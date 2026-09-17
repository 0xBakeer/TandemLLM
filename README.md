# qwen38-spark-engine

A single-stream inference engine for Qwen3.8-27B on one DGX Spark (GB10, 121 GiB of unified memory,
about 273 GB/s). It reads the vendor's FP8 checkpoint as it lies on disk, quantises the parts that
pay for it, and spends the rest of its budget on making a decode step carry more than one token.

The model is a hybrid: 48 of its 64 layers are Gated DeltaNet and keep a recurrent state, the other
16 are full attention and keep a KV cache. That split is what most of this code is about. Rolling
back a rejected speculative block is a pointer move on a KV cache and an identity on a recurrent
state, and the identity is what lets this engine verify a draft **tree** across all 64 layers.

What is in here:

- Triton kernels for the checkpoint's block-scaled FP8 format, for NVFP4 at group 16, for the
  vocabulary projection, for RMS norm, and for the gated delta rule at one token and over a block.
- A quantiser with an activation-weighted clip search, and the quality gate that decides whether its
  output ships.
- Block speculation with a block drafter, a prediction head, a lookup drafter over a token
  corpus, and a router that prices them against each other every step.
- An OpenAI-compatible server, standard library only.

Measured numbers live in [RESULTS.md](RESULTS.md), what they do not cover in
[LIMITATIONS.md](LIMITATIONS.md), and every measurement with the command that produced it in
[notes/SPEED-LEDGER.md](notes/SPEED-LEDGER.md). The design and the arithmetic behind it are in
[notes/ARCHITECTURE.md](notes/ARCHITECTURE.md).

## Running it

One board runs one engine. A second one loading 15 GB of weights beside the first will wedge the
machine.

Python 3.11, torch 2.13 with CUDA 13.0, Triton 3.7, transformers 5.12. No other dependency.

### 1. Quantise

Three artifacts, built once, each from the checkpoint:

```bash
# per-input-channel activation statistics, from a calibration corpus
python tools/quant_nvfp4.py stats  --corpus bench/calib.txt --out ~/nvfp4/stats-all.pt --tokens 8192

# NVFP4 weights, group of 16, clip search weighted by those statistics
python tools/quant_nvfp4.py quant --mode clip --targets mlp  --stats ~/nvfp4/stats-all.pt \
    --out ~/nvfp4/mlp-clip.safetensors
python tools/quant_nvfp4.py quant --mode clip --targets gdn  --stats ~/nvfp4/stats-all.pt \
    --out ~/nvfp4/gdn-clip.safetensors
python tools/quant_nvfp4.py quant --mode clip --targets attn --stats ~/nvfp4/stats-all.pt \
    --out ~/nvfp4/attn-clip.safetensors

# the vocabulary projection at e4m3, one scale per row
python tools/quant_head.py build --out ~/nvfp4/head-fp8.safetensors --ratios 1.0,0.95,0.90
```

Then gate them, because a quantiser that is not gated is a guess:

```bash
python tools/quality_gate.py --tokens 2048 --gen 900 \
    --nvfp4 ~/nvfp4/mlp-clip.safetensors,~/nvfp4/gdn-clip.safetensors,~/nvfp4/attn-clip.safetensors
python tools/quant_head.py gate --head ~/nvfp4/head-fp8.safetensors \
    --nvfp4 ~/nvfp4/mlp-clip.safetensors --tokens 2048
```

The gate reports held-out loss on prose and on code, argmax agreement against the unquantised
engine restricted to the positions where it is confident, and a free generation of 900 tokens with
its repetition statistics. All three matter. Teacher-forced loss never lets an error compound, so
it cannot see a model that fails to stay on its own trajectory; it has rated a degenerate
configuration better than a healthy one before.

### 2. Serve

```bash
python server/app.py --port 8000 --max-len 4096 \
    --drafter dflash2 --dflash2-blocks 1 --dflash2-path greedy \
    --nvfp4 ~/nvfp4/mlp-clip.safetensors,~/nvfp4/gdn-clip.safetensors,~/nvfp4/attn-clip.safetensors \
    --fp8-head ~/nvfp4/head-fp8.safetensors
```

`/v1/completions` and `/v1/models`, streaming, one request at a time. Temperature above zero is
refused rather than answered greedily. `--tree` verifies a draft tree instead of a chain, and
`--drafter merged` prices the block drafter against the lookup drafter every step.

`--dflash2-ckpt <dir>` loads a drafter trained by `tools/train_dflash2.py` on this target's own
output instead of the released one. `--think-budget N`, the per-request `max_reasoning_tokens` and
`--reasoning-effort low|medium|xhigh` cap how long the model reasons; the budget closes the
reasoning block itself when it is spent, which changes the answer. LIMITATIONS says how.

### 3. Measure

```bash
python tools/profile_decode.py  --breakdown          # step time, and where it goes
python tools/profile_block.py   --blocks 1,4,8,16    # what verifying B tokens costs
python tools/profile_prefill.py --lens 256,2048,8192 --breakdown
python tools/verify_spec.py     --dflash2 1 --new 48 --k 8   # speculation must not change output
python tools/bench_decode.py    --new 128 --no-engram --dflash2 1 --think off
python tools/train_data.py      --gen 96 --corpus-seqs 200 --passage-only   # record, then
python tools/train_dflash2.py   --data train/data --train all --lr 3e-5 --steps 500  # train
```

Take a kernel timing from inside the engine or not at all. The same NVFP4 configuration measured
0.254 ms and 0.627 ms in two processes that differed only in what they had allocated before, so a
standalone microbenchmark here is a ranking and never a budget.

## The corpus

The lookup drafter can read a memory-mapped token corpus built by `tools/build_corpus.py`. It holds
token ids and no text, it is built from public sources on the machine that serves, it is excluded
from version control, and it is never copied off the box. Do not put anything into it that you
would not publish, and do not put model output from your own evaluation prompts into it: a store
that can contain the test produces a number about the store. The build tool says so when you try.

## The serving-time caches

Four of them, in `engine/cache.py`, all on by default except the last. None of them may change what
the engine writes, and what "may not change" means is spelled out per cache below and measured by
`tools/cache_gate.py`.

```bash
python server/app.py --port 8000 --drafter dflash2 \
    --cache-budget-gb 24 --prefix-chunk 256 \
    --suffix-store ~/.qwen38-spark-engine/suffix \
    --response-cache                       # opt-in, see below

curl -s localhost:8000/v1/cache/stats | python -m json.tool
python tools/cache_gate.py --stage exact   # run this one before believing the others
```

**A session's state, kept.** After a turn the engine's whole state -- the 16 KV caches, the 48
recurrent states, the 48 convolution states and the drafter's own position-indexed KV -- stays in
RAM, keyed by the tokens that produced it. The next turn of the same conversation begins with all
of those tokens, so it resumes there and forwards only the chat template's glue and the new
message. `--no-session-cache` turns it off.

**A shared prefix, checkpointed.** During a prefill the same state is snapshotted every
`--prefix-chunk` tokens. A request whose prompt starts with a prefix some earlier request has
already read resumes at the longest checkpoint they share, which is what makes a system prompt free
from the second request on. `--no-prefix-cache` turns it off.

Both are one store under one byte budget, least-recently-used first out. The budget is a number of
*conversations* long before it is a number of tokens long: measured from the server's own
`/v1/cache/stats`, a snapshot is **151.0 MB of recurrent state whatever the prefix length**, plus
2.9 MB of convolution state, plus 65.5 kB a token of KV and 20.0 kB a token of drafter KV. A 1,536
token boundary is 285 MB.

**`--prefix-chunk` is the one knob that is a real trade and it is not the obvious one.** A chunk
costs a whole pass over the 16.35 GB of weights, because a forward reads them all whatever it
carries -- the same economics the rest of this engine is built on. Measured on a 1,724-token
prompt: a cold prefill is 2,508 ms unchunked, 2,858 ms at `--prefix-chunk 1024` and 3,940 ms at
256. So a fine grid makes every prompt nobody ever shares 57 % slower. 1024 is the default for
that reason; drop it to 256 when one system prompt really is shared, where the finer grid returns
far more than it costs (582 ms a request against 1,169 ms).

A hash is only ever a hint here. Every hit re-checks the stored token prefix element for element
before any state is restored, because a 64-bit collision would answer one request with another
request's state and nothing downstream would catch it.

**Exactness, which is two claims and not one.** A prefix-cache resume lands on the same chunk grid
a cold prefill uses, so the warm run is bit-identical to the cold one by construction. A session
resume is not: its boundary is wherever the previous turn stopped, and the state there was written
by speculative verify blocks rather than by prefill chunks. It is the state that really produced
the previous turn -- not the state a re-read of the conversation would compute -- and it is held to
the gate the rest of this engine is held to, the same argmax.

**A persistent suffix store.** `--suffix-store DIR` keeps an append-only log of the token ids this
engine has read and written, with a suffix array over it, and hands it to the lookup drafter beside
`--corpus`. Warm text from last week's session then drafts at a 0.2 ms lookup instead of a 4 ms
prediction head. Token ids only, never text; mode 0700; outside this repository; capped by
`--suffix-store-mb`, over which the oldest half is forgotten at a document boundary. It is off if
you pass an empty path, and everything the corpus section below says applies to it as well.

**An exact-prompt response cache**, `--response-cache`, opt-in. Greedy decoding is a function of
(prompt, params), so an identical request has an identical answer and this is memoisation rather
than an approximation. It is opt-in because a server that answers from a dictionary is not a server
a benchmark should ever be pointed at, and because it is not consulted at all under a relaxed
accept rule, where the engine is not answering the greedy question.

## Credits

The checkpoint and its published reference implementation are the vendor's. The kernels, the
quantiser, the drafters, the router, the server and the measurements here are this repository's.


