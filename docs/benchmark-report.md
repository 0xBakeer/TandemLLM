# The benchmark report

TandemLLM bare engine: raw /v1/completions, no conductor or chat layer.

`tools/release_report` measures a running TandemLLM server on your own GPU and writes one HTML page: decode speed by prompt length, prefill, requests at once, chats that share a long system prompt, latency under load, a conversation that comes back, the repository's card tool, and whether speculation changes the text. It works for any model the server serves. The plan is sized from what your server reports, so a 24 GB card gets shorter prompts and fewer requests at once than a DGX Spark.

Every request goes to `/v1/completions` with `X-Tandem: raw`. The prompt is the model's chat template with thinking off, rendered on the client and checked id for id against the server's `/v1/tokenize`, as [tools/parallel_card.py](../tools/parallel_card.py) does. A response that does not come back with `X-Tandem: raw` makes its row invalid, and an invalid row shows its reason, never a number.

The page is a view of [tools/measure](measurement.md#one-measuring-method)'s records: `python -m tools.measure plan` says which rows a release still needs, `run` measures only those, and `render report` writes the page, with every row's source (a gate run, an earlier report, a measuring run). The commands below still measure a whole report in one go.

## Run it

Start your server as you normally do, wait until it is idle, then run the tool with the engine's Python (it needs the model's tokenizer, which `transformers` loads from the weights folder):

    python -m tools.release_report plan                       # what it will run, and how long
    python -m tools.release_report run --out ~/tandem-report --quick
    python -m tools.release_report render ~/tandem-report/data.json

`--url` defaults to the local server, `http://127.0.0.1:8000`. The page is `~/tandem-report/report.html`. Drop `--quick` for the full report.

Before it sends a measured request, the tool:

- refuses a server that is busy (requests running, waiting or decoding). `--wait-idle 300` waits up to 300 s first.
- refuses a server with the response cache on, since it could answer a repeated greedy request without the engine. Start the server with `RESPONSE_CACHE=off`.
- sends one probe request of about 2,000 tokens and refuses the server if the answer is not served raw.
- prints the plan, the lengths it skips and why, and a time estimate from the probe's speed.

A loaded conductor is fine: its router hands raw requests to `/v1/completions` back unchanged, and the tool records it. The tool never changes a server setting and calls no admin route. It writes only under `--out`. `run` and `render` make no network call except to your server.

## How long it takes

The estimate comes from your box. On one DGX Spark with Qwen3.8-Flash-Next, `--quick` takes 20 to 25 minutes and the full run about 2.5 hours (measured on v0.15.1: the sections took 136 minutes, plus a server start per part), most of it in long prompts: a 240k-token prompt prefills for about 7 minutes. Run the full report in parts with `--only`; every run adds its sections to the same `data.json`:

    python -m tools.release_report run --out ~/tandem-report --only a h
    python -m tools.release_report run --out ~/tandem-report --only c e i
    python -m tools.release_report run --out ~/tandem-report --only i d f g    # after a restart

The order matters for section i. After the rest of the suite it measures the longest prompts on a warm engine; as the first requests of a freshly started server it measures them fresh. The page compares the two.

## What each section measures

| Section | What it sends | What the page shows |
|-|-|-|
| a, b | One request at a time at 1k, 4k, 16k, 32k, 55k, 64k, 96k, 128k, 192k and 240k tokens, up to what fits your server's window. Real prose with a fresh nonce line first, greedy, thinking off, 512 tokens out. 3 runs up to 16k, 2 above | Decode tok/s (completion tokens after the first, over the time from the first to the last token), whether the engine drafted and its tokens per pass (from the response's `timings`), prefill tok/s and the time to the first token |
| c | N requests released together, N from 1 to the server's slot limit, each with its own ~2k-token prompt and 512 tokens out; then N prompts of ~32k tokens | Aggregate tok/s over the wall time, prefill included, and the slowest first token. `/health` is read every 0.2 s: a round counts only if N requests were decoding at once, and a level without such a round is invalid |
| d | 5 chats and as many as the server has slots, all at once on one new ~28k-token system prompt | The first and the last chat's first token, and the tokens each found in the cache |
| e | A ~64k prompt arriving while two chats decode; then a ~97k conversation: turn 1, turn 2, three other ~60k prompts, turn 3 | The long prompt's first token and the decoders' combined speed before and during its prefill; each turn's first token |
| f | One ~56k prompt, up to 1,024 tokens, three tasks: a summary, code, and a copy of text in the prompt | Mean decode and the best one-second window |
| g | `tools/parallel_card.py` against the same server, its default arguments and `--fixed-prompts` | The card tool's own rows |
| h | Eight fixed prompts, greedy, 512 tokens; needles in ~32k and ~128k of prose | The texts, and, after a second run on a server with speculation off, whether they are identical |
| i | The 128k and 192k prompts after the rest of the suite, and as the first requests of a fresh server | Warm against fresh (or against section a's rows when there is no fresh run) |
| j | A multiple-choice set (`--hle --hle-data FILE`, a local JSON-lines file), off by default | The score with its 95 % interval |
| k | Nothing | Every limit the server runs with, the GPU (name, memory, compute capability, driver, CUDA, torch) and the box during each run: GPU clock and power under load, the highest temperature, the lowest free memory |

A section that does not fit your server is listed with the reason: a prompt over the largest slot window, a concurrency test on a server with one slot, a needle longer than the window. Fidelity against plain decoding needs a second run of the same build with speculation off:

    python -m tools.release_report run --out ~/tandem-report --variant plain    # server started with --drafter none

A server that runs no drafter skips that comparison, since all its texts are plain ones.

The prompt text is `bench/heldout_prose.txt` by default, repeated to the length. A long prompt then repeats the same 1,224 words, and the page says so. `--prose wikitext2` reads WikiText-2 from your local Hugging Face cache instead (about 2.3 M tokens that do not repeat), and `--prose FILE` reads any text file. Nothing is downloaded.

## Launch mode

`--ref vX.Y.Z --pack ID` starts a fresh server of that tag on your box, from a `git archive` of the tag, with its own config copy, on a free port from 8100, with the conductor, the response cache and the chat service off ([tools/gate](quality-gate.md) starts its servers the same way). It measures through the same path as `--url`, stops the server, and for section h starts a second one with speculation off:

    python -m tools.release_report run --ref v0.15.1 --pack flashnext-3.05 --out ~/tandem-report

## Several packs of one model

Run each pack into its own folder, then put them on one page: an overview of the packs side by side (their headline figures, one request's decode by prompt length, the card tool's levels), then each pack's own sections.

    python -m tools.release_report combine q4/data.json q6/data.json --label "EXL3 4.0" --label "EXL3 6.0" --model Qwen3.8-27B --out 27b/data.json
    python -m tools.release_report render 27b/data.json

`combine` refuses packs of different releases.

## Share the page

The page is one HTML file of about 50 KB with its charts inline, and it loads nothing from the network. Paths, host names, user names and addresses are taken out of every string that comes from the run, and `render` refuses to write a page that still names your box. It is safe to attach to an issue or a pull request. The raw files under `--out/raw` are not scrubbed; keep them local.

For every release the maintainers run the full report on the reference DGX Spark and attach the page to the GitHub release as `tandemllm-<version>-report-<model>.html`, one page a model ([operations.md](operations.md#release-checklist)). `publish` is that upload step. It measures nothing, and you do not need it to make or share your own page.
