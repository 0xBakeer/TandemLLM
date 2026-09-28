# Exactness

Under greedy decoding, TandemLLM writes the text that its target model writes one token at a time. Drafters, tree shapes, block widths, the lookup store and the caches change how fast the text arrives, not what it says. This page states the promise precisely, names its one exception, and lists the checks that hold the engine to it.

## The promise

Our reference is plain greedy decoding on the same weights: one forward pass per token, take the highest-scoring token, repeat. For every greedy request, the engine's output tokens equal the reference's tokens.

"The same weights" matters. A profile chooses the weights. The NVFP4 profile is exact against NVFP4 plain decoding, not against the FP8 checkpoint. The difference between weight sets is a quality question, and [quantisation.md](quantisation.md) measures it with a separate gate. Speculation never adds to that difference.

The promise covers:

- any drafter, including a drafter that is wrong on purpose on every token;
- chains and trees of any shape the router builds, and both block widths;
- a request resumed from the prefix cache or the resident prefix;
- request options that change the scores (penalties, `logit_bias`, structured-output masks), because the engine applies them to every verified row exactly as plain decoding applies them to its one row.

## The one exception: near ties

Two tokens can score the same up to one rounding step of the arithmetic. Then which one wins an argmax depends on the order of a floating-point sum. A batched verify and a one-token step do not always sum in the same order, so at such a position the two can pick different tokens. After that position the texts differ, and both are correct greedy continuations of their own prefixes.

TandemLLM keeps this rare by design. The head returns fp32 logits, and every matrix-product kernel fixes its reduction order by the weight's shape and never by the row count ([kernels.md](kernels.md)). What is left comes from the BF16 residual stream: a difference of about 2e-7 in fp32, sitting on a BF16 rounding boundary, flips one value in one layer, and the layers after it carry the flip.

Our lossless gate, `tools/verify_spec.py`, prints the gap between the top two logits in BF16 units (ulps) whenever outputs differ. A difference passes only when the gap at that position is within one ulp. Anything larger is a bug.

## Stronger than the promise: release identity

Every release candidate is also held to the release before it. Before a change ships, 13 plain requests (the five benchmark workloads, streamed and not, one with thinking, one with tools, one under penalties, one seeded sampled request) must come back byte-identical to the current release (`tools/api_text_check.py`). The served configuration must also produce bit-identical logits with the new flags off (`tools/flagoff_identity.py`). A speed change that alters one token of text does not ship.

## Sampling

A sampled request (temperature above 0) is exact in distribution. At each verified node the engine draws the target's own token, and a drafted token survives exactly while the draw lands on it. When the draw lands elsewhere, that draw is the token. For a drafter that proposes one fixed token this is the standard rejection sampler, `accept with min(1, p/q)`, with no second draw. The output therefore follows the target's own sampled distribution. A drafter that samples its own proposal carries its proposal distribution, and the accept uses it.

All sampling filters (`temperature`, `top_p`, `top_k`, `min_p`) run after the penalties, through one function in `engine/sample.py`. Drafted rows and plain rows see the same distribution.

A request with a `seed` draws with noise keyed by its position in the sequence: the token at index t is the argmax of `log p_t + G_t`, where `G_t` is Gumbel noise from a generator seeded by `(seed, t)`. That is an exact sample from `p_t`, and it does not depend on how many draws the drafters made before. The same seed therefore gives the same text whatever the drafters proposed.

`tests/test_sample.py` checks the samplers statistically on the CPU, and `tools/sampled_dist.py` checks the served engine's sampled answers against the target sampled alone.

## Caches

A cache restore gives back bytes, not an approximation. The prefix cache and the resident prefix take their snapshots on the prefill chunk grid, and a resumed request forwards the rest of its prompt on the same grid. It therefore computes the same arithmetic as a cold prefill, bit for bit. `tests/test_cache.py` and `tests/test_resident.py` check every restored tensor for bit equality.

Session resumes are one step weaker. It resumes a conversation from the state its last turn ended in, which verify rounds wrote rather than prefill chunks. That state produced the previous answer, but it is not the state a cold re-read would compute. It is held to the engine's greedy gate (same tokens) and not to bit equality. [caches.md](caches.md) has the details.

An opt-in response cache (`--response-cache`) answers an identical greedy request from memory. It is off by default, so a benchmark can never be pointed at a dictionary.

## What changes output on purpose

Some features change the text because the client asked for it. They are not exceptions to the promise. Plain decoding with the same setting writes the same text.

- Penalties (`repetition_penalty`, `presence_penalty`, `frequency_penalty`) and `no_repeat_ngram_size`.
- Structured outputs and forced tool calls, which mask the tokens the constraint does not allow.
- A reasoning budget or the stall close, which end the thinking block early ([speculative-decoding.md](speculative-decoding.md)).

Two relaxed accept rules also exist (`--relax-tau`, `--relax-rank`) that accept a drafted token when it is close to the target's choice. They break the promise, exist only for measurement, and are off in every profile. When either is set, the server prints a warning line.

## The checks

| check | what it holds |
|-|-|
| `tools/verify_spec.py` | greedy output with no drafter, an adversarial drafter, a half-right drafter and the served drafters must match, up to the one-ulp near-tie rule |
| `tools/flagoff_identity.py` | the served configuration with new flags off is bit-identical to the build it replaces |
| `tools/api_text_check.py` | 13 plain requests byte-identical to the current release |
| `tools/cache_gate.py` | warm and cold runs agree (bit-identical on the chunk grid, same tokens for a session) |
| `tests/test_forward_tree.py`, `tests/test_tree.py` | tree verify equals chain verify on a tiny model, on the CPU |
| `tests/test_cache.py`, `tests/test_resident.py` | restores are bit-exact |
| `tests/test_sample.py`, `tools/sampled_dist.py` | sampled output follows the target's distribution |
