# Roadmap

Today TandemLLM runs one model (Qwen3.8-27B) on one board (DGX Spark), one request at a time. Each step below widens one of those three limits, and each keeps the two rules that hold today: the output never changes, and no measured number gets worse. We plan to take the steps in this order. Nothing here is a promise of dates.

## Next: a mixture-of-experts model

Our first new model is Qwen3.8-Flash-Next, a mixture-of-experts hybrid of the same family, in its published NVFP4 form. What it needs:

- Expert routing inside the verify. A tree of 16 nodes can route each node to different experts, so a verify reads the union of the experts its nodes chose. A verify's cost curve therefore depends on the text, and the router has to price that.
- The experts as W4A16 NVFP4, the rest of the weights in MXFP8, and a large per-layer embedding table kept memory-mapped rather than resident.
- Its built-in prediction head as the first drafter.

By the byte math it fits: about 71.3 GiB resident without the embedding table, which stays on disk (arithmetic from the checkpoint headers). Bandwidth puts its floor at about 38 tok/s at 273 GB/s, or 31 at 220 GB/s (arithmetic, not measured). It never shares the board with the 27B engine, because the two together do not fit.

After it, Qwen3.6-35B-A3B in its standard NVFP4 export, to check that the stack works on a checkpoint we did not quantise ourselves.

## Parallel requests

Several requests can share one verify pass in lockstep rounds: each request brings its own tree, and one forward over all the trees costs little more than one over a single tree, because the weights are read once. First measurements: 66.6 tok/s in total at 2 parallel requests and 78.6 at 4, against about 46 for one (single runs, needs verification). Our plan aims at about 2 times the single-request total at 2 requests, lossless, with the single-request row never worse.

Admission will count bytes. Before a request starts, the engine computes its worst-case memory and queues it if the board would fall under its floor.

Later, multi-user serving: an API key per user, with caches isolated per user, so no user can resume from another user's prefix.

## A cache tier on NVMe

Recurrent-state anchors and KV chunks written to disk, with a time to live and a 60 GB cap, so an agent conversation can resume after the resident prefix has moved on to another one. Designed, not built.

## Profiles as data

Today a profile is a `serve.env` file. The plan is a small profile format that a validator checks. It names every weight file with its format and hash, and it holds all the settings. A facts block records what we measured: memory and disk, speed at prompt lengths from 256 to 64k tokens, and quality against the reference. Each number carries a label that says how we got it, such as "measured" or "estimate".

The repository would ship a signed catalog of tested profiles, and the dashboard would get a Profiles view to install them and switch between them. A switch would do the byte math first, allow one engine at a time, and roll back if the new profile does not come up healthy.

## Other GPUs, then several GPUs

1. Other single GPUs (H100, H200, RTX Pro 6000, L40S). Most have no FP4 tensor cores, so they need to decode the 4-bit weights in software, or run an FP8 or BF16 profile. Each GPU needs its own tile tables, and the router needs new prices: an H100 reads memory more than ten times faster than the Spark, so the best tree width and drafter size change. Discrete GPUs also change the memory model, since admission must count the GPU's own memory and not the system's.
2. Several GPUs: tensor parallel inside one server, two DGX Sparks over their 200 Gb/s link, and servers with 8 GPUs. A reduction across GPUs changes the order of a floating-point sum, so row invariance needs a fixed reduction order across devices.

## Housekeeping

- Runtime names still carry the old project name, from the `QWEN38_` and `QSE_` variable prefixes to the metric names. They change together in one release, and the old variable names keep working for one release after that.
- Engine settings move from import-time globals to a settings object handed to the engine, so two engines with different settings can share a process.
- The engine moves into an installable package.
- SGLang and llama.cpp rows join the vLLM comparison, measured with the same rules.
