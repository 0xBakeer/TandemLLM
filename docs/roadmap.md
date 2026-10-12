# Roadmap

TandemLLM runs two models (Qwen3.8-27B and Qwen3.8-Flash-Next, one at a time) on one board (DGX Spark), one request at a time by default. Each step below widens one of those limits, and each keeps the two rules that hold today: the output stays the model's greedy text, up to the documented near-tie exception ([exactness.md](exactness.md)), and no measured number gets worse. We plan to take the steps in this order. Nothing here is a promise of dates.

## A mixture-of-experts model: served, with four parts open

Qwen3.8-Flash-Next is served from turboderp's EXL3 packs through the same server, caches and ops scripts as the 27B ([architecture.md](architecture.md), [quantisation.md](quantisation.md)). The plan written here before it named the NVFP4 checkpoint; the published EXL3 packs replaced it. What is still open:

- Vision. Flash-Next has its own vision tower (27 layers, EXL3), and the engine does not load it yet (VIS-97).
- The 5.05 pack codes its experts at 5 and 7 bits; the MoE kernels take 2 to 6 and 8, so it does not load until a 7-bit expert kernel exists.
- Quality against the BF16 model, per pack. The BF16 release is 360 GB and does not fit the box, so the reference run that gives KL a pack is not done.
- Parallel requests run and are lossless on it, but three at once were slower in total than one after the other ([parallel-requests.md](parallel-requests.md#on-flash-next)).

Later, other EXL3 models that fit one box, Qwen3.6-35B-A3B among them.

## Parallel requests

Several requests share one verify pass in lockstep rounds: each request brings its own tree, and one forward over all the trees costs little more than one over a single tree, because the weights are read once. Phase A is built behind `--parallel N`, off by default ([server.md](server.md), section Parallel requests): side slots, the GPU turn, the joint verify, the packed draft, and admission by bytes (a request queues while its worst case would push the board under its floor). First single runs on the box: two requests at about 1.4 times the single-request total, four at about 1.4 times, the single-request row unchanged. Next: the host side of a round, one tree cut across the slots, graphs for the joint pass and one graph pool for every slot. A second request costs at least the step from 16 to 32 rows of the verify kernel, so two requests at about 1.6 times is the target at today's kernels; 2 times needs a 32-row verify priced like 16 rows.

Later, multi-user serving: an API key per user, with caches isolated per user, so no user can resume from another user's prefix.

## A cache tier on NVMe

Built in 0.3.13-test ([caches.md](caches.md#disk-tier)): recurrent-state anchors and KV chunks on the NVMe, a 50 GB budget, a time to live and a 50 GB free-space floor, set from the dashboard. What remains: the restore's rate measured on the board after the deploy (the 0.6 s for a 30,000-token head is arithmetic from CPU-side rates), whether rows written before a restart equal a cold prefill after it when Triton autotunes again (the fingerprint holds the code and the flags, not the chosen kernel configurations), and a background backfill of the resident prompt's chunks while the engine is idle.

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
