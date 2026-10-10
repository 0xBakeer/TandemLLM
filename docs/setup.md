# The setup guide

The dashboard's `#/setup` page walks through an install in eight steps: pick a model, read its measured table, pick the weights, switch the tools beside the model on or off, set the parallel requests and the memory, set the methods, review, and install. The server half lives in `server/setup/` and answers under `/v1/setup/*`. Nothing in it is on the serving path: a chat request never reaches it.

## The model registry

Each model the guide offers is a folder under `models/` with a `model.yaml` ([models/README.md](../models/README.md) lists the fields). `models/qwen3.8-27b/model.yaml` is the one model today. Every number in it says where it comes from:

- The weights and the drafters per profile are the `[memory] load stages` lines of the 0.6.2 runs of 2026-10-10, GiB allocated, one run per EXL3 size. NVFP4's are from the 0.5.0 load of 2026-10-09.
- The tools' memory is the dashboard's live memory split of the same evening (the reference box on exl3-6.0 with 4 slots, the narrator, Laya and the shadow reader loaded). The registry's parts add up to that split's 59.34 GiB in use within 0.1 GiB.
- The measured table is the 0.6.2 run itself; `models/qwen3.8-27b/measurements/0.6.2-summary.json` is its summary file, and a test holds the two equal. The NVFP4 row is from build 0.4.8 and is marked not comparable.

Memory is in GiB (2^30 bytes) and disk in GB (10^9 bytes) everywhere in the registry and the API.

### The KV bytes a token

One token of context window costs 106,752 bytes, which is 104.25 KiB or 106.75 kB: 64 KiB of the model's attention KV (16 attention layers, 4 KV heads of 256, K and V in bf16), 20 KiB for each of the two drafters' context KV (5 layers, 8 KV heads of 128) and 256 B of rotary table. At 262,144 tokens that is 28.0 GB, or 26.1 GiB. The docs said "106.5 KiB" until 0.6.2: that was the kB figure with the KiB unit. With the KV pool on (`QSE_KV_POOL=on`) a slot reserves its window as addresses and commits pages only for the rows its conversation writes.

### The CUDA graphs and the side slots

The 6.63 GiB of CUDA graphs belong to slot 0. A side slot captures its own verify graphs, about 7.5 GiB each, only with `QSE_SIDE_GRAPHS=on`; the guide always writes it off, as the reference box runs, and a side slot then costs 1.19 GiB beyond its KV (the load stage `side_slots` took 3.58 GiB for three on 2026-10-10).
