"""The first step of the Kolibri-1 NVFP4 build: how fast can a job read the 156 GB checkpoint, and does the
bucket take the output? CPU only, no torch.

  * shard 1 of the BF16 release read sequentially through the read-only `hf://` mount;
  * shard 2 fetched with `hf_hub_download` (Xet, `HF_XET_HIGH_PERFORMANCE=1`) to local disk;
  * 64 single expert tensors read through the mount with safetensors (the random-access pattern
    a layer read has when it does not copy the shard first);
  * a 1 GB file written to and read back from the bucket prefix, then removed.

The faster of the first two decides how the build reads the checkpoint; 50 MB/s is the floor
(156 GB in under an hour).

    python tools/kolibri_probe.py --mount /models/bf16 --repo Aleph-Alpha/Kolibri-1-BF16 \
        --local /tmp/probe --bucket-dir /work/kolibri-nvfp4
"""

from __future__ import annotations

import argparse
import json
import os
import time


def rate(nbytes: int, sec: float) -> float:
    return nbytes / max(sec, 1e-6) / 1e6


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mount", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--local", default="/tmp/probe")
    ap.add_argument("--bucket-dir", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    res: dict = {}
    idx = json.load(open(os.path.join(a.mount, "model.safetensors.index.json")))["weight_map"]
    shards = sorted(set(idx.values()))

    # 1. sequential read through the mount
    p = os.path.join(a.mount, shards[0])
    n, t0 = 0, time.time()
    with open(p, "rb") as f:
        while True:
            b = f.read(64 << 20)
            if not b:
                break
            n += len(b)
    dt = time.time() - t0
    res["mount_sequential"] = {"file": shards[0], "bytes": n, "seconds": dt, "MB_s": rate(n, dt)}
    print(f"[mount] {shards[0]}: {n / 1e9:.2f} GB in {dt:.1f} s = {rate(n, dt):.0f} MB/s", flush=True)

    # 2. Xet download of another shard
    from huggingface_hub import hf_hub_download
    os.makedirs(a.local, exist_ok=True)
    t0 = time.time()
    q = hf_hub_download(a.repo, shards[1], local_dir=a.local)
    dt = time.time() - t0
    n = os.path.getsize(q)
    res["xet_download"] = {"file": shards[1], "bytes": n, "seconds": dt, "MB_s": rate(n, dt),
                           "high_performance": os.environ.get("HF_XET_HIGH_PERFORMANCE")}
    print(f"[xet] {shards[1]}: {n / 1e9:.2f} GB in {dt:.1f} s = {rate(n, dt):.0f} MB/s", flush=True)

    # 3. random access: single expert tensors through the mount (layer 2 lives in shards 2-3)
    # (the safetensors header gives each tensor's byte range; read those ranges as a loader would)
    try:
        path = os.path.join(a.mount, shards[2])
        with open(path, "rb") as f:
            hl = int.from_bytes(f.read(8), "little")
            hdr = json.loads(f.read(hl))
            keys = [k for k in hdr if ".experts." in k][:64]
            n, t0 = 0, time.time()
            for k in keys:
                s, e = hdr[k]["data_offsets"]
                f.seek(8 + hl + s)
                n += len(f.read(e - s))
        dt = time.time() - t0
        res["mount_tensors"] = {"tensors": len(keys), "bytes": n, "seconds": dt, "MB_s": rate(n, dt)}
        print(f"[mount] {len(keys)} expert tensors: {n / 1e6:.0f} MB in {dt:.1f} s = {rate(n, dt):.0f} MB/s",
              flush=True)
    except Exception as e:
        res["mount_tensors"] = {"error": repr(e)}
        print(f"[mount] tensor read: {e!r}", flush=True)

    # 4. the bucket
    os.makedirs(a.bucket_dir, exist_ok=True)
    bp = os.path.join(a.bucket_dir, "probe.bin")
    blob = os.urandom(64 << 20)
    t0 = time.time()
    with open(bp, "wb") as f:
        for _ in range(16):
            f.write(blob)
    dt_w = time.time() - t0
    t0 = time.time()
    n = 0
    with open(bp, "rb") as f:
        while True:
            b = f.read(64 << 20)
            if not b:
                break
            n += len(b)
    dt_r = time.time() - t0
    os.remove(bp)
    res["bucket"] = {"bytes": n, "write_MB_s": rate(n, dt_w), "read_MB_s": rate(n, dt_r)}
    print(f"[bucket] 1.07 GB written at {rate(n, dt_w):.0f} MB/s, read back at {rate(n, dt_r):.0f} MB/s",
          flush=True)
    best = max(res["mount_sequential"]["MB_s"], res["xet_download"]["MB_s"])
    res["faster_path"] = "mount" if res["mount_sequential"]["MB_s"] >= res["xet_download"]["MB_s"] else "xet"
    res["gate_50MB_s"] = best >= 50
    res["checkpoint_hours_at_best"] = 156.2e9 / (best * 1e6) / 3600
    print(f"[probe] faster path {res['faster_path']} at {best:.0f} MB/s: the 156 GB checkpoint in "
          f"{res['checkpoint_hours_at_best']:.2f} h; gate (>= 50 MB/s) {'PASS' if res['gate_50MB_s'] else 'FAIL'}",
          flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
