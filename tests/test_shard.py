"""CPU tests for the sharded loader, on synthetic data small enough to run anywhere.

WHAT THESE ARE PROTECTING

The recorded tensors are 109 GB and the reason they are sharded at all is that opening 5,863 files
costs tens of minutes with the GPU idle. Sharding trades that for an offset table, and an offset
table has exactly one interesting failure mode: **every count agrees, every shape agrees, and the
rows belong to the wrong sequence.** The drafter then trains on one sequence's residual stream
against another's labels, the loss still falls, and nothing in the pipeline says a word -- it is the
same shape of silent fault as the label off-by-one `test_train_dflash2.py` exists for.

So the tests here compare VALUES, sequence by sequence and field by field, between the two layouts.
A shard that round-trips every byte is a shard that cannot be misaligned.

The second thing tested is that the two layouts are interchangeable: `load_data` on a per-file
directory and `load_data` on its shards must produce the same samples, in the same order, with the
same splits, under the same `--limit`. That is what lets a run switch layouts without restating any
result.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from tools.h100.shard import FIELDS, build_shard, plan  # noqa: E402
from tools.train_dflash2 import load_data  # noqa: E402

HID = 12          # 5 taps of a tiny hidden size; the real one is 5 * 5120
TOPK = 4


def make_recording(root: str, n_seq: int = 11, seed: int = 3) -> dict:
    """A directory in exactly the shape `tools/h100/record.py` writes, with random contents.

    Lengths are deliberately uneven and deliberately include a 1-position sequence: a shard planner
    that assumes a constant length, or a loader that assumes every slice is non-degenerate, is wrong
    on the real set too, where sequences run from 32 to 4,096 positions.
    """
    g = torch.Generator().manual_seed(seed)
    os.makedirs(root, exist_ok=True)
    metas = []
    for i in range(n_seq):
        n = [1, 3, 17, 4, 29, 2, 8, 5, 13, 7, 21][i % 11]
        name = f"p{i:06d}-seq"
        rec = {
            "fused": torch.randn(n, HID, generator=g).to(torch.bfloat16),
            "ids": torch.randint(0, 1000, (n,), generator=g, dtype=torch.int32),
            "label": torch.randint(0, 1000, (n,), generator=g, dtype=torch.int32),
            "top_ids": torch.randint(0, 1000, (n, TOPK), generator=g, dtype=torch.int32),
            "top_lp": torch.randn(n, TOPK, generator=g).to(torch.float16),
        }
        meta = {"name": name, "topic": ["en", "code", "de"][i % 3], "kind": "gen",
                "n": n, "gen_start": 0, "klass": "prose",
                "think": False, "effort": "off", "source": "synthetic",
                "think_end": 0, "think_tokens": 0,
                "split": "heldout" if i % 5 == 0 else "train"}
        rec.update({k: meta[k] for k in ("name", "topic", "kind", "gen_start", "think",
                                         "effort", "think_end", "source")})
        torch.save(rec, os.path.join(root, f"{name}.pt"))
        metas.append(meta)
    man = {"created": "synthetic", "topk": TOPK, "positions": sum(m["n"] for m in metas),
           "sequences": metas}
    with open(os.path.join(root, "manifest.json"), "w") as f:
        json.dump(man, f)
    return man


def run_shard(src: str, out: str, shard_gb: float, workers: int = 1) -> None:
    """Drive the tool the way a job does, as a subprocess, so the CLI is covered too."""
    subprocess.run([sys.executable, os.path.join(ROOT, "tools", "h100", "shard.py"),
                    "--src", src, "--out", out, "--shard-gb", str(shard_gb),
                    "--workers", str(workers)],
                   check=True, cwd=ROOT, capture_output=True)


def test_every_byte_survives_the_repack():
    """Field by field, sequence by sequence: the shard slice equals the source file."""
    with tempfile.TemporaryDirectory() as tmp:
        src, out = os.path.join(tmp, "data"), os.path.join(tmp, "shards")
        make_recording(src)
        run_shard(src, out, shard_gb=2e-6)            # ~2 kB: several shards out of eleven files

        with open(os.path.join(out, "manifest.json")) as f:
            man = json.load(f)
        assert man["sharded"] is True
        assert len(man["shards"]) > 1, "the planner put everything in one shard"

        opened = {}
        for meta in man["sequences"]:
            i = meta["shard"]
            if i not in opened:
                opened[i] = torch.load(os.path.join(out, f"shard-{i:04d}.pt"),
                                       map_location="cpu", mmap=True, weights_only=True)
            want = torch.load(os.path.join(src, f"{meta['name']}.pt"), map_location="cpu")
            o, n = meta["offset"], meta["n"]
            assert n == int(want["ids"].numel())
            for f_ in FIELDS:
                got = opened[i][f_][o:o + n]
                assert got.dtype == want[f_].dtype, (meta["name"], f_)
                assert got.shape == want[f_].shape, (meta["name"], f_)
                assert torch.equal(got, want[f_]), (meta["name"], f_)


def test_offsets_tile_each_shard_with_no_gap_and_no_overlap():
    """Within a shard the sequences are laid end to end, and the total is the shard's length."""
    with tempfile.TemporaryDirectory() as tmp:
        src, out = os.path.join(tmp, "data"), os.path.join(tmp, "shards")
        make_recording(src)
        run_shard(src, out, shard_gb=2e-6)
        with open(os.path.join(out, "manifest.json")) as f:
            man = json.load(f)

        per_shard: dict[int, list[dict]] = {}
        for meta in man["sequences"]:
            per_shard.setdefault(meta["shard"], []).append(meta)
        for i, metas in per_shard.items():
            metas.sort(key=lambda m: m["offset"])
            pos = 0
            for m in metas:
                assert m["offset"] == pos, (i, m["name"], m["offset"], pos)
                pos += m["n"]
            sh = torch.load(os.path.join(out, f"shard-{i:04d}.pt"),
                            map_location="cpu", weights_only=True)
            assert sh["ids"].numel() == pos
            assert sh["fused"].shape == (pos, HID)
        assert sum(m["n"] for m in man["sequences"]) == man["positions"]


def test_load_data_agrees_across_the_two_layouts():
    """The loader is the contract, not the file format: same samples, same order, same splits."""
    with tempfile.TemporaryDirectory() as tmp:
        src, out = os.path.join(tmp, "data"), os.path.join(tmp, "shards")
        make_recording(src)
        run_shard(src, out, shard_gb=2e-6)

        flat = load_data(src, "cpu", "cpu")
        shed = load_data(out, "cpu", "cpu")
        assert len(flat) == len(shed) > 0
        for x, y in zip(flat, shed):
            assert x.name == y.name
            assert x.split == y.split and x.klass == y.klass and x.kind == y.kind
            assert x.gen_start == y.gen_start
            assert len(x) == len(y)
            assert torch.equal(x.ids, y.ids)
            assert torch.equal(x.label, y.label)
            assert torch.equal(x.top_ids, y.top_ids)
            assert torch.equal(x.top_lp, y.top_lp)
            assert torch.equal(x.fused, y.fused)
            assert x.fused.dtype == y.fused.dtype == torch.bfloat16


def test_limit_is_a_prefix_in_both_layouts():
    """`--limit N` means the first N of the manifest, whichever layout it reads."""
    with tempfile.TemporaryDirectory() as tmp:
        src, out = os.path.join(tmp, "data"), os.path.join(tmp, "shards")
        make_recording(src)
        run_shard(src, out, shard_gb=2e-6)
        for n in (1, 4, 7):
            flat = load_data(src, "cpu", "cpu", limit=n)
            shed = load_data(out, "cpu", "cpu", limit=n)
            assert [s.name for s in flat] == [s.name for s in shed]
            assert len(shed) == n


def test_fused_is_a_view_and_not_a_copy():
    """The point of the whole exercise: `fused` is 99 % of the bytes and must stay memory-mapped.

    A slice of a memory-mapped tensor shares its storage, so the storage under a sample is larger
    than the sample. If a future edit makes the loader materialise instead, this fails -- and the
    only symptom in production would be a job that runs out of host RAM at 109 GB.
    """
    with tempfile.TemporaryDirectory() as tmp:
        src, out = os.path.join(tmp, "data"), os.path.join(tmp, "shards")
        make_recording(src)
        run_shard(src, out, shard_gb=1.0)             # one shard, so every sample shares it
        shed = load_data(out, "cpu", "cpu")
        total = sum(len(s) for s in shed)
        ptrs = {s.fused.untyped_storage().data_ptr() for s in shed}
        assert len(ptrs) == 1, "the samples do not share one mapping"
        assert shed[0].fused.untyped_storage().size() == total * HID * 2


def test_a_single_oversized_sequence_still_gets_a_shard():
    """The planner must not drop or merge a sequence bigger than the shard target."""
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "data")
        man = make_recording(src)
        groups = plan(man["sequences"], src, shard_bytes=1)     # every file is oversized
        assert len(groups) == len(man["sequences"])
        assert [g[0]["name"] for g in groups] == [m["name"] for m in man["sequences"]]


def test_a_part_file_is_never_left_behind():
    """A shard that exists is a shard that is complete: the writer renames, it does not stream."""
    with tempfile.TemporaryDirectory() as tmp:
        src, out = os.path.join(tmp, "data"), os.path.join(tmp, "shards")
        man = make_recording(src)
        os.makedirs(out)
        build_shard((0, man["sequences"], src, out))
        assert os.path.exists(os.path.join(out, "shard-0000.pt"))
        assert not [n for n in os.listdir(out) if n.endswith(".part")]


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok    {name}")
            except Exception as e:                               # noqa: BLE001
                fails += 1
                print(f"FAIL  {name}: {type(e).__name__}: {e}")
    raise SystemExit(1 if fails else 0)
