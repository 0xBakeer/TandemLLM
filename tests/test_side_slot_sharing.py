"""SPD-165: a side slot reads no drafter checkpoint and no draft head again.

`dflash2.load_weights` filed its cache under the last tensor's name (its loop variable took the key's
name), so every side slot read both arms' BF16 checkpoints again: 38-40 s a slot, and the copies stayed
on the device. The 27B adapter keeps one copy per (snapshot, device, dtype); the EXL3 draft head slot 0
holds is handed to each side slot's engine."""

from __future__ import annotations

import torch

from engine.drafters import dflash2 as df2
from engine.models.qwen3_5 import serving


def test_a_side_slot_gets_the_replaced_dict_and_reads_no_checkpoint(monkeypatch):
    reads = []

    def counting(snapshot, device="cuda", dtype=torch.bfloat16):
        reads.append(snapshot)
        return {"fc.weight": torch.ones(1)}
    monkeypatch.setattr(df2, "load_weights", counting)
    monkeypatch.setattr(df2, "_EXL3_PROJ", {})
    monkeypatch.setattr(df2, "_QUANTISED", {})
    serving._share_drafter_weights()
    assert getattr(df2.load_weights, "shared", False)
    first = df2.load_weights("/ckpt/b8", "cpu")                      # slot 0: read
    assert reads == ["/ckpt/b8"] and "fc.weight" in first
    built = {"fc.weight": torch.ones(1, dtype=torch.bfloat16), "fc.exl3": "blk"}
    df2._EXL3_PROJ["/ckpt/b8|/exl3/b8.safetensors"] = built          # slot 0's arm built
    assert df2.load_weights("/ckpt/b8", "cpu") is built              # a side slot: no read
    df2._QUANTISED["/ckpt/b16"] = q = {"fc.weight": torch.ones(1, dtype=torch.bfloat16)}
    assert df2.load_weights("/ckpt/b16", "cpu") is q
    assert reads == ["/ckpt/b8"]
    df2.load_weights("/ckpt/b8", "cpu", torch.float32)               # another dtype: read
    assert reads == ["/ckpt/b8", "/ckpt/b8"]
    serving._share_drafter_weights()                                  # idempotent


def test_a_side_slot_takes_the_draft_head_slot_0_holds(monkeypatch):
    monkeypatch.setattr(df2, "DRAFT_HEAD_EXL3", "/heads/k4")
    monkeypatch.setattr(df2, "_EXL3_HEADS", {(111, "/heads/k4"): "blk", (222, "/other"): "x"})

    class E:
        pass
    e2 = E()
    serving._share_draft_head(e2)
    assert df2._EXL3_HEADS[(id(e2), "/heads/k4")] == "blk"


def test_slot_0s_head_is_read_before_a_side_slot_takes_it(monkeypatch):
    monkeypatch.setattr(df2, "DRAFT_HEAD_EXL3", "/heads/k4")
    heads = {}
    monkeypatch.setattr(df2, "_EXL3_HEADS", heads)

    class E:
        pass
    e0, e2 = E(), E()
    monkeypatch.setattr(df2, "exl3_draft_head", lambda eng: heads.setdefault((id(eng), "/heads/k4"), "blk0"))
    serving._share_draft_head(e2, e0)
    assert heads[(id(e2), "/heads/k4")] == "blk0" and heads[(id(e0), "/heads/k4")] == "blk0"


def test_no_head_yet_leaves_the_slot_to_read_its_own(monkeypatch):
    monkeypatch.setattr(df2, "DRAFT_HEAD_EXL3", "/heads/k4")
    monkeypatch.setattr(df2, "_EXL3_HEADS", {})

    class E:
        pass
    e2 = E()
    serving._share_draft_head(e2)
    assert df2._EXL3_HEADS == {}


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
