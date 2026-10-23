"""SPD-165: a side slot reads no drafter checkpoint and no draft head again.

`dflash2.load_weights` filed its cache under the last tensor's name (its loop variable took the key's
name), so every side slot read both arms' BF16 checkpoints again: 38-40 s a slot, and the copies stayed
on the device. The 27B adapter keeps one copy per (snapshot, device, dtype); the EXL3 draft head slot 0
holds is handed to each side slot's engine."""

from __future__ import annotations

import os
import tempfile

import torch
from safetensors.torch import save_file

from engine.drafters import dflash2 as df2
from engine.models.qwen3_5 import serving


def test_the_checkpoint_is_read_once_for_every_slot(monkeypatch):
    reads = []
    orig = df2.load_weights

    def counting(snapshot, device="cuda", dtype=torch.bfloat16):
        reads.append(snapshot)
        return orig(snapshot, device, dtype)
    monkeypatch.setattr(df2, "load_weights", counting)
    serving._share_drafter_weights()
    assert getattr(df2.load_weights, "shared", False)
    with tempfile.TemporaryDirectory() as d:
        save_file({"fc.weight": torch.ones(2, 2), "layers.0.x": torch.zeros(3)}, os.path.join(d, "m.safetensors"))
        a = df2.load_weights(d, "cpu")
        b = df2.load_weights(d + "/", "cpu")
        c = df2.load_weights(d, "cpu", torch.float32)
    assert a is b and a is not c
    assert len(reads) == 2, reads              # bf16 once, fp32 once
    assert a["fc.weight"].dtype == torch.bfloat16
    serving._share_drafter_weights()            # idempotent: no second wrapper


def test_a_side_slot_takes_the_draft_head_slot_0_holds(monkeypatch):
    monkeypatch.setattr(df2, "DRAFT_HEAD_EXL3", "/heads/k4")
    monkeypatch.setattr(df2, "_EXL3_HEADS", {(111, "/heads/k4"): "blk", (222, "/other"): "x"})

    class E:
        pass
    e2 = E()
    serving._share_draft_head(e2)
    assert df2._EXL3_HEADS[(id(e2), "/heads/k4")] == "blk"


def test_no_head_yet_leaves_the_slot_to_read_its_own(monkeypatch):
    monkeypatch.setattr(df2, "DRAFT_HEAD_EXL3", "/heads/k4")
    monkeypatch.setattr(df2, "_EXL3_HEADS", {})

    class E:
        pass
    e2 = E()
    serving._share_draft_head(e2)
    assert df2._EXL3_HEADS == {}
