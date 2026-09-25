"""CPU tests for TRN-7's Draft-OPD trainer (tools/h100/train_opd.py), on a tiny random drafter.

The packed walk is what makes the on-policy part affordable -- every candidate anchor drafted in a
few packed passes instead of one call a block -- and it is only worth anything if it walks the loop
the one-block-at-a-time reference walks: the same anchors, the same first misses. That is tested
on labels built so the reference walk sees every run length. The rest is the arithmetic the loss
and the gate are made of: the batched lattice and walk against the module's own, the first-miss
table, the per-slot rates, the slot weights, and the weighted loss against a hand computation.
"""

from __future__ import annotations

import os
import random
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.drafters.dflash2 import DFlash2Config, DFlash2Module  # noqa: E402
from tools.h100.train_opd import (  # noqa: E402
    anchor_pass, drafts_at, first_miss, lattice_batch, opd_loss, parse_slot_w, summarise,
    walk_batch, walk_sequence,
)
from tools.train_dflash2 import Sample  # noqa: E402

V, H, K = 40, 32, 4


def tiny(block: int = 8, seed: int = 0) -> tuple[DFlash2Module, torch.Tensor, torch.Tensor]:
    raw = {"hidden_size": H, "intermediate_size": 64, "num_hidden_layers": 2,
           "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8, "vocab_size": V,
           "rms_norm_eps": 1e-6, "layer_types": ["sliding_attention"] * 2, "sliding_window": 2048,
           "dflash_config": {"block_size": block, "conv_kernel_size": 2, "conv_group_size": 8,
                             "mask_token_id": V - 1, "selector_rank": 8, "selector_top_k": K,
                             "target_layer_ids": [1, 2, 3, 4, 5]}}
    cfg = DFlash2Config(raw)
    g = torch.Generator().manual_seed(seed)
    w = {}
    for name, shape in cfg.expected_tensors().items():
        if name.endswith("norm.weight") or name.endswith("_norm.weight") or name.endswith("layernorm.weight"):
            w[name] = torch.ones(shape)
        else:
            w[name] = torch.randn(shape, generator=g) * 0.3
    embed = torch.randn(V, H, generator=g)
    head = torch.randn(V, H, generator=g)
    return DFlash2Module(cfg, w), embed, head


def sample(n: int = 90, gen_start: int = 20, seed: int = 1, name: str = "s0") -> Sample:
    g = torch.Generator().manual_seed(seed)
    blob = {"ids": torch.randint(0, V - 1, (n,), generator=g),
            "fused": torch.randn(n, 5 * H, generator=g),
            "label": torch.randint(0, V - 1, (n,), generator=g),
            "top_ids": torch.randint(0, V - 1, (n, 8), generator=g),
            "top_lp": torch.randn(n, 8, generator=g)}
    meta = {"name": name, "kind": "gen", "topic": "chat", "split": "train", "gen_start": gen_start}
    return Sample(meta, blob, "cpu", "cpu")


def plant_runs(m, embed, head, s: Sample, block: int, seed: int = 3) -> list[int]:
    """Rewrite `s.label` so the loop walked from the first anchor accepts a chosen run at each block:
    the drafter's own chain up to the run, then a different token. Drafts do not read labels, so
    this fixes the walk the reference must find. Returns the planted runs."""
    rng = random.Random(seed)
    l = block - 1
    p, n, runs = max(1, s.gen_start - 1), len(s), []
    while p <= n - block:
        d = drafts_at(m, embed, head, s, torch.tensor([p]), "cpu", block=block)[0]
        r = rng.choice(range(l + 1))
        s.label[p:p + r] = d[:r]
        if r < l:
            s.label[p + r] = (int(d[r]) + 1) % (V - 1)
        runs.append(r)
        p += r + 1
    return runs


def test_packed_walk_is_the_reference_walk():
    for block in (8, 16):
        m, embed, head = tiny(block)
        s = sample(n=120, seed=block)
        runs = plant_runs(m, embed, head, s, block)
        assert len(set(runs)) > 3, "the planted walk should see several run lengths"
        ref = anchor_pass(m, embed, head, [s], "cpu", block=block)[s.name]
        assert [r for _, r in ref] == runs
        for chunk in (1, 7, 64):
            got = walk_sequence(m, embed, head, s, "cpu", block=block, chunk=chunk)
            assert got == ref, (block, chunk)


def test_batched_lattice_and_walk_are_the_modules():
    m, embed, head = tiny(8)
    g = torch.Generator().manual_seed(5)
    b, l = 6, 7
    pred = torch.randn(b, l, H, generator=g)
    logits = F.linear(pred.reshape(-1, H), head)
    cand, unary = m.unary_candidates(logits)
    cand, unary = cand.view(b, l, -1), unary.view(b, l, -1)
    anchors = torch.randint(0, V - 1, (b,), generator=g)
    sc = lattice_batch(m, pred, cand, unary, anchors)
    toks = walk_batch(cand, sc)
    for i in range(b):
        ref = m.lattice(pred[i], cand[i], unary[i], int(anchors[i]))
        assert torch.allclose(sc[i], ref, atol=1e-5)
        assert torch.equal(toks[i], m.walk(cand[i], ref))


def test_first_miss_table():
    label = torch.arange(30)
    anchors = torch.tensor([0, 5, 10])
    d = torch.stack([label[0:3], label[5:8], label[10:13]]).clone()   # label[p : p+3]: all right
    d[1, 1] = 99
    d[2, 0] = 99
    assert first_miss(d, label, anchors).tolist() == [3, 1, 0]


def test_summarise_counts_rounds_and_slots():
    s = sample()
    walks = {s.name: [(19, 0), (20, 2), (23, 3)]}                  # block 4: 3 drafted slots
    ev = summarise(walks, [s], 4)
    assert ev["blocks"] == 3
    assert abs(ev["ALL"] - (1 + 3 + 4) / 3) < 1e-9 and abs(ev["prose"] - ev["ALL"]) < 1e-9
    # slot 1 offered 3 accepted 2; slot 2 offered 2 accepted 2; slot 3 offered 2 accepted 1
    assert ev["a"] == [2 / 3, 1.0, 0.5]


def test_parse_slot_w():
    assert parse_slot_w("1:0.5,2-8:1,9-:0.5", 15) == [0.5] + [1.0] * 7 + [0.5] * 7
    assert parse_slot_w("1:0.5,2-8:1,9-:0.5", 7) == [0.5] + [1.0] * 6
    assert parse_slot_w("", 3) == [1.0, 1.0, 1.0]


def test_opd_loss_is_the_weighted_cross_entropy():
    class A:
        w_acc, w_rej, w_past, kl = 0.3, 1.0, 0.2, 0.0
    s = sample()
    g = torch.Generator().manual_seed(9)
    pred = torch.randn(2, 3, H, generator=g)
    head = torch.randn(V, H, generator=g)
    anchors, rejects = torch.tensor([20, 30]), torch.tensor([1, 3])
    sw = torch.tensor([0.5, 1.0, 2.0])
    loss, _ = opd_loss(pred, head, s, anchors, rejects, A, "cpu", sw)
    want_num = want_den = 0.0
    for b in range(2):
        for j in range(3):
            lg = F.linear(pred[b, j], head)
            ce = float(F.cross_entropy(lg[None], s.label[anchors[b] + j][None]))
            r = int(rejects[b])
            wt = (A.w_acc if j < r else A.w_rej if j == r else A.w_past) * float(sw[j])
            want_num += wt * ce
            want_den += wt
    assert abs(float(loss) - want_num / want_den) < 1e-5


def _main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = 0
    for fn in fns:
        try:
            fn()
            print(f"  ok  {fn.__name__}")
        except Exception as exc:                                    # noqa: BLE001
            bad += 1
            import traceback
            traceback.print_exc()
            print(f"FAIL  {fn.__name__}: {exc}")
    print(f"\n{len(fns) - bad}/{len(fns)} passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(_main())
