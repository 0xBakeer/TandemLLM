"""On the board (SPD-249): a prompt whose head ends off the prefill grid (the conductor's notes at the
head, `QWEN38_NOTES_HEAD`) resumes at the head's mark bit for bit (engine/cache.py `Grid`).

One process, the served engine of the served model (the server's own loader, `app._load`, with the
command line `ops/start.sh` gives the server), its drafter and a state store of its own. Two prompts
share a head of `--head` rows (off the `chunk` grid) and part after it:

  * A is prefilled cold with the mark at the head's end: the store takes a snapshot at the mark;
  * B is prefilled warm (it resumes at the mark) and then cold (no store, the same mark): the last-row
    logits, every KV row, the recurrent state and the convolution tails -- `torch.equal`, through
    the snapshot the store would hold (`cache.capture`);
  * the control: B cold on the 0-anchored grid (no mark) next to B cold with the mark. Different cuts
    are a different arithmetic order; the line says whether they agree, it is no check.

Run on the box with :8000 stopped (the test loads its own copy of the weights), the served environment
and the server's flags:
    set -a; . ops/serve.env; set +a
    python tests/gpu/test_notes_head_grid_gpu.py [--head 3000 --tails 2000,2500] -- <server flags>
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from engine import cache  # noqa: E402
from server import app  # noqa: E402


def _tensors(snap) -> list:
    return [x for x in list(snap.views) + list(snap.state) if torch.is_tensor(x)]


def _prefill(eng, drafter, ids, store, chunk, mark):
    eng.reset()
    if drafter is not None:
        drafter.reset()
    with torch.no_grad():
        lg, reused, fwd = cache.prefill(eng, drafter, ids, eng_device(eng), store=store, chunk=chunk,
                                        checkpoint=store is not None, grid_mark=mark)
    torch.cuda.synchronize()
    return lg.clone(), reused, cache.capture(eng)          # the engine's rows and state


def eng_device(eng):
    return getattr(eng, "device", None) or app.STATE.get("device", "cuda")


def main() -> int:
    argv = sys.argv[1:]
    own, served = (argv[:argv.index("--")], argv[argv.index("--") + 1:]) if "--" in argv else (argv, [])
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", type=int, default=3000)
    ap.add_argument("--tails", default="2000,2500")
    ap.add_argument("--seed", type=int, default=249)
    o = ap.parse_args(own)
    a = app.parser().parse_args(served)
    app._thinking_default(a)                 # what `app.main` does before its load
    app._limits_at_start(a)
    app._load(a)
    eng, drafter = app.STATE["engine"], app.STATE["drafter"]
    chunk = int(app.STATE.get("prefix_chunk") or 4096)
    if o.head % chunk == 0:
        print(f"--head {o.head} is on the {chunk}-row grid: no mark to test")
        return 2
    vocab = int(getattr(getattr(eng, "cfg", None), "vocab_size", 0) or 150000)
    rng = random.Random(o.seed)
    hi = min(vocab - 1, 100000)
    head = [rng.randint(1000, hi) for _ in range(o.head)]
    ta, tb = (int(x) for x in o.tails.split(","))
    A = head + [rng.randint(1000, hi) for _ in range(ta)]
    B = head + [rng.randint(1000, hi) for _ in range(tb)]
    store = cache.StateStore(64 << 30, chunk=chunk)
    _, reused_a, _ = _prefill(eng, drafter, A, store, chunk, o.head)
    lines = sorted(store.report()["boundaries"])
    warm, reused_b, snap_w = _prefill(eng, drafter, B, store, chunk, o.head)
    cold, _, snap_c = _prefill(eng, drafter, B, None, chunk, o.head)
    bad = []
    if reused_a != 0:
        bad.append(f"A reused {reused_a}")
    if o.head not in lines:
        bad.append(f"no snapshot at the mark {o.head}: {lines}")
    if reused_b != o.head:
        bad.append(f"B resumed at {reused_b}, not at the mark {o.head}")
    if not torch.equal(warm, cold):
        bad.append(f"logits differ: max|d| {(warm.float() - cold.float()).abs().max().item():.3e}")
    tw, tc = _tensors(snap_w), _tensors(snap_c)
    if len(tw) != len(tc):
        bad.append(f"snapshot parts {len(tw)} vs {len(tc)}")
    for i, (x, y) in enumerate(zip(tw, tc)):
        if x.shape != y.shape or not torch.equal(x, y):
            bad.append(f"snapshot part {i} differs")
    plain, _, _ = _prefill(eng, drafter, B, None, chunk, None)
    agree = torch.equal(plain, cold)
    print(f"head {o.head} (grid {chunk}), tails {ta}/{tb}: store lines {lines}; B resumed at {reused_b}; "
          f"warm == cold (marked): {not bad}; control, 0-grid cold == marked cold: {agree}")
    for b in bad:
        print("FAIL", b)
    print("PASS" if not bad else "FAIL")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
