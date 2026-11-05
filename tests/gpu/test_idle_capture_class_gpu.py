"""On the board (SPD-249): an idle capture never takes a class below the state's own, and leaves the state as it was.

THE FAULT (2026-10-30). A 2,013-token turn started in class 0 (every row dense), found a verify width without
graphs and was owed (`capture_cost`); it decoded past 2,048. The idle window then captured class 0's graph of that
width on the 2,268-token state: its rows past the dense budget are sparse (`_stage_qsa`) while a class-0 launch
selects no blocks, so `attn_rows` read K/V rows at whatever `sel` held, an illegal memory access in the capture's
warm-up, and the engine was gone.

One process, the served engine of the served model (the server's own loader, `app._load`, with the command line
`ops/start.sh` gives the server) and its drafter:

  * a random 2,013-token prompt is prefilled and decoded greedily to 2,268 rows; the state is kept (`cache.capture`);
  * the control: the state goes back (`cache.restore`) and 64 more greedy tokens; the state goes back again;
  * the turn's width is owed in class 0 (as the server's rounds leave it) and the idle captures run: no graph of a
    class below the state's own is taken (on the old code this is the fault), the state after them is the kept one
    bit for bit (KV, indexer keys, recurrent sections, the drafter's cache), and the same 64 greedy tokens follow.

Run on the box with :8000 stopped (the test loads its own copy of the weights), the served environment and the
server's flags:
    set -a; . ops/serve.env; set +a
    python tests/gpu/test_idle_capture_class_gpu.py [--prompt 2013 --to 2268] -- <server flags>
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
    out = list(snap.views) + list(snap.state)
    if snap.drafter is not None:
        out += list(snap.drafter.get("views") or ())
    return [x for x in out if torch.is_tensor(x)]


def _decode(eng, tok: int, n: int) -> tuple[list[int], torch.Tensor]:
    out, lg = [], None
    for _ in range(n):
        lg = eng.forward(torch.tensor([tok]), int(eng.kv.length), last_only=True)
        tok = int(lg.view(-1).argmax())
        out.append(tok)
    torch.cuda.synchronize()
    return out, lg.clone()


def main() -> int:
    argv = sys.argv[1:]
    own, served = (argv[:argv.index("--")], argv[argv.index("--") + 1:]) if "--" in argv else (argv, [])
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=int, default=2013)
    ap.add_argument("--to", type=int, default=2268)
    ap.add_argument("--then", type=int, default=64)
    ap.add_argument("--width", type=int, default=9)
    ap.add_argument("--seed", type=int, default=249)
    o = ap.parse_args(own)
    a = app.parser().parse_args(served)
    app._thinking_default(a)
    app._limits_at_start(a)
    app._load(a)
    eng, drafter = app.STATE["engine"], app.STATE["drafter"]
    fn = eng.fn
    if not hasattr(eng, "idle_captures"):
        print("the served engine has no idle captures: nothing to test")
        return 2
    vocab = int(getattr(getattr(eng, "cfg", None), "vocab_size", 0) or 150000)
    rng = random.Random(o.seed)
    ids = [rng.randint(1000, min(vocab - 1, 100000)) for _ in range(o.prompt)]
    chunk = int(app.STATE.get("prefix_chunk") or 4096)
    eng.reset()
    if drafter is not None:
        drafter.reset()
    with torch.no_grad():
        lg, _, _ = cache.prefill(eng, drafter, ids, app.STATE.get("device", "cuda"), store=None, chunk=chunk,
                                 checkpoint=False)
        pre, _ = _decode(eng, int(lg.view(-1).argmax()), o.to - o.prompt)
        tok = pre[-1]                        # the next token, not fed yet
        L = int(eng.kv.length)
        own = fn.qsa_class(L + 64)
        low = fn.qsa_class(o.prompt)
        kept = cache.capture(eng, drafter)
        live, _ = _decode(eng, tok, o.then)           # from the state as decoded
        cache.restore(eng, kept, drafter)
        ctrl, ctrl_lg = _decode(eng, tok, o.then)     # the control: from the kept state, no idle capture
        cache.restore(eng, kept, drafter)
        # the rounds' state: the width used, owed in the turn's starting class
        eng.uses[o.width] = 10 ** 6
        eng._owed[low] = o.prompt
        made0 = set(eng.vgraphs)
        steps = eng.idle_captures()
        n = 0 if steps is None else sum(1 for _ in steps)
        torch.cuda.synchronize()
        new = sorted(set(eng.vgraphs) - made0)
        after = cache.capture(eng, drafter)
        got, got_lg = _decode(eng, tok, o.then)
    bad = []
    below = [k for k in new if k[1] < own]
    if below:
        bad.append(f"graphs of a class below the state's own ({own}) at {L}: {below}")
    if eng._owed.get(low) != o.prompt:
        bad.append(f"the owed entry of class {low} was dropped: {eng._owed}")
    ta, tb = _tensors(kept), _tensors(after)
    if len(ta) != len(tb):
        bad.append(f"snapshot parts {len(ta)} vs {len(tb)}")
    for i, (x, y) in enumerate(zip(ta, tb)):
        if x.shape != y.shape or not torch.equal(x, y):
            bad.append(f"state part {i} moved in the idle captures")
    if got != ctrl or not torch.equal(got_lg, ctrl_lg):
        k = next((i for i, (x, y) in enumerate(zip(got, ctrl)) if x != y), None)
        bad.append(f"tokens after the idle captures differ from the control (first at {k})")
    print(f"state {L} rows (class {own}), owed class {low} width {o.width}; idle captures: {n} steps, new graphs "
          f"{new}; state after == kept: {not any('state part' in b for b in bad)}; {o.then} tokens == control: "
          f"{got == ctrl} (control == as decoded: {ctrl == live})")
    for b in bad:
        print("FAIL", b)
    print("PASS" if not bad else "FAIL")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
