"""SPD-261, on the board: a verify block just under a context-class edge, the next conversation's capture, and every
kind of failed capture (verify, commit, draft) on real CUDA graphs.

THE FAULT (2026-11-01, SRV-765, fd627ac3). A 13-row tree whose deepest row sat at 4,095 (class 1,024) had no graph of
(13, 1,024); `verify` captured one over a chain of 13 rows, whose last row (4,097) is class 2,048, and the warm-up's
`class_check` raised `ClassError`. The uncaptured graph stayed in `vgraphs` and the slot's next two requests replayed
it: `'NoneType' object has no attribute 'replay'`.

One process, the served engine of the served model (the server's own loader, `app._load`, with the command line
`ops/start.sh` gives the server). Each case prefills a random prompt, decodes the control (greedy, each step's logits
kept) and puts the state back; then:

  edge     a 13-row tree from 4,085 (rows 0-10 the control's chain, rows 11-12 two more children of row 9) without its
           (13, 1,024) graph: no error, no graph kept, rows 0-10 pick the control's tokens; row 11 (a sibling at slot
           4,096, past the class edge) picks what a second control decoding along its path picks; the moved path
           through row 11 is committed and 13 greedy tokens follow that control;
  short    a 2,066-token conversation (class 1,024), the same width as a chain: its graph is captured and kept;
  mtp      a draft whose chain capture raises inside the capture: no graph kept, the draft runs eagerly; the next
           launch of that key (the head rewound) captures and replays it, and both drafts are bit-equal;
  cwarm    a commit whose capture warm-up raises after it applied the commit: no graph kept, the state put back, the
           commit runs eagerly (once), and 13 tokens follow the control;
  vwarm    after a committed chain (its GDN rows parked), a verify whose capture warm-up raises at the third GDN layer
           (the parked rows folded into two layers, the count not zeroed): the state put back, the block verifies
           eagerly, its rows pick the control's tokens, its commit and 13 tokens follow;
  vraise   a verify whose capture raises inside the captured call: the block verifies eagerly;
  vinval   a verify whose capture is invalidated (a synchronize inside it: the capture's end raises, and torch leaves
           the thread on its capture stream): the thread back on the default stream, the block verifies eagerly;
  after    the next conversation captures that width again and replays it.

Logits are compared with the decode step's bit for bit (each failure case gates on them as on the tokens).

Run on the box with :8000 stopped (the test loads its own copy of the weights), the served environment and the
server's flags:
    set -a; . ops/serve.env; set +a
    python tests/gpu/test_verify_class_edge_gpu.py [--edge 4085 --short 2066] -- <server flags>
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


def _decode(eng, tok: int, n: int) -> tuple[list[int], list[torch.Tensor]]:
    out, lgs = [], []
    for _ in range(n):
        lg = eng.forward(torch.tensor([tok]), int(eng.kv.length), last_only=True)
        lgs.append(lg.view(-1).clone())
        tok = int(lg.view(-1).argmax())
        out.append(tok)
    torch.cuda.synchronize()
    return out, lgs


def _forced(eng, forced: list[int], n: int) -> tuple[list[int], list[torch.Tensor]]:
    """The decode step over `forced`, then greedy: the greedy token after the last forced one and n more, with the
    logits each came from."""
    lg = None
    for t in forced:
        lg = eng.forward(torch.tensor([t]), int(eng.kv.length), last_only=True)
    first = int(lg.view(-1).argmax())
    out, lgs = _decode(eng, first, n)
    return [first] + out, [lg.view(-1).clone()] + lgs


def _prompt(eng, drafter, n: int, seed: int, chunk: int):
    vocab = int(getattr(getattr(eng, "cfg", None), "vocab_size", 0) or 150000)
    rng = random.Random(seed)
    ids = [rng.randint(1000, min(vocab - 1, 100000)) for _ in range(n)]
    eng.reset()
    if drafter is not None:
        drafter.reset()
    lg, _, _ = cache.prefill(eng, drafter, ids, app.STATE.get("device", "cuda"), store=None, chunk=chunk,
                             checkpoint=False)
    return int(lg.view(-1).argmax())


def _same(lg: torch.Tensor, rows: list[torch.Tensor], k: int) -> int:
    return sum(int(torch.equal(lg[i].float(), rows[i].float())) for i in range(k))


def _default_stream() -> bool:
    return torch.cuda.current_stream() == torch.cuda.default_stream()


class _Once:
    """Patch `obj.name` on the instance with `wrap(real, *args)` until it says it fired; then the real one again."""

    def __init__(self, obj, name: str, wrap):
        self.obj, self.name, self.fired = obj, name, False
        self.prev = obj.__dict__.get(name)              # an instance patch already there (the runner's timers)
        real = getattr(obj, name)

        def call(*a, **k):
            if self.fired:
                return real(*a, **k)
            return wrap(self, real, *a, **k)
        setattr(obj, name, call)

    def off(self) -> None:
        if self.prev is not None:
            setattr(self.obj, self.name, self.prev)
        else:
            self.obj.__dict__.pop(self.name, None)


def _warm() -> bool:
    """A capture's warm-up runs on a stream of its own, outside the capture."""
    return not torch.cuda.is_current_stream_capturing() and not _default_stream()


def _case(eng, drafter, name: str, n: int, parents: list[int], seed: int, chunk: int, bad: list, drop: bool) -> str:
    """Prefill n tokens, the control, then the block (`parents`; rows 0..k the control's chain) and a commit. `drop`:
    the block's key starts without graphs (the round's case); else what the earlier case left is kept. A block that is
    not a chain (the edge's) also has row 11 checked against a second control along its path, and commits the moved
    path through it."""
    fn = eng.fn
    M = len(parents)
    depth = [0] * M
    for i, p in enumerate(parents):
        depth[i] = 0 if p < 0 else depth[p] + 1
    chain = [i for i in range(M) if i == 0 or parents[i] == i - 1]          # rows 0..k, one per depth
    k = len(chain)
    tok = _prompt(eng, drafter, n, seed, chunk)
    L = int(eng.kv.length)
    kept = cache.capture(eng, drafter)
    ctrl, ctrl_lg = _decode(eng, tok, k + 13)
    cache.restore(eng, kept, drafter)
    cls = fn.qsa_class(L + max(depth))
    key = (M, cls)
    if drop:
        eng.vgraphs.pop(key, None)
        for c in [c for c in eng.cgraphs if c[0] == M and c[-1] == cls]:
            eng.cgraphs.pop(c)
    toks = [tok] + ctrl[:k - 1]
    block = toks + [(toks[-1] + 1 + j) % 100000 + 1000 for j in range(M - k)]
    sib = None
    if M > k:
        # row k: a sibling of row k - 1 (a child of row k - 2); the second control decodes along its path
        assert parents[k] == k - 2, parents
        c2, c2_lg = _forced(eng, toks[:k - 1] + [block[k]], 13)
        cache.restore(eng, kept, drafter)
        sib = (k, c2, c2_lg)
    eager0 = eng.stats.get("verify_eager", 0)
    try:
        lg = eng.forward_tree(torch.tensor(block), parents, L)
    except Exception as exc:                                                 # noqa: BLE001 (fd627ac3: ClassError)
        bad.append(f"{name}: verify raised {type(exc).__name__}: {exc}")
        return f"{name}: {type(exc).__name__}"
    picks = eng.picks.tolist()
    torch.cuda.synchronize()
    want = ctrl[:k]
    if picks[:k] != want:
        j = next(i for i, (a, b) in enumerate(zip(picks, want)) if a != b)
        bad.append(f"{name}: verify picks differ from the decode step's at row {j}")
    lg_same = _same(lg, ctrl_lg, k)
    g = eng.vgraphs.get(key)
    graph = "none" if g is None else ("captured" if g.graph is not None else "HALF-MADE")
    eager = eng.stats.get("verify_eager", 0) - eager0
    if g is not None and g.graph is None:
        bad.append(f"{name}: a half-made graph of {key} is kept")
    if name in ("short", "after") and graph != "captured":
        bad.append(f"{name}: the block's graph {key} was not captured")
    line = ""
    if sib is not None:
        r, c2, c2_lg = sib
        s_pick, s_same = picks[r] == c2[0], bool(torch.equal(lg[r].float(), c2_lg[0].float()))
        if not s_pick:
            bad.append(f"{name}: sibling row {r} picks {picks[r]}, its control {c2[0]}")
        path = list(range(k - 1)) + [r]
        eng.commit_tree(path)
        if int(eng.kv.length) != L + k:
            bad.append(f"{name}: the moved commit left {eng.kv.length} rows, not {L + k}")
        nxt, nxt_lg = _decode(eng, c2[0], 13)
        ok = nxt == c2[1:14]
        if not ok:
            bad.append(f"{name}: the 13 tokens after the moved commit differ from the sibling control's")
        after_same = sum(int(torch.equal(a, b)) for a, b in zip(nxt_lg, c2_lg[1:14]))
        line = (f"; sibling row {r} (slot {L + M - 2}) pick == its control {s_pick} (logits bit-equal {s_same}); "
                f"moved commit through it, 13 tokens == its control {ok} (logits bit-equal {after_same}/13)")
    else:
        eng.commit_tree(list(range(k)))
        if int(eng.kv.length) != L + k:
            bad.append(f"{name}: the commit left {eng.kv.length} rows, not {L + k}")
        nxt, nxt_lg = _decode(eng, want[-1], 13)
        ok = nxt == ctrl[k:k + 13]
        if not ok:
            bad.append(f"{name}: the 13 tokens after the commit differ from the control's")
        after_same = sum(int(torch.equal(a, b)) for a, b in zip(nxt_lg, ctrl_lg[k:k + 13]))
        line = f"; after the commit 13 tokens == control {ok} (logits bit-equal {after_same}/13)"
    return (f"{name}: {M} rows from {L} (deepest {L + max(depth)}, class {cls}; chain to {L + M - 1}, class "
            f"{fn.qsa_class(L + M - 1)}): eager verifies {eager}, graph {key} {graph}; picks == decode {picks[:k] == want} "
            f"({k} rows, logits bit-equal {lg_same}/{k})" + line)


def _fail_case(eng, drafter, name: str, n: int, M: int, seed: int, chunk: int, bad: list, arm, lead: int = 0,
               commit: int = 0) -> str:
    """A chain of M rows (the control's tokens) after `lead` committed rows, whose verify (or, with `commit` > 0,
    whose commit of that many rows) captures a graph that `arm(fn)` makes fail. Gates: no graph kept for the failed
    key, the thread on the default stream, picks and logits bit-equal to the decode step's, the commit's length, and
    13 tokens and their logits after it equal to the control's."""
    fn = eng.fn
    tok = _prompt(eng, drafter, n, seed, chunk)
    L0 = int(eng.kv.length)
    kept = cache.capture(eng, drafter)
    keep = commit or M
    ctrl, ctrl_lg = _decode(eng, tok, lead + M + 13)
    cache.restore(eng, kept, drafter)
    seq = [tok] + ctrl                       # the token at L0 + i is seq[i]; its greedy successor ctrl[i]
    if lead:
        eng.forward_tree(torch.tensor(seq[:lead]), list(range(-1, lead - 1)), L0)
        eng.commit_tree(list(range(lead)))
    L = int(eng.kv.length)
    cls = fn.qsa_class(L + M - 1)
    if commit:
        ck = (M, commit, False, cls)
        eng.cgraphs.pop(ck, None)
    else:
        eng.vgraphs.pop((M, cls), None)
        for c in [c for c in eng.cgraphs if c[0] == M and c[-1] == cls]:
            eng.cgraphs.pop(c)
    eager0, cap0 = eng.stats.get("verify_eager", 0), len(eng.vgraphs)
    once = arm(fn)
    try:
        lg = eng.forward_tree(torch.tensor(seq[lead:lead + M]), list(range(-1, M - 1)), L)
        torch.cuda.synchronize()
        picks = eng.picks.tolist()
        stream_ok = _default_stream()
        eng.commit_tree(list(range(keep)))
        stream_ok = stream_ok and _default_stream()
    except Exception as exc:                                                 # noqa: BLE001
        once.off()
        bad.append(f"{name}: the round raised {type(exc).__name__}: {exc}")
        return f"{name}: {type(exc).__name__}: {str(exc)[:200]}"
    once.off()
    fired = once.fired
    want = ctrl[lead:lead + M]
    same = _same(lg, ctrl_lg[lead:], M)
    eager = eng.stats.get("verify_eager", 0) - eager0
    if commit:
        left = ck in eng.cgraphs
        what = f"commit graph {ck} kept {left}"
    else:
        left = (M, cls) in eng.vgraphs
        what = f"verify graph {(M, cls)} kept {left}, eager verifies {eager}"
    if not fired:
        bad.append(f"{name}: the injected failure never fired")
    if left:
        bad.append(f"{name}: the failed capture's graph is kept")
    if not stream_ok:
        bad.append(f"{name}: the thread is not on the default stream after the round")
    if picks != want or same != M:
        bad.append(f"{name}: picks == decode {picks == want}, logits bit-equal {same}/{M}")
    if int(eng.kv.length) != L + keep:
        bad.append(f"{name}: the commit left {eng.kv.length} rows, not {L + keep}")
    nxt, nxt_lg = _decode(eng, ctrl[lead + keep - 1], 13)
    want2 = ctrl[lead + keep:lead + keep + 13]
    after_same = sum(int(torch.equal(a, b)) for a, b in zip(nxt_lg, ctrl_lg[lead + keep:lead + keep + 13]))
    if nxt != want2 or after_same != 13:
        bad.append(f"{name}: after the commit 13 tokens == control {nxt == want2}, logits bit-equal {after_same}/13")
    return (f"{name}: {M} rows from {L}{f' after {lead} committed' if lead else ''} (class {cls}): failure fired "
            f"{fired}; {what}; default stream {stream_ok}; picks == decode {picks == want} (logits bit-equal "
            f"{same}/{M}); commit of {keep} -> {int(eng.kv.length) - 13 == L + keep}; 13 tokens == control "
            f"{nxt == want2} (logits bit-equal {after_same}/13); new verify graphs {len(eng.vgraphs) - cap0}")


def _mtp_case(eng, drafter, n: int, seed: int, chunk: int, bad: list) -> str:
    """A draft whose chain capture raises inside the captured call: no graph kept, the draft eager; the head rewound,
    the next launch of that key captures and replays it; both drafts bit-equal."""
    from engine.methods import draft_meet
    fn = eng.fn
    head = eng.heads["mtp"]
    tok = _prompt(eng, drafter, n, seed, chunk)
    kept = cache.capture(eng, drafter)
    meet, draft_meet.MEET = draft_meet.MEET, None
    head.keyed = None
    L0, nexts = int(head.st.length), [tok, (tok + 1) % 100000 + 1000]
    rows = eng.last_streams().reshape(1, -1).expand(len(nexts), -1).contiguous()
    cls = fn.qsa_class(L0 + len(nexts) + head.depth - 1)
    key = (len(nexts), cls)
    head.graphs.pop(key, None)
    c0 = head.captures

    def raising(once, real, st_m, k, depth, cls=0, keyed=False):
        if torch.cuda.is_current_stream_capturing():
            once.fired = True
            raise RuntimeError("injected: a raise inside the draft capture")
        return real(st_m, k, depth, cls, keyed=keyed)
    once = _Once(fn, "mtp_chain_dev", raising)
    try:
        head.launch(rows, nexts)
        tok_e, prob_e = head.read()
        e_left, stream_ok, c1 = key in head.graphs, _default_stream(), head.captures
        head.st.length = L0
        head.launch(rows, nexts)
        tok_g, prob_g = head.read()
        g = head.graphs.get(key)
    except Exception as exc:                                                 # noqa: BLE001
        bad.append(f"mtp: the draft raised {type(exc).__name__}: {exc}")
        return f"mtp: {type(exc).__name__}: {str(exc)[:200]}"
    finally:
        once.off()
        draft_meet.MEET = meet
        head.st.length = L0
        cache.restore(eng, kept, drafter)
    same = tok_e == tok_g and prob_e == prob_g
    if not once.fired:
        bad.append("mtp: the injected failure never fired")
    if e_left:
        bad.append("mtp: the failed capture's graph is kept")
    if not stream_ok:
        bad.append("mtp: the thread is not on the default stream after the draft")
    if g is None or g.graph is None:
        bad.append("mtp: the next launch kept no graph")
    if not same:
        bad.append("mtp: the eager draft differs from the graph's")
    if c1 != c0 + 1:
        bad.append(f"mtp: the failed round counted {c1 - c0} captures, not 1 (the drafter would price it)")
    return (f"mtp: feed of {len(nexts)} at {L0} (class {cls}, depth {head.depth}): failure fired {once.fired}; graph "
            f"kept after it {e_left}; default stream {stream_ok}; next launch captured {g is not None}; eager draft == "
            f"graph draft {same} (tokens {[t[0] for t in tok_e]}); captures counted {c1 - c0}")


def _commit_warm_raise(fn):
    def wrap(once, real, st, key):
        real(st, key)
        if _warm():
            once.fired = True
            raise RuntimeError("injected: the commit capture's warm-up, after the commit was applied")
    return _Once(fn, "commit_dev", wrap)


def _verify_warm_raise(fn):
    seen = {"n": 0}

    def wrap(once, real, x, li, st):
        if _warm():
            seen["n"] += 1
            if seen["n"] == 3:
                once.fired = True
                raise RuntimeError("injected: the verify capture's warm-up, at its third GDN layer")
        return real(x, li, st)
    return _Once(fn, "_gdn1", wrap)


def _verify_capture_raise(fn):
    def wrap(once, real, st, M, write=False, cls=0):
        if torch.cuda.is_current_stream_capturing():
            once.fired = True
            raise RuntimeError("injected: a raise inside the verify capture")
        return real(st, M, write=write, cls=cls)
    return _Once(fn, "rows_step", wrap)


def _verify_capture_invalid(fn):
    def wrap(once, real, st, M, write=False, cls=0):
        out = real(st, M, write=write, cls=cls)
        if torch.cuda.is_current_stream_capturing():
            once.fired = True
            torch.cuda.synchronize()        # not permitted while capturing: the capture is invalidated
        return out
    return _Once(fn, "rows_step", wrap)


def main() -> int:
    argv = sys.argv[1:]
    own, served = (argv[:argv.index("--")], argv[argv.index("--") + 1:]) if "--" in argv else (argv, [])
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge", type=int, default=4085)
    ap.add_argument("--short", type=int, default=2066)
    ap.add_argument("--seed", type=int, default=261)
    o = ap.parse_args(own)
    a = app.parser().parse_args(served)
    app._thinking_default(a)
    app._limits_at_start(a)
    app._load(a)
    eng, drafter = app.STATE["engine"], app.STATE["drafter"]
    if not hasattr(eng, "vgraphs") or not getattr(eng.fn.cfg, "has_indexer", False):
        print("the served engine has no indexer classes: nothing to test")
        return 2
    chunk = int(app.STATE.get("prefix_chunk") or 4096)
    bad: list = []
    s = o.seed
    cases = [lambda: _case(eng, drafter, "edge", o.edge, list(range(-1, 10)) + [9, 9], s, chunk, bad, True),
             lambda: _case(eng, drafter, "short", o.short, list(range(-1, 12)), s + 1, chunk, bad, False),
             lambda: _mtp_case(eng, drafter, 2200, s + 2, chunk, bad),
             lambda: _fail_case(eng, drafter, "cwarm", 2300, 13, s + 3, chunk, bad, _commit_warm_raise, commit=6),
             lambda: _fail_case(eng, drafter, "vwarm", 2400, 9, s + 4, chunk, bad, _verify_warm_raise, lead=2),
             lambda: _fail_case(eng, drafter, "vraise", 2500, 5, s + 5, chunk, bad, _verify_capture_raise),
             lambda: _fail_case(eng, drafter, "vinval", 2600, 6, s + 6, chunk, bad, _verify_capture_invalid),
             lambda: _case(eng, drafter, "after", 2700, list(range(-1, 5)), s + 7, chunk, bad, False)]
    with torch.no_grad():
        for c in cases:
            print(c(), flush=True)
    for b in bad:
        print("FAIL", b)
    print("PASS" if not bad else "FAIL")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
