"""The server's idle graph captures (`graphs_once`, the capture gap of 2026-10-12): an engine that names
the graphs its context class lacks (`idle_captures`, Flash-Next's) gets them captured in the idle
window, one a step, under the engine lock, and a request that waits stops it between two.

Run: python tests/test_idle_graphs.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import test_forward_tree as T  # noqa: E402,F401  (sets the CPU environment before the engine imports)

from server import app  # noqa: E402


class _Eng:
    """Owes `n` graphs; each step takes one."""

    def __init__(self, n):
        self.owed, self.kv, self.locked = n, type("KV", (), {"length": 32_000})(), []

    def idle_captures(self, ahead=True, wants=None):
        if not self.owed:
            return None
        return self._steps()

    def _steps(self):
        while self.owed:
            self.locked.append(app.LOCK.locked())
            self.owed -= 1
            yield


def _with(eng, idle_for=5.0, drafter=None):
    app.STATE["engine"], app.STATE["drafter"] = eng, drafter
    app.STATE["last_end_mono"] = time.monotonic() - idle_for
    app.STATE.pop("draining", None)


def test_an_idle_engine_captures_what_its_class_lacks_under_the_lock():
    eng = _Eng(3)
    _with(eng)
    assert app.graphs_once() == "captured" and eng.owed == 0 and eng.locked == [True] * 3
    assert not app.LOCK.locked()
    assert app.graphs_once() == "nothing"
    return "3 graphs under the engine lock, the lock released; then nothing"


def test_a_waiting_request_stops_it_and_a_recent_one_holds_it():
    eng = _Eng(5)
    _with(eng, idle_for=0.0)
    assert app.graphs_once() == "recent" and eng.owed == 5
    _with(eng)
    real, calls = app._waiting, {"n": 0}

    def waiting():
        calls["n"] += 1
        return calls["n"] >= 2                     # a request arrives after the second graph
    app._waiting = waiting
    try:
        assert app.graphs_once() == "stopped"
    finally:
        app._waiting = real
    assert eng.owed == 3 and not app.LOCK.locked()
    assert app.graphs_once() == "captured" and eng.owed == 0
    return "recent: none; a waiter stops it after 2 of 5, the next look takes the rest"


def test_spd273_idle_captures_pause_while_the_reserve_is_at_the_peak():
    """SPD-273: each capture keeps a private pool no shed or `empty` lowers; on 2026-11-02 the idle windows took 635
    of 1,390 captures and one window moved the reserve from 76.1 to 80.0 GiB over a 75.8 GiB peak. With the memory
    policy on, no idle capture while the reserve plus the smallest prefill's workspace is over the peak ("peak", one
    line until the reserve is back under it); under it, captured as before; without a policy, as before."""
    import contextlib
    import io
    GiB = 1 << 30

    class _Mem:
        on, ws_floor, peak = True, int(2.75 * GiB), int(75.82 * GiB)

        def __init__(self, res):
            self.res = res

        def memory(self):
            return 60 * GiB, int(self.res * GiB)

    eng = _Eng(3)
    _with(eng)
    mem = _Mem(74.0)
    app.STATE["mem"] = mem
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            assert app.graphs_once() == "peak" and eng.owed == 3
            assert app.graphs_once() == "peak" and eng.owed == 3
        lines = [ln for ln in out.getvalue().splitlines() if ln.startswith("[graphs] idle captures paused")]
        assert lines == ["[graphs] idle captures paused: reserve 74.0 GiB + the workspace floor 2.75 > peak 75.8"], \
            out.getvalue()
        assert not app.LOCK.locked()
        mem.res = 70.0                              # 70 + 2.75 under 75.82: captured as before
        assert app.graphs_once() == "captured" and eng.owed == 0
        eng.owed, mem.res = 2, 74.0
        with contextlib.redirect_stdout(out):
            assert app.graphs_once() == "peak" and eng.owed == 2
        assert len([ln for ln in out.getvalue().splitlines() if "idle captures paused" in ln]) == 2, \
            "said again after the reserve was back under the peak"
        mem.on = False                              # the policy off: the peak is no bound
        assert app.graphs_once() == "captured" and eng.owed == 0
    finally:
        app.STATE.pop("mem", None)
        app.STATE.pop("graphs_peak_said", None)
    eng.owed = 1
    assert app.graphs_once() == "captured" and eng.owed == 0
    return "74.0 + 2.75 > 75.8: paused, one line; 70.0: captured; policy off or none: captured"


def test_spd273_a_window_stops_before_the_capture_after_the_reserve_crosses_the_bound():
    """SPD-273: the bound was read once, before the window; one window of 222 graphs took the reserve from 67.6 to
    74.2 GiB (2026-11-03), over the 75.82 GiB peak less the 2.75 floor (73.07), so a window that starts just under
    the bound ended over it. Now it is read again before each capture: from 72.0 GiB with 0.4 GiB a graph, the
    third graph takes the reserve to 73.2 and the window stops there ("peak", one line), the fourth is not taken;
    the next look says nothing more. A read that raises inside the window stops it too."""
    import contextlib
    import io
    GiB = 1 << 30

    class _Mem:
        on, ws_floor, peak = True, int(2.75 * GiB), int(75.82 * GiB)

        def __init__(self, eng, start, per):
            self.eng, self.start, self.per, self.reads, self.fail_at = eng, start, per, [], None

        def memory(self):
            if self.fail_at is not None and len(self.reads) >= self.fail_at:
                raise RuntimeError("no reading")
            res = self.start + self.per * len(self.eng.locked)      # one entry a capture taken
            self.reads.append(round(res, 2))
            return 60 * GiB, int(res * GiB)

    eng = _Eng(5)
    _with(eng)
    mem = _Mem(eng, 72.0, 0.4)
    app.STATE["mem"] = mem
    app.STATE.pop("graphs_peak_said", None)
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            got = app.graphs_once()
            assert got == "peak" and eng.owed == 2 and eng.locked == [True] * 3, (got, eng.owed, out.getvalue())
            assert mem.reads == [72.0, 72.4, 72.8, 73.2], mem.reads
            assert not app.LOCK.locked()
            assert app.graphs_once() == "peak" and eng.owed == 2 and len(mem.reads) == 5
        lines = out.getvalue().splitlines()
        paused = [ln for ln in lines if ln.startswith("[graphs] idle captures paused")]
        assert paused == ["[graphs] idle captures paused: reserve 73.2 GiB + the workspace floor 2.75 > peak 75.8"], \
            out.getvalue()
        assert [ln for ln in lines if ln.startswith("[graphs] idle captures over")][0].startswith(
            "[graphs] idle captures over 1 engine(s): 3 graphs, peak, "), out.getvalue()
        mem.start, mem.per = 70.0, 0.0             # back under: the rest is captured, the flag cleared
        assert app.graphs_once() == "captured" and eng.owed == 0
        assert app.STATE.get("graphs_peak_said") is False
        eng.owed, eng.locked, mem.reads, mem.fail_at = 3, [], [], 1     # the read after the first graph raises
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            assert app.graphs_once() == "peak" and eng.owed == 2 and eng.locked == [True]
        assert "no reading" in err.getvalue() and not app.LOCK.locked()
    finally:
        app.STATE.pop("mem", None)
        app.STATE.pop("graphs_peak_said", None)
    return "72.0 + 0.4 a graph: stopped after 3 of 5 at 73.2 (4 reads, one line); a raising read: stopped after 1"


def test_a_length_the_drafter_takes_plain_captures_nothing():
    eng = _Eng(4)
    seen = []
    drafter = type("D", (), {"wants_graphs": lambda self, n: seen.append(n) or n < 100_000})()
    _with(eng, drafter=drafter)
    eng.kv.length = 260_800
    assert app.graphs_once() == "plain" and eng.owed == 4 and seen == [260_800]
    eng.kv.length = 32_000
    assert app.graphs_once() == "captured" and eng.owed == 0
    return "260,800 (plain turns there): nothing captured; 32,000: captured"


def test_spd271_one_engines_failure_leaves_the_others_their_window():
    """SPD-271: slot 1's capture raised (observed: a ClassError in the MTP chain's warm-up) and ended the loop over the
    engines, so slots 2-7 got no idle captures, and the next look 2 s later did the same. Now the failing engine
    gets one line and the next engine still captures what it owes; the engine lock is released."""
    import contextlib
    import io
    import types

    class _Failing(_Eng):
        def _steps(self):
            raise RuntimeError("mtp_chain_dev: class 16384 below 32768, the class of position 85010")
            yield                                   # noqa: unreachable (a generator, as idle_captures returns)

    e1, e2 = _Failing(2), _Eng(3)
    _with(_Eng(0))

    def slot(i, e):
        return types.SimpleNamespace(index=i, values={"engine": e, "drafter": None})
    app.STATE["pool"] = types.SimpleNamespace(side=lambda: [slot(1, e1), slot(2, e2)])
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            got = app.graphs_once()
    finally:
        app.STATE.pop("pool", None)
    lines = [ln for ln in out.getvalue().splitlines() if ln.startswith("[graphs] slot 1: idle capture failed")]
    assert got == "captured" and e2.owed == 0 and e2.locked == [True] * 3, (got, out.getvalue())
    assert len(lines) == 1 and "class 16384 below 32768" in lines[0], out.getvalue()
    assert not app.LOCK.locked()
    return "slot 1 failed (one line), slot 2's 3 graphs captured, the lock released"


def test_an_engine_without_idle_captures_is_left_alone():
    _with(object())
    assert app.graphs_once() == "off"
    return "off for an engine that names no graphs (the 27B)"


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<70} ok   {fn()}")
        except Exception as exc:                                   # noqa: BLE001
            fails += 1
            import traceback
            traceback.print_exc()
            print(f"  {name:<70} FAIL ({exc})")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
