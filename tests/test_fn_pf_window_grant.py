"""SPD-249: a layer-major prefill window asks the memory policy for its bytes before it allocates them, no GPU.

  * the price of a window (`layer_major.window_bytes`, `plan_bytes`): above the offline peaks it was fitted on, the
    largest window of a plan, a state buffer a chunk with lines, nothing for a window of one chunk;
  * the slot verb (`FlashNextServing.window_bytes`) and the chunk loop's ask (`cache._window_plan`, `WINDOW_GRANT`):
    a plan only when the policy says yes, the chunk-major path when it says no, no ask of an engine that does not
    price its windows;
  * the engine's peak (`Budget.window`): granted once a prefill, held in its workspace until it has run, refused
    over the peak with every other prefill in flight counted, nothing held after a refusal or an undo;
  * the box's floor (`Admission.window`): the reserve raised and seen by every other check, left as it was when the
    floor does not hold it, given back with the prefill's workspace;
  * the server's hook (`app._window_grant`): the peak, then the floor for a request's prefill, the peak's grant
    undone when the floor says no.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import threading
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from engine import cache  # noqa: E402
from engine.cache import GIB, Budget  # noqa: E402
from engine.models.flashnext import layer_major as LM  # noqa: E402
from server import slots  # noqa: E402
from test_budget import FakeAlloc, budget  # noqa: E402

C = 4096


def test_the_window_prices_its_bytes():
    assert LM.window_bytes(C, C) == 0 and LM.window_bytes(100, C) == 0           # one chunk: chunk-major
    w8 = LM.window_bytes(8192, C)
    assert w8 == int(LM.WIN_MARGIN * (LM.WIN_FIXED + LM.WIN_ROW * C))
    # the offline 32k peaks above the held bytes it was fitted on: 4.68 - 2.57 GiB (8,192), 8.97 - 2.57 (32,768)
    assert w8 >= (4.68 - 2.57) * GIB and LM.window_bytes(32768, C) >= (8.97 - 2.57) * GIB
    assert LM.plan_bytes([C, C], 8192, C) == w8
    assert LM.plan_bytes([C, C, C, C], 8192, C) == w8                            # two windows, one after another
    assert LM.plan_bytes([C, C, 100], 8192, C) == w8
    assert LM.plan_bytes([C, 100], 8192, C) == LM.window_bytes(C + 100, C)
    assert LM.plan_bytes([C, C], C, C) == 0                                      # windows of one chunk: none
    assert LM.plan_bytes([C, C], 8192, C, section=10, lines=True) == w8 + 20    # a state buffer a chunk
    assert LM.plan_bytes([C, C], C, C, section=10, lines=True) == 0
    return "a window of 8,192 rows asks %.2f GiB, of 32,768 %.2f" % (w8 / GIB, LM.window_bytes(32768, C) / GIB)


def test_the_loop_asks_for_a_slots_windows_and_runs_chunk_major_without_them(monkeypatch):
    from test_fn_pf_window import _WEng
    from engine.models.flashnext import serving as SV
    from engine.models.flashnext.serving import FlashNextServing
    monkeypatch.setattr(SV, "PREFILL_CHUNK", 4)
    monkeypatch.setattr(LM, "ROWS", 8)
    a = FlashNextServing(_WEng(), 64)
    sec = sum(t.numel() * t.element_size() for t in a.state.buffers().values())
    assert sec > 0
    assert a.window_bytes([(0, 4), (4, 4), (8, 3)], False) == LM.window_bytes(8, 4)
    assert a.window_bytes([(0, 4), (4, 4), (8, 3)], True) == LM.window_bytes(8, 4) + 3 * sec
    asks = []
    monkeypatch.setattr(cache, "WINDOW_GRANT", lambda nb: asks.append(nb) and False)
    assert cache._window_plan(a, 0, 11, 4, 8) is None                           # no: one chunk the usual way
    assert asks == [LM.window_bytes(8, 4)]
    monkeypatch.setattr(cache, "WINDOW_GRANT", lambda nb: asks.append(nb) or True)
    assert cache._window_plan(a, 0, 11, 4, 8, lines=True) == [(0, 4), (4, 4)]
    assert asks[-1] == LM.window_bytes(8, 4) + 2 * sec
    assert cache._window_plan(a, 8, 11, 4, 8) is None and len(asks) == 2        # one chunk left: nothing asked
    monkeypatch.setattr(cache, "WINDOW_GRANT", None)
    assert cache._window_plan(a, 0, 11, 4, 8) == [(0, 4), (4, 4)]               # no policy: as before

    class Plain:                                                                 # an engine that prices no window
        def window_ok(self, i):
            return True

        def prefill_window(self, *a):
            raise AssertionError

    def never(nb):
        raise AssertionError("asked")
    monkeypatch.setattr(cache, "WINDOW_GRANT", never)
    assert cache._window_plan(Plain(), 0, 100, 16, 48) == [(0, 16), (16, 16), (32, 16)]
    return "asked before the plan; no: chunk-major; no hook or no price: the plan as before"


def _other(b: Budget, ws: int) -> None:
    t = threading.Thread(target=lambda: setattr(b, "ws_now", ws))
    t.start()
    t.join()


def test_the_engine_peak_grants_a_window_and_holds_it_until_the_prefill_has_run():
    fa = FakeAlloc(60, 2)                                  # 60 GiB allocated, 62 reserved; peak 77
    b = budget(fa)
    ws, w = 4 * GIB, int(2.4 * GIB)
    b.ws_now = ws                                          # this thread's prefill, as the gate admitted it
    assert b.window(w) == w and b.ws_own() == ws + w and b.ws_now == ws + w
    assert b.window(w) == 0 and b.ws_own() == ws + w      # asked again in the same prefill: nothing more
    assert b.window(w + GIB) == GIB and b.ws_own() == ws + w + GIB
    b.window_undo(GIB)
    assert b.ws_own() == ws + w and b.stats["windows"] == 2
    b.prefill_done()                                       # the prefill has run: workspace and window gone
    assert b.ws_own() == 0 and not b._win
    b.ws_now = ws
    assert b.window(12 * GIB) is None and b.ws_own() == ws         # 62 + 4 + 12 > 77, not in the free blocks
    assert b.stats["window_refused"] == 1 and b.events[-1]["kind"] == "window_refused"
    _other(b, 9 * GIB)                                     # another prefill in flight counts: 62 + 4 + 2.4 + 9 > 77
    assert b.window(w) is None and b.ws_own() == ws
    b.window_undo(GIB)                                     # nothing granted: nothing to undo
    assert b.ws_own() == ws
    off = Budget(0, 0, 0, probe=fa.probe, empty=fa.empty, peak_probe=fa.peak_probe)
    assert off.window(w) == 0                              # no policy: yes, nothing held
    return "granted under the peak and held to the prefill's end; over it, or with others in flight, refused"


def test_the_admission_raises_the_reserve_for_a_window_or_leaves_it():
    GiB = slots.GiB
    avail = [30 * GiB]
    adm = slots.Admission(10 * GiB, meminfo=lambda: {"MemAvailable": avail[0]})
    assert adm.reserve("r1", 4 * GiB, rows=32768)[0]
    w = int(2.4 * GiB)
    ok, d = adm.window("r1", w)
    assert ok and d["window"]["more"] == w and adm.reserved() == 4 * GiB + w
    assert adm.window("r1", w) == (True, None) and adm.reserved() == 4 * GiB + w   # held: nothing more
    _, d2 = adm.check(0, count=False)
    assert d2["in_flight_ws"] == 4 * GiB + w                                      # every other check sees it
    adm.prefilled("r1")
    assert adm.reserved() == 0 and "r1" not in adm.win
    assert adm.reserve("r2", 4 * GiB, rows=32768)[0]
    ok, d = adm.window("r2", 17 * GiB)                                            # 30 - 4 - 17 = 9 < 10
    assert not ok and d["after"] < d["floor"] and adm.reserved() == 4 * GiB and "r2" not in adm.win
    assert adm.stats["windows"] == 1 and adm.stats["window_refused"] == 1
    assert adm.window("nobody", GiB) == (True, None)                             # a prefill it does not hold
    adm.release("r2")
    assert adm.reserved() == 0
    return "raised at the floor and seen by others; refused: as it was; gone with the prefill's workspace"


def _app():
    import test_forward_tree  # noqa: F401  (the CPU environment before the engine imports)
    from server import app
    return app


def test_the_server_asks_the_peak_then_the_floor():
    app = _app()
    GiB = slots.GiB
    fa = FakeAlloc(60, 2)
    b = budget(fa)
    avail = [30 * GiB]
    adm = slots.Admission(10 * GiB, meminfo=lambda: {"MemAvailable": avail[0]})
    saved = dict(app.STATE)
    out = io.StringIO()
    try:
        app.STATE.clear()
        app.STATE["mem"], app.STATE["pool"] = b, types.SimpleNamespace(admission=adm)
        assert adm.reserve("r1", 4 * GiB, rows=32768)[0]
        b.ws_now = 4 * GiB
        w = int(2.4 * GiB)
        with contextlib.redirect_stdout(out), app._forward_ask("r1", 32768):
            assert app._window_grant(w) is True
            assert adm.reserved() == 4 * GiB + w and b.ws_own() == 4 * GiB + w
            avail[0] = 15 * GiB                            # the floor no longer holds 5 GiB more
            assert app._window_grant(w + 5 * GiB) is False
            assert adm.reserved() == 4 * GiB + w and b.ws_own() == 4 * GiB + w      # the peak's grant undone
        assert "granted 2.40 GiB" in out.getvalue() and "not there" in out.getvalue()
        assert app._PF_REQ == {}
        b.prefill_done()
        b.ws_now = 4 * GiB
        with contextlib.redirect_stdout(out):
            assert app._window_grant(w) is True            # the engine's own prefill: the peak alone
            assert app._window_grant(20 * GiB) is False and "over the engine's peak" in out.getvalue()
        assert adm.reserved() == 4 * GiB + w               # the admission was not asked
    finally:
        app.STATE.clear()
        app.STATE.update(saved)
    return "the peak, then the request's floor; the floor's no undoes the peak's yes"


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
