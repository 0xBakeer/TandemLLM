"""SPD-249, no GPU: Flash-Next's joint pass tables and joint graph keys.

  * an eager joint pass starts on clean entry tables: after a pass that raised between its attention layers (its
    tables' `k` short of their end) the next pass maps its own rows to its own lanes, and the failed pass's queued
    copies are recorded first; after a pass that ended nothing is recorded (the tables are at their end already);
  * a joint graph's key holds the lanes' K/V strides: slot 0's caches and a side slot's of another cap get two
    keys (a graph bakes the strides `attn_rows_multi` takes), lanes of one cap one key;
  * `joint_graph_stats` hands /health copies: the kept keys and the counts are copied before they are walked, so
    the runner reordering the kept graphs meanwhile never breaks it; the state is made once (setdefault);
  * /health's `joint_stats` part is a safe report: a family's report that raises gives an error entry.

Run: python tests/test_fn_joint_tables.py
"""
from __future__ import annotations

import os
import sys
import threading
import time
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.models.flashnext.engine import FlashNextEngine  # noqa: E402

LAYERS = 8                      # attention at 3 and 7, GDN at the rest


class _Event:
    """torch.cuda.Event's two calls the tables make, counted."""

    def __init__(self):
        self.records = self.syncs = 0

    def record(self):
        self.records += 1

    def synchronize(self):
        self.syncs += 1


def _cfg():
    lin = [li for li in range(LAYERS) if li % 4 != 3]
    return types.SimpleNamespace(num_hidden_layers=LAYERS, is_linear=lambda li: li in lin, linear_layers=lin,
                                 attention_layers=[li for li in range(LAYERS) if li not in lin])


def _eng():
    eng = object.__new__(FlashNextEngine)
    eng.cfg = _cfg()
    La = len(eng.cfg.attention_layers)
    eng._aent = {"ap_h": torch.zeros(La, 8, 4, dtype=torch.int64), "lane_h": torch.zeros(32, dtype=torch.int32),
                 "lane_dd": torch.zeros(32, dtype=torch.int32), "copied": _Event(), "k": La}
    eng._gent = {"h": torch.zeros(len(eng.cfg.linear_layers), 8, 4, dtype=torch.int64), "copied": _Event(),
                 "k": len(eng.cfg.linear_layers)}
    return eng


def _rows(*m):
    out, s = [], 0
    for x in m:
        out.append(slice(s, s + x))
        s += x
    return out


def _lanes(rows):
    return [j for j, r in enumerate(rows) for _ in range(r.stop - r.start)]


def test_a_pass_after_a_failed_one_maps_its_own_lanes():
    La = 2
    eng = _eng()
    a = _rows(10, 14)                            # the failed pass: lanes of 10 and 14 rows
    ent = eng._attn_entries(2, 24, a)
    assert ent["lane_d"].tolist() == _lanes(a)
    ent["k"] += 1                                # its first attention layer ran, then the pass raised
    eng._gent["k"] = 3
    b = _rows(20, 4)                             # the next pass: other lanes
    stale = eng._attn_entries(2, 24, b)["lane_d"].tolist()
    assert stale == _lanes(a) != _lanes(b), "without the begin the failed pass's map stays (the bug)"
    eng._aent["k"] = 1
    rec = eng._aent["copied"].records
    eng._joint_tables_begin()
    assert eng._aent["k"] == La and eng._gent["k"] == len(eng.cfg.linear_layers)
    assert eng._aent["copied"].records == rec + 1 and eng._gent["copied"].records == 1, "the queued copies recorded"
    got = eng._attn_entries(2, 24, b)
    assert got["lane_d"].tolist() == _lanes(b) and got["k"] == 0
    # k left at 0 (raised between the entries and the layer's step): the same
    eng._joint_tables_begin()
    assert eng._aent["k"] == La
    # after a pass that ended (k at the end) the begin records and waits for nothing
    r0, g0 = eng._aent["copied"].records, eng._gent["copied"].records
    eng._joint_tables_begin()
    assert (eng._aent["copied"].records, eng._gent["copied"].records) == (r0, g0)
    fresh = object.__new__(FlashNextEngine)
    fresh.cfg = _cfg()
    fresh._joint_tables_begin()                  # no tables yet: nothing to do
    return "rows 10-19 of the next pass map to their own lane; an ended pass costs no event"


class _St:
    def __init__(self, cap, D=4, Hk=2, layers=(3, 7)):
        self.k = {li: torch.zeros(Hk, cap, D) for li in layers}
        self.v = {li: torch.zeros(Hk, cap, D) for li in layers}


def test_a_graph_key_holds_the_kv_strides():
    eng = object.__new__(FlashNextEngine)
    eng.cfg = _cfg()
    slot0, side, side2 = _St(256), _St(64), _St(64)
    k0 = eng._jg_key([([1], [-1], slot0)], 16)
    k1 = eng._jg_key([([1], [-1], side)], 16)
    assert k0[:3] == k1[:3] == (1, 16, 0) and k0 != k1, (k0, k1)
    assert k0[3] == (256 * 4, 4, 256 * 4, 4), k0
    assert eng._jg_key([([1], [-1], side), ([2], [-1], side2)], 16) == (2, 16, 0, k1[3])
    return f"{k0} and {k1}: two graphs"


def test_the_stats_are_copies_and_survive_a_reordering_runner():
    eng = object.__new__(FlashNextEngine)
    eng.cfg = _cfg()
    g = eng._jg_state()
    assert eng._jg_state() is g and eng.__dict__["_jgs"] is g
    for e in range(1, 9):
        g["graphs"][(e, 16, 0, (1024, 4, 1024, 4))] = object()
    g["stats"]["eager"]["sparse"] = 2
    st = eng.joint_graph_stats()
    assert st["kept"][0] == [1, 16, 0, [1024, 4, 1024, 4]] and len(st["kept"]) == 8, st["kept"]
    st["eager"]["sparse"] = 99
    assert g["stats"]["eager"]["sparse"] == 2, "a copy, not the runner's dict"
    stop, errs = threading.Event(), []

    def runner():
        keys = list(g["graphs"])
        i = 0
        while not stop.is_set():
            k = keys[i % len(keys)]
            g["graphs"][k] = g["graphs"].pop(k)              # a replay: most recently used last
            g["stats"]["eager"][f"r{i % 5}"] = i
            i += 1
    th = threading.Thread(target=runner, daemon=True)
    th.start()
    n, t0 = 0, time.monotonic()
    try:
        while time.monotonic() - t0 < 0.5:
            try:
                assert len(eng.joint_graph_stats()["kept"]) in (7, 8)     # a key in the middle of its move: 7
            except RuntimeError as exc:                      # 'dictionary changed size during iteration'
                errs.append(repr(exc))
            n += 1
    finally:
        stop.set()
        th.join()
    assert not errs, errs[:3]
    return f"{n} reports beside a reordering runner, none failed"


def test_health_joint_stats_is_a_safe_part():
    from server import app as A
    from server import slots as S

    class Eng:
        def joint_stats(self):
            raise RuntimeError("dictionary changed size during iteration")
    saved = {k: dict.get(A.STATE, k) for k in ("pool", "engine", "disk_side")}
    try:
        A.STATE["pool"] = S.Pool([S.Slot(0, 4096), S.Slot(1, 1024)])
        A.STATE["engine"] = Eng()
        A.STATE.pop("disk_side", None)
        out = A._par_health()
    finally:
        for k, v in saved.items():
            if v is None:
                A.STATE.pop(k, None)
            else:
                A.STATE[k] = v
    assert out["joint_stats"] == {"error": "RuntimeError: dictionary changed size during iteration"}, out
    return "an error entry, the rest of /health built"


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
