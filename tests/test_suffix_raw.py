"""SPD-151: a request that asks for the raw engine reads the suffix store and writes nothing into it.

On 2026-10-23 the same config ran a repeated benchmark prompt at a code CV of 5.3 % with the store off
and 17.9 % with the store on after the earlier runs had written their answers into it: the store
learned the benchmark and the runs measured a different engine each time. `suffix_learns` decides it
from the headers (`X-Tandem: raw` / `off`, the locked `raw` and `bench` profiles), `_remember` takes it
as `learn`, and `QSE_SUFFIX_RAW=write` restores the old rule.
"""

from __future__ import annotations

import contextlib
import os
import tempfile

from server import app


class _H(dict):
    def get(self, k, d=None):            # the handler's headers are case-insensitive
        for key, v in self.items():
            if key.lower() == k.lower():
                return v
        return d


CASES = [({}, True), ({"X-Tandem": "on"}, True), ({"X-Tandem": "shadow"}, True), ({"X-Tandem": "raw"}, False),
         ({"x-tandem": "RAW"}, False), ({"X-Tandem": "off"}, False), ({"X-Tandem-Profile": "bench"}, False),
         ({"X-Tandem-Profile": "raw"}, False), ({"X-Tandem-Profile": "coding"}, True)]


def test_which_requests_learn():
    for headers, learns in CASES:
        assert app.suffix_learns(_H(headers)) is learns, headers


def test_no_headers_learns():
    assert app.suffix_learns(None) is True


def test_write_restores_the_old_rule():
    saved = app.SUFFIX_RAW
    try:
        app.SUFFIX_RAW = "write"
        assert app.suffix_learns(_H({"X-Tandem": "raw"})) is True
    finally:
        app.SUFFIX_RAW = saved


@contextlib.contextmanager
def _store():
    from engine.cache import PersistentSuffixStore
    saved = {k: app.STATE.get(k) for k in ("suffix_store", "suffix_scope", "engine", "drafter")}
    try:
        with tempfile.TemporaryDirectory() as root:
            st = PersistentSuffixStore(os.path.join(root, "suffix"), tokenizer_sha="t" * 64).open()
            app.STATE.update(suffix_store=st, suffix_scope="all", engine=None, drafter=None)
            yield st
    finally:
        app.STATE.update(saved)


def _n(st) -> int:
    return int(st.report()["tokens"]) + int(st.report().get("pending") or 0)


def test_a_raw_turn_leaves_the_store_as_it_was():
    with _store() as store:
        before = _n(store)
        app._remember([5, 6, 7, 8], [9, 10, 11], None, cold=True, learn=False)
        assert _n(store) == before


def test_any_other_turn_is_learned():
    with _store() as store:
        before = _n(store)
        app._remember([5, 6, 7, 8], [9, 10, 11], None, cold=True)
        assert _n(store) > before


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
