"""SPD-249: Flash-Next's PLE prefetch (QWEN38_FN_PLE_PREFETCH): the row cache read from two threads gives the
file's rows, a prefetch only warms the cache, and the round runner hands its drafted trees over before the verify."""

import threading
import types

import numpy as np

from engine.models.flashnext import engine as E
from engine.models.flashnext.ple import DirectRows


def _table(tmp_path, rows=4000, words=11):
    data = (np.arange(rows * words, dtype=np.int64) % 30011).astype(np.int16).reshape(rows, words)
    p = tmp_path / "t.bin"
    p.write_bytes(b"\0" * 100 + data.tobytes())
    return DirectRows(str(p), 100, words * 2, rows, cache_bytes=64 * words * 2 * 4), data


def test_rows_from_two_threads_are_the_files_rows(tmp_path):
    t, data = _table(tmp_path)
    rng = np.random.default_rng(1)
    bad = []

    def run(seed):
        r = np.random.default_rng(seed)
        for _ in range(60):
            ids = r.integers(0, data.shape[0], size=int(r.integers(1, 90)))
            if not np.array_equal(t.rows_of(ids), data[ids]):
                bad.append(seed)
    ths = [threading.Thread(target=run, args=(int(s),)) for s in rng.integers(0, 1000, size=4)]
    for th in ths:
        th.start()
    for th in ths:
        th.join(60)
    assert not bad
    assert t.stats["misses"] > 0 and t.stats["hits"] > 0
    t.close()


def test_a_warmed_row_is_a_hit_with_the_same_bits(tmp_path):
    t, data = _table(tmp_path)
    ids = np.array([5, 17, 3999, 17, 250])
    t.rows_of(ids)                                       # the prefetch
    m0 = t.stats["misses"]
    assert np.array_equal(t.rows_of(ids), data[ids])     # the verify's read
    assert t.stats["misses"] == m0
    t.close()


class _Pool:
    def __init__(self):
        self.jobs = []

    def submit(self, fn, *a):
        self.jobs.append((fn, a))


def _fake(layers=(3,)):
    f = types.SimpleNamespace(cfg=types.SimpleNamespace(ple_layers=list(layers)), pool=_Pool(), read=[])
    f._ple_reader = lambda: f.pool
    f._ple_warm = lambda items: f.read.append(items)
    return f


def test_the_prefetch_is_off_by_default_and_queues_one_read_when_on(monkeypatch):
    f = _fake()
    items = [([1, 2, 3], [-1, 0, 1], object())]
    monkeypatch.setattr(E, "PLE_PREFETCH", False)
    E.FlashNextEngine.ple_prefetch(f, items)
    assert f.pool.jobs == []
    monkeypatch.setattr(E, "PLE_PREFETCH", True)
    E.FlashNextEngine.ple_prefetch(f, items)
    E.FlashNextEngine.ple_prefetch(f, [])
    assert len(f.pool.jobs) == 1 and f.pool.jobs[0][1] == (items,)
    E.FlashNextEngine.ple_prefetch(_fake(layers=()), items)          # no PLE layers: nothing


def test_the_runner_hands_drafted_trees_over():
    from server import round_runner as RR
    got = []

    class Eng:
        @staticmethod
        def prefetch_rows(items):
            got.append(items)

    class NoPf:
        pass

    def task(eng, tokens, parents, n_draft=1, done=False):
        tree = None if tokens is None else types.SimpleNamespace(tokens=tokens, parents=parents, n_draft=n_draft)
        return types.SimpleNamespace(done=done, tree=tree, eng=eng)
    e1, e2 = Eng(), Eng()
    tasks = [task(e1, [7, 8], [-1, 0]), task(e2, [9], [-1], n_draft=0), task(e1, None, None),
             task(e2, [4, 5, 6], [-1, 0, 0], done=True), task(NoPf(), [1, 2], [-1, 0]), task(e2, [3, 4], [-1, 0])]
    RR.Runner._prefetch(None, tasks)
    assert got == [[(e1, [7, 8], [-1, 0]), (e2, [3, 4], [-1, 0])]]


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
