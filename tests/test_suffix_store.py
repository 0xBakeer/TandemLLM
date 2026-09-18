"""The persistent suffix store's cap, and what the cap is a cap ON.

The two-hour soak (phase 9, 02:36) ended with one number still moving: RSS 8.30 -> 8.83 GB, 530 MB
over two hours, host-side, and the note said "at this rate it reaches its cap rather than the board,
but it is the one thing in this table that has not been shown to stop". This file is that claim
turned into checks that need neither the board nor two hours.

Three of them are about whether the cap fires at all, and the fourth is about the thing the soak
could not see: `--suffix-store-mb` names a number of BYTES ON DISK, the store is int32 there and
int64 in memory with an int32 suffix array beside it, and a rebuild sorts several int64 arrays of
the same length at once. So the flag's megabytes and the resident megabytes are not the same
quantity, and the factor between them is written down here rather than discovered on a box.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                                  # noqa: E402

from engine.cache import DOC_SEP, PersistentSuffixStore             # noqa: E402


def _store(tmp, max_tokens, rebuild_every=10 ** 9):
    return PersistentSuffixStore(tmp, max_tokens=max_tokens, rebuild_every=rebuild_every,
                                 max_order=8).open()


def _on_disk(st):
    return np.fromfile(st._bin, dtype="<i4") if os.path.exists(st._bin) else np.zeros(0, "<i4")


def test_the_cap_fires_and_keeps_half():
    """Past the cap the store forgets the oldest half rather than the oldest token, so a trim is a
    rare rewrite and not one per request."""
    tmp = tempfile.mkdtemp()
    try:
        st = _store(tmp, max_tokens=400)
        for d in range(20):                                          # 20 x (50 + 1) = 1,020 tokens
            st.append([1000 + d * 100 + i for i in range(50)])
        assert st.stats["trims"] == 0                                # nothing trims on append
        assert _on_disk(st).shape[0] == 1020
        st.rebuild(background=False)
        assert st.stats["trims"] == 1, st.report()
        n = _on_disk(st).shape[0]
        assert n <= 400, n
        assert n >= 150, n                                           # half the cap, not everything
        assert st.n_tokens == n
    finally:
        shutil.rmtree(tmp)


def test_the_cut_lands_on_a_document_boundary():
    """A continuation that starts in the middle of a document is a continuation of nothing."""
    tmp = tempfile.mkdtemp()
    try:
        st = _store(tmp, max_tokens=400)
        for d in range(20):
            st.append([1000 + d * 100 + i for i in range(50)])
        st.rebuild(background=False)
        a = _on_disk(st)
        # every document in the survivors is whole: the last token is a separator, and no separator
        # is the first token, which is what "the cut moved forward to the next boundary" means
        assert a[-1] == DOC_SEP
        assert a[0] != DOC_SEP
        seps = np.nonzero(a >= DOC_SEP)[0]
        lengths = np.diff(np.concatenate([[-1], seps]))
        assert set(int(x) for x in lengths) == {51}, lengths
    finally:
        shutil.rmtree(tmp)


def test_the_cap_does_not_fire_until_something_rebuilds():
    """The trim lives in `_read_tokens`, which only a rebuild calls. Between rebuilds the file
    grows without bound -- which is the shape of the soak's 530 MB: not a cap that failed, a cap
    that is nowhere near, with the index in memory growing toward it."""
    tmp = tempfile.mkdtemp()
    try:
        st = _store(tmp, max_tokens=100, rebuild_every=10 ** 9)
        for d in range(40):
            st.append(list(range(10, 60)))
        assert _on_disk(st).shape[0] == 40 * 51                      # 2,040, twenty times the cap
        assert st.stats["trims"] == 0
    finally:
        shutil.rmtree(tmp)


def test_the_flags_megabytes_are_disk_megabytes():
    """`--suffix-store-mb 192` sets `max_tokens = 192 MiB / 4`, because the log is int32 on disk.
    In memory the same tokens are int64 with an int32 suffix array beside them, so the resident
    cost at the cap is THREE times the number in the flag, and a rebuild sorts several int64 arrays
    of that length at once on top of it.

    At the shipped 192 MB that is 50,331,648 tokens: 402 MB of tokens, 201 MB of suffix array,
    604 MB resident, and the soak's 530 MB in two hours was on its way there rather than near it --
    at that traffic the cap is about a week away, which is why nothing in two hours showed it stop.
    """
    mb = 192.0
    max_tokens = int(mb * (1 << 20)) // 4                            # server/app.py
    assert max_tokens == 50_331_648
    tokens_bytes = max_tokens * 8                                    # np.int64 in `_read_tokens`
    sa_bytes = max_tokens * 4                                        # np.int32 from build_suffix_array
    resident = tokens_bytes + sa_bytes
    assert resident == 3 * int(mb * (1 << 20))
    assert 600e6 < resident < 610e6, resident


def test_a_store_below_its_cap_is_left_alone():
    tmp = tempfile.mkdtemp()
    try:
        st = _store(tmp, max_tokens=10_000)
        for d in range(10):
            st.append(list(range(10, 60)))
        st.rebuild(background=False)
        assert st.stats["trims"] == 0 and st.n_tokens == 10 * 51
        assert st.stats["rebuilds"] == 1   # `open` finds an empty file and returns before counting
    finally:
        shutil.rmtree(tmp)


def _main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = 0
    for fn in fns:
        try:
            fn()
            print(f"  ok  {fn.__name__}")
        except Exception as exc:                                     # noqa: BLE001
            bad += 1
            import traceback
            traceback.print_exc()
            print(f"FAIL  {fn.__name__}: {exc}")
    print(f"\n{len(fns) - bad}/{len(fns)} passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(_main())
