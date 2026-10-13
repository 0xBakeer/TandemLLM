"""0.8.2: a module whose bytes changed without changing a stored row keeps the saved conversations
(engine/methods/manifest.py `SAME_ROWS`). CPU only.

#1 (0xWhiteMage, a6f2142c) changed engine/loader.py to load a standalone EXL3 pack. The file is in
the 27B's v3 manifest (engine/models/qwen3_5/adapter.py MODULES) and in v2's engine walk, so its
bytes alone changed every 27B fingerprint, and the next 27B start would have swept every saved chat
(on testers' boxes too), although a 27B profile (QWEN38_EXL3 set, or NVFP4) takes the old path.

  * the entry names this tree's loader.py, and with it the 27B's v2 code fingerprint is 0.7.2's
    (9ac3d8a94c8a9199, the number the release notes give since 0.7.2);
  * a fingerprint with `same` reads the replaced bytes: v2 and v3 of the new tree equal the old
    tree's; without it they differ;
  * the legacy list holds the old tree's fp4 (its extra by content, as a start computes it), and a
    record the old tree's start left adopts it, with what it had adopted, once, under a new key;
  * a directory whose fingerprint did not change (Flash-Next's: loader.py is not in its manifest)
    keeps what it had adopted when the new key adopts again.

Run: python tests/test_fp_same_rows.py
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_fp_equivalents import _edit, _tree as _code  # noqa: E402
from test_tier_identity import Env, _args, _tree  # noqa: E402

from engine import disk_cache as dc  # noqa: E402
from engine import spill as sp  # noqa: E402
from engine.methods import manifest as mf  # noqa: E402
from engine.models.qwen3_5 import adapter  # noqa: E402
from engine.models.flashnext import serving as fn_serving  # noqa: E402
from server import diskcache  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOADER = "engine/loader.py"
M27 = adapter.manifest()


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def test_the_entry_names_this_loader_and_keeps_the_27bs_code_fingerprint():
    now = _sha(os.path.join(ROOT, LOADER))
    assert (LOADER, now) in mf.SAME_ROWS, "SAME_ROWS names another loader.py: update it"
    assert LOADER in M27.modules
    code = dc.fingerprint(ROOT, {}, skip=sp.NOT_ARITHMETIC)
    same = dc.fingerprint(ROOT, {}, skip=sp.NOT_ARITHMETIC, same=mf.SAME_ROWS)
    assert same.startswith("9ac3d8a94c8a9199"), same
    assert not code.startswith("9ac3d8a94c8a9199"), "the loader's bytes count without the entry"
    assert mf.adoption() != mf.FORMULA and mf.adoption().startswith(mf.FORMULA + "+same-")
    return f"v2 code {code[:16]} on the bytes, {same[:16]} with SAME_ROWS (0.7.2's)"


def test_flash_next_does_not_read_the_loader():
    files = {os.path.relpath(p, ROOT) for p in mf.files(ROOT, fn_serving.manifest())}
    assert LOADER not in files, files


def test_a_fingerprint_with_same_reads_the_replaced_bytes():
    with tempfile.TemporaryDirectory() as d:
        old, new = _code(os.path.join(d, "old")), _code(os.path.join(d, "new"))
        _edit(os.path.join(new, LOADER), "class Weights", "# a standalone pack's path\nclass Weights")
        same = {(LOADER, _sha(os.path.join(new, LOADER))): _sha(os.path.join(old, LOADER))}
        x = {"weights": "a", "grid": 1024}
        v2 = lambda t, s=None: dc.fingerprint(t, x, skip=sp.NOT_ARITHMETIC, same=s)  # noqa: E731
        assert v2(new) != v2(old) and v2(new, same) == v2(old)
        assert mf.fingerprint(new, M27, x) != mf.fingerprint(old, M27, x)
        assert mf.fingerprint(new, M27, x, same=same) == mf.fingerprint(old, M27, x)
        assert mf.fingerprint(new, M27, x, formula="fp3", same=same) == \
            mf.fingerprint(old, M27, x, formula="fp3")
        # another file's change is not covered by the entry
        _edit(os.path.join(new, "engine", "tree.py"), "from __future__", "# x\nfrom __future__")
        assert mf.fingerprint(new, M27, x, same=same) != mf.fingerprint(old, M27, x)


def test_the_old_trees_fp4_is_adopted_once_with_its_chain():
    d = tempfile.mkdtemp(prefix="qse-same-")
    real = mf.SAME_ROWS
    try:
        old, new = _code(os.path.join(d, "old")), _code(os.path.join(d, "new"))
        _edit(os.path.join(new, LOADER), "class Weights", "# a standalone pack's path\nclass Weights")
        mf.SAME_ROWS = {(LOADER, _sha(os.path.join(new, LOADER))): _sha(os.path.join(old, LOADER))}
        t = _tree(d, "box", mtime=1.7e9)
        tier = os.path.join(d, "tier")
        os.makedirs(tier)
        with Env(t, added=True):
            extra = diskcache.fingerprint_extra(_args(t))
            fp_old, fp_new = mf.fingerprint(old, M27, extra), mf.fingerprint(new, M27, extra)
            leg = diskcache.legacy_fingerprints(new, M27, _args(t), tree=t)
        assert fp_old != fp_new and fp_old in leg
        # what the old tree's first fp4 start left: its fp4, and what that one adopted (0.7.2's)
        with open(os.path.join(tier, dc.EQUIV_FILE), "w") as f:
            json.dump({"formulas": {mf.FORMULA: fp_old}, fp_old: ["0.7.2's v2", "the RC's fp3"]}, f)
        also = dc.equivalents(tier, fp_new, leg, formula=mf.adoption())
        assert {fp_old, "0.7.2's v2", "the RC's fp3"} <= set(also), also
        assert dc.equivalents(tier, fp_new, ["anything"], formula=mf.adoption()) == also, "once"
        # without the entry (the formula's key alone) nothing is adopted: the files would be swept
        with open(os.path.join(tier, dc.EQUIV_FILE), "w") as f:
            json.dump({"formulas": {mf.FORMULA: fp_old}, fp_old: ["0.7.2's v2"]}, f)
        assert dc.equivalents(tier, fp_new, leg, formula=mf.FORMULA) == ()
        return f"{fp_new[:12]} adopts {fp_old[:12]} and its {len(also) - 1} older ones"
    finally:
        mf.SAME_ROWS = real
        shutil.rmtree(d, ignore_errors=True)


def test_an_unchanged_fingerprint_keeps_what_it_adopted():
    with tempfile.TemporaryDirectory() as tier:
        with open(os.path.join(tier, dc.EQUIV_FILE), "w") as f:
            json.dump({"formulas": {mf.FORMULA: "flash-next"}, "flash-next": ["the RC's fp3", "v2"]}, f)
        also = dc.equivalents(tier, "flash-next", ["flash-next", "v2 on the old loader"],
                              formula=mf.adoption())
        assert set(also) == {"the RC's fp3", "v2", "v2 on the old loader"}, also


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<72} ok   {fn() or ''}", flush=True)
        except AssertionError as e:
            fails += 1
            print(f"  {name:<72} FAIL {e}", flush=True)
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
