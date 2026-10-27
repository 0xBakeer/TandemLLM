"""SPD-199, QWEN38_EXL3_TALL: the 27B's dispatch to the decode-once EXL3 tile (engine/models/qwen3_5/tall.py).

On the CPU, with the tile and the shared kernel stood in: a tagged block at 33..CHUNK_UNTIL rows goes to the
tile, at 1..32 rows or with a forced tiling to the shared function; an untagged block (another family's)
always to the shared function; `install` wraps the shared function once and tags only the engine's EXL3
blocks of the built rates.

Run: python tests/test_exl3_tall_dispatch.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from engine.models.qwen3_5 import tall  # noqa: E402
from tools import exl3_linear as X  # noqa: E402


class _Blk(X.EXL3Block):
    def __init__(self, kb):                       # no tensors: the dispatch reads kb and the tag only
        self.kb = kb


def test_install_tags_the_engines_blocks_and_dispatches_by_rows():
    calls = []
    real_mt, real_orig, real_fn = tall.MT, tall._ORIG, X.exl3_matmul
    import tools.exl3_tall as TT
    real_t64 = TT.t64_matmul
    try:
        tall.MT, tall._ORIG = 4, None
        X.exl3_matmul = lambda x, w, **kw: calls.append(("shared", x.shape[0], kw.get("cfg"))) or "s"
        TT.t64_matmul = lambda x, w, mt, xh=None, out=None: calls.append(("tile", x.shape[0], mt)) or "t"

        class W:
            pass
        w = W()
        w.q = {"a": _Blk(4), "b": _Blk(2)}
        w.g, w.t = {}, {"lm_head.weight": _Blk(6), "norm": torch.zeros(1)}
        eng = W()
        eng.w = w
        assert tall.install(eng) == 2 and tall.install(eng) == 0
        assert X.exl3_matmul is tall._dispatch
        other = _Blk(4)                            # another family's block: never tagged
        x = lambda m: torch.zeros(m, 8)
        x.__name__ = "x"
        x_cuda = lambda m: type("T", (), {"shape": (m, 8), "is_cuda": True, "stride": lambda s, i: 1 if i else 8,
                                          "data_ptr": lambda s: 0})()
        for m, blk, kw, want in ((16, w.q["a"], {}, "shared"), (32, w.q["a"], {}, "shared"),
                                 (33, w.q["a"], {}, "tile"), (128, w.q["a"], {}, "tile"),
                                 (X.CHUNK_UNTIL + 1, w.q["a"], {}, "shared"), (64, w.q["a"], {"cfg": 2}, "shared"),
                                 (64, w.q["b"], {}, "shared"), (64, other, {}, "shared"),
                                 (64, w.t["lm_head.weight"], {}, "tile")):
            calls.clear()
            tall._dispatch(x_cuda(m), blk, **kw)
            assert calls and calls[0][0] == want, (m, blk.kb, kw, calls)
    finally:
        tall.MT, tall._ORIG = real_mt, real_orig
        X.exl3_matmul = real_fn
        TT.t64_matmul = real_t64
    return "tagged 33..CHUNK_UNTIL rows -> the tile; 1..32, forced tiling, unbuilt rates, untagged -> shared"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:66s} ok   {fn() or ''}")
            passed += 1
    print(f"{passed} passed")
