"""On the board: the fused GDN prefill preparation (`tools/gdn_prep_kernels.gdn_prep`: conv + SiLU + the
l2 norm of q and k) against engine/gdn.py's conv_update / conv_prefill and l2norm at the key heads, on
fresh and continued sequences and short chunks. Run on the box: python tests/gpu/test_gdn_prep_gpu.py
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from tools import gdn_prep_kernels as P  # noqa: E402


def test_prep_matches_the_reference():
    for T in (1, 2, 3, 64, 1024, 1350):
        for state in (False, True):
            r = P.check(T=T, state=state)
            for name, v in r.items():
                assert v["equal_frac"] == 1.0, (T, state, name, v)


if __name__ == "__main__":
    test_prep_matches_the_reference()
    print("ok")
