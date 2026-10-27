"""On the board: the fused gated norm reading its gate in place (SPD-245, tools/norm_kernels.py `gate_rows`)
gives the dense gate's bits: the gate as the column slice of a fused qkvz output, as that slice's rows of
several lanes joined into one view (engine/model.py `_join`), and as a [1, 1, H, dv] decode gate; and
`gate2d` makes those layouts views, never copies. Run on the box under the lock:
python tests/gpu/test_gate_rows_gpu.py
"""

from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from engine.model import _join, gate2d  # noqa: E402
from tools import norm_kernels as N  # noqa: E402
from tools.exl3_slab_gemm import rms_norm_gated_out  # noqa: E402

H, DV, MIXED = 48, 128, 10240          # the 27B's value heads, head width, q|k|v columns of qkvz


def test_strided_gate_bits():
    g = torch.Generator(device="cuda").manual_seed(0)
    w = (torch.randn(DV, device="cuda", generator=g) * 0.1 + 1).to(torch.bfloat16)
    for T in (1, 8, 16, 64, 1024):
        y = torch.randn(T, MIXED + H * DV, device="cuda", generator=g).to(torch.bfloat16)
        z = y[:, MIXED:]                                     # the split's gate: a strided view
        o = torch.randn(T * H, DV, device="cuda", generator=g).to(torch.bfloat16)
        zv = gate2d(z)
        assert zv.data_ptr() == z.data_ptr() and (T == 1 or not zv.is_contiguous()), T   # one row is a plain view
        ref = N.rms_norm_gated(o, z.reshape(-1, DV).contiguous(), w, 1e-6)
        assert torch.equal(N.rms_norm_gated(o, zv, w, 1e-6), ref), T
        out = torch.empty_like(o)
        rms_norm_gated_out(o, zv, w, 1e-6, out)
        assert torch.equal(out, ref), T
        z4 = z.reshape(1, T, H, DV)                          # the prefill path's [B, T, H, dv] view
        assert gate2d(z4).data_ptr() == z.data_ptr()
        assert torch.equal(N.rms_norm_gated(o, gate2d(z4), w, 1e-6), ref), T


def test_joined_lanes_gate_bits():
    g = torch.Generator(device="cuda").manual_seed(1)
    w = (torch.randn(DV, device="cuda", generator=g) * 0.1 + 1).to(torch.bfloat16)
    rows = [8, 8, 16, 8]
    y = torch.randn(sum(rows), MIXED + H * DV, device="cuda", generator=g).to(torch.bfloat16)
    parts = [p[:, MIXED:] for p in y.split(rows, dim=0)]      # each lane's gate, as the split hands it
    zj = _join([gate2d(p) for p in parts], 0)
    assert zj.data_ptr() == parts[0].data_ptr(), "the joined gate is a view"
    o = torch.randn(sum(rows) * H, DV, device="cuda", generator=g).to(torch.bfloat16)
    ref = N.rms_norm_gated(o, torch.cat([p.reshape(-1, DV) for p in parts]), w, 1e-6)
    assert torch.equal(N.rms_norm_gated(o, zj, w, 1e-6), ref)


def test_decode_gate_bits():
    g = torch.Generator(device="cuda").manual_seed(2)
    w = (torch.randn(DV, device="cuda", generator=g) * 0.1 + 1).to(torch.bfloat16)
    zf = torch.randn(1, H * DV, device="cuda", generator=g).to(torch.bfloat16)
    z = zf.view(1, 1, H, DV)
    o = torch.randn(H, DV, device="cuda", generator=g).to(torch.bfloat16)
    ref = N.rms_norm_gated(o, z.reshape(-1, DV), w, 1e-6)
    assert torch.equal(N.rms_norm_gated(o, gate2d(z), w, 1e-6), ref)


if __name__ == "__main__":
    test_strided_gate_bits()
    test_joined_lanes_gate_bits()
    test_decode_gate_bits()
    print("ok")
