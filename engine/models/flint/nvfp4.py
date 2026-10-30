"""Flint's routed experts from an NVFP4 pack (tools/flint_nvfp4.py): every expert's gate|up [2 I, H] (one tensor
scale for both) and down [H, I], stacked on the device, through tools/nvfp4_relu2.py's pair kernel (one program a
(row, slot) pair and 32 outputs, K walked in order, fp32 products: a verify block's row is the decode step's).

  gate|up   y = W_gu x, rounded to bf16 (no activation in the kernel)
  act       a = bf16(bf16(silu(g)) u)   (the BF16 experts' roundings, engine/models/lfm2_moe/kernels.py)
  down      yd = W_d a in fp32, rounded once to bf16 values

No fp16 anywhere, so the top layers' activations past 65,504 (|silu(g) u| up to 2.6e5) are taken as they are.
A prompt chunk runs each expert's rows through cuBLAS on its weights unpacked to bf16.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tools.nvfp4_relu2 import HAVE_TRITON, dequant, e4m3_table, fp4_table

if HAVE_TRITON:
    from tools.nvfp4_relu2 import _nvfp4_bank, _nvfp4_unpack


class NVFP4GLUBank:
    def __init__(self, gw, gs, g2, dw, ds, d2):
        """gw [E, 2 I, H / 2] uint8, gs [E, 2 I, H / 16] e4m3 bytes, g2 [E] fp32; dw [E, H, I / 2], ds, d2."""
        self.gw, self.gs = gw.contiguous(), gs.contiguous().view(torch.uint8)
        self.dw, self.ds = dw.contiguous(), ds.contiguous().view(torch.uint8)
        self.g2, self.d2 = g2.float().contiguous(), d2.float().contiguous()
        self.E, I2, half = gw.shape
        self.I, self.H = I2 // 2, half * 2
        if tuple(dw.shape) != (self.E, self.H, self.I // 2):
            raise ValueError(f"NVFP4 experts: gate|up {tuple(gw.shape)}, down {tuple(dw.shape)}")
        self._f4, self._f8 = fp4_table(gw.device), e4m3_table(gw.device)
        self.w13 = None          # not a BF16 bank

    @classmethod
    def stack(cls, experts: list, device) -> "NVFP4GLUBank":
        """From [((gate w, s), (up w, s), s2_gu, (down w, s), s2_d)] an expert, host tensors as stored."""
        E = len(experts)
        (g0, gs0), (u0, _), _, (d0, ds0), _ = experts[0]
        I, H2 = g0.shape
        gw = torch.empty(E, 2 * I, H2, dtype=torch.uint8, device=device)
        gs = torch.empty(E, 2 * I, gs0.shape[1], dtype=torch.uint8, device=device)
        dw = torch.empty(E, *d0.shape, dtype=torch.uint8, device=device)
        ds = torch.empty(E, *ds0.shape, dtype=torch.uint8, device=device)
        g2 = torch.empty(E, dtype=torch.float32)
        d2 = torch.empty(E, dtype=torch.float32)
        for e, ((gw_, gs_), (uw_, us_), s2gu, (dw_, ds_), s2d) in enumerate(experts):
            gw[e, :I].copy_(gw_)
            gw[e, I:].copy_(uw_)
            gs[e, :I].copy_(gs_.view(torch.uint8))
            gs[e, I:].copy_(us_.view(torch.uint8))
            dw[e].copy_(dw_)
            ds[e].copy_(ds_.view(torch.uint8))
            g2[e], d2[e] = float(s2gu), float(s2d)
        return cls(gw, gs, g2.to(device), dw, ds, d2.to(device))

    def bank(self, device):
        return self

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.gw, self.gs, self.dw, self.ds, self.g2, self.d2))

    def expert_bf16(self, e: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Expert e's (gate|up [2 I, H], down [H, I]) in bf16, for a prompt chunk's cuBLAS."""
        return self._unpack(self.gw[e], self.gs[e], self.g2[e], 2 * self.I, self.H), \
            self._unpack(self.dw[e], self.ds[e], self.d2[e], self.H, self.I)

    def expert(self, e: int):
        """(gate, up, down) bf16 weights of expert e (the reference forward's view)."""
        gu, d = self.expert_bf16(e)
        return gu[:self.I], gu[self.I:], d

    def _unpack(self, w, s, s2, N, K) -> torch.Tensor:
        if not w.is_cuda:
            return dequant(w, s.view(torch.float8_e4m3fn), float(s2)).to(torch.bfloat16)
        y = torch.empty(N, K, dtype=torch.bfloat16, device=w.device)
        BK = 256 if K % 256 == 0 else 64
        _nvfp4_unpack[(N, K // BK)](w, s, self._f4, self._f8, float(s2), y, K=K, BK=BK, num_warps=2)
        return y

    def moe_glu(self, x: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        """x [M, H] bf16, ids int32 [M, S] -> yd [M S, H] fp32 (bf16 values, unweighted, in pair order)."""
        M, S = ids.shape
        P = M * S
        flat = ids.reshape(-1).contiguous()
        if not x.is_cuda:
            return self._reference(x, flat, S)
        gu = torch.empty(P, 2 * self.I, dtype=torch.bfloat16, device=x.device)
        yd = torch.empty(P, self.H, dtype=torch.float32, device=x.device)
        _nvfp4_bank[(P, 2 * self.I // 32)](x, x.stride(0), flat, self.gw, self.gs, self.g2, self._f4, self._f8, gu,
                                           NOUT=2 * self.I, KIN=self.H, S_=S, BN=32, BK=128, RELU2=False, num_warps=4)
        from engine.models.flint.kernels import glu_act
        a = glu_act(gu, self.I)
        _nvfp4_bank[(P, self.H // 32)](a, a.stride(0), flat, self.dw, self.ds, self.d2, self._f4, self._f8, yd,
                                       NOUT=self.H, KIN=self.I, S_=1, BN=32, BK=128, RELU2=False, num_warps=4)
        return yd

    def _reference(self, x, flat, S) -> torch.Tensor:
        out = []
        for p, e in enumerate(flat.tolist()):
            gu = (dequant(self.gw[e], self.gs[e].view(torch.float8_e4m3fn), float(self.g2[e])) @ x[p // S].float())
            gu = gu.to(torch.bfloat16)
            a = (F.silu(gu[:self.I]) * gu[self.I:]).float()
            y = dequant(self.dw[e], self.ds[e].view(torch.float8_e4m3fn), float(self.d2[e])) @ a
            out.append(y.to(torch.bfloat16).float())
        return torch.stack(out)
