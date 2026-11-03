"""EXL3 experts with a non-gated relu^2 MLP (NemotronH: up, down, no gate) on the board, row-invariant (ENG-E12).

The SwiGLU bank of tools/fn_kernels.py (`moe_gu` / `moe_down`) reads a gate and an up projection and
forms bf16(bf16(silu(g)) * u); NemotronH's experts have no gate and take bf16(relu(bf16(u))^2). These
two kernels are that bank's for the relu^2 shape, with the same parts: a CTA owns one (row, slot) pair
and one 128-column block, rotates the pair's input with the expert's own suh into shared memory, runs
`RowGemm` (the trellis GEMV over its K range in contiguous warp slices, the warps added in warp order)
and applies the output rotation and svh:

  up      a[p] = bf16(relu(bf16(u))^2),   u = up_e(x[p / S])            bf16 [P, I]
  down    yd[p] = bf16-rounded fp32 of down_e(a[p])                     fp32 [P, X]

which are the roundings of the BF16 experts and the IQ2_XXS bank (engine/models/nemotron_h/kernels.py
`moe_relu2`, tools/iq2_linear.py `IQ2Bank.moe_relu2`): the combine weights and adds them in fp32. A pair's
arithmetic does not depend on the other pairs or on M, so a verify block's row is the decode step's.
The trellis decode, the Hadamard, the GEMV body (`RowGemm`, `red_rot`) and `MoEArgs` are
tools/fn_kernels.py's (exllamav3's bit extraction and codebook, MIT, Copyright (c) 2025 Turboderp; the
notice travels with tools/csrc/exl3_linear/exl3.cu). Built on first use for sm_121a.

On the CPU (the tests) `Relu2Bank.moe_relu2` runs the same arithmetic in torch, each projection through
tools/exl3_linear.py `reference_matmul` (the kernels' yardstick).
"""

from __future__ import annotations

import os
import sys

if os.path.dirname(os.path.dirname(os.path.abspath(__file__))) not in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from engine import native_build as nb  # noqa: E402
from tools import exl3_linear as X  # noqa: E402
from tools import fn_kernels as FK  # noqa: E402

# fn_kernels' device code up to and including MoEArgs: the helpers, RowGemm, red_rot (inside the
# anonymous namespace that tools/exl3_linear.py's helpers open)
_BASE = FK._CUDA[: FK._CUDA.index("template <int KB, int MOE_ST, int WARPS>\n__device__ __forceinline__ void moe_gu_body")]
# and its grouping (SPD-125): MOE_GR, moe_groups_kernel, RowGemmG, red_rot_row
_GROUPS = FK._CUDA[FK._CUDA.index("#define MOE_GR 8"):
                   FK._CUDA.index("template <int KB, int MOE_ST, int WARPS>\n__device__ __forceinline__ void moe_gu_group_body")]

_CUDA = nb.source(__file__, "relu2.cu")

_CPP = nb.source(__file__, "relu2.cpp")

_MOD = None

#: SPD-125: a verify block's pairs that share an expert decode its tiles once (a group of up to
#: tools/fn_kernels.MOE_GR pairs a CTA); "off" = one CTA a pair
GROUP = os.environ.get("QSE_NH_BANK_GROUP", "on") != "off"


def source() -> str:
    return "#define EXL3_ST 3\n" + FK._HELPERS + _BASE + _GROUPS + _CUDA


def _module():
    global _MOD
    if _MOD is None:
        src = source()
        tag = nb.tag(_CPP, src)
        _MOD = nb.load(f"exl3_relu2_{tag}", cpp_sources=[_CPP], cuda_sources=[src],
                       functions=["nh_relu2_up", "nh_relu2_down", "nh_relu2_groups", "nh_relu2_up_group",
                                  "nh_relu2_down_group"],
                       extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_121a,code=sm_121a"],
                       verbose=False)
    return _MOD


class Relu2Bank:
    """The routed experts of one NemotronH MoE layer as EXL3 blocks: up [I, X] and down [X, I] of each
    expert (engine orientation), their pointers for the kernels ([E, 2]: up, down) and the blocks for
    the CPU path and the prompt chunk (`expert`)."""

    def __init__(self, blocks: list):
        """blocks: [(up EXL3Block, down EXL3Block)] in expert order."""
        self.blocks = blocks
        self.E = len(blocks)
        up, down = blocks[0]
        self.I, self.X = up.N, up.K
        for e, (u, d) in enumerate(blocks):
            if (u.N, u.K, d.N, d.K) != (self.I, self.X, self.X, self.I):
                raise ValueError(f"expert {e}: up {u.shape}, down {d.shape}; the bank wants [{self.I}, {self.X}] and its transpose")
        self._dev = None

    @property
    def nbytes(self) -> int:
        return sum(b.nbytes for e in self.blocks for b in e)

    def expert(self, e: int):
        return self.blocks[e]

    def _pointers(self):
        if self._dev is None:
            tr, kb, su, sv = [], [], [], []
            for blks in self.blocks:
                for b in blks:
                    assert b.t.is_contiguous() and b.t.data_ptr() % 16 == 0 and b.suh.shape[0] == 1
                    tr.append(b.t.data_ptr())
                    kb.append(b.kb)
                    su.append(b.suh[0].data_ptr())
                    sv.append(b.svh.data_ptr())
            dev = self.blocks[0][0].t.device
            ks = set(kb)
            if not ks <= set(FK.MOE_K):
                raise ValueError(f"expert K {sorted(ks)}: the kernels take K 2..8, 1.5, 2.5 and 3.5")
            self._dev = (torch.tensor(tr, dtype=torch.int64, device=dev), torch.tensor(kb, dtype=torch.int32, device=dev),
                         torch.tensor(su, dtype=torch.int64, device=dev), torch.tensor(sv, dtype=torch.int64, device=dev),
                         max(ks, key=X.tile_width))
        return self._dev

    def moe_relu2(self, x: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        """x [M, X] bf16, ids int32 [M, S] -> yd [M S, X] fp32 (bf16 values, unweighted, pair order)."""
        M, S = ids.shape
        P = M * S
        flat = ids.reshape(-1).to(torch.int32).contiguous()
        if not x.is_cuda:
            yd = torch.empty(P, self.X, dtype=torch.float32)
            for p, e in enumerate(flat.tolist()):
                up, down = self.blocks[e]
                a = torch.relu(X.reference_matmul(x[p // S:p // S + 1], up)).square().to(torch.bfloat16)
                yd[p] = X.reference_matmul(a, down)[0].float()
            return yd
        tr, kb, su, sv, kbmax = self._pointers()
        if x.stride(1) != 1 or x.stride(0) % 4:
            x = x.contiguous()
        a = torch.empty(P, self.I, dtype=torch.bfloat16, device=x.device)
        yd = torch.empty(P, self.X, dtype=torch.float32, device=x.device)
        mod = _module()
        if M > 1 and GROUP:
            # SPD-125: a verify block's pairs of one expert share its tile decode (each pair's bits its lone call's)
            g = torch.empty(P, 2 + FK.MOE_GR, dtype=torch.int32, device=x.device)
            ng = torch.empty(1, dtype=torch.int32, device=x.device)
            mod.nh_relu2_groups(flat, FK.expert_order(flat), g, ng)
            mod.nh_relu2_up_group(x, g, ng, S, tr, kb, su, sv, a, kbmax)
            mod.nh_relu2_down_group(a, g, ng, tr, kb, su, sv, yd, kbmax)
            return yd
        mod.nh_relu2_up(x, flat, tr, kb, su, sv, a, kbmax)
        mod.nh_relu2_down(a, flat, tr, kb, su, sv, yd, kbmax)
        return yd
