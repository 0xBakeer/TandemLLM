"""NVFP4 numerics for many small matrices at once, in plain torch: the Kolibri-1 experts.

`tools/quant_nvfp4.py` quantises one projection at a time, and its GPTQ loop is one Python
iteration per input column. A Kolibri-1 layer has 384 experts of three projections each, so the
same loop run per expert is 384 x (2,560 + 512) column steps a layer. Every function here takes a leading expert axis instead (`W [E, N, K]`, `H [E, K, K]`), so a
column step updates all experts at once and a layer is one pass of 2,560 + 512 steps.

The arithmetic is the one `tools/quant_nvfp4.py` and `tools/nvfp4_linear.py` define, written again
without their Triton imports so that it runs on any CPU (the tests) and so that a batch of one gives
the same codes as the serial function (checked on the GPU by `kolibri_quant.py selftest`):

  * the format: `weight` uint8 [N, K/2] (two e2m1 codes, low nibble = even K), `weight_scale`
    e4m3 [N, K/16], `weight_scale_2` fp32 per tensor; w = e2m1(code) * scale * scale_2;
  * `scale_2 = amax(tensor) / (6 * 448)`, so the largest group scale lands on e4m3's maximum;
  * the clip search: per group of 16, the scale multiplier in `CLIP_RATIOS` that minimises the
    activation-weighted squared error `sum_k act[k] * (w_k - q(w_k))^2`;
  * GPTQ (Frantar et al., 2022) on that grid: columns in stored order, each group's scale chosen by
    the clip search on the weights as updated so far, weighted by diag(H).

`scale_2` is per tensor, and a call may stack two tensors along N (an expert's `gate_proj` and
`up_proj` share their input, so they share H and are one call): `s2` is therefore given per row.
"""

from __future__ import annotations

import torch

GROUP = 16
E2M1_MAX = 6.0
E4M3_MAX = 448.0
FP4_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
CLIP_RATIOS = (1.0, 0.95, 0.90, 0.85, 0.80)
TINY = torch.finfo(torch.float32).tiny


def round_e2m1(a: torch.Tensor) -> torch.Tensor:
    """|a| in [0, 6] -> index into FP4_GRID, nearest with ties to even (the hardware converter).
    Identical to `tools.nvfp4_linear._round_e2m1`."""
    grid = FP4_GRID.to(a.device)
    mid = (grid[1:] + grid[:-1]) / 2
    dn = torch.bucketize(a, mid, right=False, out_int32=True)
    up = torch.bucketize(a, mid, right=True, out_int32=True)
    return torch.where((up != dn) & (dn % 2 == 1), up, dn).to(torch.uint8)


def scale_2_of(W: torch.Tensor) -> torch.Tensor:
    """The per-tensor scale of each matrix of `W [..., N, K]`, as fp32 [...]."""
    amax = W.detach().abs().float().amax(dim=(-2, -1))
    return (amax / (E2M1_MAX * E4M3_MAX)).clamp_min(TINY)


def pack(nib: torch.Tensor) -> torch.Tensor:
    """uint8 nibbles [..., K] -> codes [..., K/2], low nibble = even K."""
    return (nib[..., 0::2] | (nib[..., 1::2] << 4)).contiguous()


def dequant(codes: torch.Tensor, scale: torch.Tensor, s2, dtype=torch.float32) -> torch.Tensor:
    """codes [..., N, K/2], scale e4m3 [..., N, K/16], s2 scalar / [...] / [..., N, 1] -> [..., N, K].

    value * float(scale) * s2 in fp32 and one rounding to `dtype`: the arithmetic of
    `NVFP4Block.dequant`."""
    grid = FP4_GRID.to(codes.device)
    lo = (codes & 0x0F).long()
    hi = (codes >> 4).long()
    vlo = grid[lo & 7] * torch.where(lo >= 8, -1.0, 1.0)
    vhi = grid[hi & 7] * torch.where(hi >= 8, -1.0, 1.0)
    vals = torch.stack([vlo, vhi], dim=-1).flatten(-2)
    s2t = torch.as_tensor(s2, dtype=torch.float32, device=codes.device)
    while s2t.dim() and s2t.dim() < codes.dim():
        s2t = s2t[..., None]
    s = scale.float().repeat_interleave(GROUP, -1) * s2t
    return (vals * s).to(dtype)


def _rows_s2(s2: torch.Tensor, E: int, N: int, dev) -> torch.Tensor:
    s2 = torch.as_tensor(s2, dtype=torch.float32, device=dev)
    if s2.dim() == 0:
        s2 = s2.expand(E)
    if s2.dim() == 1:
        s2 = s2[:, None].expand(E, N)
    return s2.reshape(E, N, 1).contiguous()


def _search(x: torch.Tensor, w_k: torch.Tensor, s2: torch.Tensor, ratios) -> torch.Tensor:
    """One group for all experts and rows: x [E, N, 16], w_k [E, 16], s2 [E, N, 1] -> the chosen
    e4m3 scale as float [E, N, 1]. The search of `quant_nvfp4._group_scale`."""
    grid = FP4_GRID.to(x.device)
    gmax = x.abs().amax(dim=-1, keepdim=True)
    best_err, best_s8 = None, None
    for ratio in ratios:
        s8 = ((gmax * ratio / E2M1_MAX) / s2).clamp(0.0, E4M3_MAX).to(torch.float8_e4m3fn).float()
        eff = (s8 * s2).clamp_min(TINY)
        q = round_e2m1((x / eff).abs().clamp(0.0, E2M1_MAX)).long()
        err = (w_k[:, None, :] * (x - grid[q] * torch.sign(x) * eff).pow(2)).sum(-1, keepdim=True)
        if best_err is None:
            best_err, best_s8 = err, s8
        else:
            take = err < best_err
            best_err = torch.where(take, err, best_err)
            best_s8 = torch.where(take, s8, best_s8)
    return best_s8


def clip_batched(W: torch.Tensor, act: torch.Tensor | None, s2=None, *,
                 ratios=CLIP_RATIOS) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The clip search for every matrix of `W [E, N, K]`, weighted per expert by `act [E, K]`
    (mean square input of each channel; None = unweighted). Returns (codes uint8 [E, N, K/2],
    scales e4m3 [E, N, K/16], s2 [E, N, 1]). Same codes as `quant_nvfp4.quantize_clipped`."""
    E, N, K = W.shape
    dev = W.device
    s2r = _rows_s2(scale_2_of(W) if s2 is None else s2, E, N, dev)
    x = W.float().reshape(E, N, K // GROUP, GROUP)
    if act is None:
        act = torch.ones(E, K, device=dev)
    w_k = act.to(dev).float().reshape(E, 1, K // GROUP, GROUP)
    grid = FP4_GRID.to(dev)
    gmax = x.abs().amax(dim=-1, keepdim=True)
    s2g = s2r[..., None]
    best_err, best_s8 = None, None
    for ratio in ratios:
        s8 = ((gmax * ratio / E2M1_MAX) / s2g).clamp(0.0, E4M3_MAX).to(torch.float8_e4m3fn).float()
        eff = (s8 * s2g).clamp_min(TINY)
        q = round_e2m1((x / eff).abs().clamp(0.0, E2M1_MAX)).long()
        err = (w_k * (x - grid[q] * torch.sign(x) * eff).pow(2)).sum(-1, keepdim=True)
        if best_err is None:
            best_err, best_s8 = err, s8
        else:
            take = err < best_err
            best_err = torch.where(take, err, best_err)
            best_s8 = torch.where(take, s8, best_s8)
        del s8, eff, q, err
    eff = (best_s8 * s2g).clamp_min(TINY)
    v = x / eff
    idx = round_e2m1(v.abs().clamp(0.0, E2M1_MAX))
    nib = (idx | ((v < 0).to(torch.uint8) * 8)).reshape(E, N, K)
    return pack(nib), best_s8.reshape(E, N, K // GROUP).to(torch.float8_e4m3fn), s2r


def gptq_batched(W: torch.Tensor, H: torch.Tensor, s2=None, *, ratios=CLIP_RATIOS,
                 damp: float = 0.01, block: int = 128, inplace_h: bool = False,
                 act_order: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    """GPTQ onto the NVFP4 grid for every matrix of `W [E, N, K]` with its own `H [E, K, K]`.

    `quant_nvfp4.gptq_nvfp4` with a leading axis: the damping, the dead-channel rule, the clip
    search at each group's first column and the error propagation are the same, step for step.
    A matrix whose H is not positive definite after the first damping gets ten times more, as in
    the serial function, without touching the others. Returns (codes, scales, s2 [E, N, 1],
    damping attempts per matrix). With `inplace_h` the caller's H is overwritten (saves E*K*K*4
    bytes on a 384-expert layer).

    `act_order` rounds the columns in order of falling diag(H), so the channels the inputs excite
    most are rounded first and their error is pushed onto the quieter ones. An NVFP4 group is 16
    consecutive stored columns, so with act order the group scales are fixed before the loop by
    the clip search on the original weights ("static groups"), not chosen on the updated ones."""
    E, N, K = W.shape
    assert K % GROUP == 0 and block % GROUP == 0, (K, block)
    dev = W.device
    s2r = _rows_s2(scale_2_of(W) if s2 is None else s2, E, N, dev)
    W = W.float().clone()
    H = H.float() if inplace_h and H.dtype == torch.float32 else H.float().clone()
    diag = torch.diagonal(H, dim1=1, dim2=2)
    dead = diag <= 0
    if dead.any():
        e_i, k_i = dead.nonzero(as_tuple=True)
        H[e_i, k_i, k_i] = 1.0
    act_w = diag.clamp_min(0).clone()
    ar = torch.arange(K, device=dev)
    d = damp * diag.mean(dim=1)
    attempts = [0] * E
    perm = eff_col = None
    if act_order:
        _, s_static, _ = clip_batched(W, act_w, s2r, ratios=ratios)
        s8_static = s_static.float()
        eff_col = (s8_static * s2r).clamp_min(TINY).repeat_interleave(GROUP, -1)
        perm = torch.argsort(act_w, dim=1, descending=True, stable=True)
        pn = perm[:, None, :].expand(E, N, K)
        W = W.gather(2, pn)
        eff_col = eff_col.gather(2, pn)
        H = H.gather(1, perm[:, :, None].expand(E, K, K)).gather(2, perm[:, None, :].expand(E, K, K))
        diag = torch.diagonal(H, dim1=1, dim2=2)
    H[:, ar, ar] += d[:, None]
    L, info = torch.linalg.cholesky_ex(H)
    bad = (info != 0).nonzero(as_tuple=True)[0].tolist()
    for a in range(1, 6):
        if not bad:
            break
        for e in bad:
            H[e, ar, ar] += d[e] * (10 ** a)
            attempts[e] = a
        Lb, ib = torch.linalg.cholesky_ex(H[bad])
        L[bad] = Lb
        bad = [e for e, i in zip(bad, ib.tolist()) if i != 0]
    if bad:
        raise RuntimeError(f"H of matrices {bad[:8]} not positive definite even with 1e5 x damping")
    del H, diag
    Hinv = torch.cholesky_inverse(L)
    del L
    Hinv = torch.linalg.cholesky(Hinv, upper=True)
    grid = FP4_GRID.to(dev)
    nib = torch.empty(E, N, K, dtype=torch.uint8, device=dev)
    s8_all = torch.empty(E, N, K // GROUP, dtype=torch.float32, device=dev)
    for i1 in range(0, K, block):
        i2 = min(i1 + block, K)
        W1 = W[:, :, i1:i2].clone()
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[:, i1:i2, i1:i2]
        eff = None
        for j in range(i2 - i1):
            col = i1 + j
            if act_order:
                eff = eff_col[:, :, col]
            elif col % GROUP == 0:
                s8 = _search(W1[:, :, j:j + GROUP], act_w[:, col:col + GROUP], s2r, ratios)
                s8_all[:, :, col // GROUP] = s8[..., 0]
                eff = (s8 * s2r).clamp_min(TINY)[..., 0]
            w = W1[:, :, j]
            v = w / eff
            q = round_e2m1(v.abs().clamp(0.0, E2M1_MAX))
            nib[:, :, col] = q | ((v < 0).to(torch.uint8) * 8)
            deq = grid[q.long()] * torch.sign(v) * eff
            err = (w - deq) / Hinv1[:, j, j][:, None]
            W1[:, :, j:] -= err[:, :, None] * Hinv1[:, j, j:][:, None, :]
            Err1[:, :, j] = err
        if i2 < K:
            W[:, :, i2:] -= torch.bmm(Err1, Hinv[:, i1:i2, i2:])
        del W1, Err1
    if act_order:
        out = torch.empty_like(nib)
        out.scatter_(2, perm[:, None, :].expand(E, N, K), nib)
        return pack(out), s8_static.to(torch.float8_e4m3fn), s2r, attempts
    return pack(nib), s8_all.to(torch.float8_e4m3fn), s2r, attempts


def rel_output_error(W: torch.Tensor, Wq: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """tr(E H E^T) / tr(W H W^T) per matrix, E = W - Wq; W, Wq [E, N, K], H [E, K, K] -> [E].
    The relative squared error of the projection's output over the inputs H was summed from (the
    number `quant_nvfp4.output_error` reports)."""
    Wf = W.float()
    Ef = Wf - Wq.float()
    num = (torch.bmm(Ef, H) * Ef).sum(dim=(1, 2))
    den = (torch.bmm(Wf, H) * Wf).sum(dim=(1, 2))
    return num / den.clamp_min(1e-30)


def quant_head_e4m3(w: torch.Tensor, *, ratios=(1.0,), rows: int = 8192):
    """bf16 [V, K] -> (e4m3 codes [V, K], fp32 scale [V]): one scale per vocabulary row,
    `amax(row) * ratio / 448`. The arithmetic of `tools.head_gemv.quantize_head_fp8`."""
    N, K = w.shape
    codes = torch.empty(N, K, dtype=torch.float8_e4m3fn, device=w.device)
    scale = torch.empty(N, dtype=torch.float32, device=w.device)
    for r0 in range(0, N, rows):
        r1 = min(r0 + rows, N)
        ref = w[r0:r1].float()
        amax = ref.abs().amax(dim=1, keepdim=True).clamp_min(TINY)
        best_err, best_s = None, None
        for ratio in ratios:
            s = (amax * ratio / 448.0).clamp_min(TINY)
            q = (ref / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * s
            err = (ref - q).pow(2).sum(dim=1, keepdim=True)
            if best_err is None:
                best_err, best_s = err, s
            else:
                take = err < best_err
                best_err = torch.where(take, err, best_err)
                best_s = torch.where(take, s, best_s)
        codes[r0:r1] = (ref / best_s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        scale[r0:r1] = best_s[:, 0]
    return codes, scale
