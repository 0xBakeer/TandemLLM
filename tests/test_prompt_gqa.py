"""SRV-681: attention over the KV heads as they are (`enable_gqa` on SDPA's flash kernel) in place of a copy
repeated per query head, on three paths, each against the code it replaced, bit for bit on the GPU:

  * Ornith's prompt chunks and MTP primes (engine/models/qwen3_5_moe/kernels.py `prompt_attention`,
    SRV-673), now with blocks of one row: alone, the flash kernel folds one row's query heads into rows
    (another split, other bits), so a one-row block runs as two rows, the row twice;
  * LFM2.5's prompt chunks (engine/models/lfm2_moe/engine.py `chunk_attention`, Hq 32, Hk 8, D 64);
  * NemotronH's MTP head (engine/models/nemotron_h/mtp.py `gqa_attention`, Hq 32, Hk 2, D 128): its steps
    (one row, no mask) and its primes (rows causal to the lower right).

Small shapes, a few MB on the GPU; no weights. On the CPU the new paths stand aside (the callers keep the
copy) and Ornith's matches its fp32 twin.

Run: python tests/test_prompt_gqa.py
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.models.lfm2_moe import engine as LE  # noqa: E402
from engine.models.nemotron_h import mtp as NM  # noqa: E402
from engine.models.qwen3_5_moe import kernels as QK  # noqa: E402

CUDA = torch.cuda.is_available()
DEV = "cuda" if CUDA else "cpu"


def _cache(HK, cap, D, seed):
    g = torch.Generator().manual_seed(seed)
    kc = (torch.randn(HK, cap, D, generator=g) * 0.5).to(torch.bfloat16).to(DEV)
    vc = torch.randn(HK, cap, D, generator=g).to(torch.bfloat16).to(DEV)
    return kc, vc, g


def _copy(qh, k, v, mask):
    """The code the paths replaced: SDPA over K and V repeated to every query head."""
    rep = qh.shape[1] // k.shape[1]
    return F.scaled_dot_product_attention(qh, k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1), attn_mask=mask)


def test_ornith_prompt_attention_one_row_blocks():
    """Blocks of 128 over T = 1, 2, 129 and 257 rows (a last block of one row) at L from 0 to 3,000:
    each block equals SDPA over the repeated copy bit for bit (0.13.0's arithmetic)."""
    from torch.nn.attention.bias import causal_lower_right
    HQ, HK, D = 32, 2, 256
    kc, vc, g = _cache(HK, 3400, D, 2)
    n = 0
    for L in (0, 1, 127, 1000, 3000):
        for T in (1, 2, 129, 257):
            if L + T > kc.shape[1]:
                continue
            q = torch.randn(T, HQ, D, generator=g).to(torch.bfloat16).to(DEV)
            o = QK.prompt_attention(q, kc, vc, L, block=128)
            if not CUDA:
                ctl = torch.tensor([[L + t + 1, L + t + 1] for t in range(T)], dtype=torch.int32)
                ref = QK.attn_rows_reference(q.cpu(), kc.cpu(), vc.cpu(), ctl, torch.zeros(T, 1, dtype=torch.int32))
                assert (o.float().cpu() - ref).abs().max() < 2e-2 * ref.abs().max(), (L, T)
                n += 1
                continue
            for q0 in range(0, T, 128):
                q1 = min(q0 + 128, T)
                k, v = kc[:, :L + q1].unsqueeze(0), vc[:, :L + q1].unsqueeze(0)
                rep = _copy(q[q0:q1].transpose(0, 1).unsqueeze(0), k, v, causal_lower_right(q1 - q0, L + q1))
                assert torch.equal(rep[0].transpose(0, 1), o[q0:q1]), (L, T, q0)
                n += 1
    if CUDA:
        assert QK.FLASH_GQA is True, "the flash kernel refused the KV heads as they are on this GPU"
    return f"{n} blocks equal"


def test_lfm25_chunk_attention():
    """LFM2.5's prompt chunk (Hq 32, Hk 8, D 64) over a strided cache view: the KV heads as they are equal
    the copy bit for bit at T = 1, 2, 37 and 600 rows after 0 to 5,000 cached rows."""
    from torch.nn.attention.bias import causal_lower_right
    HQ, HK, D = 32, 8, 64
    kc, vc, g = _cache(HK, 5700, D, 3)
    n = 0
    for L in (0, 1, 63, 1000, 5000):
        for T in (1, 2, 37, 600):
            k, v = kc[:, :L + T].unsqueeze(0), vc[:, :L + T].unsqueeze(0)
            qh = torch.randn(T, HQ, D, generator=g).to(torch.bfloat16).to(DEV).transpose(0, 1).unsqueeze(0)
            o = LE.chunk_attention(qh, k, v)
            if not CUDA:
                assert o is None, "the CPU keeps the repeated copy"
                continue
            assert o is not None and LE.FLASH_GQA is True, "the flash kernel refused the KV heads as they are"
            assert torch.equal(o, _copy(qh, k, v, causal_lower_right(T, L + T))), (L, T)
            n += 1
    return f"{n} chunks equal" if CUDA else "cpu: the copy"


def test_nemotron_h_mtp_attention():
    """NemotronH's MTP head (Hq 32, Hk 2, D 128): a step (one row, no mask) and a prime (rows causal to the
    lower right) equal the copy bit for bit; the head's cache is read as a view, its rows as written."""
    from torch.nn.attention.bias import causal_lower_right
    HQ, HK, D = 32, 2, 128
    kc, vc, g = _cache(HK, 9000, D, 4)
    n = 0
    for pos0 in (0, 1, 500, 4096, 8000):
        for T in (1, 2, 3, 300):
            k, v = kc[:, :pos0 + T].unsqueeze(0), vc[:, :pos0 + T].unsqueeze(0)
            qh = torch.randn(T, HQ, D, generator=g).to(torch.bfloat16).to(DEV).transpose(0, 1).unsqueeze(0)
            o = NM.gqa_attention(qh, k, v)
            if not CUDA:
                assert o is None, "the CPU keeps the repeated copy"
                continue
            assert o is not None and NM.FLASH_GQA is True, "the flash kernel refused the KV heads as they are"
            ref = _copy(qh, k, v, None if T == 1 else causal_lower_right(T, pos0 + T))
            assert torch.equal(o, ref), (pos0, T)
            n += 1
    return f"{n} calls equal" if CUDA else "cpu: the copy"


def test_one_row_alone_takes_other_bits():
    """Why one row runs as two: on this GPU the flash kernel with the KV heads as they are and a single
    query row gives other bits than the copy (it folds the query heads into rows). When a torch build
    stops doing so the test says so, and the two-row step can go."""
    if not CUDA:
        return "cpu: nothing to show"
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.nn.attention.bias import causal_lower_right
    HQ, HK, D, L = 32, 2, 256, 5000
    kc, vc, g = _cache(HK, L + 8, D, 5)
    qh = torch.randn(1, HQ, D, generator=g).to(torch.bfloat16).to(DEV).transpose(0, 1).unsqueeze(0)
    k, v = kc[:, :L + 1].unsqueeze(0), vc[:, :L + 1].unsqueeze(0)
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        alone = F.scaled_dot_product_attention(qh, k, v, attn_mask=causal_lower_right(1, L + 1), enable_gqa=True)
    ref = _copy(qh, k, v, causal_lower_right(1, L + 1))
    same = torch.equal(alone, ref)
    print("    one row alone:", "the copy's bits on this build" if same else "other bits (the two-row step is needed)")


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
