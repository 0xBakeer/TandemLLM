"""SPD-249 P4: QWEN38_FN_PF_NOCOPY's switch and the PLE states from the rows' words (CPU; the kernels' bits are
tests/gpu/test_fn_pf_nocopy_gpu.py's and tools/fn_pf_hc_check.py's).

  * QWEN38_FN_PF_NOCOPY is a Flash-Next knob, off by default, owned by the engine and read by the prompt chunk's
    paths only (`_gdn_pre`, `_attn_pre`, `_ple_pre`) and the layer-major window
  * `ple.row_states` gives `decode_rows`' states for every K a 16-bit state allows, and `decode_rows(words=True)`
    the same rows bit for bit
  * `fn_prefill_kernels.gated_norm_ok` refuses CPU tensors, and `qsa_scores_rows(lean=True)` is the torch path's
    scores off the device

Run: python tests/test_fn_pf_nocopy.py
"""

from __future__ import annotations

import inspect
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def test_knob_is_registered_off():
    from engine.settings import KNOBS
    from engine.models.flashnext import engine as E
    assert KNOBS["FN_PF_NOCOPY"] == ("0", "engine/models/flashnext/engine.py")
    assert E.PF_NOCOPY is (os.environ.get("QWEN38_FN_PF_NOCOPY") == "1")


def test_knob_is_read_by_the_prompt_paths_only():
    from engine.models.flashnext import engine as E
    from engine.models.flashnext import layer_major as LM
    src = inspect.getsource(E.FlashNextEngine)
    assert src.count("PF_NOCOPY") == 4, src.count("PF_NOCOPY")
    g = inspect.getsource(E.FlashNextEngine._gdn_pre)
    assert "view=PF_NOCOPY" in g and "if PF_NOCOPY and PK.gated_norm_ok(o, z)" in g
    assert "lean=PF_NOCOPY" in inspect.getsource(E.FlashNextEngine._attn_pre)
    assert "words=PF_NOCOPY" in inspect.getsource(E.FlashNextEngine._ple_pre)
    assert inspect.getsource(LM.window).count("E.PF_NOCOPY") == 1
    from engine.models.flashnext.ple import decode_rows
    assert inspect.signature(decode_rows).parameters["words"].default is False


def _stream_states(packed: torch.Tensor, K: int) -> torch.Tensor:
    """`decode_rows`' states as it has built them: the rows' bit stream, 16 bits a state."""
    from engine.models.flashnext.ple import ROW_DIM, _ring_src
    wd = packed[:, 1:].view(torch.uint16).to(torch.int32)
    stream = ((wd.unsqueeze(-1) >> torch.arange(16, dtype=torch.int32)) & 1).reshape(packed.shape[0], ROW_DIM * K)
    return (stream[:, _ring_src(K, packed.device)] << torch.arange(16, dtype=torch.int32)).sum(-1)


def test_row_states_are_the_bit_streams():
    from engine.models.flashnext.ple import ROW_DIM, row_states
    g = torch.Generator().manual_seed(3)
    for K in range(1, 17):
        packed = torch.randint(-32768, 32768, (37, 1 + ROW_DIM * K // 16), generator=g, dtype=torch.int16)
        packed[0, 1:] = -1                                   # every bit set
        packed[1, 1:] = 0
        got = row_states(packed, K)
        want = _stream_states(packed, K)
        assert got.dtype == torch.int32 and got.shape == (37, ROW_DIM), (K, got.dtype, got.shape)
        assert int(got.min()) >= 0 and int(got.max()) <= 0xffff, K
        assert torch.equal(got.long(), want.long()), K


def test_decode_rows_words_is_bitwise():
    from engine.models.flashnext.ple import ROW_DIM, decode_rows, mul1_codebook
    g = torch.Generator().manual_seed(4)
    cb = mul1_codebook()
    for K in (2, 3, 4, 5, 6, 8):
        packed = torch.randint(-32768, 32768, (21, 1 + ROW_DIM * K // 16), generator=g, dtype=torch.int16)
        packed[:, 0] = (torch.rand(21, generator=g) * 0.1 + 0.01).half().view(torch.int16)
        heads = torch.randint(0, 3, (21,), generator=g)
        bias = (torch.randn(3, ROW_DIM, generator=g) * 0.01).half()
        a = decode_rows(packed, heads, bias, K, cb)
        b = decode_rows(packed, heads, bias, K, cb, words=True)
        assert a.dtype == b.dtype == torch.float32
        assert torch.equal(a.view(torch.int32), b.view(torch.int32)), K


def test_kernel_paths_refuse_or_fall_back_off_the_device():
    from tools import fn_prefill_kernels as PK
    o = torch.zeros(6, 128, dtype=torch.bfloat16)
    z = torch.zeros(2, 3 * 128 + 5, dtype=torch.bfloat16)[:, 5:]
    assert not PK.gated_norm_ok(o, z)
    g = torch.Generator().manual_seed(5)
    iq = (torch.randn(9, 4, 128, generator=g) * 0.3).to(torch.bfloat16)
    pool = (torch.randn(40, 128, generator=g) * 0.3).to(torch.bfloat16)
    nblk = torch.randint(0, 41, (9,), generator=g)
    a = PK.qsa_scores_rows(iq, pool, nblk, 64)
    b = PK.qsa_scores_rows(iq, pool, nblk, 64, lean=True)
    assert torch.equal(a.view(torch.int32), b.view(torch.int32))
    assert inspect.signature(PK.mix_tail).parameters["out"].default is None


if __name__ == "__main__":
    fails = 0
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        try:
            fn()
            print(f"  {name:55s} ok")
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"  {name:55s} FAIL {type(e).__name__}: {e}")
    print(f"{len(tests) - fails} passed, {fails} failed")
    sys.exit(1 if fails else 0)
