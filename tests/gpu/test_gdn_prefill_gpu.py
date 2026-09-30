"""On the board: the fused prefill's state at a chunk line inside the call (`state_out`).

The resident prefix's end anchor asks the prefill's last forward for the recurrent state at a
64-row line inside it. Two things have to hold, both bit for bit:

  * asking changes nothing: the output and the final state are the same with and without it;
  * the state it hands back is the state a call over only the rows before the line returns.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tools.gdn_prefill_kernels import CHUNK, fused_chunk_prefill  # noqa: E402

G = torch.Generator(device="cuda").manual_seed(7)


def _inputs(T: int, H: int = 48, D: int = 128):
    q = (torch.randn(1, T, H, D, device="cuda", generator=G) * 0.5).to(torch.bfloat16)
    k = (torch.randn(1, T, H, D, device="cuda", generator=G) * 0.5).to(torch.bfloat16)
    v = (torch.randn(1, T, H, D, device="cuda", generator=G) * 0.5).to(torch.bfloat16)
    g = -torch.rand(1, T, H, device="cuda", generator=G) * 0.2
    beta = torch.rand(1, T, H, device="cuda", generator=G).to(torch.bfloat16)
    S0 = torch.randn(1, H, D, D, device="cuda", generator=G) * 0.1
    return q, k, v, g, beta, S0


def test_asking_for_the_state_changes_nothing():
    n = 0
    for T in (64, 640, 1000, 1024):
        q, k, v, g, beta, S0 = _inputs(T)
        o_a, S_a = fused_chunk_prefill(q, k, v, g, beta, S0)
        o_r, S_r = fused_chunk_prefill(q, k, v, g, beta, S0)
        assert torch.equal(o_a, o_r) and torch.equal(S_a, S_r), (T, "not deterministic")
        nc = (T + CHUNK - 1) // CHUNK
        for j in sorted({1, nc // 2 or 1, nc}):
            out = torch.empty_like(S0)
            o_b, S_b = fused_chunk_prefill(q, k, v, g, beta, S0, state_at=j, state_out=out)
            assert torch.equal(o_a, o_b) and torch.equal(S_a, S_b), (T, j)
            n += 1
    return f"{n} calls: output and final state bit-equal with and without state_out"


def test_the_state_is_the_one_a_shorter_call_returns():
    n = 0
    for T in (640, 1000, 1024):
        q, k, v, g, beta, S0 = _inputs(T)
        nc = (T + CHUNK - 1) // CHUNK
        for j in range(1, nc + 1, 3):
            r = min(j * CHUNK, T)
            if r % CHUNK and r != T:
                continue
            out = torch.empty_like(S0)
            fused_chunk_prefill(q, k, v, g, beta, S0, state_at=j, state_out=out)
            _, S_r = fused_chunk_prefill(q[:, :r], k[:, :r], v[:, :r], g[:, :r], beta[:, :r], S0)
            assert torch.equal(out, S_r), (T, j, (out - S_r).abs().max().item())
            n += 1
    return f"{n} lines: the state at the line equals a call over the rows before it, bit for bit"


def test_a_bad_buffer_is_refused():
    q, k, v, g, beta, S0 = _inputs(128)
    try:
        fused_chunk_prefill(q, k, v, g, beta, S0, state_at=1,
                            state_out=torch.empty_like(S0, dtype=torch.bfloat16))
    except ValueError:
        return "a bf16 buffer is refused"
    raise AssertionError("a bf16 state buffer was accepted")


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:60s} ok   {fn() or ''}", flush=True)
            passed += 1
    print(f"{passed} passed")
