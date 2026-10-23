"""Row copies recorded and run as one launch (SPD-201 host cut, QSE_RUNNER_ROWCOPY).

A round's tails move rows and nothing else: each slot's accepted rows of the drafters' tap layers, of the
last hidden state, and of the attention cache down to the path's positions. Issued one torch op at a time,
that is about fifteen launches and three host-to-device copies a slot, ~120 launches a round of eight slots,
and the GPU waits for the host between them. While a recording is open on the thread (`recording`), the
helpers below allocate their outputs and record each row as an entry `(source address, destination address,
4-byte words)`; `flush` sends the table in one copy and runs one kernel over it (two for a cache compaction,
whose rows move inside one buffer). The bits are the bits: a copy is a copy.

A consumer that reads a recorded output before the round's flush calls `flush` first (the helpers that read
them do). Without a recording every helper is the torch statement it stands for.
"""

from __future__ import annotations

import threading

import numpy as np
import torch

_LOCAL = threading.local()


def _rec():
    return getattr(_LOCAL, "rec", None)


def active() -> bool:
    return _rec() is not None


class _Recorder:
    def __init__(self):
        self.a: list = []          # (src, dst, words) run first
        self.b: list = []          # run after `a` (a compaction's second half)
        self.arr_a: list = []      # the same as [k, 3] int64 arrays (a compaction's rows)
        self.arr_b: list = []
        self.keep: list = []       # outputs and scratch the launches write or read, alive until they ran
        self.memo: dict = {}
        self.dev = None

    def _rows(self, out, dst_stride: int, src, rows, col_bytes: int, phase: list) -> None:
        esz = src.element_size()
        sb, db = src.data_ptr(), out.data_ptr()
        rs = src.stride(0) * esz
        ds = dst_stride
        w = col_bytes // 4
        for j, r in enumerate(rows):
            phase.append((sb + int(r) * rs, db + j * ds, w))

    def flush(self) -> None:
        for phase, arrs in ((self.a, self.arr_a), (self.b, self.arr_b)):
            if not phase and not arrs:
                continue
            parts = ([np.asarray(phase, dtype=np.int64).reshape(-1, 3)] if phase else []) + arrs
            tab = torch.from_numpy(np.ascontiguousarray(np.concatenate(parts)) if len(parts) > 1 else parts[0])
            n = tab.shape[0]
            if torch.cuda.is_available():
                tab = tab.pin_memory().to(self.dev, non_blocking=True)
            _launch(tab, n)
            self.keep.append(tab)
            phase.clear()
            arrs.clear()
        self.memo.clear()
        # the launches are queued on the stream; the caching allocator only reuses these blocks after
        # work queued later on the same stream, so dropping the references now is safe
        self.keep.clear()


class recording:
    """`with recording():` the thread's row copies from here are recorded; flushed on exit. Inside an open
    recording it records into that one and flushes what is pending on exit."""

    def __enter__(self):
        self.nested = _rec() is not None
        if not self.nested:
            _LOCAL.rec = _Recorder()
        return _LOCAL.rec

    def __exit__(self, *exc):
        if self.nested:
            _LOCAL.rec.flush()
            return
        rec, _LOCAL.rec = _LOCAL.rec, None
        rec.flush()


def flush() -> None:
    """Run what the thread recorded so far (a consumer about to read a recorded output)."""
    rec = _rec()
    if rec is not None:
        rec.flush()


def _ok(t: torch.Tensor) -> bool:
    return (t.is_cuda and t.dim() == 2 and t.stride(1) == 1
            and (t.shape[1] * t.element_size()) % 4 == 0 and t.data_ptr() % 4 == 0
            and (t.stride(0) * t.element_size()) % 4 == 0)


def rows_cat(srcs: list, rows: list) -> torch.Tensor:
    """`torch.cat([s[rows] for s in srcs], dim=-1)`: the same rows of several [R, C_i] tensors side by side."""
    rec = _rec()
    if rec is None or not all(_ok(s) for s in srcs) or len({s.dtype for s in srcs}) != 1:
        from engine.model import h2d
        sel = h2d(list(rows), torch.long, srcs[0].device)
        return torch.cat([s[sel] for s in srcs], dim=-1)
    key = ("cat", tuple((s.data_ptr(), s.stride(0), s.shape[1]) for s in srcs), tuple(int(r) for r in rows))
    hit = rec.memo.get(key)
    if hit is not None:
        return hit
    rec.dev = srcs[0].device
    width = sum(s.shape[1] for s in srcs)
    out = torch.empty((len(rows), width), dtype=srcs[0].dtype, device=srcs[0].device)
    esz = out.element_size()
    col = 0
    for s in srcs:
        rec._rows(out[:, col:], width * esz, s, rows, s.shape[1] * esz, rec.a)
        col += s.shape[1]
    rec.keep.append(out)
    rec.memo[key] = out
    return out


def rows_of(src: torch.Tensor, rows: list) -> torch.Tensor:
    """`src[rows]` for a [R, C] tensor."""
    rec = _rec()
    if rec is None or not _ok(src):
        from engine.model import h2d
        return src[h2d(list(rows), torch.long, src.device)]
    return rows_cat([src], rows)


def compact(buf: torch.Tensor, start: int, path: list) -> bool:
    """`buf[..., start:start + L, :] = buf[..., start + path, :]` for a contiguous [..., N, D] cache buffer, recorded
    (the rows go to a scratch buffer first, then down, so no row is read after it was written). False, and
    nothing recorded, when the thread has no recording or the buffer does not qualify (the caller runs the
    statement)."""
    rec = _rec()
    if rec is None or not buf.is_cuda or not buf.is_contiguous() or buf.dim() < 2:
        return False
    esz = buf.element_size()
    N, D = buf.shape[-2], buf.shape[-1]
    rb = D * esz
    if rb % 4 or buf.data_ptr() % 4:
        return False
    regions = buf.numel() // (N * D)
    L = len(path)
    rec.dev = buf.device
    tmp = torch.empty((regions, L, D), dtype=buf.dtype, device=buf.device)
    base = _bases(buf.data_ptr(), regions, N * rb)
    j = np.arange(L, dtype=np.int64)
    a = np.empty((regions, L, 3), dtype=np.int64)
    b = np.empty((regions, L, 3), dtype=np.int64)
    a[:, :, 0] = base + ((start + np.asarray(path, dtype=np.int64)) * rb)[None, :]
    a[:, :, 1] = tmp.data_ptr() + (np.arange(regions, dtype=np.int64)[:, None] * L + j[None, :]) * rb
    a[:, :, 2] = rb // 4
    b[:, :, 0] = a[:, :, 1]
    b[:, :, 1] = base + ((start + j) * rb)[None, :]
    b[:, :, 2] = rb // 4
    rec.arr_a.append(a.reshape(-1, 3))
    rec.arr_b.append(b.reshape(-1, 3))
    rec.keep.append(tmp)
    return True


_BASES: dict = {}


def _bases(ptr: int, regions: int, stride: int) -> np.ndarray:
    """[regions, 1] the start address of every region of a buffer (cached: a cache buffer never moves)."""
    key = (ptr, regions, stride)
    b = _BASES.get(key)
    if b is None:
        if len(_BASES) > 256:
            _BASES.clear()
        b = _BASES[key] = ptr + np.arange(regions, dtype=np.int64)[:, None] * stride
    return b


_KERNEL = None


def _launch(tab: torch.Tensor, n: int) -> None:
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = _build()
    _KERNEL[(n,)](tab, BLOCK=1024, num_warps=4)


def _build():
    import triton
    import triton.language as tl

    @triton.jit
    def _copy_rows(TAB, BLOCK: tl.constexpr):
        i = tl.program_id(0)
        src = tl.load(TAB + 3 * i).to(tl.pointer_type(tl.int32))
        dst = tl.load(TAB + 3 * i + 1).to(tl.pointer_type(tl.int32))
        n = tl.load(TAB + 3 * i + 2)
        offs = tl.arange(0, BLOCK)
        for o in range(0, n, BLOCK):
            m = o + offs < n
            tl.store(dst + o + offs, tl.load(src + o + offs, mask=m), mask=m)

    return _copy_rows


def check(seed: int = 0) -> list[str]:
    """The recorded copies against the torch statements, bit for bit (a GPU)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    out = []
    taps = [torch.randn(16, 5120, device="cuda", generator=g).to(torch.bfloat16) for _ in range(5)]
    hid = torch.randn(1, 16, 5120, device="cuda", generator=g).to(torch.bfloat16)
    for rows in ([0], [0, 1, 2], [0, 2, 5, 9], [0, 1, 3, 7, 12, 15]):
        ref = torch.cat([t[torch.tensor(rows, device="cuda")] for t in taps], dim=-1)
        refh = hid[0][torch.tensor(rows, device="cuda")]
        kv = torch.randn(16, 1, 4, 64, 256, device="cuda", generator=g).to(torch.bfloat16)
        kref = kv.clone()
        start = 20
        sel = start + torch.tensor(rows, device="cuda")
        kref[..., start:start + len(rows), :] = kref[..., sel, :]
        with recording():
            got = rows_cat(taps, rows)
            got2 = rows_cat(taps, rows)
            goth = rows_of(hid[0], rows)
            assert compact(kv, start, rows)
        torch.cuda.synchronize()
        assert got is got2
        assert torch.equal(got, ref), rows
        assert torch.equal(goth, refh), rows
        assert torch.equal(kv, kref), rows
        out.append(f"rows {rows}: taps, hidden and the cache compaction equal the torch statements")
    return out
