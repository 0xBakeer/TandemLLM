"""On the board: SPD-177's window buffer over real driver pages (engine/drafters/dflash2.py `_window_buffer`).
Rows past the floor keep what was written; rows below it read and write one shared page without a fault;
a lowered floor maps fresh zeroed pages and keeps the rows above the old floor; the pool's account follows.
Run on the box under the lock: python tests/gpu/test_df2_kv_window_gpu.py
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import torch  # noqa: E402

from engine import kvpool  # noqa: E402
from engine.drafters import dflash2 as D  # noqa: E402

SHAPE = (5, 8, 65536, 128)


def test_window_over_driver_pages():
    assert kvpool.Driver.get() is not None, "needs CUDA virtual memory management"
    pool = kvpool.KVPool(hard_floor=0, margin_rows=0, meminfo=lambda: 1 << 50)
    W = D._window_buffer(kvpool)
    b = W(SHAPE, 2, torch.bfloat16, "cuda", owner="w", pool=pool, window=2048)
    t = b.tensor
    b.ensure(20000)
    t[:, :, :20000] = torch.arange(20000, device="cuda", dtype=torch.float32)[:, None].to(torch.bfloat16)
    b.ensure(60000)                                   # the floor moves to 49152
    assert b.floor == 49152 and b.alias
    t[:, :, 50000:60000] = 7.0
    torch.cuda.synchronize()
    t[:, :, :40000].float().sum().item()              # below the floor: the shared page, no fault
    t[:, :, 1000:3000] = 3.0                          # a write below the floor lands in the shared page
    torch.cuda.synchronize()
    assert bool((t[:, :, 50000:60000] == 7.0).all()), "rows above the floor keep what was written"
    assert pool.owner_bytes("w") == b.committed_bytes() == 2 * 40 * b.page
    b.set_floor(b._floor_for(51000))                  # a resume at 51K: its window mapped again
    assert b.floor == 32768
    b.ensure(51000)
    torch.cuda.synchronize()
    assert b.floor == 32768
    assert bool((t[:, :, 32768:49152] == 0).all()), "the uncovered pages are fresh and zeroed"
    assert bool((t[:, :, 50000:51000] == 7.0).all()), "rows above the old floor are kept"
    b.trim(0)
    assert b.floor == 0 and not b.alias and b.committed_bytes() == 0
    b.close()
    return "floor up, rows kept, alias reads/writes, floor down, zeroed, trimmed"


if __name__ == "__main__":
    print(test_window_over_driver_pages())
    print("1 passed")
