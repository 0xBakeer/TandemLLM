"""SRV-629: the slot choice for a request with images. CPU only, no engine: `_par_side_need`,
`_side_home`, the view keep and the admission with tests/test_par_continue.py's stand-ins.

THE CASE (live 2026-10-19). Two Chat-tab agent runs called render_page; every later request
carried 3 to 8 screenshots and `_par_side_need` refused each one a side slot, so after the first
carry into slot 0 all 31 requests of both chats ran there (56.6 min of wall time against 18.9
min of decoding) while slot 1 stayed idle. tests/test_side_images.py proves a side slot's
engine decodes an image request as slot 0's does; this file checks the choice:

  1  the rows: the window, the output cut and the admission count each image's expanded rows,
     not its one placeholder (8 screenshots: about 7,700 rows); `_mm_keys` is the context's
     `key_ids`, on the host
  2  `SIDE_IMAGES` off: slot 0 only, as 0.12.0; on: a side slot when it fits

Run: python tests/test_side_images_route.py
"""

from __future__ import annotations

import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from engine import vision as V  # noqa: E402
from server import app as A  # noqa: E402
from server import slots as S  # noqa: E402
from test_par_continue import CHUNK, HEAD, FakeResident, _restore, _session, _setup  # noqa: E402

IMG = 900_001                     # the template's image placeholder in these stand-in ids
MERGE = 2


def shot(seed: int, h: int = 54, w: int = 84) -> V.Image:
    """A preprocessed screenshot stand-in: `h x w` patches, `h * w / 4` rows (54 x 84: 1,134)."""
    g = torch.Generator().manual_seed(seed)
    pv = torch.randn(h * w, 4, generator=g)
    grid = (1, h, w)
    return V.Image(pv, grid, V.digest_of(pv, grid), "data")


def tower(own=None):
    """The vision tower as the router reads it: its placeholder and its merge (the Qwen towers),
    or a tower with its own `context` (Super VL)."""
    t = types.SimpleNamespace(cfg=types.SimpleNamespace(image_token_id=IMG,
                                                        spatial_merge_size=MERGE))
    if own is not None:
        t.context = own
    return t


def with_shots(ids: list[int], n: int) -> list[int]:
    return list(ids) + [IMG] * n


def need_img(ids, images, max_new=32_000, plan_active=True):
    """`_par_side_need` for a request with `images`, the renderer answering `ids`."""
    real = A.build_prompt

    class Plan:
        active = plan_active

    def fake(body, client=None, *, count=False, left_out=0):
        assert count == "ids"
        return list(ids)
    A.build_prompt = fake
    try:
        why: dict = {}
        need = A._par_side_need({"messages": []}, None, 0, max_new, Plan(), images, None, why)
        return need, why
    finally:
        A.build_prompt = real


class flag:
    """`SIDE_IMAGES` and the tower for the block."""

    def __init__(self, value: str, t=None):
        self.value, self.t = value, t if t is not None else tower()

    def __enter__(self):
        self.was, self.had = A.SIDE_IMAGES, A.STATE.get("vision")
        A.SIDE_IMAGES = self.value
        A.STATE["vision"] = self.t
        return self

    def __exit__(self, *exc):
        A.SIDE_IMAGES = self.was
        if self.had is None:
            A.STATE.pop("vision", None)
        else:
            A.STATE["vision"] = self.had


# ------------------------------------------------------------------ 1. the rows
def test_mm_keys_are_the_contexts_key_ids():
    ims = [shot(1), shot(2, 40, 64), shot(1)]                   # the same screenshot twice
    ids = with_shots([5, 6, 7], 1) + [8, 9] + with_shots([], 1) + [10] + with_shots([], 1) + [11]
    with flag("on"):
        keys = A._mm_keys(ids, ims)
        full, spans = V.expand(ids, ims, IMG, MERGE)
        eng_cfg = types.SimpleNamespace(rotary_dim=16, mrope_section=[2, 1, 1])
        mm = V.MMContext(full, spans, ims, A.STATE["vision"], eng_cfg)
        assert keys == list(mm.key_ids), "the host keys are the caches' keys"
        assert len(keys) == len(ids) - 3 + 1_134 + 640 + 1_134
        assert A._mm_rows(len(ids), ims) == len(keys)
        assert keys[3] == keys[3 + 1_134 + 2 + 640 + 1] == V.key_id(ims[0].digest)
        assert keys[3 + 1_134 + 2] != keys[3]
        assert A._mm_keys([5, IMG], ims) is None, "a placeholder short: the handler's 400"
    # Super VL builds its own context: its images stay on slot 0; its rows are still counted
    own = tower(own=lambda ids, images, cache=None, on_encode=None: (
        [t for i in ids for t in ([IMG - 1] + [IMG] * 9 + [IMG + 1] if i == IMG else [i])], None))
    with flag("on", own):
        assert A._mm_keys(ids, ims) is None
        assert A._mm_rows(len(ids), ims) == len(ids) + 3 * 10
    return f"{len(keys)} rows for {len(ids)} rendered tokens"


def test_the_window_counts_the_expanded_rows():
    """Eight screenshots (9,072 rows) in a 27,000-token conversation on a 65,536-token side slot:
    counted as eight placeholders the output limit would be cut to 30,269 and the generation
    would run past the window; the rows leave it 21,205."""
    r1 = _session(1, HEAD, 1_500) + [7, 8, 9]
    ims = [shot(10 + i) for i in range(8)]
    ids = with_shots(_session(4, HEAD, 20_000), 8) + [7, 8, 9]
    pool, saved = _setup(FakeResident(r1), view=r1)
    try:
        with flag("on"):
            need, why = need_img(ids, ims)
            margin = A.SIDE_MARGIN + A.PLAN_MARGIN
            rows = len(ids) - 8 + 8 * 1_134
            assert why["n_rows"] == rows and why["n_images"] == 8, why
            assert need == rows + margin + A.SIDE_MIN_ROOM and why["clip"], (need, why)
            room = A._side_room(pool.side()[0], why["n_rows"], margin, 32_000)
            assert room == 65_536 - rows - margin == 21_205, room
    finally:
        _restore(saved)
    return f"{rows} rows, need {need}, max_tokens 32000 -> {room}"


def test_the_admission_reserves_the_expanded_rows():
    seen = []

    class Mem:
        def workspace(self, n):
            seen.append(n)
            return n

    class H:
        def _client_gone(self):
            return False
    adm = S.Admission(0, meminfo=lambda: {"MemAvailable": 1 << 40})
    pool = types.SimpleNamespace(admission=adm, turn=None)
    saved = A.STATE.get("mem")
    A.STATE["mem"] = Mem()
    try:
        assert A.Handler._par_admit(H(), pool, "a", torch.zeros(600)) is True
        assert A.Handler._par_admit(H(), pool, "b", torch.zeros(600), rows=9_672) is True
    finally:
        if saved is None:
            A.STATE.pop("mem", None)
        else:
            A.STATE["mem"] = saved
    assert seen == [600, 9_672] and adm.held["b"] == 9_672, (seen, adm.held)


# ------------------------------------------------------------------ 2. the flag
def test_flag_off_keeps_images_on_slot_0():
    r1 = _session(1, HEAD, 1_500) + [7, 8, 9]
    ids = with_shots(_session(4, HEAD, 900), 2) + [7, 8, 9]
    pool, saved = _setup(FakeResident(r1), view=r1)
    try:
        with flag("off"):
            need, why = need_img(ids, [shot(1), shot(2)], max_new=4_000)
            assert need is None and why == {"images": True}, why
    finally:
        _restore(saved)


def test_flag_on_gives_a_new_chat_with_images_a_side_slot():
    r1 = _session(1, HEAD, 1_500) + [7, 8, 9]
    ids = with_shots(_session(4, HEAD, 900), 2) + [7, 8, 9]
    pool, saved = _setup(FakeResident(r1), view=r1)
    try:
        with flag("on"):
            need, why = need_img(ids, [shot(1), shot(2)], max_new=4_000)
            margin = A.SIDE_MARGIN + A.PLAN_MARGIN
            assert need == len(ids) - 2 + 2 * 1_134 + 4_000 + margin, (need, why)
            assert "images" not in why and not why.get("wait"), why
    finally:
        _restore(saved)
    return f"need {need}"


if __name__ == "__main__":
    n = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            out = fn()
            print(f"ok {name}" + (f"  {out}" if out else ""), flush=True)
            n += 1
    print(f"{n} passed")
