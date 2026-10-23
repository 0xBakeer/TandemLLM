"""SPD-169: which chunks of a layer-major prefill window share their projections (`_rowwise_run`).
Only a contiguous run of at least two chunks that each take the EXL3 slab path alone (more than
EXL3_CHUNK_UNTIL rows), and only when every projection of the model is EXL3; the knobs are off by
default."""

import types

import torch

from engine import model as M
from engine.settings import KNOBS


class _E:                                              # stands in for an EXL3Block
    pass


def _eng(all_exl3=True, layers=4):
    cfg = types.SimpleNamespace(num_hidden_layers=layers, is_linear=lambda l: (l + 1) % 4 != 0)
    w = types.SimpleNamespace(group=lambda name: _E() if all_exl3 else object(),
                              proj=lambda name: _E())
    return types.SimpleNamespace(cfg=cfg, w=w, trace=None, tree=None, _gv=None)


def _toks(*rows):
    return [torch.zeros(r, dtype=torch.long) for r in rows]


X3 = types.SimpleNamespace(CHUNK_UNTIL=512, EXL3Block=_E)


def test_full_chunks_join_and_partial_ones_keep_their_call():
    run = M.Qwen38Engine._rowwise_run
    assert run(_eng(), _toks(1024, 1024, 1024, 300), X3) == (0, 3)
    assert run(_eng(), _toks(200, 1024, 1024, 904), X3) == (1, 4)     # 904 > 512: the slab path alone
    assert run(_eng(), _toks(1024, 300), X3) is None                   # one chunk is not a run
    assert run(_eng(), _toks(512, 1024), X3) is None                   # 512 rows take the skinny kernel


def test_no_join_without_exl3_or_during_a_verify():
    run = M.Qwen38Engine._rowwise_run
    assert run(_eng(all_exl3=False), _toks(1024, 1024), X3) is None
    e = _eng()
    e.trace = object()
    assert run(e, _toks(1024, 1024), X3) is None
    assert run(_eng(), _toks(1024, 1024), None) is None


def test_knobs_default_off():
    for k in ("WINDOW_ROWWISE", "GDN_PREP", "EXL3_SLAB_TRITON"):
        assert KNOBS[k][0] == "0", k


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            n += 1
    print(f"{n} passed")
