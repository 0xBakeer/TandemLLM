"""ENG-406: the 27B family's builder, drafter stack and side-slot hooks, moved from server/app.py into
engine/models/qwen3_5/serving.py, run on the CPU with stand-ins for the weights, the engine, the tokenizer and
the drafters (the 27B's weights are not needed: what is checked is which parts the builder asks for, with
which arguments, and what it returns to the shared loader).

  1  every `--drafter` builds the stack it built in server/app.py: none, mtp, router, merged (its verify
     table), dflash2 (the depth from the block width), lenrouter (with and without the tree, the narrow arm
     off); the builder asks the server's ctx for the KV pool and the vision tower, in that order
  2  the side-slot hooks: the drafters they are built for, the bytes per token with and without the pool and
     the constant from ctx, a slot's engine and drafter over the same weights reading the lookup's corpus,
     the precapture widths
  3  the registry names this builder for the default family

CPU. Run: python tests/test_qwen3_5_serving.py
"""

from __future__ import annotations

import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from engine.models import default_arch  # noqa: E402
from engine.models.qwen3_5 import serving as S  # noqa: E402


class _Cfg:
    path = "/snap"
    block_size = 16


class _Drafter:
    """Every drafter class the builder may name: records how it was made."""
    made: list = []

    def __init__(self, *args, **kw):
        self.args, self.kw = args, kw
        self.cfg = types.SimpleNamespace(block_size=kw.get("block", 16))
        self.corpus = "corpus"
        _Drafter.made.append((type(self).__name__, args, kw))

    def _build(self):
        self.built = True


class MTPDrafter(_Drafter): pass                 # noqa: E701
class RouterDrafter(_Drafter): pass              # noqa: E701
class DFlash2Drafter(_Drafter): pass             # noqa: E701
class NgramDrafter(_Drafter): pass               # noqa: E701
class MergedRouter(_Drafter): pass               # noqa: E701
class LengthRouter(_Drafter): pass               # noqa: E701


class NoArm(_Drafter):
    def __init__(self, eng, block):
        super().__init__(eng, block=block)


class _Engine:
    def __init__(self, cfg, w, max_len):
        self.cfg, self.w, self.max_len = cfg, w, max_len
        self.captured = None

    def _graphs_for(self, n, c):
        return object()

    @property
    def _graphs(self):
        eng = self

        class G:
            def precapture(self, widths):
                eng.captured = list(widths)
        return G()


class _Weights:
    def __init__(self, path, skip_mtp, nvfp4, fp8_head):
        self.path, self.skip_mtp, self.nvfp4, self.fp8_head = path, skip_mtp, nvfp4, fp8_head
        self.nvfp4_source, self.fp8_head_source = None, None

    def report(self):
        return "stand-in weights"


def _patch():
    """The stand-ins in the modules the builder imports from; returns the originals."""
    import engine.config
    import engine.drafters.dflash2
    import engine.drafters.mtp
    import engine.drafters.ngram
    import engine.lenrouter
    import engine.loader
    import engine.models.qwen3_5.adapter as ad
    import engine.router
    import transformers
    tok = types.SimpleNamespace(from_pretrained=lambda p: f"tok:{p}")
    table = {
        (engine.config, "load_config"): lambda p: _Cfg(),
        (engine.loader, "Weights"): _Weights,
        (ad, "Qwen38Engine"): _Engine,
        (transformers, "AutoTokenizer"): tok,
        (engine.drafters.mtp, "MTPDrafter"): MTPDrafter,
        (engine.router, "RouterDrafter"): RouterDrafter,
        (engine.router, "MergedRouter"): MergedRouter,
        (engine.drafters.dflash2, "DFlash2Drafter"): DFlash2Drafter,
        (engine.drafters.ngram, "NgramDrafter"): NgramDrafter,
        (engine.lenrouter, "LengthRouter"): LengthRouter,
        (engine.lenrouter, "NoArm"): NoArm,
    }
    saved = {k: getattr(*k) for k in table}
    for (mod, name), v in table.items():
        setattr(mod, name, v)
    return saved


def _unpatch(saved):
    for (mod, name), v in saved.items():
        setattr(mod, name, v)


def _args(**kw):
    base = dict(model="/snap", nvfp4=None, fp8_head=None, drafter="none", max_len=4096, depth=4, draft_head=None,
                dflash2_ckpt="ck8", dflash2_ckpt16="ck16", dflash2_path=None, dflash2_blocks=1, df2_temp=0.0,
                corpus="", budget=17, narrow_arm="on", tree=False, sampled_tree="det", relax_tau=1.0, relax_rank=1,
                k=None, len_fixed=None, len_explore=0, len_latch=False, drop_idle=False, deep=0, deep_after=2,
                learn_cost=False, parallel=2, side_graphs="on", side_precapture="on")
    base.update(kw)
    return types.SimpleNamespace(**base)


def _ctx(order):
    return types.SimpleNamespace(
        keys=lambda path, paths: order.append(("keys", path)), mark=lambda st: order.append(st),
        kv_pool=lambda: order.append("kv_pool"), vision=lambda path: order.append(("vision", path)),
        state={"tokenizer_sha": "sha"}, prices={"tree_ms": None}, side_const_gb=lambda: 2.0)


def _build(**kw):
    saved = _patch()
    order = []
    _Drafter.made = []
    try:
        import engine.router as R
        tree = (R.served_tree_table, R.tree_nodes)
        R.served_tree_table = lambda t: {8: 10.0, 16: 18.0}
        R.tree_nodes = lambda b: b + 4
        try:
            a = _args(**kw)
            m = S.serve(a, _ctx(order))
        finally:
            R.served_tree_table, R.tree_nodes = tree
    finally:
        _unpatch(saved)
    return a, m, order, list(_Drafter.made)


def test_the_builder_asks_the_server_in_order_and_returns_the_parts():
    a, m, order, made = _build()
    assert order == [("keys", "/snap"), "weights", "kv_pool", "engine", ("vision", "/snap"), "vision", "drafters"]
    assert m["drafter"] is None and m["tok"] == "tok:/snap" and isinstance(m["engine"], _Engine)
    assert m["w"].skip_mtp is True and m["serving"]["tree"] is False and m["serving"]["sampled_tree"] == "det"
    assert set(m["side_slots"]) == {"check", "make", "bytes", "precapture"} and "drafter=none" in m["describe"]


def test_every_drafter_builds_its_stack():
    _, m, _, made = _build(drafter="mtp")
    assert type(m["drafter"]).__name__ == "MTPDrafter" and m["w"].skip_mtp is False
    _, m, _, made = _build(drafter="router")
    assert type(m["drafter"]).__name__ == "RouterDrafter"
    a, m, _, made = _build(drafter="dflash2", dflash2_blocks=2)
    assert type(m["drafter"]).__name__ == "DFlash2Drafter" and a.depth == 30 and m["drafter"].built
    a, m, _, made = _build(drafter="merged", nvfp4="nv")
    assert type(m["drafter"]).__name__ == "MergedRouter" and a.depth == 16
    ng = [x for x in made if x[0] == "NgramDrafter"][0][2]
    from engine.router import VERIFY_MS_NVFP4
    assert ng["verify_base_ms"] == VERIFY_MS_NVFP4[min(VERIFY_MS_NVFP4)] and ng["tokenizer_sha"] == "sha"


def test_the_length_router_with_and_without_the_tree():
    a, m, _, made = _build(drafter="lenrouter")
    assert type(m["drafter"]).__name__ == "LengthRouter" and a.depth == 15
    assert [x[2].get("block") for x in made if x[0] == "DFlash2Drafter"] == [8, 16]
    a, m, _, made = _build(drafter="lenrouter", tree=True, deep=24)
    assert type(m["drafter"]).__name__ == "LengthRouter" and a.depth == 23 and m["drafter"].kw["tree"] is True
    assert [x[0] for x in made].count("NgramDrafter") == 1, "one lookup drafter for both arms"
    a, m, _, made = _build(drafter="lenrouter", narrow_arm="off")
    assert any(x[0] == "NoArm" for x in made)
    try:
        _build(drafter="lenrouter", dflash2_ckpt16=None)
    except SystemExit as exc:
        assert "dflash2-ckpt16" in str(exc)
    else:
        raise AssertionError("lenrouter without the wide checkpoint must stop")
    try:
        _build(drafter="mtp", narrow_arm="off")
    except SystemExit as exc:
        assert "--narrow-arm off" in str(exc)
    else:
        raise AssertionError("--narrow-arm off is the length router's")


def test_the_side_slot_hooks():
    saved = _patch()
    try:
        a = _args(drafter="lenrouter")
        ctx = _ctx([])
        h = S.side_slots(a, ctx, _Cfg(), "w")
        h["check"]()
        assert h["bytes"](4096, True) == (256, 2 << 30) and h["bytes"](4096, False) == (106_752, 2 << 30)
        import engine.router as R
        tree = (R.served_tree_table, R.tree_nodes)
        R.served_tree_table, R.tree_nodes = (lambda t: {8: 10.0, 16: 18.0}), (lambda b: b + 4)
        try:
            e2, d2 = h["make"](2048, types.SimpleNamespace(ngram=types.SimpleNamespace(corpus="C")))
        finally:
            R.served_tree_table, R.tree_nodes = tree
        assert e2.max_len == 2048 and e2.w == "w" and e2.graphs_off is False
        assert type(d2).__name__ == "LengthRouter"
        h["precapture"](e2, d2)
        assert e2.captured == list(range(2, 17))
        h2 = S.side_slots(_args(drafter="mtp"), ctx, _Cfg(), "w")
        try:
            h2["check"]()
        except SystemExit as exc:
            assert "built for --drafter lenrouter" in str(exc)
        else:
            raise AssertionError("side slots are built for the length router or none")
        e3, d3 = S.side_slots(_args(drafter="none", side_graphs="off"), ctx, _Cfg(), "w")["make"](1024, None)
        assert d3 is None and e3.graphs_off is True
    finally:
        _unpatch(saved)


def test_the_default_family_names_this_builder():
    assert default_arch().serve == "engine.models.qwen3_5.serving:serve"
    assert default_arch().serve_builder() is S.serve


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
