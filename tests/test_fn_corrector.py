"""SPD-249: the corrector of Flash-Next's coupled MTP drafts (engine/models/flashnext/corrector.py).

CPU:
  * knob off: no corrector, a keyed row's parameters are the plain coupled pick's bit for bit (the flag slot 0), the
    round's cut inputs the plain coupled round's; the keys (the target's keyed noise) are the same with the knob on or off;
  * knob on: the flag is set only for requests sampled as the file was trained for (T, top-k, top-p, no min-p);
    a flagged round's cut reads the file's calibration, an unflagged one the coupled calibration file's;
  * knob on and no file: one log line, no corrector, every keyed row keeps the plain coupled parameters (the flag slot 0);
  * knob on and a file the kernel cannot take (truncated, a tensor of another shape than the metadata's or the model's,
    no metadata, kc or hh not a power of 2, a calibration row that is not 10 bins) or one trained at another tau: one
    "unusable" line, no corrector, the flag 0, and `status` says why; a good file passes `check` and loads;
  * the auto path is ops/correctors/<model>/<profile>.corrector.safetensors, the profile as the price files resolve it,
    and the Flash-Next profile's comment names that path;
  * `status` (the build's info, /health weights.corrector): loaded or not and why, the file's name and sha256, its sizes
    and trained sampling, and which keyed heads its table covers (`serving.corrector_heads`); no paths;
  * an untrained corrector (w2 0, beta 1, kappa high) picks what the plain coupled pick picks over the top kc.
GPU (skipped without CUDA and Triton):
  * `_corr` equals the torch reference (`scores_ref`, the noise from `torch.rand(V)` after `manual_seed(key)` at the
    candidates) token for token, both statistics, over random rows, parameters and corrector weights;
  * rows whose flag or `on` is 0 keep the plain coupled pair bit for bit; `pick` without `x` (the knob-off path) is the plain coupled pick alone;
  * flat scores pick the argmax of the target's noise at the top kc (the noise is the target's);
  * the pick and the row part captured in a CUDA graph replay what the eager launch writes.
"""

from __future__ import annotations

import json
import math
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402

from engine.models.flashnext import coupled as CP  # noqa: E402
from engine.models.flashnext import corrector as CR  # noqa: E402
from engine.sample import Sampler, _key  # noqa: E402


def _skip(why: str) -> Exception:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from scriptmode import Skip
    return Skip(why)


def _gpu():
    if not torch.cuda.is_available() or not CR.HAVE_TRITON:
        raise _skip("no CUDA/Triton")
    return torch.device("cuda")


def _corrector(dev="cpu", kc=8, hh=32, H=64, nhot=4096, seed=0, w2=0.5, cal=None, stat="q"):
    g = torch.Generator().manual_seed(seed)
    cp = torch.zeros(4, 2 + kc)
    cp[:, 0] = 1.0 + 0.1 * torch.randn(4, generator=g)
    cp[:, 1] = 3.0
    cp[:, 2:] = 0.3 * torch.randn(4, kc, generator=g)
    t = {"A": torch.randn(hh, H, generator=g) / math.sqrt(H), "a": 0.1 * torch.randn(4, hh, generator=g),
         "T": torch.randn(nhot, hh, generator=g), "C": 0.2 * torch.randn(CR.NF, hh, generator=g),
         "w2": w2 * torch.randn(hh, generator=g), "cp": cp}
    meta = {"kc": kc, "hh": hh, "D": 4, "tau": 0.85, "sampling": {"temperature": 1.0, "top_k": 20, "top_p": 0.95},
            "stat": stat, "cal": cal or {}}
    return CR.Corrector(t, meta, "test", device=dev)


class _On:
    """The knobs of coupled.py and corrector.py for a test, and the served corrector."""

    def __init__(self, corr=None, **kw):
        self.kw = {"ON": True, **kw}
        self.corr = corr

    def __enter__(self):
        self.old = {k: getattr(CP, k) for k in self.kw}
        for k, v in self.kw.items():
            setattr(CP, k, v)
        self.old_cr = (CR.ON, dict(CR.Corrector._cache))
        CR.ON = self.corr is not None
        CR.Corrector._cache.clear()
        if self.corr is not None:
            CR.Corrector._cache["c"] = self.corr

    def __exit__(self, *exc):
        for k, v in self.old.items():
            setattr(CP, k, v)
        CR.ON = self.old_cr[0]
        CR.Corrector._cache.clear()
        CR.Corrector._cache.update(self.old_cr[1])


def test_knob_off_is_the_plain_coupled_pick():
    sm = Sampler(temperature=1.0, top_k=20, top_p=0.95, seed=1000)
    with _On(TAU=0.85, FILTER=True, SIB=True, SIB_MARGIN=1.3):
        assert CR.get() is None
        keys, par = CP.keyed_for(sm, 717, 4)
    assert keys == [_key(1000, 717 + d) for d in range(4)]
    assert par == [0.85, 20.0, 0.95, 1.3, 1.0, 0.0, 0.0, 0.0]
    return "no corrector; the plain coupled parameters bit for bit (flag 0); keys _key(seed, start + d)"


def test_knob_on_without_a_file_keeps_the_plain_coupled_pick():
    import contextlib
    import io
    sm = Sampler(temperature=1.0, top_k=20, top_p=0.95, seed=1000)
    old = (CR.ON, CR.FILE, dict(CR.Corrector._cache))
    out = io.StringIO()
    try:
        CR.ON, CR.FILE = True, os.path.join(os.path.dirname(os.path.abspath(__file__)), "no-such-corrector.safetensors")
        CR.Corrector._cache.clear()
        with contextlib.redirect_stdout(out), _On(TAU=0.85, FILTER=True, SIB=True, SIB_MARGIN=1.3):
            CR.ON = True                                   # _On sets ON from its corrector (none): on again
            CR.Corrector._cache.clear()
            assert CR.get() is None and CR.get() is None
            assert CR.flag_for(sm) == 0.0
            keys, par = CP.keyed_for(sm, 717, 4)
    finally:
        CR.ON, CR.FILE = old[0], old[1]
        CR.Corrector._cache.clear()
        CR.Corrector._cache.update(old[2])
    lines = [ln for ln in out.getvalue().splitlines() if ln.startswith("[corrector]")]
    assert len(lines) == 1 and "no file" in lines[0], lines
    assert keys == [_key(1000, 717 + d) for d in range(4)]
    assert par == [0.85, 20.0, 0.95, 1.3, 1.0, 0.0, 0.0, 0.0]
    return "knob on, no file: one log line, no corrector, the plain coupled parameters bit for bit (flag 0)"


def _write(path, t, meta, key="corrector"):
    from safetensors.torch import save_file
    save_file({k: v.contiguous() for k, v in t.items()}, path, metadata={key: json.dumps(meta)} if meta is not None else None)


def _file_parts(kc=8, hh=32, H=64, nhot=512):
    c = _corrector(kc=kc, hh=hh, H=H, nhot=nhot)
    t = {"A": c.A.float(), "a": c.a, "T": c.T.float(), "C": c.C, "w2": c.w2, "cp": c.cp}
    tab = {"spine": [[0.5] * 10] * 4, "sib": [[0.3] * 10]}
    return t, dict(c.meta, cal={str(nhot): tab})


def _get_from(path, hidden=None, tau=0.85):
    """CR.get() with the knob on, FN_MTP_CORRECTOR_FILE = `path` and FN_MTP_COUPLED_TAU = `tau`: (corrector, status,
    the [corrector] lines)."""
    import contextlib
    import io
    old = (CR.ON, CR.FILE, dict(CR.Corrector._cache), CP.TAU)
    out = io.StringIO()
    try:
        CR.ON, CR.FILE, CP.TAU = True, path, tau
        CR.Corrector._cache.clear()
        with contextlib.redirect_stdout(out):
            c = CR.get(hidden=hidden)
            assert CR.get() is c                               # decided once
            st = CR.status()
            flag = CR.flag_for(Sampler(temperature=1.0, top_k=20, top_p=0.95, seed=1000))
    finally:
        CR.ON, CR.FILE, CP.TAU = old[0], old[1], old[3]
        CR.Corrector._cache.clear()
        CR.Corrector._cache.update(old[2])
    return c, st, flag, [ln for ln in out.getvalue().splitlines() if ln.startswith("[corrector]")]


def test_a_file_the_kernel_cannot_take_keeps_the_plain_coupled_pick():
    import tempfile
    d = tempfile.mkdtemp(prefix="corrector-")
    t, meta = _file_parts()
    good = os.path.join(d, "good.safetensors")
    _write(good, t, meta)
    blob = open(good, "rb").read()
    cases = {}
    cases["truncated"] = os.path.join(d, "truncated.safetensors")
    open(cases["truncated"], "wb").write(blob[:len(blob) // 2])
    bad_cp = dict(t, cp=torch.zeros(4, 9))
    cases["cp of another kc"] = os.path.join(d, "cp.safetensors")
    _write(cases["cp of another kc"], bad_cp, meta)
    cases["no T"] = os.path.join(d, "noT.safetensors")
    _write(cases["no T"], {k: v for k, v in t.items() if k != "T"}, meta)
    cases["no metadata"] = os.path.join(d, "nometa.safetensors")
    _write(cases["no metadata"], t, None)
    cases["kc 6"] = os.path.join(d, "kc6.safetensors")
    _write(cases["kc 6"], dict(t, cp=torch.zeros(4, 8)), dict(meta, kc=6))
    cases["a short calibration row"] = os.path.join(d, "cal.safetensors")
    _write(cases["a short calibration row"], t, dict(meta, cal={"512": {"spine": [[0.5] * 9]}}))
    cases["a value that is not finite"] = os.path.join(d, "nan.safetensors")
    _write(cases["a value that is not finite"], dict(t, w2=torch.full_like(t["w2"], float("nan"))), meta)
    for why, path in cases.items():
        c, st, flag, lines = _get_from(path)
        assert c is None and flag == 0.0, why
        assert len(lines) == 1 and "unusable" in lines[0] and "stay plain" in lines[0], (why, lines)
        assert st["loaded"] is False and st["why"].startswith("unusable"), (why, st)
    c, st, flag, lines = _get_from(good, hidden=128)          # A is [32, 64]: the model's MTP output is 128 wide
    assert c is None and "A [32, 64]" in st["why"], st
    c, st, flag, lines = _get_from(good, tau=0.9)             # trained at 0.85
    assert c is None and "tau" in st["why"], st
    c, st, flag, lines = _get_from(os.path.join(d, "absent.safetensors"))
    assert c is None and st == {"loaded": False, "why": "no file", "file": "absent.safetensors", "sha256": None}, st
    # the good file passes the check and loads (CPU here; the server loads it on the device)
    good_c = CR.Corrector.load(good, device="cpu", hidden=64)
    assert good_c.kc == 8 and good_c.hh == 32 and good_c.T.shape[0] == 512
    assert good_c.calibration(512) is not None
    first_run = os.path.join(d, "first-run.safetensors")      # a first-run file: its JSON is the only entry
    _write(first_run, t, meta, key="first")
    assert CR.Corrector.load(first_run, device="cpu", hidden=64).kc == 8
    return f"{len(cases) + 3} bad or missing files: one line, no corrector, flag 0, status why; a good one loads"


def test_the_auto_path():
    from engine.models.flashnext.serving import ROOT
    want = os.path.join(ROOT, "ops", "correctors", "qwen3.8-flash-next", "flashnext-3.05.corrector.safetensors")
    assert CR.default_path("flashnext-3.05") == want
    old = os.environ.pop("QSE_WEIGHTS", None)
    try:
        assert CR.default_path(pack="/weights/3.05/") == want                  # flashnext-<pack dir>, as the prices
        os.environ["QSE_WEIGHTS"] = "flashnext-4.05"
        assert CR.default_path(pack="/weights/3.05") == want.replace("3.05", "4.05")
    finally:
        os.environ.pop("QSE_WEIGHTS", None)
        if old is not None:
            os.environ["QSE_WEIGHTS"] = old
    prof = open(os.path.join(ROOT, "ops", "profiles", "_flashnext.env")).read()
    assert "QWEN38_FN_MTP_CORRECTOR=1" in prof
    assert os.path.relpath(want, ROOT) in prof, "the profile's comment names another file"
    return os.path.relpath(want, ROOT)


def test_the_status_names_the_heads_the_table_covers():
    from engine.models.flashnext.serving import corrector_heads
    ns = types.SimpleNamespace
    served = ns(coupled_head="c", coupled_ids=torch.arange(65536), lone_ids=torch.arange(32768), draft_head="d",
                draft_ids=torch.arange(32768))
    assert corrector_heads(served, 65536) == {65536: True, 32768: True}
    assert corrector_heads(served, 32768) == {65536: False, 32768: True}
    draft_only = ns(coupled_head=None, coupled_ids=None, lone_ids=None, draft_head="d", draft_ids=torch.arange(32768))
    assert corrector_heads(draft_only, 65536) == {32768: True}
    full = ns(coupled_head=None, coupled_ids=None, lone_ids=None, draft_head=None, draft_ids=None)
    assert corrector_heads(full, 65536) == {}
    c = _corrector(nhot=65536)
    old = (CR.ON, dict(CR.Corrector._cache))
    try:
        CR.ON = False
        assert CR.status() == {"loaded": False, "why": "QWEN38_FN_MTP_CORRECTOR is off"}
        CR.ON = True
        CR.Corrector._cache.clear()
        CR.Corrector._cache.update({"c": c, "why": None, "file": ("f.corrector.safetensors", "ab" * 32)})
        st = CR.status(corrector_heads(served, int(c.T.shape[0])))
    finally:
        CR.ON = old[0]
        CR.Corrector._cache.clear()
        CR.Corrector._cache.update(old[1])
    assert st["loaded"] and st["why"] is None and st["file"] == "f.corrector.safetensors" and st["sha256"] == "ab" * 32
    assert (st["kc"], st["hh"], st["D"], st["hot_rows"], st["tau"]) == (8, 32, 4, 65536, 0.85)
    assert st["sampling"] == {"temperature": 1.0, "top_k": 20, "top_p": 0.95}
    assert st["heads"] == {"65536": True, "32768": True}
    assert "/" not in json.dumps(st)                                   # no paths
    return "heads covered by the table's hot rows; status: loaded, file, sha256, sizes, sampling, heads; off: why"


def test_the_flag_follows_the_trained_sampling():
    c = _corrector()
    with _On(corr=c, TAU=0.85, FILTER=True, SIB=True, SIB_MARGIN=1.3):
        k1, p1 = CP.keyed_for(Sampler(temperature=1.0, top_k=20, top_p=0.95, seed=1000), 717, 4)
        _, p2 = CP.keyed_for(Sampler(temperature=0.7, top_k=20, top_p=0.95, seed=1000), 717, 4)
        _, p3 = CP.keyed_for(Sampler(temperature=1.0, top_k=40, top_p=0.95, seed=1000), 717, 4)
        _, p4 = CP.keyed_for(Sampler(temperature=1.0, top_k=20, top_p=0.95, min_p=0.05, seed=1000), 717, 4)
    assert k1 == [_key(1000, 717 + d) for d in range(4)]            # the keyed noise is the same with the knob on
    assert p1[CR.FLAG] == 1.0 and p1[:5] == [0.85, 20.0, 0.95, 1.3, 1.0]
    assert p2[CR.FLAG] == 0.0 and p3[CR.FLAG] == 0.0 and p4[CR.FLAG] == 0.0
    return "flag 1 at T 1 / top-k 20 / top-p 0.95; 0 otherwise; keys unchanged"


def test_a_flagged_rounds_cut_reads_the_files_calibration():
    from engine.methods.staircut import DepthDeciles
    from engine.models.flashnext.mtp import FlashNextMTPHead
    tab = {"spine": [[0.11] * 10, [0.22] * 10, [0.33] * 10, [0.44] * 10], "sib": [[0.5] * 10]}
    c = _corrector(cal={"32768": tab})
    h = FlashNextMTPHead.__new__(FlashNextMTPHead)
    h._khot = 32768
    with _On(corr=c, CAL=False):
        h.keyed = ([1, 2, 3, 4], [0.85, 20.0, 0.95, 1.3, 1.0, 1.0, 0.0, 0.0])
        ci = h.cut_inputs(True, 1)
        assert isinstance(ci["cal"], DepthDeciles) and abs(ci["cal"].q("spine", 0.5, 1) - 0.22) < 1e-9
        h.keyed = ([1, 2, 3, 4], [0.85, 20.0, 0.95, 1.3, 1.0, 0.0, 0.0, 0.0])
        assert h.cut_inputs(True, 1)["cal"] is None          # unflagged: the calibration file's (off here: the cut's own)
        h._khot = 65536
        h.keyed = ([1, 2, 3, 4], [0.85, 20.0, 0.95, 1.3, 1.0, 1.0, 0.0, 0.0])
        assert h.cut_inputs(True, 8)["cal"] is None          # no table for that head in the file
    return "flagged: the file's table for the head's size; unflagged or no table: the calibration file's"


def _rows_cpu(R, Vh, V, kp, seed):
    g = torch.Generator().manual_seed(seed)
    lg = torch.randn(R, Vh, generator=g) * 3.0
    hids = torch.randperm(V, generator=g)[:Vh]
    keys = [int(torch.randint(0, 2 ** 62, (1,), generator=g)) for _ in range(R)]
    v, i = torch.topk(lg, kp, sorted=True)
    return lg, hids, keys, v, i


def test_an_untrained_corrector_is_the_plain_coupled_pick_over_the_top_k():
    R, Vh, V, kc = 64, 4096, 9000, 8
    lg, hids, keys, v, i = _rows_cpu(R, Vh, V, 32, 3)
    c = _corrector(kc=kc, w2=0.0)
    p = {"A": c.A, "a": c.a, "T": c.T, "C": c.C, "w2": c.w2, "cp": c.cp.clone()}
    p["cp"][:, 0], p["cp"][:, 1], p["cp"][:, 2:] = 1.0, 30.0, 0.0
    keep = CR.keep_mask(v, torch.full((R,), 0.85), 20, 0.95)
    u = torch.rand(R, kc)
    g = -torch.log(-torch.log(u.clamp_min(1e-20)))
    s = CR.scores_ref(p, torch.randn(R, 64), v, i, torch.logsumexp(lg, 1), keep, torch.full((R,), 0.85),
                      torch.zeros(R, dtype=torch.long), kc)
    plain = (v[:, :kc] / 0.85).masked_fill(~keep[:, :kc], float("-inf"))
    assert torch.equal(torch.argmax(s + g, 1), torch.argmax(plain + g, 1))
    return f"{R} rows: argmax(s + G) = the plain coupled pick's over the top {kc}"


# ---- GPU ----------------------------------------------------------------------------------------------------------

def _gpu_case(dev, R, kc, seed, stat, flags=None, on=None):
    V, Vh = 248320, 4096
    g = torch.Generator().manual_seed(seed)
    lg = (torch.randn(R, Vh, generator=g) * 3.0).to(dev)
    hids = torch.randperm(V, generator=g)[:Vh].to(dev)
    keys = [int(torch.randint(0, 2 ** 62, (1,), generator=g)) for _ in range(R)]
    shapes = [(0.85, 20, 0.95, 1.3), (1.0, 0, 1.0, -1.0), (0.7, 5, 0.5, 9.0), (0.85, 48, 0.99, 0.5)]
    par = torch.tensor([list(shapes[r % 4]) + [1.0 if on is None else on[r], 1.0 if flags is None else flags[r],
                                               0.0, 0.0] for r in range(R)], dtype=torch.float32, device=dev)
    x = torch.randn(R, 64, generator=g).to(dev, torch.bfloat16)
    c = _corrector(dev, kc=kc, nhot=Vh, seed=seed, stat=stat)
    return V, lg, hids, keys, par, x, c


def _coupled_then_corr(c, V, lg, hids, keys, par, x, depth, use_x=True):
    dev = lg.device
    R = lg.shape[0]
    z = torch.exp(lg - lg.amax(-1, keepdim=True)).sum(-1, keepdim=True)
    tok = torch.full((R, 2), -7, dtype=torch.long, device=dev)
    prob = torch.full((R, 2), -7.0, device=dev)
    kt = torch.tensor(keys, dtype=torch.int64, device=dev)
    with _On(corr=c):
        CP.pick(lg, hids, kt, par, z, tok, prob, V, x=x if use_x else None, depth=depth)
    torch.cuda.synchronize()
    return tok.cpu(), prob.cpu()


def test_the_kernel_is_the_reference():
    dev = _gpu()
    n = 0
    for kc in (4, 8):
        for stat in ("q", "c"):
            for seed in range(6):
                R = 16
                V, lg, hids, keys, par, x, c = _gpu_case(dev, R, kc, seed, stat)
                depth = seed % 4
                tok, prob = _coupled_then_corr(c, V, lg, hids, keys, par, x, depth)
                v, i = torch.topk(lg, CP.KP, sorted=True)
                lse = torch.logsumexp(lg, 1)
                tau = par[:, 0]
                p = {"A": c.A, "a": c.a, "T": c.T, "C": c.C, "w2": c.w2, "cp": c.cp}
                for r in range(R):
                    keep = CR.keep_mask(v[r:r + 1], tau[r:r + 1], int(par[r, 1]), float(par[r, 2]))
                    s = CR.scores_ref(p, x[r:r + 1], v[r:r + 1], i[r:r + 1], lse[r:r + 1], keep, tau[r:r + 1],
                                      torch.tensor([depth], device=dev), kc)[0]
                    gen = torch.Generator(device=dev)
                    gen.manual_seed(keys[r])
                    u = torch.rand(V, generator=gen, device=dev)
                    cand = hids[i[r, :kc]]
                    sc = s + (-torch.log(-torch.log(u[cand].clamp_min(1e-20))))
                    t2 = torch.topk(sc, 2)
                    assert int(tok[r, 0]) == int(cand[t2.indices[0]]), (kc, stat, seed, r)
                    if stat == "c":
                        q = torch.softmax(s, 0)
                        want = float(q[t2.indices[0]])
                    else:
                        want = float(torch.exp(v[r, t2.indices[0]] - lse[r]))
                    assert abs(float(prob[r, 0]) - want) < 2e-4 * max(1.0, want) + 2e-5, (kc, stat, seed, r, prob[r], want)
                    n += 1
    return f"{n} rows: token = argmax(scores_ref + torch.rand noise), statistic q and c"


def test_unflagged_rows_keep_the_plain_coupled_pair_bit_for_bit():
    dev = _gpu()
    R = 12
    flags = [1.0, 0.0] * 6
    on = [1.0] * 10 + [0.0, 0.0]
    V, lg, hids, keys, par, x, c = _gpu_case(dev, R, 8, 11, "q", flags=flags, on=on)
    t_plain, p_plain = _coupled_then_corr(c, V, lg, hids, keys, par, x, 0, use_x=False)     # the knob-off path: the plain pick alone
    t_c, p_c = _coupled_then_corr(c, V, lg, hids, keys, par, x, 0)
    with _On():
        t_off, p_off = _coupled_then_corr(None, V, lg, hids, keys, par, x, 0)       # knob off, x given: the plain pick alone
    assert torch.equal(t_off, t_plain) and torch.equal(p_off.view(torch.int32), p_plain.view(torch.int32))
    for r in range(R):
        if flags[r] == 0.0 or on[r] == 0.0:
            assert torch.equal(t_c[r], t_plain[r]) and torch.equal(p_c[r].view(torch.int32), p_plain[r].view(torch.int32)), r
    assert t_c[10:].tolist() == [[-7, -7]] * 2
    changed = sum(int(not torch.equal(t_c[r], t_plain[r])) for r in range(0, 10, 2))
    return f"flag 0 / on 0 rows = the plain pick bit for bit; knob off with x = the plain pick; {changed} of 5 flagged rows changed"


def test_flat_scores_pick_the_targets_noise():
    dev = _gpu()
    R, kc = 8, 8
    V, lg, hids, keys, par, x, c = _gpu_case(dev, R, kc, 21, "q")
    c.cp.zero_()
    c.w2.zero_()
    par[:, 1], par[:, 2] = 0.0, 1.0                    # no filter: every top-kc candidate kept
    tok, _ = _coupled_then_corr(c, V, lg, hids, keys, par, x, 2)
    v, i = torch.topk(lg, kc)
    for r in range(R):
        gen = torch.Generator(device=dev)
        gen.manual_seed(keys[r])
        u = torch.rand(V, generator=gen, device=dev)
        cand = hids[i[r]]
        assert int(tok[r, 0]) == int(cand[torch.argmax(u[cand])]), r
    return f"{R} rows: flat scores = argmax of torch.rand(V) at the top {kc} (the target's noise)"


def test_a_captured_pick_replays_the_eager_one():
    dev = _gpu()
    R = 8
    V, lg, hids, keys, par, x, c = _gpu_case(dev, R, 8, 31, "c")
    t_e, p_e = _coupled_then_corr(c, V, lg, hids, keys, par, x, 1)
    z = torch.exp(lg - lg.amax(-1, keepdim=True)).sum(-1, keepdim=True)
    tok = torch.full((R, 2), -7, dtype=torch.long, device=dev)
    prob = torch.full((R, 2), -7.0, device=dev)
    kt = torch.tensor(keys, dtype=torch.int64, device=dev)
    with _On(corr=c):
        s_ = torch.cuda.Stream()
        s_.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s_):
            CP.pick(lg, hids, kt, par, z, tok, prob, V, x=x, depth=1)
        torch.cuda.current_stream().wait_stream(s_)
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            CP.pick(lg, hids, kt, par, z, tok, prob, V, x=x, depth=1)
        tok.fill_(-7)
        prob.fill_(-7.0)
        gr.replay()
    torch.cuda.synchronize()
    assert torch.equal(tok.cpu(), t_e) and torch.equal(prob.cpu().view(torch.int32), p_e.view(torch.int32))
    return "graph replay = eager, bit for bit"


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from scriptmode import Skip
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<56} ok   {fn() or ''}")
        except Skip as e:
            print(f"  {name:<56} skip {e}")
        except AssertionError as e:
            fails += 1
            print(f"  {name:<56} FAIL {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
