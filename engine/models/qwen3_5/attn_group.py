"""SPD-245: the attention of every lane of a graphed joint pass in one launch a layer (QWEN38_JOINT_ATTN_GROUP).

In a joint verify pass each lane's attention layer ran on its own: the KV scatter at the lane's static
slots (two launches), the split-KV kernel and its combine (tools/attn_kernels.py `decode_attention_dev`),
so an 8-lane pass issued about 128 small launches a layer, each a few programs on 48 SMs. Here the
lanes' blocks are one request of the joint pass (`"attn"`, engine/model.py `_attention_steps`): one
scatter launch, one split launch over every lane's row groups and one combine, the lanes told apart by
a table of pointers and lengths on the device.

The arithmetic is `_attn_split` / `_attn_combine`'s, tile for tile: a row's output does not depend on
which other rows share the launch (the shared kernels' row independence), a split past a lane's length
runs no iteration and the combine weighs it exactly zero, so each row keeps the bits the lone launch
gave it. The table is written once a graph at capture (every tensor it names is static for that graph)
and copied from pinned memory as the graph's own node.
"""

from __future__ import annotations

import math

import torch

from engine.settings import SETTINGS as _S

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:                                            # pragma: no cover
    HAVE_TRITON = False

#: on (1): the lanes' attention grouped where the joint pass is graphed; 0: a launch a lane, as before
ON = _S.get("JOINT_ATTN_GROUP") == "1"
#: table fields (int64): pointers and lengths, one row a lane
NF = 16
(F_Q, F_KN, F_VN, F_KC, F_VC, F_LENP, F_BMASK, F_SLOTS, F_NS, F_CHUNK, F_SKCH, F_SVCH, F_SKNH, F_SVNH,
 F_SKNT, F_SVNT) = range(16)
STATS = {"grouped": 0, "lone": 0}


if HAVE_TRITON:

    @triton.jit
    def _kv_scatter_grp(TAB, T, NKV, NF: tl.constexpr, D: tl.constexpr):
        """Lane e's new keys and values into its cache at its static slots: one program a (lane, position,
        kv head)."""
        e = tl.program_id(0)
        t = tl.program_id(1)
        h = tl.program_id(2)
        row = TAB + e * NF
        kn = tl.load(row + 1).to(tl.pointer_type(tl.bfloat16))
        vn = tl.load(row + 2).to(tl.pointer_type(tl.bfloat16))
        kc = tl.load(row + 3).to(tl.pointer_type(tl.bfloat16))
        vc = tl.load(row + 4).to(tl.pointer_type(tl.bfloat16))
        slots = tl.load(row + 7).to(tl.pointer_type(tl.int64))
        s_kch = tl.load(row + 10)
        s_vch = tl.load(row + 11)
        s_knh = tl.load(row + 12)
        s_vnh = tl.load(row + 13)
        s_knt = tl.load(row + 14)
        s_vnt = tl.load(row + 15)
        slot = tl.load(slots + t)
        od = tl.arange(0, D)
        k = tl.load(kn + h * s_knh + t * s_knt + od)
        v = tl.load(vn + h * s_vnh + t * s_vnt + od)
        tl.store(kc + h * s_kch + slot * D + od, k)
        tl.store(vc + h * s_vch + slot * D + od, v)

    @triton.jit
    def _attn_split_grp(TAB, PM, PL, PACC, T, R, NKV, GROUPS, NSMAX, NF: tl.constexpr,
                        SCALE: tl.constexpr, REP: tl.constexpr, D: tl.constexpr,
                        BM: tl.constexpr, BN: tl.constexpr):
        """`tools.attn_kernels._attn_split` (DEVLEN, not DIRECT) over the lanes of the table: program
        (lane * GROUPS + row group, kv head, split)."""
        pid = tl.program_id(0)
        e = pid // GROUPS
        rg = pid % GROUPS
        kvh = tl.program_id(1)
        sp = tl.program_id(2)
        row = TAB + e * NF
        Q = tl.load(row + 0).to(tl.pointer_type(tl.bfloat16))
        K = tl.load(row + 3).to(tl.pointer_type(tl.bfloat16))
        V = tl.load(row + 4).to(tl.pointer_type(tl.bfloat16))
        LENP = tl.load(row + 5).to(tl.pointer_type(tl.int32))
        BMASK = tl.load(row + 6).to(tl.pointer_type(tl.int8))
        CHUNK = tl.load(row + 9)
        s_kh = tl.load(row + 10)
        s_vh = tl.load(row + 11)
        START = tl.load(LENP)
        LC = tl.load(LENP + 1)
        s_qh = T * D
        s_qt = D
        rows = rg * BM + tl.arange(0, BM)                    # flattened (group, position)
        rmask = rows < R
        g = rows // T
        t = rows % T
        qh = kvh * REP + g
        od = tl.arange(0, D)
        q = tl.load(Q + qh[:, None] * s_qh + t[:, None] * s_qt + od[None, :],
                    mask=rmask[:, None], other=0.0)
        c0 = sp * CHUNK
        c1 = tl.minimum(c0 + CHUNK, LC)
        m_i = tl.full([BM], -float("inf"), dtype=tl.float32)
        l_i = tl.zeros([BM], dtype=tl.float32)
        acc = tl.zeros([BM, D], dtype=tl.float32)
        for n0 in range(c0, c1, BN):
            cols = n0 + tl.arange(0, BN)
            cmask = cols < c1
            k = tl.load(K + kvh * s_kh + cols[:, None] * D + od[None, :],
                        mask=cmask[:, None], other=0.0)
            s = tl.dot(q, tl.trans(k.to(tl.bfloat16)))
            s = s * SCALE
            inblk = cols - START
            bm = tl.load(BMASK + t[:, None] * T + inblk[None, :],
                         mask=rmask[:, None] & (inblk[None, :] >= 0) & cmask[None, :], other=0)
            vis = (cols[None, :] < START) | (bm != 0)
            vis = vis & cmask[None, :] & rmask[:, None]
            s = tl.where(vis, s, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
            p = tl.exp(s - m_safe[:, None])
            alpha = tl.exp(m_i - m_safe)
            l_i = l_i * alpha + tl.sum(p, 1)
            v = tl.load(V + kvh * s_vh + cols[:, None] * D + od[None, :],
                        mask=cmask[:, None], other=0.0)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v.to(tl.bfloat16))
            m_i = m_new
        base = (e * NSMAX + sp) * (NKV * R) + kvh * R + rows
        tl.store(PM + base, m_i, mask=rmask)
        tl.store(PL + base, l_i, mask=rmask)
        tl.store(PACC + base[:, None] * D + od[None, :], acc, mask=rmask[:, None])

    @triton.jit
    def _attn_combine_grp(TAB, PM, PL, PACC, OUT, T, R, NKV, NSMAX, s_oh, s_ot, s_oe, NF: tl.constexpr,
                          REP: tl.constexpr, D: tl.constexpr, NSP: tl.constexpr):
        """`tools.attn_kernels._attn_combine` a lane: program (lane * NKV + kv head, row)."""
        pid = tl.program_id(0)
        e = pid // NKV
        kvh = pid % NKV
        r = tl.program_id(1)
        NS = tl.load(TAB + e * NF + 8)
        g = r // T
        t = r % T
        qh = kvh * REP + g
        stride = NKV * R
        sps = tl.arange(0, NSP)
        smask = sps < NS
        idx = (e * NSMAX + sps) * stride + kvh * R + r
        m = tl.load(PM + idx, mask=smask, other=-float("inf"))
        l = tl.load(PL + idx, mask=smask, other=0.0)
        mx = tl.max(m, 0)
        mx = tl.where(mx == -float("inf"), 0.0, mx)
        w = tl.where(smask, tl.exp(m - mx), 0.0)
        od = tl.arange(0, D)
        acc = tl.load(PACC + idx[:, None] * D + od[None, :], mask=smask[:, None], other=0.0)
        num = tl.sum(acc * w[:, None], 0)
        den = tl.sum(l * w, 0)
        o = num / tl.where(den == 0.0, 1.0, den)
        tl.store(OUT + e * s_oe + t * s_ot + qh * s_oh + od, o.to(tl.bfloat16))


class _Table:
    """One layer's table of a lane set: pinned host rows and their device copy."""

    __slots__ = ("host", "dev", "np")

    def __init__(self, E: int, device):
        self.host = torch.zeros(E, NF, dtype=torch.int64).pin_memory()
        self.np = self.host.numpy()
        self.dev = torch.zeros(E, NF, dtype=torch.int64, device=device)


_TABLES: dict = {}


def _table(engs: list, layer: int, device) -> _Table:
    key = (tuple(id(e) for e in engs), layer)
    t = _TABLES.get(key)
    if t is None:
        if len(_TABLES) > 4096:
            _TABLES.clear()
        t = _TABLES[key] = _Table(len(engs), device)
    return t


def run_attn(name: str, payloads: list) -> list:
    """The `"attn"` requests of a joint pass: per lane (engine, q [1, Hq, T, D], k, v [1, Hkv, T, D] after the
    norms and the rotary), the engine graphed (`eng._jg`). Returns each lane's o [1, Hq, T, D]."""
    from tools.attn_kernels import STAGES3, decode_attention_dev, pick_launch
    layer = int(name)
    engs = [p[0] for p in payloads]
    if len(payloads) == 1 or not ON or not HAVE_TRITON:
        STATS["lone"] += len(payloads)
        out = []
        for eng, q, k, v in payloads:
            jg = eng._jg
            i = eng.kv.slot[layer]
            eng.kv.k[i].index_copy_(2, jg.slots, k.to(eng.kv.k.dtype))
            eng.kv.v[i].index_copy_(2, jg.slots, v.to(eng.kv.v.dtype))
            out.append(decode_attention_dev(q, eng.kv.k[i], eng.kv.v[i], jg.lenp, eng.tree.anc_incl, jg.max_lc))
        return out
    eng0, q0, k0, v0 = payloads[0]
    E = len(payloads)
    _, hq, T, D = q0.shape
    hkv = k0.shape[1]
    rep = hq // hkv
    dev = q0.device
    scale = 1.0 / math.sqrt(D)
    bm, groups, _, _ = pick_launch(T, rep, 1024)
    R = rep * T
    tab = _table(engs, layer, dev)
    ns_max = 2
    for e, (eng, q, k, v) in enumerate(payloads):
        jg = eng._jg
        i = eng.kv.slot[layer]
        kc, vc = eng.kv.k[i], eng.kv.v[i]
        if not q.is_contiguous():
            q = q.contiguous()
        assert k.stride(3) == 1 and v.stride(3) == 1 and k.dtype == v.dtype == torch.bfloat16, "attn group: k/v rows"
        assert kc.dtype == torch.bfloat16 and kc.stride(3) == 1 and kc.stride(2) == D, "attn group: the cache rows"
        _, _, ns, chunk = pick_launch(T, rep, jg.max_lc)
        ns = max(ns, 2)
        ns_max = max(ns_max, ns)
        mask = eng.tree.anc_incl
        assert mask.dtype == torch.bool and mask.is_contiguous() and mask.shape == (T, T)
        row = tab.np[e]
        row[F_Q] = q.data_ptr()
        row[F_KN] = k.data_ptr()
        row[F_VN] = v.data_ptr()
        row[F_KC] = kc.data_ptr()
        row[F_VC] = vc.data_ptr()
        row[F_LENP] = jg.lenp.data_ptr()
        row[F_BMASK] = mask.data_ptr()
        row[F_SLOTS] = jg.slots.data_ptr()
        row[F_NS] = ns
        row[F_CHUNK] = chunk
        row[F_SKCH] = kc.stride(1)
        row[F_SVCH] = vc.stride(1)
        row[F_SKNH] = k.stride(1)
        row[F_SVNH] = v.stride(1)
        row[F_SKNT] = k.stride(2)
        row[F_SVNT] = v.stride(2)
    tab.dev.copy_(tab.host, non_blocking=True)
    _kv_scatter_grp[(E, T, hkv)](tab.dev, T, hkv, NF=NF, D=D, num_warps=1)
    out = torch.empty(E, T, hq, D, dtype=torch.bfloat16, device=dev)
    pm = torch.empty(E * ns_max * R * hkv, dtype=torch.float32, device=dev)
    pl = torch.empty_like(pm)
    pacc = torch.empty(E * ns_max * R * hkv, D, dtype=torch.float32, device=dev)
    _attn_split_grp[(E * groups, hkv, ns_max)](
        tab.dev, pm, pl, pacc, T, R, hkv, groups, ns_max, NF=NF,
        SCALE=scale, REP=rep, D=D, BM=bm, BN=32, num_warps=4, num_stages=3 if STAGES3 else 2)
    _attn_combine_grp[(E * hkv, R)](tab.dev, pm, pl, pacc, out, T, R, hkv, ns_max,
                                    out.stride(2), out.stride(1), out.stride(0), NF=NF,
                                    REP=rep, D=D, NSP=triton.next_power_of_2(ns_max), num_warps=4)
    STATS["grouped"] += E
    return [out[e].unsqueeze(0).transpose(1, 2) for e in range(E)]
