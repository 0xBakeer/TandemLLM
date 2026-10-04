"""Train a DSpark-style block drafter for Kolibri-1, online against the served NVFP4 set.

WHAT IS TRAINED
---------------
The module is the one the engine already runs for Qwen's DFlash2 and DSpark checkpoints
(`engine/drafters/dflash2.DFlash2Module` + `engine/drafters/dspark.DSparkModule`), with Kolibri's
shapes. Nothing in the forward is new code: the trainer builds the module from a parameter dict
and calls the same `project_context`, `context_kv`, `forward_block`, `markov_latent` and
`markov_bias` the engine's drafter calls, so a checkpoint written here loads there.

  * taps: the residual stream AFTER target layers `target_layer_ids` (0-based; 49 is the last
    layer, before the final norm), fp32 in the engine, handed over as bf16, concatenated in that
    order -> `fc` [2560, 5 * 2560] -> `hidden_norm` (plain RMSNorm, `x * w`);
  * a block of 8 rows [anchor, MASK x 7] at positions p..p+7, embedded with Kolibri's own
    `embed_tokens` (mask id = `dflash_config.mask_token_id`); rows 1..7 propose the tokens at
    p+1..p+7; the anchor is the last token the target decided, the context is every position < p;
  * N Qwen3-style decoder layers (GQA, q/k RMSNorm, RoPE, sliding window over the context,
    non-causal inside the block), the final `norm`, Kolibri's own head (the set's e4m3 head,
    dequantised to bf16 once), plus DSpark's rank-r Markov bias `markov_w2(markov_w1[prev])`
    from the token one slot back (teacher-forced here, the drafter's own proposal at serving).

THE TARGET
----------
`engine/kolibri` loads the served NVFP4 set (kvfullsh8) on the job's GPU and runs its own prefill
path over each training sequence once: the taps, the argmax label and the top-32 log-probs at every
position come from the arithmetic the engine serves, so a label here is what the engine's verify
will compare against.

THE LOSS
--------
Per slot j (1..7): CE to the target's argmax + `--kl` x KL(target top-32 renormalised || drafter),
weighted exp(-(j-1)/`--gamma`), because slot j only counts when slots < j were accepted.

THE GATE (`acceptance`)
-----------------------
Every answer position of every held-out sequence is an anchor in one batched pass (blocks are
isolated by the mask, so a block's proposals equal what a lone draft call would propose), then the
serving loop is walked over them: from p, the matching prefix m against the target's argmax labels,
commit m + 1, next anchor p + m + 1. Exact for greedy decoding on the held-out text; per-slot
conditional rates and mean committed tokens a round, by language, kind and effort.
`--real-loop N` also runs N held-out prompts through a real speculative loop against the target
(draft, verify all eight rows with the engine, truncate the KV to the accepted prefix).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

H = 2560
VOCAB = 128000
MASK_ID = 127999                    # <|reserved-token-76|>
EOS = (127906, 127901)


# ----------------------------------------------------------------------------- the drafter

def drafter_config(layers: int, inter: int, heads: int, kv_heads: int, taps: list[int],
                   window: int, block: int, markov: int, mask_id: int = MASK_ID, hidden: int = H,
                   vocab: int = VOCAB, head_dim: int = 128) -> dict:
    return {
        "architectures": ["DSparkDraftModel"],
        "model_type": "qwen3",
        "hidden_size": hidden, "intermediate_size": inter, "num_hidden_layers": layers,
        "num_attention_heads": heads, "num_key_value_heads": kv_heads, "head_dim": head_dim,
        "vocab_size": vocab, "rms_norm_eps": 1e-6, "hidden_act": "silu",
        "rope_parameters": {"rope_type": "default", "rope_theta": 1e6},
        "layer_types": ["sliding_attention"] * layers, "sliding_window": window,
        "is_causal": False, "block_size": block,
        "markov_rank": markov, "markov_head_type": "vanilla", "enable_confidence_head": False,
        "has_embed_tokens": False, "has_lm_head": False,
        "dflash_config": {
            "block_size": block, "mask_token_id": mask_id, "target_layer_ids": list(taps),
            "tap": "residual stream after target layer L (0-based, 49 = last layer before the final norm)",
        },
        "target": {"model": "Aleph-Alpha/Kolibri-1", "set": "Kolibri-1 NVFP4 (kvfullsh8)",
                   "embed_tokens": "target's own (bf16)", "lm_head": "target's own (e4m3, row scales)"},
    }


def init_params(cfg, device, seed: int) -> dict[str, torch.nn.Parameter]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = {}
    std = 0.02
    for name, shape in cfg.expected_tensors().items():
        if name.endswith("norm.weight"):
            t = torch.ones(shape)
        elif name == "markov_head.markov_w2.weight":
            t = torch.zeros(shape)
        else:
            t = torch.randn(shape, generator=g) * std
            if name.endswith(("o_proj.weight", "down_proj.weight")):
                t = t / math.sqrt(2 * cfg.num_hidden_layers)
        out[name] = torch.nn.Parameter(t.to(device=device, dtype=torch.float32))
    return out


def build_module(raw: dict, params: dict):
    from engine.drafters.dspark import DSparkConfig, DSparkModule
    cfg = DSparkConfig(raw)
    return cfg, DSparkModule(cfg, params)


# ----------------------------------------------------------------------------- the target

def load_target(set_dir: str, device: str, max_len: int, chunk: int):
    from engine.kolibri.model import KolibriEngine
    # graphs on the GPU: the verify path proves itself against the captured decode step
    eng = KolibriEngine.load(set_dir, None, device=device, max_len=max_len,
                             graphs=str(device).startswith("cuda"))
    eng.chunk = chunk
    return eng


@torch.no_grad()
def target_pass(eng, ids: torch.Tensor, taps: list[int], start: int = 0, topk: int = 32):
    """Teacher-forced pass over `ids` at positions start.. (KV rows written). Returns the taps
    [T, len(taps) * H] bf16, the argmax label [T] (the token the target predicts AFTER each
    position) and the top-k ids / log-probs [T, k]."""
    from engine.kolibri.kernels import add_rms2, rms_fused
    c = eng.cfg
    H = c.hidden
    T = ids.numel()
    want = {L: i for i, L in enumerate(taps)}
    tap = torch.empty(T, len(taps) * H, dtype=torch.bfloat16, device=ids.device)
    label = torch.empty(T, dtype=torch.long, device=ids.device)
    top_ids = torch.empty(T, topk, dtype=torch.int32, device=ids.device)
    top_lp = torch.empty(T, topk, dtype=torch.float16, device=ids.device)
    n = len(eng.layers)
    for c0 in range(0, T, eng.chunk):
        c1 = min(T, c0 + eng.chunk)
        r = eng.emb[ids[c0:c1]].float()
        x = rms_fused(r, eng.layers[0].n_in, c.eps)
        for i, lw in enumerate(eng.layers):
            a = eng.attn.attn_prefill(x, lw, eng.kv, start + c0)
            r, x2 = add_rms2(r, a, lw.n_pa, lw.n_pal, c.eps)
            y = eng._moe(x2, lw)
            if i + 1 < n:
                r, x = add_rms2(r, y, lw.n_pf, eng.layers[i + 1].n_in, c.eps)
            else:
                r, x = add_rms2(r, y, lw.n_pf, eng.final_norm, c.eps, torch.float32)
            if i in want:
                j = want[i]
                tap[c0:c1, j * H:(j + 1) * H] = r.to(torch.bfloat16)
        for r0 in range(c0, c1, 1024):
            r1 = min(c1, r0 + 1024)
            lg = eng.head.logits(x[r0 - c0:r1 - c0])
            lse = torch.logsumexp(lg, -1, keepdim=True)
            v, ix = lg.topk(topk, -1)
            label[r0:r1] = ix[:, 0]
            top_ids[r0:r1] = ix.int()
            top_lp[r0:r1] = (v - lse).half()
    eng.kv.length = start + T
    return tap, label, top_ids, top_lp


# ----------------------------------------------------------------------------- data

class Seq:
    __slots__ = ("id", "src", "kind", "lang", "effort", "mode", "ids", "gen_start")

    def __init__(self, r: dict, max_len: int):
        self.id, self.src, self.kind, self.lang = r["id"], r["src"], r["kind"], r["lang"]
        self.effort, self.mode = r["effort"], r["mode"]
        ids = list(r["prompt_ids"]) + list(r["gen_ids"])
        self.ids = ids[:max_len]
        self.gen_start = len(r["prompt_ids"])


def load_gen(pattern: str, max_len: int, min_gen: int = 16, split: str | None = None,
             skip: set | None = None) -> list[Seq]:
    """Every record of `split` in the shards matching `pattern`; shard basenames in `skip` are passed
    over and every shard read is added to it."""
    out = []
    files = sorted(set(f for pat in pattern.split(",") if pat for f in glob.glob(pat)))
    for f in files:
        if skip is not None:
            if os.path.basename(f) in skip:
                continue
            skip.add(os.path.basename(f))
        for r in torch.load(f, weights_only=False):
            if split and r.get("split") != split:
                continue
            s = Seq(r, max_len)
            if len(s.ids) - s.gen_start >= min_gen:
                out.append(s)
    return out


# ----------------------------------------------------------------------------- one block pass

def block_masks(anchor_pos: torch.Tensor, ctx_len: int, block: int, window: int | None, device):
    """(full, sliding) masks for B packed blocks: a block reads context strictly before its anchor
    (within the window) and every row of its own block, nothing of another block."""
    b = anchor_pos.numel()
    t = b * block
    qblk = torch.arange(t, device=device) // block
    qpos = anchor_pos[qblk] + (torch.arange(t, device=device) % block)
    kpos = torch.cat([torch.arange(ctx_len, device=device), qpos])
    kblk = torch.cat([torch.full((ctx_len,), -1, device=device, dtype=torch.long), qblk])
    is_ctx = torch.arange(ctx_len + t, device=device) < ctx_len
    same = kblk[None, :] == qblk[:, None]
    before = kpos[None, :] < anchor_pos[qblk][:, None]
    full = torch.where(is_ctx[None, :], before, same)
    if window is None:
        return full, full
    inw = (qpos[:, None] - kpos[None, :]) < window
    return full, torch.where(is_ctx[None, :], before & inw, same)


def run_blocks(m, cfg, emb: torch.Tensor, ids: torch.Tensor, taps: torch.Tensor, anchors: torch.Tensor):
    """Draft hidden states [B, block-1, H] for anchors (sorted or not) of one sequence."""
    bs = cfg.block_size
    dev = ids.device
    b = anchors.numel()
    lo = 0
    hi = int(anchors.max().item())
    if cfg.sliding_window:
        lo = max(0, int(anchors.min().item()) - cfg.sliding_window)
    ctx_pos = torch.arange(lo, hi, device=dev)
    x = taps[lo:hi].to(m.dtype)
    scale = getattr(m, "tap_scale", None)
    if scale is not None:            # training space: every tap channel at unit RMS (see --tap-norm)
        x = x / scale
    ctx_hidden = m.project_context(x)
    ctx_kv = m.context_kv(ctx_hidden, ctx_pos)
    block_ids = torch.full((b, bs), cfg.mask_token_id, dtype=torch.long, device=dev)
    block_ids[:, 0] = ids[anchors]
    noise = F.embedding(block_ids.reshape(-1), emb).to(m.dtype)
    positions = (anchors[:, None] + torch.arange(bs, device=dev)[None, :]).reshape(-1)
    full, slide = block_masks(anchors - lo, hi - lo, bs, cfg.sliding_window, dev)
    out = m.forward_block(noise, positions, ctx_kv, ctx_pos, block_size=bs, masks=(full, slide))
    return out.view(b, bs, -1)[:, 1:, :]


def draft_logits(m, cfg, pred: torch.Tensor, head: torch.Tensor, prev: torch.Tensor | None):
    """[B, L, H] -> [B, L, V] logits (bf16 matmul, fp32 out), plus the Markov bias from `prev`."""
    b, l, h = pred.shape
    lg = F.linear(pred.reshape(-1, h).to(head.dtype), head).float()
    if cfg.markov_rank and prev is not None:
        lg = lg + m.markov_bias(m.markov_latent(prev.reshape(-1))).float()
    return lg.view(b, l, -1)


def propose(m, cfg, pred, head, anchors_tok: torch.Tensor, markov: bool = True) -> torch.Tensor:
    """The engine's rule: unbiased argmax, then one Markov refinement with those as `prev`."""
    lg = draft_logits(m, cfg, pred, head, None)
    ids = lg.argmax(-1)
    if markov and cfg.markov_rank:
        prev = torch.cat([anchors_tok[:, None], ids[:, :-1]], dim=1)
        ids = draft_logits(m, cfg, pred, head, prev).argmax(-1)
    return ids


# ----------------------------------------------------------------------------- loss

def loss_of(m, cfg, pred, head, ids, label, top_ids, top_lp, anchors, kl_w: float, gamma: float):
    b, l, _ = pred.shape
    dev = pred.device
    T = ids.numel()
    j = torch.arange(1, l + 1, device=dev)
    idx = anchors[:, None] + j[None, :] - 1                 # label[p+j-1] = token at p+j
    valid = idx <= T - 2
    idx_c = idx.clamp(max=T - 2)
    prev = ids[idx_c]                                        # token at p+j-1 (teacher-forced)
    lg = draft_logits(m, cfg, pred, head, prev)              # [b, l, V] fp32
    lab = label[idx_c]
    w = torch.exp(-(j - 1).float() / gamma)[None, :].expand(b, l) * valid
    # log-softmax never materialised: lse once, then gathers (one [b*l, V] fp32 tensor and its grad)
    lse = torch.logsumexp(lg, -1)
    ce = lse - lg.gather(-1, lab[..., None])[..., 0]
    tid = top_ids[idx_c].long()
    p = torch.softmax(top_lp[idx_c].float(), -1)
    q = lg.gather(-1, tid) - lse[..., None]
    kl = (p * (torch.log(p.clamp_min(1e-9)) - q)).sum(-1)
    wsum = w.sum().clamp_min(1e-6)
    loss = ((ce + kl_w * kl) * w).sum() / wsum
    hit = ((lg.detach().argmax(-1) == lab) & valid).float().sum(0)
    return loss, (ce * w).sum() / wsum, (kl * w).sum() / wsum, hit, valid.float().sum(0)


# ----------------------------------------------------------------------------- gate

@torch.no_grad()
def acceptance(m, cfg, eng, emb, head, seqs: list[Seq], taps_l: list[int], dev, block_chunk: int = 512,
               markov: bool = True, limit_tokens: int = 0, given: dict | None = None) -> dict:
    """Per-slot conditional acceptance and mean committed tokens a round on held-out sequences."""
    l = cfg.block_size - 1
    groups: dict[str, dict] = {}

    def grp(k):
        return groups.setdefault(k, {"tried": [0] * l, "acc": [0] * l, "tokens": 0, "rounds": 0, "seqs": 0,
                                     "ct": [0] * (l + 1), "cr": [0] * (l + 1)})

    ac = torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(dev).startswith("cuda"))
    for s in seqs:
        ids = torch.tensor(s.ids, device=dev)
        if given is not None and s.id in given:      # taps and labels from elsewhere (kd_tapcheck)
            taps, label = given[s.id]
        else:
            eng.reset()
            taps, label, _, _ = target_pass(eng, ids, taps_l)
        T = ids.numel()
        last = T - 2
        anchors_all = torch.arange(s.gen_start, last + 1, device=dev)
        if anchors_all.numel() == 0:
            continue
        drafts = torch.empty(anchors_all.numel(), l, dtype=torch.long, device=dev)
        for a0 in range(0, anchors_all.numel(), block_chunk):
            an = anchors_all[a0:a0 + block_chunk]
            with ac:
                pred = run_blocks(m, cfg, emb, ids, taps, an)
                drafts[a0:a0 + an.numel()] = propose(m, cfg, pred, head, ids[an], markov)
        drafts = drafts.cpu().tolist()
        lab = label.cpu().tolist()
        keys = ["ALL", f"lang:{s.lang}", f"kind:{s.kind}", f"effort:{s.effort}", f"mode:{s.mode}",
                f"src:{s.src}"]
        gs = [grp(k) for k in keys]
        for g in gs:
            g["seqs"] += 1
        p = s.gen_start
        while p <= last:
            d = drafts[p - s.gen_start]
            mm = 0
            for jj in range(l):
                if p + jj > last:
                    break
                for g in gs:
                    g["tried"][jj] += 1
                if d[jj] == lab[p + jj]:
                    mm += 1
                    for g in gs:
                        g["acc"][jj] += 1
                else:
                    break
            step = mm + 1
            for g in gs:
                g["tokens"] += min(step, last + 2 - p)
                g["rounds"] += 1
            p += step
        for k in range(1, l + 1):          # the same walk with the chain cut at k slots
            p, tok, rnd = s.gen_start, 0, 0
            while p <= last:
                d = drafts[p - s.gen_start]
                mm = 0
                while mm < k and p + mm <= last and d[mm] == lab[p + mm]:
                    mm += 1
                tok += min(mm + 1, last + 2 - p)
                rnd += 1
                p += mm + 1
            for g in gs:
                g["ct"][k] += tok
                g["cr"][k] += rnd
    out = {}
    for k, g in groups.items():
        out[k] = {"seqs": g["seqs"], "rounds": g["rounds"],
                  "tokens_per_round": g["tokens"] / max(1, g["rounds"]),
                  "slot_rates": [a / t if t else None for a, t in zip(g["acc"], g["tried"])],
                  "tried": list(g["tried"]),
                  "chains": {str(k): g["ct"][k] / max(1, g["cr"][k]) for k in range(1, l + 1)}}
    return out


def make_verify(eng, prompt_ids: list[int], log=print):
    """The bit-exact verify path (engine/kolibri/verify.py): its GPU twins when the load-time
    self-check proves them equal to the decode step, else its reference semantics (`ReplayVerifier`:
    each node decoded alone), which is exact by construction and only slower."""
    from engine.kolibri.verify import ReplayVerifier, make_verifier
    # inference mode around all of it: the engine's KV and graphs were made there (stage one s1b died
    # in `capture` with "inplace update to inference tensor outside InferenceMode")
    v = make_verifier(eng, log=log, selfcheck_ids=list(prompt_ids)) if eng.device.type == "cuda" else None
    if v is None:
        log("[loop] verify: ReplayVerifier (the decode step per node)")
        return ReplayVerifier(eng), "replay"
    v.capture(range(2, 9))
    return v, "gpu-twins"


@torch.no_grad()
def real_loop(m, cfg, eng, emb, head, ver, prompt_ids: list[int], taps_l: list[int], dev,
              max_new: int = 256, markov: bool = True, chain: int = 0) -> dict:
    """A real speculative loop for greedy decoding through the bit-exact verify path: draft `chain` tokens
    (0 = the block's 7), verify [anchor + drafts] as a chain (rows bit-equal to the decode step, no
    cache write), keep the matching prefix plus the target's own next token, commit that path.

    The taps of the committed rows come from a prefill pass over them before the commit (the
    commit then overwrites those KV rows with the verify path's own, so the text stays the decode
    step's); prefill taps differ from decode taps by about one BF16 ulp, noise to a drafter."""
    l = chain or (cfg.block_size - 1)
    ac = torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(dev).startswith("cuda"))
    eng.reset()
    ids = torch.tensor(prompt_ids, device=dev)
    taps, label, _, _ = target_pass(eng, ids, taps_l)
    toks = list(prompt_ids) + [int(label[-1])]
    tap_rows = [taps]
    accepted = []
    t0 = time.time()
    while len(toks) - len(prompt_ids) < max_new and toks[-1] not in EOS:
        p = len(toks) - 1
        assert eng.kv.length == p, (eng.kv.length, p)
        allt = torch.cat(tap_rows) if len(tap_rows) > 1 else tap_rows[0]
        tap_rows = [allt]
        idt = torch.tensor(toks, device=dev)
        an = torch.tensor([p], device=dev)
        with ac:
            pred = run_blocks(m, cfg, emb, idt, allt, an)
            d = propose(m, cfg, pred, head, idt[an], markov)[0].tolist()[:l]
        lg = ver.verify([toks[-1]] + d, [-1] + list(range(len(d))))
        vl = lg.argmax(-1).tolist()
        mm = 0
        while mm < len(d) and d[mm] == vl[mm]:
            mm += 1
        accepted.append(mm)
        new = d[:mm] + [vl[mm]]
        for i, t in enumerate(new):
            if t in EOS:
                new = new[:i + 1]
                break
        vtap, _, _, _ = target_pass(eng, torch.tensor([toks[-1]] + d[:mm], device=dev), taps_l, start=p)
        ver.commit(list(range(mm + 1)))
        tap_rows.append(vtap)
        toks += new
    dt = time.time() - t0
    return {"gen": toks[len(prompt_ids):], "accepted": accepted, "seconds": dt,
            "tokens_per_round": (len(toks) - len(prompt_ids) - 1) / max(1, len(accepted))}


@torch.no_grad()
def greedy_plain(eng, prompt_ids: list[int], taps_l, dev, max_new: int) -> list[int]:
    """Plain greedy through the engine's own decode step (the reference the verify path equals).
    The first token comes from the same prefill pass the loop uses."""
    eng.reset()
    ids = torch.tensor(prompt_ids, device=dev)
    _, label, _, _ = target_pass(eng, ids, taps_l)
    out = [int(label[-1])]
    while len(out) < max_new and out[-1] not in EOS:
        out.append(int(eng.decode(out[-1]).argmax()))
    return out


# ----------------------------------------------------------------------------- export / resume

def export(params: dict, raw: dict, out_dir: str, meta: dict, tap_scale: torch.Tensor | None = None) -> str:
    """bf16 weights + config. With a tap scale, `fc` is folded back to raw taps: fc(x / s) = (W / s) x,
    so the engine feeds the target's residual rows as they are."""
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    flat = {}
    for k, v in params.items():
        v = v.detach().float()
        if k == "fc.weight" and tap_scale is not None:
            v = v / tap_scale.to(v.device)[None, :]
        flat[k] = v.to(torch.bfloat16).contiguous().cpu()
    tmp = os.path.join(out_dir, "model.safetensors.part")
    save_file(flat, tmp)
    os.replace(tmp, os.path.join(out_dir, "model.safetensors"))
    json.dump(raw, open(os.path.join(out_dir, "config.json"), "w"), indent=2)
    json.dump(meta, open(os.path.join(out_dir, "train_meta.json"), "w"), indent=1)
    return out_dir


def save_state(path: str, params, opt, step: int, cursor: int, epoch: int, rng: random.Random, extra: dict):
    """Written locally, then copied whole (the bucket mount takes sequential writes best)."""
    blob = {"params": {k: v.detach().cpu() for k, v in params.items()},
            "opt": opt.state_dict(), "step": step, "cursor": cursor, "epoch": epoch,
            "rng": rng.getstate(), "torch_rng": torch.get_rng_state(), "extra": extra}
    local = path + ".local" if not path.startswith("/work") else "/tmp/kd_resume.pt"
    torch.save(blob, local)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    import shutil
    shutil.copyfile(local, path + ".part")
    os.replace(path + ".part", path)
    if local != path:
        os.remove(local)


def mirror(src: str, dst_dir: str | None) -> None:
    """Copy a file or a directory of files from the job's disk into the bucket directory."""
    if not dst_dir:
        return
    import shutil
    if os.path.isdir(src):
        d = os.path.join(dst_dir, os.path.basename(src.rstrip("/")))
        os.makedirs(d, exist_ok=True)
        for f in os.listdir(src):
            shutil.copyfile(os.path.join(src, f), os.path.join(d, f))
    elif os.path.exists(src):
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copyfile(src, os.path.join(dst_dir, os.path.basename(src)))


# ----------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True, help="the served set directory (with config.json)")
    ap.add_argument("--gen", required=True, help="glob of training generation shards")
    ap.add_argument("--held", default=None, help="glob of held-out generation shards")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--inter", type=int, default=6144)
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--kv-heads", type=int, default=4)
    ap.add_argument("--taps", default="1,13,25,37,49")
    ap.add_argument("--window", type=int, default=2048)
    ap.add_argument("--block", type=int, default=8)
    ap.add_argument("--markov", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=6144)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--anchors", type=int, default=256, help="anchors a sequence per step (256 anchors "
                    "from each of 8 sequences: the same drafter work as 1,024 from 2, four times the contexts)")
    ap.add_argument("--accum", type=int, default=8, help="sequences per optimizer step")
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--min-lr", type=float, default=6e-5)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--decay-start", type=int, default=0)
    ap.add_argument("--steps", type=int, default=0, help="0: one pass over --epochs of the data")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--kl", type=float, default=0.5)
    ap.add_argument("--gamma", type=float, default=4.0)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--eval-seqs", type=int, default=200)
    ap.add_argument("--save-every-min", type=float, default=40.0)
    ap.add_argument("--budget-min", type=float, default=0.0)
    ap.add_argument("--gen-live", default=None, help="glob (on the bucket mount) rescanned every --rescan-min "
                    "for shards a running generation job wrote since the start; new sequences join the rest of "
                    "the epoch at random places")
    ap.add_argument("--rescan-min", type=float, default=10.0)
    ap.add_argument("--gen-exclude", default="", help="comma-separated shard basenames never read for "
                    "training (also on rescans): the A/B of fresh against repeated answers")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--ckpt-dir", default=None, help="bucket directory: resume state, exports and the log "
                    "are copied there as they are written, so a killed job loses at most one interval")
    ap.add_argument("--init", default=None, help="start from this exported drafter")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tap-norm", action="store_true", help="train on taps divided by their per-channel "
                    "RMS (measured once on the first training sequences); folded into fc at export. The "
                    "taps' channels span 0.3 to 800 in RMS, and Adam's per-weight steps then move a large "
                    "channel's contribution 1,000 times more than a small one's")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--mask-id", type=int, default=MASK_ID)
    ap.add_argument("--pilot-steps", type=int, default=12)
    ap.add_argument("--pilot", action="store_true", help="measure throughput and stop")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--real-loop", type=int, default=0)
    ap.add_argument("--loop-chain", type=int, default=3, help="drafted tokens a round in the real loop")
    ap.add_argument("--loop-tokens", type=int, default=256)
    ap.add_argument("--stop-step", type=int, default=0, help="stop at this step (resume state kept, the "
                    "schedule's --steps unchanged): a stage of a longer run")
    a = ap.parse_args()
    t_start = time.time()
    dev = a.device
    cuda = dev.startswith("cuda")
    torch.manual_seed(a.seed)
    taps_l = [int(x) for x in a.taps.split(",")]
    t0 = time.time()
    eng = load_target(a.set, dev, max(a.max_len + 64, 8192), a.chunk)
    emb = eng.emb
    head = (eng.head.w.float() * eng.head.s[:, None]).to(torch.bfloat16 if cuda else torch.float32)
    if cuda:
        print(f"[kd] target loaded in {time.time() - t0:.0f} s; GPU {torch.cuda.memory_allocated() / 1e9:.1f} GB",
              flush=True)
    rn = emb.float().norm(dim=-1)
    print(f"[kd] embedding row norms: median {rn.median():.3f}, mask id {a.mask_id}: {rn[a.mask_id]:.3f}, "
          f"min {rn.min():.4f}", flush=True)
    raw = drafter_config(a.layers, a.inter, a.heads, a.kv_heads, taps_l, a.window, a.block, a.markov,
                         a.mask_id, eng.cfg.hidden, eng.cfg.vocab, a.head_dim)
    if a.init:
        raw = json.load(open(os.path.join(a.init, "config.json")))
        taps_l = raw["dflash_config"]["target_layer_ids"]
    from engine.drafters.dspark import DSparkConfig
    cfg0 = DSparkConfig(raw)
    params = init_params(cfg0, dev, a.seed)
    # The drafter's hidden goes through Kolibri's own head, whose rows are small (median norm 0.032):
    # the target's final RMSNorm carries a weight near 43. Starting the drafter's `norm` at 1 gives
    # logits 40 times too small (std 0.03 instead of 1.4), a near-uniform softmax, and a loss that
    # must first grow that weight through Adam steps of about lr each. Start it at the target's.
    with torch.no_grad():
        params["norm.weight"].copy_(eng.final_norm.float().to(params["norm.weight"].device))
    if a.init:
        from safetensors.torch import load_file
        sd = load_file(os.path.join(a.init, "model.safetensors"))
        for k, v in sd.items():
            params[k].data.copy_(v.float())
    cfg, m = build_module(raw, params)
    nparam = sum(p.numel() for p in params.values())
    print(f"[kd] drafter {nparam / 1e6:.1f} M params ({a.layers} layers, inter {a.inter}, "
          f"{a.heads}/{a.kv_heads} heads, taps {taps_l}, window {a.window}, markov {a.markov})", flush=True)


    seen_shards: set = set(x for x in a.gen_exclude.split(",") if x)
    train = load_gen(a.gen, a.max_len, split="train", skip=seen_shards)
    held = load_gen(a.held, a.max_len, split="heldout") if a.held else []
    print(f"[kd] train {len(train)} sequences, {sum(len(s.ids) - s.gen_start for s in train)} answer tokens; "
          f"held-out {len(held)}", flush=True)
    rng = random.Random(a.seed)
    held_eval = sorted(held, key=lambda s: s.id)[: a.eval_seqs]

    tap_scale = None
    if a.tap_norm:
        blob0 = torch.load(a.resume, map_location="cpu", weights_only=False) if (a.resume and os.path.exists(a.resume)) else None
        if blob0 is not None and blob0.get("extra", {}).get("tap_scale") is not None:
            tap_scale = blob0["extra"]["tap_scale"].to(dev)
        else:
            acc, nrow = None, 0
            for s0 in train[:8]:
                eng.reset()
                tp, *_ = target_pass(eng, torch.tensor(s0.ids, device=dev), taps_l)
                sq = tp.float().pow(2).sum(0)
                acc = sq if acc is None else acc + sq
                nrow += tp.shape[0]
            tap_scale = (acc / nrow).sqrt().clamp_min(1e-3)
        m.tap_scale = tap_scale
        print(f"[kd] tap norm: channel RMS median {tap_scale.median():.3f}, max {tap_scale.max():.1f}", flush=True)
    decay = [p for k, p in params.items() if not k.endswith("norm.weight")]
    nodecay = [p for k, p in params.items() if k.endswith("norm.weight")]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd},
                             {"params": nodecay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95), eps=1e-8, fused=cuda)
    total = a.steps or max(1, int(len(train) * a.epochs / a.accum))
    step, cursor, epoch = 0, 0, 0
    order = list(range(len(train)))
    random.Random(a.seed).shuffle(order)
    if a.resume and os.path.exists(a.resume):
        blob = torch.load(a.resume, map_location="cpu", weights_only=False)
        for k, v in blob["params"].items():
            params[k].data.copy_(v.to(dev))
        opt.load_state_dict(blob["opt"])
        if tap_scale is not None and blob.get("extra", {}).get("tap_scale") is None:
            # the state was trained on raw taps: fc(x) = (W * s)(x / s), and fc's Adam moments restart
            params["fc.weight"].data.mul_(tap_scale[None, :])
            opt.state.pop(params["fc.weight"], None)
            print("[kd] resume: fc moved into the tap-norm space, its Adam state restarted", flush=True)
        step, cursor, epoch = blob["step"], blob["cursor"], blob["epoch"]
        rng.setstate(blob["rng"])
        torch.set_rng_state(blob["torch_rng"])
        order = list(range(len(train)))
        random.Random(a.seed + epoch).shuffle(order)
        print(f"[kd] resumed at step {step}, cursor {cursor}, epoch {epoch}", flush=True)

    decay_start = a.decay_start if a.decay_start else a.warmup

    def lr_at(s):
        """Warmup, then constant until --decay-start (0: right after warmup, i.e. plain cosine), then
        cosine to --min-lr at the last step. A constant middle lets a later job extend the run."""
        if s < a.warmup:
            return a.lr * (s + 1) / a.warmup
        if s < decay_start:
            return a.lr
        f = min(1.0, (s - decay_start) / max(1, total - decay_start))
        return a.min_lr + 0.5 * (a.lr - a.min_lr) * (1 + math.cos(math.pi * f))

    os.makedirs(a.out, exist_ok=True)
    log_eval = open(os.path.join(a.out, "eval_log.jsonl"), "a")
    step_now = [0]

    def evaluate(tag, seqs=None):
        seqs = held_eval if seqs is None else seqs
        if not seqs:
            return None
        te = time.time()
        res = acceptance(m, cfg, eng, emb, head, seqs, taps_l, dev)
        g = res.get("mode:greedy")
        if g:
            print(f"[gate {tag}] greedy subset ({g['seqs']} answers): chain 3 {g['chains']['3']:.3f} tokens a "
                  f"round; all ({res['ALL']['seqs']}): {res['ALL']['chains']['3']:.3f}; tried per slot (all) "
                  f"{res['ALL']['tried']}", flush=True)
        allr = res["ALL"]
        print(f"[eval {tag}] chain 3: {allr['chains']['3']:.3f} tokens a round; chains 1..7: "
              + " ".join(f"{allr['chains'][str(k)]:.3f}" for k in range(1, cfg.block_size))
              + "; slots " + " ".join(f"{x:.3f}" for x in allr["slot_rates"] if x is not None)
              + f" ({allr['seqs']} seqs, {time.time() - te:.0f} s)", flush=True)
        for k in sorted(res):
            if k != "ALL":
                print(f"   {k:22s} chain 3 {res[k]['chains']['3']:.3f}, chain 7 {res[k]['tokens_per_round']:.3f}"
                      f"  ({res[k]['seqs']} seqs)", flush=True)
        if log_eval is not None:
            log_eval.write(json.dumps({"tag": tag, "step": step_now[0], "eval": res}) + "\n")
            log_eval.flush()
        return res

    ac = torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda)
    os.makedirs(a.out, exist_ok=True)
    if a.eval_only:
        res = evaluate(f"only (all {len(held)} held-out)", held)
        json.dump(res, open(os.path.join(a.out, "eval.json"), "w"), indent=1)
        if a.real_loop:
            loops(a, m, cfg, eng, emb, head, held, taps_l, dev)
        return

    log = open(os.path.join(a.out, "train_log.jsonl"), "a")
    stat = {"tgt_tok": 0, "tgt_s": 0.0, "drf_s": 0.0, "anchors": 0, "loss": 0.0, "ce": 0.0, "kl": 0.0, "n": 0,
            "hit": torch.zeros(cfg.block_size - 1), "tried": torch.zeros(cfg.block_size - 1)}
    last_save = time.time()
    pilot_left = a.pilot_steps if a.pilot else None
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    last_scan = time.time()
    while step < total:
        step_now[0] = step
        if a.stop_step and step >= a.stop_step:
            print(f"[kd] stop step {a.stop_step} reached", flush=True)
            break
        if a.gen_live and (time.time() - last_scan) / 60 > a.rescan_min:
            last_scan = time.time()
            new = load_gen(a.gen_live, a.max_len, split="train", skip=seen_shards)
            if new:
                n0 = len(train)
                train.extend(new)
                rest = order[cursor:] + list(range(n0, len(train)))
                random.Random(a.seed + step).shuffle(rest)
                order = order[:cursor] + rest
                if not a.steps:
                    total = max(total, int(len(train) * a.epochs / a.accum))
                print(f"[kd] rescan: +{len(new)} sequences, {len(train)} in all, total steps {total}", flush=True)
        if a.budget_min and (time.time() - t_start) / 60 > a.budget_min:
            print(f"[kd] budget {a.budget_min} min reached at step {step}", flush=True)
            break
        opt.zero_grad(set_to_none=True)
        for _ in range(a.accum):
            if cursor >= len(order):
                epoch += 1
                cursor = 0
                order = list(range(len(train)))
                random.Random(a.seed + epoch).shuffle(order)
            s = train[order[cursor]]
            cursor += 1
            ids = torch.tensor(s.ids, device=dev)
            sync()
            t1 = time.time()
            eng.reset()
            taps, label, top_ids, top_lp = target_pass(eng, ids, taps_l)
            sync()
            t2 = time.time()
            T = ids.numel()
            cand = torch.arange(s.gen_start, T - 1, device=dev)
            if cand.numel() > a.anchors:
                cand = cand[torch.randperm(cand.numel(), device=dev)[: a.anchors]]
            with ac:
                pred = run_blocks(m, cfg, emb, ids, taps, cand)
                loss, ce, kl, hit, tried = loss_of(m, cfg, pred, head, ids, label, top_ids, top_lp, cand,
                                                   a.kl, a.gamma)
            (loss / a.accum).backward()
            sync()
            t3 = time.time()
            stat["tgt_tok"] += T
            stat["tgt_s"] += t2 - t1
            stat["drf_s"] += t3 - t2
            stat["anchors"] += cand.numel()
            stat["loss"] += float(loss)
            stat["ce"] += float(ce)
            stat["kl"] += float(kl)
            stat["n"] += 1
            stat["hit"] += hit.cpu()
            stat["tried"] += tried.cpu()
        gn = float(torch.nn.utils.clip_grad_norm_(list(params.values()), 1.0))
        stat["gn"] = stat.get("gn", 0.0) + gn
        stat["gn_n"] = stat.get("gn_n", 0) + 1
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.step()
        step += 1
        if step % 20 == 0 or (pilot_left is not None):
            n = max(1, stat["n"])
            acc = (stat["hit"] / stat["tried"].clamp_min(1)).tolist()
            rec = {"step": step, "epoch": epoch, "cursor": cursor, "lr": lr_at(step),
                   "loss": stat["loss"] / n, "ce": stat["ce"] / n, "kl": stat["kl"] / n,
                   "slot_acc": [round(x, 4) for x in acc],
                   "tgt_tok_s": stat["tgt_tok"] / max(1e-6, stat["tgt_s"]),
                   "anchors_s": stat["anchors"] / max(1e-6, stat["drf_s"]),
                   "s_per_step": (stat["tgt_s"] + stat["drf_s"]) / max(1, n / a.accum),
                   "tgt_share": stat["tgt_s"] / max(1e-6, stat["tgt_s"] + stat["drf_s"]),
                   "grad_norm": stat.get("gn", 0.0) / max(1, stat.get("gn_n", 0)),
                   "mem_gb": torch.cuda.max_memory_allocated() / 1e9 if cuda else 0.0,
                   "elapsed_min": (time.time() - t_start) / 60}
            print("[kd] " + json.dumps(rec), flush=True)
            if log:
                log.write(json.dumps(rec) + "\n")
                log.flush()
            stat["gn"], stat["gn_n"] = 0.0, 0
            for k in ("tgt_tok", "tgt_s", "drf_s", "anchors", "loss", "ce", "kl", "n"):
                stat[k] = 0 if isinstance(stat[k], int) else 0.0
            stat["hit"].zero_()
            stat["tried"].zero_()
        if pilot_left is not None:
            pilot_left -= 1
            if pilot_left <= 0:
                break
        if a.eval_every and step % a.eval_every == 0:
            step_now[0] = step
            res = evaluate(f"step {step}")
            mirror(export(params, raw, os.path.join(a.out, f"step-{step:06d}"), {"step": step, "eval": res}, tap_scale),
                   a.ckpt_dir)
            mirror(os.path.join(a.out, "train_log.jsonl"), a.ckpt_dir)
            mirror(os.path.join(a.out, "eval_log.jsonl"), a.ckpt_dir)
        if (time.time() - last_save) / 60 > a.save_every_min:
            save_state(os.path.join(a.ckpt_dir or a.out, "resume.pt"), params, opt, step, cursor, epoch, rng,
                       {"tap_scale": None if tap_scale is None else tap_scale.cpu()})
            mirror(os.path.join(a.out, "train_log.jsonl"), a.ckpt_dir)
            last_save = time.time()
            print(f"[kd] resume state saved at step {step}", flush=True)
    step_now[0] = step
    res = evaluate(f"final step {step} (all {len(held)} held-out)", held)
    mirror(os.path.join(a.out, "eval_log.jsonl"), a.ckpt_dir)
    meta = {"step": step, "epoch": epoch, "cursor": cursor, "total": total, "eval": res, "args": vars(a),
            "minutes": (time.time() - t_start) / 60}
    mirror(export(params, raw, os.path.join(a.out, "final"), meta, tap_scale), a.ckpt_dir)
    mirror(os.path.join(a.out, "train_log.jsonl"), a.ckpt_dir)
    if not a.pilot:
        save_state(os.path.join(a.ckpt_dir or a.out, "resume.pt"), params, opt, step, cursor, epoch, rng,
                       {"tap_scale": None if tap_scale is None else tap_scale.cpu()})
    if a.real_loop:
        loops(a, m, cfg, eng, emb, head, held, taps_l, dev)
        mirror(os.path.join(a.out, "real_loop.json"), a.ckpt_dir)
    print(f"[kd] done: step {step}, {(time.time() - t_start) / 60:.1f} min", flush=True)


@torch.inference_mode()
def loops(a, m, cfg, eng, emb, head, held, taps_l, dev):
    """Real speculative loops on held-out GREEDY prompts through the bit-exact verify path, beside a plain
    greedy decode of the same prompt and the teacher-forced walk over the loop's own text."""
    picks = [s for s in sorted(held, key=lambda s: s.id) if s.mode == "greedy"][: a.real_loop]
    if not picks:
        return
    if str(dev).startswith("cuda"):
        torch.cuda.empty_cache()
    ver, kind = make_verify(eng, picks[0].ids[: picks[0].gen_start])
    rows = []
    for s in picks:
        prompt = s.ids[: s.gen_start]
        r = real_loop(m, cfg, eng, emb, head, ver, prompt, taps_l, dev, max_new=a.loop_tokens,
                      chain=a.loop_chain)
        g = greedy_plain(eng, prompt, taps_l, dev, max_new=len(r["gen"]))
        same = g == r["gen"][: len(g)]
        first_diff = next((i for i, (x, y) in enumerate(zip(g, r["gen"])) if x != y), None)
        own = Seq({"id": s.id, "src": s.src, "kind": s.kind, "lang": s.lang, "effort": s.effort,
                   "mode": "greedy", "prompt_ids": prompt, "gen_ids": r["gen"]}, 10 ** 9)
        tf = acceptance(m, cfg, eng, emb, head, [own], taps_l, dev)["ALL"]["chains"]
        k = a.loop_chain or (cfg.block_size - 1)
        rows.append({"id": s.id, "lang": s.lang, "kind": s.kind, "tokens": len(r["gen"]),
                     "rounds": len(r["accepted"]), "tokens_per_round": r["tokens_per_round"],
                     "teacher_forced_same_text": tf[str(k)], "same_as_plain_greedy": same,
                     "first_diff": first_diff, "seconds": r["seconds"],
                     "accept_hist": [r["accepted"].count(j) for j in range(k + 1)]})
        print(f"[loop] {s.id} {s.lang}/{s.kind}: {len(r['gen'])} tokens, {r['tokens_per_round']:.2f} a round "
              f"(teacher-forced on the same text {tf[str(k)]:.2f}), same as plain greedy: {same} "
              f"(first diff {first_diff})", flush=True)
    tpr = sum(r["tokens"] for r in rows) / max(1, sum(r["rounds"] for r in rows))
    same_all = all(r["same_as_plain_greedy"] for r in rows)
    print(f"[loop] verify {kind}, chain {a.loop_chain or cfg.block_size - 1}: mean {tpr:.3f} tokens a round over "
          f"{len(rows)} prompts; all equal to plain greedy: {same_all}", flush=True)
    json.dump({"verify": kind, "chain": a.loop_chain, "rows": rows, "tokens_per_round": tpr,
               "all_same": same_all}, open(os.path.join(a.out, "real_loop.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
