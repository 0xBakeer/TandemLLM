"""Fine-tune the block drafter on the target's own distribution, offline.

WHY THIS EXISTS
---------------
The engine's speed is acceptance times step rate, and the step is at the weight format's floor. The
row this program is measured against is fresh prose and chat, where the released drafter accepts
3.28 tokens a block against 6.35 on an edit. That is not a kernel problem: it is a distribution
problem. The drafter was trained on a generic English instruction and code mixture, and it is asked
here to predict a particular 27 B model on a particular workload.

Nothing about output can change. A drafter proposes; the target verifies every token against its
own argmax and keeps the matching prefix. Training it wrong makes the engine slower and never wrong,
which is why this file has no quality gate and the losslessness gate in `tools/verify_spec.py` is
the only correctness statement needed.

THE OBJECTIVE
-------------
The drafter's forward pass is non-causal over a block of eight rows: row 0 carries the last token
the target committed, rows 1..7 are mask tokens, and row j predicts the token at anchor + j. So the
loss is over those seven rows, per block:

    L = CE(draft_logits[j], target_argmax[anchor + j])
        + w * KL(target_top64[anchor + j] || draft[j])

The hard term is the one that matches the gate exactly -- greedy verification accepts row j if and
only if its argmax equals the target's -- and the soft term is what stops the hard term from
over-fitting a single token when the target itself was nearly indifferent.

WHAT MAKES IT CHEAP
-------------------
The target never runs. `tools/train_data.py` has already written its five tap tensors and its top-64
for every position, so a step reads tensors from disk and touches 1.9 B parameters, not 27 B. The
drafter's context is materialised the same way serving materialises it -- `project_context` then
`context_kv` -- and many blocks of one sequence are packed into a single pass behind a mask that
keeps them from seeing each other.

THE GATE
--------
`--eval-only`, or the evaluation that runs at every checkpoint, replays the serving loop against
held-out self-generated sequences: draft seven tokens from the anchor, count the matching prefix,
commit that many plus the target's bonus token, move the anchor, repeat. The number it reports is
the number the ledger calls accepted tokens per block, and it is exact rather than estimated,
because the target's greedy continuation of those sequences is known.

Held-out sequences are named in the data manifest and are never sampled for training.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.drafters.dflash2 import (  # noqa: E402
    DFlash2Module, load_config, load_weights,
)


# ----------------------------------------------------------------------------- target tensors

def target_tensors(snapshot: str, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """`embed_tokens.weight` and `lm_head.weight`, and nothing else of the 27 B target.

    The drafter embeds its noise block with the target's embedding table and turns its output into
    tokens with the target's head. Those two tensors are 5.1 GB together; the other 25 GB of the
    checkpoint has no role in training.
    """
    from safetensors import safe_open
    import glob
    want = {"embed_tokens.weight": None, "lm_head.weight": None}
    for path in sorted(glob.glob(os.path.join(snapshot, "*.safetensors"))):
        with safe_open(path, framework="pt", device="cpu") as f:
            keys = set(f.keys())
            for name in list(want):
                for cand in (name, f"model.{name}", f"model.language_model.{name}"):
                    if cand in keys and want[name] is None:
                        want[name] = f.get_tensor(cand).to(torch.bfloat16).to(device)
                        break
    if want["embed_tokens.weight"] is None:
        raise RuntimeError(f"no embed_tokens.weight under {snapshot}")
    if want["lm_head.weight"] is None:                       # tied embeddings, if it ever happens
        want["lm_head.weight"] = want["embed_tokens.weight"]
    return want["embed_tokens.weight"], want["lm_head.weight"]


# ----------------------------------------------------------------------------- data

def klass(topic: str) -> str:
    """The three regimes the gate is written in terms of, from the prompt's topic.

    The row this program is measured against is chat-shaped prompts across eleven topics; what
    separates the acceptance regimes is not the topic but whether the output is code, German, or
    English prose, which is how the ledger has reported acceptance since 10:17."""
    if topic in ("code",):
        return "code"
    if topic in ("multilingual", "de"):
        return "de"
    if topic == "en":
        return "prose"
    return "prose"


class Sample:
    __slots__ = ("name", "kind", "topic", "klass", "split", "ids", "fused", "label", "top_ids",
                 "top_lp", "gen_start")

    def __init__(self, meta: dict, blob: dict, device: str, keep_on: str):
        self.name = meta["name"]
        self.kind = meta["kind"]
        self.topic = meta["topic"]
        self.klass = klass(meta["topic"])
        self.split = meta["split"]
        self.gen_start = int(meta.get("gen_start", 0))
        dev = device if keep_on == "cuda" else "cpu"
        self.ids = blob["ids"].to(dev).long()
        self.fused = blob["fused"].to(dev)
        self.label = blob["label"].to(dev).long()
        self.top_ids = blob["top_ids"].to(dev).long()
        self.top_lp = blob["top_lp"].to(dev)

    def __len__(self) -> int:
        return int(self.ids.numel())


def load_data(path: str, device: str, keep_on: str, limit: int = 0) -> list[Sample]:
    with open(os.path.join(path, "manifest.json")) as f:
        manifest = json.load(f)
    out = []
    for meta in manifest["sequences"]:
        f = os.path.join(path, f"{meta['name']}.pt")
        if not os.path.exists(f):
            continue
        out.append(Sample(meta, torch.load(f, map_location="cpu"), device, keep_on))
        if limit and len(out) >= limit:
            break
    return out


# ----------------------------------------------------------------------------- masks

def block_masks(anchor_pos: torch.Tensor, ctx_len: int, block: int, window: int | None,
                device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """The (full, sliding) attention masks for `B` packed blocks over a shared context.

    Query rows are `B * block` of them, keys are `ctx_len` context rows followed by the same query
    rows. A query in block b may read a context position strictly before its anchor -- that is the
    serving rule, `ctx_len == anchor position` -- and may read every row of its OWN block and no row
    of any other. Inside a block the pass is deliberately non-causal.
    """
    b = anchor_pos.numel()
    t = b * block
    qblk = torch.arange(t, device=device) // block
    qpos = anchor_pos[qblk] + (torch.arange(t, device=device) % block)
    ctx_pos = torch.arange(ctx_len, device=device)
    kpos = torch.cat([ctx_pos, qpos])
    kblk = torch.cat([torch.full((ctx_len,), -1, device=device, dtype=torch.long), qblk])
    is_ctx = torch.arange(ctx_len + t, device=device) < ctx_len

    same_block = kblk[None, :] == qblk[:, None]
    before_anchor = kpos[None, :] < anchor_pos[qblk][:, None]
    full = torch.where(is_ctx[None, :], before_anchor, same_block)
    if window is None:
        return full, full
    in_window = (qpos[:, None] - kpos[None, :]) < window
    slide = torch.where(is_ctx[None, :], before_anchor & in_window, same_block)
    return full, slide


# ----------------------------------------------------------------------------- one training step

def run_blocks(m: DFlash2Module, embed: torch.Tensor, s: Sample, anchors: torch.Tensor,
               device: str) -> torch.Tensor:
    """Draft hidden states for `len(anchors)` blocks of one sequence. [B, block-1, H]."""
    cfg = m.cfg
    bs = cfg.block_size
    b = anchors.numel()
    ctx_len = int(anchors.max().item())               # every block reads strictly before its anchor
    fused = s.fused[:ctx_len].to(device, non_blocking=True)
    ctx_hidden = m.project_context(fused.to(m.dtype))
    ctx_pos = torch.arange(ctx_len, device=device)
    ctx_kv = m.context_kv(ctx_hidden, ctx_pos)

    ids = s.ids.to(device, non_blocking=True)
    block_ids = torch.full((b, bs), cfg.mask_token_id, dtype=torch.long, device=device)
    block_ids[:, 0] = ids[anchors]
    noise = F.embedding(block_ids.reshape(-1), embed).to(m.dtype)
    positions = (anchors[:, None] + torch.arange(bs, device=device)[None, :]).reshape(-1)
    masks = block_masks(anchors, ctx_len, bs, cfg.sliding_window, device)
    out = m.forward_block(noise, positions, ctx_kv, ctx_pos, masks=masks)
    return out.view(b, bs, -1)[:, 1:, :]


def loss_of(pred: torch.Tensor, head: torch.Tensor, s: Sample, anchors: torch.Tensor,
            kl_weight: float, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Cross-entropy to the target's argmax plus KL to its top-64. Returns (loss, hits)."""
    b, l, h = pred.shape
    logits = F.linear(pred.reshape(-1, h), head).float()                  # [b*l, V]
    # Row j of a block (j counted from 1, because row 0 is the anchor and dead) predicts the token
    # at anchor + j, and `label[i]` is the token that follows position i -- so the label wanted is
    # `label[anchor + j - 1]`. Getting this one index wrong costs nothing visible: the loss still
    # falls, on a task one position to the left of the one the verify pass grades.
    slots = anchors[:, None] + torch.arange(1, l + 1, device=device)[None, :]
    idx = (slots - 1).reshape(-1)
    label = s.label.to(device, non_blocking=True)[idx]
    ce = F.cross_entropy(logits, label)
    hits = (logits.argmax(-1) == label).float().sum()
    if kl_weight <= 0:
        return ce, hits
    top_ids = s.top_ids.to(device, non_blocking=True)[idx]
    top_lp = s.top_lp.to(device, non_blocking=True)[idx].float()
    p = torch.softmax(top_lp, dim=-1)                       # renormalised over the stored top-k
    q = torch.log_softmax(logits, dim=-1).gather(-1, top_ids)
    kl = (p * (torch.log(p.clamp_min(1e-9)) - q)).sum(-1).mean()
    return ce + kl_weight * kl, hits


# ----------------------------------------------------------------------------- the gate

@torch.no_grad()
def acceptance(m: DFlash2Module, embed: torch.Tensor, head: torch.Tensor, samples: list[Sample],
               device: str, use_selector: bool = True, max_blocks: int = 64) -> dict:
    """Replay the serving loop on held-out self-generated sequences. Exact, not estimated.

    A block anchored at `p` drafts seven tokens; the verify pass accepts the matching prefix and
    adds one bonus token of its own, so `m + 1` positions are committed and the next anchor is
    `p + m + 1`. That is the number the ledger calls accepted tokens per block.
    """
    cfg = m.cfg
    bs = cfg.block_size
    per_topic: dict[str, list[float]] = {}
    ac = torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda"))
    for s in samples:
        n = len(s)
        fused = s.fused.to(device, non_blocking=True)
        ids = s.ids.to(device, non_blocking=True)
        label = s.label.to(device, non_blocking=True)
        with ac:
            ctx_hidden = m.project_context(fused.to(m.dtype))
            ctx_pos_all = torch.arange(n, device=device)
            kv_all = [(k.to(torch.bfloat16), v.to(torch.bfloat16))
                      for k, v in m.context_kv(ctx_hidden, ctx_pos_all)]
        p = max(1, s.gen_start - 1)
        got: list[float] = []
        while p <= n - bs and len(got) < max_blocks:
            lo = 0 if cfg.sliding_window is None else max(0, p - cfg.sliding_window + 1)
            ctx_kv = [(k[:, lo:p], v[:, lo:p]) for k, v in kv_all]
            ctx_pos = ctx_pos_all[lo:p]
            block_ids = torch.full((bs,), cfg.mask_token_id, dtype=torch.long, device=device)
            block_ids[0] = ids[p]
            noise = F.embedding(block_ids, embed).to(m.dtype)
            positions = torch.arange(p, p + bs, device=device)
            with ac:
                pred = m.forward_block(noise, positions, ctx_kv, ctx_pos)[1:]
                logits = F.linear(pred.to(head.dtype), head).float()
            if use_selector and cfg.selector_rank:
                cand, unary = m.unary_candidates(logits)
                scores = m.lattice(pred, cand, unary, int(ids[p]))
                draft = m.walk(cand, scores)
            else:
                draft = logits.argmax(-1)
            want = label[p:p + bs - 1]                       # label[p+j] is the token at p+j+1
            match = int((draft != want).float().argmax()) if (draft != want).any() else bs - 1
            got.append(match + 1.0)
            p += match + 1
        per_topic.setdefault(s.klass, []).extend(got)
    out = {k: sum(v) / len(v) for k, v in per_topic.items() if v}
    allv = [x for v in per_topic.values() for x in v]
    out["ALL"] = sum(allv) / max(1, len(allv))
    out["blocks"] = float(len(allv))
    return out


# ----------------------------------------------------------------------------- export

def export(w: dict[str, torch.Tensor], snapshot: str, out_dir: str) -> None:
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    for name in ("config.json", "tokenizer_config.json", "generation_config.json"):
        src = os.path.join(snapshot, name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(out_dir, name))
    flat = {k: v.detach().to(torch.bfloat16).contiguous().cpu() for k, v in w.items()}
    save_file(flat, os.path.join(out_dir, "model.safetensors"))


# ----------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.expanduser("~/qwen38-spark-engine/train/data"))
    ap.add_argument("--ckpt", default=None, help="the drafter to start from")
    ap.add_argument("--model", default=None, help="the target snapshot, for embed and head")
    ap.add_argument("--out", default=os.path.expanduser("~/qwen38-spark-engine/train/ft"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--blocks", type=int, default=32, help="blocks packed into one pass. The "
                    "optimiser costs the same whatever the batch is -- 1.8 B parameters read and "
                    "written five times -- so the batch is what amortises it")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--kl", type=float, default=0.5, help="weight of the soft term")
    ap.add_argument("--train", default="all", help="all | fc | lastN (e.g. last2)")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-blocks", type=int, default=64)
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--keep-on", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gen-weight", type=float, default=0.5,
                    help="share of steps drawn from the self-generated sequences. They are the "
                         "serving distribution -- the drafter conditions on hidden states of text "
                         "the target itself wrote -- and the corpus half is there to keep the "
                         "drafter from narrowing onto a few hundred generations")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--budget-min", type=float, default=0.0)
    ap.add_argument("--log", default=None, help="jsonl of every logged step")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    dev = a.device

    cfg, snap = load_config(a.ckpt)
    base = load_weights(snap, device="cpu")
    w = {k: v.detach().clone().to(dev).float() if v.is_floating_point() else v.to(dev)
         for k, v in base.items()}
    m = DFlash2Module(cfg, w)

    from engine.config import load_config as load_target
    tcfg = load_target(a.model)
    embed, head = target_tensors(tcfg.path, dev)

    data = load_data(a.data, dev, a.keep_on, a.limit)
    train = [s for s in data if s.split == "train"]
    held = [s for s in data if s.split == "heldout"]
    if not held:
        raise SystemExit("no held-out sequences in the manifest")
    train_gen = [s for s in train if s.kind == "gen"]
    train_corp = [s for s in train if s.kind != "gen"]
    print(f"drafter {snap}\n{len(train)} training sequences "
          f"({sum(len(s) for s in train)} positions, {len(train_gen)} self-generated / "
          f"{len(train_corp)} corpus), {len(held)} held out", flush=True)

    # Which parameters move. The selector codebooks stay frozen: they are 254 MB of sparsely
    # gathered rows whose gradient at this batch size is almost all zeros, and the lattice they
    # score is a re-ranking of the head's own top-16, not the prediction itself.
    if a.train == "all":
        trainable = [k for k in w if not k.startswith("candidate_selector.")]
    elif a.train == "fc":
        trainable = ["fc.weight", "hidden_norm.weight", "norm.weight"]
    elif a.train.startswith("last"):
        n = int(a.train[4:])
        keep = set(range(cfg.num_hidden_layers - n, cfg.num_hidden_layers))
        trainable = ["fc.weight", "hidden_norm.weight", "norm.weight"]
        trainable += [k for k in w if k.startswith("layers.")
                      and int(k.split(".")[1]) in keep]
    else:
        raise SystemExit(f"unknown --train {a.train}")
    params = []
    for k in trainable:
        if w[k].is_floating_point():
            w[k].requires_grad_(True)
            params.append(w[k])
    n_par = sum(p.numel() for p in params)
    print(f"--train {a.train}: {len(params)} tensors, {n_par/1e9:.3f} B parameters", flush=True)

    ev = acceptance(m, embed, head, held, dev, max_blocks=a.eval_blocks)
    print("accept/block before: " + " ".join(f"{k}={v:.3f}" for k, v in sorted(ev.items())),
          flush=True)
    if a.eval_only:
        return

    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd, betas=(0.9, 0.95), eps=1e-8)
    logf = open(a.log, "a") if a.log else None
    best = ev["ALL"]
    t0 = time.perf_counter()
    t_eval = 0.0
    bs = cfg.block_size
    hist = []
    for step in range(1, a.steps + 1):
        for g in opt.param_groups:
            g["lr"] = a.lr * min(1.0, step / max(1, a.warmup)) * \
                (0.5 * (1 + math.cos(math.pi * min(1.0, step / a.steps))) * 0.9 + 0.1)
        pool = train_gen if (train_gen and train_corp and rng.random() < a.gen_weight) \
            else (train_corp or train_gen)
        s = pool[rng.randrange(len(pool))]
        n = len(s)
        lo = max(1, s.gen_start - 1) if s.kind == "gen" else 1
        hi = n - bs
        if hi <= lo:
            continue
        anchors = torch.tensor(sorted(rng.sample(range(lo, hi + 1),
                                                 min(a.blocks, hi - lo + 1))),
                               dtype=torch.long, device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            pred = run_blocks(m, embed, s, anchors, dev)
            loss, hits = loss_of(pred, head, s, anchors, a.kl, dev)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, a.clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        acc = float(hits) / (anchors.numel() * (bs - 1))
        hist.append((float(loss.detach()), acc))
        if step % 25 == 0 or step == 1:
            k = hist[-25:]
            msg = (f"step {step:5d}  loss {sum(x[0] for x in k)/len(k):.4f}  "
                   f"row-hit {sum(x[1] for x in k)/len(k):.3f}  gn {float(gn):.2f}  "
                   f"lr {opt.param_groups[0]['lr']:.2e}  "
                   f"{(time.perf_counter()-t0-t_eval)/step*1000:.0f} ms/step")
            print(msg, flush=True)
            if logf:
                logf.write(json.dumps({"step": step, "loss": sum(x[0] for x in k)/len(k),
                                       "row_hit": sum(x[1] for x in k)/len(k)}) + "\n")
                logf.flush()
        if a.eval_every and step % a.eval_every == 0:
            t_ev = time.perf_counter()
            ev = acceptance(m, embed, head, held, dev, max_blocks=a.eval_blocks)
            t_eval += time.perf_counter() - t_ev
            print(f"  [eval @{step}] " + " ".join(f"{k}={v:.3f}"
                                                  for k, v in sorted(ev.items())), flush=True)
            if logf:
                logf.write(json.dumps({"step": step, "eval": ev}) + "\n")
                logf.flush()
            if ev["ALL"] > best:
                best = ev["ALL"]
                export(w, snap, a.out)
                print(f"  [saved] {a.out} at {best:.3f} accepted/block", flush=True)
        if a.budget_min and (time.perf_counter() - t0) / 60 > a.budget_min:
            print(f"[budget] stopping at step {step}", flush=True)
            break

    ev = acceptance(m, embed, head, held, dev, max_blocks=a.eval_blocks)
    print("accept/block after:  " + " ".join(f"{k}={v:.3f}" for k, v in sorted(ev.items())),
          flush=True)
    if ev["ALL"] >= best:
        export(w, snap, a.out)
        print(f"[saved] {a.out} at {ev['ALL']:.3f} accepted/block")
    if logf:
        logf.write(json.dumps({"final": ev, "best": best}) + "\n")
        logf.close()


if __name__ == "__main__":
    main()
