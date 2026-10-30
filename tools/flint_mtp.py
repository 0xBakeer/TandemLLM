"""SPD-146: a multi-token-prediction draft head for Flint, distilled from the model itself on one GPU.

    PYTHONPATH=<repo> python tools/flint_mtp.py train --model <bf16 dir> --corpus train.txt --heldout a.txt,b.txt \
        --tokens 4000000 --out mtp.pt
    PYTHONPATH=<repo> python tools/flint_mtp.py eval --model <dir> --head mtp.pt --heldout a.txt,b.txt

The head (~70M parameters of its own) reads the trunk's last hidden state h_t (before the final norm) and the
embedding of token t+1, and predicts token t+2:

  x = fc([enorm(embed(t+1)) | hnorm(h_t)])               4,096 -> 2,048 (h's half starts as the identity)
  x = x + attention(input_norm(x))                       one Flint attention layer: q/k RMS norm, NEOX rotary,
                                                         a sink per head, window 513 (starts as layer 47's)
  x = x + swiglu(post_norm(x))                           2,048 -> 1,024 -> 2,048 (starts as layer 47's shared expert)
  logits = lm_head(norm(x))                              the trunk's final norm and head, frozen

Deeper drafts feed the head its own x and the token it drafted. The target at t is the trunk's own distribution at
t+1 (its next-next token given the text through t+1): the head learns to draft what the model will write, not the
corpus. Loss: KL from the trunk's top-32 probabilities (renormalised) plus the cross-entropy of its top-1. The trunk
(this engine on the BF16 checkpoint) runs each 2,048-token chunk once: hidden states and logits on the fly.
Accuracy reported: depth-1 acceptance = the head's top-1 equals the trunk's top-1 on held-out text.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

INIT_LAYER = 47


def rms(x, w, eps):
    x32 = x.float()
    return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * w


class FlintMTP(nn.Module):
    def __init__(self, cfg, W: dict):
        super().__init__()
        c = cfg
        H, D, Hq, Hk = c.hidden_size, c.head_dim, c.num_attention_heads, c.num_key_value_heads
        S = c.shared_expert_intermediate_size or 1024
        self.c = c
        p = f"layers.{INIT_LAYER}."
        g = lambda n: nn.Parameter(W[n].detach().float().clone())
        self.enorm = nn.Parameter(torch.ones(H))
        self.hnorm = nn.Parameter(torch.ones(H))
        fc = torch.zeros(H, 2 * H)
        fc[:, :H].normal_(0, 0.01)
        fc[:, H:] = torch.eye(H)
        self.fc = nn.Parameter(fc)
        self.input_norm = g(p + "input_layernorm.weight")
        self.q = g(p + "self_attn.q_proj.weight")
        self.k = g(p + "self_attn.k_proj.weight")
        self.v = g(p + "self_attn.v_proj.weight")
        self.o = g(p + "self_attn.o_proj.weight")
        self.q_norm = g(p + "self_attn.q_norm.weight")
        self.k_norm = g(p + "self_attn.k_norm.weight")
        self.sink = g(p + "self_attn.softmax_offset")
        self.post_norm = g(p + "post_attention_layernorm.weight")
        s = p + "mlp.shared_expert."
        self.gate = g(s + "gate_proj.weight")
        self.up = g(s + "up_proj.weight")
        self.down = g(s + "down_proj.weight")
        inv = 1.0 / (c.rope_theta ** (torch.arange(0, D, 2, dtype=torch.float64) / D))
        self.register_buffer("inv_freq", inv.float(), persistent=False)

    def step(self, h, nxt_emb, pos):
        """h [T, H] (the trunk's or the head's own x), nxt_emb [T, H], pos [T] -> x [T, H] (before the final norm)."""
        c = self.c
        eps = c.rms_norm_eps
        T = h.shape[0]
        D, Hq, Hk = c.head_dim, c.num_attention_heads, c.num_key_value_heads
        x = torch.cat([rms(nxt_emb, self.enorm, eps), rms(h, self.hnorm, eps)], -1) @ self.fc.T
        n = rms(x, self.input_norm, eps)
        q = rms((n @ self.q.T).view(T, Hq, D), self.q_norm, eps)
        k = rms((n @ self.k.T).view(T, Hk, D), self.k_norm, eps)
        v = (n @ self.v.T).view(T, Hk, D)
        f = pos.float()[:, None] * self.inv_freq[None]
        cos, sin = torch.cat([f, f], -1).cos()[:, None], torch.cat([f, f], -1).sin()[:, None]
        rot = lambda t: torch.cat([-t[..., D // 2:], t[..., :D // 2]], -1)
        q, k = q * cos + rot(q) * sin, k * cos + rot(k) * sin
        k, v = k.repeat_interleave(Hq // Hk, 1), v.repeat_interleave(Hq // Hk, 1)
        s = torch.einsum("thd,shd->hts", q, k) / math.sqrt(D)
        i, j = pos[:, None], pos[None, :]
        allow = (j <= i) & (i - j < c.sliding_window)
        s = s.masked_fill(~allow[None], float("-inf"))
        s = torch.cat([s, self.sink.view(Hq, 1, 1).expand(Hq, T, 1)], -1)
        o = torch.einsum("hts,shd->thd", torch.softmax(s.float(), -1)[..., :T].to(v.dtype), v).reshape(T, Hq * D)
        x = x + o @ self.o.T
        n = rms(x, self.post_norm, eps)
        return x + (F.silu(n @ self.gate.T) * (n @ self.up.T)) @ self.down.T


class Trunk:
    """The model on this engine: per chunk, every position's hidden state before the final norm and its logits."""

    def __init__(self, model: str, max_len: int):
        from engine.models.flint.engine import FlintEngine
        from engine.models.flint.load import load
        idx, W, experts, _ = load(model, "cuda", log=None)
        self.cfg, self.W = idx.cfg, W
        self.eng = FlintEngine(idx.cfg, W, experts, device="cuda", max_len=max_len)
        self.max_len = max_len

    @torch.no_grad()
    def run(self, ids: torch.Tensor):
        st = self.eng.new_state(self.max_len)
        lg = self.eng.prefill(ids, st, head="all")
        return self.eng.prompt_hidden_pre.clone(), lg

    def embed(self, ids):
        return self.W["embed_tokens.weight"].index_select(0, ids.cuda())

    def head(self, x):
        return self.eng.head(rms(x.to(torch.bfloat16), self.W["norm.weight"], self.cfg.rms_norm_eps))


def chunks(path: str, tok, T: int, limit: int | None = None):
    ids = tok(open(path, encoding="utf-8", errors="ignore").read(), add_special_tokens=False)["input_ids"]
    out = [ids[i:i + T] for i in range(0, len(ids) - T + 1, T)]
    return out[:limit] if limit else out


def loss_of(mtp, trunk, h, lg, ids):
    """The head on positions 0..T-3: input h_t and token t+1, target the trunk's distribution at t+1."""
    T = ids.numel()
    pos = torch.arange(T - 2, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        x = mtp.step(h[:T - 2].float(), trunk.embed(ids[1:T - 1]).float(), pos)
    # logits through the frozen head with gradients to x
    hw = trunk.W["lm_head.weight"]
    nx = rms(x.float(), trunk.W["norm.weight"].float(), trunk.cfg.rms_norm_eps)
    tgt = lg[1:T - 1].float()
    tp, ti = torch.topk(torch.softmax(tgt, -1), 32, -1)
    tp = tp / tp.sum(-1, keepdim=True)
    total, acc = 0.0, 0
    for a in range(0, T - 2, 512):
        b = min(a + 512, T - 2)
        q = F.log_softmax((nx[a:b].to(torch.bfloat16) @ hw.T).float(), -1)
        kl = -(tp[a:b] * q.gather(1, ti[a:b])).sum(-1).mean()
        ce = F.nll_loss(q, ti[a:b, 0])
        total = total + (kl + ce) * (b - a) / (T - 2)
        acc += int((q.argmax(-1) == ti[a:b, 0]).sum())
    return total, acc / (T - 2), x


@torch.no_grad()
def evaluate(mtp, trunk, held, depth: int = 2) -> dict:
    hits, n = [0] * depth, [0] * depth
    for ids in held:
        ids = torch.tensor(ids, device="cuda")
        h, lg = trunk.run(ids)
        T = ids.numel()
        top = lg.argmax(-1)                                     # the trunk's top-1 at every position
        x = h[:T - 2].float()
        nxt = ids[1:T - 1]
        pos = torch.arange(T - 2, device="cuda")
        for d in range(depth):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                x = mtp.step(x, trunk.embed(nxt).float(), pos)
            pred = trunk.head(x).argmax(-1)
            # depth d drafts token t+2+d; the trunk's top-1 for it was read at position t+1+d
            m = T - 2 - d
            ok = pred[:m] == top[1 + d:1 + d + m]
            hits[d] += int(ok.sum())
            n[d] += m
            nxt = pred
    return {f"acc_d{d + 1}": round(hits[d] / max(n[d], 1), 4) for d in range(depth)}


def cmd_train(a) -> None:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    trunk = Trunk(a.model, a.chunk + 64)
    mtp = FlintMTP(trunk.cfg, trunk.W).cuda()
    train = []
    for p in a.corpus.split(","):
        train += chunks(p, tok, a.chunk)
    random.Random(0).shuffle(train)
    held = []
    for p in a.heldout.split(","):
        held += chunks(p, tok, 1024, a.eval_chunks)
    steps = min(len(train), a.tokens // a.chunk)
    opt = torch.optim.AdamW(mtp.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    log = {"config": vars(a), "steps": steps, "train_chunks": len(train), "eval": []}
    e0 = evaluate(mtp, trunk, held)
    print(json.dumps({"step": 0, **e0}), flush=True)
    log["eval"].append({"step": 0, **e0})
    t0 = time.time()
    run_acc = 0.0
    for s in range(steps):
        ids = torch.tensor(train[s], device="cuda")
        h, lg = trunk.run(ids)
        loss, acc, _ = loss_of(mtp, trunk, h, lg, ids)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mtp.parameters(), 1.0)
        opt.step()
        sched.step()
        run_acc = 0.98 * run_acc + 0.02 * acc if s else acc
        if (s + 1) % a.log_every == 0:
            print(json.dumps({"step": s + 1, "tokens": (s + 1) * a.chunk, "loss": round(float(loss), 4),
                              "train_acc": round(run_acc, 4), "tok_s": round((s + 1) * a.chunk / (time.time() - t0))}),
                  flush=True)
        if (s + 1) % a.eval_every == 0 or s + 1 == steps:
            e = evaluate(mtp, trunk, held)
            print(json.dumps({"step": s + 1, **e}), flush=True)
            log["eval"].append({"step": s + 1, "tokens": (s + 1) * a.chunk, **e})
            torch.save({"state": mtp.state_dict(), "log": log}, a.out)
    json.dump(log, open(a.out + ".json", "w"), indent=1)


def cmd_eval(a) -> None:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    trunk = Trunk(a.model, 1024 + 64)
    mtp = FlintMTP(trunk.cfg, trunk.W).cuda()
    mtp.load_state_dict(torch.load(a.head)["state"])
    held = []
    for p in a.heldout.split(","):
        held += chunks(p, tok, 1024, a.eval_chunks)
    print(json.dumps(evaluate(mtp, trunk, held, depth=3)))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--model", required=True)
    t.add_argument("--corpus", required=True)
    t.add_argument("--heldout", required=True)
    t.add_argument("--tokens", type=int, default=4_000_000)
    t.add_argument("--chunk", type=int, default=2048)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--eval-chunks", type=int, default=8)
    t.add_argument("--eval-every", type=int, default=200)
    t.add_argument("--log-every", type=int, default=25)
    t.add_argument("--out", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--model", required=True)
    e.add_argument("--head", required=True)
    e.add_argument("--heldout", required=True)
    e.add_argument("--eval-chunks", type=int, default=8)
    a = ap.parse_args()
    {"train": cmd_train, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
