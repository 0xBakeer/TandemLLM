"""The resident prefix's guest path on Flash-Next, in one process (the P0 follow-up of 2026-10-12).

    PYTHONPATH=<repo>:~/fn-ref/site python tools/fn_guest_check.py --pack ~/flashnext-packs/3.05

The server answered a 6,000-token prompt differently from a cold prefill when it came after a
1,000-token guest of a 40,000-token resident conversation (the three prompts share a head). Here the
same sequence runs through engine/cache.py `prefill` with a resident prefix as the server builds it
(chunk 4,096, a stash, a small park and pin): A (40,000), B (1,000, a guest, then a few decode steps
as its answer), C (6,000). C's last logits are compared with a cold prefill's, and so is every section
of the state and every view row below C's length, so the first buffer that differs is named.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import cache  # noqa: E402
from engine.models.flashnext.serving import build  # noqa: E402
from tools.fn_fit_run import prompt  # noqa: E402

GIB = 1 << 30


def ids_of(tok, pack, n):
    text, _ = prompt(pack, n)
    chat = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    return tok(chat, add_special_tokens=False)["input_ids"]


@torch.no_grad()
def decode(eng, lg, n, steps):
    t = int(lg.view(-1).argmax())
    for i in range(steps):
        lg = eng.forward(torch.tensor([t], device="cuda"), n + i)
        t = int(lg.view(-1).argmax())


def state_of(eng, n):
    secs = {k: v.clone() for k, v in eng.cache_state().sections().items()}
    views = {name: t.narrow(ax, 0, n).clone() for name, t, ax in eng.cache_state().views()}
    return secs, views


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--steps", type=int, default=64, help="the guest's answer, decode steps")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    pack = os.path.expanduser(a.pack)
    tok = AutoTokenizer.from_pretrained(pack)
    eng, _, _ = build(pack, 65536, tok, spec=False, log=lambda s: None)
    A, B, C = (ids_of(tok, pack, n) for n in (40000, 1000, 6000))
    # the truth: C cold
    eng.reset()
    cold = cache.prefill(eng, None, C, "cuda", chunk=4096)[0].clone()
    cs, cv = state_of(eng, len(C))
    res = cache.ResidentPrefix(int(2.5 * GIB), 4096, stash_bytes=GIB, guest_max=8192, pin_bytes=GIB // 4,
                               park_bytes=GIB // 4, end=True)
    store = cache.StateStore(2 * GIB, chunk=4096, max_entry_bytes=GIB, per_prefill=1)
    kw = dict(chunk=4096, resident=res, store=store, checkpoint=True)
    eng.reset()
    lg = cache.prefill(eng, None, A, "cuda", reach=len(A) + 120, **kw)[0]
    decode(eng, lg, len(A), 8)
    lg = cache.prefill(eng, None, B, "cuda", reach=len(B) + 120, **kw)[0]
    print(f"[guest] B a guest: stash {res._stash is not None and (res._stash['start'], res._stash['rows'])}", flush=True)
    decode(eng, lg, len(B), a.steps)
    info = {}
    warm = cache.prefill(eng, None, C, "cuda", reach=len(C) + 120, info=info, **kw)
    lw, reused = warm[0], warm[1]
    ws, wv = state_of(eng, len(C))
    print(f"[guest] C resumed at {reused} ({info.get('kind')}); logits max diff {float((lw - cold).abs().max()):.4g}, "
          f"argmax same {int(lw.argmax()) == int(cold.argmax())}", flush=True)
    for k in cs:
        if not torch.equal(cs[k], ws[k]):
            print(f"[guest] section {k} differs", flush=True)
    for k in cv:
        d = (cv[k].float() - wv[k].float()).abs()
        if d.max() > 0:
            ax = 1 if k.startswith(("k.", "v.")) else 0
            rows = d.amax(dim=tuple(i for i in range(d.dim()) if i != ax)).nonzero().view(-1)
            print(f"[guest] view {k}: rows {int(rows[0])}..{int(rows[-1])} differ ({rows.numel()} rows)", flush=True)


if __name__ == "__main__":
    main()
