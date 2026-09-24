"""A fake engine for the dashboard's end-to-end tests (VIS-18): the real server, no model.

    QSE_ADMIN_TOKEN=$(python3 -c 'print("e2e-" + "0" * 40)') \\
    python server/app.py --fake-engine --port 8011 --served-model qwen38-spark-engine \\
        --usage-ledger /tmp/qse-e2e/ledger.sqlite3 --dashboard-dir ../qwen38-spark-engine-dash/dashboard/dist

Everything a browser can reach is the real code -- the handler, the streaming, usage and timings,
the usage ledger, the dashboard API, the log stream, /metrics, the access control, the static
files. Only `generate_stream` and the tokenizer are replaced: a character-level tokenizer and a
deterministic writer that answers every prompt with the same text for the same prompt, at
`--fake-tps` tokens a second in blocks of four, and fills the loop's own counters (blocks, the
first-miss histogram, the prefill's reused/forwarded) so every number the dashboard shows is
populated. It needs CPU torch and nothing else: no weights, no triton, no transformers, no GPU.
It refuses the production ledger (it is a test run).

Words in the last user message steer it, for the e2e's error and edge states:

    FAKE_ERROR    the generation raises after five tokens (finish_reason error)
    FAKE_SLOW     ten tokens a second
    FAKE_LONG     an answer about ten times as long
    a message containing "tool" with `tools` in the request: one call to the first tool

With thinking on (the chat template's default), the answer is preceded by a short reasoning block.
A second request that repeats a prompt's first 64+ characters reports them as a prefix-cache hit.
"""

from __future__ import annotations

import hashlib
import re
import time
import types

import torch

THINK_OPEN, THINK_CLOSE, EOS = 0x110001, 0x110002, 0x110003
SPECIAL = {"<think>": THINK_OPEN, "</think>": THINK_CLOSE, "<|im_end|>": EOS}
WORDS = ("the engine reads one sequence at a time and writes several tokens a block when the "
         "draft agrees with the target so the speed depends on the text more than on the prompt "
         "length a quotation runs at a hundred tokens a second and fresh prose at forty").split()


class FakeTokenizer:
    """One token per character; `<think>`, `</think>` and `<|im_end|>` are single special ids."""

    unk_token_id = -1
    eos_token_id = EOS

    def convert_tokens_to_ids(self, text):
        return SPECIAL.get(text, -1)

    def encode_text(self, text: str) -> list[int]:
        out, i = [], 0
        while i < len(text):
            for s, t in SPECIAL.items():
                if text.startswith(s, i):
                    out.append(t)
                    i += len(s)
                    break
            else:
                out.append(ord(text[i]))
                i += 1
        return out

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        ids = self.encode_text(text or "")
        if return_tensors == "pt":
            ids = torch.tensor([ids])
        return types.SimpleNamespace(input_ids=ids)

    def decode(self, seq, skip_special_tokens=True):
        inv = {v: k for k, v in SPECIAL.items()}
        out = []
        for t in seq:
            t = int(t)
            if t == EOS:
                if not skip_special_tokens:
                    out.append("<|im_end|>")
            elif t in inv:
                out.append(inv[t])
            else:
                out.append(chr(t))
        return "".join(out)

    def apply_chat_template(self, messages, add_generation_prompt=True, enable_thinking=True,
                            tools=None, return_tensors=None, return_dict=False, **kw):
        parts = []
        if tools:
            names = [((t.get("function") or {}).get("name") or "?") for t in tools
                     if isinstance(t, dict)]
            parts.append(f"<|im_start|>system\n# Tools: {', '.join(names)}<|im_end|>\n")
        for m in messages:
            c = m.get("content")
            if isinstance(c, list):
                c = "".join(p.get("text", "") for p in c if isinstance(p, dict))
            parts.append(f"<|im_start|>{m.get('role', 'user')}\n{c or ''}<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n" + ("<think>\n" if enable_thinking
                                                        else "<think>\n\n</think>\n\n"))
        ids = self.encode_text("".join(parts))
        return {"input_ids": torch.tensor([ids])}


class _FakeCfg:
    vocab_size = EOS + 1
    attention_layers: list = []
    num_key_value_heads = 0
    head_dim = 0


class _FakeState:
    S = torch.zeros(1)
    conv = torch.zeros(1)


class FakeEngine:
    """What `server/app.py` reads off an engine outside the decode loop."""

    cfg = _FakeCfg()
    state = _FakeState()
    kv = types.SimpleNamespace(length=0)
    w = types.SimpleNamespace(nvfp4_source="", fp8_head_source=None)


def _reply(user: str, tools: list[str], think: bool) -> tuple[str, str]:
    """(reasoning, answer) for a prompt, the same every time."""
    h = int(hashlib.sha256(user.encode()).hexdigest(), 16)
    words = [WORDS[(h >> (3 * i)) % len(WORDS)] for i in range(24 + h % 40)]
    if "FAKE_LONG" in user:
        words = words * 10
    answer = " ".join(words).capitalize() + "."
    if tools and "tool" in user.lower():
        answer = (f"<tool_call>\n<function={tools[0]}>\n<parameter=query>\n{user[:40]}\n"
                  f"</parameter>\n</function>\n</tool_call>")
    reasoning = f"The question has {len(user)} characters; answer it plainly.\n" if think else ""
    return reasoning, answer


def load(app, a) -> None:
    """Put the fake in `app.STATE` and replace `app.generate_stream`. Called from `main()`."""
    tok = FakeTokenizer()
    tps = max(1.0, float(getattr(a, "fake_tps", 200.0)))
    seen: list[list[int]] = []

    def generate_stream(prompt, max_new, eos, think=None, conv_id=None, deadline=None, pen=None,
                        pstop=None, sampler=None):
        ctx = [int(t) for t in prompt.tolist()]
        text = tok.decode(ctx, skip_special_tokens=False)
        users = re.findall(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", text, flags=re.S)
        user = users[-1] if users else text
        tools = re.findall(r"# Tools: (.*?)<\|im_end\|>", text)
        tools = [t.strip() for t in tools[0].split(",")] if tools else []
        opened = text.rstrip().endswith("<think>")
        # the prefill: a prefix hit when an earlier prompt shares 64+ tokens, on a 64 grid
        best = max((next((i for i, (x, y) in enumerate(zip(p, ctx)) if x != y), min(len(p), len(ctx)))
                    for p in seen[-32:]), default=0)
        reused = (min(best, len(ctx) - 1) // 64) * 64
        t0 = time.perf_counter()
        time.sleep(min(2.0, 0.02 + 0.0001 * (len(ctx) - reused)))
        app.STATE["last_prefill"] = {"reused": reused, "forwarded": len(ctx) - reused,
                                     "ms": (time.perf_counter() - t0) * 1e3,
                                     "kind": "prefix" if reused else None}
        seen.append(ctx)
        reasoning, answer = _reply(user, tools, opened)
        ids = tok.encode_text(reasoning) + ([THINK_CLOSE] + tok.encode_text("\n\n") if opened
                                            else []) + tok.encode_text(answer) + [EOS]
        rate = 10.0 if "FAKE_SLOW" in user else tps
        bs = app.STATE["blocks"] = app.BlockStats()
        app.STATE["last_ctx"] = ctx
        n = 0
        for i in range(0, len(ids), 4):
            block = ids[i:i + 4]
            if n:
                time.sleep(len(block) / rate)
                bs.block(15, len(block) - 1)
            for t in block:
                if "FAKE_ERROR" in user and n == 5:
                    raise RuntimeError("the fake engine failed on purpose (FAKE_ERROR)")
                if n == 0:
                    bs.first()
                ctx.append(t)
                n += 1
                yield t
                if t in eos or n >= max_new:
                    return
            if deadline is not None and deadline.expired():
                return

    app.STATE.update(
        engine=FakeEngine(), tok=tok, drafter=None, k=15, device="cpu", tree=False,
        sampled_tree=False, relax=app.Relax(1.0, 1), model=a.served_model,
        started=int(time.time()), verbose=a.verbose, max_len=int(a.max_len),
        default_max_tokens=int(a.default_max_tokens), reasoning_format=a.reasoning_format,
        request_timeout=float(a.request_timeout), max_queue=int(a.max_queue),
        queue_timeout=float(a.queue_timeout), draining=False, cfg_eos=EOS,
        think_budget=a.think_budget, think_stall=False, reasoning_effort=a.reasoning_effort,
        pen_spec=app.PenaltySpec(a.rep_penalty, a.presence_penalty, a.frequency_penalty,
                                 a.no_repeat_ngram),
        temperature=a.temperature, top_p=a.top_p, top_k=a.top_k, pattern_stop=None,
        state_store=None, session_cache=False, prefix_cache=False, prefix_chunk=0,
        response_cache=None, suffix_store=None, suffix_scope="all",
        usage_default=(a.usage_default == "on"))
    app.generate_stream = generate_stream
    print(f"[server] FAKE ENGINE: no model, a deterministic character writer at {tps:g} tok/s "
          f"(server/fake_engine.py) -- for tests only", flush=True)
