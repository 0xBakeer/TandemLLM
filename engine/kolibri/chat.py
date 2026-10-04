"""Kolibri-1's tokenizer and chat template, as the server needs them.

The tokenizer and the template are the release's own files (`tokenizer.json`,
`tokenizer_config.json` with its `chat_template`), loaded with `transformers.AutoTokenizer`;
nothing here re-implements the template. What this module adds is the handful of places where an
OpenAI-shaped request does not fit the template as written:

* The template reads `message.content` as a string (`content.startswith(...)`, `+`), so the
  OpenAI list form (`[{"type": "text", "text": ...}]`) is joined to its text. Kolibri-1 reads no
  images; an image part is dropped and counted in `dropped_parts`.
* `developer` (the newer OpenAI name for the system role) is rendered as `system`; the template
  would otherwise drop it without a word.
* The template decides thinking from `reasoning_effort` FIRST and only looks at `enable_thinking`
  when no effort is given. The server always passes its default effort, so a client's
  `enable_thinking: false` would be ignored. Here `enable_thinking: false` sets the effort to
  `none`, which is what the template means by thinking off.
* Effort words: Kolibri's template knows none, minimal, low, medium, high, xhigh and max (xhigh and
  max render as high, minimal as low). Anything else is passed as given and renders as high.

THINKING IS OPENED BY THE MODEL. With thinking on, the generation prompt ends with
`<|im_start|>assistant\\n` and the model writes `<think>` (id 127907) itself as its first token;
with thinking off the prompt ends `<think>\\n\\n</think>\\n\\n`. Qwen's template opens the block in
the prompt instead, and the server's reasoning split assumed that. `thinking_mode(prompt_text)`
says which case a rendered prompt is: "model" (the model will open the block: the server's
`in_think = "model"`, server/stream.py) or False.

Tool calls are Hermes JSON: `<tool_call>\\n{"name": ..., "arguments": {...}}\\n</tool_call>`, which
`server/toolcall.py` reads as its third tier; tool results go back as `role: tool` messages, which
the template wraps in `<tool_response>` inside a user turn.

Stop ids: `<|im_end|>` 127906 (the config's eos) and `<|endoftext|>` 127901 (generation_config).
"""

from __future__ import annotations

import json
import os

IM_END = 127906
END_OF_TEXT = 127901
THINK_OPEN_ID = 127907
THINK_CLOSE_ID = 127908
TOOL_CALL_ID = 127909
EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
GEN_PROMPT = "<|im_start|>assistant\n"


def _text(content) -> tuple[str, int]:
    """OpenAI content (a string, None, or a list of parts) as the template's string, and how many
    parts were dropped (anything that is not text)."""
    if content is None:
        return "", 0
    if isinstance(content, str):
        return content, 0
    if isinstance(content, list):
        out, dropped = [], 0
        for p in content:
            if isinstance(p, str):
                out.append(p)
            elif isinstance(p, dict) and p.get("type") in ("text", "input_text", "output_text"):
                out.append(str(p.get("text") or ""))
            else:
                dropped += 1
        return "".join(out), dropped
    return str(content), 0


def normalize_messages(messages: list) -> tuple[list, int]:
    """The messages as the template can read them, and the number of dropped content parts."""
    out, dropped = [], 0
    for m in messages:
        if not isinstance(m, dict):
            continue
        m = dict(m)
        role = m.get("role") or "user"
        if role == "developer":
            role = "system"
        m["role"] = role
        m["content"], d = _text(m.get("content"))
        dropped += d
        for key in ("reasoning", "reasoning_content"):
            if key in m and m[key] is not None and not isinstance(m[key], str):
                m[key], _ = _text(m[key])
        out.append(m)
    return out, dropped


def template_kwargs(kwargs: dict) -> dict:
    """`enable_thinking: false` wins over a default effort; an effort word passes through."""
    kw = dict(kwargs)
    if kw.get("enable_thinking") is False:
        kw["reasoning_effort"] = "none"
    eff = kw.get("reasoning_effort")
    if isinstance(eff, str):
        kw["reasoning_effort"] = eff.strip().lower()
    return kw


def thinking_mode(prompt_text: str):
    """"model" when the rendered prompt leaves the model to open `<think>` itself, else False."""
    return "model" if prompt_text.endswith(GEN_PROMPT) else False


class KolibriTokenizer:
    """`transformers` tokenizer with the request fixes above in `apply_chat_template`; every other
    attribute is the wrapped tokenizer's."""

    def __init__(self, tok):
        self._tok = tok
        self.dropped_parts = 0

    def __getattr__(self, name):
        return getattr(self._tok, name)

    def __call__(self, *a, **kw):
        return self._tok(*a, **kw)

    def __len__(self) -> int:
        # the grammar's vocabulary (tool_choice required / named) walks every id
        return len(self._tok)

    def apply_chat_template(self, messages, *a, **kw):
        msgs, dropped = normalize_messages(messages)
        self.dropped_parts = dropped
        return self._tok.apply_chat_template(msgs, *a, **template_kwargs(kw))

    def render(self, messages, tools=None, **kw) -> str:
        """The prompt as text (for tests and the log)."""
        msgs, _ = normalize_messages(messages)
        return self._tok.apply_chat_template(msgs, tools=tools, tokenize=False,
                                             add_generation_prompt=True,
                                             **template_kwargs(kw))


def tokenizer_dir(*candidates: str) -> str:
    """The first directory that has both `tokenizer.json` and `tokenizer_config.json`."""
    for d in candidates:
        if d and os.path.isfile(os.path.join(d, "tokenizer.json")) \
                and os.path.isfile(os.path.join(d, "tokenizer_config.json")):
            return d
    raise FileNotFoundError("no directory with tokenizer.json and tokenizer_config.json among "
                            + ", ".join(c for c in candidates if c))


def load_tokenizer(path: str) -> KolibriTokenizer:
    from transformers import AutoTokenizer
    return KolibriTokenizer(AutoTokenizer.from_pretrained(path))


def stop_ids(path: str) -> list[int]:
    """`<|im_end|>` and whatever generation_config.json lists (127901 `<|endoftext|>`)."""
    out = [IM_END]
    gc = os.path.join(path, "generation_config.json")
    if os.path.isfile(gc):
        e = json.load(open(gc)).get("eos_token_id")
        for t in (e if isinstance(e, list) else [e]):
            if isinstance(t, int) and t not in out:
                out.append(t)
    if END_OF_TEXT not in out:
        out.append(END_OF_TEXT)
    return out


def sampling_defaults(path: str) -> dict:
    """The release's generation_config sampling (temperature 1.0, top_p 0.97, top_k 128)."""
    gc = os.path.join(path, "generation_config.json")
    if not os.path.isfile(gc):
        return {}
    g = json.load(open(gc))
    return {k: g[k] for k in ("temperature", "top_p", "top_k") if k in g}
