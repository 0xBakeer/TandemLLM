"""DeepSeek's API (OpenAI-shaped chat completions).

Its usage carries the cache split as `prompt_cache_hit_tokens` and `prompt_cache_miss_tokens`
(verified on api-docs.deepseek.com on 2026-10-04): the hits fill `cached_tokens`. The misses are
checked against `prompt_tokens - cached_tokens`; a mismatch is one log line, never a correction.
Reasoning arrives as `delta.reasoning_content` on `deepseek-reasoner`, and thinking is switched by
the model pick (`deepseek-chat` off, `deepseek-reasoner` on). It cannot prefill an assistant
message except in its beta prefix mode, so `continue_final_message` is off.
"""

from __future__ import annotations

from proxy.adapters.base import CHAT, COMPLETIONS, Adapter, _int


class DeepSeek(Adapter):
    kind = "deepseek"
    cloud = True

    def __init__(self, backend: dict, log=None):
        super().__init__(backend)
        self.log = log or (lambda msg: None)

    def routes(self) -> tuple[str, ...]:
        return (CHAT, COMPLETIONS)

    def capabilities(self) -> dict:
        return {"continue_final_message": False, "choice": "text",
                "reasoning_field": "reasoning_content",
                "cache_counts": ["prompt_cache_hit_tokens", "prompt_cache_miss_tokens"],
                "thinking_switch": "model_pick"}

    def read_usage(self, obj, path: str = CHAT) -> dict:
        out = super().read_usage(obj, path)
        if not isinstance(obj, dict):
            return out
        hit = _int(obj.get("prompt_cache_hit_tokens"))
        miss = _int(obj.get("prompt_cache_miss_tokens"))
        if hit is not None:
            out["cached_tokens"] = hit
        if hit is not None and miss is not None and out["prompt_tokens"] is not None and \
                out["prompt_tokens"] - hit != miss:
            self.log(f"[proxy] {self.backend.get('id')}: prompt_cache_miss_tokens {miss} != "
                     f"prompt_tokens {out['prompt_tokens']} - hits {hit}; kept as reported")
        return out
