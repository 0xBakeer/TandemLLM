"""One adapter class per backend kind (proxy/adapters/base.py has the interface)."""

from __future__ import annotations

from proxy.adapters.base import Adapter, Stamps, ENDPOINTS  # noqa: F401
from proxy.adapters.deepseek import DeepSeek
from proxy.adapters.generic import Generic, TandemLLM
from proxy.adapters.openai import OpenAI

KINDS = {"generic": Generic, "tandemllm": TandemLLM, "openai": OpenAI, "deepseek": DeepSeek}


def make(backend: dict, log=None) -> Adapter:
    cls = KINDS.get(backend.get("kind"), Generic)
    if cls is DeepSeek:
        return cls(backend, log=log)
    return cls(backend)
