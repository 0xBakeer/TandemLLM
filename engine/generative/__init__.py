"""Small generative models inside the engine: the slot (P1 of "One engine for all models", 2026-10-02).

The decision models (engine/decision) answer typed questions in one forward. The models here
write text: a few dozen tokens, greedy, with their own tokenizer and their own KV cache. The first
is the narrator, MiniCPM5-2B, which writes the one-line summaries of TandemLLM's states
(server/conductor/narrator.py). The pieces: the interface and the role contract (base.py), the
Llama-architecture forward in plain PyTorch (llama.py), a checkpoint directory as a model
(model.py), the slot that loads it through the residency planner and runs it inside a time budget
(slot.py), and a fake for the tests (fake.py).

Nothing here touches Qwen: no tensor, stream or allocator pool is shared (the slot uses the
decision models' Device and Placement), and with `QWEN38_NARRATOR_MODEL` empty the package is never
imported.
"""

from __future__ import annotations

FAKE_PREFIXES = ("fake-",)


def model(name: str, path: str, *, allow_fake: bool = False):
    """The model `QWEN38_NARRATOR_MODEL` names. A fake name only in a test or with the fake
    engine; an unknown name stops the start."""
    if name.startswith(FAKE_PREFIXES):
        if not allow_fake:
            raise ValueError(f"{name!r} is a test double; it is served only by a fake engine")
        from engine.generative.fake import FakeGenerative
        return FakeGenerative(name)
    from engine.generative.model import NAMES, LlamaModel
    if name not in NAMES:
        raise ValueError(f"unknown generative model {name!r}; this engine serves "
                         f"{', '.join(NAMES)}")
    return LlamaModel(name, path)
