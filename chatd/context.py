"""The Context panel's report (CONTRACT section 4, `ContextReport`; SRV-349, Chat 4).

Built from what the server says, never from a model name: the window from the model's card (or the
engine's own `max_len`), the split from chatd's counts and the last usage frame, the cache from the
engine's peek before a send (when the server has one) or from the last request's `cache_source`,
the live figures from the live row of the run, the memory from `/v1/dashboard/system`. A server
that says less gives `null`s and zeros, never invented figures. No message text goes in a report.
"""

from __future__ import annotations

FLOOR_BYTES = 15 * 1024 ** 3          # the engine's admission floor (docs/operations.md)
COMPACT_SHARE = 0.8


def estimate_tokens(text: str) -> int:
    """A count for text no tokenizer has seen yet: about 4 characters a token."""
    return (len(text) + 3) // 4 if text else 0


def report(*, model: dict, settings: dict, branch_tokens: int, files_tokens: int,
           stats: dict | None, run: dict | None, row: dict | None, facts: dict,
           peek: dict | None, harness: dict | None, split: dict | None = None) -> dict:
    """`model`: the chat's model as `service.model_info` gives it. `branch_tokens`: the branch as
    the next request carries it (the last request's prompt plus its answer, plus what came since).
    `stats`: the last finished answer's TurnStats; `run`/`row`: the run in flight and its live row.
    `split`: the last usage frame's `tandem.parts` (SRV-349): the system prompt's and the tool
    schemas' tokens as the engine counted them, which come out of the conversation's share."""
    window = int(model.get("max_len") or 0)
    skills = 0
    if row and row.get("hidden_tokens"):
        skills = int(row["hidden_tokens"])
    elif stats and stats.get("hidden_tokens"):
        skills = int(stats["hidden_tokens"])
    thinking = int((run or {}).get("thinking", {}).get("used") or 0) if run else 0
    messages = max(0, branch_tokens - files_tokens)
    if row and row.get("prompt_tokens"):
        messages = max(0, int(row["prompt_tokens"]) - files_tokens)
    reserved = int(model.get("max_output") or 0)
    system = int((split or {}).get("system") or 0)
    tools = int((split or {}).get("tools") or 0)
    messages = max(0, messages - system - tools)
    parts = [
        {"kind": "system", "label": "system prompt", "tokens": system},
        {"kind": "tools", "label": "tool schemas", "tokens": tools},
        {"kind": "skills", "label": "skills and packages", "tokens": skills},
        {"kind": "files", "label": "files", "tokens": files_tokens},
        {"kind": "messages", "label": "conversation", "tokens": messages},
        {"kind": "thinking", "label": "thinking, this turn", "tokens": thinking},
        {"kind": "reserved", "label": "kept for the answer", "tokens": reserved},
    ]
    used = sum(p["tokens"] for p in parts if p["kind"] != "reserved")

    if row and row.get("prompt_tokens") is not None:
        cached = int(row.get("cached_tokens") or 0)
        cache = {"warm": cached, "cold": max(0, int(row["prompt_tokens"]) - cached),
                 "cold_eta_ms": None, "source": row.get("cache_source") or "none",
                 "slot": str(row["lane"]) if row.get("lane") is not None else None}
    elif peek:
        cache = {"warm": int(peek.get("cached_tokens") or 0),
                 "cold": int(peek.get("reread_tokens") or 0),
                 "cold_eta_ms": peek.get("reread_eta_ms"),
                 "source": peek.get("tier") or "none", "slot": None}
    elif stats:
        cached = int(stats.get("cached_tokens") or 0)
        cache = {"warm": cached, "cold": max(0, int(stats.get("prompt_tokens") or 0) - cached),
                 "cold_eta_ms": None, "source": stats.get("cache_source") or "unknown",
                 "slot": None}
    else:
        cache = {"warm": 0, "cold": used, "cold_eta_ms": None, "source": "none", "slot": None}

    mem = facts.get("memory") or {}
    ours = bool(facts.get("engine"))
    return {
        "window": window, "used": used, "parts": parts,
        "compact_at": int(window * COMPACT_SHARE),
        "cache": cache,
        "harness": harness,
        "split": split,
        "memory": {"kv_bytes": None, "resident_bytes": None, "sandbox_disk_bytes": None,
                   "available_bytes": mem.get("unified_available_bytes"),
                   "floor_bytes": FLOOR_BYTES if ours else None, "sandbox_bytes": None},
        # the profile the engine serves (flashnext-3.05, exl3-4.5), else the chat's conductor profile
        "model": {"name": model.get("id") or "",
                  "profile": model.get("profile") or settings.get("profile"),
                  "weights": model.get("weights") or "", "drafter": model.get("drafter"),
                  "max_len": window, "effort": settings.get("effort") or "auto",
                  "thinking": bool(settings.get("thinking", True)),
                  "backend": model.get("backend") or "engine"},
    }
