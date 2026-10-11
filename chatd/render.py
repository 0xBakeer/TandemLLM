"""The request of one model call, the same bytes for the same conversation (D5, SRV-345).

The engine finds a conversation's cache by the longest shared token prefix, so a request whose head
changes re-reads the whole conversation (the lesson of SRV-209: Open WebUI's working directory in
the system prompt did that to every turn). So the renderer puts nothing in the head that changes
between turns -- no time, no ids, no counts -- builds every object with its keys in one fixed
order, and writes compact JSON. What is new in a turn (the next user message, a steer from the
queue) only ever appends. `tests/test_chatd_render.py` pins it: the same conversation renders to the
same bytes across calls and across a restart (a fresh store read).

    messages        the branch, root to head: user text (text attachments as a fenced block after
                    it, images as `image_url` parts when the model reads images), assistant text with
                    its reasoning as `reasoning_content` (the engine's template decides what to keep);
                    notices (role `system` in the tree) are the UI's and are left out
    effort          `reasoning_effort` only when the chat picked one (`auto` lets the harness decide)
    thinking off    `chat_template_kwargs.enable_thinking: false` (read by our engine, vLLM and
                    SGLang alike)
    continue        an interrupted answer resumes as the last assistant message with
                    `continue_final_message: true` and no generation prompt (SRV-283's Continue)
"""

from __future__ import annotations

import base64
import json

SESSION_PREFIX = "chat-"


def session_id(conversation_id: str) -> str:
    """`x-session-id`: the router's session key becomes `c:chat-<id>` (server/conductor/router.py),
    the park, the spill and the live row's conversation id follow it. A branch shares its prefix up
    to the edited message, so one key is right for all of a chat's branches."""
    return SESSION_PREFIX + conversation_id


def text_of(parts: list[dict], kind: str = "text") -> str:
    return "".join(p.get("text") or "" for p in parts if p.get("type") == kind)


def message(m: dict, vision: bool, blob=None) -> dict | None:
    """One branch message as an OpenAI message; None for what the model does not see."""
    role = m.get("role")
    parts = m.get("parts") or []
    if role == "user":
        text = text_of(parts)
        images = []
        for a in m.get("attachments") or []:
            data = blob(a) if blob is not None else None
            if data is None:
                continue
            if a.get("kind") == "image":
                if vision:
                    images.append({"type": "image_url", "image_url": {
                        "url": f"data:{a.get('mime') or 'image/png'};base64,"
                               + base64.b64encode(data).decode()}})
            elif a.get("kind") == "text":
                body = data.decode("utf-8", "replace")
                text += f"\n\n```{a.get('name') or 'file'}\n{body}\n```"
        if images:
            return {"role": "user", "content": [{"type": "text", "text": text}, *images]}
        return {"role": "user", "content": text}
    if role == "assistant":
        text = text_of(parts)
        thinking = text_of(parts, "thinking")
        if not text and not thinking:
            return None
        out = {"role": "assistant", "content": text}
        if thinking:
            out["reasoning_content"] = thinking
        return out
    return None


def body(model: str, settings: dict, branch: list[dict], *, system: str | None = None,
         vision: bool = False, blob=None, continue_final: bool = False,
         max_tokens: int | None = None) -> dict:
    msgs: list[dict] = []
    if system:
        msgs.append({"role": "system", "content": system})
    for m in branch:
        x = message(m, vision, blob)
        if x is not None:
            msgs.append(x)
    out: dict = {"model": model, "messages": msgs, "stream": True,
                 "stream_options": {"include_usage": True}}
    effort = settings.get("effort") or "auto"
    if not settings.get("thinking", True):
        out["chat_template_kwargs"] = {"enable_thinking": False}
    elif effort != "auto":
        out["reasoning_effort"] = effort
    if max_tokens:
        out["max_tokens"] = int(max_tokens)
    if continue_final and msgs and msgs[-1]["role"] == "assistant":
        out["continue_final_message"] = True
        out["add_generation_prompt"] = False
    return out


def encode(obj: dict) -> bytes:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode()


def headers(conversation_id: str, settings: dict) -> dict:
    """The call's headers, also fixed: the session id, the client kind (the dashboard's: the
    conductor maps it to its chat profile), and a profile only when the chat named one."""
    h = {"x-session-id": session_id(conversation_id), "X-QSE-Client": "dashboard"}
    if settings.get("profile"):
        h["X-Tandem-Profile"] = str(settings["profile"])
    return h
