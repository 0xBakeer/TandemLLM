"""The request of one model call, the same bytes for the same conversation (D5, SRV-345).

The engine finds a conversation's cache by the longest shared token prefix, so a request whose head
changes re-reads the whole conversation (the lesson of SRV-209: Open WebUI's working directory in
the system prompt did that to every turn). So the renderer puts nothing in the head that changes
between turns -- no time, no ids, no counts -- builds every object with its keys in one fixed
order, and writes compact JSON. What is new in a turn (the next user message, a steer from the
queue) only ever appends. `tests/test_chatd_render.py` pins it: the same conversation renders to the
same bytes across calls and across a restart (a fresh store read).

    messages        the branch, root to head: user text (text attachments as a fenced block after
                    it, one line with its workspace path for one over the budget, chatd/attach.py;
                    images as `image_url` parts when the model reads images), assistant text with
                    its reasoning as `reasoning_content` (the engine's template decides what to keep);
                    notices (role `system` in the tree) are the UI's and are left out
    tool images     in the tool message that returned them, the newest one only: an older
                    screenshot is a line saying a newer one follows (TOOL_IMAGES_KEPT)
    effort          `reasoning_effort` only when the chat picked one (`auto` lets the harness decide)
    thinking off    `chat_template_kwargs.enable_thinking: false` (read by our engine, vLLM and
                    SGLang alike)
    continue        an interrupted answer resumes as the last assistant message with
                    `continue_final_message: true` and no generation prompt (SRV-283's Continue)
"""

from __future__ import annotations

import base64
import json

from chatd import attach

SESSION_PREFIX = "chat-"


def session_id(conversation_id: str) -> str:
    """`x-session-id`: the router's session key becomes `c:chat-<id>` (server/conductor/router.py),
    the park, the spill and the live row's conversation id follow it. A branch shares its prefix up
    to the edited message, so one key is right for all of a chat's branches."""
    return SESSION_PREFIX + conversation_id


def text_of(parts: list[dict], kind: str = "text") -> str:
    return "".join(p.get("text") or "" for p in parts if p.get("type") == kind)


def inlined(a: dict) -> bool:
    """A text attachment the prompt carries whole: as placed when its message was sent (SRV-487);
    one from before that, while it stays within INLINE_MAX_TOKENS."""
    if a.get("inline") is not None:
        return bool(a["inline"])
    return int(a.get("tokens") or 0) <= attach.INLINE_MAX_TOKENS


def _user(m: dict, vision: bool, blob=None) -> dict:
    parts = m.get("parts") or []
    text = text_of(parts)
    todos = text_of(parts, "todos")                 # the open todo list at the run's start
    if todos:
        text += "\n\n" + todos
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
            if not inlined(a):
                text += attach.line(a)
                continue
            text += f"\n\n```{a.get('name') or 'file'}\n{body}\n```"
    if images:
        return {"role": "user", "content": [{"type": "text", "text": text}, *images]}
    return {"role": "user", "content": text}


def steps(parts: list[dict]) -> list[list[dict]]:
    """An assistant message's parts by model call (`step`); parts from before phase 2 are one."""
    out: dict[int, list[dict]] = {}
    for p in parts:
        out.setdefault(int(p.get("step") or 1), []).append(p)
    return [out[k] for k in sorted(out)]


#: the tool images a request carries as images, the newest first: older ones become a line with
#: their path. Live (2026-10-13), five 300 KB screenshots of one run stayed in every later request
TOOL_IMAGES_KEPT = 1
LEFT_OUT = "\n[The image of this result is left out: a newer screenshot follows.]"


def _image_part(a: dict, data: bytes) -> dict:
    return {"type": "image_url", "image_url": {
        "url": f"data:{a.get('mime') or 'image/png'};base64," + base64.b64encode(data).decode()}}


def kept_tool_images(branch: list[dict], n: int | None = None) -> set:
    """The ids of the newest `n` (TOOL_IMAGES_KEPT) images tool results returned on `branch`."""
    n = TOOL_IMAGES_KEPT if n is None else n
    ids = [a.get("id") for m in branch if m.get("role") == "assistant"
           for p in m.get("parts") or [] if p.get("type") == "tool_call"
           for a in (p.get("output") or {}).get("images") or []]
    return set(ids[-n:]) if n > 0 else set()


def _assistant(m: dict, vision: bool, blob=None, keep_reasoning: bool = True,
               keep: set | None = None, inline: bool = True, from_step: int = 0) -> list[dict]:
    """One assistant message as the model saw it: per step its text, reasoning and tool calls,
    then one `tool` message per call with the result the model read. The images of those results
    (vision models only, the ones in `keep`; None: all) go in the tool message itself with
    `inline` (our engine's template renders them in the tool response, so the conversation keeps
    one user turn: a separate user message per screenshot read to the model as the user sending
    again, live 2026-10-13), else in a user message after the results. `from_step`: the steps
    before it were compacted (the summary is in the system block)."""
    out: list[dict] = []
    for group in steps([p for p in m.get("parts") or [] if int(p.get("step") or 1) >= from_step]):
        text = text_of(group)
        thinking = text_of(group, "thinking") if keep_reasoning else ""
        calls = [p for p in group if p.get("type") == "tool_call"]
        if not text and not thinking and not calls:
            continue
        msg: dict = {"role": "assistant", "content": text}
        if thinking:
            msg["reasoning_content"] = thinking
        if calls:
            msg["tool_calls"] = [{"id": c["id"], "type": "function",
                                  "function": {"name": c.get("name") or "",
                                               "arguments": c.get("input") or "{}"}}
                                 for c in calls]
        out.append(msg)
        images = []
        for c in calls:
            text, own = result_text(c), []
            for a in ((c.get("output") or {}).get("images") or []) if vision else []:
                if keep is not None and a.get("id") not in keep:
                    text += LEFT_OUT
                    continue
                data = blob(a) if blob is not None else None
                if data:
                    own.append(_image_part(a, data))
            if own and inline:
                out.append({"role": "tool", "tool_call_id": c["id"],
                            "content": [{"type": "text", "text": text}, *own]})
            else:
                out.append({"role": "tool", "tool_call_id": c["id"], "content": text})
                images += own
        if images:
            out.append({"role": "user", "content": [
                {"type": "text", "text": "The images the tool calls above returned."}, *images]})
    return out


def result_text(c: dict) -> str:
    """What the model reads for one call: the saved model text, else the visible output, else the
    call's state in words (every call gets an answer, or the next request is invalid)."""
    if c.get("model_content"):
        return c["model_content"]
    out = c.get("output") or {}
    if out.get("content"):
        return out["content"]
    st = c.get("status")
    if st == "denied":
        return "The user denied this call."
    if st in ("cancelled", "running", "asking"):
        return "The call was stopped before it finished."
    return "(no output)"


def message(m: dict, vision: bool, blob=None, keep_reasoning: bool = True, keep: set | None = None,
            inline: bool = True, from_step: int = 0) -> list[dict]:
    """One branch message as OpenAI messages; [] for what the model does not see (notices)."""
    if m.get("role") == "user":
        return [_user(m, vision, blob)]
    if m.get("role") == "assistant":
        return _assistant(m, vision, blob, keep_reasoning, keep, inline, from_step)
    return []


def body(model: str, settings: dict, branch: list[dict], *, system: str | None = None,
         vision: bool = False, blob=None, continue_final: bool = False,
         max_tokens: int | None = None, tools: list[dict] | None = None,
         keep_reasoning_from: int = 0, nudge: str | None = None,
         tandem: dict | None = None, inline_images: bool = True,
         step_from: tuple[int, int] | None = None) -> dict:
    """`keep_reasoning_from`: the branch index from which assistant reasoning goes back (the
    context guard drops it before that); `nudge`: a hidden last user line (the empty-turn retry);
    `inline_images`: tool images in the tool message (our engine) or in a user message after it;
    `step_from`: (a branch index, a step): that assistant message's steps before the step were
    compacted (a long run: the summary is in the system block)."""
    msgs: list[dict] = []
    if system:
        msgs.append({"role": "system", "content": system})
    keep = kept_tool_images(branch)
    for i, m in enumerate(branch):
        msgs.extend(message(m, vision, blob, keep_reasoning=i >= keep_reasoning_from, keep=keep,
                            inline=inline_images,
                            from_step=step_from[1] if step_from and i == step_from[0] else 0))
    if nudge:
        msgs.append({"role": "user", "content": nudge})
    out: dict = {"model": model, "messages": msgs, "stream": True,
                 "stream_options": {"include_usage": True}}
    if tools:
        out["tools"] = tools
        out["parallel_tool_calls"] = True
    effort = settings.get("effort") or "auto"
    if not settings.get("thinking", True):
        out["chat_template_kwargs"] = {"enable_thinking": False}
    elif effort != "auto":
        out["reasoning_effort"] = effort
    if max_tokens:
        out["max_tokens"] = int(max_tokens)
    if continue_final and msgs and msgs[-1]["role"] == "assistant" and \
            not msgs[-1].get("tool_calls"):
        out["continue_final_message"] = True
        out["add_generation_prompt"] = False
    if tandem:
        out["tandem"] = tandem
    return out


def system_block(*, today: str | None, tools: bool, summary: str | None = None,
                 skills: str | None = None, prompt: str | None = None,
                 plan: str | None = None) -> str | None:
    """The system message: the chat's own prompt, then the tool rules, the skill index and the
    summary of compacted turns. None for a plain chat with nothing to say."""
    parts = []
    if prompt:
        parts.append(prompt.strip())
    if tools:
        parts.append(
            f"You are the assistant in TandemLLM's chat{f'. Today is {today}' if today else ''}. "
            "You can use tools. Files live in /workspace, this chat's own Linux sandbox; the "
            "internet is reachable through a proxy, the local network is not. Read a file before "
            "you edit it, and prefer edit_file for small changes. Use present to show the user a "
            "page, a file or this run's changes. When the task is done, answer in plain words; do "
            "not call tools you do not need.")
    if plan:
        parts.append(plan)
    if skills:
        parts.append("Skills you can read with library_read when a task needs one:\n" + skills)
    if summary:
        parts.append("Summary of the earlier part of this conversation (the messages before it "
                     "were compacted):\n" + summary.strip())
    return "\n\n".join(parts) if parts else None


def encode(obj: dict) -> bytes:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode()


def headers(conversation_id: str, settings: dict) -> dict:
    """The call's headers, also fixed: the session id, the client kind (`chat-app`: the conductor
    maps it to its chat-app profile, SRV-351), and a profile only when the chat named one."""
    h = {"x-session-id": session_id(conversation_id), "X-QSE-Client": "chat-app"}
    if settings.get("profile"):
        h["X-Tandem-Profile"] = str(settings["profile"])
    return h
