"""A recorded Chat-tab conversation replayed through chatd in seconds, no model (SRV-667).

A chat's snapshot (`GET /v1/chat-app/conversations/<id>`, or a replay script's file that holds it
under `conv`) records what the model said at every step: its reasoning, its text and its tool
calls with their arguments. This feeds those steps back to a chatd in front of the test suite's
fake engine (tests/fake_backend.py), turn by turn with the chat's own settings, so every request
chatd builds for that conversation is built again by the code of this tree:

    replay   the conversation through chatd: file tools run on a scratch workspace, bash answers
             with the recorded output and exit code (a fake sandbox), the network and page tools
             with their recorded results; plan mode on for a turn whose message carried the plan
             rule, Approve for "Carry out the approved plan."; Continue after a resumable stop
             (the call cap, the run monitor) while the turn has recorded steps left; `cap` sets
             the chat's call limit (and the replay stops there)
    check    what chatd must do with it: no write's text replaced by a stub, a history note sent
             as content refused with what is on disk, every request the one before it plus its
             new messages (the engine's prefix cache), the same tool list and system block in and
             out of plan mode, a call cap leaving the chat idle, bash status from the exit code

    monitor  the steps where the run monitor's stall rule (chatd/stall.py) wrote its note into a
             call's result, steer or stop, printed after the replay
    reread   per request, where it parts from the one before it and about how many tokens the
             engine reads again from there (`--reread`): the requests that do not only append
             (a fold point, a compaction, a screenshot that became a line), estimated from their
             characters and image sizes as chatd's context guard counts them

    python tools/chat_trace.py <trace.json> [--cap N] [--bodies out.jsonl] [--store DIR]
        [--max-images N] [--reread]

The recorded tool images (render_page's screenshots) come back with their results, as chatd
renders them for a model that reads images (SRV-676): with `--store` (a chat data folder:
chat.sqlite3 and blobs/) the recorded bytes, else a stand-in PNG per image (the fake engine reads
none; the bodies then only fit the trace replay, not tools/prefix_check.py). `--max-images`: the
engine's image cap the model card names (16 when not given).

`--bodies` writes the requests chatd built, one JSON a line, for tools/prefix_check.py (the same
requests against the live engine) and tools/chat_fork.py (the live model from a step on). The
methods: CONTRIBUTING.md, "Before a pull request". Standard library and the test suite's helpers.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

#: tools whose results come from the record (the network, the page check, the panes, the library)
RECORDED = ("web_fetch", "web_search", "render_page", "present", "library_search", "library_read")
APPROVED = "Carry out the approved plan."


def load(path: str) -> dict:
    with open(path) as fh:
        d = json.load(fh)
    return d.get("conv") if isinstance(d.get("conv"), dict) else d


def turns(conv: dict) -> list[dict]:
    """The conversation's branch as turns: the user's text, whether it was a plan turn or the
    Approve, and the model's steps (reasoning, text, calls with their recorded results)."""
    out, cur = [], None
    for m in conv.get("messages") or []:
        parts = m.get("parts") or []
        if m.get("role") == "user":
            text = "".join(p.get("text") or "" for p in parts if p.get("type") == "text")
            cur = {"text": text, "plan": any(p.get("type") == "plan" for p in parts),
                   "approve": text.strip() == APPROVED, "steps": []}
            out.append(cur)
        elif m.get("role") == "assistant" and cur is not None:
            by: dict[int, dict] = {}
            for p in parts:
                s = by.setdefault(int(p.get("step") or 1), {"reasoning": "", "content": "",
                                                            "calls": []})
                if p.get("type") == "thinking":
                    s["reasoning"] += p.get("text") or ""
                elif p.get("type") == "text":
                    s["content"] += p.get("text") or ""
                elif p.get("type") == "tool_call" and p.get("name"):
                    try:
                        args = json.loads(p.get("input") or "{}")
                    except ValueError:
                        args = {}
                    out_ = p.get("output") or {}
                    s["calls"].append({"name": p["name"], "arguments": args,
                                       "result": out_.get("content") or "",
                                       "exit_code": out_.get("exit_code"),
                                       "is_error": bool(out_.get("is_error")),
                                       "status": p.get("status"),
                                       "images": list(out_.get("images") or [])})
            cur["steps"] += [by[k] for k in sorted(by)]
    return out


def _users(body: dict) -> int:
    def text(m):
        c = m.get("content") or ""
        return c if isinstance(c, str) else "".join(
            p.get("text") or "" for p in c if isinstance(p, dict) and p.get("type") == "text")
    return sum(1 for m in body.get("messages") or [] if m.get("role") == "user"
               and not text(m).startswith("The images the tool calls"))


def stand_in_png(key: str, w: int = 1280, h: int = 720) -> bytes:
    """A PNG of w x h with `key` in a text chunk: a screenshot's size (the context guard counts
    an image by its pixels) and bytes of its own, without drawing one."""
    import struct
    import zlib

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
    raw = b"".join(b"\x00" + b"\x20" * w for _ in range(h))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"tEXt", b"id\x00" + key.encode())
            + chunk(b"IEND", b""))


def image_reader(store: str | None):
    """`attachment -> (mime, bytes)`: the recorded bytes from a chat data folder (its
    chat.sqlite3 names each attachment's sha256, blobs/ holds them), else a stand-in PNG."""
    def read(a: dict) -> tuple[str, bytes]:
        if store:
            import sqlite3
            db = sqlite3.connect("file:" + os.path.join(store, "chat.sqlite3") + "?mode=ro",
                                 uri=True)
            try:
                r = db.execute("SELECT sha256, mime FROM attachments WHERE id = ?",
                               (a.get("id"),)).fetchone()
            finally:
                db.close()
            if r:
                path = os.path.join(store, "blobs", r[0][:2], r[0])
                if os.path.exists(path):
                    with open(path, "rb") as fh:
                        return r[1] or "image/png", fh.read()
        return a.get("mime") or "image/png", stand_in_png(str(a.get("id")))
    return read


def replay(conv: dict, cap: int | None = None, window: int | None = None,
           store: str | None = None, max_images: int | None = None,
           vision: bool | None = None, fork: tuple[int, int] | None = None, live=None,
           patch: str | None = None) -> dict:
    """The conversation through a chatd of this tree. Returns {"requests", "snapshot", "calls",
    "statuses"}: the bodies chatd sent (in order), the final snapshot, the calls as chatd ran them
    and the chat's status after each turn. The recorded tool images come back with their results
    (`image_reader`); `vision` (default: the record has images) makes the model read them.

    `fork`: (turn, step), both from 1: the recorded steps go up to the step before it, and from
    it on every model call of that turn is `live(body, headers)` (tools/chat_fork.py: the live
    engine), which returns the step as the fake engine's script takes it ({"content",
    "reasoning", "calls"}) or None to end the run; the replay ends after that turn. In the fork
    the recorded tools have no record left: they answer that they did not run, as bash does for
    a command the record does not hold. `patch`: the text that replaces the result of the last
    recorded call before the fork (the result as a fix would return it)."""
    from chatd import tools as tools_mod
    from chatd.tools import Result
    from test_chatd_agent import chat_with_sandbox, requests, small_window

    ts = turns(conv)
    if fork is not None:
        ft, fs = fork
        if not 1 <= ft <= len(ts) or not 1 <= fs <= len(ts[ft - 1]["steps"]) + 1:
            raise ValueError(f"no step {fs} in turn {ft} (the record has {len(ts)} turns, "
                             f"{[len(t['steps']) for t in ts]} steps)")
        ts = ts[:ft]
        ts[-1] = dict(ts[-1], steps=ts[-1]["steps"][:fs - 1])
        if patch is not None:
            done = [c for s in ts[-1]["steps"] for c in s["calls"]]
            if not done:
                raise ValueError("--patch: no recorded call before the fork step")
            # that call's result, whichever tool it is: the n-th run of its tool answers it
            name = done[-1]["name"]
            patch_at = (name, sum(1 for t in ts for s in t["steps"] for c in s["calls"]
                                  if c["name"] == name), patch, done[-1]["is_error"])
    patch_at = locals().get("patch_at")
    image_of = image_reader(store)
    has_images = any(c.get("images") for t in ts for s in t["steps"] for c in s["calls"])
    vision = has_images if vision is None else vision
    steps_by_turn = [list(t["steps"]) for t in ts]
    recorded: dict[str, list] = {n: [] for n in RECORDED}
    bash: list = []
    for t in ts:
        for s in t["steps"]:
            for c in s["calls"]:
                if c["name"] in recorded:
                    recorded[c["name"]].append(c)
                elif c["name"] == "bash":
                    bash.append(c)
    c = chat_with_sandbox()
    saved = {n: tools_mod.BY_NAME[n].executor for n in RECORDED if n in tools_mod.BY_NAME}
    if patch_at and patch_at[0] in tools_mod.BY_NAME:
        saved.setdefault(patch_at[0], tools_mod.BY_NAME[patch_at[0]].executor)
    try:
        for n in [x for x in saved if x in recorded]:
            queue = recorded[n]

            def answer(ctx, a, queue=queue):
                got = queue.pop(0) if queue else {
                    "result": "(not run: this replay has no record of the call)" if fork is None
                    else "(not run in the fork: this tool answers from the record only)",
                    "is_error": True}
                return Result(got["result"], is_error=got["is_error"],
                              images=[image_of(x) for x in got.get("images") or []])
            tools_mod.BY_NAME[n].executor = answer
        if patch_at and patch_at[0] in tools_mod.BY_NAME:
            inner, runs = tools_mod.BY_NAME[patch_at[0]].executor, {"n": 0}

            def patched(ctx, a, inner=inner, runs=runs):
                runs["n"] += 1
                if runs["n"] != patch_at[1]:
                    return inner(ctx, a)
                try:
                    inner(ctx, a)                   # it runs (a file the fork reads), as before
                except Exception:                                   # noqa: BLE001
                    pass
                return Result(patch_at[2], is_error=patch_at[3])
            tools_mod.BY_NAME[patch_at[0]].executor = patched
        live_on = {"n": 0}
        boxes = c.svc.sandboxes
        boxes.run_real = boxes.run

        def run(cid, cmd, *, timeout=600, on_output=None, cancel=None):
            line = str(cmd[-1]) if cmd else ""
            k = next((i for i, x in enumerate(bash)
                      if str(x["arguments"].get("command") or "\0") in line), None)
            if k is None:
                if fork is not None and live_on["n"] and on_output:
                    on_output("(not run in the fork: bash answers from the record only)")
                    return 1, False
                return 0, False                     # chatd's own command (a start, a snapshot)
            got = bash.pop(k)
            text = got["result"]
            for tail in (f"\n[exit code {got.get('exit_code')}]",):
                if text.endswith(tail):
                    text = text[:-len(tail)]
            if on_output and text:
                on_output(text)
            return int(got.get("exit_code") or 0), False
        boxes.run = run
        if window:
            small_window(c, window=window)
        if vision or max_images:
            info_of = c.svc.model_info

            def model_info(mid, info_of=info_of):
                info = dict(info_of(mid))
                # our engine's reading: the images inside their tool messages (render.py)
                info["vision"], info["backend"] = bool(vision), "engine"
                if max_images:
                    info["max_images"] = int(max_images)
                return info
            c.svc.model_info = model_info
        cid = c.new()
        settings = dict(conv.get("settings") or {})
        keep = {k: settings[k] for k in ("effort", "thinking", "tools", "permission_mode",
                                         "verify", "agents", "max_steps") if k in settings}
        keep["verify"] = False                      # the page check is the record's
        if cap:
            keep["max_steps"] = cap
        c.ok("PATCH", f"/conversations/{cid}/settings", keep)
        turn_of = {"n": 0}

        def script(body, headers):
            if headers.get("x-session-id", headers.get("X-Session-Id", "")).endswith("-task"):
                return None
            k = _users(body) - 1
            if k < 0 or k >= len(steps_by_turn):
                return {"content": "done"}
            turn_of["n"] = k
            q = steps_by_turn[k]
            if fork is not None and k == len(steps_by_turn) - 1 and not q:
                live_on["n"] += 1
                got = live(body, headers)
                return got if got is not None else {"content": "(the fork ends here)"}
            if not q:
                return {"content": "done"}
            s = q.pop(0)
            out = {"content": s["content"]}
            if s["reasoning"]:
                out["reasoning"] = s["reasoning"]
            if s["calls"]:
                out["calls"] = [{"name": x["name"], "arguments": x["arguments"]}
                                for x in s["calls"]]
            return out
        c.fake.script = script
        statuses, continued = [], []
        for i, t in enumerate(ts):
            if t["approve"]:
                c.ok("POST", f"/conversations/{cid}/plan/approve", {})
            else:
                if t["plan"]:
                    c.ok("PATCH", f"/conversations/{cid}/settings", {"plan_mode": True})
                c.send(cid, t["text"])
            snap = c.wait_idle(cid, 120)
            if cap and snap["messages"][-1].get("status") == "interrupted":
                statuses.append(snap.get("status"))
                break
            # the record goes on past a resumable stop (the call cap, the run monitor): the user
            # pressed Continue there, or the record had no such stop; at most 20 times a turn
            for _ in range(20):
                last = snap["messages"][-1]
                why = (last.get("stats") or {}).get("interrupted")
                waiting = fork is not None and i == len(ts) - 1 and not live_on["n"]
                if last.get("status") != "interrupted" or why not in ("limit", "stall") or \
                        not (steps_by_turn[i] or waiting):
                    break
                continued.append((i + 1, why))
                c.ok("POST", f"/conversations/{cid}/continue", {})
                snap = c.wait_idle(cid, 120)
            statuses.append(snap.get("status"))
        snap = c.snap(cid)
        made = [p for m in snap["messages"] if m["role"] == "assistant"
                for p in m["parts"] if p.get("type") == "tool_call"]
        return {"requests": requests(c), "snapshot": snap, "calls": made, "statuses": statuses,
                "continued": continued}
    finally:
        for n, ex in saved.items():
            tools_mod.BY_NAME[n].executor = ex
        c.close()


def _prefix_kept(prev: dict, cur: dict) -> bool:
    n = len(prev["messages"])
    enc = lambda x: json.dumps(x, ensure_ascii=False, sort_keys=True)  # noqa: E731
    return (enc(cur["messages"][:n]) == enc(prev["messages"]) and
            enc(cur.get("tools")) == enc(prev.get("tools")))


def _image_count(messages: list[dict]) -> int:
    return sum(1 for m in messages if isinstance(m.get("content"), list)
               for p in m["content"] if isinstance(p, dict) and p.get("type") == "image_url")


def check(res: dict, cap: int | None = None, max_images: int | None = None) -> list[str]:
    """What chatd must do with the conversation; a list of failures (empty: all held)."""
    from chatd import render
    bad = []
    reqs = res["requests"]
    done = {p["id"]: p for p in res["calls"]}
    for k, b in enumerate(reqs):
        for m in b["messages"]:
            for tc in m.get("tool_calls") or []:
                fn = tc["function"]
                if fn["name"] != "write_file":
                    continue
                part = done.get(tc["id"])
                if part is None or part.get("status") != "done":
                    continue
                try:
                    a = json.loads(fn["arguments"])
                except ValueError:
                    bad.append(f"request {k}: write {tc['id']} arguments are not JSON")
                    continue
                c = a.get("content")
                if isinstance(c, str) and (render.WROTE_STUB.match(c.strip()) or
                                           render.FOLDED_NOTE.match(c.strip())):
                    bad.append(f"request {k}: write {tc['id']} shows a stub as its content")
    for p in res["calls"]:
        c = (json.loads(p.get("input") or "{}") if p.get("name") == "write_file" else {})
        c = c.get("content") if isinstance(c, dict) else None
        if isinstance(c, str) and (render.WROTE_STUB.match(c.strip()) or
                                   render.FOLDED_NOTE.match(c.strip())):
            out = (p.get("output") or {}).get("content") or ""
            if p.get("status") != "failed" or "nothing was written" not in out:
                bad.append(f"call {p['id']}: a note sent as content was not refused with what is "
                           f"on disk ({out[:80]!r})")
        if p.get("name") == "bash" and p.get("status") in ("done", "failed"):
            code = (p.get("output") or {}).get("exit_code")
            want = "failed" if code else "done"
            if code is not None and p["status"] != want:
                bad.append(f"call {p['id']}: bash exit {code} marked {p['status']}")
    room = max_images or getattr(render, "TOOL_IMAGES_CAP", 16)
    for k in range(1, len(reqs)):
        if not _prefix_kept(reqs[k - 1], reqs[k]):
            notes = sum("is left out of this conversation" in str(m.get("content") or "")
                        for m in reqs[k]["messages"])
            summary = "Summary of the earlier part" in str(reqs[k]["messages"][0].get("content"))
            prev, cur = reqs[k - 1]["messages"], reqs[k]["messages"]
            j = next((i for i, (x, y) in enumerate(zip(prev, cur)) if x != y), None)
            if j is not None and render_left_out(prev[j], cur[j]):
                # a screenshot became a line: only at a fold point, when the request had no room
                # left for the next image (SRV-676)
                had = _image_count(prev)
                if had < room:
                    bad.append(f"request {k}: a screenshot became a line with {had} of {room} "
                               f"images in the request (not a fold point)")
            elif not summary and not notes:
                bad.append(f"request {k} does not start with request {k - 1} (prefix changed)")
    if cap and res["snapshot"]["messages"][-1].get("status") == "interrupted" and \
            res["snapshot"].get("status") != "idle":
        bad.append(f"the call cap left the chat {res['snapshot'].get('status')}, not idle")
    return bad


def reread(reqs: list[dict], ratio: float | None = None) -> list[dict]:
    """Per request after the first that does not start with the one before it: where it parts
    (the message index), and the tokens from there to its end as chatd's guard estimates them
    (chatd/compaction.py `estimate`: characters at `ratio`, images by their pixels). A request
    that only appends reads its new messages; one that parts reads everything after the change
    again (the engine resumes at the 4,096 grid line below it, not counted here)."""
    from chatd import compaction as comp
    enc = lambda x: json.dumps(x, ensure_ascii=False, sort_keys=True)  # noqa: E731
    out = []
    for k in range(1, len(reqs)):
        prev, cur = reqs[k - 1]["messages"], reqs[k]["messages"]
        j = next((i for i, (a, b) in enumerate(zip(prev, cur)) if enc(a) != enc(b)),
                 min(len(prev), len(cur)))
        tools_same = enc(reqs[k].get("tools")) == enc(reqs[k - 1].get("tools"))
        if j >= len(prev) and tools_same:
            continue
        j = 0 if not tools_same else j
        tail = comp.estimate({"messages": cur[j:]}, ratio) - 16
        out.append({"request": k, "at": j, "of": len(cur), "tokens": tail,
                    "images": comp.measure({"messages": cur[j:]})[1],
                    "why": "head" if j == 0 else
                           "image" if render_left_out(prev[j], cur[j]) else "other"})
    return out


def monitor_notes(snap: dict) -> list[tuple[int, int, str]]:
    """(turn, step, steer | stop) for every call result the run monitor's stall rule wrote a note
    into (chatd/stall.py)."""
    out, turn = [], 0
    for m in snap.get("messages") or []:
        if m.get("role") == "user":
            turn += 1
            continue
        for p in m.get("parts") or []:
            text = ((p.get("output") or {}).get("content") or "") if p.get("type") == \
                "tool_call" else ""
            if "[run monitor: still the same outcome" in text:
                out.append((turn, int(p.get("step") or 0), "stop"))
            elif "[run monitor:" in text:
                out.append((turn, int(p.get("step") or 0), "steer"))
    return out


def render_left_out(a: dict, b: dict) -> bool:
    """The message became the line of a left-out screenshot."""
    from chatd import render
    return isinstance(a.get("content"), list) and render.LEFT_OUT in str(b.get("content"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("trace")
    ap.add_argument("--cap", type=int, default=None)
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--bodies", default=None)
    ap.add_argument("--store", default=None, help="a chat data folder: the recorded screenshots")
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--reread", action="store_true", help="the requests that do not only append")
    ap.add_argument("--ratio", type=float, default=None, help="tokens a character (--reread)")
    a = ap.parse_args(argv)
    conv = load(a.trace)
    res = replay(conv, cap=a.cap, window=a.window, store=a.store, max_images=a.max_images)
    if a.bodies:
        with open(a.bodies, "w") as fh:
            for b in res["requests"]:
                fh.write(json.dumps(b, ensure_ascii=False) + "\n")
    bad = check(res, a.cap, a.max_images)
    ts = turns(conv)
    print(f"{a.trace}: {len(ts)} turns, {sum(len(t['steps']) for t in ts)} recorded steps, "
          f"{len(res['requests'])} requests, {len(res['calls'])} calls, statuses {res['statuses']}"
          + (f", continued {res['continued']}" if res.get("continued") else ""))
    for turn, step, level in monitor_notes(res["snapshot"]):
        print(f"  run monitor: turn {turn} step {step}: {level}")
    if a.reread:
        rows = reread(res["requests"], a.ratio)
        for r in rows:
            print(f"  request {r['request']:>4}: parts at message {r['at']:>4} of {r['of']:>4} "
                  f"({r['why']}), ~{r['tokens']:,} tokens read again ({r['images']:,} of images)")
        print(f"reread: {len(rows)} of {len(res['requests'])} requests part from the one before, "
              f"~{sum(r['tokens'] for r in rows):,} tokens in all "
              f"({sum(1 for r in rows if r['why'] == 'image')} at a screenshot)")
    for line in bad:
        print("FAIL", line)
    print("ok" if not bad else f"{len(bad)} failure(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
