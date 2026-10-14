"""The Context panel's figures (SRV-445, SRV-449, SRV-451, SRV-453). Seen on a phone on 2026-10-14
after a 48-step run that wrote an HTML file: "tool calls 0" after about 60 calls, "files 0" after a
write_file of 7.8 KB and seven edit_file calls, "KV for this chat —", "resident prefix —" and
"sandbox running, 0 B".

  1  the usage counts every tool call of the branch
  2  the context's `files` part counts what the file tools carried (a write's content, an edit's
     old and new text, a read's result), from the first message a compaction kept
  3  the memory says where the engine holds the chat: its rows in the slot or the park, by the
     owner's tag (or its short key from an engine without tags), else the tier the next turn reads
  4  a running sandbox's memory is the runtime's figure (usage less the inactive file cache),
     read at most every 5 s, and null when the runtime does not say
  5  the harness's picks and the prompt's split outlive a chatd restart (SRV-455)

CPU only. Run: python -m pytest -q tests/test_chatd_panel.py
"""

from __future__ import annotations

import hashlib
import json
import shutil

from chatd_util import Chat
from test_chatd_agent import calls

from chatd import context as ctx_mod
from chatd import sandbox as sb_mod


def test_the_usage_counts_every_tool_call_and_the_files_part_what_the_file_tools_carried():
    c = Chat()
    try:
        cid = c.new()
        page = "<!doctype html><h1>" + "x" * 4000 + "</h1>"
        c.send(cid, calls(("write_file", {"path": "site/a.html", "content": page}),
                          ("edit_file", {"path": "site/a.html", "old": "<h1>", "new": "<h2>"})))
        s = c.wait_idle(cid, 20)
        made = [p for m in s["messages"] if m["role"] == "assistant" for p in m["parts"]
                if p["type"] == "tool_call"]
        assert len(made) == 2 and s["usage"]["tool_calls"] == 2, s["usage"]
        files = next(p for p in s["context"]["parts"] if p["kind"] == "files")["tokens"]
        assert files >= len(page) // 4, s["context"]["parts"]
        c.send(cid, calls(("read_file", {"path": "site/a.html"})))
        s = c.wait_idle(cid, 20)
        assert s["usage"]["tool_calls"] == 3
        files2 = next(p for p in s["context"]["parts"] if p["kind"] == "files")["tokens"]
        assert files2 > files + len(page) // 4 - 10, (files, files2)
        msgs = next(p for p in s["context"]["parts"] if p["kind"] == "messages")["tokens"]
        assert msgs >= 0
    finally:
        c.close()


def test_file_tokens_start_at_the_compaction_boundary():
    w = {"type": "tool_call", "name": "write_file", "step": 1,
         "input": json.dumps({"path": "a", "content": "y" * 400})}
    late = dict(w, step=3)
    branch = [{"role": "user", "parts": []}, {"role": "assistant", "parts": [w]},
              {"role": "user", "parts": []}, {"role": "assistant", "parts": [w, late]}]
    assert ctx_mod.file_tokens(branch) == 300
    assert ctx_mod.file_tokens(branch, 2) == 200
    assert ctx_mod.file_tokens(branch, 2, (3, 2)) == 100
    bad = {"type": "tool_call", "name": "write_file", "input": "{\"path\": "}
    failed_read = {"type": "tool_call", "name": "read_file",
                   "output": {"content": "no such file", "is_error": True}}
    assert ctx_mod.file_tokens([{"role": "assistant", "parts": [bad, failed_read]}]) == 0


def _res(**kw):
    base = {"valid": 37614, "bytes": 3_000_000, "park_bytes": 1_000_000, "pin_bytes": 500_000,
            "park_rows": 900, "owner": None, "park_owner": None}
    return {"caches": {"resident": {**base, **kw}}, "engine": {"base_model": "x"}}


def _tag(s):
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def test_the_memory_says_where_the_engine_holds_the_chat():
    me, other = "chat-01J9Z3K7Q2W5E8R4T6Y1U3I5O7", "chat-01J9Z3K0A1S2D3F4G5H6J7K8L9"
    warm = {"warm": 1000, "source": "resident"}
    h = ctx_mod.holding(_res(owner="chat-01J9Z3K…", owner_tag=_tag(me)), me, warm)
    assert h == {"kv_tokens": 37614, "kv_where": "slot", "resident_bytes": 1_500_000,
                 "resident_owner": "this"}, h
    # the tag decides: the same short key, another chat
    h = ctx_mod.holding(_res(owner="chat-01J9Z3K…", owner_tag=_tag(other)), me, warm)
    assert h["kv_where"] == "other" and h["resident_owner"] == "other" and h["kv_tokens"] is None
    # an engine without tags: the short key
    assert ctx_mod.holding(_res(owner="chat-01J9Z3K…"), me, warm)["kv_where"] == "slot"
    h = ctx_mod.holding(_res(owner="chat-01J9Z3X…", park_owner="chat-01J9Z3K…",
                             park_owner_tag=_tag(me), owner_tag=_tag(other)), me, warm)
    assert h == {"kv_tokens": 900, "kv_where": "park", "resident_bytes": 1_000_000,
                 "resident_owner": "this"}, h
    spilled = {"warm": 30000, "source": "nvme_spill"}
    assert ctx_mod.holding(_res(owner="x", owner_tag=_tag(other)), me, spilled)["kv_where"] == \
        "nvme_spill"
    assert ctx_mod.holding(_res(), me, {"warm": 0, "source": "none"}) == {
        "kv_tokens": None, "kv_where": "none", "resident_bytes": None, "resident_owner": "none"}
    assert ctx_mod.holding({}, me, warm)["kv_where"] is None


def test_the_report_carries_the_holding_for_our_engine_only():
    model = {"id": "m", "max_len": 1000, "max_output": 100}
    common = dict(model=model, settings={}, branch_tokens=10, files_tokens=0, stats=None,
                  run=None, row=None, peek=None, harness=None)
    me = "chat-abc"
    r = ctx_mod.report(**common, facts=_res(owner=me, owner_tag=_tag(me)), session=me)
    assert r["memory"]["kv_where"] == "slot" and r["memory"]["kv_tokens"] == 37614
    r = ctx_mod.report(**common, facts={}, session=me)
    assert r["memory"]["kv_where"] is None and r["memory"]["resident_owner"] is None


class _Docker:
    def __init__(self, answer):
        self.answer, self.calls = answer, 0

    def call(self, method, path, body=None, timeout=60.0):
        self.calls += 1
        if isinstance(self.answer, Exception):
            raise self.answer
        assert path.startswith("/containers/box-1/stats?stream=false"), path
        return self.answer


def test_a_running_sandbox_reports_its_memory_in_use(tmp_path):
    t = [100.0]
    eng = _Docker((200, {"memory_stats": {"usage": 64_405_504,
                                          "stats": {"inactive_file": 32_956_416}}}))
    boxes = sb_mod.Sandboxes(str(tmp_path), engine=eng, clock=lambda: t[0])
    b = boxes.box("c1")
    assert boxes.memory_used(b) is None, "a stopped box has no figure"
    b.state, b.container = "running", "box-1"
    assert boxes.memory_used(b) == 31_449_088
    t[0] += 1
    assert boxes.memory_used(b) == 31_449_088 and eng.calls == 1, "read at most every 5 s"
    t[0] += 10
    eng.answer = OSError("socket gone")
    assert boxes.memory_used(b) is None
    t[0] += 10
    eng.answer = (200, {"memory_stats": {}})
    assert boxes.memory_used(b) is None, "a runtime that does not say"


def test_the_harness_and_the_split_outlive_a_chatd_restart():
    """SRV-455: the usage frame's `tandem` (the harness's picks, the system and tool tokens) lived
    in chatd's memory only; after a restart the panel lost the skills and showed the system prompt
    and the tool schemas as 0 until the next turn. The answer's stats keep them now."""
    c = Chat()
    try:
        cid = c.new()
        c.send(cid, "FAKE_TANDEM hi there")
        s = c.wait_idle(cid)
        want = s["context"]["harness"]
        assert want and want["packages"][0]["name"] == "coding", want
        assert s["messages"][-1]["stats"]["tandem"]["harness"] == want
        c.owned_fake = False                       # the same fake engine serves the restart
        c.close(keep_data=True)
        again = Chat(data=c.data, fake=(c.fake_srv, c.fake_port))
        try:
            ctx = again.snap(cid)["context"]
            parts = {p["kind"]: p["tokens"] for p in ctx["parts"]}
            assert ctx["harness"] == want and parts["system"] == 2 and parts["tools"] == 3, ctx
        finally:
            again.close()
    finally:
        c.fake_srv.shutdown()
        shutil.rmtree(c.data, ignore_errors=True)
