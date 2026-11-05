"""SRV-765 (QA SRV-766): the notes head carries no session address, so new chats of one user
share it.

The Chat tab (chatd) offers tools with a shell. The memory index named its routes on the session's
own address (`{tandem_url}` = `http://host/s/<token>`, server/session_tokens.py), and with
`QWEN38_NOTES_HEAD=on` the index sits in the head at the system message's end: 30 sessions sent 30
index texts, one after masking the token, so no chat reused another's head and every new chat read
3-4.8k hidden tokens again.

  1  two fresh Chat-tab chats with a shell, notes head on: the served prompts are the same up to the
     grid mark, the marks equal, no session address in the head; each tail names its own session
                                                    test_two_fresh_chats_share_the_notes_head
  2  the address is a tail note once a session; its token names the session, the memory route
     answers under it with the session's notes; the next turn replays it in place
                                                    test_the_address_is_a_tail_note_of_the_session
  3  `QWEN38_MEMORY_ROUTES=index`: the address in the index, no address note, heads differ (the
     layout before), and the turn's row names the head note that carries the token
                                                    test_routes_index_is_the_layout_before
  4  no shell: the index without routes, no address note, the heads shared
                                                    test_no_shell_no_address_note
  5  the head's order: memory card, index, then the class's note (`memory`, the default);
     `class`: the class's note first                test_head_order
  6  `index_text` under `note` is the same text for every session and address; `index` names the
     address                                        test_index_text_is_session_free
  7  `QWEN38_NOTES_HEAD` off (the shipped default) under `auto`: a fresh chat with a shell and its
     follow-up are served token for token as under `index` (fd627ac3's layout): the index names
     the session's address, no address note       test_head_off_is_the_prompt_before
  8  `note` forced with the head off: the index and the address note both at the tail, each once,
     replayed in place on the next turn           test_routes_note_with_the_head_off
  9  head on, `auto`: an address note that drops while the session's index stays live comes back
     at the next turn's tail; a session whose first request is not a conversation's first turn
     (no head) gets the index of before at the tail
                                                    test_auto_follows_the_session_and_the_head
 10  `routes_layout` alone: the knob, the head, a ranked index, the proxy's plan (`caps`), a
     compaction, a later turn, the session's live index
                                                    test_routes_layout
 11  the Chat tab's code_edit class (as when the class question falls back) with a package that
     names `{tandem_url}` in the head: `extra.head_session` names the class note's parts, the
     memory notes stay out of it                   test_head_session_names_a_class_note

On the tree before the fix (fd627ac3) every test but 4 fails; 1 is the regression (the heads differ
by the session's token). With the address note on every request with a shell (`note` as the
default) 6, 7, 9 and 10 fail; 7 is that regression (the head-off prompt changed).
CPU torch (the fake engine in process).
Run: python tests/test_notes_head_session_route.py
"""

from __future__ import annotations

import contextlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from test_memory_router import CHAT_APP, _chat_app, _index_store, _cfg  # noqa: E402
from test_ui_states import TOOLS, _go  # noqa: E402

from server import app, session_tokens as st  # noqa: E402
from server.conductor import memory as M  # noqa: E402
from server.conductor import router as R  # noqa: E402

SYSTEM = "You are a helpful assistant. Today is Sunday, 1 November 2026."
Q = "What do you know about my garden?"
ADDR = "Memory address: "


def _text(ids):
    return app.STATE["tok"].decode(ids, skip_special_tokens=False)


@contextlib.contextmanager
def box(head="on", reg=None, **knobs):
    """The Chat tab's profile with memory on, two user notes, a session-token registry (`reg`, or a
    new one), the notes head as given (on) and the given module knobs (`order`, `routes`); yields
    (cond, seen, marks, reg), `marks` the grid mark of each generation."""
    cond, seen = _chat_app()
    reg = reg if reg is not None else st.SessionTokens(None)
    cond.router.session_tokens = app.STATE["session_tokens"] = reg
    for t in ("garden: tomatoes in the south bed", "units: metric"):
        cond.notes.write("owner", {"op": "add", "scope": "user", "kind": "reference",
                                   "title": t, "body": f"The user noted {t}."}, writer="human")
    marks, real = [], app.generate_stream

    def rec(prompt, *a, **k):
        marks.append(k.get("grid_mark"))
        return real(prompt, *a, **k)
    was = {"head": R.NOTES_HEAD}
    R.NOTES_HEAD = head
    if "order" in knobs:
        was["order"] = R.NOTES_HEAD_ORDER
        R.NOTES_HEAD_ORDER = knobs["order"]
    if "routes" in knobs:
        was["routes"] = M.MEMORY_ROUTES
        M.MEMORY_ROUTES = knobs["routes"]
    app.generate_stream = rec
    try:
        yield cond, seen, marks, reg
    finally:
        app.generate_stream = real
        R.NOTES_HEAD = was["head"]
        if "order" in was:
            R.NOTES_HEAD_ORDER = was["order"]
        if "routes" in was:
            M.MEMORY_ROUTES = was["routes"]


def chat(text, sid, history=(), tools=True):
    body = {"model": "TandemLLM", "stream": True, "max_tokens": 2000,
            "reasoning_effort": "medium", "reasoning_format": "reasoning_content",
            "messages": [{"role": "system", "content": SYSTEM}, *history,
                         {"role": "user", "content": text}]}
    if tools:
        body["tools"] = TOOLS
    return _go(body, {**CHAT_APP, "Host": "box:8000", "x-session-id": sid})


def _row(cond, sid):
    cond.journal.flush()
    return cond.journal.rows(session="c:" + sid, limit=1)[0]


def test_two_fresh_chats_share_the_notes_head():
    with box() as (cond, seen, marks, reg):
        chat(Q, "chat-h1")
        p1, m1 = seen[-1], marks[-1]
        chat(Q, "chat-h2")
        p2, m2 = seen[-1], marks[-1]
    assert m1 and m1 == m2, (m1, m2)
    head = _text(p1[:m1])
    assert "Your memory: notes the runtime keeps" in head and "garden: tomatoes" in head, \
        head[-800:]
    assert "/s/" not in head, "the head carries no session address"
    assert p1[:m1] == p2[:m2], "two new chats of one user: the same head, token for token"
    t1, t2 = reg.token_for("c:chat-h1"), reg.token_for("c:chat-h2")
    assert t1 != t2
    tail1, tail2 = _text(p1[m1:]), _text(p2[m2:])
    assert f"http://box:8000/s/{t1}/v1/memory/search" in tail1 and t2 not in tail1, tail1[-900:]
    assert f"http://box:8000/s/{t2}/v1/memory/search" in tail2 and t1 not in tail2
    assert "head_session" not in (_row(cond, "chat-h2").get("extra") or {})


class _Route:
    """What `Handler._strip_session` and `memory_api.answer` read of a request."""

    def __init__(self, path):
        self.path = path

    def _json(self, status, body, **_):
        raise AssertionError((status, body))


def test_the_address_is_a_tail_note_of_the_session():
    from server import memory_api
    with box() as (cond, seen, marks, reg):
        out = chat(Q, "chat-r1")
        p1 = _text(seen[-1])
        tok = reg.token_for("c:chat-r1")
        user = p1.index("<|im_start|>user")
        assert p1.count(ADDR) == 1 and p1.index(ADDR) > user, "once, at the tail"
        assert f"{ADDR}http://box:8000/s/{tok} is this conversation's own address" in p1
        # the address reaches the session's own notes through the server's own token strip
        h = _Route(f"/s/{tok}/v1/memory/search?q=garden+tomatoes")
        assert app.Handler._strip_session(h) is False and h._sess == "c:chat-r1", vars(h)
        _, status, got, _ = memory_api.answer(cond, h, "GET", None)
        assert status == 200 and got and got[0]["title"].startswith("garden"), got
        # the next turn: the note replayed at its place, not put in again
        hist = [{"role": "user", "content": Q},
                {"role": "assistant", "content": out["content"],
                 "reasoning_content": out["reasoning"]}]
        chat("And which units do I use?", "chat-r1", hist)
        p2 = _text(seen[-1])
    end1 = p1.rindex("</runtime-note>") + len("</runtime-note>")
    assert p2.startswith(p1[:end1]), (p1[end1 - 300:end1], p2[end1 - 300:end1 + 100])
    assert p2.count(ADDR) == 1 and p2.count("Your memory: notes the runtime keeps") == 1


def test_routes_index_is_the_layout_before():
    with box(routes="index") as (cond, seen, marks, reg):
        chat(Q, "chat-o1")
        p1, m1 = seen[-1], marks[-1]
        chat(Q, "chat-o2")
        p2, m2 = seen[-1], marks[-1]
        row = _row(cond, "chat-o2")
    t1 = reg.token_for("c:chat-o1")
    head1 = _text(p1[:m1])
    assert f"GET http://box:8000/s/{t1}/v1/memory/<id>" in head1, head1[-800:]
    assert ADDR not in _text(p1) and ADDR not in _text(p2)
    assert p1[:m1] != p2[:m2], "each head carries its own session's address"
    assert "memory_index" in (row.get("extra") or {}).get("head_session", []), row.get("extra")


def test_no_shell_no_address_note():
    with box() as (cond, seen, marks, reg):
        chat(Q, "chat-n1", tools=False)
        p1, m1 = seen[-1], marks[-1]
        chat(Q, "chat-n2", tools=False)
        p2, m2 = seen[-1], marks[-1]
    text = _text(p1)
    assert "there is no tool to read or search them here" in text and ADDR not in text
    assert "/v1/memory" not in text
    assert m1 and p1[:m1] == p2[:m2]


def _order(cond, sid):
    """The served start of each injected part's note in the turn's row, by the part's kind (the
    class's note carries `rules`)."""
    at = {}
    for x in _row(cond, sid).get("injected") or []:
        for p in x.get("parts") or []:
            at.setdefault(p["kind"], x["at"])
    return at


def test_head_order():
    with box() as (cond, seen, marks, reg):
        chat(Q, "chat-o3")
        m = marks[-1]
        at = _order(cond, "chat-o3")
    assert "rules" in at, ("the class's note (its rules part)", at)
    assert at["memory_card"] < at["memory_index"] < at["rules"] < m < at["memory_routes"], at
    with box(order="class") as (cond, seen, marks, reg):
        chat(Q, "chat-o4")
        m = marks[-1]
        at = _order(cond, "chat-o4")
    assert at["rules"] < at["memory_card"] < at["memory_index"] < m < at["memory_routes"], at


def test_index_text_is_session_free():
    st_, proj, gate, _ = _index_store(n_user=3)
    cfg = _cfg()
    a = M.index_text(st_, "owner", proj, "c:s2", cfg, None, "http://h:8000/s/AAAAAAAAAAAA",
                     routes="note")
    b = M.index_text(st_, "owner", proj, "c:s3", cfg, None, "http://other:9/s/BBBBBBBBBBBB",
                     routes="note")
    assert a == b and "/s/" not in a[0] and "GET <address>/v1/memory/<id>" in a[0], a[0][:400]
    c = M.index_text(st_, "owner", proj, "c:s2", cfg, None, "http://h:8000/s/AAAAAAAAAAAA",
                     routes="index")
    assert "GET http://h:8000/s/AAAAAAAAAAAA/v1/memory/<id>" in c[0] and c[1] == a[1]
    # `auto` is resolved per request (`routes_layout`); unresolved it is the text before
    d = M.index_text(st_, "owner", proj, "c:s2", cfg, None, "http://h:8000/s/AAAAAAAAAAAA",
                     routes="auto")
    assert d == c
    assert M.MEMORY_ROUTES == "auto" and R.NOTES_HEAD_ORDER == "memory", "the shipped defaults"
    assert R.NOTES_HEAD == "off", "the head stays off by default until its quality gate passes"


class _OneAddress(st.SessionTokens):
    """Every session gets the same token, so two sessions' prompts compare token for token."""

    def token_for(self, key: str) -> str:
        return "T0k3nT0k3nAA"


def _two_turns(seen, sid):
    """A fresh chat with a shell and its follow-up: the two served prompts (ids)."""
    out = chat(Q, sid)
    p1 = seen[-1]
    hist = [{"role": "user", "content": Q},
            {"role": "assistant", "content": out["content"], "reasoning_content": out["reasoning"]}]
    chat("And which units do I use?", sid, hist)
    return p1, seen[-1]


def test_head_off_is_the_prompt_before():
    with box(head="off", reg=_OneAddress(None)) as (cond, seen, marks, reg):
        a1, a2 = _two_turns(seen, "chat-f1")
        was, M.MEMORY_ROUTES = M.MEMORY_ROUTES, "index"
        try:
            b1, b2 = _two_turns(seen, "chat-f1b")       # fd627ac3's layout
        finally:
            M.MEMORY_ROUTES = was
    assert a1 == b1 and a2 == b2, "head off: the served prompts of fd627ac3's layout"
    t1, t2 = _text(a1), _text(a2)
    assert ADDR not in t1 and ADDR not in t2 and M.ADDRESS not in t1
    assert "GET http://box:8000/s/T0k3nT0k3nAA/v1/memory/<id>" in t1, t1[-1500:]
    assert t1.index("Your memory: notes the runtime keeps") > t1.index("<|im_start|>user"), \
        "head off: the index at the tail"
    assert t2.count("Your memory: notes the runtime keeps") == 1


def test_routes_note_with_the_head_off():
    with box(head="off", routes="note") as (cond, seen, marks, reg):
        out = chat(Q, "chat-f2")
        p1 = _text(seen[-1])
        hist = [{"role": "user", "content": Q},
                {"role": "assistant", "content": out["content"],
                 "reasoning_content": out["reasoning"]}]
        chat("And which units do I use?", "chat-f2", hist)
        p2 = _text(seen[-1])
        tok = reg.token_for("c:chat-f2")
    user = p1.index("<|im_start|>user")
    idx = p1.index("Your memory: notes the runtime keeps")
    assert user < idx < p1.index(ADDR), "both at the tail, the index first"
    assert "/s/" not in p1[idx:p1.index(ADDR)] and f"{ADDR}http://box:8000/s/{tok} is" in p1
    end1 = p1.rindex("</runtime-note>") + len("</runtime-note>")
    assert p2.startswith(p1[:end1]), "the next turn replays both in place"
    assert p2.count(ADDR) == 1 and p2.count("Your memory: notes the runtime keeps") == 1


def test_auto_follows_the_session_and_the_head():
    with box() as (cond, seen, marks, reg):
        out = chat(Q, "chat-a1")
        p1, m1 = _text(seen[-1]), marks[-1]
        assert m1 and "<address>" in _text(seen[-1][:m1]) and p1.count(ADDR) == 1
        sess = cond.router.session("c:chat-a1")
        with sess.lock:            # the address note drops (its message was rewritten)
            sess.notes = [n for n in sess.notes
                          if not any(p.get("kind") == "memory_routes" for p in n.parts or ())]
        hist = [{"role": "user", "content": Q},
                {"role": "assistant", "content": out["content"],
                 "reasoning_content": out["reasoning"]}]
        chat("And which units do I use?", "chat-a1", hist)
        p2 = _text(seen[-1])
        tok = reg.token_for("c:chat-a1")
        # a session first seen past its conversation's first turn: no head, the index of before
        chat("And which units do I use?", "chat-a2", hist)
        p3 = _text(seen[-1])
        tok3 = reg.token_for("c:chat-a2")
    assert p2.count(ADDR) == 1 and p2.index(ADDR) > p2.rindex("<|im_start|>user"), \
        "the address note comes back at the new tail"
    assert f"{ADDR}http://box:8000/s/{tok} is" in p2
    assert p2.count("Your memory: notes the runtime keeps") == 1, "the index stays, once"
    assert ADDR not in p3 and M.ADDRESS not in p3
    assert f"GET http://box:8000/s/{tok3}/v1/memory/<id>" in p3, p3[-1500:]


class _P:
    def __init__(self, fresh=True, compaction=False, caps=None):
        self.fresh, self.compaction, self.caps = fresh, compaction, caps


class _S:
    def __init__(self, notes=()):
        self.notes = list(notes)


def test_routes_layout():
    cfg, ranked = {"index_order": "recent"}, {"index_order": "ranked"}
    idx_note = R.Injection(0, "h", [], f"... GET {M.ADDRESS}/v1/memory/<id> ...",
                           [{"kind": "memory_index", "name": "index"}], {"memory"}, "k", 1, 1)
    old_note = R.Injection(0, "h", [], "... GET http://h/s/AAAAAAAAAAAA/v1/memory/<id> ...",
                           [{"kind": "memory_index", "name": "index"}], {"memory"}, "k", 1, 1)
    was = (R.NOTES_HEAD, M.MEMORY_ROUTES)
    try:
        M.MEMORY_ROUTES = "auto"
        R.NOTES_HEAD = "off"
        assert M.routes_layout(_P(), _S(), cfg) == "index"
        assert M.routes_layout(_P(), _S([idx_note]), cfg) == "index", "the head off decides"
        R.NOTES_HEAD = "on"
        assert M.routes_layout(_P(), _S(), cfg) == "note"
        assert M.routes_layout(_P(), _S(), ranked) == "index"
        assert M.routes_layout(_P(caps={"memory": "on"}), _S(), cfg) == "index", "the proxy"
        assert M.routes_layout(_P(compaction=True), _S(), cfg) == "index"
        assert M.routes_layout(_P(fresh=False), _S(), cfg) == "index"
        assert M.routes_layout(_P(fresh=False), _S([idx_note]), cfg) == "note"
        assert M.routes_layout(_P(fresh=False), _S([old_note]), cfg) == "index"
        for forced in ("note", "index"):
            M.MEMORY_ROUTES = forced
            R.NOTES_HEAD = "off"
            assert M.routes_layout(_P(fresh=False, caps={}), _S(), ranked) == forced
    finally:
        R.NOTES_HEAD, M.MEMORY_ROUTES = was


PKG = "laya-ask-test"


def test_head_session_names_a_class_note():
    with box() as (cond, seen, marks, reg):
        _, pk = cond.store.get("packages")
        pk["packages"][PKG] = {"title": "Ask Laya", "kind": "package", "source": "inline",
                               "description": "Ask Laya through the session's address.",
                               "body": 'Ask Laya: curl -s "{tandem_url}/v1/laya/ask"',
                               "max_tokens": 200, "enabled": True, "general": True}
        cond.store.put("packages", None, pk, note="a package with the address", author="test")
        _, r = cond.store.get("router")
        for c in r["classes"]:
            if c["id"] == "code_edit":
                c["policy"]["packages"] = [PKG]
        cond.store.put("router", None, r, note="code_edit names it", author="test")
        _, prof = cond.store.get("profiles", "chat-app")
        assert prof["default_class"] == "code_edit"
        prof["decisions"]["classify"] = "off"          # the class question's fallback
        cond.store.put("profiles", "chat-app", prof, note="code_edit", author="test")
        chat(Q, "chat-c1")
        p1, m1 = seen[-1], marks[-1]
        row = _row(cond, "chat-c1")
        tok = reg.token_for("c:chat-c1")
    assert row.get("class_id") == "code_edit", row.get("class_id")
    head = _text(p1[:m1])
    assert f"http://box:8000/s/{tok}/v1/laya/ask" in head, head[-1200:]
    cls_kinds = set()
    for x in row.get("injected") or []:
        kinds = {p["kind"] for p in x.get("parts") or []}
        if "package" in kinds and any(p.get("name") == PKG for p in x.get("parts") or []):
            cls_kinds = kinds
    bound = (row.get("extra") or {}).get("head_session")
    assert bound and "package" in bound and set(bound) == cls_kinds, (bound, cls_kinds)
    assert not {"memory_index", "memory_card", "memory_routes"} & set(bound), bound


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:60s} ok")
            passed += 1
    print(f"{passed} passed")
