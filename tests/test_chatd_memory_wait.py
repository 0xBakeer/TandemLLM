"""SRV-787: a model call the engine holds for memory, as chatd sees it.

On 2026-11-05 a 128k turn waited 900 s for memory with no byte from the engine, and chatd's `READ_TIMEOUT_S` cut it
(the engine's log: "left while waiting for memory (900194 ms)"). The engine now opens the stream at once and sends an
SSE comment every 15 s while the call waits (server/queue_wait.py `mem_comment`), and ends a wait at its bound with an
error event (`memory_wait_timeout`).

  * the read timeout is per read: a stream kept alive by comments outlives 2 x `READ_TIMEOUT_S` of waiting and is
    read to its end; the same silence without them breaks it, as before;
  * the comment sets the Now line ("Waiting for memory: ..."), not "In the engine's line";
  * the bound's error event (or its 503) ends the run `interrupted` with the engine's reason under the answer, not
    retried and not a stalled model; Continue sends it again.

Run: python tests/test_chatd_memory_wait.py
"""

from __future__ import annotations

import json
import os
import re
import socket
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatd import engine as engine_mod  # noqa: E402
from chatd import runs as runs_mod  # noqa: E402
from server import queue_wait  # noqa: E402

REFUSAL = ("Not enough memory for this turn right now (needed 9.3 GiB, 7.7 free above the guard). Try again when "
           "the other chat finishes, or start a shorter conversation.")


def _server(wait_s: float, every_s: float | None):
    """One connection: the request read, the stream's headers, then `wait_s` of waiting with a memory comment every
    `every_s` (None: silence), then one content chunk and [DONE]."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)

    def run():
        c, _ = srv.accept()
        data = b""
        while b"\r\n\r\n" not in data:
            data += c.recv(65536)
        head, _, rest = data.partition(b"\r\n\r\n")
        size = int(re.search(rb"(?i)\r\ncontent-length:\s*(\d+)", head).group(1))
        while len(rest) < size:
            rest += c.recv(65536)
        try:
            c.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nX-Tandem-Wait: memory\r\n"
                      b"Connection: close\r\n\r\n")
            t0 = time.monotonic()
            while time.monotonic() - t0 < wait_s:
                if every_s is not None:
                    waited = time.monotonic() - t0
                    c.sendall(queue_wait.mem_comment(1, 7.7, waited, 600 - waited))
                time.sleep(every_s if every_s is not None else wait_s)
            chunk = {"id": "chatcmpl-m", "choices": [{"index": 0, "delta": {"content": "done"}}]}
            c.sendall(f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode())
        except OSError:
            pass
        c.close()
        srv.close()
    threading.Thread(target=run, daemon=True).start()
    return srv.getsockname()[1]


def _read(port):
    call = engine_mod.Engine(f"http://127.0.0.1:{port}").stream(b"{}", {})
    return [kind for kind, _ in call.events()]


def test_a_call_kept_alive_by_comments_outlives_twice_the_read_timeout():
    was = engine_mod.READ_TIMEOUT_S
    engine_mod.READ_TIMEOUT_S = 0.4
    try:
        t0 = time.monotonic()
        kinds = _read(_server(2 * 0.4 + 0.4, 0.1))           # 1.2 s of waiting, a comment every 0.1 s
        took = time.monotonic() - t0
        silent = _read(_server(2 * 0.4 + 0.4, None))         # the same wait with no byte
    finally:
        engine_mod.READ_TIMEOUT_S = was
    assert took >= 1.2 and kinds.count("comment") >= 8 and kinds[-3:] == ["data", "done", "end"], (took, kinds)
    assert silent == ["broken"], silent
    return f"{kinds.count('comment')} comments over {took:.1f} s (timeout 0.4 s), then the answer; silent: broken"


def _bare(frames):
    r = runs_mod.Run.__new__(runs_mod.Run)
    r.call = type("C", (), {"events": staticmethod(lambda: iter(frames))})()
    r.request_id, r.request_ids, r.usage, r.step_usage = None, set(), None, []
    r.stop_mode, r.carry, r.in_tag, r.finish_reason = None, "", False, None
    r.error = r.notice = None
    r.phase, r.phases = "prefill", []
    r.set_phase = lambda phase, now, **kw: r.phases.append((phase, now))   # type: ignore[method-assign]
    r._content = r._add = r._tool_delta = lambda *a: None                  # type: ignore[method-assign]
    return r


def test_the_wait_shows_and_the_bound_ends_the_run_with_the_engines_reason():
    comment = queue_wait.mem_comment(1, 7.7, 120.4, 479.6).decode()[1:].strip()
    frame = {"id": "chatcmpl-m", "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
             "error": {"message": REFUSAL, "type": "server_busy", "code": "memory_wait_timeout", "status": 503}}
    r = _bare([("comment", comment), ("data", frame), ("done", None)])
    assert r._consume() == "memory_refused" and r.error == REFUSAL, (r.error,)
    assert r.phases == [("queued", "Memory is tight: this chat waits for 7.7 GiB to free (2 min so far) and continues "
                                   "by itself. Stop cancels; it gives up after 10 min without memory")], r.phases
    # 2026-11-06: the first end of the wait in a run compacts the chat (a lower compaction line) and sends it again
    r.memory_compacted, r.request_est, r.compact_next = False, 208_326, False
    r.conv = type("Cv", (), {"memory_limit": None})()
    assert r._memory_compact(r.error) and r.compact_next and r.conv.memory_limit == 208_326, vars(r.conv)
    assert r.phases[-1] == ("queued", "Memory stayed full: compacting this chat to fit, then trying again")
    assert not r._memory_compact(r.error), "once a run: the second end of the wait ends the run"
    assert r._memory_refused(r.error) == "interrupted"
    assert r.notice == REFUSAL[:-1] + ". Continue goes on from here.", r.notice
    # a 503 before the stream (an engine that never opened it) is the same refusal, not a retry
    exc = engine_mod.EngineError(503, REFUSAL, True, 30.0, kind="server_busy", code="memory_wait_timeout")
    assert runs_mod.memory_refused(exc) and not runs_mod.memory_full(exc)
    assert not runs_mod.memory_refused(engine_mod.EngineError(503, "busy", True, code="memory_full"))
    # the queue's own comment still reads as the line
    q = _bare([("comment", queue_wait.comment(2, 40.0, 12.0).decode()[1:].strip()), ("done", None)])
    q._consume()
    assert q.phases == [("queued", "In the engine's line, 2 ahead")], q.phases
    return r.notice


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            n += 1
            print(f"ok  {name}")
    print(f"{n} passed")
