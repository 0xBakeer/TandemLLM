"""Hardening of chatd's exec/MCP edges and the conductor's query params (upstream fixes).

  1  a read past EOF promises no "[N more lines]" (a negative rest made the model read forever),
     and a negative limit reads lines instead of nothing
  2  garbage /conductor query params go through _int: no uncaught ValueError (a 500), and
     limit=-1 clamps instead of becoming SQLite LIMIT -1 = unlimited
  3  a stdio MCP server's reader closes the stream (on_line(None)) even when the chat's box is
     gone, and keeps the busy count exact under threads
  4  a frame claiming over 64 MiB drops the exec stream instead of buffering it
  5  _kill_exec leaves a recycled pid alone (its /proc starttime moved)

Run: python tests/test_chatd_hardening.py
"""

from __future__ import annotations

import os
import socket
import struct
import subprocess
import sys
import threading
import time
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatd import mcp as mcp_mod                                    # noqa: E402
from chatd import sandbox as sb_mod                                 # noqa: E402
from chatd import tools as T                                        # noqa: E402
from chatd.workspace import Workspace                               # noqa: E402
from server.conductor import routes                                 # noqa: E402


# --------------------------------------------------------------- 1  read past EOF
class _WsCtx:
    def __init__(self, ws):
        self.ws = ws

    def changed(self, path):
        pass


def test_a_read_past_eof_promises_no_more_lines():
    ws = Workspace("/tmp")  # not written to; the file lives in a throwaway workspace below
    import tempfile
    ws = Workspace(tempfile.mkdtemp(prefix="rd-"))
    ws.write("f.txt", "\n".join(f"line {i}" for i in range(1, 6)).encode())
    ctx = _WsCtx(ws)
    # an offset past EOF: empty body, no "more lines" fragment
    out = T._read(ctx, {"path": "f.txt", "offset": 10}).content
    assert out == "", out
    # a huge offset too
    assert T._read(ctx, {"path": "f.txt", "offset": 10**9}).content == ""
    # a normal read still shows the tail fragment
    out = T._read(ctx, {"path": "f.txt", "offset": 4, "limit": 1}).content
    assert out.endswith("[1 more lines]"), out
    # the last full page says nothing more
    assert "more lines" not in T._read(ctx, {"path": "f.txt", "offset": 1, "limit": 5}).content
    # a negative limit reads (it used to yield an empty read plus a bogus fragment)
    out = T._read(ctx, {"path": "f.txt", "limit": -5}).content
    assert out.startswith("     1\tline 1"), out


# --------------------------------------------------------------- 2  conductor query params
class _Journal:
    def __init__(self):
        self.rows_kwargs = []

    def rows(self, **kw):
        self.rows_kwargs.append(kw)
        return []

    def with_events(self, rows, evs=None):
        return rows

    def get(self, _id):
        return None

    def info(self):
        return {}

    def status(self):
        return {}


def _cond():
    return types.SimpleNamespace(store=None, journal=_Journal(), state={},
                                 roles_health=lambda: {}, router=types.SimpleNamespace())


def test_garbage_query_params_do_not_500_and_negative_limit_clamps():
    cond = _cond()
    # garbage numbers fall back to the defaults instead of an uncaught ValueError (a 500)
    for path, q in ((["events"], {"since": "abc", "limit": "x", "before": "y"}),
                    (["decisions"], {"since": "abc", "limit": "x", "before": "y"}),
                    (["stats"], {"hours": "abc"})):
        st, _, _ = routes._route(None, cond, "GET", path, q, {}, None)
        assert st == 200, (path, st)
    # limit=-1 clamps to the low bound: never SQLite LIMIT -1 = a full-table scan
    routes._route(None, cond, "GET", ["events"], {"limit": "-1"}, {}, None)
    assert cond.journal.rows_kwargs[-1]["limit"] == 1
    routes._route(None, cond, "GET", ["decisions"], {"limit": "-1"}, {}, None)
    assert cond.journal.rows_kwargs[-1]["limit"] == 1
    # the ceilings still hold
    routes._route(None, cond, "GET", ["events"], {"limit": "999999"}, {}, None)
    assert cond.journal.rows_kwargs[-1]["limit"] == 2000
    routes._route(None, cond, "GET", ["decisions"], {"limit": "999999"}, {}, None)
    assert cond.journal.rows_kwargs[-1]["limit"] == 500
    # negative ms timestamps are pinned at 0, and 0 means "no bound" (None)
    routes._route(None, cond, "GET", ["events"], {"since": "-5", "before": "-5"}, {}, None)
    kw = cond.journal.rows_kwargs[-1]
    assert kw["since_ms"] == 0 and kw["before_ms"] is None, kw


# --------------------------------------------------------------- 3  the reader's exit paths
def test_a_stdio_mcp_reader_closes_the_stream_when_the_box_is_gone():
    class GoneSandboxes:
        def ensure(self, cid):
            pass

        def open_exec(self, cid, cmd, **kw):
            a, b = socket.socketpair()
            return "exec1", a, b""

        def box(self, cid):
            raise KeyError(cid)          # the chat's sandbox vanished mid-connect

    seen = []
    hook = threading.excepthook
    threading.excepthook = lambda args: None   # the KeyError in the reader thread is the point
    try:
        pipe = mcp_mod.exec_pipe(GoneSandboxes(), "c1", "mcp-server", [], {})
        write, start, closer = pipe(lambda ln: seen.append(ln))
        start()
        for _ in range(100):
            if seen:
                break
            time.sleep(0.05)
    finally:
        threading.excepthook = hook
    assert seen == [None], seen          # on_line(None) ran despite the KeyError


def test_the_busy_count_stays_exact_under_threads():
    b = sb_mod.Box("c1")
    def worker():
        for _ in range(200):
            with b.lock:                 # the fix: the deltas ride the lock
                b.busy += 1
            with b.lock:
                b.busy -= 1
    ts = [threading.Thread(target=worker) for _ in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert b.busy == 0, b.busy           # a racy counter strands the sandbox's eviction


# --------------------------------------------------------------- 4  the frame cap
def test_a_frame_over_the_cap_drops_the_stream():
    def framed(size, payload=b""):
        return struct.pack(">BxxxI", 1, size) + payload

    # normal frames still demux
    a, b = socket.socketpair()
    got = []
    b.sendall(framed(5, b"hello") + framed(2, b"ok"))
    b.close()                            # EOF: demux reads to the end of the stream
    sb_mod.demux(a, b"", lambda s, d: got.append((s, d)))
    assert got == [(1, b"hello"), (1, b"ok")], got
    # a frame claiming 100 MiB with only a drip of bytes: dropped, not buffered
    a, b = socket.socketpair()
    b.sendall(framed(100 << 20) + b"x" * 1024)
    try:
        sb_mod.demux(a, b"", lambda s, d: None)
        raise AssertionError("the oversized frame was accepted")
    except OSError as exc:
        assert "64 MiB" in str(exc), exc
    # and the buffered bytes never exceed one frame: the drip stops being read


# --------------------------------------------------------------- 5  the recycled pid
class _Engine:
    def __init__(self, pid, running=True):
        self.pid, self.running = pid, running

    def call(self, method, path, body=None, timeout=None):
        assert path.startswith("/exec/exec1/json")
        return 200, {"Running": self.running, "ExitCode": None, "Pid": self.pid}


def _sleeper():
    return subprocess.Popen(["sleep", "60"], start_new_session=True)


def _kill(proc):
    proc.kill()
    proc.wait()


def test_kill_exec_signals_the_execs_own_group():
    proc = _sleeper()
    try:
        sb = sb_mod.Sandboxes("/tmp/none", engine=_Engine(proc.pid))
        sb._kill_exec("exec1")
        assert proc.poll() is not None   # TERM reached the group
    finally:
        _kill(proc)


def test_kill_exec_leaves_a_recycled_pid_alone():
    proc = _sleeper()
    born = sb_mod._proc_starttime(proc.pid)
    assert born is not None
    real = sb_mod._proc_starttime
    try:
        # between reading the exec's pid and the signal the pid is recycled: its starttime moved
        # (the first read still matches, the re-check before each signal does not)
        state = {"n": 0}
        def moved(pid):
            state["n"] += 1
            return born if state["n"] == 1 else born + 1
        sb_mod._proc_starttime = moved
        sb = sb_mod.Sandboxes("/tmp/none", engine=_Engine(proc.pid))
        sb._kill_exec("exec1")
        assert proc.poll() is None       # untouched
        # an already-dead exec gives nothing to signal either
        sb_mod._proc_starttime = lambda pid: None
        sb._kill_exec("exec1")
        assert proc.poll() is None
    finally:
        sb_mod._proc_starttime = real
        _kill(proc)


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")