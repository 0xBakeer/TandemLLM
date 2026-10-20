"""A stdio MCP server's closer (chatd.mcp.exec_pipe) ends the exec, not just the socket:
a restart/drop of the supervisor SIGTERM's the server process instead of orphaning it, and
the reader thread reaps the box slot when the stream dies. socketpair + a fake sandboxes,
no engine. CPU only.

Run: python tests/test_chatd_mcp_stdio_kill.py
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatd import mcp as mcp_mod


class FakeBox:
    def __init__(self):
        self.busy = 0
        self.lock = threading.RLock()          # the real Box grew one (exec-reader
                                               # threads take it around busy +-)


class FakeSandboxes:
    """One exec on a socketpair; end_exec kills the far end, as a dying process would."""

    def __init__(self):
        self.ended: list[str] = []
        self.box_ = FakeBox()
        self.peer: socket.socket | None = None
        self.sock: socket.socket | None = None

    def ensure(self, cid):
        pass

    def box(self, cid):
        return self.box_

    def open_exec(self, cid, cmd, *, tty=False, stdin=False, env=None, workdir="/workspace"):
        a, b = socket.socketpair()
        self.peer = b
        self.sock = a
        return "exec-1", a, b""

    def end_exec(self, exec_id):
        self.ended.append(exec_id)
        if self.peer is not None:                       # the process is gone: the engine
            self.peer.close()                            # closes the attach stream
            self.peer = None


def test_closer_ends_the_exec_and_reaps_the_slot():
    sb = FakeSandboxes()
    lines = []
    open_pipe = mcp_mod.exec_pipe(sb, "c1", "server", [], {})
    _write, start_reader, closer = open_pipe(lines.append)
    start_reader()
    time.sleep(0.05)
    assert sb.box_.busy == 1
    closer()
    t0 = time.monotonic()
    while time.monotonic() - t0 < 2.0 and None not in lines:
        time.sleep(0.02)
    assert sb.ended == ["exec-1"], f"end_exec not called once: {sb.ended}"
    assert None in lines, "the reader did not reap (no end-of-stream)"
    assert sb.box_.busy == 0, "the box slot was not released"


def test_session_close_goes_through_the_closer():
    """Mcp's drop path: Session.close -> StdioTransport.close -> the same closer."""
    sb = FakeSandboxes()
    t = mcp_mod.StdioTransport(mcp_mod.exec_pipe(sb, "c1", "server", [], {}))
    s = mcp_mod.Session(t)
    s.close()
    assert sb.ended == ["exec-1"]


def test_a_failing_end_exec_ends_nothing_further():
    """drop() calls the closer for every server of a chat, under a lock, and session()
    marks the server failed after it. An OSError escaping here stopped drop() halfway
    and left the server unmarked: the closer logs it and closes the socket anyway."""
    class Raising(FakeSandboxes):
        def end_exec(self, exec_id):
            FakeSandboxes.end_exec(self, exec_id)
            raise OSError("engine gone mid-kill")
    sb = Raising()
    _write, _start, closer = mcp_mod.exec_pipe(sb, "c1", "server", [], {})(lambda line: None)
    closer()                                                     # must not raise
    assert sb.ended == ["exec-1"], "end_exec did not run"
    assert sb.sock.fileno() == -1, "a failing end_exec still skipped sock.close()"


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    n = ok = 0
    for name, fn in tests:
        n += 1
        try:
            fn()
            print(f"  {name:56s} ok")
            ok += 1
        except Exception as exc:                                       # noqa: BLE001
            print(f"  {name:56s} FAIL {type(exc).__name__}: {exc}")
    print(f"{ok}/{n}")
    sys.exit(0 if ok == n else 1)
