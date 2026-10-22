"""Free TCP ports for the tests' servers (OPS-231).

A port picked with bind(("127.0.0.1", 0)), closed and handed to a server that binds it later is free only for that
moment: with the suite's files running side by side, another file's server or client could take it in between
(test_disk_cache_server failed so on a hosted runner and passed alone). Here every test process takes the next
port of one counter kept under a lock file in the system temp dir, below the kernels' ephemeral ranges (which
bind(0) and outgoing connections draw from), and checks it with a bind: two tests never get the same port.
"""

from __future__ import annotations

import fcntl
import os
import socket
import tempfile

LOW, SPAN = 20_000, 12_000          # 20000-31999: under Linux's 32768 and macOS's 49152


def free_port() -> int:
    path = os.path.join(tempfile.gettempdir(), "tandem-test-ports")
    with open(path + ".lock", "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            with open(path) as f:
                n = int(f.read().strip() or 0)
        except (OSError, ValueError):
            n = 0
        for _ in range(SPAN):
            n = (n + 1) % SPAN
            port = LOW + n
            s = socket.socket()
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
            finally:
                s.close()
            with open(path, "w") as f:
                f.write(str(n))
            return port
    raise RuntimeError(f"no free port in {LOW}-{LOW + SPAN - 1}")
