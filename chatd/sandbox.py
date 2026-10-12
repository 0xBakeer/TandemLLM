"""One sandbox container per chat, and the memory guard that keeps them away from the engine
(SRV-355, Chat 6).

THE RUNTIME. chatd speaks the Docker Engine API over a unix socket (`QSE_SANDBOX_SOCKET`, default
/var/run/docker.sock). Podman serves the same API, so the plan's rootless Podman under its own user
(`ops/sandbox/install.sh`, which needs root once) is a socket path away: nothing here changes.
On the Spark today the socket is rootful Docker, which is why every container is locked down here
rather than relying on the runtime:

- `tandem-chat-<id>` from `tandem-sandbox:1` (ops/sandbox/Containerfile), as uid 1000 (`agent`),
  every capability dropped, no-new-privileges, the default seccomp profile, a read-only root, tmpfs
  /tmp (512 MB), no devices and no GPU;
- `--network none`: the container has a loopback and nothing else. Its internet goes through
  chatd's filtering proxy for that chat only (chatd/egress.py), which refuses the LAN and this box;
- mounts: /workspace (the chat's files, chatd/workspace.py), /home/agent (pip, npm, uv caches) and
  /run/tandem (the proxy's socket, read-only). Nothing of chatd's or the engine's is mounted;
- limits: 4 GiB memory with no swap, 4 CPUs, 512 pids (a fork bomb stops there).

THE MEMORY GUARD (the engine comes first; unified memory makes a sandbox's RAM the engine's KV):
- admission: a sandbox starts only when MemAvailable minus its limit stays at or above 19 GiB
  (floor 15 + margin 4), and MemAvailable is at least QSE_SANDBOX_MIN_START_GB (0 by default; 30 on
  the Spark while the Flash-Next builder runs its memory fits). Else the chat waits ("Waiting for
  memory") until it fits or the run is stopped;
- the running sandboxes' limits add up to at most QSE_SANDBOX_TOTAL_GB (16), and at most
  QSE_SANDBOX_MAX_RUNNING run at once (the least recently used idle one is stopped to make room);
- every second: under 15 GiB, stop the least recently used idle sandbox; under 12 GiB, kill every
  sandbox. Stop, not pause: pausing frees nothing. The files stay; the chat says why
  (`stopped_reason: memory_guard`);
- 30 minutes with no tool call and no terminal: stop (`idle`).

chatd restarts adopt running containers by their label and do not recreate them.
"""

from __future__ import annotations

import http.client
import json
import os
import shutil
import socket
import struct
import threading
import time
import urllib.parse

from chatd.egress import Proxy

GiB = 1024 ** 3
IMAGE = os.environ.get("QSE_SANDBOX_IMAGE", "tandem-sandbox:1")
SOCKET = os.environ.get("QSE_SANDBOX_SOCKET", "/var/run/docker.sock")
LIMIT_BYTES = 4 * GiB
CPUS = 4
PIDS = 512
DISK_LIMIT = 10 * GiB
FLOOR_GB, MARGIN_GB, STOP_GB, KILL_GB = 15.0, 4.0, 15.0, 12.0
IDLE_S = 30 * 60
LABEL = "tandem.chat"


class Unavailable(Exception):
    """No sandbox for this call: the image is missing, the runtime is down, or the run stopped
    while it waited for memory."""


def meminfo(path: str = "/proc/meminfo") -> dict:
    out = {}
    try:
        with open(path) as fh:
            for line in fh:
                k, _, v = line.partition(":")
                out[k.strip()] = int(v.split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return out


class _UnixConn(http.client.HTTPConnection):
    def __init__(self, path: str, timeout: float):
        super().__init__("localhost", timeout=timeout)
        self.unix_path = path

    def connect(self):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(self.timeout)
        s.connect(self.unix_path)
        self.sock = s


class Engine:
    """The container engine's API (Docker's, which Podman also serves)."""

    def __init__(self, path: str = SOCKET):
        self.path = path

    def call(self, method: str, path: str, body=None, timeout: float = 60.0):
        c = _UnixConn(self.path, timeout)
        try:
            data = json.dumps(body).encode() if body is not None else None
            c.request(method, path, body=data,
                      headers={"Content-Type": "application/json"} if data else {})
            r = c.getresponse()
            raw = r.read()
            try:
                return r.status, json.loads(raw) if raw else None
            except ValueError:
                return r.status, raw.decode("utf-8", "replace")
        finally:
            c.close()

    def hijack(self, path: str, body: dict) -> tuple[socket.socket, bytes]:
        """POST that upgrades to the raw stream (exec start with attach)."""
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(30)
        s.connect(self.path)
        data = json.dumps(body).encode()
        s.sendall(f"POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n"
                  f"Connection: Upgrade\r\nUpgrade: tcp\r\nContent-Length: {len(data)}\r\n\r\n"
                  .encode() + data)
        head = b""
        while b"\r\n\r\n" not in head:
            d = s.recv(4096)
            if not d:
                raise OSError("the engine closed the exec stream")
            head += d
        status = head.split(b" ", 2)[1] if b" " in head else b"0"
        if status not in (b"101", b"200"):
            s.close()
            raise OSError(f"exec start answered {status.decode()}")
        s.settimeout(None)
        return s, head.split(b"\r\n\r\n", 1)[1]


def demux(sock: socket.socket, first: bytes, on_data) -> None:
    """A non-TTY attach stream: 8-byte frames [stream, 0, 0, 0, size(4, big-endian)]."""
    buf = first
    while True:
        while len(buf) >= 8:
            stream, size = buf[0], struct.unpack(">I", buf[4:8])[0]
            if len(buf) < 8 + size:
                break
            on_data(stream, buf[8:8 + size])
            buf = buf[8 + size:]
        try:
            d = sock.recv(65536)
        except OSError:
            return
        if not d:
            return
        buf += d


class Box:
    """What chatd knows of one chat's sandbox."""

    def __init__(self, cid: str):
        self.cid = cid
        self.container: str | None = None
        self.state = "stopped"               # running, stopped, creating, waiting_memory, error
        self.stopped_reason: str | None = None
        self.started = 0.0
        self.last_used = 0.0
        self.busy = 0                        # execs and attached terminals
        self.error: str | None = None
        self.proxy: Proxy | None = None
        self.lock = threading.RLock()


class Sandboxes:
    def __init__(self, data_dir: str, *, engine: Engine | None = None, image: str = IMAGE,
                 log=None, on_change=None, network_mode=None, meminfo_fn=meminfo, clock=time.time):
        self.data = data_dir
        self.engine = engine or Engine()
        self.image = image
        self.log = log or (lambda line: None)
        self.on_change = on_change or (lambda cid, status: None)
        self.network_mode = network_mode or (lambda cid: "on")
        self.meminfo = meminfo_fn
        self.clock = clock
        self.boxes: dict[str, Box] = {}
        self.lock = threading.Lock()
        self.min_start_gb = float(os.environ.get("QSE_SANDBOX_MIN_START_GB") or 0)
        self.total_gb = float(os.environ.get("QSE_SANDBOX_TOTAL_GB") or 16)
        self.max_running = int(os.environ.get("QSE_SANDBOX_MAX_RUNNING") or 4)
        self._avail: tuple[float, bool] | None = None
        self.stopping = threading.Event()

    # ------------------------------------------------------------------ availability
    def available(self) -> bool:
        """The image exists and the engine answers (cached for 60 s)."""
        if self._avail is not None and self.clock() - self._avail[0] < 60:
            return self._avail[1]
        try:
            st, _ = self.engine.call("GET", f"/images/{urllib.parse.quote(self.image, safe='')}"
                                     f"/json", timeout=5)
            ok = st == 200
        except OSError:
            ok = False
        self._avail = (self.clock(), ok)
        return ok

    def box(self, cid: str) -> Box:
        with self.lock:
            b = self.boxes.get(cid)
            if b is None:
                b = self.boxes[cid] = Box(cid)
            return b

    def name(self, cid: str) -> str:
        return "tandem-chat-" + cid.lower()

    def paths(self, cid: str) -> tuple[str, str, str]:
        return (os.path.join(self.data, "workspaces", cid), os.path.join(self.data, "homes", cid),
                os.path.join(self.data, "egress", cid))

    # ------------------------------------------------------------------ adopt, status
    def adopt(self) -> int:
        """At start: the containers a previous chatd left running keep running, and come back."""
        try:
            st, rows = self.engine.call("GET", "/containers/json?all=1&filters=" +
                                        urllib.parse.quote(json.dumps({"label": [LABEL]})))
        except OSError:
            return 0
        n = 0
        for r in rows if st == 200 and isinstance(rows, list) else []:
            cid = (r.get("Labels") or {}).get(LABEL)
            if not cid:
                continue
            b = self.box(cid)
            b.container = r.get("Id")
            if r.get("State") == "running":
                b.state, b.started, b.last_used = "running", self.clock(), self.clock()
                self._proxy(b)
                n += 1
        return n

    def status(self, cid: str) -> dict:
        b = self.box(cid)
        ws, _, _ = self.paths(cid)
        mode = self.network_mode(cid)
        return {"state": b.state if self.available() or b.state == "running" else "error",
                "image": self.image, "cwd": "/workspace", "cpus": CPUS,
                "memory_bytes": 0, "memory_limit_bytes": LIMIT_BYTES,
                "disk_bytes": _du(ws), "disk_limit_bytes": DISK_LIMIT,
                "network": "on" if mode == "on" else "off", "allowlist": [],
                "uptime_s": int(self.clock() - b.started) if b.state == "running" else 0,
                "error": b.error or (None if self.available() else
                                     "the sandbox image is not installed on this box"),
                "stopped_reason": b.stopped_reason, "ports": []}

    def _changed(self, b: Box) -> None:
        try:
            self.on_change(b.cid, self.status(b.cid))
        except Exception:                                           # noqa: BLE001
            pass

    # ------------------------------------------------------------------ start
    def _running(self) -> list[Box]:
        with self.lock:
            return [b for b in self.boxes.values() if b.state == "running"]

    def admit(self) -> str | None:
        """None when a new sandbox may start now, else the reason to wait."""
        avail = self.meminfo().get("MemAvailable", 0) / GiB
        if avail and avail < self.min_start_gb:
            return f"MemAvailable {avail:.1f} GiB is under the {self.min_start_gb:.0f} GiB start line"
        if avail and avail - LIMIT_BYTES / GiB < FLOOR_GB + MARGIN_GB:
            return f"MemAvailable {avail:.1f} GiB leaves the engine under {FLOOR_GB + MARGIN_GB:.0f}"
        running = self._running()
        if (len(running) + 1) * LIMIT_BYTES / GiB > self.total_gb or \
                len(running) >= self.max_running:
            idle = sorted((b for b in running if not b.busy), key=lambda b: b.last_used)
            if idle:
                self.stop(idle[0].cid, "idle")
                return None if len(self._running()) < self.max_running else "making room"
            return f"{len(running)} sandbox(es) running, the most at once"
        return None

    def ensure(self, cid: str, cancel: threading.Event | None = None, on_wait=None) -> str:
        """The running container's id: created or started when needed, after admission."""
        if not self.available():
            raise Unavailable("the sandbox image is not installed on this box "
                              "(ops/sandbox/build.sh)")
        b = self.box(cid)
        with b.lock:
            b.last_used = self.clock()
            if b.state == "running" and b.container:
                return b.container
            waited = False
            while True:
                why = self.admit()
                if why is None:
                    break
                if not waited:
                    b.state, waited = "waiting_memory", True
                    self._changed(b)
                if on_wait is not None:
                    on_wait(why)
                if (cancel is not None and cancel.wait(2.0)) or self.stopping.is_set():
                    b.state = "stopped"
                    self._changed(b)
                    raise Unavailable("stopped while waiting for memory")
                if cancel is None:
                    time.sleep(2.0)
            b.state, b.error = "creating", None
            self._changed(b)
            try:
                b.container = self._start(cid)
            except Unavailable as exc:
                b.state, b.error = "error", str(exc)
                self._changed(b)
                raise
            b.state, b.stopped_reason = "running", None
            b.started = b.last_used = self.clock()
            self._proxy(b)
            self._changed(b)
            self.log(f"[sandbox] {cid} running")
            return b.container

    def _proxy(self, b: Box) -> None:
        if b.proxy is None:
            _, _, eg = self.paths(b.cid)
            b.proxy = Proxy(os.path.join(eg, "proxy.sock"), lambda: self.network_mode(b.cid),
                            self.log)
            b.proxy.start()

    def _start(self, cid: str) -> str:
        ws, home, eg = self.paths(cid)
        for d in (ws, home, eg):
            os.makedirs(d, exist_ok=True)
        name = self.name(cid)
        st, info = self.engine.call("GET", f"/containers/{name}/json")
        if st == 404:
            proxy = "http://127.0.0.1:3128"
            body = {
                "Image": self.image, "User": "1000:1000", "WorkingDir": "/workspace",
                "Cmd": ["python3", "/opt/tandem/relay.py", "serve"],
                "Env": ["HOME=/home/agent", "TERM=xterm-256color", "LANG=C.UTF-8",
                        "PATH=/home/agent/.local/bin:/home/agent/.npm-global/bin:/usr/local/sbin:"
                        "/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                        f"HTTP_PROXY={proxy}", f"HTTPS_PROXY={proxy}", f"http_proxy={proxy}",
                        f"https_proxy={proxy}", "NO_PROXY=localhost,127.0.0.1",
                        "no_proxy=localhost,127.0.0.1", "npm_config_prefix=/home/agent/.npm-global",
                        "PIP_DISABLE_PIP_VERSION_CHECK=1", "PYTHONUNBUFFERED=1"],
                "Labels": {LABEL: cid},
                "HostConfig": {
                    "Binds": [f"{ws}:/workspace:rw", f"{home}:/home/agent:rw",
                              f"{eg}:/run/tandem:ro"],
                    "NetworkMode": "none", "Memory": LIMIT_BYTES, "MemorySwap": LIMIT_BYTES,
                    "NanoCpus": CPUS * 10 ** 9, "PidsLimit": PIDS, "CapDrop": ["ALL"],
                    "SecurityOpt": ["no-new-privileges"], "ReadonlyRootfs": True,
                    "Tmpfs": {"/tmp": "rw,nosuid,nodev,size=512m"}, "Init": True,
                    "Devices": [], "Privileged": False},
            }
            st, info = self.engine.call("POST", f"/containers/create?name={name}", body)
            if st not in (200, 201):
                raise Unavailable(f"the sandbox could not be created ({st}: "
                                  f"{str((info or {}).get('message') if isinstance(info, dict) else info)[:200]})")
            container = info["Id"]
        elif st == 200:
            container = info["Id"]
        else:
            raise Unavailable(f"the container engine answered {st}")
        st, info = self.engine.call("POST", f"/containers/{container}/start")
        if st not in (204, 304):
            raise Unavailable(f"the sandbox did not start ({st})")
        # a fresh workspace is a git repository, for the agent's own history
        self.run(container, ["bash", "-lc", "[ -d .git ] || git init -q -b main . 2>/dev/null || "
                             "true"], timeout=20)
        return container

    # ------------------------------------------------------------------ stop, reset
    def stop(self, cid: str, reason: str = "user") -> None:
        b = self.box(cid)
        if b.container and b.state == "running":
            try:
                self.engine.call("POST", f"/containers/{b.container}/stop?t=3", timeout=30)
            except OSError:
                pass
            self.log(f"[sandbox] {cid} stopped ({reason})")
        b.state, b.stopped_reason = "stopped", reason
        if b.proxy is not None:
            b.proxy.close()
            b.proxy = None
        self._changed(b)

    def kill_all(self, reason: str = "memory_guard") -> None:
        for b in self._running():
            try:
                self.engine.call("POST", f"/containers/{b.container}/kill", timeout=10)
            except OSError:
                pass
            b.state, b.stopped_reason = "stopped", reason
            if b.proxy is not None:
                b.proxy.close()
                b.proxy = None
            self._changed(b)
        self.log(f"[sandbox] every sandbox killed ({reason})")

    def remove(self, cid: str, files: bool = False) -> None:
        self.stop(cid, "user")
        b = self.box(cid)
        try:
            self.engine.call("DELETE", f"/containers/{self.name(cid)}?force=1", timeout=30)
        except OSError:
            pass
        b.container = None
        if files:
            for d in self.paths(cid):
                shutil.rmtree(d, ignore_errors=True)

    # ------------------------------------------------------------------ exec
    def open_exec(self, cid: str, cmd: list[str], *, tty: bool = False, stdin: bool = False,
                  env: list[str] | None = None, workdir: str = "/workspace"
                  ) -> tuple[str, socket.socket, bytes]:
        container = self.box(cid).container
        st, info = self.engine.call("POST", f"/containers/{container}/exec", {
            "AttachStdin": stdin, "AttachStdout": True, "AttachStderr": True, "Tty": tty,
            "Cmd": cmd, "WorkingDir": workdir, "User": "1000:1000", "Env": env or []})
        if st != 201:
            raise Unavailable(f"exec failed ({st})")
        exec_id = info["Id"]
        sock, first = self.engine.hijack(f"/exec/{exec_id}/start", {"Detach": False, "Tty": tty})
        return exec_id, sock, first

    def exit_code(self, exec_id: str) -> tuple[bool, int | None, int | None]:
        st, info = self.engine.call("GET", f"/exec/{exec_id}/json", timeout=10)
        if st != 200 or not isinstance(info, dict):
            return False, None, None
        return bool(info.get("Running")), info.get("ExitCode"), info.get("Pid")

    def run(self, container_or_cid: str, cmd: list[str], *, timeout: float = 600,
            on_output=None, cancel: threading.Event | None = None) -> tuple[int, bool]:
        """Run `cmd` to its end; (exit code, timed out). Output goes to `on_output(text)`."""
        cid = next((c for c, b in self.boxes.items() if b.container == container_or_cid),
                   container_or_cid)
        b = self.box(cid)
        b.busy += 1
        b.last_used = self.clock()
        try:
            exec_id, sock, first = self.open_exec(cid, ["setsid", "--wait", *cmd])
            done = threading.Event()

            def pump():
                demux(sock, first, lambda s, d: on_output and on_output(
                    d.decode("utf-8", "replace")))
                done.set()
            threading.Thread(target=pump, daemon=True).start()
            t0, timed_out = time.monotonic(), False
            while not done.wait(0.2):
                if time.monotonic() - t0 > timeout or (cancel is not None and cancel.is_set()):
                    timed_out = time.monotonic() - t0 > timeout
                    self._kill_exec(exec_id)
                    done.wait(5)
                    break
            try:
                sock.close()
            except OSError:
                pass
            for _ in range(20):
                running, code, _ = self.exit_code(exec_id)
                if not running:
                    return (code if code is not None else -1), timed_out
                time.sleep(0.1)
            return -1, timed_out
        finally:
            b.busy -= 1
            b.last_used = self.clock()

    def _kill_exec(self, exec_id: str) -> None:
        """The exec's process group (setsid made it a leader): TERM, then KILL. Without user
        namespaces its pid is a pid of this host, owned by chatd's user."""
        _, _, pid = self.exit_code(exec_id)
        if not pid:
            return
        import signal
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(pid, sig)
            except (ProcessLookupError, PermissionError):
                return
            time.sleep(2)

    # ------------------------------------------------------------------ the guard
    def guard_once(self) -> str | None:
        """One look: what it did, for the tests and the log."""
        avail = self.meminfo().get("MemAvailable", 0) / GiB
        running = self._running()
        if not avail or not running:
            return None
        if avail < KILL_GB:
            self.kill_all("memory_guard")
            return "kill"
        if avail < STOP_GB:
            idle = sorted((b for b in running if not b.busy), key=lambda b: b.last_used)
            victim = idle[0] if idle else sorted(running, key=lambda b: b.last_used)[0]
            self.stop(victim.cid, "memory_guard")
            return f"stop {victim.cid}"
        now = self.clock()
        for b in running:
            if not b.busy and now - b.last_used > IDLE_S:
                self.stop(b.cid, "idle")
                return f"idle {b.cid}"
        return None

    def guard(self) -> None:
        while not self.stopping.wait(1.0):
            try:
                self.guard_once()
            except Exception as exc:                                # noqa: BLE001
                self.log(f"[sandbox] guard {type(exc).__name__}")

    def touch(self, cid: str) -> None:
        self.box(cid).last_used = self.clock()


def _du(path: str) -> int:
    total = 0
    for dirpath, _, files in os.walk(path, followlinks=False):
        for f in files:
            try:
                total += os.lstat(os.path.join(dirpath, f)).st_size
            except OSError:
                pass
    return total
