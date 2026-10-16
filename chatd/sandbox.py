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
- limits: 1 GiB memory with no swap (QSE_SANDBOX_MEM_GB), 4 CPUs, 512 pids (a fork bomb stops
  there). A process the kernel kills at the limit is reported to the model as "out of memory (1 GiB
  sandbox limit)" (`oom_kills`, the cgroup's memory.events).

THE MEMORY GUARD. The sandbox has priority over the engine's idle memory (ruling of 2026-10-15): a
chat's tool call gets its sandbox, and when memory is short the engine gives way. The engine keeps
itself at its own floors: its KV pages pass the pool's hard floor and its requests the admission
floor (10 GiB each on the Spark), and below them it gives back idle conversations or waits. What a
sandbox must leave is only what the engine takes without asking: a prefill's workspace, beyond
the free blocks its allocator already holds. Measured on the Spark (27B EXL3 6.0, 4 slots, KV pool,
2026-10-15, the engine's own `last_prefill_peak` and `observed.workspace`): 2.7 GiB for a
203,052-token prefill, 3.5 GiB scaled to the 262,144-token window; the allocator held 10 GiB of
free blocks at the time. So:
- admission: a sandbox starts when MemAvailable minus its limit stays at or above FLOOR_GB (4,
  QSE_SANDBOX_FLOOR_GB). When it does not, chatd asks the engine to give back the difference
  (`release`, the engine's POST /v1/cache/release: free allocator blocks, caches, the small models,
  idle conversations) and looks again;
- the running sandboxes' limits add up to at most QSE_SANDBOX_TOTAL_GB (QSE_SANDBOX_MAX_RUNNING
  times the limit), and at most QSE_SANDBOX_MAX_RUNNING (4) run at once (the least recently used
  idle one is stopped to make room);
- a call waits at most WAIT_S (60 s, QSE_SANDBOX_WAIT_S). Then it fails with the reason and the
  numbers, which the model reads as the tool's result and the chat shows on the call: it never
  hangs;
- every second: under STOP_GB (3) the engine is asked first, then the least recently used idle
  sandbox stops (a busy one when none is idle); under KILL_GB (2) every sandbox is killed. Both
  are under the admission's floor, so the guard never stops a sandbox the admission just let in.
  Stop, not pause: pausing frees nothing. The files stay; the chat says why (`stopped_reason:
  memory_guard`);
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
LIMIT_BYTES = int(float(os.environ.get("QSE_SANDBOX_MEM_GB") or 1) * GiB)
CPUS = 4
PIDS = 512
DISK_LIMIT = 10 * GiB
FLOOR_GB = float(os.environ.get("QSE_SANDBOX_FLOOR_GB") or 4)
STOP_GB, KILL_GB = 3.0, 2.0
WAIT_S = float(os.environ.get("QSE_SANDBOX_WAIT_S") or 60)
RELEASE_EVERY_S = 10.0
IDLE_S = 30 * 60
CGROUP = "/sys/fs/cgroup"
MEM_TTL_S = 5.0
LABEL = "tandem.chat"


class Unavailable(Exception):
    """No sandbox for this call: the image is missing, the runtime is down, the run stopped while
    it waited for memory, or memory stayed short for WAIT_S. The message is the tool result the
    model reads; `data` is the call's `output.data` for the chat (the reason and the numbers)."""

    def __init__(self, message: str, data: dict | None = None):
        super().__init__(message)
        self.data = data


def limit_words() -> str:
    return f"{LIMIT_BYTES / GiB:g} GiB"


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
        self.mem: tuple[float, int | None] = (0.0, None)   # (when read, bytes in use)


class Sandboxes:
    def __init__(self, data_dir: str, *, engine: Engine | None = None, image: str = IMAGE,
                 log=None, on_change=None, network_mode=None, meminfo_fn=meminfo, clock=time.time,
                 release=None):
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
        self.max_running = int(os.environ.get("QSE_SANDBOX_MAX_RUNNING") or 4)
        self.total_gb = float(os.environ.get("QSE_SANDBOX_TOTAL_GB") or
                              self.max_running * LIMIT_BYTES / GiB)
        # `release(nbytes) -> bytes given back`: the engine makes room (chatd/engine.py
        # `release_memory`); None where there is no engine to ask
        self.release = release
        self.released = 0                    # bytes the engine gave back at the last ask
        self._asked: float | None = None
        self.wait_s, self.poll_s, self.release_every_s = WAIT_S, 2.0, RELEASE_EVERY_S
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

    def memory_used(self, b: Box) -> int | None:
        """A running container's memory in use, as `docker stats` shows it (the cgroup's usage
        less its inactive file cache), read at most every MEM_TTL_S; None when the runtime does
        not say. It was a constant 0 before: the Context panel showed "running, 0 B"."""
        if b.state != "running" or not b.container:
            return None
        now = self.clock()
        if now - b.mem[0] < MEM_TTL_S:
            return b.mem[1]
        used = None
        try:
            st, s = self.engine.call("GET", f"/containers/{b.container}/stats?stream=false&"
                                     "one-shot=true", timeout=3)
            m = (s or {}).get("memory_stats") if st == 200 and isinstance(s, dict) else None
            if isinstance(m, dict) and m.get("usage") is not None:
                stats = m.get("stats") or {}
                cache = stats.get("inactive_file", stats.get("total_inactive_file", 0)) or 0
                used = max(0, int(m["usage"]) - int(cache))
        except (OSError, ValueError, TypeError):
            used = None
        b.mem = (now, used)
        return used

    def oom_kills(self, cid: str) -> int | None:
        """How many processes the kernel has killed at this sandbox's memory limit: `oom_kill` in
        its cgroup's memory.events (cgroup v2, Docker's systemd or cgroupfs layout); None where
        the file cannot be read (another runtime, cgroup v1)."""
        ctr = self.box(cid).container
        if not ctr:
            return None
        for path in (f"{CGROUP}/system.slice/docker-{ctr}.scope/memory.events",
                     f"{CGROUP}/docker/{ctr}/memory.events"):
            try:
                with open(path) as fh:
                    for line in fh:
                        k, _, v = line.partition(" ")
                        if k == "oom_kill":
                            return int(v)
            except (OSError, ValueError):
                continue
        return None

    def status(self, cid: str) -> dict:
        b = self.box(cid)
        ws, _, _ = self.paths(cid)
        mode = self.network_mode(cid)
        return {"state": b.state if self.available() or b.state == "running" else "error",
                "image": self.image, "cwd": "/workspace", "cpus": CPUS,
                "memory_bytes": self.memory_used(b), "memory_limit_bytes": LIMIT_BYTES,
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

    def _avail_gb(self) -> float:
        return self.meminfo().get("MemAvailable", 0) / GiB

    def _ask_engine(self, short_gb: float) -> int:
        """The engine gives way: it is asked for `short_gb` (at most every RELEASE_EVERY_S, since
        what it holds idle does not come back between two looks). Bytes it gave back."""
        now = time.monotonic()
        if self.release is None or (self._asked is not None and
                                    now - self._asked < self.release_every_s):
            return 0
        self._asked = now
        try:
            got = int(self.release(int(short_gb * GiB)) or 0)
        except Exception:                                           # noqa: BLE001
            got = 0
        self.released = got
        if got:
            self.log(f"[sandbox] the engine gave back {got / GiB:.1f} GiB for a sandbox")
        return got

    def short_words(self, avail: float) -> str:
        return (f"MemAvailable {avail:.1f} GiB: a {limit_words()} sandbox would leave the engine "
                f"under its {FLOOR_GB:g} GiB floor")

    def admit(self) -> str | None:
        """None when a new sandbox may start now, else the reason to wait. Short of memory, the
        engine is asked to give back the difference first."""
        avail = self._avail_gb()
        short = FLOOR_GB + LIMIT_BYTES / GiB - avail
        if avail and short > 0:
            if self._ask_engine(short):
                avail = self._avail_gb()
            if avail and avail - LIMIT_BYTES / GiB < FLOOR_GB:
                return self.short_words(avail)
        running = self._running()
        if (len(running) + 1) * LIMIT_BYTES / GiB > self.total_gb or \
                len(running) >= self.max_running:
            idle = sorted((b for b in running if not b.busy), key=lambda b: b.last_used)
            if idle:
                self.stop(idle[0].cid, "idle")
                return None if len(self._running()) < self.max_running else "making room"
            return (f"{len(running)} sandbox(es) of other chats are running commands, the most "
                    f"at once")
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
            waited, t0 = False, time.monotonic()
            while True:
                why = self.admit()
                if why is None:
                    break
                spent = time.monotonic() - t0
                if not waited:
                    b.state, waited = "waiting_memory", True
                    self._changed(b)
                if spent >= self.wait_s:
                    raise self._gave_up(b, why, spent)
                if on_wait is not None:
                    on_wait(f"{why}; {spent:.0f} of {self.wait_s:.0f} s")
                if (cancel is not None and cancel.wait(self.poll_s)) or self.stopping.is_set():
                    b.state = "stopped"
                    self._changed(b)
                    raise Unavailable("stopped while waiting for memory")
                if cancel is None:
                    time.sleep(self.poll_s)
            b.state, b.error = "creating", None
            self._changed(b)
            try:
                b.container = self._start(cid)
                # a fresh workspace is a git repository, for the agent's own history
                self.run(cid, ["bash", "-lc", "[ -d .git ] || git init -q . 2>/dev/null; true"],
                         timeout=20)
            except (Unavailable, OSError) as exc:
                b.state, b.error = "error", str(exc)
                self._changed(b)
                raise Unavailable(str(exc)) from None
            b.state, b.stopped_reason = "running", None
            b.started = b.last_used = self.clock()
            self._proxy(b)
            self._changed(b)
            self.log(f"[sandbox] {cid} running")
            return b.container

    def _gave_up(self, b: Box, why: str, spent: float) -> Unavailable:
        """The wait's end: the reason and the numbers, for the model (the tool's result), the
        chat (the call's `output.data`) and the Sandbox panel (`error`)."""
        avail = self._avail_gb()
        freed = self.released / GiB
        b.state, b.error = "stopped", f"did not start after {spent:.0f} s: {why}"
        self._changed(b)
        self.log(f"[sandbox] {b.cid} not started after {spent:.0f} s (MemAvailable {avail:.1f} "
                 f"GiB, the engine gave back {freed:.1f} GiB)")
        memory = why.startswith("MemAvailable")
        msg = (f"no sandbox for this call: after waiting {spent:.0f} s it still could not start "
               f"({why}" + (f"; the engine gave back {freed:.1f} GiB" if memory else "") +
               "). The command did not run. Do not call bash or render_page again for now: answer "
               "with what you already have, or without tools, and tell the user the sandbox could "
               "not start " + ("for lack of memory." if memory else "because other chats' "
                                                                     "sandboxes are busy."))
        return Unavailable(msg, {"sandbox": {
            "reason": "no_memory" if memory else "busy", "why": why,
            "waited_s": round(spent), "mem_available_gib": round(avail, 1),
            "limit_gib": LIMIT_BYTES / GiB, "floor_gib": FLOOR_GB,
            "engine_released_gib": round(freed, 1)}})

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
        cid = container_or_cid
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

    def end_exec(self, exec_id: str) -> None:
        """TERM to an exec's own process (not its group) when it still runs."""
        running, _, pid = self.exit_code(exec_id)
        if running and pid:
            import signal
            try:
                os.kill(pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass

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
            # the engine first: what it holds idle goes before a sandbox does
            if self._ask_engine(STOP_GB + LIMIT_BYTES / GiB - avail) and \
                    self._avail_gb() >= STOP_GB:
                return "engine released"
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
    """The bytes of the files under `path`, a tree the sandbox writes into: os.fwalk walks by
    fds, so a folder swapped for a symlink mid-walk is not entered (os.walk would list what it
    points to, the host's / included)."""
    total = 0
    if not os.path.isdir(path):
        return 0
    for _, _, files, dirfd in os.fwalk(path, follow_symlinks=False):
        for f in files:
            try:
                total += os.stat(f, dir_fd=dirfd, follow_symlinks=False).st_size
            except OSError:
                pass
    return total
