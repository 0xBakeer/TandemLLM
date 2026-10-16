"""The sandbox's one network helper, baked into tandem-sandbox:1 at /opt/tandem/relay.py.

The container runs with no network but its loopback. Two jobs:

    relay.py serve          (the container's main process) listen on 127.0.0.1:3128 and pass every
                            connection to chatd's proxy for this chat, /run/tandem/proxy.sock
                            (HTTP_PROXY and HTTPS_PROXY point here; chatd/egress.py checks where
                            each one goes: the internet yes, the LAN and the host no)
    relay.py connect PORT   (an exec per request) connect to localhost:PORT and pipe stdin to it
                            and its answer to stdout: chatd's way to a dev server in the sandbox
                            for the preview pane. ::1, then 127.0.0.1, by address (not /etc/hosts).
                            Live (2026-10-16) Vite 5 on `localhost` listened on [::1]:5173 only
                            (Node 17+ takes the resolver's order) and a dial to 127.0.0.1 alone
                            was refused while curl in the sandbox answered 200
    relay.py ports          the TCP ports something listens on in the sandbox, as JSON: [{port,
                            addresses, command}] (render_page's check and its error)

chatd sends this file with each exec (`python3 -c <this file> connect PORT`, chatd/sandbox.py
`relay_cmd`), so a container made from an older image dials the same way.
"""

import json
import os
import socket
import sys
import threading

SOCK = "/run/tandem/proxy.sock"


def pump(src, dst):
    try:
        while True:
            d = src.recv(65536)
            if not d:
                break
            dst.sendall(d)
    except OSError:
        pass
    try:
        dst.shutdown(socket.SHUT_WR)
    except OSError:
        pass


def handle(c):
    u = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        u.connect(SOCK)
    except OSError:
        msg = b"chatd's proxy is not there; the network is off for this sandbox\n"
        c.sendall(b"HTTP/1.1 502 Bad Gateway\r\nContent-Type: text/plain\r\nContent-Length: " +
                  str(len(msg)).encode() + b"\r\nConnection: close\r\n\r\n" + msg)
        c.close()
        return
    t = threading.Thread(target=pump, args=(u, c), daemon=True)
    t.start()
    pump(c, u)
    t.join(600)
    c.close()
    u.close()


def serve():
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", 3128))
    s.listen(128)
    while True:
        c, _ = s.accept()
        threading.Thread(target=handle, args=(c,), daemon=True).start()


def _addr(hexaddr):
    b = bytes.fromhex(hexaddr)
    if len(b) == 4:
        return socket.inet_ntop(socket.AF_INET, b[::-1])
    return socket.inet_ntop(socket.AF_INET6, b"".join(b[i:i + 4][::-1] for i in range(0, 16, 4)))


def listening(proc="/proc"):
    """[{port, addresses, command}] by port: the listening TCP sockets of /proc/net/tcp and tcp6,
    each with the command line of the process that holds it (when it is ours to read). The relay's
    own proxy port is left out."""
    socks = {}
    for name in ("tcp", "tcp6"):
        try:
            with open(os.path.join(proc, "net", name)) as fh:
                rows = fh.read().splitlines()[1:]
        except OSError:
            continue
        for row in rows:
            f = row.split()
            if len(f) > 9 and f[3] == "0A":
                a, p = f[1].split(":")
                socks[f[9]] = (int(p, 16), _addr(a))
    owners = {}
    try:
        pids = os.listdir(proc)
    except OSError:
        pids = []
    for pid in pids:
        if not pid.isdigit():
            continue
        try:
            fds = os.listdir(os.path.join(proc, pid, "fd"))
        except OSError:
            continue
        for fd in fds:
            try:
                link = os.readlink(os.path.join(proc, pid, "fd", fd))
            except OSError:
                continue
            ino = link[8:-1] if link.startswith("socket:[") else None
            if ino in socks and ino not in owners:
                try:
                    with open(os.path.join(proc, pid, "cmdline"), "rb") as fh:
                        raw = fh.read().replace(b"\0", b" ")
                    owners[ino] = raw.decode("utf-8", "replace").strip()
                except OSError:
                    pass
    out = {}
    for ino, (port, addr) in socks.items():
        cmd = owners.get(ino, "")
        if "relay.py serve" in cmd:
            continue
        row = out.setdefault(port, {"port": port, "addresses": [], "command": ""})
        if addr not in row["addresses"]:
            row["addresses"].append(addr)
        row["command"] = row["command"] or cmd[:120]
    return [out[p] for p in sorted(out)]


def ports_words(ports):
    if not ports:
        return "nothing listens on any port in the sandbox"
    return "listening now: " + ", ".join(
        f"{r['port']} on {' and '.join(r['addresses'])}" + (f" ({r['command']})" if r["command"]
                                                              else "") for r in ports)


def dial(port):
    """::1, then 127.0.0.1, by address: /etc/hosts may map localhost to 127.0.0.1 only."""
    for host in ("::1", "127.0.0.1"):
        try:
            return socket.create_connection((host, port), timeout=10)
        except OSError:
            pass
    raise OSError(f"nothing listens on port {port}")


def connect(port):
    try:
        s = dial(port)
    except OSError:
        msg = (f"Nothing listens on port {port} in the sandbox (dialed localhost: ::1 and "
               f"127.0.0.1); {ports_words(listening())}.\n").encode()
        os.write(1, b"HTTP/1.1 502 Bad Gateway\r\nContent-Type: text/plain\r\nContent-Length: " +
                 str(len(msg)).encode() + b"\r\n\r\n" + msg)
        return
    s.settimeout(None)

    def up():
        try:
            while True:
                d = os.read(0, 65536)
                if not d:
                    break
                s.sendall(d)
            s.shutdown(socket.SHUT_WR)
        except OSError:
            pass
    threading.Thread(target=up, daemon=True).start()
    try:
        while True:
            d = s.recv(65536)
            if not d:
                break
            os.write(1, d)
    except OSError:
        pass


if __name__ == "__main__":
    if sys.argv[1:2] == ["serve"]:
        serve()
    elif sys.argv[1:2] == ["connect"]:
        connect(int(sys.argv[2]))
    elif sys.argv[1:2] == ["ports"]:
        print(json.dumps(listening()))
    else:
        sys.exit("relay.py serve | connect PORT | ports")
