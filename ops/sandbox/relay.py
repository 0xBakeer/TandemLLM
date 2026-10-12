"""The sandbox's one network helper, baked into tandem-sandbox:1 at /opt/tandem/relay.py.

The container runs with no network but its loopback. Two jobs:

    relay.py serve          (the container's main process) listen on 127.0.0.1:3128 and pass every
                            connection to chatd's proxy for this chat, /run/tandem/proxy.sock
                            (HTTP_PROXY and HTTPS_PROXY point here; chatd/egress.py checks where
                            each one goes: the internet yes, the LAN and the host no)
    relay.py connect PORT   (an exec per request) connect to 127.0.0.1:PORT and pipe stdin to it
                            and its answer to stdout: chatd's way to a dev server in the sandbox
                            for the preview pane
"""

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


def connect(port):
    try:
        s = socket.create_connection(("127.0.0.1", port), timeout=10)
    except OSError:
        msg = f"Nothing listens on port {port} in the sandbox.\n".encode()
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
    else:
        sys.exit("relay.py serve | connect PORT")
