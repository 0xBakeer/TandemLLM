#!/bin/bash
# Graceful stop: SIGTERM makes the server drain (new requests get 503 + Retry-After, the
# generation in flight finishes), and only a server that will not go after the grace period is
# killed. A hard kill mid-verify truncates whoever is streaming.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a; . "$HERE/serve.env"; set +a
GRACE="${1:-60}"
PID="$(pgrep -f "server/app.py --host .* --port $PORT" | head -1)"
[ -z "$PID" ] && { echo "[stop] nothing on :$PORT"; exit 0; }
echo "[stop] SIGTERM $PID, up to ${GRACE}s to drain"
kill -TERM "$PID"
for _ in $(seq 1 "$GRACE"); do kill -0 "$PID" 2>/dev/null || { echo "[stop] stopped"; exit 0; }; sleep 1; done
echo "[stop] still alive after ${GRACE}s, SIGKILL"; kill -KILL "$PID" 2>/dev/null; sleep 2
echo "[stop] killed"
