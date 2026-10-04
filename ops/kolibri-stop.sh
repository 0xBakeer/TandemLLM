#!/bin/bash
# Stop Kolibri-1 on :8001: SIGTERM drains (the request in flight finishes), SIGKILL after the grace.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${KOLIBRI_PORT:-8001}"
GRACE="${1:-60}"
. "$HERE/engines.sh"
PIDS="$(engine_pids "$PORT")"
[ -z "$PIDS" ] && { echo "[kolibri-stop] nothing on :$PORT"; exit 0; }
echo "[kolibri-stop] SIGTERM $(echo $PIDS | tr '\n' ' '), up to ${GRACE}s to drain"
for P in $PIDS; do kill -TERM "$P" 2>/dev/null; done
for _ in $(seq 1 "$GRACE"); do
    LEFT=""; for P in $PIDS; do kill -0 "$P" 2>/dev/null && LEFT="$LEFT $P"; done
    [ -z "$LEFT" ] && { echo "[kolibri-stop] stopped"; exit 0; }
    sleep 1
done
echo "[kolibri-stop] still alive after ${GRACE}s, SIGKILL:$LEFT"
for P in $LEFT; do kill -KILL "$P" 2>/dev/null; done
