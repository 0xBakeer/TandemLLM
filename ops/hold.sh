#!/bin/bash
# Run a box-alone command: the service stopped, the watchdog paused and kept fresh, then restored.
#
# The 2026-09-18 wedges came from a second engine beside :8000 (twice) and from the watchdog
# resuming mid-hold (its pause ages out at 2400 s) and starting :8000 under a measurement. This
# script makes the correct sequence the easy one:
#
#   ops/hold.sh 30 -- python tools/verify_spec.py ...
#
# It arms the pause, keeps refreshing it every minute for the life of the hold, stops the service,
# runs the command, restarts the service (which waits for /health), and removes the pause on every
# exit path including a killed command.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
MIN="${1:?usage: hold.sh <minutes> -- <command...>}"; shift
[ "${1:-}" = "--" ] && shift
[ $# -gt 0 ] || { echo "[hold] no command given"; exit 2; }
cd "$REPO" || exit 1

touch .watchdog.off
( while [ -f .watchdog.off ]; do touch .watchdog.off; sleep 60; done ) &
KEEPER=$!
restore() {
    rm -f .watchdog.off
    kill "$KEEPER" 2>/dev/null
    echo "[hold] restarting the service"
    bash ops/start.sh 2>&1 | tail -2
    rm -f .watchdog.off
}
trap 'restore' EXIT

echo "[hold] stopping the service (up to ${MIN}m budget for the command)"
bash ops/stop.sh 120
if pgrep -f "[s]erver/app.py" >/dev/null; then
    echo "[hold] REFUSING: an engine process is still alive after stop.sh"; exit 1
fi
echo "[hold] running: $*"
"$@"
RC=$?
echo "[hold] command finished rc=$RC"
exit $RC
