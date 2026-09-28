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
. "$HERE/engines.sh"
# How long the restore waits for an engine the command left behind; a knob only for the tests.
ENGINE_WAIT="${HOLD_ENGINE_WAIT:-60}"

# The keeper is what makes a hold of ANY length safe: the watchdog and start.sh ignore a
# pause file older than WATCHDOG_PAUSE_MAX (2400 s), which is the safety for a pause nobody is
# holding any more, and this loop keeps the file younger than HOLD_REFRESH for as long as the hold
# lives. HOLD_REFRESH exists so tests/test_ops_scripts.py can prove that on a compressed clock.
# It keeps nothing above stdio: `kill $KEEPER` ends the loop but not the sleep inside it, and under
# a plain `flock LOCKFILE hold.sh ...` that sleep held the box lock for up to a minute after the
# hold had returned.
touch .watchdog.off
( for fd in $(seq 3 254); do eval "exec $fd>&-"; done 2>/dev/null
  while [ -f .watchdog.off ]; do touch .watchdog.off; sleep "${HOLD_REFRESH:-60}"; done ) &
KEEPER=$!
restore() {
    # Order matters: the pause stays armed until the service is HEALTHY. Removing it first opens
    # a race with the cron watchdog (one check a minute), which then starts the engine and this
    # script's own start.sh reports "already healthy" -- observed 2026-09-18 22:17.
    kill "$KEEPER" 2>/dev/null
    touch .watchdog.off
    # One engine at a time is the rule this whole script exists for. Whatever the command
    # started must be gone before the service comes back; if something is still alive after a
    # minute, the box is better with no service than with two engines -- say so and do not start.
    # An engine on ANY port counts (row3's :8011 is one); a process that only mentions the path in
    # its command line does not (a lock holder's did, and the service stayed down).
    for _ in $(seq 1 "$ENGINE_WAIT"); do [ -z "$(engine_pids)" ] && break; sleep 1; done
    LEFT="$(engine_pids)"
    if [ -n "$LEFT" ]; then
        echo "[hold] REFUSING to restart the service: an engine is still alive:" \
             "$(echo $LEFT)"
        rm -f .watchdog.off
        return
    fi
    echo "[hold] restarting the service"
    # Nothing above stdio goes to the service. A hold run as `flock LOCKFILE hold.sh ...` has the
    # box lock on an open descriptor, and the engine used to inherit it: on 2026-09-23 at 09:24 the
    # restarted :8000 engine held the lock and every later `flock` waited on the operator's service.
    ( for fd in $(seq 3 254); do eval "exec $fd>&-"; done 2>/dev/null
      HOLD_RESTART=1 bash ops/start.sh ) 2>&1 | tail -2
    rm -f .watchdog.off
}
trap 'restore' EXIT

echo "[hold] stopping the service (up to ${MIN}m budget for the command)"
bash ops/stop.sh 120
if [ -n "$(engine_pids)" ]; then
    echo "[hold] REFUSING: an engine process is still alive after stop.sh: $(echo $(engine_pids))"
    exit 1
fi
echo "[hold] running: $*"
# The command runs in a process group of its own, so a signal to the HOLD -- an ssh session that
# dropped, a tool's timeout, a `kill` -- reaches all of it. It used to run in the foreground: the
# hold died, its EXIT trap restarted :8000, and the command (a row3 with its own engine) kept
# running beside it, which is the two-engine state that wedges this box.
setsid "$@" &
CMD=$!
stop_command() {
    trap - TERM HUP INT
    echo "[hold] signalled: stopping the command (process group $CMD) before the restore"
    kill -TERM -- -"$CMD" 2>/dev/null
    for _ in $(seq 1 120); do kill -0 -- -"$CMD" 2>/dev/null || break; sleep 1; done
    kill -KILL -- -"$CMD" 2>/dev/null
    exit 143
}
trap stop_command TERM HUP INT
wait "$CMD"
RC=$?
echo "[hold] command finished rc=$RC"
exit $RC
