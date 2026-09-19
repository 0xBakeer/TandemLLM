#!/bin/bash
# One health check. Meant for cron every minute, or a loop; it restarts only what it is sure about.
#
# Three states and only one of them is a restart:
#   healthy   -> nothing
#   draining  -> nothing. A graceful stop is in progress and killing it is the bug, not the fix.
#   silent    -> restart, after CONSECUTIVE failures, so one slow health check during a long
#                verify does not bounce a working server.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a; . "$HERE/serve.env"; set +a
LOGS="$REPO/logs"; mkdir -p "$LOGS"
STATE="$LOGS/watchdog.fails"; WD="$LOGS/watchdog.log"
NEED="${WATCHDOG_FAILS:-3}"
say() { echo "$(date -Is) $*" >> "$WD"; }

# A fourth state, added 2026-09-18: PAUSED. A measurement takes the board lock and drains :8000 on
# purpose, and a supervisor that cannot tell a deliberate stop from a crash restarts the service
# under the measurement -- which happened, and put a second full engine on a bandwidth-bound board
# for three atlas rows (SPEED-LEDGER, phase 10, 03:51). The pause file is how a hold says "this
# silence is mine"; it carries who and why, and the hold removes it in its EXIT trap so a crashed
# hold cannot leave the service unsupervised for long.
PAUSE="$REPO/.watchdog.off"
if [ -f "$PAUSE" ]; then
    AGE=$(( $(date +%s) - $(stat -c %Y "$PAUSE") ))
    if [ "$AGE" -lt "${WATCHDOG_PAUSE_MAX:-2400}" ]; then exit 0; fi
    say "pause file is ${AGE}s old, older than the longest hold; ignoring it"
fi

CODE="$(curl -s -o /dev/null -m 10 -w '%{http_code}' "http://127.0.0.1:$PORT/health" || echo 000)"
if [ "$CODE" = "200" ]; then echo 0 > "$STATE"; rm -f "$LOGS/loading.since"; exit 0; fi
if [ "$CODE" = "503" ]; then say "draining (503), leaving it alone"; echo 0 > "$STATE"; exit 0; fi
# A process that is LOADING answers nothing and is not absent (the 2026-09-18 entry in the ledger,
# relearned at every cold boot: the load reads 29 GB from cold disk and outruns the three-strike
# window; the old behaviour killed the loader and started another while the first one's memory was
# still draining -- a second full load is how this box wedges). So: if a server process exists,
# give it time; after LOADING_MAX minutes of consecutive failure, restart for real.
if pgrep -f "[s]erver/app.py --host .* --port $PORT" >/dev/null; then
    [ -f "$LOGS/loading.since" ] || date +%s > "$LOGS/loading.since"
    AGE=$(( $(date +%s) - $(cat "$LOGS/loading.since") ))
    if [ "$AGE" -lt "${LOADING_MAX:-900}" ]; then
        say "loading (code $CODE, ${AGE}s), leaving it alone"
        echo 0 > "$STATE"
        exit 0
    fi
    say "still not answering after ${AGE}s, treating as hung"
fi
N=$(( $(cat "$STATE" 2>/dev/null || echo 0) + 1 ))
echo "$N" > "$STATE"
say "health check failed (code $CODE), $N/$NEED"
[ "$N" -lt "$NEED" ] && exit 0
say "restarting"
"$HERE/stop.sh" 30 >> "$WD" 2>&1
"$HERE/start.sh" >> "$WD" 2>&1 && say "restarted" || say "RESTART FAILED"
echo 0 > "$STATE"
