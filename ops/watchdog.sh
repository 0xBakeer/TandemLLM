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

CODE="$(curl -s -o /dev/null -m 10 -w '%{http_code}' "http://127.0.0.1:$PORT/health" || echo 000)"
if [ "$CODE" = "200" ]; then echo 0 > "$STATE"; exit 0; fi
if [ "$CODE" = "503" ]; then say "draining (503), leaving it alone"; echo 0 > "$STATE"; exit 0; fi
N=$(( $(cat "$STATE" 2>/dev/null || echo 0) + 1 ))
echo "$N" > "$STATE"
say "health check failed (code $CODE), $N/$NEED"
[ "$N" -lt "$NEED" ] && exit 0
say "restarting"
"$HERE/stop.sh" 30 >> "$WD" 2>&1
"$HERE/start.sh" >> "$WD" 2>&1 && say "restarted" || say "RESTART FAILED"
echo 0 > "$STATE"
