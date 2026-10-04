#!/bin/bash
# Start Kolibri-1 (server/app.py --kolibri). One engine at a time: refuses while any engine process
# is alive, on any port. With KOLIBRI_LOCK set, the lock file is held for the server's whole life (a
# `flock -o` parent that waits on the server), so a GPU job queued with `flock $KOLIBRI_LOCK` waits
# until the server stops. On a board whose GPU shares system memory, a second large GPU process
# beside the resident server can exhaust that memory, which is why the lock covers the whole life
# and not only the load.
#
#   KOLIBRI_SET=./Kolibri-1-NVFP4 bash ops/kolibri-start.sh
#   KOLIBRI_SET=... KOLIBRI_FP8=./Kolibri-1 bash ops/kolibri-start.sh   # attention from the FP8 release
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${KOLIBRI_REPO:-$(dirname "$HERE")}"
PY="${PY:-python3}"
PORT="${KOLIBRI_PORT:-8001}"
HOST="${KOLIBRI_HOST:-0.0.0.0}"
MAX_LEN="${KOLIBRI_MAX_LEN:-262144}"
SET="${KOLIBRI_SET:?set KOLIBRI_SET to the downloaded Kolibri-1 NVFP4 set}"
FP8="${KOLIBRI_FP8-}"
TOK="${KOLIBRI_TOKENIZER:-$SET}"
LOCK="${KOLIBRI_LOCK:-}"
LOGS="${KOLIBRI_LOGS:-$REPO/logs}"
mkdir -p "$LOGS"
LOG="$LOGS/kolibri-$(date +%Y%m%d-%H%M%S).log"
. "$HERE/engines.sh"

if curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "[kolibri-start] already healthy on :$PORT"; exit 0
fi
LIVE="$(engine_pids)"
if [ -n "$LIVE" ]; then
    echo "[kolibri-start] REFUSING: an engine is alive (pid $(echo $LIVE)); one engine at a time"
    exit 1
fi
cd "$REPO" || exit 1
SHA="$(git rev-parse --short HEAD 2>/dev/null || cat .git-sha 2>/dev/null || echo unknown)"
echo "[kolibri-start] $REPO ($SHA) set=$SET fp8=${FP8:-<set>} max_len=$MAX_LEN -> :$PORT, log $LOG"
export LOG PY HOST PORT MAX_LEN SET FP8 TOK
WRAP=()
if [ -n "$LOCK" ]; then
    # the lock: free now, or nothing starts (a GPU job holds it; it ends, or stop it first)
    if ! flock -n "$LOCK" true; then
        echo "[kolibri-start] $LOCK is held (a GPU job runs); nothing started"; exit 1
    fi
    WRAP=(flock -o -w 30 "$LOCK")
fi
setsid nohup ${WRAP[@]+"${WRAP[@]}"} "$PY" -u server/app.py --kolibri --host "$HOST" --port "$PORT" \
    --served-model Kolibri-1 --max-len "$MAX_LEN" --kolibri-set "$SET" \
    --kolibri-fp8 "$FP8" --kolibri-tokenizer "$TOK" --reasoning-format tags \
    --default-max-tokens 32768 --request-timeout 3600 > "$LOG" 2>&1 < /dev/null &
echo $! > "$LOGS/kolibri.wrap.pid"
RC=4
for i in $(seq 1 200); do
    if curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then RC=0; break; fi
    kill -0 "$(cat "$LOGS/kolibri.wrap.pid")" 2>/dev/null || { RC=3; break; }
    sleep 3
done
engine_pids "$PORT" | head -1 > "$LOGS/kolibri.pid"
case $RC in
    0) echo "[kolibri-start] healthy on :$PORT (pid $(cat "$LOGS/kolibri.pid"))";;
    3) echo "[kolibri-start] the server died while loading:"; tail -20 "$LOG";;
    *) echo "[kolibri-start] not healthy after 600 s (rc $RC); see $LOG";;
esac
exit $RC
