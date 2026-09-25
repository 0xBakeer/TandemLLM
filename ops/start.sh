#!/bin/bash
# Start the engine on :8000. Idempotent: if a healthy one is already there, say so and stop.
#
# The port is the contract. your-host.example reaches this box's nginx, which proxies :8000, so
# whatever answers on :8000 is what Open WebUI calls the model -- and exactly one thing may.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a; . "$HERE/serve.env"; set +a
LOGS="$REPO/logs"; mkdir -p "$LOGS"
PIDFILE="$LOGS/engine.pid"
LOG="$LOGS/engine-$(date +%Y%m%d-%H%M%S).log"

# A hold owns the board: the pause file is how it says so, and this script must respect it too.
# The @reboot cron line calls this directly (not through the watchdog, which already respects the
# pause), and on 2026-09-18 21:38 it started :8000 under a running hold -- a second engine beside
# the hold's own, which contaminated a row3 (the numbers read 3.5 % slower TTFT). The hold's own
# restart passes HOLD_RESTART=1, because that start is the point of the pause.
if [ -f "$REPO/.watchdog.off" ] && [ "${HOLD_RESTART:-0}" != "1" ]; then
    # A hold cannot survive a reboot, but its pause file can, and then the @reboot line refuses to
    # start the service for up to WATCHDOG_PAUSE_MAX with nothing holding anything. Older than the
    # longest hold means nobody is holding: the watchdog already ignores such a file, and this
    # removes it. A real refusal exits non-zero, so cron and the unit can see it.
    AGE=$(( $(date +%s) - $(stat -c %Y "$REPO/.watchdog.off") ))
    if [ "$AGE" -ge "${WATCHDOG_PAUSE_MAX:-2400}" ]; then
        echo "[start] pause file is ${AGE}s old, older than the longest hold; removing it"
        rm -f "$REPO/.watchdog.off"
    else
        echo "[start] a hold owns the board (.watchdog.off is armed, ${AGE}s); refusing to start"
        exit 1
    fi
fi
if curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "[start] already healthy on :$PORT"; exit 0
fi
BUSY="$(ss -ltnp 2>/dev/null | grep ":$PORT " || true)"
if [ -n "$BUSY" ]; then
    echo "[start] REFUSING: something else holds :$PORT and it is not answering /health:"
    echo "$BUSY"; exit 1
fi

cd "$REPO" || exit 1
# The access tokens (SRV-31): QSE_ADMIN_TOKEN (dashboard, cache routes) and QSE_METRICS_TOKEN
# (Prometheus), made by ops/make-secrets.sh, mode 600, never in this repository. Without the file
# the dashboard API does not exist (404) and /metrics answers only the box itself.
SECRETS="${QSE_SECRETS:-$HOME/.qwen38-spark-engine/secrets.env}"
if [ -f "$SECRETS" ]; then set -a; . "$SECRETS"; set +a; fi
# The usage ledger (SRV-28) is this service's alone: the server does not read QSE_USAGE_LEDGER from
# its environment, so a benchmark server started with serve.env exported stays off.
export PYTHONPATH TZ=Europe/Berlin QWEN38_FUSE_PROJ="$FUSE_PROJ" \
       QWEN38_DF2_TREE_MODE="$QWEN38_DF2_TREE_MODE" \
       QWEN38_TREE_ALIAS_STATE="$QWEN38_TREE_ALIAS_STATE"
LATCH_FLAG=""; [ "${LEN_LATCH:-0}" = "1" ] && LATCH_FLAG="--len-latch"
DROP_FLAG=""; [ "${DROP_IDLE:-0}" = "1" ] && DROP_FLAG="--drop-idle"
# The anti-repetition flags (ENG-17) are passed only when serve.env sets them; empty means the
# server's own defaults, which are the identities (off) -- a run without them is byte-identical.
PEN_FLAGS=""
[ -n "${REP_PENALTY:-}" ] && PEN_FLAGS="$PEN_FLAGS --rep-penalty $REP_PENALTY"
[ -n "${PRESENCE_PENALTY:-}" ] && PEN_FLAGS="$PEN_FLAGS --presence-penalty $PRESENCE_PENALTY"
[ -n "${FREQUENCY_PENALTY:-}" ] && PEN_FLAGS="$PEN_FLAGS --frequency-penalty $FREQUENCY_PENALTY"
[ -n "${PATTERN_STOP:-}" ] && PEN_FLAGS="$PEN_FLAGS --pattern-stop $PATTERN_STOP"
SAMPLE_FLAGS=""
[ -n "${TEMPERATURE:-}" ] && SAMPLE_FLAGS="$SAMPLE_FLAGS --temperature $TEMPERATURE"
[ -n "${TOP_P:-}" ] && SAMPLE_FLAGS="$SAMPLE_FLAGS --top-p $TOP_P"
[ -n "${TOP_K:-}" ] && SAMPLE_FLAGS="$SAMPLE_FLAGS --top-k $TOP_K"
# ENG-109: a sampled request's verify. det (or 1): the greedy request's tree, walked by drawing the
# target's token at each node; mixed: the sampled chain as the tree's spine, accepted against its q;
# unset: the q-aware chain (ENG-102).
case "${SAMPLED_TREE:-}" in
    1|det) SAMPLE_FLAGS="$SAMPLE_FLAGS --sampled-tree=det" ;;
    mixed) SAMPLE_FLAGS="$SAMPLE_FLAGS --sampled-tree=mixed" ;;
esac
# The engine inherits stdin/stdout/stderr and nothing else (OPS-15). Holds run as
# `flock ~/.qwen38-box.flock bash ops/hold.sh ...`, and flock hands its lock descriptor to the
# command unless told `-o`: it came down through hold.sh and this script into the restarted engine,
# which then held the box lock for its whole life, and the next hold waited on a server that does
# not exit. 255 is bash's own handle on this script, close-on-exec already.
for fd in /proc/$$/fd/*; do
    fd=${fd##*/}
    case "$fd" in 0|1|2|255|*[!0-9]*) ;; *) eval "exec $fd>&-" 2>/dev/null ;; esac
done
setsid nohup "$PY" -u server/app.py \
    --host "$HOST" --port "$PORT" --served-model "$SERVED_MODEL" \
    --max-len "$MAX_LEN" --default-max-tokens "$DEFAULT_MAX_TOKENS" \
    --reasoning-format "$REASONING_FORMAT" --reasoning-effort "$REASONING_EFFORT" \
    --drafter lenrouter --len-fixed "$LEN_FIXED" $LATCH_FLAG $DROP_FLAG --dflash2-path greedy \
    $PEN_FLAGS $SAMPLE_FLAGS \
    --tree --budget "$BUDGET" --corpus "$CORPUS" \
    --dflash2-ckpt "$CKPT8" --dflash2-ckpt16 "$CKPT16" \
    --nvfp4 "$NV" --fp8-head "$HEAD" --cache-budget-gb "$CACHE_GB" \
    --request-timeout "$REQUEST_TIMEOUT" --max-queue "$MAX_QUEUE" \
    --queue-timeout "$QUEUE_TIMEOUT" --verbose \
    --usage-ledger "${QSE_USAGE_LEDGER:-off}" \
    >"$LOG" 2>&1 < /dev/null &
echo $! > "$PIDFILE"
ln -sfn "$LOG" "$LOGS/engine.log"
echo "[start] pid $(cat "$PIDFILE"), log $LOG"
for i in $(seq 1 60); do
    if curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        echo "[start] healthy after ${i}0s"
        # SPD-18 (2026-09-25): the weights are on the device now and nothing reads their files
        # again, but their page cache is ~52 GB of the board, counted as available and not given
        # back fast enough: an 8,192-row prefill took MemFree to 1 GB and the driver logged
        # NV_ERR_NO_MEMORY; with the cache dropped the same request left 53 GB free.
        if [ "${DROP_PAGE_CACHE:-0}" = "1" ]; then "$PY" tools/drop_page_cache.py 2>&1 | tail -1; fi
        exit 0
    fi
    sleep 10
done
echo "[start] FAILED to become healthy in 600s; tail of $LOG:"; tail -20 "$LOG"; exit 1
