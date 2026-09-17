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

if curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "[start] already healthy on :$PORT"; exit 0
fi
BUSY="$(ss -ltnp 2>/dev/null | grep ":$PORT " || true)"
if [ -n "$BUSY" ]; then
    echo "[start] REFUSING: something else holds :$PORT and it is not answering /health:"
    echo "$BUSY"; exit 1
fi

cd "$REPO" || exit 1
export PYTHONPATH TZ=Europe/Berlin QWEN38_FUSE_PROJ="$FUSE_PROJ"
LATCH_FLAG=""; [ "${LEN_LATCH:-0}" = "1" ] && LATCH_FLAG="--len-latch"
setsid nohup "$PY" -u server/app.py \
    --host "$HOST" --port "$PORT" --served-model "$SERVED_MODEL" \
    --max-len "$MAX_LEN" --default-max-tokens "$DEFAULT_MAX_TOKENS" \
    --reasoning-format "$REASONING_FORMAT" --reasoning-effort "$REASONING_EFFORT" \
    --drafter lenrouter --len-fixed "$LEN_FIXED" $LATCH_FLAG --dflash2-path greedy \
    --tree --budget "$BUDGET" --corpus "$CORPUS" \
    --dflash2-ckpt "$CKPT8" --dflash2-ckpt16 "$CKPT16" \
    --nvfp4 "$NV" --fp8-head "$HEAD" --cache-budget-gb "$CACHE_GB" \
    --request-timeout "$REQUEST_TIMEOUT" --max-queue "$MAX_QUEUE" \
    --queue-timeout "$QUEUE_TIMEOUT" --verbose \
    >"$LOG" 2>&1 < /dev/null &
echo $! > "$PIDFILE"
ln -sfn "$LOG" "$LOGS/engine.log"
echo "[start] pid $(cat "$PIDFILE"), log $LOG"
for i in $(seq 1 60); do
    if curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        echo "[start] healthy after ${i}0s"; exit 0
    fi
    sleep 10
done
echo "[start] FAILED to become healthy in 600s; tail of $LOG:"; tail -20 "$LOG"; exit 1
