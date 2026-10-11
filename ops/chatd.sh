#!/bin/bash
# chatd, the Chat tab's service (python -m chatd, chatd/__init__.py), started and stopped by hand:
#
#   bash ops/chatd.sh start [--front 0.0.0.0:4180]   start it in the background (127.0.0.1:8016)
#   bash ops/chatd.sh stop                          SIGTERM: runs on end `interrupted` (Continue)
#   bash ops/chatd.sh status                        the pid and GET /health
#
# Its own process, never the engine's: an engine restart does not touch it, and it does not touch
# the engine (ops/engines.sh does not count it). Data in ~/.tandemllm/chat (QSE_CHAT_DATA), the log
# in logs/chatd.log (ids and counts, never message text), the pid in ~/.tandemllm/chat/chatd.pid.
# `--front HOST:PORT` also serves the dashboard and the Chat tab from this process, for an engine
# whose server has no /v1/chat-app forward yet (chatd/front.py). QSE_CHAT_FORWARD_KEY comes from
# secrets.env when ops/make-secrets.sh wrote one; else chatd makes ~/.tandemllm/chat/forward.key.
set -eu
HERE="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${QSE_CHAT_DATA:-$HOME/.tandemllm/chat}"
PIDF="$DATA/chatd.pid"
PY="${CHATD_PY:-python3}"
SECRETS="${QSE_STATE_DIR:-$HOME/.qwen38-spark-engine}/secrets.env"
cmd="${1:-status}"; shift || true
alive() { [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; }
case "$cmd" in
    start)
        if alive; then echo "[chatd] already running, pid $(cat "$PIDF")"; exit 0; fi
        mkdir -p "$DATA" "$HERE/logs"
        if [ -f "$SECRETS" ]; then
            k="$(grep -E '^QSE_CHAT_FORWARD_KEY=' "$SECRETS" | tail -1 | cut -d= -f2- || true)"
            [ -n "$k" ] && export QSE_CHAT_FORWARD_KEY="$k"
        fi
        FRONT=()
        if [ "${1:-}" = "--front" ]; then FRONT=(--front "$2" --dashboard "$HERE/dashboard/dist"); fi
        cd "$HERE"
        nohup env PYTHONPATH="$HERE" "$PY" -u -m chatd --data "$DATA" "${FRONT[@]}" \
            >> "$HERE/logs/chatd.log" 2>&1 < /dev/null &
        echo $! > "$PIDF"
        for _ in $(seq 1 50); do
            curl -sf -m 1 http://127.0.0.1:8016/health > /dev/null 2>&1 && break
            sleep 0.2
        done
        echo "[chatd] pid $(cat "$PIDF"): $(curl -s -m 2 http://127.0.0.1:8016/health)"
        ;;
    stop)
        if ! alive; then echo "[chatd] not running"; rm -f "$PIDF"; exit 0; fi
        pid="$(cat "$PIDF")"
        kill -TERM "$pid"
        for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
        kill -0 "$pid" 2>/dev/null && kill -KILL "$pid"
        rm -f "$PIDF"
        echo "[chatd] stopped"
        ;;
    status)
        if alive; then
            echo "[chatd] pid $(cat "$PIDF"): $(curl -s -m 2 http://127.0.0.1:8016/health)"
        else
            echo "[chatd] not running"; exit 1
        fi
        ;;
    *) echo "usage: $0 start [--front HOST:PORT] | stop | status"; exit 2 ;;
esac
