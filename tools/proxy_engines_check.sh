#!/bin/bash
# The L1 adapters (P2, P3) against real vLLM and SGLang servers, for the night pass on the engine's
# host. One engine at a time, a small model, beside the live engine, and only when memory allows.
#
#   bash tools/proxy_engines_check.sh [--out DIR] [--only vllm|sglang] [--engine-port 8000]
#
# For each engine:
#   1 memory   MemAvailable must stay at least 30 GiB after the live engine's own headroom (its
#              memory budget's ceiling less what it holds, from :8000's system body); without that
#              figure the margin is 46 GiB. Otherwise the engine is SKIPPED, never squeezed in.
#   2 lock     the box lock (~/.qwen38-box.flock, flock -w 1800) for the whole start-check-stop:
#              never two loads at once
#   3 start    the engine's container with the model from the local Hugging Face cache (offline),
#              a small KV share and eager mode; vLLM with --enable-prompt-tokens-details,
#              --reasoning-parser qwen3 and n-gram speculation (retried without speculation when
#              the start fails), SGLang with --enable-metrics, --enable-cache-report and
#              --reasoning-parser qwen3. A watcher in this script stops the container when
#              MemAvailable falls under 8 GiB during the load or the checks
#   4 check    tools/proxy_engines_live.py (discovery, the page, the relay, the mapped page against
#              the engine's own, the window figures, the choice constraint, X-Request-Id on vLLM)
#   5 stop     the container is removed (also on any exit of this script) and the lock released
#
# Images: VLLM_IMAGE and SGLANG_IMAGE, by default the release tags the tables were checked against
# (vllm/vllm-openai:v0.30.0 and lmsysorg/sglang:v0.5.21-cu130, both with an arm64 image on Docker
# Hub), pulled before the lock is taken when the box does not have them; set either to an image
# already on the box to skip the pull. MODEL defaults to Qwen/Qwen3.5-0.8B and must be in the local
# Hugging Face cache. The versions the servers report are in each report.json.
set -u
cd "$(dirname "$0")/.." || exit 2
PY=${PY:-python3}
OUT=""; ONLY=""; EPORT=8000
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;
    --engine-port) EPORT="$2"; shift 2 ;;
    *) echo "unknown argument $1"; exit 2 ;;
  esac
done
OUT=${OUT:-results/proxy-engines-$(date +%Y%m%d-%H%M)}
mkdir -p "$OUT"
MODEL=${MODEL:-Qwen/Qwen3.5-0.8B}
VLLM_IMAGE=${VLLM_IMAGE:-vllm/vllm-openai:v0.30.0}
SGLANG_IMAGE=${SGLANG_IMAGE:-lmsysorg/sglang:v0.5.21-cu130}
LOCK=${BOX_LOCK:-$HOME/.qwen38-box.flock}
HF=${HF_HOME:-$HOME/.cache/huggingface}
NAME=""
WATCH=""

cleanup() {
  [ -n "$WATCH" ] && kill "$WATCH" 2>/dev/null
  [ -n "$NAME" ] && docker rm -f "$NAME" >/dev/null 2>&1
  WATCH=""; NAME=""
}
trap cleanup EXIT INT TERM

mem_ok() {                                  # prints the decision; exit 0 when the load may run
  $PY - "$EPORT" <<'EOF'
import json, sys, urllib.request
GiB = 1 << 30
avail = None
for line in open("/proc/meminfo"):
    if line.startswith("MemAvailable:"):
        avail = int(line.split()[1]) * 1024
head = None
try:
    body = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/v1/dashboard/system",
                                             timeout=10).read())
    b = ((body.get("memory") or {}).get("limits") or {}).get("budget")
    if b:
        head = max(0, int(b["ceiling"]) - int(b["reserved"]))
except Exception as exc:                                       # noqa: BLE001
    print(f"engine budget unreadable ({type(exc).__name__}); using a 16 GiB margin")
need = 30 * GiB + (head if head is not None else 16 * GiB)
print(f"MemAvailable {avail / GiB:.1f} GiB, engine headroom "
      f"{'?' if head is None else f'{head / GiB:.1f} GiB'}, need {need / GiB:.1f} GiB")
sys.exit(0 if avail is not None and avail >= need else 1)
EOF
}

watch_mem() {                               # stop the container when memory runs low
  ( while sleep 2; do
      a=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
      if [ "${a:-0}" -lt 8 ]; then
        echo "MemAvailable ${a} GiB: stopping $1" >> "$OUT/memory-stop.log"
        docker rm -f "$1" >/dev/null 2>&1; exit 0
      fi
    done ) &
  WATCH=$!
}

wait_up() {                                 # wait_up <port> <seconds>
  local t=0
  until curl -sf "http://127.0.0.1:$1/v1/models" >/dev/null 2>&1; do
    sleep 5; t=$((t+5))
    [ $t -ge "$2" ] && return 1
    docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null | grep -q true || return 1
  done
}

start_vllm() {                              # start_vllm <spec 0|1>
  local sp=()
  [ "$1" = 1 ] && sp=(--speculative-config '{"method":"ngram","num_speculative_tokens":3,"prompt_lookup_max":4}')
  NAME=px-l1-vllm
  docker run -d --name "$NAME" --gpus all --ipc=host --network host \
    -v "$HF:/root/.cache/huggingface" -e HF_HUB_OFFLINE=1 \
    "$VLLM_IMAGE" "$MODEL" --host 127.0.0.1 --port 8101 --gpu-memory-utilization 0.08 \
    --max-model-len 4096 --enforce-eager --enable-prefix-caching \
    --enable-prompt-tokens-details --reasoning-parser qwen3 "${sp[@]}" \
    > "$OUT/vllm-container.id" 2> "$OUT/vllm-start.err"
}

run_vllm() {
  local spec=1
  start_vllm 1 && watch_mem "$NAME"
  if ! wait_up 8101 900; then
    docker logs "$NAME" > "$OUT/vllm-server-spec.log" 2>&1
    cleanup
    echo "vLLM did not start with n-gram speculation; again without it" | tee -a "$OUT/summary.txt"
    spec=0
    start_vllm 0 && watch_mem "$NAME"
    wait_up 8101 900 || { docker logs "$NAME" > "$OUT/vllm-server.log" 2>&1; cleanup; return 1; }
  fi
  local flag=(); [ $spec = 1 ] && flag=(--spec)
  $PY tools/proxy_engines_live.py --kind vllm --url http://127.0.0.1:8101/v1 --model "$MODEL" \
    --out "$OUT/vllm" "${flag[@]}" > "$OUT/vllm-check.log" 2>&1
  local rc=$?
  docker logs "$NAME" > "$OUT/vllm-server.log" 2>&1
  cleanup
  return $rc
}

run_sglang() {
  NAME=px-l1-sglang
  docker run -d --name "$NAME" --gpus all --ipc=host --network host \
    -v "$HF:/root/.cache/huggingface" -e HF_HUB_OFFLINE=1 \
    "$SGLANG_IMAGE" python3 -m sglang.launch_server --model-path "$MODEL" --host 127.0.0.1 \
    --port 8102 --mem-fraction-static 0.08 --context-length 4096 --disable-cuda-graph \
    --enable-metrics --enable-cache-report --reasoning-parser qwen3 \
    > "$OUT/sglang-container.id" 2> "$OUT/sglang-start.err" || return 1
  watch_mem "$NAME"
  wait_up 8102 900 || { docker logs "$NAME" > "$OUT/sglang-server.log" 2>&1; cleanup; return 1; }
  $PY tools/proxy_engines_live.py --kind sglang --url http://127.0.0.1:8102/v1 --model "$MODEL" \
    --out "$OUT/sglang" > "$OUT/sglang-check.log" 2>&1
  local rc=$?
  docker logs "$NAME" > "$OUT/sglang-server.log" 2>&1
  cleanup
  return $rc
}

FAILS=0
: > "$OUT/summary.txt"
for kind in vllm sglang; do
  [ -n "$ONLY" ] && [ "$ONLY" != "$kind" ] && continue
  if ! mem_ok > "$OUT/$kind-memory.txt" 2>&1; then
    echo "SKIP $kind: $(tail -1 "$OUT/$kind-memory.txt")" | tee -a "$OUT/summary.txt"
    continue
  fi
  img=$VLLM_IMAGE; [ $kind = sglang ] && img=$SGLANG_IMAGE
  if ! docker image inspect "$img" >/dev/null 2>&1; then
    docker pull "$img" > "$OUT/$kind-pull.log" 2>&1 || {
      echo "SKIP $kind: could not pull $img" | tee -a "$OUT/summary.txt"; continue; }
  fi
  t0=$(date +%s)
  exec 9>"$LOCK"
  if ! flock -w 1800 9; then
    echo "SKIP $kind: the box lock stayed held for 30 minutes" | tee -a "$OUT/summary.txt"
    exec 9>&-; continue
  fi
  # memory again under the lock: a hold that ended just now may have changed it
  if ! mem_ok >> "$OUT/$kind-memory.txt" 2>&1; then
    echo "SKIP $kind: $(tail -1 "$OUT/$kind-memory.txt")" | tee -a "$OUT/summary.txt"
    exec 9>&-; continue
  fi
  if "run_$kind"; then r=PASS; else r=FAIL; FAILS=$((FAILS+1)); fi
  exec 9>&-
  echo "$r $kind $(( $(date +%s) - t0 ))s" | tee -a "$OUT/summary.txt"
done
echo "failed: $FAILS (logs in $OUT)" | tee -a "$OUT/summary.txt"
exit $FAILS
