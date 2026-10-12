#!/bin/bash
# Switch the :8000 service to another weight profile (ops/profiles/<name>.env), one at a time.
#
#   ops/use-profile.sh <name>              check files (fetch them if missing), stop, start, smoke, roll back on failure
#   ops/use-profile.sh <name> --fetch-only check / fetch / verify the profile's files, touch nothing else
#   ops/use-profile.sh <name> --dry-run    print what a switch would do
#   ops/use-profile.sh --list              the profiles and which one is active
#
# Everything but the weights is shared: serve.env, the conductor, ledger, caches, crons. The active profile
# is ops/active-profile, which serve.env reads, so start.sh, the watchdog and the @reboot line always start
# it. The switch holds the box lock (~/.qwen38-box.flock), so the watchdog leaves the restart alone.
# QSE_NO_ROLLBACK=1 leaves a failed switch to the caller (the setup guide's apply job).
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAME="${1:-}"; MODE="${2:-switch}"
# the profile's view of serve.env, without switching
pv() { QSE_PROFILE_OVERRIDE="$NAME" bash -c "set -a; . '$HERE/serve.env'; echo \"\${$1:-}\""; }
# whether the profile sets a key at all (an empty value included)
pvset() { QSE_PROFILE_OVERRIDE="$NAME" bash -c "set -a; . '$HERE/serve.env'; [ -n \"\${$1+x}\" ]"; }
# the active-profile file and the profiles are the ones serve.env reads, under its REPO: this folder's
# parent on a checkout, the source tree on an install.sh box (whose serve.env sits in DIR/run/ops)
REPO_DIR="$(pv REPO)"; REPO_DIR="${REPO_DIR:-$(dirname "$HERE")}"
ACTIVE="$REPO_DIR/ops/active-profile"
PROFILES="$REPO_DIR/ops/profiles"
SMOKE_URL="${SMOKE_URL:-https://ai-api.khaled.de/v1}"
SMOKE_MODEL="${SMOKE_MODEL:-TandemLLM}"   # the served raw name when the conductor is off (the setup guide)
MEM_NEED_GB="${MEM_NEED_GB:-}"          # MemAvailable before a start: the profile's QSE_MEM_NEED_GIB, else 80
                                        # (serve8011's guard; the 27B stack takes ~70)
DISK_MARGIN_GB="${DISK_MARGIN_GB:-60}"  # free disk left after a download (disk tier + spill grow into it)
BOX_LOCK="${BOX_LOCK:-$HOME/.qwen38-box.flock}"
say() { echo "[use-profile $(date +%H:%M:%S)] $*"; }

if [ "$NAME" = "--list" ]; then
    CUR="$(cat "$ACTIVE" 2>/dev/null || echo '(none: serve.env values)')"
    for f in "$PROFILES"/[!_]*.env; do
        n="$(basename "$f" .env)"
        d="$(QSE_PROFILE_OVERRIDE=$n bash -c "set -a; . '$HERE/serve.env'; echo \"\$QSE_WEIGHTS_DIR\"")"
        printf '%-16s %-3s %s %s\n' "$n" "$([ "$n" = "$CUR" ] && echo '*')" "$d" "$([ -e "$d" ] && echo '(on disk)' || echo '(not on disk: fetched on switch)')"
    done
    exit 0
fi
[ -n "$NAME" ] && [ -f "$PROFILES/$NAME.env" ] || { echo "usage: $0 <profile>|--list [--fetch-only|--dry-run]; profiles: $(ls "$PROFILES" | grep -v '^_' | sed 's/\.env$//' | tr '\n' ' ')"; exit 2; }

WDIR="$(pv QSE_WEIGHTS_DIR)"; HF="$(pv QSE_WEIGHTS_HF)"; GB="$(pv QSE_WEIGHTS_GB)"; FILES="$(pv QSE_WEIGHTS_FILES)"
REV="$(pv QSE_WEIGHTS_REVISION)"; ALSO="$(pv QSE_WEIGHTS_ALSO)"
[ -n "$MEM_NEED_GB" ] || MEM_NEED_GB="$(pv QSE_MEM_NEED_GIB)"; MEM_NEED_GB="${MEM_NEED_GB:-80}"
PY="$(pv PY)"; PORT="$(pv PORT)"
DRAFT="$(pv QWEN38_DRAFT_EXL3)"; DHEAD="$(pv QWEN38_DRAFT_HEAD_EXL3)"; C8="$(pv CKPT8)"; C16="$(pv CKPT16)"

# 1. files: the target's weights (fetched when missing), the drafters and the draft head (never fetched:
#    they live on this box only), the base snapshot every 27B profile reads; a profile that sets
#    QSE_ASSETS names its own (Flash-Next: none, its pack holds its draft layer)
weights_ok() {
    [ -d "$WDIR" ] || return 1
    if [ -n "$FILES" ]; then for f in $FILES; do [ -s "$WDIR/$f" ] || return 1; done; return 0; fi
    [ -s "$WDIR/config.json" ] && [ -s "$WDIR/model.safetensors.index.json" ] || return 1
    for f in $ALSO; do [ -s "$WDIR/$f" ] || return 1; done
    "$PY" -c "import json,os,sys; d=sys.argv[1]; m=json.load(open(os.path.join(d,'model.safetensors.index.json')))['weight_map']; sys.exit(0 if all(os.path.isfile(os.path.join(d,f)) for f in set(m.values())) else 1)" "$WDIR"
}
MISSING=""
if pvset QSE_ASSETS; then
    for p in $(pv QSE_ASSETS); do [ -e "$p" ] || MISSING="$MISSING $p"; done
else
    for p in ${DRAFT:+$DRAFT/ft-b8-v2.safetensors $DRAFT/ft-b16.safetensors} ${DHEAD:+$DHEAD/model.safetensors} \
             ${DRAFT:-$C8 $C16}; do [ -e "$p" ] || MISSING="$MISSING $p"; done
    ls -d "$HOME"/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/* >/dev/null 2>&1 || MISSING="$MISSING (the Qwen3.8-27B-FP8 base snapshot)"
fi
[ -z "$MISSING" ] || { say "REFUSING: $NAME needs files that are not on this box and have no download:$MISSING"; exit 3; }
if ! weights_ok; then
    PARENT="$(dirname "$WDIR")"; while [ ! -d "$PARENT" ]; do PARENT="$(dirname "$PARENT")"; done
    FREE=$(df -BG --output=avail "$PARENT" | tail -1 | tr -dc 0-9)
    NEED=$(awk -v g="$GB" -v m="$DISK_MARGIN_GB" 'BEGIN{printf "%d", g + m + 0.999}')
    say "$WDIR is not complete: fetch $HF${REV:+@$REV} (~${GB} GB), free ${FREE} GB, need ${NEED} GB (margin ${DISK_MARGIN_GB})"
    [ "$MODE" = "--dry-run" ] && { say "dry run: would fetch"; } || {
        [ "$FREE" -ge "$NEED" ] || { say "REFUSING the download: ${FREE} GB free < ${NEED} GB"; exit 4; }
        "$PY" "$HERE/fetch_weights.py" ${REV:+--revision "$REV"} "$HF" "$WDIR" $FILES || { say "download/verify FAILED"; exit 5; }
        weights_ok || { say "the download is not a complete set"; exit 5; }
    }
else
    say "$NAME: weights $WDIR present"
fi
[ "$MODE" = "--fetch-only" ] && { say "fetch-only: done"; exit 0; }

PREV="$(cat "$ACTIVE" 2>/dev/null || echo nvfp4)"
if [ "$MODE" = "--dry-run" ]; then
    say "dry run: lock $BOX_LOCK; check MemAvailable; $HERE/stop.sh 120; wait for MemAvailable >= ${MEM_NEED_GB} GB;"
    say "dry run: echo $NAME > $ACTIVE; $HERE/start.sh; smoke $SMOKE_URL; on failure back to $PREV"
    exit 0
fi

exec 9<>"$BOX_LOCK"
flock -w 900 9 || { say "the box lock stayed held for 15 min; not switching"; exit 6; }
# start.sh refuses under a fresh pause file: find that out before the stop, not after it
if [ -f "$REPO_DIR/.watchdog.off" ]; then
    AGE=$(( $(date +%s) - $(stat -c %Y "$REPO_DIR/.watchdog.off") ))
    [ "$AGE" -lt "${WATCHDOG_PAUSE_MAX:-2400}" ] && { say "REFUSING: a hold owns the board ($REPO_DIR/.watchdog.off, ${AGE}s old)"; exit 6; }
fi

memavail() { awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo; }
start_profile() {   # $1 = profile; returns start.sh's status
    echo "$1" > "$ACTIVE"
    "$HERE/start.sh" < /dev/null 9>&-
}
smoke() {   # a /health that names the profile, then one conductor turn through the reverse proxy
    local h; h="$(curl -s -m 10 "http://127.0.0.1:$PORT/health")"
    echo "$h" | "$PY" -c "import json,sys; w=json.load(sys.stdin).get('weights') or {}; sys.exit(0 if w.get('name')=='$1' else 1)" \
        || { say "smoke: /health does not name $1"; return 1; }
    local r; r="$(curl -s -m 300 -H 'Content-Type: application/json' -H 'X-Tandem-Memory: off' "$SMOKE_URL/chat/completions" \
        -d '{"model":"'"$SMOKE_MODEL"'","messages":[{"role":"user","content":"Reply with the single word: ready"}],"max_tokens":512,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}')"
    echo "$r" | "$PY" -c "import json,sys; d=json.load(sys.stdin); c=d['choices'][0]['message'].get('content') or ''; print('[smoke] ' + c.strip()[:80]); sys.exit(0 if c.strip() else 1)" \
        || { say "smoke: the turn through $SMOKE_URL failed: ${r:0:300}"; return 1; }
}

T0=$(date +%s)
say "switching $PREV -> $NAME (MemAvailable $(memavail) GB)"
STATUS_REASON="switching the weights to $NAME" "$HERE/stop.sh" 120 9>&-
for _ in $(seq 1 60); do [ "$(memavail)" -ge "$MEM_NEED_GB" ] && break; sleep 2; done
T1=$(date +%s)
# QSE_NO_ROLLBACK=1: the caller rolls back itself (the setup guide's apply job puts its old
# run/setup.env back first; a rollback here would start the old profile under the new overlay)
NO_ROLLBACK="${QSE_NO_ROLLBACK:-0}"
if [ "$(memavail)" -lt "$MEM_NEED_GB" ]; then
    [ "$NO_ROLLBACK" = "1" ] && { say "MemAvailable $(memavail) GB < ${MEM_NEED_GB} GB after the stop; the caller rolls back"; exit 7; }
    say "MemAvailable $(memavail) GB < ${MEM_NEED_GB} GB after the stop; starting $PREV again"
    start_profile "$PREV"; exit 7
fi
if start_profile "$NAME" && smoke "$NAME"; then
    T2=$(date +%s)
    say "OK: $NAME live on :$PORT; stop $((T1-T0)) s, start+smoke $((T2-T1)) s, total $((T2-T0)) s; MemAvailable $(memavail) GB"
    exit 0
fi
[ "$NO_ROLLBACK" = "1" ] && { say "FAILED on $NAME; the caller rolls back"; exit 1; }
say "FAILED on $NAME; rolling back to $PREV"
STATUS_REASON="rolling back to $PREV" "$HERE/stop.sh" 120 9>&-
for _ in $(seq 1 60); do [ "$(memavail)" -ge "$MEM_NEED_GB" ] && break; sleep 2; done
if start_profile "$PREV" && smoke "$PREV"; then say "rolled back: $PREV live"; else say "ROLLBACK FAILED too: see $REPO_DIR/logs/engine.log"; fi
exit 1
