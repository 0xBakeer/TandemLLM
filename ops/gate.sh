#!/bin/bash
# The standing test protocol of the 2026-09-24 plan (§3), one command (OPS-19). Run it on the box,
# from the candidate's box directory, under the box lock -- as a SCRIPT FILE, so nothing on the
# lock holder's command line looks like an engine to stop.sh / hold.sh (OPS-18):
#
#   flock -o ~/.qwen38-box.flock bash ~/qwen38-spark-engine/ops/hold.sh 150 -- \
#       bash ~/qwen38-spark-engine-p1/ops/gate.sh spd29 --flags "QWEN38_VERIFY_GRAPH=1 QWEN38_GDN_AB=1"
#
# Steps, in order, each with a PASS/FAIL line; a failing step stops the script (exit 1):
#   1 suite      every tests/test_*.py on the CPU, in a clean environment (no serve.env, no QWEN38_*)
#   2 gpu        every tests/gpu/test_*.py, served environment
#   3 identity   tools/flagoff_identity.py --from-env in the BASE directory and here, new flags off:
#                the served engine must be bit-identical to the one it replaces; then, with the
#                candidate flags on, what they change (measured, not gated)
#   4 lossless   tools/verify_spec.py with the served flags + the candidate flags: GATE PASS
#   5 rows       tools/row3.py <label>-nostore, <label>-nostore-r2 (store off), <label>-clean,
#                on :8011 at 262,144, the served flags + the candidate flags as --env
#   6 compare    row3 --compare and tools/gatecheck.py against the base reports (and the phase
#                baseline): --mode adopt needs the mean resolved better in every pair, both modes
#                need nothing resolved worse (mean, p50, p90, max, TTFT, wall, tok/blk, ms/blk)
#   7 ledger     a dated stub with every command and its output tail, appended to
#                notes/SPEED-LEDGER.md (append-only) and kept in results/gate/<label>/ledger-stub.md
#
# Options:
#   --flags "K=V ..."        the candidate's environment on top of ops/serve.env
#   --base-dir DIR           the checkout the identity compares against (default ~/qwen38-spark-engine-p1base)
#   --base-nostore F         store-off base report (default results/row3/rc4k-nostore.json)
#   --base-clean F           clean-store base report (default results/row3/rc4k-clean.json)
#   --phase-nostore F        a second store-off baseline, the phase's own (optional)
#   --phase-clean F          a second clean baseline (optional)
#   --mode noworse|adopt|block|tokens
#                            the exit rule on the rows against the base reports (default adopt with
#                            --flags, noworse without); block / tokens are the per-item rule of
#                            2026-09-24: ms/blk / tok/blk resolved better, nothing resolved worse
#   --phase-mode MODE        the rule against the phase baseline (default noworse: a ticket stacked on
#                            the adopted set is often below the row's resolution on the mean alone,
#                            and its own gain is read off ms/blk, which the rows resolve)
#   --skip-suite --skip-gpu --skip-identity --skip-lossless --skip-row
#   --rows "nostore nostore-r2 clean"   which rows (default all three)
set -u
D="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="${1:?usage: gate.sh <label> [options]}"; shift
FLAGS=""; BASE_DIR="${GATE_BASE_DIR:-$HOME/qwen38-spark-engine-p1base}"
BASE_NS=results/row3/rc4k-nostore.json; BASE_CL=results/row3/rc4k-clean.json
PH_NS=""; PH_CL=""; MODE=""; PMODE=noworse; ROWS="nostore nostore-r2 clean"
SKIP_SUITE=0; SKIP_GPU=0; SKIP_ID=0; SKIP_LOSSLESS=0; SKIP_ROW=0
while [ $# -gt 0 ]; do
  case "$1" in
    --flags) FLAGS="$2"; shift 2 ;;
    --base-dir) BASE_DIR="$2"; shift 2 ;;
    --base-nostore) BASE_NS="$2"; shift 2 ;;
    --base-clean) BASE_CL="$2"; shift 2 ;;
    --phase-nostore) PH_NS="$2"; shift 2 ;;
    --phase-clean) PH_CL="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --phase-mode) PMODE="$2"; shift 2 ;;
    --rows) ROWS="$2"; shift 2 ;;
    --skip-suite) SKIP_SUITE=1; shift ;;
    --skip-gpu) SKIP_GPU=1; shift ;;
    --skip-identity) SKIP_ID=1; shift ;;
    --skip-lossless) SKIP_LOSSLESS=1; shift ;;
    --skip-row) SKIP_ROW=1; shift ;;
    *) echo "[gate] unknown option $1"; exit 2 ;;
  esac
done
[ -n "$MODE" ] || { [ -n "$FLAGS" ] && MODE=adopt || MODE=noworse; }
cd "$D" || exit 1

# The served configuration. serve.env's REPO is the serving directory; the tile table and the code
# are this directory's.
set -a; . ops/serve.env; set +a
PY="${GATE_PY:-$PY}"
export QWEN38_SKINNY_TILES="$D/ops/skinny-tiles.json" QWEN38_FUSE_PROJ="$FUSE_PROJ"
export QWEN38_NVFP4="$NV" QWEN38_FP8_HEAD="$HEAD" TZ=Europe/Berlin
export PYTHONPATH="$D:$HOME/pylibs"
SERVED_ENV=$(env | grep -E '^QWEN38_' | grep -vE '^QWEN38_(NVFP4|FP8_HEAD)=' | sort)
ROW_ENV=""
for kv in $SERVED_ENV $FLAGS; do ROW_ENV="$ROW_ENV --env $kv"; done

OUT="results/gate/$LABEL"; mkdir -p "$OUT" results/row3
STUB="$OUT/ledger-stub.md"
FAIL=0
trap 'echo "[gate] signalled: stopping"; exit 143' TERM INT HUP

say() { echo "[gate] $*"; }
stub() { printf '%s\n' "$*" >> "$STUB"; }
# run <name> <cmd...>: output to $OUT/<name>.log; the command and its tail into the stub
run() {
  local name="$1"; shift
  stub '```'; stub "\$ $*"
  "$@" > "$OUT/$name.log" 2>&1; local rc=$?
  tail -${TAIL:-12} "$OUT/$name.log" >> "$STUB"; stub '```'
  return $rc
}
step() { say "$1 $2 $(date +%H:%M:%S)"; stub ""; stub "**$1** $2 -- $(date +%H:%M:%S)"; }
verdict() {  # verdict <step> <rc>
  if [ "$2" = 0 ]; then say "$1 PASS"; stub "$1: PASS"; else say "$1 FAIL"; stub "$1: FAIL"; fi
}
abort() {
  say "ABORT at $1 $(date '+%Y-%m-%dT%H:%M:%S%z')"; stub ""; stub "ABORTED at $1."
  finish 1
}
finish() {
  stub ""
  { echo; cat "$STUB"; } >> notes/SPEED-LEDGER.md
  say "ledger stub appended to notes/SPEED-LEDGER.md ($STUB)"
  exit "$1"
}

if pgrep -f "[s]erver/app.py" >/dev/null; then
  say "REFUSING: an engine is running ($(pgrep -f '[s]erver/app.py' | tr '\n' ' ')); run me inside ops/hold.sh"
  exit 1
fi
if [ $SKIP_ROW = 0 ]; then
  for f in "$BASE_NS" "$BASE_CL" $PH_NS $PH_CL; do
    [ -f "$f" ] || { say "REFUSING: the base report $f is not there, so nothing could be compared"; exit 1; }
  done
fi
CODE=$("$PY" -c "import sys; sys.path.insert(0, '$D'); from tools.row3 import code_hash; print(code_hash('$D'))" 2>/dev/null)
: > "$STUB"
stub "## $(date '+%Y-%m-%d %H:%M') -- gate $LABEL (ops/gate.sh)"
stub ""
stub "Candidate \`$D\`, code \`${CODE:0:16}\`; flags: \`${FLAGS:-none}\`; mode $MODE; base reports"
stub "\`$BASE_NS\` / \`$BASE_CL\`${PH_NS:+, phase \`$PH_NS\`}${PH_CL:+ / \`$PH_CL\`}; identity base \`$BASE_DIR\`."
say "start $(date '+%Y-%m-%dT%H:%M:%S%z') label=$LABEL code=${CODE:0:16} flags='${FLAGS}' mode=$MODE"

# 1 -- the suite, on the CPU, nothing of serve.env in its environment
if [ $SKIP_SUITE = 0 ]; then
  step 1 "suite (CPU, clean environment)"
  ok=0; bad=0; failed=""
  for t in tests/test_*.py; do
    if env -i HOME="$HOME" PATH="$PATH" PYTHONPATH="$D:$HOME/pylibs" CUDA_VISIBLE_DEVICES="" \
         TZ=Europe/Berlin timeout 900 "$PY" -u "$t" > "$OUT/suite-$(basename "$t" .py).log" 2>&1
    then ok=$((ok+1)); else bad=$((bad+1)); failed="$failed $t"; fi
  done
  stub '```'; stub "\$ CUDA_VISIBLE_DEVICES=\"\" python tests/<file>, every tests/test_*.py (env -i)"
  stub "files=$((ok+bad)) ok=$ok fail=$bad${failed:+ --$failed}"; stub '```'
  say "suite files=$((ok+bad)) ok=$ok fail=$bad$failed"
  verdict "1 suite" $bad; [ $bad = 0 ] || abort "1 suite"
fi

# 2 -- the GPU batteries
if [ $SKIP_GPU = 0 ]; then
  step 2 "GPU batteries"
  rc=0
  for t in tests/gpu/test_*.py; do
    TAIL=3 run "gpu-$(basename "$t" .py)" env $FLAGS "$PY" -u "$t" || rc=1
  done
  verdict "2 gpu" $rc; [ $rc = 0 ] || abort "2 gpu"
fi

# 3 -- flag-off identity against the base checkout, then what the flags change
if [ $SKIP_ID = 0 ]; then
  step 3 "identity (served configuration, base \`$BASE_DIR\`)"
  [ -d "$BASE_DIR/engine" ] || abort "3 identity: no base checkout at $BASE_DIR"
  # the base checkout runs this candidate's identity tool (the scenario and --from-env), on its
  # own engine; its own copy of the tool is left alone
  cp tools/flagoff_identity.py "$BASE_DIR/tools/flagoff_identity_gate.py"
  B=/tmp/gate-$LABEL-base.pt; C=/tmp/gate-$LABEL-cand.pt; F=/tmp/gate-$LABEL-flags.pt
  # both dumps run with the BASE's flag set: a flag this directory's serve.env adopted since the base
  # is not the base's, and the question here is whether the code with it off is the base's code
  UNSET=""
  for v in $(comm -23 <(env | grep -o '^QWEN38_[A-Z0-9_]*' | sort -u) \
                      <(grep -o '^QWEN38_[A-Z0-9_]*' "$BASE_DIR/ops/serve.env" | sort -u)); do
    case "$v" in QWEN38_NVFP4|QWEN38_FP8_HEAD|QWEN38_FUSE_PROJ) ;; *) UNSET="$UNSET -u $v" ;; esac
  done
  [ -n "$UNSET" ] && stub "identity with the base's flags: env$UNSET"
  TAIL=3 run identity-base env $UNSET bash -c "cd '$BASE_DIR' && PYTHONPATH='$BASE_DIR:$HOME/pylibs' '$PY' -u tools/flagoff_identity_gate.py --from-env --dump $B" \
    || abort "3 identity (base dump)"
  TAIL=3 run identity-cand env $UNSET "$PY" -u tools/flagoff_identity.py --from-env --dump $C || abort "3 identity (candidate dump)"
  run identity-compare "$PY" -u tools/flagoff_identity.py --compare $B $C; rc=$?
  verdict "3 identity (new flags off == base)" $rc
  if [ $rc = 0 ] && [ -n "$FLAGS" ]; then
    TAIL=3 run identity-flags env $UNSET $FLAGS "$PY" -u tools/flagoff_identity.py --from-env --dump $F
    TAIL=16 run identity-flags-compare "$PY" -u tools/flagoff_identity.py --compare $C $F
    say "3 flags on vs off: $(tail -1 "$OUT/identity-flags-compare.log") (measured, not gated)"
  fi
  rm -f $B $C $F
  [ $rc = 0 ] || abort "3 identity"
fi

# 4 -- the lossless gate with the candidate on
if [ $SKIP_LOSSLESS = 0 ]; then
  step 4 "lossless gate (served + candidate flags)"
  TAIL=14 run lossless env $FLAGS "$PY" -u tools/verify_spec.py --new 96 --k 15 --chat --dflash2 1 \
      --dflash2-ckpt "$CKPT8" --lenrouter "$CKPT16" --drop-idle --tree-router --corpus "$CORPUS" \
      --extra-prompts
  grep -A2 "^GATE" "$OUT/lossless.log" | grep -q PASS; rc=$?
  verdict "4 lossless" $rc; [ $rc = 0 ] || abort "4 lossless"
fi

# 5 + 6 -- the rows and the rule
if [ $SKIP_ROW = 0 ]; then
  G=0
  for r in $ROWS; do
    case "$r" in
      clean) store=clean; base="$BASE_CL"; ph="$PH_CL" ;;
      *) store=off; base="$BASE_NS"; ph="$PH_NS" ;;
    esac
    lab="$LABEL-$r"
    step 5 "row3 $lab (store $store)"
    TAIL=9 run "row3-$lab" "$PY" -u tools/row3.py --label "$lab" --store "$store" \
        --clean-store "$HOME/qwen38-suffix-norow-0923" --runs 3 --port 8011 --max-len 262144 \
        --server-arg=--drop-idle $ROW_ENV || abort "5 row3 $lab"
    for b in "$base" $ph; do
      m=$MODE; [ "$b" = "$base" ] || m=$PMODE
      step 6 "compare $lab vs $(basename "$b" .json) ($m)"
      TAIL=14 run "cmp-$lab-$(basename "$b" .json)" "$PY" tools/gatecheck.py --mode "$m" \
          "$b" "results/row3/$lab.json" || G=1
      tail -1 "$OUT/cmp-$lab-$(basename "$b" .json).log"
    done
  done
  step 6 "tokens/block and ms/block beside tok/s"
  run factors "$PY" - $(for r in $ROWS; do echo "results/row3/$LABEL-$r.json"; done) "$BASE_NS" "$BASE_CL" $PH_NS $PH_CL <<'EOF'
import json, sys
print(f"{'report':<28}{'mean':>8}{'p50':>8}{'tok/blk':>9}{'ms/blk':>8}  code")
for p in sys.argv[1:]:
    d = json.load(open(p)); s = d["summary"]
    f = lambda k, fmt: format(s[k]["median"], fmt) if s.get(k) else "n/a"
    print(f"{d['label']:<28}{s['mean']['median']:>8.2f}{s['p50']['median']:>8.2f}"
          f"{f('tok_blk', '.2f'):>9}{f('ms_blk', '.1f'):>8}  {d.get('code_sha256', 'n/a')[:12]}")
EOF
  cat "$OUT/factors.log"
  verdict "6 rows ($MODE)" $G
  [ $G = 0 ] || { stub ""; stub "The rows did not pass the $MODE rule."; finish 1; }
fi
say "done $(date '+%Y-%m-%dT%H:%M:%S%z'): PASS"
stub ""; stub "GATE $LABEL: every step PASS."
finish 0
