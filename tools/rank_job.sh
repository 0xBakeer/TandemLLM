#!/bin/bash
# RND-37: the drafter ranking fine-tune, run INSIDE one Hugging Face Jobs container (h200).
#
# Mounts (set by the launcher):
#   /work          the private bucket (code under $W/code, recorded taps under $W/data/train)
#   /models/b8     the served block-8 drafter (bf16), /models/b16 the block-16 one
#   /models/target the FP8 target snapshot: config, embed_tokens and lm_head only are read
# Secrets: HF_TOKEN (for the bucket sync).
#
# What it does: copy the taps to local disk, then per arm (b8, b16, in parallel) a --probe over the
# ranking weights at $STEPS steps, then the best weight (in-loop tree eval on a validation split of
# the training pool, never the gate's traces) again at $FINAL_STEPS. Every configuration's final and
# best weights go to $OUT, which a background loop mirrors to the bucket every few minutes, so a
# timeout loses at most the configuration that was running.
#
# Env: STEPS (4000) FINAL_STEPS (8000) LAMBDAS (0.5,1,2) LR (3e-5) SEL_LR (1e-4) SMOKE (0)
set -uo pipefail
STEPS=${STEPS:-4000}; FINAL_STEPS=${FINAL_STEPS:-8000}; LAMBDAS=${LAMBDAS:-0.5,1,2}
LR=${LR:-3e-5}; SEL_LR=${SEL_LR:-1e-4}; SMOKE=${SMOKE:-0}
BUCKET=${BUCKET:-hf://buckets/0xBakeer/qwen38-engine-work}
RUN=${RUN:-rank/run1}
W=/work/rank
OUT=/tmp/out; DATA=/tmp/data; CODE=/tmp/code
mkdir -p $OUT $DATA $CODE
log() { echo "[rank_job $(date -u +%H:%M:%S)] $*"; }

log "gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
pip install -q --no-input safetensors numpy 'huggingface_hub>=1.31' >/dev/null 2>&1 || \
  pip install -q safetensors numpy huggingface_hub
cp -r $W/code/. $CODE/
cd $CODE
export PYTHONPATH=$CODE PY=python3 QWEN38_MODEL=/models/target

log "taps: bucket -> local disk"
t0=$(date +%s)
if [ "$SMOKE" = 1 ]; then
  hf buckets sync "$BUCKET/rank/data/smoke" $DATA >/dev/null 2>&1 || cp -r $W/data/smoke/. $DATA/
else
  hf buckets sync "$BUCKET/rank/data/train" $DATA >/dev/null 2>&1 || cp -r $W/data/train/. $DATA/
fi
log "taps: $(du -sh $DATA | cut -f1), $(ls $DATA | wc -l) files in $(( $(date +%s) - t0 )) s"
VAL=$(cat $W/val.txt 2>/dev/null | tr -d '\n')
[ "$SMOKE" = 1 ] && VAL=$(python3 -c "import json;print(json.load(open('$DATA/manifest.json'))['sequences'][0]['name'])")

( while true; do sleep 300; hf buckets sync $OUT "$BUCKET/$RUN" >/dev/null 2>&1; done ) &
SYNC=$!

arm() {   # arm <b8|b16> <block> <ckpt>
  local A=$1 B=$2 CK=$3 specs="" l
  for l in ${LAMBDAS//,/ }; do specs="$specs,$A-l$l=rank@$l:$LR:$B"; done
  specs=${specs#,}
  log "$A probe: $specs, $STEPS steps"
  python3 -u tools/train_dflash2.py --data $DATA --ckpt $CK --model /models/target \
    --probe "$specs" --probe-out $OUT/$A --export-final --heldout "$VAL" --steps $STEPS \
    --selector-lr $SEL_LR --eval-tree auto --eval-every $(( STEPS / 4 )) --gen-weight 1.0 \
    > $OUT/$A-probe.log 2>&1
  local best
  best=$(grep -E "^  probe $A-l" $OUT/$A-probe.log | sed -E 's/^  probe ([^ ]+) -> .*ALL=([0-9.]+).*/\2 \1/' \
         | sort -rn | head -1 | awk '{print $2}')
  log "$A best: $best ($(grep -E "^  probe $A-l" $OUT/$A-probe.log | sed -E 's/.*probe ([^ ]+) -> .*ALL=([0-9.]+).*/\1=\2/' | tr '\n' ' '))"
  [ -z "$best" ] && return 1
  [ "$FINAL_STEPS" = 0 ] && return 0
  l=${best#$A-l}
  log "$A final: lambda $l, $FINAL_STEPS steps"
  python3 -u tools/train_dflash2.py --data $DATA --ckpt $CK --model /models/target \
    --probe "$A-l$l-s$FINAL_STEPS=rank@$l:$LR:$B" --probe-out $OUT/$A --export-final \
    --heldout "$VAL" --steps $FINAL_STEPS --selector-lr $SEL_LR --eval-tree auto \
    --eval-every $(( FINAL_STEPS / 4 )) --gen-weight 1.0 > $OUT/$A-final.log 2>&1
  log "$A final: $(grep -E '^  probe ' $OUT/$A-final.log | tail -1)"
}

held_eval() {   # held_eval <block> <base ckpt> <label> [ckpt,...]: the gate's 62 + 20 traces, tree eval, bf16
  local B=$1 CK=$2 L=$3 E=${4:-base}
  [ -d /tmp/held ] || { mkdir -p /tmp/held; hf buckets sync "$BUCKET/rank/data/held" /tmp/held >/dev/null 2>&1 \
    || cp -r $W/data/held/. /tmp/held/; }
  python3 -u tools/train_dflash2.py --data /tmp/held --ckpt $CK --model /models/target --block $B \
    --eval-only --eval-ckpts "$E" --eval-tree auto --eval-blocks 1000 > $OUT/held-$L.log 2>&1
  log "held $L: $(grep -vE '^\s*\[|^data|^drafter|^--train|training seq' $OUT/held-$L.log | tail -n +1 | tr '\n' ' ' | cut -c1-900)"
}

if [ "$SMOKE" = 1 ]; then
  held_eval 8 /models/b8 base-b8
  held_eval 16 /models/b16 base-b16
fi

arm b8 8 /models/b8 & P8=$!
arm b16 16 /models/b16 & P16=$!
wait $P8; R8=$?; wait $P16; R16=$?
if [ "$SMOKE" != 1 ]; then
  for A in b8 b16; do
    B=8; [ $A = b16 ] && B=16
    E=base; for d in $(ls -d $OUT/$A/*/ 2>/dev/null); do E="$E,${d%/}"; done
    held_eval $B /models/$A held-$A "$E"
  done
fi
kill $SYNC 2>/dev/null
log "arms done (b8 rc $R8, b16 rc $R16); final sync"
hf buckets sync $OUT "$BUCKET/$RUN" 2>&1 | tail -2
log done
