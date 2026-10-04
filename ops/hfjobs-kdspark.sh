#!/usr/bin/env bash
#
# The Kolibri-1 block drafter (DSpark-style) on Hugging Face Jobs. Every job has a --timeout; check
# the spend after each job.
#
#   bash ops/hfjobs-kdspark.sh push-code              tools/ engine/ into the bucket (committed tree)
#   bash ops/hfjobs-kdspark.sh push-prompts FILE      the prompt list (tools/kd_prompts.py output)
#   F=h200 SPLIT=train START=0 LIMIT=1024 BUDGET=25 T=45m bash ops/hfjobs-kdspark.sh gen NAME
#                                                     Kolibri's own answers, vLLM + Aleph's plugin, FP8
#   F=rtx-pro-6000 T=40m EXTRA="--pilot" bash ops/hfjobs-kdspark.sh train TAG
#                                                     the drafter, online against the served NVFP4 set
#   bash ops/hfjobs-kdspark.sh drafttime | tapstats | tapcheck
#   bash ops/hfjobs-kdspark.sh spend | ps | logs ID | inspect ID | cancel ID
#
# Set HF_NS, BUCKET (a storage bucket you own), PREFIX (this run's folder in it) and SET_PREFIX (the
# folder that holds the NVFP4 set: layers/, outside.safetensors, manifest.json) first; nothing has a
# default. Code reaches a job through the bucket ($PREFIX/code/<sha>). The FP8 release and the
# tokenizer come from the Hub.

set -uo pipefail
cd "$(dirname "$0")/.."

HF=${HF:-hf}
NS=${HF_NS:?set HF_NS to the namespace that owns the bucket}
BUCKET=${BUCKET:?set BUCKET to a Hugging Face storage bucket}
PREFIX=${PREFIX:?set PREFIX to the run folder inside the bucket}
SETW=/work/${SET_PREFIX:?set SET_PREFIX to the bucket folder of the NVFP4 set}
FP8_REPO=Aleph-Alpha/Kolibri-1
IMG_TORCH=${IMG_TORCH:-pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel}
IMG_VLLM=${IMG_VLLM:-vllm/vllm-openai:v0.29.0}
SHA=$(git rev-parse --short=12 HEAD)
W=/work/$PREFIX

[ -n "${HF_TOKEN:-}" ] || { echo "export HF_TOKEN=\$(cat ~/.cache/huggingface/token)"; exit 2; }

PRE='set -eo pipefail; export HF_XET_HIGH_PERFORMANCE=1 PYTHONPATH=/tmp/code GIT_SHA='"$SHA"' PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True;
echo "[job] start $(date -u +%H:%M:%S) code '"$SHA"'"; mkdir -p /tmp/code && cp -r '"$W"'/code/'"$SHA"'/. /tmp/code/ && cd /tmp/code;
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true;'

run() {   # run <name> <flavor> <timeout> <image> -- <command>
  local name=$1 flavor=$2 timeout=$3 image=$4; shift 4
  [ "$1" = "--" ] && shift
  "$HF" jobs run ${DRY:+--dry-run} --detach --name "$name" --flavor "$flavor" --timeout "$timeout" \
    --secrets HF_TOKEN -v "hf://buckets/$NS/$BUCKET:/work" "$image" bash -c "$*"
}

cmd=${1:-help}; shift || true
case "$cmd" in
push-code)
  git diff --quiet HEAD -- tools engine || { echo "uncommitted changes in tools/engine; commit first"; exit 1; }
  st=$(mktemp -d)
  git archive HEAD tools engine | tar -x -C "$st"
  "$HF" buckets sync "$st" "hf://buckets/$NS/$BUCKET/$PREFIX/code/$SHA"
  rm -rf "$st"
  echo "code $SHA -> hf://buckets/$NS/$BUCKET/$PREFIX/code/$SHA"
  ;;
push-prompts)
  "$HF" buckets cp "${1:?prompts.jsonl}" "hf://buckets/$NS/$BUCKET/$PREFIX/prompts.jsonl"
  ;;
gen)
  name=${1:?name}
  run "kd-gen-$name" "${F:-h200}" "${T:-45m}" "$IMG_VLLM" -- "$PRE"'
    pip install -q --no-deps aleph-alpha-inference==1.0.0;
    t0=$(date +%s); python3 -c "from huggingface_hub import snapshot_download as s; s('"'"''"$FP8_REPO"''"'"', local_dir='"'"'/tmp/fp8'"'"')" >/dev/null; echo "[job] fp8 download $(( $(date +%s) - t0 )) s";
    mkdir -p '"$W"'/gen; cp '"$W"'/prompts.jsonl /tmp/prompts.jsonl;
    python3 -u tools/kd_gen.py --model /tmp/fp8 --prompts /tmp/prompts.jsonl --out /tmp/gen \
        --split '"${SPLIT:-train}"' --start '"${START:-0}"' --limit '"${LIMIT:-0}"' \
        --budget-min '"${BUDGET:-0}"' --max-num-seqs '"${SEQS:-256}"' --mirror '"$W"'/gen '"${EXTRA:-}"' 2>&1 | grep -v "it/s\]$" ;
    cp /tmp/gen/* '"$W"'/gen/; ls -la '"$W"'/gen | tail -5'
  ;;
train)
  tag=${1:?tag}
  live=${LIVE-"--gen-live '$W/gen/gen-*.pt'"}   # LIVE= (empty) turns the rescan off
  run "kd-train-$tag" "${F:-rtx-pro-6000}" "${T:-40m}" "$IMG_TORCH" -- "$PRE"'
    pip install -q -U safetensors "huggingface_hub[hf_xet]" "tokenizers>=0.22" numpy 2>&1 | tail -2;
    t0=$(date +%s); mkdir -p /tmp/set; cp -r '"$SETW"'/layers '"$SETW"'/outside.safetensors '"$SETW"'/manifest.json /tmp/set/;
    echo "[job] set copied in $(( $(date +%s) - t0 )) s: $(du -sh /tmp/set | cut -f1)";
    python3 -c "from huggingface_hub import hf_hub_download as d; import shutil; shutil.copy(d('"'"''"$FP8_REPO"''"'"', '"'"'config.json'"'"'), '"'"'/tmp/set/config.json'"'"')";
    mkdir -p /tmp/gen && cp '"$W"'/gen/gen-*.pt /tmp/gen/ && echo "[job] gen shards: $(ls /tmp/gen | wc -l)";
    rm -f /tmp/gen/*.part;
    mkdir -p '"$W"'/train/'"$tag"';
    if [ -n "'"${TAPCHECK:-}"'" ]; then python3 -c "from huggingface_hub import hf_hub_download as d; import shutil; shutil.copy(d('"'"''"$FP8_REPO"''"'"', '"'"'tokenizer.json'"'"'), '"'"'/tmp/set/tokenizer.json'"'"')"; python3 -u tools/kd_tapcheck.py --set /tmp/set --out /tmp/tapcheck.json || true; cp /tmp/tapcheck.json '"$W"'/train/'"$tag"'/ || true; fi;
    python3 -u tools/kd_train.py --set /tmp/set --gen "/tmp/gen/'"${GEN:-gen-*.pt}"'" --held "/tmp/gen/gen-*.pt" '"$live"' \
        --out /tmp/out --ckpt-dir '"$W"'/train/'"$tag"' --resume '"${RESUME:-$W/train/$tag/resume.pt}"' '"${EXTRA:-}"' 2>&1 | tee /tmp/train.log;
    cp -r /tmp/out/. '"$W"'/train/'"$tag"'/; cp /tmp/train.log '"$W"'/train/'"$tag"'/train-$(date -u +%H%M).log'
  ;;
drafttime)
  # one draft call in a CUDA graph for several drafter shapes and two runtimes (tools/kd_draft_time.py)
  run "kd-drafttime" "${F:-rtx-pro-6000}" "${T:-20m}" "$IMG_TORCH" -- "$PRE"'
    pip install -q -U safetensors 2>&1 | tail -1;
    mkdir -p '"$W"'/drafttime;
    python3 -u tools/kd_draft_time.py --set '"$SETW"' --json /tmp/dt.json '"${EXTRA:-}"';
    cp /tmp/dt.json '"$W"'/drafttime/'"${F:-rtx-pro-6000}"'.json'
  ;;
tapstats)
  run "kd-tapstats" "${F:-rtx-pro-6000}" "${T:-25m}" "$IMG_TORCH" -- "$PRE"'
    pip install -q -U safetensors "huggingface_hub[hf_xet]" 2>&1 | tail -1;
    mkdir -p /tmp/set; cp -r '"$SETW"'/layers '"$SETW"'/outside.safetensors '"$SETW"'/manifest.json /tmp/set/;
    python3 -c "from huggingface_hub import hf_hub_download as d; import shutil; shutil.copy(d('"'"''"$FP8_REPO"''"'"', '"'"'config.json'"'"'), '"'"'/tmp/set/config.json'"'"')";
    mkdir -p '"$W"'/diag; python3 -u tools/kd_tapstats.py --set /tmp/set --gen "'"$W"'/gen/gen-train-001024.pt" --out /tmp/taps.json;
    cp /tmp/taps.json '"$W"'/diag/tapstats.json'
  ;;
tapcheck)
  # decode-step taps against the trainer's prefill taps, and a drafter's walk with each (DRAFTER=bucket dir)
  run "kd-tapcheck" "${F:-rtx-pro-6000}" "${T:-30m}" "$IMG_TORCH" -- "$PRE"'
    pip install -q -U safetensors "huggingface_hub[hf_xet]" "tokenizers>=0.22" 2>&1 | tail -1;
    mkdir -p /tmp/set; cp -r '"$SETW"'/layers '"$SETW"'/outside.safetensors '"$SETW"'/manifest.json /tmp/set/;
    python3 -c "from huggingface_hub import hf_hub_download as d; import shutil; [shutil.copy(d('"'"''"$FP8_REPO"''"'"', f), '"'"'/tmp/set/'"'"' + f) for f in ('"'"'config.json'"'"', '"'"'tokenizer.json'"'"')]";
    mkdir -p /tmp/gen && cp '"$W"'/gen/gen-all-*.pt /tmp/gen/; cp -r '"${DRAFTER:?DRAFTER}"' /tmp/drafter;
    python3 -u tools/kd_tapcheck.py --set /tmp/set --drafter /tmp/drafter --held "/tmp/gen/gen-all-*.pt" --seqs '"${SEQS:-24}"' --out /tmp/tc.json || true;
    mkdir -p '"$W"'/diag; cp /tmp/tc.json '"$W"'/diag/tapcheck-$(date -u +%H%M).json'
  ;;
spend)
  curl -s -H "Authorization: Bearer $HF_TOKEN" https://huggingface.co/api/settings/billing/usage/jobs \
    | python3 -c 'import json,sys; u=json.load(sys.stdin)["usage"]; print("jobs this period: $%.2f, %s min" % (u["usedMicroUsd"] / 1e6, u["totalMinutes"])); [print(" ", j) for j in u.get("jobDetails", [])[-8:]]'
  ;;
ps) "$HF" jobs ps -a | head -20 ;;
logs) "$HF" jobs logs "${1:?job id}" ;;
inspect) "$HF" jobs inspect "${1:?job id}" ;;
cancel) "$HF" jobs cancel "${1:?job id}" ;;
*) sed -n 3,19p "$0" ;;
esac
