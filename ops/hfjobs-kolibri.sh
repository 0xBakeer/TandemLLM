#!/usr/bin/env bash
#
# The Kolibri-1 NVFP4 build on Hugging Face Jobs, one job a step, each started only after the one
# before it passed its gate:
#
#   bash ops/hfjobs-kolibri.sh push-code     the committed tools/, engine/, bench/ into the bucket
#   bash ops/hfjobs-kolibri.sh probe         cpu-upgrade: mount vs Xet read rate, the corpus
#   bash ops/hfjobs-kolibri.sh dry           rtx-pro-6000: layers 0-1, 50k calibration tokens
#   bash ops/hfjobs-kolibri.sh crosscheck    rtx-pro-6000: vLLM + Aleph Alpha's plugin vs ours
#   bash ops/hfjobs-kolibri.sh build         rtx-pro-6000: all 50 layers, the gate, greedy
#   bash ops/hfjobs-kolibri.sh mixgate | attn | build-fp8t | final
#   bash ops/hfjobs-kolibri.sh spend | ps | logs <id> | inspect <id> | cancel <id>
#
# Set HF_NS (your user or org) and BUCKET (a storage bucket you own) first; PREFIX defaults to
# kolibri-nvfp4. Code reaches a job through the bucket ($PREFIX/code/<sha>, copied to /tmp/code
# first: the bucket mount reads small files slowly). Outputs go to $PREFIX/ in the same bucket.
# Every job has a --timeout. The checkpoints are public and are read from the Hub (mount for the
# probe, Xet downloads to local disk for the GPU jobs).

set -uo pipefail
cd "$(dirname "$0")/.."

HF=${HF:-hf}
NS=${HF_NS:?set HF_NS to the namespace that owns the bucket}
BUCKET=${BUCKET:?set BUCKET to a Hugging Face storage bucket}
PREFIX=${PREFIX:-kolibri-nvfp4}
BF16_REPO=Aleph-Alpha/Kolibri-1-BF16
FP8_REPO=Aleph-Alpha/Kolibri-1
IMG_TORCH=${IMG_TORCH:-pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel}
IMG_VLLM=${IMG_VLLM:-vllm/vllm-openai:v0.29.0}
IMG_CPU=${IMG_CPU:-python:3.12}
SHA=$(git rev-parse --short=12 HEAD)
W=/work/$PREFIX

[ -n "${HF_TOKEN:-}" ] || { echo "export HF_TOKEN=\$(cat ~/.cache/huggingface/token)"; exit 2; }

PRE='set -eo pipefail; export HF_XET_HIGH_PERFORMANCE=1 PYTHONPATH=/tmp/code GIT_SHA='"$SHA"';
echo "[job] start $(date -u +%H:%M:%S) code '"$SHA"'"; mkdir -p /tmp/code && cp -r '"$W"'/code/'"$SHA"'/. /tmp/code/ && cd /tmp/code;
echo "[job] code copied: $(ls /tmp/code)";'

run() {   # run <name> <flavor> <timeout> <image> <extra -v ...> -- <command>
  local name=$1 flavor=$2 timeout=$3 image=$4; shift 4
  local extra=()
  while [ "${1:-}" != "--" ]; do extra+=("$1"); shift; done
  shift
  "$HF" jobs run ${DRY:+--dry-run} --detach --name "$name" --flavor "$flavor" --timeout "$timeout" --secrets HF_TOKEN \
    -v "hf://buckets/$NS/$BUCKET:/work" ${extra[@]+"${extra[@]}"} "$image" bash -c "$*"
}

cmd=${1:-help}; shift || true
case "$cmd" in
push-code)
  git diff --quiet HEAD -- tools engine bench || { echo "uncommitted changes in tools/engine/bench; commit first"; exit 1; }
  st=$(mktemp -d)
  git archive HEAD tools engine bench | tar -x -C "$st"
  "$HF" buckets sync "$st" "hf://buckets/$NS/$BUCKET/$PREFIX/code/$SHA"
  rm -rf "$st"
  echo "code $SHA -> hf://buckets/$NS/$BUCKET/$PREFIX/code/$SHA"
  ;;
probe)
  run kolibri-probe cpu-upgrade "${T:-45m}" "$IMG_CPU" -v "hf://$BF16_REPO:/models/bf16" -- "$PRE"'
    echo "[job] arch $(uname -m)"; pip install -q --prefer-binary "huggingface_hub[hf_xet]" "tokenizers>=0.21" pyarrow numpy 2>&1 | tail -3; echo "[job] pip done";
    mkdir -p '"$W"'/probe;
    python3 -u tools/kolibri_probe.py --mount /models/bf16 --repo '"$BF16_REPO"' --local /tmp/probe \
        --bucket-dir '"$W"'/probe --out /tmp/probe.json; cp /tmp/probe.json '"$W"'/probe/probe.json;
    rm -rf /tmp/probe;
    python3 -u tools/kolibri_corpus.py --tokenizer /models/bf16/tokenizer.json --out /tmp/corpus \
        --cache /tmp/hfcache;
    mkdir -p '"$W"'/corpus && cp /tmp/corpus/* '"$W"'/corpus/ && ls -la '"$W"'/corpus'
  ;;
dry)
  run kolibri-dry rtx-pro-6000 "${T:-40m}" "$IMG_TORCH" -- "$PRE"'
    pip install -q -U safetensors "huggingface_hub[hf_xet]" "tokenizers>=0.22" numpy 2>&1 | tail -3; echo "[job] pip done";
    nvidia-smi --query-gpu=name,memory.total --format=csv;
    python3 -u tools/kolibri_quant.py selftest 2>&1 | tee /tmp/selftest.log;
    mkdir -p '"$W"'/dry && cp /tmp/selftest.log '"$W"'/dry/;
    python3 -u tools/kolibri_quant.py build --bf16-repo '"$BF16_REPO"' --local /tmp/bf16 \
        --fp8-repo '"$FP8_REPO"' --fp8-local /tmp/fp8 --corpus '"$W"'/corpus --out '"$W"'/dry \
        --layers 0-1 --calib-tokens '"${CALIB:-50000}"' --force-head --greedy --greedy-tokens 16 '"${EXTRA:-}"''
  ;;
crosscheck)
  run kolibri-crosscheck "${F:-rtx-pro-6000}" "${T:-60m}" "$IMG_VLLM" -- "$PRE"'
    pip install -q --no-deps aleph-alpha-inference==1.0.0;
    nvidia-smi --query-gpu=name,memory.total --format=csv;
    t0=$(date +%s); python3 -c "from huggingface_hub import snapshot_download as s; s('"'"''"$FP8_REPO"''"'"', local_dir='"'"'/tmp/fp8'"'"')" >/dev/null; echo "fp8 download $(( $(date +%s) - t0 )) s, $(du -sh /tmp/fp8 | cut -f1)";
    mkdir -p '"$W"'/crosscheck;
    VLLM_USE_DEEP_GEMM=0 python3 -u tools/kolibri_vllm_check.py --model /tmp/fp8 --corpus '"$W"'/corpus \
        --out /tmp/xc --max-model-len '"${MAXLEN:-4096}"' 2>&1 | tee /tmp/xc-vllm.log;
    cp /tmp/xc/* /tmp/xc-vllm.log '"$W"'/crosscheck/;
    python3 -u tools/kolibri_quant.py ref --fp8 /tmp/fp8 --seqs /tmp/xc/seqs.json --out /tmp/xc/ours.pt;
    python3 -u tools/kolibri_quant.py compare --vllm /tmp/xc/vllm.pt --ours /tmp/xc/ours.pt \
        --tokenizer /tmp/fp8/tokenizer.json --out /tmp/xc/crosscheck.json;
    cp /tmp/xc/ours.pt /tmp/xc/crosscheck.json '"$W"'/crosscheck/'
  ;;
crosscheck-ab)
  # The cross-check's A/B: our forward again on the same sequences, with vLLM's W8A8 activation rounding
  # emulated (the one difference between the two paths that is there by design), against the
  # vllm.pt the cross-check wrote. Torch image, no vLLM.
  run kolibri-crosscheck-ab rtx-pro-6000 "${T:-25m}" "$IMG_TORCH" -- "$PRE"'
    pip install -q -U safetensors "huggingface_hub[hf_xet]" "tokenizers>=0.22" numpy 2>&1 | tail -3; echo "[job] pip done";
    t0=$(date +%s); python3 -c "from huggingface_hub import snapshot_download as s; s('"'"''"$FP8_REPO"''"'"', local_dir='"'"'/tmp/fp8'"'"')" >/dev/null; echo "fp8 download $(( $(date +%s) - t0 )) s";
    mkdir -p /tmp/xc && cp '"$W"'/crosscheck/vllm.pt '"$W"'/crosscheck/seqs.json /tmp/xc/;
    python3 -u tools/kolibri_quant.py ref --fp8 /tmp/fp8 --seqs /tmp/xc/seqs.json --out /tmp/xc/ours-actfp8.pt --act-fp8;
    python3 -u tools/kolibri_quant.py compare --vllm /tmp/xc/vllm.pt --ours /tmp/xc/ours-actfp8.pt \
        --tokenizer /tmp/fp8/tokenizer.json --out /tmp/xc/crosscheck-actfp8.json;
    cp /tmp/xc/ours-actfp8.pt /tmp/xc/crosscheck-actfp8.json '"$W"'/crosscheck/'
  ;;
mixgate)
  # Diagnosis after a failed build gate: the same gate streams with one part of the written set put
  # back in BF16 or FP8 (attention, shared expert, both, head), so the deltas say where the loss is.
  run kolibri-mixgate rtx-pro-6000 "${T:-40m}" "$IMG_TORCH" -- "$PRE"'
    pip install -q -U safetensors "huggingface_hub[hf_xet]" "tokenizers>=0.22" numpy 2>&1 | tail -3; echo "[job] pip done";
    python3 -u tools/kolibri_quant.py mixgate --bf16-repo '"$BF16_REPO"' --local /tmp/bf16 \
        --fp8-repo '"$FP8_REPO"' --fp8-local /tmp/fp8 --set '"$W"'/set --corpus '"$W"'/corpus \
        --out /tmp/mixgate.json; cp /tmp/mixgate.json '"$W"'/set/mixgate.json'
  ;;
attn)
  # Attention only, four recipes (GPTQ toward the FP8 release's attention weights, act order,
  # finer clip search), the experts of set/ kept. Writes set-attn2/ and never touches set/.
  run kolibri-attn rtx-pro-6000 "${T:-45m}" "$IMG_TORCH" -- "$PRE"'
    pip install -q -U safetensors "huggingface_hub[hf_xet]" "tokenizers>=0.22" numpy 2>&1 | tail -3; echo "[job] pip done";
    python3 -u tools/kolibri_quant.py attn --bf16-repo '"$BF16_REPO"' --local /tmp/bf16 \
        --fp8-repo '"$FP8_REPO"' --fp8-local /tmp/fp8 --set '"$W"'/set --corpus '"$W"'/corpus \
        --out '"$W"'/set-attn2 --greedy '"${EXTRA:-}"''
  ;;
build-fp8t)
  # The rebuild: GPTQ toward the FP8 release's weights (the QAT policy, the gate's
  # primary reference), second moments from the policy's own stream. Writes set-fp8t/, never set/.
  run kolibri-build-fp8t rtx-pro-6000 "${T:-90m}" "$IMG_TORCH" -- "$PRE"'
    pip install -q -U safetensors "huggingface_hub[hf_xet]" "tokenizers>=0.22" numpy 2>&1 | tail -3; echo "[job] pip done";
    nvidia-smi --query-gpu=name,memory.total --format=csv; df -h /tmp | tail -1;
    python3 -u tools/kolibri_quant.py build --bf16-repo '"$BF16_REPO"' --local /tmp/bf16 \
        --fp8-repo '"$FP8_REPO"' --fp8-local /tmp/fp8 --corpus '"$W"'/corpus --out '"$W"'/'"${OUT:-set-fp8t}"' \
        --target fp8 --calib-tokens '"${CALIB:-600000}"' '"${EXTRA:-}"''
  ;;
final)
  # The pooled gate over the candidate splits (tools/kolibri_final.py) and the final set + manifest.
  run kolibri-final rtx-pro-6000 "${T:-2h}" "$IMG_TORCH" -- "$PRE"'
    pip install -q -U safetensors "huggingface_hub[hf_xet]" "tokenizers>=0.22" numpy 2>&1 | tail -3; echo "[job] pip done";
    nvidia-smi --query-gpu=name,memory.total --format=csv; df -h /tmp | tail -1;
    python3 -u tools/kolibri_final.py --bf16-repo '"$BF16_REPO"' --local /tmp/bf16 \
        --fp8-repo '"$FP8_REPO"' --fp8-local /tmp/fp8 --corpus '"$W"'/corpus2 --chat '"$W"'/crosscheck \
        --set '"$W"'/'"${SET:-set-fp8t}"' --set-old '"$W"'/set --attn-old '"$W"'/set-attn2/attn/fp8t \
        --out '"$W"'/'"${OUT:-final}"' --greedy '"${EXTRA:-}"''
  ;;
build)
  run kolibri-build rtx-pro-6000 "${T:-5h}" "$IMG_TORCH" -- "$PRE"'
    pip install -q -U safetensors "huggingface_hub[hf_xet]" "tokenizers>=0.22" numpy 2>&1 | tail -3; echo "[job] pip done";
    nvidia-smi --query-gpu=name,memory.total --format=csv; df -h /tmp | tail -1;
    python3 -u tools/kolibri_quant.py build --bf16-repo '"$BF16_REPO"' --local /tmp/bf16 \
        --fp8-repo '"$FP8_REPO"' --fp8-local /tmp/fp8 --corpus '"$W"'/corpus --out '"$W"'/set \
        --calib-tokens '"${CALIB:-600000}"' --greedy '"${EXTRA:-}"''
  ;;
spend)
  curl -s -H "Authorization: Bearer $HF_TOKEN" https://huggingface.co/api/settings/billing/usage/jobs \
    | python3 -c 'import json,sys; u=json.load(sys.stdin)["usage"]; print("jobs this period: $%.2f, %s min" % (u["usedMicroUsd"] / 1e6, u["totalMinutes"])); [print(" ", j) for j in u.get("jobDetails", [])]'
  ;;
ps) "$HF" jobs ps -a | head -20 ;;
logs) "$HF" jobs logs "${1:?job id}" ;;
inspect) "$HF" jobs inspect "${1:?job id}" ;;
cancel) "$HF" jobs cancel "${1:?job id}" ;;
*) sed -n '3,20p' "$0" ;;
esac
