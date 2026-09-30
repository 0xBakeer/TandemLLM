#!/usr/bin/env bash
# TandemLLM installer: Qwen3.8-27B served with StairCut on the published NVFP4 weights.
#
#   curl -fsSL https://raw.githubusercontent.com/0xBakeer/TandemLLM/main/install.sh | bash
#   bash install.sh [--dir DIR] [--port PORT] [--no-start] [--yes] ...
#
# What it does, in order (each step is skipped when it is already done, so a re-run updates):
#   1. checks: NVIDIA GPU and driver, CUDA compiler, disk, memory, Python, Hugging Face access
#   2. the engine: clones (or updates) the repository into DIR/src at a pinned release
#   3. a Python venv in DIR/venv with the pinned dependencies (torch for CUDA 13.0, Triton, ...)
#   4. the weights from Hugging Face, into the normal Hugging Face cache
#   5. a lookup corpus built from public text (tools/build_corpus.py); --no-corpus skips it
#   6. the config: DIR/run/ops/serve.env (the served profile with this machine's paths) and the
#      control script DIR/bin/tandem (start | stop | status | smoke | logs)
#   7. start, wait for /health, a smoke test; --no-start stops after step 6
#
# The primary target is one NVIDIA DGX Spark (GB10, aarch64, sm_121). Other CUDA GPUs are best
# effort: the served profile's skinny NVFP4 kernel is built for sm_121a only and is switched off
# there, and StairCut's cost tables were measured on the Spark.
#
# Nothing here needs root. The repository's own ops/serve.env is never modified.
set -euo pipefail

# ---------------------------------------------------------------------------------------------
# Pinned versions
# ---------------------------------------------------------------------------------------------

# The engine release. v0.3.0 adds image input; text requests decode as in v0.2.0-staircut, the StairCut
# paper's version (merge c111fa7).
# Any tag, branch or commit of the repository works here (TANDEM_REF=main for the newest code).
TANDEM_REF="${TANDEM_REF:-v0.3.0}"
TANDEM_REPO="${TANDEM_REPO:-https://github.com/0xBakeer/TandemLLM.git}"

# The weights. The base checkpoint gives every tensor the NVFP4 overlays do not replace
# (embeddings, norms, the recurrent layers' small tensors) and the tokenizer.
BASE_REPO="Qwen/Qwen3.8-27B-FP8"
NVFP4_REPO="0xBakeer/TandemLLM-Qwen3.8-27B-NVFP4"
B8_REPO="0xBakeer/TandemLLM-Qwen3.8-27B-DFlash2-b8"
B16_REPO="0xBakeer/TandemLLM-Qwen3.8-27B-DFlash2-b16"
# The public text the lookup corpus is built from (Wikipedia, CC BY-SA 4.0), one shard each.
WIKI_REPO="wikimedia/wikipedia"
WIKI_EN="20231101.en/train-00000-of-00041.parquet"
WIKI_DE="20231101.de/train-00000-of-00020.parquet"

TORCH_INDEX="https://download.pytorch.org/whl/cu130"
UV_VERSION="0.12.12"     # only fetched when neither uv nor python3.11 is on the machine
PYPI_INDEX="https://pypi.org/simple"
# The Python packages the engine imports, and everything they pull in, pinned to the versions the
# served engine runs (resolved for Linux aarch64 and x86_64, Python 3.11 and 3.12). torch and
# the nvidia-* wheels come from the CUDA 13.0 index. pyarrow is for the corpus build only.
LOCK="$(cat <<'EOF'
annotated-doc==0.0.5
anyio==4.15.1
certifi==2026.7.22
click==8.5.0
cuda-bindings==13.3.1
cuda-pathfinder==1.8.1
cuda-toolkit==13.0.3.0
filelock==3.32.6
fsspec==2026.6.0
h11==0.16.0
hf-xet==1.6.0
httpcore==1.0.9
httpx==0.28.1
huggingface-hub==1.30.0
idna==3.19
jinja2==3.1.6
markdown-it-py==4.2.0
markupsafe==3.0.3
mdurl==0.1.2
mpmath==1.3.0
networkx==3.6.1
ninja==1.13.2
numpy==2.3.5
nvidia-cublas==13.1.1.3
nvidia-cuda-cupti==13.0.85
nvidia-cuda-nvrtc==13.0.88
nvidia-cuda-runtime==13.0.96
nvidia-cudnn-cu13==9.20.0.48
nvidia-cufft==12.0.0.61
nvidia-cufile==1.15.1.6
nvidia-curand==10.4.0.35
nvidia-cusolver==12.0.4.66
nvidia-cusparse==12.6.3.3
nvidia-cusparselt-cu13==0.8.1
nvidia-nccl-cu13==2.29.7
nvidia-nvjitlink==13.3.33
nvidia-nvshmem-cu13==3.4.5
nvidia-nvtx==13.0.85
packaging==26.3
pyarrow==25.0.1
pygments==2.21.0
pyyaml==6.0.1
regex==2026.9.3
rich==15.0.0
safetensors==0.8.0
setuptools==84.0.0
shellingham==1.5.4
sympy==1.14.0
tokenizers==0.22.2
torch==2.13.0+cu130
tqdm==4.70.0
transformers==5.12.1
triton==3.7.1
typer==0.27.2
typing-extensions==4.16.0
EOF
)"

# Sizes in GiB, for the disk check: what each repository's download adds to the HF cache (the
# base checkpoint without its MTP file, the four NVFP4 files, one drafter each, two Wikipedia
# shards), and the rest of an install.
BASE_GIB=28; NVFP4_GIB=14; DRAFTER_GIB=4; WIKI_GIB=1
LOCAL_GIB=12         # the venv (~8 GiB with the CUDA wheels), the corpus, the logs

# ---------------------------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------------------------

DIR="${TANDEM_DIR:-$HOME/TandemLLM}"
PORT="${TANDEM_PORT:-8000}"
HOST="${TANDEM_HOST:-127.0.0.1}"
SERVED_MODEL="${TANDEM_MODEL_NAME:-qwen3.8-27b-tandem}"
START=1
CORPUS_MODE=build
YES=0
DRY=0
SYSTEMD=0
UNINSTALL=0
REPO_SET=0

usage() {
    cat <<EOF
TandemLLM installer: Qwen3.8-27B with StairCut on one NVIDIA DGX Spark.

Usage: bash install.sh [options]
       curl -fsSL https://raw.githubusercontent.com/0xBakeer/TandemLLM/main/install.sh | bash -s -- [options]

  --dir DIR         install directory (default ~/TandemLLM)
  --port PORT       the API port (default 8000)
  --host ADDR       the address to bind (default 127.0.0.1; 0.0.0.0 serves the LAN, with no auth
                    on the OpenAI routes)
  --ref REF         the engine release: tag, branch or commit (default $TANDEM_REF)
  --repo URL        the git repository to clone (default $TANDEM_REPO)
  --hf-token TOKEN  a Hugging Face token (or set HF_TOKEN; a stored 'hf auth login' works too)
  --no-corpus       skip the lookup corpus (the lookup drafter then copies from the conversation only)
  --no-start        install and configure, but do not start the engine
  --systemd         also install a systemd user unit, tandemllm.service (off by default)
  --yes, -y         non-interactive: answer yes to every question
  --dry-run         print every step and command, change nothing
  --uninstall       stop the engine, remove the unit and DIR (the weights stay in the HF cache)
  -h, --help        this text

Environment: TANDEM_REF, TANDEM_REPO, TANDEM_DIR, TANDEM_PORT, TANDEM_HOST, HF_TOKEN, HF_HOME.
EOF
}

die() { printf '\n[tandem] ERROR: %s\n' "$*" >&2; exit 1; }
need_arg() { [ $# -ge 2 ] && [ -n "$2" ] || die "$1 needs a value"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --dir) need_arg "$@"; DIR="$2"; shift 2 ;;
        --dir=*) DIR="${1#*=}"; shift ;;
        --port) need_arg "$@"; PORT="$2"; shift 2 ;;
        --port=*) PORT="${1#*=}"; shift ;;
        --host) need_arg "$@"; HOST="$2"; shift 2 ;;
        --host=*) HOST="${1#*=}"; shift ;;
        --ref) need_arg "$@"; TANDEM_REF="$2"; shift 2 ;;
        --ref=*) TANDEM_REF="${1#*=}"; shift ;;
        --repo) need_arg "$@"; TANDEM_REPO="$2"; REPO_SET=1; shift 2 ;;
        --repo=*) TANDEM_REPO="${1#*=}"; REPO_SET=1; shift ;;
        --hf-token) need_arg "$@"; export HF_TOKEN="$2"; shift 2 ;;
        --hf-token=*) export HF_TOKEN="${1#*=}"; shift ;;
        --no-corpus) CORPUS_MODE=none; shift ;;
        --no-start) START=0; shift ;;
        --systemd) SYSTEMD=1; shift ;;
        --yes|-y) YES=1; shift ;;
        --dry-run) DRY=1; shift ;;
        --uninstall) UNINSTALL=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; die "unknown option: $1" ;;
    esac
done

case "$PORT" in ''|*[!0-9]*) die "--port must be a number, got '$PORT'" ;; esac
[ "$PORT" -ge 1 ] && [ "$PORT" -le 65535 ] || die "--port out of range: $PORT"
# shellcheck disable=SC2088  # a literal "~" that the caller's shell did not expand
case "$DIR" in "~"|"~/"*) DIR="$HOME${DIR#"~"}" ;; esac
case "$DIR" in /*) ;; *) DIR="$PWD/$DIR" ;; esac
DIR="${DIR%/}"
case "$DIR" in *[[:space:]]*) die "the install directory may not contain spaces: '$DIR'" ;; esac
[ "$DIR" != "$HOME" ] && [ -n "$DIR" ] || die "the install directory cannot be \$HOME itself"

SRC="$DIR/src"
VENV="$DIR/venv"
PY="$VENV/bin/python"
RUN="$DIR/run"
CORPUS_DIR="$DIR/corpus"
MARK="$DIR/.tandemllm-install"
UNIT="$HOME/.config/systemd/user/tandemllm.service"
HF_HUB_DIR="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"
# fast parallel Xet downloads (the defaults leave most of a fast line unused)
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export HF_XET_NUM_CONCURRENT_RANGE_GETS="${HF_XET_NUM_CONCURRENT_RANGE_GETS:-64}"

# ---------------------------------------------------------------------------------------------
# Logging and helpers
# ---------------------------------------------------------------------------------------------

STEP=0
STEPS=7
if [ -t 1 ]; then B=$'\033[1m'; Y=$'\033[33m'; G=$'\033[32m'; N=$'\033[0m'; else B=""; Y=""; G=""; N=""; fi
step() { STEP=$((STEP + 1)); printf '\n%s==> [%d/%d] %s%s\n' "$B" "$STEP" "$STEPS" "$*" "$N"; }
info() { printf '    %s\n' "$*"; }
ok() { printf '    %sok%s %s\n' "$G" "$N" "$*"; }
warn() { printf '    %sWARNING:%s %s\n' "$Y" "$N" "$*" >&2; }
# A failed check stops the install; under --dry-run it is reported and the plan goes on.
fail() { if [ "$DRY" = 1 ]; then warn "(dry-run, would stop here) $*"; else die "$*"; fi; }
# run CMD...: execute, or under --dry-run print it
run() {
    if [ "$DRY" = 1 ]; then printf '    [dry-run] %s\n' "$(printf '%q ' "$@")"; return 0; fi
    "$@"
}
# note TEXT: a step that --dry-run describes instead of printing a command
note() { if [ "$DRY" = 1 ]; then printf '    [dry-run] %s\n' "$*"; fi; }

# confirm QUESTION: yes under --yes; asks on the terminal when there is one (stdin is the script
# itself under curl | bash); with no terminal and no --yes the answer is the default, yes, since
# running the installer was the request. --uninstall asks for --yes explicitly instead.
confirm() {
    [ "$YES" = 1 ] && return 0
    [ "$DRY" = 1 ] && return 0
    local ans=""
    if { : </dev/tty; } 2>/dev/null; then
        printf '    %s [Y/n] ' "$1"
        read -r ans </dev/tty || ans=""
        case "$ans" in ""|y|Y|yes|YES) return 0 ;; *) return 1 ;; esac
    fi
    return 0
}

gib_avail_at() {   # free GiB on the filesystem holding PATH (or its nearest existing parent)
    local p="$1"
    while [ ! -e "$p" ]; do p="$(dirname "$p")"; done
    df -Pk "$p" | awk 'NR == 2 { printf "%d", $4 / 1048576 }'
}
fs_of() {
    local p="$1"
    while [ ! -e "$p" ]; do p="$(dirname "$p")"; done
    df -Pk "$p" | awk 'NR == 2 { print $1 }'
}
meminfo_gib() {
    awk -v k="$1:" '$1 == k { printf "%d", $2 / 1048576; f = 1 } END { if (!f) print 0 }' /proc/meminfo 2>/dev/null || echo 0
}
hf_repo_dir() { printf '%s/models--%s' "$HF_HUB_DIR" "${1//\//--}"; }
sha_of() {
    if command -v sha256sum >/dev/null 2>&1; then sha256sum | cut -c1-16; else shasum -a 256 | cut -c1-16; fi
}

# ---------------------------------------------------------------------------------------------
# --uninstall
# ---------------------------------------------------------------------------------------------

if [ "$UNINSTALL" = 1 ]; then
    printf '%sTandemLLM uninstall: %s%s\n' "$B" "$DIR" "$N"
    [ -d "$DIR" ] || die "nothing installed at $DIR"
    [ -f "$MARK" ] || die "$DIR was not made by this installer (no $MARK); remove it by hand"
    if [ "$YES" != 1 ] && [ "$DRY" != 1 ]; then
        if { : </dev/tty; } 2>/dev/null; then
            printf '    Stop the engine and delete %s? [y/N] ' "$DIR"
            read -r ans </dev/tty || ans=""
            case "$ans" in y|Y|yes|YES) ;; *) die "cancelled" ;; esac
        else
            die "no terminal to confirm on: re-run with --uninstall --yes"
        fi
    fi
    if [ -x "$DIR/bin/tandem" ]; then run "$DIR/bin/tandem" stop 120 || warn "the stop reported an error"; fi
    if [ -f "$UNIT" ] && grep -q "$DIR/bin/tandem" "$UNIT" 2>/dev/null; then
        run systemctl --user disable --now tandemllm.service || true
        run rm -f "$UNIT"
        run systemctl --user daemon-reload || true
    fi
    run rm -rf "$DIR"
    printf '\n    %s %s. The weights stay in the Hugging Face cache (%s):\n' \
        "$([ "$DRY" = 1 ] && echo 'Would remove' || echo 'Removed')" "$DIR" "$HF_HUB_DIR"
    for r in "$BASE_REPO" "$NVFP4_REPO" "$B8_REPO" "$B16_REPO"; do printf '      %s\n' "$(hf_repo_dir "$r")"; done
    printf '      %s/datasets--wikimedia--wikipedia\n' "$HF_HUB_DIR"
    printf '    Delete those directories to free the space (other tools may share the base checkpoint).\n'
    exit 0
fi

printf '%sTandemLLM installer%s  ref %s, into %s, port %s%s\n' "$B" "$N" "$TANDEM_REF" "$DIR" "$PORT" \
    "$([ "$DRY" = 1 ] && echo ', DRY RUN: nothing is changed')"

# ---------------------------------------------------------------------------------------------
# 1. Checks
# ---------------------------------------------------------------------------------------------

step "Checking this machine"
SPARK=0
GPU_NAME=""
GPU_MEM_GIB=0
[ "$(uname -s)" = Linux ] || fail "TandemLLM runs on Linux with an NVIDIA GPU (this is $(uname -s))"
ARCH="$(uname -m)"
case "$ARCH" in aarch64|x86_64) ok "Linux $ARCH" ;; *) fail "unsupported CPU architecture: $ARCH" ;; esac
for tool in git curl awk df; do command -v "$tool" >/dev/null 2>&1 || fail "'$tool' is not installed"; done

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    IFS=',' read -r GPU_NAME DRIVER CC GPU_MEM <<<"$(nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total \
        --format=csv,noheader,nounits | head -1 | sed 's/ *, */,/g')"
    ok "GPU $GPU_NAME, compute capability $CC, driver $DRIVER"
    [ "$(nvidia-smi -L | wc -l)" -gt 1 ] && info "more than one GPU: the engine uses GPU 0 (set CUDA_VISIBLE_DEVICES to choose)"
    [ "${DRIVER%%.*}" -ge 580 ] 2>/dev/null \
        || fail "driver $DRIVER is too old: the CUDA 13.0 wheels need driver 580 or newer"
    if [ "$CC" = "12.1" ]; then
        SPARK=1
        GPU_MEM_GIB="$(meminfo_gib MemTotal)"   # unified memory: the GPU's is the board's
        ok "DGX Spark class board (sm_121), ${GPU_MEM_GIB} GiB unified memory"
    else
        case "$GPU_MEM" in ''|*[!0-9]*) GPU_MEM_GIB="$(meminfo_gib MemTotal)" ;; *) GPU_MEM_GIB=$((GPU_MEM / 1024)) ;; esac
        warn "this is not a DGX Spark (GB10, sm_121). Other GPUs are best effort and untested:"
        warn "the skinny NVFP4 kernel is sm_121a-only and is switched off, StairCut's cost tables"
        warn "were measured on the Spark, and speeds will differ from the published numbers."
        [ "$GPU_MEM_GIB" -ge 40 ] || fail "$GPU_NAME has ${GPU_MEM_GIB} GiB; the engine needs at least 40 GiB of GPU memory"
        confirm "Continue on $GPU_NAME anyway?" || die "stopped at the GPU check"
    fi
else
    fail "no NVIDIA GPU found (nvidia-smi is missing or cannot talk to the driver)"
fi

# The CUDA compiler: the skinny NVFP4 kernel is compiled on first use (torch cpp_extension).
CUDA_HOME_FOUND=""
for c in "${CUDA_HOME:-}" /usr/local/cuda-13.0 /usr/local/cuda-13 /usr/local/cuda; do
    if [ -n "$c" ] && [ -x "$c/bin/nvcc" ]; then CUDA_HOME_FOUND="$c"; break; fi
done
if [ -z "$CUDA_HOME_FOUND" ] && command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME_FOUND="$(dirname "$(dirname "$(command -v nvcc)")")"
fi
if [ -n "$CUDA_HOME_FOUND" ]; then
    NVCC_VER="$("$CUDA_HOME_FOUND/bin/nvcc" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')"
    ok "nvcc $NVCC_VER at $CUDA_HOME_FOUND"
    case "$NVCC_VER" in 13.*) ;; *) warn "nvcc $NVCC_VER: the engine is built and tested with CUDA 13.0" ;; esac
elif [ "$SPARK" = 1 ]; then
    fail "no CUDA compiler (nvcc) found: install the CUDA 13.0 toolkit, or set CUDA_HOME"
else
    warn "no CUDA compiler (nvcc) found; only needed for the sm_121a kernel, which is off here"
fi
if ! command -v g++ >/dev/null 2>&1 && [ "$SPARK" = 1 ]; then
    fail "no C++ compiler (g++): the NVFP4 kernel build needs it (apt install g++)"
fi

# Disk: the weights go to the Hugging Face cache, the rest to DIR.
# still_to_get DIR GIB: GiB of a repository not in the cache yet (0 once it is about complete)
still_to_get() {
    local have=0
    [ -d "$1" ] && have="$(du -sk "$1" 2>/dev/null | awk '{printf "%d", $1 / 1048576 + 0.5}')"
    [ "$have" -ge "$2" ] && echo 0 || echo $(( $2 - have ))
}
TO_GET=$(( $(still_to_get "$(hf_repo_dir "$BASE_REPO")" "$BASE_GIB") + $(still_to_get "$(hf_repo_dir "$NVFP4_REPO")" "$NVFP4_GIB")
         + $(still_to_get "$(hf_repo_dir "$B8_REPO")" "$DRAFTER_GIB") + $(still_to_get "$(hf_repo_dir "$B16_REPO")" "$DRAFTER_GIB") ))
[ "$CORPUS_MODE" = build ] && [ ! -f "$CORPUS_DIR/meta.json" ] \
    && TO_GET=$((TO_GET + $(still_to_get "$HF_HUB_DIR/datasets--wikimedia--wikipedia" "$WIKI_GIB")))
NEED_HF=$((TO_GET + 1))
NEED_DIR=$LOCAL_GIB
[ -d "$VENV" ] && NEED_DIR=2
FREE_HF="$(gib_avail_at "$HF_HUB_DIR")"
FREE_DIR="$(gib_avail_at "$DIR")"
if [ "$(fs_of "$HF_HUB_DIR")" = "$(fs_of "$DIR")" ]; then
    NEED=$((NEED_HF + NEED_DIR))
    if [ "$FREE_DIR" -ge "$NEED" ]; then
        ok "disk: ${FREE_DIR} GiB free, about ${NEED} GiB needed (${TO_GET} GiB to download)"
    else
        fail "disk: ${FREE_DIR} GiB free, about ${NEED} GiB needed (${TO_GET} GiB to download)"
    fi
elif [ "$FREE_HF" -lt "$NEED_HF" ]; then
    fail "disk: ${FREE_HF} GiB free for the HF cache ($HF_HUB_DIR), about ${NEED_HF} GiB needed"
elif [ "$FREE_DIR" -lt "$NEED_DIR" ]; then
    fail "disk: ${FREE_DIR} GiB free at $DIR, about ${NEED_DIR} GiB needed"
else
    ok "disk: ${FREE_HF} GiB free for the HF cache, ${FREE_DIR} GiB at $DIR"
fi

# Memory: the served stack takes about 64 GiB of the board at load (docs/operations.md).
MEM_AVAIL="$(meminfo_gib MemAvailable)"
MIN_FREE=64; [ "$SPARK" = 1 ] || MIN_FREE=16      # host RAM on a discrete-GPU box
if [ "$MEM_AVAIL" -ge "$MIN_FREE" ]; then
    ok "memory: ${MEM_AVAIL} GiB available"
elif [ "$START" = 1 ]; then
    fail "memory: ${MEM_AVAIL} GiB available, the engine needs about ${MIN_FREE} GiB to load. Stop other GPU work, or install with --no-start"
else
    warn "memory: ${MEM_AVAIL} GiB available; about ${MIN_FREE} GiB must be free when you start the engine"
fi
if pgrep -f 'server/app\.py' >/dev/null 2>&1; then
    if [ "$START" = 1 ]; then
        fail "a TandemLLM engine is already running (pgrep 'server/app.py'). One engine per board: stop it, or install with --no-start"
    fi
    warn "an engine is already running; this install will not start a second one (--no-start)"
fi
if [ "$START" = 1 ] && command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | awk '{print $4}' | grep -q ":$PORT\$"; then
    fail "port $PORT is in use; choose another with --port"
fi

# Python 3.11, the served engine's: tests/test_grammar.py passes under 3.11 and fails under 3.12
# (the structured-output grammar's empty-mask check). uv when it is on PATH, else a system
# python3.11 with venv, else uv (pinned) is fetched into DIR/uv and brings its own 3.11 into
# DIR/python -- user-local, no root.
PY_MODE=""
UV=""
if command -v uv >/dev/null 2>&1; then
    PY_MODE=uv; UV="$(command -v uv)"; ok "uv $(uv --version | awk '{print $2}') (the venv gets Python 3.11)"
elif command -v python3.11 >/dev/null 2>&1 && python3.11 -c 'import venv, ensurepip' 2>/dev/null; then
    PY_MODE=venv; ok "$(python3.11 -V) with venv"
else
    PY_MODE=fetch-uv; UV="$DIR/uv/uv"
    info "no uv and no python3.11: uv $UV_VERSION goes into $DIR/uv and fetches Python 3.11 (user-local)"
fi

# Hugging Face access: the published repositories are private until the release.
HF_TOKEN_FILE="${HF_TOKEN_PATH:-${HF_HOME:-$HOME/.cache/huggingface}/token}"
HF_AUTH=""
if [ -n "${HF_TOKEN:-}" ]; then HF_AUTH="$HF_TOKEN"; ok "HF token from HF_TOKEN/--hf-token"
elif [ -r "$HF_TOKEN_FILE" ]; then HF_AUTH="$(cat "$HF_TOKEN_FILE")"; ok "HF token from $HF_TOKEN_FILE"
else info "no Hugging Face token: fine once the weights are public"
fi
for r in "$BASE_REPO" "$NVFP4_REPO" "$B8_REPO" "$B16_REPO"; do
    # the token goes to curl on stdin, never on its command line
    code="$( { [ -n "$HF_AUTH" ] && printf 'header = "Authorization: Bearer %s"\n' "$HF_AUTH"; true; } \
        | curl -s -K - -o /dev/null -w '%{http_code}' -m 20 "https://huggingface.co/api/models/$r" || true)"
    case "$code" in
        200) ;;
        401|403|404) fail "cannot read $r on Hugging Face (HTTP $code). The repositories are private until the release: pass --hf-token, set HF_TOKEN, or run 'hf auth login' with an account that has access" ;;
        *) fail "cannot reach huggingface.co for $r (HTTP ${code:-none}); check the network" ;;
    esac
done
ok "Hugging Face: the four model repositories are readable"
unset HF_AUTH

cat <<EOF

    The plan: engine $TANDEM_REF into $SRC, a venv, about $TO_GET GiB of
    downloads into $HF_HUB_DIR, $([ "$CORPUS_MODE" = build ] && echo "a public lookup corpus," || echo "no corpus,") the config$([ "$START" = 1 ] && echo ", then start on port $PORT" || echo " (no start)").
EOF
confirm "Go ahead?" || die "cancelled"
if [ "$DRY" != 1 ]; then mkdir -p "$DIR" "$RUN/ops" "$DIR/bin" "$DIR/state"; touch "$MARK"; fi

# ---------------------------------------------------------------------------------------------
# 2. The engine
# ---------------------------------------------------------------------------------------------

step "Engine source: $TANDEM_REF"
export GIT_TERMINAL_PROMPT=0     # a private repository fails here with a message, not a prompt
if [ -d "$SRC/.git" ]; then
    [ "$REPO_SET" = 1 ] && run git -C "$SRC" remote set-url origin "$TANDEM_REPO"
    if [ -n "$(git -C "$SRC" status --porcelain --untracked-files=no)" ]; then
        fail "$SRC has local changes; commit or stash them (git -C $SRC stash), then re-run"
    fi
    run git -C "$SRC" fetch --quiet --tags --force --prune origin || die "git fetch failed (origin: $(git -C "$SRC" remote get-url origin))"
else
    [ -e "$SRC" ] && die "$SRC exists and is not a git checkout; move it away"
    run git clone --quiet "$TANDEM_REPO" "$SRC" \
        || die "git clone $TANDEM_REPO failed. While the repository is private, clone access is needed: --repo git@github.com:0xBakeer/TandemLLM.git (an SSH key) or a git credential helper"
fi
if [ "$DRY" = 1 ] && [ ! -d "$SRC/.git" ]; then
    note "git checkout --detach $TANDEM_REF"
else
    TARGET=""
    for cand in "refs/tags/$TANDEM_REF" "refs/remotes/origin/$TANDEM_REF" "$TANDEM_REF"; do
        if TARGET="$(git -C "$SRC" rev-parse -q --verify "$cand^{commit}" 2>/dev/null)"; then break; fi
        TARGET=""
    done
    [ -n "$TARGET" ] || die "no tag, branch or commit '$TANDEM_REF' in $TANDEM_REPO"
    run git -C "$SRC" -c advice.detachedHead=false checkout --quiet --detach "$TARGET"
    ok "$SRC at $TANDEM_REF (${TARGET:0:7}), version $(cat "$SRC/VERSION" 2>/dev/null || echo '?')"
fi

# ---------------------------------------------------------------------------------------------
# 3. Python
# ---------------------------------------------------------------------------------------------

step "Python environment: $VENV"
LOCK_SHA="$(printf '%s\n' "$LOCK" | sha_of)"
if [ -x "$PY" ] && ! "$PY" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 11))' 2>/dev/null; then
    info "$VENV is not Python 3.11 ($("$PY" -V 2>&1)); rebuilding it"
    run rm -rf "$VENV"
fi
if [ -x "$PY" ] && [ "$(cat "$VENV/.tandem-lock" 2>/dev/null)" = "$LOCK_SHA" ]; then
    ok "up to date (lock $LOCK_SHA)"
else
    if [ "$PY_MODE" = fetch-uv ]; then
        # uv's own Python downloads stay inside DIR too
        export UV_PYTHON_INSTALL_DIR="$DIR/python"
        if [ ! -x "$UV" ]; then
            note "curl -LsSf https://astral.sh/uv/$UV_VERSION/install.sh | UV_INSTALL_DIR=$DIR/uv UV_NO_MODIFY_PATH=1 sh"
            if [ "$DRY" != 1 ]; then
                curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" \
                    | env UV_INSTALL_DIR="$DIR/uv" UV_NO_MODIFY_PATH=1 sh >/dev/null \
                    || die "could not install uv; install python3.11 with venv, or uv, and re-run"
                [ -x "$UV" ] || UV="$DIR/uv/bin/uv"
                [ -x "$UV" ] || die "uv was installed but not found under $DIR/uv"
            fi
        fi
    fi
    if [ ! -x "$PY" ]; then
        if [ "$PY_MODE" = venv ]; then run python3.11 -m venv "$VENV"
        else run "$UV" venv --quiet --python 3.11 "$VENV"
        fi
    fi
    if [ "$DRY" != 1 ]; then printf '%s\n' "$LOCK" > "$RUN/requirements.lock"; fi
    info "installing $(printf '%s\n' "$LOCK" | wc -l | tr -d ' ') pinned packages (torch 2.13.0+cu130, triton 3.7.1, transformers 5.12.1, ...)"
    if [ "$PY_MODE" = venv ]; then
        run "$PY" -m pip install --disable-pip-version-check -r "$RUN/requirements.lock" \
            --index-url "$PYPI_INDEX" --extra-index-url "$TORCH_INDEX"
    else
        run "$UV" pip install --python "$PY" -r "$RUN/requirements.lock" \
            --index-url "$PYPI_INDEX" --extra-index-url "$TORCH_INDEX" --index-strategy unsafe-best-match
    fi
    if [ "$DRY" != 1 ]; then printf '%s\n' "$LOCK_SHA" > "$VENV/.tandem-lock"; fi
fi
if [ "$DRY" != 1 ]; then
    "$PY" - <<'EOF' || die "the Python environment does not see the GPU (see above)"
import torch, triton, transformers
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
cc = ".".join(map(str, torch.cuda.get_device_capability(0)))
print(f"    ok torch {torch.__version__} (CUDA {torch.version.cuda}), triton {triton.__version__}, "
      f"transformers {transformers.__version__}; {torch.cuda.get_device_name(0)} sm_{cc.replace('.', '')}")
EOF
fi

# ---------------------------------------------------------------------------------------------
# 4. Weights
# ---------------------------------------------------------------------------------------------

step "Weights (Hugging Face cache: $HF_HUB_DIR)"
# hf_get REPO PATTERN...: download (or verify) the files matching the patterns; prints the
# snapshot directory on stdout, progress on stderr. Resumes an interrupted download.
hf_get() {
    "$PY" - "$@" <<'EOF'
import sys
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError, GatedRepoError
repo, patterns = sys.argv[1], sys.argv[2:]
try:
    path = snapshot_download(repo, allow_patterns=patterns, max_workers=16)
except (RepositoryNotFoundError, GatedRepoError) as e:
    sys.exit(f"cannot read {repo}: {type(e).__name__}. Private until the release: pass --hf-token or set HF_TOKEN.")
except HfHubHTTPError as e:
    sys.exit(f"download of {repo} failed: {e}")
print(path)
EOF
}
# The board shares one memory between CPU and GPU, and page cache is not handed back fast enough
# for the GPU's allocations (docs/operations.md): drop the cache of what is downloaded as it lands.
drop_cache_loop() {
    while :; do "$PY" "$SRC/tools/drop_page_cache.py" "$@" >/dev/null 2>&1 || true; sleep 5; done
}
if [ "$DRY" = 1 ]; then
    note "snapshot_download $BASE_REPO  (layers-*.safetensors, outside.safetensors, tokenizer and config; not mtp.safetensors)"
    note "snapshot_download $NVFP4_REPO  (mlp, gdn, attn, head-fp8 .safetensors)"
    note "snapshot_download $B8_REPO, $B16_REPO  (config.json, model.safetensors)"
    SNAP_BASE="$(hf_repo_dir "$BASE_REPO")/snapshots/<rev>"; SNAP_NV="$(hf_repo_dir "$NVFP4_REPO")/snapshots/<rev>"
    SNAP_B8="$(hf_repo_dir "$B8_REPO")/snapshots/<rev>"; SNAP_B16="$(hf_repo_dir "$B16_REPO")/snapshots/<rev>"
else
    drop_cache_loop "$(hf_repo_dir "$BASE_REPO")" "$(hf_repo_dir "$NVFP4_REPO")" \
        "$(hf_repo_dir "$B8_REPO")" "$(hf_repo_dir "$B16_REPO")" &
    DROPPER=$!
    trap 'kill "$DROPPER" 2>/dev/null || true' EXIT
    info "$BASE_REPO"
    SNAP_BASE="$(hf_get "$BASE_REPO" '*.json' '*.jinja' 'merges.txt' 'LICENSE' 'layers-*.safetensors' 'outside.safetensors' | tail -n 1)" \
        || die "base checkpoint download failed"
    info "$NVFP4_REPO"
    SNAP_NV="$(hf_get "$NVFP4_REPO" 'mlp.safetensors' 'gdn.safetensors' 'attn.safetensors' 'head-fp8.safetensors' \
        'README.md' 'LICENSE' 'quality/sha256.txt' | tail -n 1)" || die "NVFP4 download failed"
    info "$B8_REPO"
    SNAP_B8="$(hf_get "$B8_REPO" 'config.json' 'model.safetensors' 'README.md' 'LICENSE' | tail -n 1)" || die "b8 drafter download failed"
    info "$B16_REPO"
    SNAP_B16="$(hf_get "$B16_REPO" 'config.json' 'model.safetensors' 'README.md' 'LICENSE' | tail -n 1)" || die "b16 drafter download failed"
    kill "$DROPPER" 2>/dev/null || true
    trap - EXIT
    "$PY" "$SRC/tools/drop_page_cache.py" "$SNAP_BASE" "$SNAP_NV" "$SNAP_B8" "$SNAP_B16" >/dev/null 2>&1 || true
    for f in "$SNAP_BASE/config.json" "$SNAP_BASE/tokenizer.json" "$SNAP_BASE/outside.safetensors" \
             "$SNAP_NV/mlp.safetensors" "$SNAP_NV/gdn.safetensors" "$SNAP_NV/attn.safetensors" \
             "$SNAP_NV/head-fp8.safetensors" "$SNAP_B8/model.safetensors" "$SNAP_B16/model.safetensors"; do
        [ -s "$f" ] || die "missing after download: $f"
    done
    ok "base      $SNAP_BASE"
    ok "NVFP4     $SNAP_NV"
    ok "drafters  $SNAP_B8"
    ok "          $SNAP_B16"
fi

# ---------------------------------------------------------------------------------------------
# 5. Lookup corpus
# ---------------------------------------------------------------------------------------------

# The served engine's corpus (38 M tokens) is not published. This builds a smaller one of the same
# kinds of public text with the repository's own tool: English and German Wikipedia (one shard
# each) and the Python sources of the installed transformers, torch and triton packages, about
# 31 M tokens. The paper's numbers were measured with the unpublished corpus; with this one, or
# none, the lookup drafter proposes different continuations and the speed differs (not measured).
step "Lookup corpus"
if [ "$CORPUS_MODE" = none ]; then
    info "skipped (--no-corpus): the lookup drafter copies from the conversation only"
    CORPUS_VALUE=""
elif [ -f "$CORPUS_DIR/meta.json" ]; then
    ok "already built: $CORPUS_DIR ($(sed -n 's/.*"n_tokens": \([0-9]*\).*/\1/p' "$CORPUS_DIR/meta.json") tokens)"
    CORPUS_VALUE="$CORPUS_DIR"
else
    CORPUS_VALUE="$CORPUS_DIR"
    if [ "$DRY" = 1 ]; then
        note "hf_hub_download $WIKI_REPO $WIKI_EN and $WIKI_DE (public, CC BY-SA 4.0, 1.2 GB)"
        note "python tools/build_corpus.py --model <base> --out $CORPUS_DIR --parquet <en>:text::12000000 --parquet <de>:text::7000000 --src <site-packages>/{transformers,torch,triton} --src engine --src tools"
    else
        info "fetching one English and one German Wikipedia shard ($WIKI_REPO, CC BY-SA 4.0)"
        WIKI_PATHS="$("$PY" - "$WIKI_REPO" "$WIKI_EN" "$WIKI_DE" <<'EOF'
import sys
from huggingface_hub import hf_hub_download
repo = sys.argv[1]
for f in sys.argv[2:]:
    print(hf_hub_download(repo, f, repo_type="dataset"))
EOF
)" || die "Wikipedia download failed (or re-run with --no-corpus)"
        EN_PQ="$(sed -n 1p <<<"$WIKI_PATHS")"; DE_PQ="$(sed -n 2p <<<"$WIKI_PATHS")"
        SITE="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
        info "tokenising and indexing (CPU only, a few minutes)"
        rm -rf "$CORPUS_DIR.partial"
        (cd "$SRC" && nice -n 10 "$PY" tools/build_corpus.py --model "$SNAP_BASE" --out "$CORPUS_DIR.partial" \
            --parquet "$EN_PQ:text::12000000" --parquet "$DE_PQ:text::7000000" \
            --src "$SITE/transformers:6000000" --src "$SITE/torch:5000000" --src "$SITE/triton:1500000" \
            --src "$SRC/engine" --src "$SRC/tools" --max-tokens 40000000) 2>&1 | sed 's/^/    | /' \
            || die "the corpus build failed (re-run with --no-corpus to skip it)"
        [ -f "$CORPUS_DIR.partial/meta.json" ] || die "the corpus build wrote no meta.json"
        mv "$CORPUS_DIR.partial" "$CORPUS_DIR"
        ok "built: $CORPUS_DIR"
    fi
fi

# ---------------------------------------------------------------------------------------------
# 6. Config and control script
# ---------------------------------------------------------------------------------------------

step "Configuration: $RUN/ops/serve.env"
TEMPLATE="$SRC/ops/serve.env"
if [ ! -f "$TEMPLATE" ] && [ "$DRY" = 1 ]; then
    here="$(cd "$(dirname "${BASH_SOURCE[0]:-.}")" 2>/dev/null && pwd || true)"
    [ -n "$here" ] && [ -f "$here/ops/serve.env" ] && TEMPLATE="$here/ops/serve.env"
fi
q() { printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"; }    # single-quoted for the env file
# The keys this install sets. Every other line of the release's ops/serve.env -- StairCut, the
# tree, the kernels, every QWEN38_* switch -- is copied as it is.
ENV_KEYS=()
ENV_ASSIGN=()
setv() { ENV_KEYS+=("$1"); ENV_ASSIGN+=("TV_$1=$2"); }
setv REPO "$(q "$SRC")"
setv PY "$(q "$PY")"
setv PYTHONPATH ""
setv HOST "$(q "$HOST")"
setv PORT "$PORT"
setv SERVED_MODEL "$(q "$SERVED_MODEL")"
setv NV "$(q "$SNAP_NV/mlp.safetensors,$SNAP_NV/gdn.safetensors,$SNAP_NV/attn.safetensors")"
setv HEAD "$(q "$SNAP_NV/head-fp8.safetensors")"
setv CKPT8 "$(q "$SNAP_B8")"
setv CKPT16 "$(q "$SNAP_B16")"
setv CORPUS "$(q "$CORPUS_VALUE")"
setv QSE_USAGE_LEDGER "$(q "$DIR/state/usage/ledger.sqlite3")"
setv DROP_PAGE_CACHE 0              # bin/tandem drops the weights' page cache with this install's paths
setv QWEN38_MODEL "$(q "$SNAP_BASE")"
setv QSE_SECRETS "$(q "$DIR/state/secrets.env")"
if [ -n "$CUDA_HOME_FOUND" ]; then setv CUDA_HOME "$(q "$CUDA_HOME_FOUND")"; fi
if [ "$SPARK" != 1 ]; then
    # best effort off the Spark: the skinny kernel is sm_121a code; a smaller context below 100 GiB
    setv QWEN38_NVFP4_SKINNY 0
    setv QWEN38_SKINNY_SRUN 0
    if [ "$GPU_MEM_GIB" -lt 100 ]; then setv MAX_LEN 65536; fi
fi
render_env() {
    printf '# Written by install.sh on %s for %s. Regenerated on every run of the installer:\n' "$(date -Iseconds 2>/dev/null || date)" "$DIR"
    printf '# put your own settings in %s/local.env, which is read last.\n' "$RUN"
    printf '# The template is the release'"'"'s ops/serve.env (%s); these keys are this machine'"'"'s:\n' "$TANDEM_REF"
    printf '#   %s\n\n' "${ENV_KEYS[*]}"
    # shellcheck disable=SC2016  # the $0/$1 are awk's
    env "${ENV_ASSIGN[@]}" awk -v keys="${ENV_KEYS[*]}" '
        BEGIN { n = split(keys, k, " "); for (i = 1; i <= n; i++) want[k[i]] = 1 }
        match($0, /^[A-Za-z_][A-Za-z0-9_]*=/) {
            key = substr($0, 1, RLENGTH - 1)
            if (key in want) { if (!(key in seen)) print key "=" ENVIRON["TV_" key]; seen[key] = 1; next }
        }
        { print }
        END {
            printf "\n# set by install.sh (not in the template)\n"
            for (i = 1; i <= n; i++) if (!(k[i] in seen)) print k[i] "=" ENVIRON["TV_" k[i]]
        }' "$TEMPLATE"
    printf '\nif [ -f %s ]; then . %s; fi\n' "$(q "$RUN/local.env")" "$(q "$RUN/local.env")"
}
if [ ! -f "$TEMPLATE" ]; then
    note "render $SRC/ops/serve.env with this machine's paths into $RUN/ops/serve.env"
else
    if [ "$DRY" = 1 ]; then
        note "would write $RUN/ops/serve.env; the lines this install changes:"
        render_env | grep -E "^($(IFS='|'; echo "${ENV_KEYS[*]}"))=" | sed 's/^/        /'
    else
        render_env > "$RUN/ops/serve.env.tmp"
        # it is sourced by bash: prove it parses before it replaces the old one
        bash -n "$RUN/ops/serve.env.tmp" || die "the generated serve.env does not parse"
        mv "$RUN/ops/serve.env.tmp" "$RUN/ops/serve.env"
        # the release's own service scripts, next to this env file (they source ./serve.env)
        for s in start.sh stop.sh engines.sh; do install -m 755 "$SRC/ops/$s" "$RUN/ops/$s"; done
        mkdir -p "$DIR/state/usage"
        ok "wrote $RUN/ops/serve.env (StairCut on: QWEN38_LEN_SWITCH=$(sed -n 's/^QWEN38_LEN_SWITCH=//p' "$RUN/ops/serve.env"), QWEN38_LEN_MODE=$(sed -n 's/^QWEN38_LEN_MODE=//p' "$RUN/ops/serve.env"))"
    fi
fi

# DIR/bin/tandem: start | stop | restart | status | smoke | logs
write_control() {
    printf '#!/usr/bin/env bash\n# TandemLLM control script, written by install.sh.\nTANDEM_DIR=%q\n' "$DIR"
    cat <<'CTL'
set -euo pipefail
RUN="$TANDEM_DIR/run"
ENV_FILE="$RUN/ops/serve.env"
[ -f "$ENV_FILE" ] || { echo "[tandem] no $ENV_FILE; re-run install.sh" >&2; exit 1; }
set -a
# shellcheck source=/dev/null
. "$ENV_FILE"
set +a
URL="http://127.0.0.1:$PORT"
healthy() { curl -sf -m 3 "$URL/health" >/dev/null 2>&1; }
engine_alive() { [ -f "$REPO/logs/engine.pid" ] && kill -0 "$(cat "$REPO/logs/engine.pid")" 2>/dev/null; }

memcheck() {
    local avail need
    [ "${TANDEM_SKIP_MEMCHECK:-0}" = 1 ] && return 0
    if nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | grep -q '^12\.1'; then
        avail=$(awk '$1 == "MemAvailable:" { printf "%d", $2 / 1048576 }' /proc/meminfo)
        need="${TANDEM_MIN_FREE_GIB:-64}"
    else
        avail=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | awk '{printf "%d", $1 / 1024}')
        need="${TANDEM_MIN_FREE_GIB:-36}"
    fi
    if [ "$avail" -lt "$need" ]; then
        echo "[tandem] ${avail} GiB free, the engine needs about ${need} GiB to load; stop other GPU work first" >&2
        echo "         (TANDEM_SKIP_MEMCHECK=1 skips this check)" >&2
        exit 1
    fi
}

start() {
    if healthy; then echo "[tandem] already healthy on :$PORT"; return 0; fi
    memcheck
    local rc=0 t0 limit
    bash "$RUN/ops/start.sh" || rc=$?
    if [ "$rc" != 0 ]; then
        # start.sh waits 600 s; a first load also compiles kernels, so keep waiting while it lives
        engine_alive || { echo "[tandem] the engine did not start; log: $REPO/logs/engine.log" >&2; return 1; }
        limit="${TANDEM_HEALTH_TIMEOUT:-1800}"; t0=$(date +%s)
        echo "[tandem] still loading; waiting up to ${limit}s for /health"
        until healthy; do
            engine_alive || { echo "[tandem] the engine exited; tail of the log:" >&2; tail -30 "$REPO/logs/engine.log" >&2; return 1; }
            [ $(( $(date +%s) - t0 )) -lt "$limit" ] || { echo "[tandem] not healthy after ${limit}s" >&2; return 1; }
            sleep 10
        done
    fi
    # the weights are on the device now: drop their page cache (docs/operations.md)
    local -a nv
    IFS=, read -r -a nv <<<"$NV"
    "$PY" "$REPO/tools/drop_page_cache.py" "$QWEN38_MODEL" "${nv[@]}" "$HEAD" "$CKPT8" "$CKPT16" 2>&1 | tail -1 || true
    echo "[tandem] healthy: $URL/v1"
}

smoke() {
    "$PY" - "$URL" "$SERVED_MODEL" <<'PY'
import json, sys, time, urllib.request
url, model = sys.argv[1], sys.argv[2]
with urllib.request.urlopen(url + "/v1/models", timeout=30) as r:
    models = [m["id"] for m in json.load(r)["data"]]
print(f"[smoke] /v1/models: {', '.join(models)}")
body = {"model": model, "stream": True, "stream_options": {"include_usage": True},
        "temperature": 0, "max_tokens": 256, "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content":
                      "Write a Python function that returns the n-th Fibonacci number iteratively, "
                      "with a docstring and two doctests."}]}
req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
t0 = time.perf_counter(); first = None; n = 0; usage = None; text = []
with urllib.request.urlopen(req, timeout=600) as r:
    for raw in r:
        line = raw.decode().strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        chunk = json.loads(line[5:])
        if chunk.get("usage"):
            usage = chunk["usage"]
        for ch in chunk.get("choices") or []:
            piece = (ch.get("delta") or {}).get("content")
            if piece:
                if first is None:
                    first = time.perf_counter()
                n += 1
                text.append(piece)
t1 = time.perf_counter()
tokens = (usage or {}).get("completion_tokens") or n
if first is None or tokens < 2:
    sys.exit("[smoke] FAILED: no content came back")
print("[smoke] reply: " + "".join(text)[:160].replace("\n", " ") + " ...")
print(f"[smoke] {tokens} tokens, first token after {1000 * (first - t0):.0f} ms, "
      f"{(tokens - 1) / (t1 - first):.1f} tok/s decode (one request, a cold engine's first)")
PY
}

case "${1:-}" in
    start) start ;;
    stop) bash "$RUN/ops/stop.sh" "${2:-60}" ;;
    restart) bash "$RUN/ops/stop.sh" "${2:-60}"; start ;;
    status)
        if healthy; then echo "[tandem] healthy on :$PORT"; curl -s -m 5 "$URL/v1/models"; echo
        elif engine_alive; then echo "[tandem] loading (pid $(cat "$REPO/logs/engine.pid"))"
        else echo "[tandem] not running"; exit 1; fi ;;
    smoke) smoke ;;
    logs) tail -n "${2:-100}" -f "$REPO/logs/engine.log" ;;
    env) cat "$ENV_FILE" ;;
    *) echo "usage: $0 start | stop [grace-s] | restart | status | smoke | logs [n] | env" >&2; exit 2 ;;
esac
CTL
}
if [ "$DRY" = 1 ]; then
    note "would write $DIR/bin/tandem (start | stop | restart | status | smoke | logs | env)"
else
    write_control > "$DIR/bin/tandem.tmp" && chmod 755 "$DIR/bin/tandem.tmp" && mv "$DIR/bin/tandem.tmp" "$DIR/bin/tandem"
    ok "wrote $DIR/bin/tandem"
fi

if [ "$SYSTEMD" = 1 ]; then
    info "systemd user unit: $UNIT"
    info "one engine per board: enable it only if nothing else (cron, another unit) starts one here;"
    info "bin/tandem start refuses while any engine is alive, but two supervisors can still race at boot"
    if [ "$DRY" = 1 ]; then
        note "would write $UNIT (ExecStart=$DIR/bin/tandem start) and enable it"
    else
        mkdir -p "$(dirname "$UNIT")"
        cat > "$UNIT" <<EOF
[Unit]
Description=TandemLLM: Qwen3.8-27B with StairCut, OpenAI-compatible API on :$PORT
After=network-online.target

[Service]
# bin/tandem start runs the release's ops/start.sh, which starts the server in the background
# and returns once /health answers; the server stays in this unit's cgroup.
Type=oneshot
RemainAfterExit=yes
ExecStart=$DIR/bin/tandem start
ExecStop=$DIR/bin/tandem stop 120
TimeoutStartSec=2400
TimeoutStopSec=180

[Install]
WantedBy=default.target
EOF
        if systemctl --user daemon-reload && systemctl --user enable tandemllm.service >/dev/null; then
            ok "enabled tandemllm.service (user). To start it at boot without a login: sudo loginctl enable-linger $USER"
        else
            warn "systemctl --user failed (no user session bus?); the unit file is at $UNIT"
        fi
    fi
fi

# ---------------------------------------------------------------------------------------------
# 7. Start, health, smoke test
# ---------------------------------------------------------------------------------------------

step "Start"
if [ "$START" != 1 ]; then
    info "skipped (--no-start). Start it with: $DIR/bin/tandem start"
elif [ "$DRY" = 1 ]; then
    note "$DIR/bin/tandem start   (ops/start.sh with $RUN/ops/serve.env; waits for /health)"
    note "$DIR/bin/tandem smoke   (/v1/models and one short chat completion, prints tok/s)"
else
    info "loading the model (a few minutes; the first start also compiles kernels)"
    if [ "$SYSTEMD" = 1 ] && systemctl --user is-enabled tandemllm.service >/dev/null 2>&1; then
        systemctl --user start tandemllm.service || die "tandemllm.service did not start: journalctl --user -u tandemllm; $SRC/logs/engine.log"
    else
        "$DIR/bin/tandem" start || die "the engine did not come up; log: $SRC/logs/engine.log"
    fi
    "$DIR/bin/tandem" smoke || die "the smoke test failed; log: $SRC/logs/engine.log"
fi

API_HOST="$HOST"; [ "$HOST" = 0.0.0.0 ] && API_HOST="$(hostname -I 2>/dev/null | awk '{print $1}')"; API_HOST="${API_HOST:-127.0.0.1}"
cat <<EOF

${B}Done.${N}  TandemLLM $TANDEM_REF in $DIR

  Control:   $DIR/bin/tandem start | stop | status | smoke | logs
  Config:    $RUN/ops/serve.env   (your overrides: $RUN/local.env)
  Update:    re-run the installer (same --dir); downloads resume, finished steps are skipped
  Remove:    bash install.sh --dir $DIR --uninstall   (the weights stay in the HF cache)
  Dashboard: http://$API_HOST:$PORT/dashboard (no sign-in: anyone who reaches the port can read it)
             to require the admin token: QSE_DASHBOARD_LOGIN=on in $RUN/local.env,
             QSE_STATE_DIR=$DIR/state bash $SRC/ops/make-secrets.sh, $DIR/bin/tandem restart;
             the token: grep QSE_ADMIN_TOKEN $DIR/state/secrets.env

  OpenAI-compatible API at http://$API_HOST:$PORT/v1, model "$SERVED_MODEL", no API key needed:

    export OPENAI_BASE_URL=http://$API_HOST:$PORT/v1 OPENAI_API_KEY=none
    curl -s \$OPENAI_BASE_URL/chat/completions -H 'Content-Type: application/json' \\
      -d '{"model": "$SERVED_MODEL", "messages": [{"role": "user", "content": "Say hello."}]}'

  opencode (~/.config/opencode/opencode.json):

    {
      "\$schema": "https://opencode.ai/config.json",
      "provider": {
        "tandem": {
          "npm": "@ai-sdk/openai-compatible",
          "name": "TandemLLM",
          "options": { "baseURL": "http://$API_HOST:$PORT/v1" },
          "models": { "$SERVED_MODEL": { "name": "Qwen3.8-27B (TandemLLM)" } }
        }
      }
    }
EOF
