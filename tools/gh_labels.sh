#!/usr/bin/env bash
# The GitHub labels of the repository, as one table (OPS-229). Creates the missing ones and updates the
# colour and description of the rest; renames enhancement to feature and documentation to docs, which
# keeps their issues. Never deletes a label. docs/labels.md is this table (tests/test_gh_labels.py).
#
#   bash tools/gh_labels.sh [--repo OWNER/NAME] [--dry-run]     apply (needs gh with admin on the repo)
#   bash tools/gh_labels.sh --table                             print the table as markdown
#
# Bash 3.2 compatible.
set -euo pipefail

REPO="0xBakeer/TandemLLM"; DRY=0; TABLE=0
while [ $# -gt 0 ]; do
    case "$1" in
        --repo) REPO="$2"; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        --table) TABLE=1; shift ;;
        -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

# group|name|colour|description (a description is at most 100 characters)
LABELS='type|bug|0a3069|Something is broken: a wrong answer, a crash, a hang, a wrong number
type|feature|0550ae|A new capability, or a change to how something works
type|perf|0969da|Speed or memory: tok/s, time to first token, prefill, GPU memory
type|docs|218bff|README, docs/ or the manual
type|refactor|54aeff|Code that changes shape and keeps its behaviour and output
type|test|80ccff|Tests, test tools or the test suite
type|security|033d8b|A security fix or hardening. Report a vulnerability in private: SECURITY.md
type|question|b6e3ff|A question about using TandemLLM
area|engine/core|9a3412|Engine: model code, loaders, the decode and verify loop
area|engine/kernels|b45309|Engine: GPU kernels, matrix products, attention
area|engine/drafting|c2410c|Engine: drafters and speculation (lookup, StairCut, MTP, DFlash2)
area|engine/cache|ea580c|Engine: KV cache, prefix cache, disk cache
area|server/api|116329|Server: the OpenAI API, tool calls, streaming, structured output
area|server/conductor|1a7f37|Server: the conductor (session monitor, deciders, policies)
area|server/proxy|2da44e|Server: the proxy in front of the engine and other backends
area|chat|0f766e|Chat service (chatd): agent loop, sandbox, files, tools
area|dash/setup|5a32a3|Dashboard: Setup Guide
area|dash/chat|6639ba|Dashboard: Chat
area|dash/usage|7b4bc9|Dashboard: Usage
area|dash/performance|8250df|Dashboard: Performance
area|dash/dev|8d5ee0|Dashboard: Dev (logs and request inspector)
area|dash/playground|9b6ce3|Dashboard: Playground
area|dash/conductor|a475f9|Dashboard: Conductor
area|dash/memory|b392f0|Dashboard: memory screens (conductor memory, disk cache)
area|dash/backends|c297fa|Dashboard: Backends
area|dash/system|d2a8ff|Dashboard: System
area|models/new|7d4e00|Support for a model that TandemLLM does not run yet
area|models/flint|9a6700|Model: Recursive Flint
area|models/qwen3.8|bf8700|Model: Qwen3.8 27B and Qwen3.8 Flash-Next
area|models/lfm2.5|d4a72c|Model: LFM2.5-24B-A2B
area|models/nemotron|e3b341|Model: Nemotron 3.5 Lightning, Bolt and Super VL
area|models/ornith|eac54f|Model: Ornith 1.5 122B
area|quant/exl3|99286e|Weights: EXL3 packs
area|quant/nvfp4|bf3989|Weights: NVFP4 packs
area|quant/bf16|db61a2|Weights: BF16
area|quant/community|f0b6d0|Weights: community packs (models/<model>/community/)
area|install|424a53|install.sh and the first start on a new machine
area|ops|57606a|Service scripts, profiles, watchdog, release tooling
area|docs/manual|6e7781|The manual (manual/, LaTeX)
area|bench|8c959f|Benchmarks and measurement tools
area|ci|afb8c1|GitHub workflows, issue forms, repository settings
priority|p0|b60205|Broken for everyone, or data loss: first
priority|p1|d73a4a|Next release
priority|p2|e99695|Normal
priority|p3|f9d0c4|When there is time
status|needs-triage|fbca04|New: no type, area, priority or milestone yet
status|needs-repro|fcd34d|Needs steps that reproduce it
status|needs-info|fde68a|Waits for an answer from the reporter
status|blocked|d4a106|Waits for another issue or an outside change
hardware|hw/dgx-spark|006b75|Seen on an NVIDIA DGX Spark (GB10)
hardware|hw/rtx-pro-6000|0598bc|Seen on an NVIDIA RTX PRO 6000
hardware|hw/other|5fb3c4|Seen on another GPU or machine
process|run-gpu|1f2328|A maintainer runs the GPU tests on a DGX Spark
process|skip-changelog|ededed|Left out of the generated release notes'

if [ "$TABLE" = 1 ]; then
    echo "| Group | Label | Meaning |"
    echo "|---|---|---|"
    printf '%s\n' "$LABELS" | while IFS='|' read -r group name colour desc; do
        echo "| $group | \`$name\` | $desc |"
    done
    exit 0
fi

command -v gh > /dev/null || { echo "gh is not installed" >&2; exit 1; }
existing="$(gh label list -R "$REPO" --limit 500 --json name -q '.[].name')"
has() { printf '%s\n' "$existing" | grep -qxF -- "$1"; }
run() { echo "+ $*"; if [ "$DRY" = 0 ]; then "$@" > /dev/null; fi; }

# renames keep the issues that carry the old name
for pair in enhancement:feature documentation:docs; do
    old="${pair%%:*}"; new="${pair#*:}"
    if has "$old" && ! has "$new"; then
        run gh label edit "$old" -R "$REPO" --name "$new"
        existing="$(printf '%s\n%s\n' "$existing" "$new")"
    fi
done

printf '%s\n' "$LABELS" | while IFS='|' read -r group name colour desc; do
    [ "${#desc}" -le 100 ] || { echo "description over 100 characters: $name" >&2; exit 1; }
    if has "$name"; then
        run gh label edit "$name" -R "$REPO" --color "$colour" --description "$desc"
    else
        run gh label create "$name" -R "$REPO" --color "$colour" --description "$desc"
    fi
done
