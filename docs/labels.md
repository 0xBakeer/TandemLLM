# Labels

Every issue and pull request gets one type label and one or more area labels. Open issues also get a priority. A label from the other groups is added when it applies. `tools/gh_labels.sh` creates and updates these labels on GitHub, and this table is its `--table` output: change the script, then run `bash tools/gh_labels.sh --table` and paste the result here.

GitHub's own `duplicate`, `invalid`, `wontfix`, `good first issue` and `help wanted` stay as they are.

| Group | Label | Meaning |
|---|---|---|
| type | `bug` | Something is broken: a wrong answer, a crash, a hang, a wrong number |
| type | `feature` | A new capability, or a change to how something works |
| type | `perf` | Speed or memory: tok/s, time to first token, prefill, GPU memory |
| type | `docs` | README, docs/ or the manual |
| type | `refactor` | Code that changes shape and keeps its behaviour and output |
| type | `test` | Tests, test tools or the test suite |
| type | `security` | A security fix or hardening. Report a vulnerability in private: SECURITY.md |
| type | `question` | A question about using TandemLLM |
| area | `engine/core` | Engine: model code, loaders, the decode and verify loop |
| area | `engine/kernels` | Engine: GPU kernels, matrix products, attention |
| area | `engine/drafting` | Engine: drafters and speculation (lookup, StairCut, MTP, DFlash2) |
| area | `engine/cache` | Engine: KV cache, prefix cache, disk cache |
| area | `server/api` | Server: the OpenAI API, tool calls, streaming, structured output |
| area | `server/conductor` | Server: the conductor (session monitor, deciders, policies) |
| area | `server/proxy` | Server: the proxy in front of the engine and other backends |
| area | `chat` | Chat service (chatd): agent loop, sandbox, files, tools |
| area | `dash/setup` | Dashboard: Setup Guide |
| area | `dash/chat` | Dashboard: Chat |
| area | `dash/usage` | Dashboard: Usage |
| area | `dash/performance` | Dashboard: Performance |
| area | `dash/dev` | Dashboard: Dev (logs and request inspector) |
| area | `dash/playground` | Dashboard: Playground |
| area | `dash/conductor` | Dashboard: Conductor |
| area | `dash/memory` | Dashboard: memory screens (conductor memory, disk cache) |
| area | `dash/backends` | Dashboard: Backends |
| area | `dash/system` | Dashboard: System |
| area | `models/new` | Support for a model that TandemLLM does not run yet |
| area | `models/flint` | Model: Recursive Flint |
| area | `models/qwen3.8` | Model: Qwen3.8 27B and Qwen3.8 Flash-Next |
| area | `models/lfm2.5` | Model: LFM2.5-24B-A2B |
| area | `models/nemotron` | Model: Nemotron 3.5 Lightning, Bolt and Super VL |
| area | `models/ornith` | Model: Ornith 1.5 122B |
| area | `quant/exl3` | Weights: EXL3 packs |
| area | `quant/nvfp4` | Weights: NVFP4 packs |
| area | `quant/bf16` | Weights: BF16 |
| area | `quant/community` | Weights: community packs (models/<model>/community/) |
| area | `install` | install.sh and the first start on a new machine |
| area | `ops` | Service scripts, profiles, watchdog, release tooling |
| area | `docs/manual` | The manual (manual/, LaTeX) |
| area | `bench` | Benchmarks and measurement tools |
| area | `ci` | GitHub workflows, issue forms, repository settings |
| priority | `p0` | Broken for everyone, or data loss: first |
| priority | `p1` | Next release |
| priority | `p2` | Normal |
| priority | `p3` | When there is time |
| status | `needs-triage` | New: no type, area, priority or milestone yet |
| status | `needs-repro` | Needs steps that reproduce it |
| status | `needs-info` | Waits for an answer from the reporter |
| status | `blocked` | Waits for another issue or an outside change |
| hardware | `hw/dgx-spark` | Seen on an NVIDIA DGX Spark (GB10) |
| hardware | `hw/rtx-pro-6000` | Seen on an NVIDIA RTX PRO 6000 |
| hardware | `hw/other` | Seen on another GPU or machine |
| process | `run-gpu` | A maintainer runs the GPU tests on a DGX Spark |
| process | `skip-changelog` | Left out of the generated release notes |
