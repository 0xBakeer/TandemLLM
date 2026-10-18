# Benchmarks with tool-eval-bench

[tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench) scores tool calling over an OpenAI-compatible API: tool selection, arguments, multi-step chains, restraint, error recovery, injection resistance and structured output. It detects TandemLLM by its model owner or `Server` header. This page says how to run it against TandemLLM and which settings a run must name, so that another box gets the same score (OPS-189).

## Install

```bash
uv tool install git+https://github.com/SeraphimSerapis/tool-eval-bench.git
```

The numbers below are from bench 2.8.1.

## Two paths

A benchmark of the model runs the plain engine. The `X-Tandem: raw` header serves it: no turn class, no notes, no effort change, the session monitor off ([server.md](server.md#tandemllm-and-the-conductor)). Run the default path, without the header, only when the conductor itself is under test. The two runs then differ in the conductor alone, and the decision journal shows what it added to each turn.

```bash
# the model alone
tool-eval-bench run --short --base-url http://<host>:8000/v1 --model TandemLLM \
  --header X-Tandem=raw --seed 42 --timeout 600
# the conductor in front of it
tool-eval-bench run --short --base-url http://<host>:8000/v1 --model TandemLLM \
  --seed 42 --timeout 600
```

`--short` runs the 15 core scenarios. Leave it out for the full suite, 69 scenarios. `--hardmode` adds category P on top; a score with it is not comparable to one without. The bench's default temperature is 0, so the engine decodes greedily. Keep `--parallel` at 1, or at most the server's `--parallel` slots (`/health`, `parallel.max_running`): a scenario that queues behind another counts the wait as its latency. The bench's default request timeout, 120 s, cuts long reasoning and scores it as a failure, so give `--timeout 600` with thinking on.

`/v1/models` lists the relay backends beside `TandemLLM`, so pass `--model TandemLLM`. `/metrics` needs a bearer token and has none of the counters the bench's speculative-decoding report reads.

## Settings that change the score

| setting | where | what it changes |
|-|-|-|
| weight pack | `/health` `weights.name`, the setup guide's model profile, `ops/use-profile.sh` | the logits: answers on a near tie can differ between packs |
| thinking | the bench's `--no-think` | on by default; off was faster and scored lower in the runs below |
| default effort | `--reasoning-effort` (`REASONING_EFFORT` in `ops/serve.env`, `medium`; the setup guide's thinking effort) | the raw path's effort sentence and how long the model thinks; a client's own word wins, `--backend-kwargs '{"reasoning_effort": "high"}'` |
| profile | `X-Tandem-Profile`, else the profile mapped to the client kind, else `default_profile` | the conductor run only: classes, packages, effort, memory |
| memory | the profile's `memory` block | the conductor run only: a keyless client on a single-user box gets the owner's notes; the `general` profile has memory off |
| reasoning format | `reasoning_format` in the request, `--reasoning-format` | where the thinking goes; see below |
| response cache | `--response-cache` | a rerun of an identical request within the hour replays its greedy answer: timings change, answers do not |

Name the pack, the thinking setting, the effort, the profile and the path with every score. `/health` gives the version and the pack in `version`, for example `0.11.4+exl3-6.0`, and the bench records it.

A plain API client gets the thinking in `reasoning_content` and the answer alone in `content` (SRV-245). Up to 0.11.4 the raw path left it inline as `<think>...</think>` in `content`. Scenarios that check the exact answer then failed: valid JSON only (TC-22), a terse answer (TC-24). Since SRV-617 the raw path follows the same rule for an SDK, curl or an unnamed client, as the bench is. Against an older server, send the format yourself:

```bash
--backend-kwargs '{"reasoning_format": "reasoning_content"}'
```

## Results on a DGX Spark

Qwen3.8-27B in the EXL3 6.0 pack, TandemLLM 0.11.4, default effort `xhigh`, seed 42, temperature 0, timeout 600 s, one scenario at a time. The bench ran on another machine on the LAN. Each row is one run (n = 1).

Core 15:

| path | score | deployability | responsiveness | time | not passed |
|-|-|-|-|-|-|
| raw, thinking on | 100 | 89 | 63 | 102 s | none |
| raw, thinking on, effort `medium` | 97 | 82 | 48 | 117 s | TC-14 partial |
| raw, thinking off | 90 | 81 | 60 | 88 s | TC-03, TC-14 partial |
| conductor, `chat` profile with memory | 93 | 80 | 49 | 187 s | TC-03 |
| conductor, `general` profile | 93 | 79 | 47 | 161 s | TC-03 |

Full suite, 69 scenarios:

| path | score | deployability | responsiveness | time |
|-|-|-|-|-|
| raw, thinking on | 93 | 80 | 51 | 14 min |
| raw, thinking off | 88 | 80 | 60 | 10 min |
| conductor, `chat` profile | 89 | 68 | 20 | 30 min |

The first core row ran on main after SRV-617. The other rows ran on 0.11.4: the full raw row with thinking on sent `reasoning_format: reasoning_content` itself, and the `medium` row got the thinking inline in `content`.

TC-03 asks the model to tell Sarah the meeting moved. The conductor classes the turn as message writing, adds the write-message package, and the model drafts the email instead of looking up the contact and sending it (SRV-615). On the raw path TC-03 sits near a tie: one run of the core 15 missed it, the next passed. On every path the structured-output scenarios TC-65 to TC-69 end partial or failed, and TC-68, schema violation resistance, fails.

The first request of each scenario pays the prefill of the tool prompt, about 4,400 tokens; on the Spark the first token arrives after about 7 s, which the responsiveness score counts. Later turns reuse the prefix.
