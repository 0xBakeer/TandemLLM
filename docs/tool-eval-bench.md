# tool-eval-bench results

We ran [tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench) against TandemLLM on one DGX Spark on 2026-10-19: the raw path at five thinking settings, the conductor in four profiles, and the bench's hard mode on both paths. Thinking adds 3 to 5 points. Packages double the time a tool client waits and add no points, so the `tools` profile, which tool clients get by default, adds none. [benchmarks.md](benchmarks.md) says how to run the bench and which settings change a score.

## Setup

- Model: Qwen3.8-27B in the EXL3 6.0 pack, on the 0.12.0 code. `/health` still said `0.11.4+exl3-6.0`, because the runs came before the version bump.
- Server: default effort `xhigh` (`--reasoning-effort xhigh`), two slots, response cache on.
- Bench: 2.8.1.dev13, commit [f2e368b](https://github.com/SeraphimSerapis/tool-eval-bench/commit/f2e368b), on another machine on the LAN.
- Every run: `--seed 42 --temperature 0 --timeout 600`, `--parallel` 1, one run at a time on the box, each row one run (n = 1).
- The server restarted before the batch, which empties the response cache. No answer in a row below came from the cache. The first raw `xhigh` run did replay 190 answers of the `high` run, which sends the same prompt, so the server restarted and that row ran again.

## Full suite, 69 scenarios

| path | thinking | packages | score | deployability | responsiveness | time | not passed |
|-|-|-|-|-|-|-|-|
| raw | off | | 92 | 83 | 63 | 8.0 min | TC-14, TC-48, TC-50, TC-51, TC-60 partial; TC-03, TC-62, TC-68 failed |
| raw | `low` | | 96 | 81 | 47 | 12.7 min | TC-14, TC-50, TC-52 partial; TC-68 failed |
| raw | `medium` | | 96 | 82 | 49 | 13.4 min | TC-14, TC-49, TC-52 partial; TC-68 failed |
| raw | `high` | | 95 | 83 | 54 | 11.9 min | TC-23, TC-58, TC-62 partial; TC-51, TC-68 failed |
| raw | `xhigh` | | 97 | 84 | 55 | 12.4 min | TC-23, TC-58 partial; TC-68 failed |
| conductor, `tools` | the server's (`xhigh`) | off | 95 | 82 | 52 | 15.0 min | TC-23, TC-49, TC-67 partial; TC-62, TC-68 failed |
| conductor, `bench-nopk` | by turn class | off | 96 | 83 | 53 | 13.5 min | TC-23, TC-49, TC-62 partial; TC-68 failed |
| conductor, `bench-pk` | by turn class | on | 94 | 75 | 31 | 26.0 min | TC-50, TC-63 partial; TC-03, TC-31, TC-68 failed |
| conductor, `chat` | by turn class | on | 91 | 71 | 23 | 28.0 min | TC-40, TC-50, TC-62 partial; TC-03, TC-31, TC-43, TC-63, TC-68 failed |

## Hard mode, 28 scenarios

`--hardmode-only` runs category P alone. Its scores are not comparable to the full suite's.

| path | score | deployability | responsiveness | time | not passed |
|-|-|-|-|-|-|
| raw, `xhigh` | 84 | 71 | 41 | 11.0 min | TC-85, TC-90, TC-91, TC-94, TC-97 partial; TC-88, TC-92 failed |
| conductor, `tools` | 91 | 77 | 43 | 10.2 min | TC-85, TC-90, TC-94 partial; TC-92 failed |

## How each row ran

The base command, then each row's flags:

```bash
tool-eval-bench run --base-url http://<host>:8000/v1 --model TandemLLM --backend tandemllm \
  --seed 42 --temperature 0 --timeout 600
```

| row | flags |
|-|-|
| raw, off | `--header X-Tandem=raw --no-think` |
| raw, an effort | `--header X-Tandem=raw --backend-kwargs '{"reasoning_effort": "low"}'` (`medium`, `high`, `xhigh`) |
| conductor, `tools` | none: an unnamed client that offers tools gets `tools` |
| conductor, another profile | `--header X-Tandem-Profile=<profile>` |
| hard mode | add `--hardmode-only` |

`low` caps thinking at 1,024 tokens and `medium` at 8,192. `high` and `xhigh` send the template's `xhigh` effort sentence, and `high` caps thinking at 16,384 tokens.

`chat` is the profile Open WebUI gets. It classes each turn, sets the effort and adds packages and memory notes. `bench-pk` and `bench-nopk` are test profiles that do not ship. Each is `tools` with the turn class and the effort on and the conductor's prompt, and `bench-pk` adds packages. Run this on the box to create them; a write from another machine needs the admin token:

```bash
for p in on off; do
  name=bench-pk; [ $p = off ] && name=bench-nopk
  curl -s http://127.0.0.1:8000/v1/conductor/profiles/tools \
    | jq --arg p $p '{note: "bench, packages \($p)", doc: (.doc
        | .decisions += {classify: "on", effort: "on", packages: $p}
        | .prompt = "conductor")}' \
    | curl -s -X PUT -H 'Content-Type: application/json' --data @- \
        http://127.0.0.1:8000/v1/conductor/profiles/$name
done
```

## What the rows show

Run-to-run noise is about 2 points. The raw `high` and `xhigh` rows send the same prompt and scored 95 and 97. The `tools` profile scored 96 on the same build the night before, and 95 here. Read rows within 2 points of each other as the same.

- Thinking on scores 95 to 97 against 92 with thinking off. With thinking off the suite runs in 8.0 min against about 12 to 13.
- Packages cost time. `bench-pk` and `bench-nopk` differ in packages alone: 94 against 96, inside the noise, and 26.0 min against 13.5. TC-03, TC-31 and TC-63 fail only in the two rows with packages. On TC-03 the model drafts the message instead of sending it (SRV-615).
- On the full suite the conductor without packages scores the same as raw.
- In hard mode the conductor scored 91 against 84 for raw. It passed TC-88, TC-91 and TC-97, where raw failed TC-88 and got partial on the other two. That is one run a side, and we have not tested which part of the conductor makes the difference.
- TC-68, schema violation resistance, fails in every row: the scenario wants a value that the prompt leaves out, and the model declines to invent it. The graders of TC-23 and TC-49 read "doesn't provide historical data" and "no email will be sent" as failures (OPS-193).
