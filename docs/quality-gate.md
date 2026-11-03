# Quality gate

A pull request that changes the engine, a model, a kernel or the server carries one report: the base (main) and the head (the branch) measured side by side on the contributor's GPU. One command writes it, and the `gate` check refuses a pull request without a current, passing report. For an engine or kernel change, a maintainer's run on the reference box (a DGX Spark) decides the merge.

## The command

    python -m tools.gate map                  # the level your diff needs
    python -m tools.gate run                  # base origin/main, head HEAD, level from the diff

Run it with the engine's Python, from a checkout with `origin/main` fetched, on a box with the weights of the profile's reference pack and a free GPU. The gate never starts a server below port 8100 and never touches one it did not start; on a GPU that shares memory with the host it refuses to start while another engine holds that memory. It writes `results/gate/<head>-<time>/gate-report.md` and `gate-report.json`. Paste the Markdown file into the pull request's description. Editing the description runs the check again; a report posted in a comment counts from the next run.

Options: `--suite quick|quick+long|full` (default: what the diff needs), `--base`, `--head`, `--pack ID`, `--hw auto|gb10|ada|generic`, `--python` (the engine's interpreter, for the hardware fingerprint), `--port` (the first port tried), `--pairs N` (servers a side), `--site FILE` (see below), `--lock FILE` (a lock file held for the GPU part).

How it runs: the gate copies both commits with `git archive`. The head's GPU tests run first, with no server up. Then each side gets a fresh server on a free port, with its own config copy and state directory, the conductor raw, and the suffix store, the response cache, the usage ledger and the disk tier off. The suites run against each server, and each server stops before the next starts. The head without speculation and, at quick+long, both sides with the wide prefill chunk get servers of their own. The measuring code is the checkout's own, so one instrument measures both sides.

A box whose layout differs from a plain checkout (its own `ops/serve.env`, a `run/setup.env`, files linked beside the tree) names it in a site file: `serve_env`, `setup_env`, `local_env`, and `links` / `copies` maps of tree path to box path. Without one, the gate reads an install's `run/local.env`.

## What a diff needs

`tools/gate/map.yaml` maps each changed path to a level. Every file takes the highest level of the rules it matches, and the pull request takes the highest of its files. A path no rule names takes `quick`.

A new GPU test in `tests/gpu/` needs a line under `kernels` in the same file, with the paths it covers; `tests/test_gate_map.py` fails until it has one.

| Level | Paths | Suites |
|-|-|-|
| none | docs, the manual, the dashboard, `chatd/`, the proxy, packages, CPU tests, community packs | no report |
| quick | the engine, models, kernels, profiles, the server's request path | all but stability; long runs one case |
| quick+long | memory, caches, the scheduler, slots, start and stop scripts | quick, and the wide prefill chunk case |
| full | VERSION, the release scripts | everything, the stability run included, two servers a side |

The map's `kernels` table names the `tests/gpu/` files that the changed paths need.

## The suites

| Suite | What it holds |
|-|-|
| `lossless` | Greedy and seeded outputs of a fixed set of 8 prompts are the same text on head and base. Then the head with speculation against the head without: the same text, or a difference at a near tie (a top-2 gap of at most 0.125 nats, [exactness.md](exactness.md)). |
| `decode` | One request with a short prompt and with a 1k prompt; N requests at once (8 on a GB10, fewer on smaller cards). Fixed prompts and seeds, warm-up rounds dropped, the median of 3 rounds. |
| `prefill` | Cold solo prompts of 2k and 8k tokens (32k and 64k at full, where they fit), and a burst of N 2k prompts at once: prefill tok/s and the median first token. |
| `long` | The longest prompt the profile allows (64k on a GB10, 128k at quick+long) with a needle: it completes, the answer names the needle, and the reserved memory stays under the declared peak. |
| `api` | A chat stream and a plain answer, a forced tool call, a cancel mid-stream, a client that leaves while it waits for a lane, and the `/health` fields. |
| `kernels` | The mapped `tests/gpu/` files, each in its own process. |
| `stability` | 30 minutes of mixed traffic (`tools/soak.py`): no failed request, reserve under the peak, decode drift of the last third against the first. |

Every request goes to `/v1/completions` with `X-Tandem: raw`, with the prompts rendered from the model's chat template as `tools/parallel_card.py` renders them ([measurement.md](measurement.md#the-weight-cards-speed-rows)). Before a concurrent level, the gate reads the server's live slot limit (`parallel.max_running`). During every round it samples `/health` and requires N requests decoding at once.

## The report

The header names:

- the hardware: the GPU and its memory and compute capability, with the driver, CUDA and torch versions;
- the pack, with a fingerprint of its weights;
- the base and head commits, with a hash of each tree;
- every limit knob each server ran with, defaults included: lanes, the live slot limit, queue, window, prefill chunk, caches, memory ceiling and peak, and the server's environment and flags.

Each table row reads metric | base | head | delta | noise | verdict.

| Verdict | Meaning |
|-|-|
| PASS | nothing resolved worse, or the texts are equal |
| WORSE | resolved worse by at least the profile's floor (3 %) |
| CHANGED_TEXT | the head writes other text than the base |
| FAIL | a check fails on the head and passes on the base |
| INVALID | the concurrency proof failed, a request hit a cache, a side wrote two texts for one request, or a side has fewer than 2 runs. Such a row carries no number. |
| SKIPPED | not run, with the reason; a check that also fails on the base reads SKIPPED |

The gate calls a speed difference resolved only when it exceeds the run-to-run noise and the run ranges of the two sides do not overlap (`tools/row3.py`'s rule, [measurement.md](measurement.md#noise-and-what-a-row-can-decide)). A resolved slowdown under the floor passes, and the row says so.

A WORSE or CHANGED_TEXT row needs a written reason before the check passes:

    python -m tools.gate explain results/gate/.../gate-report.json decode.c8 "why, in a sentence"

## The check

The `gate` workflow runs on every pull request with read-only permissions and no secrets. It uses the gate code of the base commit, so a pull request cannot change the rules that check it. It finds the last marked report in the description or the comments and fails when:

- the diff needs a report and there is none, or it does not parse or fails the schema;
- the report measured another head, unless every file changed since then needs no report;
- its base is not a commit of the target branch, or its level is below what the diff needs;
- a needed suite did not run, measured nothing, or a mapped GPU test is missing;
- a row is FAIL or INVALID, or a WORSE or CHANGED_TEXT row has no reason;
- the report comes from a dry run (`--head-patch`) or from another major version of the tool.

After the re-run on the reference box, a maintainer adds `gate-verified`.

No fitting GPU? Say so in the pull request. A maintainer adds `gate-maintainer`, the check passes, and the maintainer runs the gate on the reference box and posts the report.

## Hardware profiles

`tools/gate/hw/` holds one adapter per GPU. Each adapter answers the gate's questions about that GPU, from the lanes and prompt sizes to the memory limits and the reference pack. Detection reads `nvidia-smi` and the engine's torch. A GPU no adapter claims gets `generic`, with small sizes and no reference pack (pass `--pack`). `python -m tools.gate hw` prints the profile for this box.

| Profile | GPU | Lanes | Long prompt | Pack |
|-|-|-|-|-|
| `gb10` | NVIDIA GB10 (sm_121, unified memory) | 8 | 64k, 128k at quick+long | `flashnext-3.05` |
| `ada` | sm_89 cards | 4 from 40 GiB, else 2 | 32k from 40 GiB, else 16k | `lfm25-exl3-4.0` from 24 GiB, else `lfm25-exl3-2.5` |
| `generic` | anything else | 4 from 40 GiB, else 2 | 16k from 40 GiB, else 8k | none |

The Ada sizes are a starting point; no Ada card has run the gate yet.
