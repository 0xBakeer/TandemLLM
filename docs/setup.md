# The setup guide

The dashboard's `#/setup` page walks through an install in eight steps, from picking a model and its weights to the tools beside it, the parallel requests and the memory they take. The last two steps show what will change and then install it. A small assistant called Wren answers questions on the side and can change the page when you ask. The server half lives in `server/setup/` and `server/setup_api.py` and answers under `/v1/setup/*`. None of it is on the serving path: a chat request never reaches it, and the engine imports it only when a `/v1/setup` route is called.

## Setup mode

A new box has no configured engine yet. `install.sh --setup` installs the code and the Python environment, downloads only Wren's model (Qwen3.5-2B, 4.3 GB), writes the config with no weights, and starts setup mode:

```bash
bash install.sh --setup                     # on the box; binds 127.0.0.1 and prints the ssh tunnel
bash install.sh --setup --host 0.0.0.0      # the setup page on the network (no password yet!)
ssh -L 8000:localhost:8000 <the box>        # from your computer, then open
                                            # http://localhost:8000/dashboard/#/setup
```

On a checkout, the same mode starts with `ops/start.sh --setup` (127.0.0.1 unless `--host` names another address). Setup mode runs `server/app.py --setup`: no model of Qwen's, no caches, no state store, no drafters, no conductor and no ledger. It answers the dashboard, `/v1/setup/*` and `/health` (`"mode": "setup"`), loads Wren at the start, and gives every other route a 503 with the code `setup_mode`. The watchdog sees its `/health` answer 200 and leaves it alone.

The installer writes the marker `run/setup-pending` (with the host to bind). While it exists, a plain `ops/start.sh` (the reboot's, the watchdog's) starts setup mode again. The install job removes it before it starts the configured engine, and puts it back if that engine does not come up.

Until an admin password is set, anyone who reaches the port can run the setup, apply it and reset the box. The review step says so. `ops/make-secrets.sh --admin` sets one; the setup page's "generate" option does the same on a box that has no `secrets.env` yet.

## The model registry

Each model the guide offers is a folder under `models/` with a `model.yaml` ([models/README.md](../models/README.md) lists the fields). `models/qwen3.8-27b/model.yaml` is the one model today. Every number in it says where it comes from:

- The weights and the drafters per profile are the `[memory] load stages` lines of the 0.6.2 runs of 2026-10-10, GiB allocated, one run per EXL3 size. NVFP4's are from the 0.5.0 load of 2026-10-09.
- The tools' memory is the dashboard's live memory split of the same evening (the reference box on `exl3-6.0` with 4 slots, the narrator, Laya and the shadow reader loaded). The registry's parts add up to that split's 59.34 GiB in use within 0.1 GiB.
- The measured table is the 0.6.2 run itself; `models/qwen3.8-27b/measurements/0.6.2-summary.json` is its summary file, and a test holds the two equal. The NVFP4 row is from build 0.4.8 and is marked not comparable.

Memory is in GiB (2^30 bytes) and disk in GB (10^9 bytes) everywhere in the registry and the API. `python -m server.setup.registry --check` checks a file.

### The KV bytes a token

One token of context window costs 106,752 bytes, which is 104.25 KiB or 106.75 kB: 64 KiB of the model's attention KV (16 attention layers, 4 KV heads of 256, K and V in BF16), 20 KiB for each of the two drafters' context KV (5 layers, 8 KV heads of 128) and 256 B of rotary table. At 262,144 tokens that is 28.0 GB, or 26.1 GiB. The docs said "106.5 KiB" until 0.6.2: that was the kB figure with the KiB unit. With the KV pool on (`QSE_KV_POOL=on`) a slot reserves its window as addresses and commits pages only for the rows its conversation writes.

### The CUDA graphs and the side slots

The 6.63 GiB of CUDA graphs belong to slot 0. A side slot captures its own verify graphs, about 7.5 GiB each, only with `QSE_SIDE_GRAPHS=on`; the guide always writes it off, as the reference box runs, and a side slot then costs 1.19 GiB beyond its KV (the load stage `side_slots` took 3.58 GiB for three on 2026-10-10).

## The memory estimate

`server/setup/estimate.py` holds the one formula; `models/estimate-cases.json` pins it with 44 cases, and the dashboard's TypeScript copy must give the same answers.

- The parts: the OS and other programs (the device's `reserved_gib`), the weights and the drafters of the profile, the CUDA graphs, the engine's warm-up workspace, the allocator's reserve, 1.19 GiB per side slot, and every tool that is on.
- The conversations (KV): with the pool, each slot at 32,768 tokens or its window if smaller (`typical_fill_tokens`); without it, every window in full, since it is allocated at load.
- `budget_gb` is the device's memory less the floor (`QSE_PARALLEL_FLOOR_GB`, 15). `fits` is the total at or under the budget, compared to the cent. `fits_worst` asks the same with every slot at its full window; when it is false the engine makes a new request wait for memory instead of running out.
- `context_tokens_fit` is what the room left after the fixed parts holds, in tokens, over all slots.

`exl3-4.5` with the recommended tools at 4 slots of 262,144: 67.35 GiB of a 106.63 GiB budget, 13.03 of it conversations at 32,768 tokens a slot. Every slot full would take 104.25 GiB of KV, so `fits_worst` is false.

## Wren

Wren is Qwen3.5-2B (Qwen3.5-0.8B when the 2B is not on disk) on the engine's own forward, in the plainest setup the engine has: one sequence, a window of 8,192 rows, thinking off, greedy, no caches. Its persona and rules are `server/setup/wren_system.md`.

- Each turn it reads the page's state (fields named like a secret arrive as `<set>`), the box's free memory and disk, and the four best sections of a BM25 search over `docs/*.md`, the manual's chapters, `models/*/model.yaml` and the packages that describe TandemLLM.
- It can ask the page to `set_field`, `toggle`, `select_profile`, `go_to_step`, `next`, `previous` and `explain`. The browser runs every call; the server checks each one against the step schema first. An invalid call never reaches the browser: Wren sees why and tries again, at most twice.
- It cannot finish, apply, reset, load or unload anything.
- On a serving engine it loads only when asked (`POST /v1/setup/assistant/load`), through the residency planner. If the narrator is loaded with the same model, Wren borrows the narrator's copy and costs no memory. It is the first thing given back when the KV pool needs memory, and it unloads itself after 5 minutes without a poll or a chat.
- It generates in slices of 150 ms while no request runs or waits, so a request that arrives waits at most one slice.
- The chat's text is never logged.

On the reference box (2026-10-11, `tests/gpu/setup_wren_gpu.py`) Wren's own copy loads in 22 s and holds 3.62 GiB on the device; MemAvailable drops 5.0 to 5.2 GiB at the first load, of which 1.1 to 1.5 GiB is the process's CUDA context, which stays. Later loads and unloads return within the box's own noise (0.2 to 0.4 GiB), with the process's RSS and the allocator flat. Borrowing the narrator costs 0.03 to 0.06 GiB, and the narrator's next line is the same to the token. A turn takes 1 to 3 s, the first words after 0.4 to 0.7 s. On ten plain requests and three questions, the 2B made the right call or answered 10 times of 13; Qwen3.5-4B (`QSE_WREN_MODEL=qwen3.5-4b`) 12 of 13, at about 2.5 times the time a turn. The 2B sometimes mixes the numbers of two profiles in an answer, so the page's own numbers stay the reference.

## The install job

`POST /v1/setup/plan` returns the review step's facts: what is wrong, what changes against `ops/serve.env`, the commands that will run and the estimate. `POST /v1/setup/apply` checks the selection again and starts a job; `GET /v1/setup/jobs/{id}` streams its steps and log lines (SSE; `Last-Event-ID` resumes, `?format=json` gives the job).

The job runs a fixed list of steps from `server/setup/apply.py`, and nothing a client sends becomes a command:

| step | what it runs |
|-|-|
| check | the selection, the memory, the disk, the files the profile needs, the HF token, the box lock |
| unload_assistant | Wren is unloaded before the job starts |
| fetch | `bash ops/use-profile.sh <profile> --fetch-only`, skipped when the weights are on disk |
| write_config | `run/setup.env`, the old one kept as `run/setup.env.bak-<time>`; with a port change, `bash ops/stop.sh 120` first; with "generate", `bash ops/make-secrets.sh --admin` |
| switch | `bash ops/use-profile.sh <profile>`: stop, start, smoke, and its own rollback of the profile |
| smoke | `/health` names the profile and one short chat answers |
| rollback | on a failed switch or smoke: the old `run/setup.env` back, then the previous profile (in setup mode, setup mode again) |

`ops/serve.env` is never edited. It reads `run/setup.env` after the weight profile, so the overlay never sets the weights. A box whose `serve.env` predates the setup guide must get those last lines first; the plan says so.

The job runs as a process of its own, because the switch replaces the server that started it. Its state is in `run/setup-jobs/<id>/`, and the status page (`ops/engine_status.py`) answers `GET /v1/setup/jobs/{id}` while the engine restarts. Apply is refused while requests run (409 `busy`), while another job runs (409 `job_running`), when the selection does not fit (409 `over_budget`), and when the box lacks something (412 `missing_assets`, `missing_token`, `disk_full`, `no_overlay_hook`).

On a fresh box today every profile fails the check: the K3 drafters, the K4 draft head and the FP8 base checkpoint have no download yet (OPS-95), and the job says which files are missing before it changes anything.

## Reset

`GET /v1/setup/reset` lists the levels with what each removes and how big it is; `POST /v1/setup/reset {level, confirm, dry_run}` removes one.

| level | what goes | confirm | restart |
|-|-|-|-|
| 1 | the page's progress and Wren's chat (the browser's own storage) | the page's | no |
| 2 | `run/setup.env` and its backups; the recommended profile starts | `true` | yes (a job, as apply) |
| 3 | the disk caches of every profile, the parked chats, the RAM caches, the lookup store (a running one is emptied by the engine, its folder stays), the shadow reader's disk | `delete caches` | no |
| 4 | the weights of every profile but the active one, and their HF cache copies; never what an engine on any port reads (its `--nvfp4`, `--fp8-head`, `QWEN38_EXL3`, `QWEN38_DRAFT_*`), nor `~/nvfp4-bf16` while `~/tandem-staircut` is there; nothing at all while the active profile is unknown (409 `active_unknown`) | `delete weights` | no |
| 5 | the usage ledger file and its `-wal`, `-shm` and `-journal` files (not its folder), the conductor's settings and journal, the memory notes | `delete history` | yes |

Levels 2 to 4 never touch the active profile's weights, the K3 drafters, the K4 draft head, the base checkpoint, the small models, the HF token, `secrets.env`, the history or the conductor's data. A cache the running engine holds is cleared by the engine's own function; the shadow reader's disk is kept while it runs. A symbolic link goes as a link and its target stays. Every reset is refused while requests run or a job runs, and every call writes one line to `run/setup-audit.log`.

## The routes

All of them are admin routes: open while no admin password is set, and the admin token, a dashboard session or a tool on the box itself once one is. Every POST must also be same-origin JSON: `Content-Type: application/json` (else 415 `json_only`), no `Sec-Fetch-Site: cross-site`, and an `Origin`, when sent, that names the server's `Host` or `X-Forwarded-Host` (else 403 `cross_site` / `cross_origin`). A page on another site cannot run apply or reset through a visitor's browser, even on an open box or through a tunnel's loopback. The shapes are `dashboard/src/api/types.ts`, contract 1.0.

```text
GET  /v1/setup/models                  a tile per model
GET  /v1/setup/models/{id}             model.yaml as JSON, with wizard_fields
GET  /v1/setup/models/{id}/steps       the step schema and Wren's tools
GET  /v1/setup/device                  memory, GPU, disk, profiles on disk, Wren's state
POST /v1/setup/estimate                {selection} -> the estimate
POST /v1/setup/plan                    {selection} -> the review step's plan
POST /v1/setup/apply                   {selection, dry_run?} -> 202 {job}
GET  /v1/setup/jobs/{id}               SSE of the job (?format=json: the job)
GET  /v1/setup/reset                   the reset levels
POST /v1/setup/reset                   {level, confirm, dry_run?}
GET  /v1/setup/assistant/status        Wren's state; a poll keeps it loaded
POST /v1/setup/assistant/load          load Wren
POST /v1/setup/assistant/unload        unload Wren
POST /v1/setup/assistant/chat          {messages, page} -> SSE (/v1/setup/chat is the same)
```
