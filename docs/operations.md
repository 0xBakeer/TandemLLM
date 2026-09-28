# Operations

A DGX Spark has 128 GB of memory that the CPU and the GPU share, and a board that runs out of it does not recover by itself: the GPU driver locks, the box stops answering, and only a power cycle brings it back. Most rules on this page exist to keep that from happening on a box nobody can reach.

## One engine per board

Two engines loading side by side will wedge the board. The served engine reserves about 69 GB, and a second one of the same size does not fit. Every script here checks for a running engine first, and a loading engine counts even though it answers nothing yet. The server binds its port only after the weights are in, so for the minutes of a load neither `/health` nor the port shows it. `ops/engines.sh` finds engines by their process instead.

## Memory rules

Keep at least 15 GiB of `MemAvailable` at the worst case of anything you run. Before any change that can raise memory (a cache budget, the context length, more parallel sequences, a new resident copy), write down its worst-case bytes first. During a trial, run a guard that kills your test process when `MemAvailable` drops under 10 GiB.

What the served stack costs:

| item | size |
|-|-|
| NVFP4 weights and the FP8 head, on the device | about 16.4 GiB |
| KV buffer, allocated for the whole context at load | 106.5 KiB a token: 64 KiB for the target, 20 KiB for each of the two drafters (27.9 GB at 262,144 tokens) |
| the whole stack at load, 262,144-token context | 53.45 GB (measured with `tools/mem_audit.py`) |
| state store | 8 GiB budget, at most a quarter of it per snapshot |
| resident prefix | 4 GiB of anchors and a 2 GiB guest stash |

Page cache counts as available memory but is not always given back in time. After a load, the engine never reads the weight files again, and `ops/start.sh` drops their page cache (`tools/drop_page_cache.py`, `posix_fadvise` with `DONTNEED`, no root needed). Without that drop, an 8,192-row prefill took free memory to 1 GB and the driver logged `NV_ERR_NO_MEMORY`. With it, the same request left 53 GB free.

The FP8 profile needs about 27.5 GiB of weights on the device instead of 16.4. The BF16 checkpoint needs about 54 GB: keep `--max-len` at 65,536 or less with it unless the byte math says otherwise.

## Running the service

`ops/serve.env` is the served configuration. It holds about 60 settings, from paths and the port to every `QWEN38_*` switch, each with the reason for its value in a comment. The scripts read it.

| script | what it does |
|-|-|
| `ops/make-secrets.sh` | creates `secrets.env` with the admin and metrics tokens (mode 600), once; `--rotate` replaces both |
| `ops/start.sh` | starts the engine on the port from `serve.env` and waits for `/health`; refuses if an engine is alive or the port is held |
| `ops/stop.sh [seconds]` | sends SIGTERM, waits for the drain (60 s by default), and kills only after that |
| `ops/watchdog.sh` | one health check, for cron every minute; restarts after 3 failures in a row |
| `ops/hold.sh <minutes> <command>` | runs a command with the board to itself, then restores the service |
| `ops/gate.sh <label>` | the release gate ([measurement.md](measurement.md)) |

`ops/start.sh` is safe to run at any time. If the engine is healthy it says so and exits. It refuses, with a non-zero exit, while any engine is alive or while something else holds the port without answering `/health`.

The watchdog restarts only what it is sure about. A healthy server is left alone, and so is a draining one, because killing a graceful stop halfway is worse than a slow stop. A loading engine is left alone for up to 900 s. Only a server that fails three checks in a row is restarted, so one slow check during a long verify does not bounce a working server.

At boot, two cron lines bring the service up:

    @reboot sleep 90 && $HOME/<checkout>/ops/start.sh >> $HOME/<checkout>/logs/boot.log 2>&1
    * * * * * $HOME/<checkout>/ops/watchdog.sh

`ops/qwen38-engine.service` is a systemd user unit for the same job. A user unit only survives logout with `loginctl enable-linger`, which needs root. Install one mechanism, never both: two supervisors for one port start the service twice.

## Profiles

A profile is a `serve.env` file. `ops/serve-fp8.env` and `ops/serve-balanced.env` source `ops/serve.env` and change two lines: `NV` (the NVFP4 files) and `HEAD` (the FP8 head).

- To gate a profile: `ops/gate.sh <label> --profile ops/serve-fp8.env`.
- To serve a profile on the service port: copy its two lines into `ops/serve.env` and restart the service through `ops/hold.sh`.
- For the engine tools, export the same values: `QWEN38_NVFP4=` and `QWEN38_FP8_HEAD=` (empty) for plain FP8.
- The FP8 profile also reads its own launch table (`QWEN38_FP8_TILES=ops/fp8-tiles.json`) and its router prices (`--price-table ops/prices.json:fp8-plain`).
- `--fp8-head build` builds the FP8 head at load, without the file.

## Holding the board

Measurements need the board to themselves, and so does anything that loads a second copy of the model. `ops/hold.sh` makes the safe sequence the easy one:

    flock ~/.qwen38-box.flock ops/hold.sh 30 python tools/verify_spec.py ...

It arms a pause file so the watchdog stands back, and keeps it fresh for the whole hold. Then it stops the service and runs the command. Afterwards it restarts the service and waits for `/health`. On every exit path, including a killed command, it removes the pause. The lock file makes two holds wait for each other. A pause file older than 2,400 s is ignored, so a crashed hold cannot leave the service unsupervised for long.

Taking the service down stops everyone who uses it. Hold the board when nobody needs the service, and keep holds short.

## Settings

`engine/settings.py` is the one registry of every `QWEN38_*` switch the engine reads (110 of them), with its default and the module that reads it. `tests/test_settings.py` fails if any module reads a switch another way. To see what is in effect:

    python -c "from engine.settings import SETTINGS; import json; print(json.dumps(SETTINGS.describe(), indent=1))"

Server behaviour (ports, limits, caches, reasoning) is set by command-line flags. `python server/app.py --help` lists them, and `ops/start.sh` builds them from `serve.env`.

## Deploying a new version

1. Copy the current serving directory to a dated backup (`<checkout>.bak-YYYYMMDD-HHMM`).
2. Copy the new files in, keeping the local `serve.env` values that are site-specific (paths, port).
3. Restart through `ops/hold.sh`, so the watchdog does not race the restart.
4. Check that `/health` says `ok` and that `VERSION` is the new one, then send a short request.

Run the gate and the client smoke set before step 1 ([measurement.md](measurement.md)). A deploy that fails step 4 goes back to the backup.

## Monitoring

`/health` returns the status, the queue counters (running, waiting, served, refused, errors, timeouts, abandoned), the cache report and the allocator's memory figures. On this board `nvidia-smi` reports no used memory, so `torch.cuda.memory_allocated` and `memory_reserved` are the numbers to watch. Allocated memory that climbs across requests is a leak. Reserved memory that climbs while allocated memory stays flat is fragmentation.

`/metrics` serves Prometheus metrics, listed in [server/METRICS.md](../server/METRICS.md). `ops/monitoring/` has a scrape config, alert rules with their tests, and a Grafana dashboard generated by `tools/grafana_dashboard.py`.

## When something is wrong

- An answer stops in the middle. Find the request's log line. `finish=length` hit the token limit, `timeout` the wall clock, `error` an exception (its traceback is in the log), and `abandoned` means the client hung up.
- An agent turn waits minutes before the first token. Check `cached_tokens` for that request in the usage ledger. On a growing conversation it should be close to the previous turn's prompt. If it is not, read the `resident` block of `/v1/cache/stats`.
- The server answers 503 or 429 while it looks idle. Read `inflight` in `/health`. A 503 means a full queue, and a 429 a request that waited longer than the queue timeout. A high `waiting` count with nothing running is a bug.
- The port answers but nothing generates. The watchdog handles three failed checks. By hand: `ops/stop.sh 30 && ops/start.sh`.
