# The proxy: one dashboard in front of any OpenAI-compatible backend

The dashboard can sit in front of other engines and cloud APIs, not only this one. Clients send their OpenAI requests to one address; the proxy routes each request by its model id to the backend that serves it and copies the answer back byte for byte. On the way it records what a client could measure: tokens from `usage`, time to first token, decode tokens a second, cached and reasoning tokens when the provider reports them, and the cost at the model's price. One ledger and one dashboard cover every backend.

Backends in this release:

| kind | what it is | routes relayed |
|-|-|-|
| `tandemllm` | another TandemLLM server, by URL | chat, completions, embeddings, Responses |
| `generic` | vLLM, SGLang, llama.cpp, LM Studio, Ollama, any OpenAI-compatible server | chat, completions, embeddings, Responses |
| `openai` | OpenAI's API with your key | chat, completions, embeddings, Responses |
| `deepseek` | DeepSeek's API with your key | chat, completions |

The proxy only observes in this release. It never changes a request's meaning: it sets the model id the backend knows, and it asks a stream for its usage frame when the client did not (then it drops that frame before the client). The conductor's features (task class, effort, packages, the loop and tool checks) come to foreign backends in a later release. Their switches are already stored per model.

## Two ways to run it

### Inside the engine's server

`server/app.py --backends ~/.tandemllm/config.sqlite3` serves the engine's own model as before and relays every request whose model belongs to a backend. A relayed request runs on its own thread and never takes the engine's lock or a place in its queue, so a slow cloud answer cannot hold up the engine and the engine's queue cannot hold up a cloud answer. On the :8000 service, set `QSE_BACKENDS` in `ops/serve.env` and restart; empty means off, and the service is exactly what it was.

### On its own

Anyone without a GPU runs the same code. It needs Python 3.11 or newer, the standard library and `cryptography`, and never pulls torch. From a checkout:

    pip install .                       # or '.[host]': adds psutil and NVML for the System view
    tandem-dash                         # 127.0.0.1:8080, ~/.tandemllm/config.sqlite3
    open http://127.0.0.1:8080/dashboard/

`python -m proxy` in the checkout does the same without installing anything but `cryptography`. The page opens on Setup while no backend exists. The engine-only panels (caches, speculation, the conductor, memory) say that this server has no engine.

From a DeepSeek key to a row in the Usage view takes five commands:

    pip install '.[host]'
    printf 'TANDEM_BACKENDS=deepseek\nTANDEM_BACKEND_DEEPSEEK_KIND=deepseek\nTANDEM_BACKEND_DEEPSEEK_KEY=%s\n' "$DEEPSEEK_API_KEY" > dash.env && chmod 600 dash.env
    tandem-dash check --env-file dash.env
    tandem-dash serve --env-file dash.env &
    curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"deepseek-chat","messages":[{"role":"user","content":"Say hi"}]}'

Then open `http://127.0.0.1:8080/dashboard/#/usage`.

### In docker

`ops/Dockerfile.dash` builds an image of about 70 MB on `python:3.12-slim`, with the built page inside and no compiler or torch in it:

    docker build -f ops/Dockerfile.dash -t tandemllm-dash .
    cp ops/dash.env.example dash.env    # fill in; no quotes around values
    docker run -d --name tandem-dash -p 127.0.0.1:8080:8080 -v tandem-dash:/data \
        --env-file dash.env tandemllm-dash

The config database, the usage ledger and the key file (when `TANDEM_SECRETS_KEY` is not set) live in the `/data` volume. Inside the container the server binds 0.0.0.0, so what the outside reaches is what `-p` publishes: `127.0.0.1:8080` for this machine, a LAN address for a local network. Turn the login on (`TANDEM_DASHBOARD_LOGIN=on` and `TANDEM_ADMIN_TOKEN`) before you publish it to anything but loopback. The container runs as an unprivileged user and answers its health check on `/health`.

### The `tandem-dash` command

| command | what it does |
|-|-|
| `tandem-dash` or `tandem-dash serve` | the server; every flag of `python -m proxy` (`--env-file`, `--config-db`, `--host`, `--port`, `--usage-ledger`, `--dashboard-login`) |
| `tandem-dash check` | reads the environment and a copy of the config database through the schema and prints what would run: the backends, their key hints and locked fields, the rules, the page, the listen address. It writes nothing and exits 1 on an error that names the variable |
| `tandem-dash secrets init` | makes the master key file beside the config database (0600) when no master key answers; `secrets status` says which source answers |
| `tandem-dash backends list` | the backends, one line each, from a copy of the database |
| `tandem-dash hostcollect` | the remote host collector (below) |
| `tandem-dash version` | the installed version |

## Configuration: the GUI and the environment

Everything lives in one SQLite file, the config database: backends, their keys (encrypted), model lists, prices, routing rules, per-model switches, users. The Backends tab edits it: add a backend, test the connection (it shows the latency and the models before anything is saved), edit, delete. Every change is a row in its `audit` table with the time and who made it, and no key value.

The same objects can be declared as `TANDEM_*` variables, for a docker start or any headless box. The environment wins, field by field: at every start the variables are written over the database's rows (a backend is created when missing), and the GUI shows each field a variable sets as locked, with the variable's name on it. A save that changes such a field answers 422 and names the variable. When the variable goes away, the GUI can edit the field again, starting from its last value. A key from the environment stays in memory and is never written to the file.

    TANDEM_BACKENDS=deepseek,vllm1
    TANDEM_BACKEND_DEEPSEEK_KIND=deepseek
    TANDEM_BACKEND_DEEPSEEK_KEY=sk-...
    TANDEM_BACKEND_VLLM1_KIND=generic
    TANDEM_BACKEND_VLLM1_URL=http://gpu-box:8000/v1
    TANDEM_BACKEND_VLLM1_FALLBACK=deepseek
    TANDEM_ROUTE_1="deepseek-*=deepseek"
    TANDEM_PRICE_1="deepseek/deepseek-flash=0.30,0.006,1.2"
    TANDEM_PRICE_1_OFFPEAK="0.15,0.003,0.6"
    TANDEM_API_KEYS="laptop:tdm-...,ci:tdm-..."

`python -m proxy --env-file proxy.env` reads such a file; a variable already in the process environment keeps its value. The full list, with the fields each backend takes (`_KIND`, `_URL`, `_KEY`, `_LABEL`, `_MODELS`, `_MODELS_MODE`, `_FALLBACK`, `_MAX_INFLIGHT`, `_ENABLED`, `_INJECT_USAGE`, `_HARNESS`, `_TIMEOUT_CONNECT`, `_TIMEOUT_FIRST_BYTE`, `_TIMEOUT_IDLE`, `_HOST`, `_HOST_URL`, `_HOST_DCGM_URL`), is at the top of `proxy/config.py`, and `ops/dash.env.example` is a file to start from. One schema (`proxy/config.schema.json`) checks both ways in: a wrong variable stops the start with a message that names it.

## Keys at rest

A backend's key is stored as AES-256-GCM ciphertext. The master key comes from the first of these that answers: `TANDEM_SECRETS_KEY` (base64 of 32 bytes), the OS keychain (`security` on macOS, `secret-tool` on Linux, read only), or `secrets.key` beside the config database, made once at mode 0600. The Backends tab names the source. No API answer, log line, ledger row or audit row holds a key: the GUI gets `key_set`, the last 4 characters and 6 hex digits of the key's sha256, and the key field is write-only.

Encryption at rest protects a copied or backed-up database. It does not protect against a process running as the same user, which can read the key file as easily as the proxy can, and a backup of the whole directory carries the key file too. Keep the master key in the environment or the keychain when backups leave the machine. A lost master key means the keys are typed in again, nothing else.

To keep it in the macOS keychain instead of the file:

    security add-generic-password -s tandemllm -a secrets-key -w "$(cat ~/.tandemllm/secrets.key)"
    rm ~/.tandemllm/secrets.key

## The machine behind a backend

The System view shows the machine each backend runs on, down to the process behind the backend's port. For a vLLM on the same machine that is its API server and its engine core, with their RSS, CPU, threads, uptime and GPU memory. A backend's host setting says where to read it:

| setting | reads | per-process figures |
|-|-|-|
| `auto` (the default) | this machine, when the backend's URL is loopback or one of this machine's addresses; nothing for a cloud API | yes |
| `same` | this machine, whatever the URL says | yes |
| `collector` | `tandem-dash hostcollect` running on the backend's machine, at `host.url` | yes |
| `exporter` | a node_exporter page at `host.url` and a DCGM exporter page at `host.dcgm_url` | no |
| `off` | nothing | |

The server's own process is always listed first, as `engine` in the engine's server and `proxy` in the standalone one. The Backends form sets the host per backend; `TANDEM_BACKEND_<ID>_HOST`, `_HOST_URL` and `_HOST_DCGM_URL` set it from the environment and lock it.

The collector reads psutil and NVML when they are installed (`pip install 'tandemllm[host]'`), else `/proc` and `nvidia-smi`, and macOS without psutil gives the machine's total memory and load only. The process behind a port is found by its listening socket: psutil, then `/proc/net/tcp` and the processes' file descriptors, then `lsof` or `ss`. A process that belongs to another user stays unreadable without root, and the view says so. Command lines show the home as `~`, and the collector masks the value after any flag whose name holds `key`, `token`, `secret`, `passw` or `auth`.

On a GB10 (the DGX Spark) the CPU and the GPU share one memory pool. The driver reports no memory for the GPU itself, so the view shows "unified" there and `MemAvailable` is the figure to watch. The GPU memory of a process is still reported, and it comes out of the same pool as its RSS: the two overlap.

Nothing is sampled while nobody looks. When the System view asks, the collector reads memory, CPU, GPUs, disks and network at most every 5 s and scans the processes at most every 15 s. One sample with a scan took 4.4 ms of CPU on a Mac with psutil, under 0.1 % of one core at that cadence, and each host's footer shows what its own last sample cost.

On another machine, run the collector next to the backend and give both sides the same token:

    TANDEM_COLLECTOR_TOKEN=$(openssl rand -hex 24) tandem-dash hostcollect --host 0.0.0.0 --port 9108

It answers `GET /host` as JSON and `GET /metrics` as Prometheus text (`tandem_host_*`), both behind the bearer token; it refuses to bind beyond loopback without one, because its answer names processes and their command lines. `--watch vllm:8000` reports a port on every answer; the proxy asks for its backends' ports itself. A machine that already runs node_exporter and NVIDIA's DCGM exporter needs nothing new: set the backend's host to `exporter` with both page URLs. That path has no per-process figures, and the panel says so. A remote read runs on its own thread, so a slow or dead host never holds the page; its panel shows the last answer, its age, or the error.

In docker the collector sees the container, not the machine. Run the container with `--pid=host` (and `--gpus all` for NVML) for the machine's view, or run `tandem-dash hostcollect` on the machine itself.

## Models, names and routing

`GET /v1/models` lists each backend's real models under their own ids (`deepseek-flash`, `Qwen/Qwen3.5-4B`), with `owned_by` set to the backend's id and a `tandem` block (backend, kind, label, upstream id, context length, switches). In the engine's server, `TandemLLM` stays the engine's own. When two backends serve one id, both are listed as `<backend>/<model>` and the bare id is not; the prefixed form is always accepted. A model can get a public id of its own in the GUI.

A request goes to, in this order: an explicit `<backend>/` prefix, the first enabled routing rule whose glob matches the id, the one backend that serves it, else OpenAI's 404 `model_not_found` (which names the prefixed forms when two backends serve the id).

A backend or a single model may name fallbacks. On a connect error, a 429, 502 or 503, before any byte reached the client, the proxy tries the next backend at once, or the same one again after 1 s, then 2 s (or its `Retry-After`, up to 30 s). At most 2 tries again in all. After the first byte a failure is an `error` frame and `[DONE]`. The ledger row records the backend that answered, `fallback_from`, `retries` and the upstream status.

Each backend takes at most `max_inflight` requests at once (default 64). Past that a request goes to a fallback with room, else it gets 503 with `Retry-After: 1`. Timeouts per backend (local: connect 5 s, first byte 600 s, idle gap 120 s; cloud: 10, 120, 60 s) end a request with `finish_reason: timeout` and the kind in `error_type`. A client that leaves closes the upstream connection within a quarter second, so the backend stops generating.

## What the dashboard shows for a relayed request

Measured or null, never estimated:

| figure | from |
|-|-|
| prompt, completion tokens | the backend's `usage`; null when it sends none |
| cached prompt tokens | `prompt_tokens_details.cached_tokens` (OpenAI, vLLM, SGLang), `prompt_cache_hit_tokens` (DeepSeek), `input_tokens_details.cached_tokens` (Responses) |
| reasoning tokens | `completion_tokens_details.reasoning_tokens`, SGLang's `reasoning_tokens` |
| TTFT | arrival at the proxy to the first frame with a token; the network and the backend's queue are inside it |
| decode tok/s | (completion tokens - 1) / (last token frame - first token frame); null without `usage`, because a frame is not a token |
| cost | the model's price when the row is written |
| proxy overhead | arrival to the upstream send, plus the first frame's delay through the proxy |

Queue, prefill time, blocks, draft acceptance and the cache source come off an engine's loop and are null for a relayed row; the views say "not reported by this backend". The Usage view filters by backend (`engine` is the engine's own rows) and by user, and shows the cost of the priced models.

## Prices and cost

A price row per model holds USD per million tokens for input, cached input, cache write and output. The cost of a request is `(prompt - cached) x input + cached x cached input + cache write x its price + completion x output`, written into the row when the request ends, so a later price change never rewrites history. A model without a price has a null cost. DeepSeek charges half outside its peak hours (01:00-04:00 and 06:00-10:00 UTC on weekdays); a price row can carry an `offpeak` block, and the DeepSeek adapter picks by the request's UTC hour. Chinese public holidays count as peak here, so their cost reads high.

The proxy never fetches a price list. `proxy/prices.seed.json` holds the list prices of OpenAI and DeepSeek as read from their pages on 2026-10-04; the model rows offer them and apply them only on a click. Check the providers' pages before you bill anyone from these numbers.

## Users and their keys

Under Backends, Users, the admin makes a key per person or tool. The key is shown once and stored as its sha256. From the first key on, the OpenAI routes need one (`Authorization: Bearer tdm-...`) for every model, the engine's own included, from any machine other than this one: a request from loopback without proxy headers passes, so the engine's own helpers keep working. The key's label goes into the ledger's `user` column. `TANDEM_API_KEYS` adds keys from the environment.

## Who may change the configuration

Reading `/v1/config` follows the dashboard's rule. Every write needs the admin token or a dashboard session, or comes from the machine itself. With no admin token set, only the machine itself can change the configuration. The default bind is 127.0.0.1; the page is meant for a local network. Never put it on the internet without the login on and TLS in front.

## Terms, in short

Using your own key through your own proxy for your own clients is ordinary API use. Putting one key in front of several people makes you the provider toward them: the provider's terms hold the key holder responsible for what those people send, and most forbid reselling access without an agreement. Per-user keys keep each person's usage visible in the ledger. This is not legal advice.

## License

The proxy, the collector and the package are part of TandemLLM and carry its licenses: AGPL-3.0-only, or the commercial license in `COMMERCIAL-LICENSE.md`. Under the AGPL, anyone who runs a changed copy of the dashboard as a service for others must offer those users the changed source. The one runtime dependency, `cryptography`, is Apache-2.0 or BSD at the user's choice; the `host` extra adds psutil (BSD-3-Clause) and nvidia-ml-py (BSD-3-Clause). `NOTICE` lists them. None of their code is copied into this repository.

## Not in this release

The conductor's features on foreign backends (the per-model switches are stored, not acted on), vLLM's and SGLang's `/metrics` tiles, an Anthropic adapter, and a release on PyPI (the wheel builds; publishing it is a separate step). The cloud adapters are tested against the providers' documented shapes, not against a paid call. The per-process GPU memory on a discrete GPU is tested against a recorded NVML answer, not yet on such a machine.
