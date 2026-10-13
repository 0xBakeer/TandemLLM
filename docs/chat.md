# The Chat tab and chatd

The dashboard's Chat tab (`#/chat`) is a chat with history whose runs live on the server: close the
browser in the middle of an answer, open the chat on a phone, and it shows the answer at the token
the engine is writing now and keeps streaming. The wire contract is
`dashboard/src/chat/CONTRACT.md`; this page is how it runs on a box.

## What runs where

```
browser  --https-->  engine server :8000  --/v1/chat-app/* (auth checked, byte forward)-->  chatd 127.0.0.1:8016
                                                                                           |  ~/.tandemllm/chat/chat.sqlite3
                                                                                           |  POST /v1/chat/completions --> :8000
```

- **chatd** (`python -m chatd`, the `chatd/` package): a standard-library Python process. It holds
  the stream to the engine, so a run does not end when a browser closes; it writes every change of
  a conversation to an event log with a sequence number, and a device attaches with a snapshot and
  then tails the log from the snapshot's `seq` (SSE, `Last-Event-ID` on a reconnect, `event: reset`
  for a device more than 2,000 events behind).
- **The forward** (`server/chat_forward.py`): the engine server (and the standalone proxy,
  `tandem-dash`) passes `/v1/chat-app/*` to chatd after the dashboard's auth check. chatd answers
  only requests that carry the forward key.
- **The model** is reached as any client reaches it, through `POST /v1/chat/completions` with
  `x-session-id: chat-<conversation id>`, so the engine's park, spill and live rows follow the chat.
  What a model can do comes from `/v1/models` (the `tandem` card, `models/<id>/card.json`), never
  from a model name.

chatd never imports the engine and an engine restart does not touch it: a run that loses the
engine retries with backoff for 10 minutes, then ends `interrupted` with its text kept, and
Continue resumes it. Stopping chatd ends its open runs `interrupted` the same way.

## Starting and stopping

```
bash ops/make-secrets.sh            # once: adds QSE_CHAT_FORWARD_KEY to secrets.env
bash ops/chatd.sh start             # 127.0.0.1:8016, log in logs/chatd.log
bash ops/chatd.sh status
bash ops/chatd.sh stop
```

`ops/chatd.service` is the systemd user unit for when linger is enabled. At most 2 runs talk to the
engine at once (`QSE_CHAT_MAX_RUNS`); the others wait as `queued`. A run that waits for a
permission prompt gives its slot back until the answer comes, and takes one again for its next call.

**An engine without the forward.** An engine server from an older build (or another engine
altogether) has no `/v1/chat-app` route. `bash ops/chatd.sh start --front 0.0.0.0:4180` also serves
the built dashboard on :4180 from chatd's own process, forwarding `/v1/chat-app/*` to chatd and
everything else to the engine (`chatd/front.py`). The front has no admin password, so it belongs on
the trusted network only.

## Auth

The dashboard's one switch (the admin password, `QSE_ADMIN_TOKEN`). With none set, every chat route
answers anyone on the trusted network and the tab shows no sign-in. With one set, every chat route
needs the dashboard session cookie or the admin bearer token (or trusted-local); the probe
(`GET /v1/chat-app/capabilities`) still answers and says `signed_in: false`, and the tab shows the
setup guide's sign-in card. A write with `Sec-Fetch-Site: cross-site` is refused.

## Data

- `~/.tandemllm/chat/chat.sqlite3` (WAL, mode 600): conversations, the message tree, runs, the event
  log, the queue, attachments' metadata. Blobs under `blobs/<sha[:2]>/<sha>`.
- A `VACUUM INTO` copy once a day in `backups/`, 7 kept. A deleted chat stays in the trash for 7
  days; text deltas in the event log are dropped after 7 days (the messages keep their text).
- chatd's log has ids, counts and times, never message text.

## The sandbox (phase 2)

Each chat's tools run in its own container, `tandem-chat-<id>` from the image `tandem-sandbox:1`
(build it once: `bash ops/sandbox/build.sh`). chatd speaks the Docker Engine API over a unix socket,
`QSE_SANDBOX_SOCKET` (default `/var/run/docker.sock`); Podman serves the same API, and
`ops/sandbox/install.sh` (root once) sets up rootless Podman under the user `tandem-sbx` with its
nftables rules. Every container is locked down whatever the runtime: uid 1000, no capabilities, a
read-only root, `--network none`, 4 GiB, 4 CPUs, 512 pids, no GPU. Its internet goes through
chatd's per-chat proxy, which refuses the LAN and this box (`sandbox_network` per chat: on or off).

The engine comes first. A sandbox starts only when MemAvailable minus its limit stays at 19 GiB or
more; the guard stops the least recently used idle sandbox under 15 GiB and kills them all under
12 GiB; an idle sandbox stops after 30 minutes. The files stay. Knobs:

- `QSE_SANDBOX_MIN_START_GB` (0): MemAvailable needed to start one (30 while other work runs memory
  fits on the box)
- `QSE_SANDBOX_MAX_RUNNING` (4) and `QSE_SANDBOX_TOTAL_GB` (16): how many run at once, and their
  limits' sum
- `QSE_SANDBOX_IMAGE` (`tandem-sandbox:1`)

Export them before `ops/chatd.sh start`. Without a runtime, starting a sandbox answers 503
`sandbox_unavailable`, bash, render_page and the terminals are not offered, and the file tools work
on the chat's workspace folder.

The unit tests (`tests/test_chatd*.py`, `tests/test_chat_forward.py`) never reach a container
runtime: `tests/chatd_util.py` points `QSE_SANDBOX_SOCKET` at a path that does not exist, and a test
that needs a sandbox installs a fake one. On the Spark, Docker and the sandbox image are there, and
before this the tests found them: `bash` was offered, the tools' order changed and the capabilities
said the sandbox was installed. Only `tests/test_sandbox_box.py` (with `QSE_SANDBOX_BOX=1`) uses
the real runtime.

MCP servers: stdio servers run inside the chat's sandbox, HTTP servers are called from chatd
through the same egress rules; their secrets are sealed in the store and never come back in an
answer.

## Permission modes

Each chat has a permission mode, and the chat settings hold the default that a new chat gets. A test run on a phone (2026-10-13) asked six times for bash commands such as `grep` and `cp`, which cost 20 s of tapping, and every one of them ran inside the chat's own sandbox. The three modes:

- Ask: every risky call asks first, as before 0.8.2.
- Auto, the default: a call that stays inside the chat's sandbox runs without asking. That covers bash, reading and writing files in `/workspace`, and `render_page`. A call that leaves the sandbox still asks: `web_fetch` to a domain this chat has not fetched before, an MCP tool that its server does not mark `readOnlyHint`, and an imported Open WebUI tool whose file imports a network or process module (`requests`, `urllib`, `socket`, `subprocess` and the like). `rm` or `rmdir` aimed at `/`, `/workspace` or a home directory asks too.
- YOLO: nothing asks, and everything runs.

Some limits hold in every mode. A deny rule refuses the call, and so does a tool set to "never". A tool set to "ask" asks. Plan mode offers only the read tools. The sandbox stays locked down: no LAN, no host, and the internet only through chatd's proxy.

A call that runs without a prompt still leaves a record. Its tool card and `tool.started` carry `approved: "auto"` or `"yolo"` (`"user"` after a prompt, `"rule"` for a rule or an always policy), and the chat's event log gets a `tool.auto_approved` event with the call's one-line summary. The run's mode is on `run.started`, on the assistant message (`permission_mode`) and, after a change during the run, on `run.mode`. A change applies from the next call. The routes:

```text
GET  /v1/chat-app/permissions/mode                       {default, modes}
PUT  /v1/chat-app/permissions/mode                       {default: ask|auto|yolo}
PATCH /v1/chat-app/defaults                              {permission_mode} -> the capabilities' defaults
GET  /v1/chat-app/conversations/{id}/permission-mode     {mode, effective, default}
PUT  /v1/chat-app/conversations/{id}/permission-mode     {mode: ask|auto|yolo}
```

`PATCH /conversations/{id}/settings` takes `permission_mode` too. A chat from before 0.8.2 has none and runs in the default mode.

## Chat tab (beta): known limitations

The Chat tab is in beta in this release, and a Beta pill beside it says so. We know of these limits and have not fixed them yet:

- The agent loop does not yet work the way Claude Code does (epic VIS-E10). Its prompts are not kept prefix-stable from step to step (SRV-375). Old tool results and old thinking stay in the context until a compaction (SRV-365). A compaction writes a general summary, not the task's state (SRV-373). Nothing notices when the model repeats the same failing approach with different calls (SRV-367). render_page passes on Chromium's console noise (SRV-369). Long tool output has no "see more" paging (SRV-371). Searches run in the main context, with no sub-agents (SRV-381).
- The engine reads a short Flash-Next chat again on its next turn when the answer ended inside a verified block (ENG-235).
- A request that arrives during a prewarm still waits 0.5 to 1 s (SRV-349).
- When memory reaches its ceiling, the engine drops the parked chat without writing it to the NVMe, so its next turn starts cold.
- Images only: there is no video input (VIS-97).
- The 5.05 Flash-Next pack does not load yet.
- Parallel decoding does not raise throughput over one request at a time: three requests together took 27.18 s against 27.03 s one after another, though the later ones start sooner.
- The phone layout's keyboard behaviour is not verified in Safari on an iPhone.

## Not yet

Importing Open WebUI tools and uploading a skill zip answer `not_available` (501). Uploads of
attachments are phase 1's.
