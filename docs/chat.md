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
engine at once (`QSE_CHAT_MAX_RUNS`); the others wait as `queued`.

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

## Not yet (phase 2)

Tools, permissions, the sandbox, terminals, files, diffs, MCP servers, prompts and presets: their
routes answer `not_available` (501) or `sandbox_unavailable` (503) and their lists are empty, so the
tab shows them as unavailable. Compaction is phase 2 too.
