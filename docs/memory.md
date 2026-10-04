# Memory notes

The model keeps notes about the user and the work, as files, and reads them back in later turns and sessions. One note is one markdown file with a frontmatter: one fact, a title of at most 120 characters, a body of at most 400 words. The files are the truth; the dashboard's Memory screen (Notes tab) reads and edits them.

## Where the files are

```text
~/.qwen38-spark-engine/memory/notes/<user>/
  INDEX.md                               generated each night: one line per live note
  user/<kind>-<slug>--<id>.md            about the person
  projects/<name>-<hash8>/<kind>-...md   about one working directory
  sessions/<session>/<kind>-...md        true for one conversation only
  revisions/<id>/<n>.md                  every earlier text of a note
  archive/<yyyy-mm>/...                  archived notes, deleted after 30 days
  usage.json                             how often each note went into a request, by day
```

`<user>` is `owner` on the engine. A request the proxy relayed carries the proxy user's name (the key's label in process, `X-Tandem-User` from a loopback address or one listed in `QWEN38_MEMORY_PROXIES`); a relayed request without one reads and writes nothing. Files are `0600`, and the store refuses a root inside a git work tree, as the transcripts do.

The eight kinds: `rule` (how the user wants things done), `ruling` (a decision), `gotcha` (what went wrong once), `runbook`, `program_state` (with `verify_by`; stale after that date and never injected), `reference`, `registry`, `archive`.

## How the model writes

It writes a block in its answer, usually after its text:

```text
<tandem_memory op="add" scope="project" kind="gotcha">
title: the build needs node 20
tags: build

npm run build fails on node 18 with "Unexpected token 'with'".
</tandem_memory>
```

`op` is `add`, `update` (with `id:`) or `archive` (with `id:` and `reason:`). The server holds the block back from the client the way it holds a `<tool_call>` (`server/memblock.py`), writes the note, and the client sees one state line, `› remembered: the build needs node 20`, when the profile shows state lines. A block that is not taken goes out as the text it was: one that never closed, one over 2,000 characters, the fifth of an answer, or one the store refused. A block inside the reasoning is left alone.

The rules are code, not prompt text (`server/conductor/notes.py`):

- A write over a cap is refused, never cut: title 120 characters, body 400 words, 20 notes a session, 500 live notes a user.
- Every title and body passes the secret-pattern refusal of the transcripts; a body that is mostly a secret is refused.
- A `rule` or `ruling` about the user needs the user's own message in the turn. A tool turn or a script cannot write one.
- The model writes `provisional` notes. Ratify, pin and delete are the dashboard's. A model revision of a ratified note makes it provisional again.
- Before an `add` lands, the same scope is searched: the same normalised title, or one of BM25's top 5 whose word overlap is at or above `dedupe_threshold` (0.6), turns the add into a revision of that note.

The model's scripts have the same write as a route, under the session's address only: `POST {tandem_url}/v1/memory`, `GET {tandem_url}/v1/memory/search?q=`, `GET {tandem_url}/v1/memory/<id>` (`server/memory_api.py`). Without a known session token the routes answer 403.

## How the model reads

A profile's `memory` switch: `off` (the default; the request is the release's), `shadow` (the card goes in and blocks are written, nothing is read back) or `on`. With `on`, three tail notes, each replayed at its place on later turns so each costs its prefill once:

- the memory card, about 150 tokens: the block's shape and when to write one;
- the index, on the first turn of a session and again after a compaction: the pinned ratified rules, then one line per note the request may see (`settings.memory.notes.index_tokens`, 1,500);
- the matching notes, on a new user message (never a tool turn): BM25 over the message, at most 3 notes that share at least 2 of its words, within 600 tokens.

Laya's question "does this message ask to remember, forget or recall" attaches the long `memory` skill (`settings.memory.notes.intent`, shadow by default: journaled only).

## What is journaled

Kind `memory_notes`: each write (`write`, `api_write`: op, id, scope, kind, why it was refused, redaction counts), each injection (`injected`: the note ids), the dashboard's edits (`human`), the nightly job. Never a title or a body.
