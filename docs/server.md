# Server and API

`server/app.py` speaks the OpenAI HTTP API, so any OpenAI client can use the engine by changing its base URL. It is built on the Python standard library (`ThreadingHTTPServer`), with no web framework. This page covers the routes and the request fields, then tool calls and structured outputs, then how the service behaves under load.

## Routes

| route | who may call it | what |
|-|-|-|
| `POST /v1/chat/completions` | anyone | chat, streamed or not |
| `POST /v1/completions` | anyone | plain text completion |
| `GET /v1/models` | anyone | the served model name, and `TandemLLM` when the conductor is on |
| `GET /health` | anyone gets the status; the full body needs a token | status, queue counters, cache report, memory |
| `GET /metrics` | the metrics or admin token | Prometheus metrics ([server/METRICS.md](../server/METRICS.md)) |
| `GET /v1/cache/stats` | the admin token | the caches' counters |
| `POST /v1/cache/clear` | the admin token, as a bearer header | empty the caches |
| `GET /dashboard/` | anyone | the dashboard's static files, which hold no data |
| `/v1/dashboard/*` | anyone by default; the admin token or a dashboard session with `--dashboard-login on` | the dashboard's data ([dashboard.md](dashboard.md)) |
| `GET /v1/conductor/*` | as `/v1/dashboard/*` | the conductor's config, decisions, sessions and reports (below) |
| `PUT`/`POST /v1/conductor/*` | the admin token or a dashboard session, or the box itself | new config versions, restores, labels |

The admin and metrics tokens live in a `secrets.env` file that `ops/make-secrets.sh` writes (mode 600, never in the repository). Without an admin token the admin routes answer 404. Requests from the box itself (loopback, no proxy headers) need no token for the read-only admin routes. The watchdog and the measurement tools depend on that.

The dashboard login is off by default (`--dashboard-login`, from `QSE_DASHBOARD_LOGIN`): the dashboard's read API, `/v1/dashboard/*`, answers anyone who reaches the port, even without a `secrets.env`, and the page opens without a sign-in screen. With `QSE_DASHBOARD_LOGIN=on` it needs the admin token or a session, and without an admin token it answers 404. The login changes nothing else: `/metrics`, the full `/health` and both cache routes keep the rules in the table, and `POST /v1/cache/clear` takes only the admin token as a bearer header. [dashboard.md](dashboard.md#access) says what the dashboard shows and how to turn the login on.

## What the API supports

`server/compat.py` gives every OpenAI request field one of three answers, and `tests/test_compat.py` pins the table.

- Honoured: `messages`, `model`, `stream`, `stream_options`, `max_tokens`, `max_completion_tokens`, `stop`, `temperature`, `top_p`, `seed`, `presence_penalty`, `frequency_penalty`, `logit_bias`, `logprobs`, `top_logprobs` (up to 20), `n` (up to 16, not streamed), `tools`, `tool_choice`, `parallel_tool_calls`, `reasoning_effort`, `response_format`, `modalities` (text output only). Messages may carry `image_url` parts (see Images below).
- Accepted and without effect by their own definition: `user`, `metadata`, `store`, `service_tier`, `prompt_cache_key`, `safety_identifier`, `prediction`.
- Refused with a 400 that names the field: `audio`, `functions`, `function_call` (use `tools`), `web_search_options`, `verbosity`, and for completions `suffix` and `echo`.

Fields that are not OpenAI's pass through without a 400, because clients send extensions meant for other servers. The engine reads its own: `top_k`, `min_p`, `repetition_penalty`, `no_repeat_ngram_size`, `max_reasoning_tokens`, `reasoning_format`, `draft_temperature` and `chat_template_kwargs`.

## Reasoning

The model thinks inside `<think>` tags. `--reasoning-format` (or `reasoning_format` in the request) picks how the thinking reaches the client:

| value | content | `reasoning_content` |
|-|-|-|
| `tags` (default) | thinking inside `<think>...</think>`, then the answer | empty |
| `reasoning_content` | the answer only | the thinking, as OpenAI-style deltas |
| `both` | as `tags` | as `reasoning_content` |

![Thinking in opencode with the live token count and rate](media/thinking.gif)

*Recorded against the served engine (published NVFP4 weights, StairCut on); sped-up parts are marked in the clip.*

The chat template opens the block in the prompt, so the model only ever writes `</think>`. Our server re-emits `<think>` as the first delta so that clients which fold on a matched pair see one. `reasoning_effort` (low, medium, xhigh) goes into the chat template, and the served default is `medium`. At `xhigh`, one request spent all 8,192 of its tokens thinking and never answered.

## Tool calls

The request's `tools` go into the chat template, and the model answers a call in its own XML form:

    <tool_call>
    <function=read>
    <parameter=offset>
    150
    </parameter>
    </function>
    </tool_call>

![A tool call in opencode next to the dashboard's live panel](media/tool_call.gif)

*Recorded against the served engine (published NVFP4 weights, StairCut on); sped-up parts are marked in the clip.*

`server/toolcall.py` turns that into OpenAI `tool_calls` with `finish_reason: "tool_calls"`, on both the streamed and the JSON path. It reads the strict form, a lenient form and a JSON call object, and it reads only the answer, never the thinking. On a stream, a call goes out as argument deltas while the model writes it: the id and the name first, then each string value in pieces. A whole-file write therefore shows progress instead of minutes of silence.

Argument values carry their schema's types. The model writes every value as text, so `offset` arrives as the characters `150`. Our parser looks up the request's own schema, and a parameter whose schema does not allow a string is returned as JSON: `150`, `true`, a list. A typed value is held back on the stream until it is complete, then sent whole. On an 8-task agent run, a strict client that validates arguments against the schema refused 58 of 80 calls before this change, and 0 of 26 after.

`tool_choice` works as OpenAI defines it. `none` renders the prompt without tools. `required` and a named function become a constraint on the answer, the same token masks structured outputs use, so the model cannot answer without the call. `parallel_tool_calls: false` keeps one call.

## Structured outputs

`response_format` with `json_object` or `json_schema`, and `structured_outputs` with a `regex`, a `choice` list or a JSON schema, constrain the answer to a regular language. `engine/grammar.py` compiles each form to a finite automaton over bytes, because the tokenizer is byte-level and a token can end halfway through a character. A token is allowed in a state when walking its bytes never reaches the dead state. The engine computes that for the whole vocabulary at once per state and caches it, so a state costs a few milliseconds once.

Each mask applies to every verified row of a draft tree, not only to one row per step. Constrained output under speculation is therefore the text plain constrained decoding writes, with the same near-tie exception as any greedy text ([exactness.md](exactness.md)). JSON schemas with recursion (`$ref` cycles) are refused, and `json_object` allows a fixed nesting depth.

## Streaming

A streamed response sends one chunk per token, carries `usage` on the last chunk when the client asks for it (`stream_options.include_usage`, and by default with `--usage-default on`), then a chunk with `finish_reason` and `data: [DONE]`. A verify round accepts several tokens at once, so they reach the socket together. Half the gaps between tokens are therefore microseconds, and the other half a whole round.

`finish_reason` is one of `stop` (end of text or a stop string), `length` (`max_tokens`), `tool_calls`, `timeout` (the request's wall-clock cap), or `error`. On `error` the partial text has already been sent and an `error` object sits beside the finish reason. A client that hung up is logged as `abandoned`, not as an error.

`server/stream.py` turns tokens into characters. A character can span tokens, because the tokenizer works on bytes, and a stream that sent every decoded prefix would send a replacement character (U+FFFD) that never goes away. Our detokenizer holds back a trailing partial character until the next token completes it. Every character it sends is final.

## A long prefill, and a client that leaves

A long prompt prefills for minutes: 208.3 s at 128k tokens on the published NVFP4 set ([caches.md](caches.md)). After every prefill chunk the handler checks its socket. If the client has gone, the prefill stops there, the request ends `abandoned`, and the engine is free at once. Rows already prefilled stay in the resident prefix ([caches.md](caches.md)), so a retry continues from there. A request whose client left while it waited in the queue is dropped before it runs.

A streamed prefill that has run for 5 s (`--prefill-heartbeat-s`) also sends an SSE comment after each chunk, at most once a second:

    : prefill 12288/95131

Clients ignore comment lines. Proxies see a live stream and do not time it out.

## Admission and limits

TandemLLM holds one sequence, so requests wait for it in a bounded queue.

| limit | served value | what happens |
|-|-|-|
| `--max-queue` | 8 | the next request gets a 503 with `Retry-After: 5` |
| `--queue-timeout` | 240 s | a request that waited that long gets a 429 with `Retry-After: 10` |
| `--request-timeout` | 3,600 s | a generation that runs that long ends with `finish_reason: "timeout"` |
| `--max-len` | 262,144 tokens | a prompt that does not fit gets a 400; `max_tokens` is clamped to what fits |
| `--default-max-tokens` | 32,768 | used when the request sends no `max_tokens` |

At load the server allocates the KV buffer for the whole context, so the context length is a memory decision taken before the first request ([operations.md](operations.md)).

## Images

A user, assistant or tool message may carry `{"type": "image_url", "image_url": {"url": ...}}` parts, where the URL is `https://` or `data:image/...;base64,`. The server fetches and decodes them before the request queues and runs the checkpoint's own image processor (`preprocessor_config.json`: 16-pixel patches, 2 × 2 merge, 65,536 to 16,777,216 pixels). The chat template writes one placeholder per image and the server expands it to one prompt row per merged patch: a 640 × 480 image is 300 rows, and those rows count as prompt tokens in `usage`. The vision tower (27 blocks, 0.92 GB, BF16 as the checkpoint stores it) runs inside the prefill, when the first chunk that holds the image runs; the live view reads "Encoding image 1 of 2" while it does.

A bad image is a 400 that names the part, for example `messages[1].content[0].image_url`: a scheme other than https or data, a download that fails, broken base64, a body in none of the five formats it reads (PNG, JPEG, WEBP, GIF, BMP), more than `--image-max-mb` (20), a header that declares more than `--image-max-decode-pixels` (64 million), more than `--max-images` (16) in a request, images that together take more than `--max-image-rows` (65,536) prompt rows, an image in a system message, a video part, and any image when the server runs with `--vision off`. `--image-act-gb` (6) is the admission check: an image whose encode would need more memory than that, by the tower's byte math, is refused before it queues; the largest image the processor admits needs about 3.4 GiB. A waiting request keeps its images' patches on the host in the tower's dtype (BF16, 12 KiB a prompt row), so the row cap also bounds what a full queue holds.

An https URL is a request the server makes for the client, so it is fenced. The host must resolve to public addresses only: loopback, private, link-local, shared (100.64/10), multicast and reserved ranges are refused, including IPv4 addresses inside IPv6. The connection goes to the address that was checked, with the certificate checked against the host name, so a second DNS answer cannot move it. A redirect is followed only to another https URL that passes the same check, at most 3 times. A URL with credentials in it is refused. `--image-fetch-timeout` (15 s) covers the whole download, redirects included. `--image-https-private on` lets URLs reach private addresses, for a server that should read images from its own network; `--image-https off` accepts data: URLs only.

Text requests do not touch any of this: the engine's image state is empty for them and every forward is the text one, bit for bit.

## TandemLLM and the conductor

With `--conductor-config DIR` (off by default; `ops/start.sh` passes serve.env's `QSE_CONDUCTOR_CONFIG` to the :8000 service only), `/v1/models` lists two models. `qwen38-spark-engine`, and any other name, is the raw engine and runs exactly as without the conductor. `TandemLLM` is the same weights with a conductor in front ([server/conductor/](../server/conductor/)). Every response carries `X-Tandem` (`raw`, `shadow` or `on`) and `X-Tandem-Profile`.

A `TandemLLM` request resolves to a profile: the `X-Tandem-Profile` header (an unknown name is a 400 with `param: X-Tandem-Profile`), else the profile mapped to the client (opencode gets `coding`, Open WebUI `chat`), else the default. `X-Tandem: raw` or `shadow` lowers the mode and nothing raises it. `raw` and `bench` are locked.

Before the request queues, the router decides which kind of task the turn is. A turn that only carries tool results keeps the session's class, so in an agent session only a new user message asks a model: arbiter's `choice` question first (`POST /v1/systemone` on :8010, used at 0.80), then the engine itself on the raw name with thinking off and `structured_outputs.choice` (used at 0.55), else the session's class. No word list decides anything. The class sets the thinking budget, thinking off (the generation prompt's `<think>\n` becomes the template's closed empty block at the tail; the template renders `reasoning_effort` as the first system tokens and `enable_thinking: false` drops that sentence, so the router never re-renders), and a note with the effort sentence, a skill or package and rules. The note goes into the last turn before its `<|im_end|>`, and later turns replay it at the same token position, so the resident prefix keeps matching. It is never streamed, never counted in `usage`, never stored in the suffix store, and a request with a note skips the response cache. A client's `max_reasoning_tokens`, `thinking_budget` or `chat_template_kwargs.enable_thinking` always wins.

The config (profiles, the router's classes and thresholds, roles, packages, settings) lives in versioned JSON files under the directory, outside any repository. `PUT` makes a new version with a required note, `POST .../restore` makes an old one current again, and the next request uses it. A file that fails to parse or validate keeps the last good version and shows in `GET /v1/conductor/config`. Every decision, with the exact text of every note, goes to `decisions.sqlite3` there for 30 days; `GET /v1/conductor/sessions/{key}/export?format=md` lists what the model saw that the client did not. The contract is [contract/conductor-v1](contract/conductor-v1/README.md), and the dashboard's Conductor view edits all of it.

## Draining

SIGTERM and SIGINT drain the server. It stops taking work, `/health` answers `draining` with a 503, new requests get a 503 with `Retry-After: 30`, the generation in flight finishes with its last chunk and `[DONE]`, and the process exits 0. `ops/stop.sh` sends SIGTERM and waits up to 60 s (its first argument) before it kills.

## Logs and privacy

Each request leaves one log line with its token counts, its finish reason and its rate. No line carries the text of a prompt or an answer. `--log-content` lets exception messages that quote a request into the log, and `--log-request-keys` logs parameter names and scalar values, never messages. A usage ledger in SQLite, one row per request with counts and timings only, feeds the dashboard.

## Sampling and penalties

Every sampling field works on every path, speculative ones included, and a seeded request reproduces its text exactly ([exactness.md](exactness.md)). Penalties apply to every row before sampling. None is set in the served profile, and a client that sends one gets it.

## The conductor's session monitor

`--conductor raw|shadow|on` switches the second-level harness (`server/conductor/monitor/`). `raw`, the default, runs none of it: the output is the raw engine's, and there is no stall close either (a client's own `max_reasoning_tokens` still closes the block with the streamed phrase). `shadow` runs every rule and journals what it would do, and executes nothing. `on` executes, each task in the mode its policy gives it. `off` is accepted for `raw`. A request's `X-Tandem: raw|shadow|on` header can lower the server's mode, never raise it, and every completion response carries `X-Tandem` with the mode the request ran in. A request with `n > 1` never acts (at most `shadow`).

The policy is a file, `--conductor-policy` (default [ops/supervisor.yaml](../ops/supervisor.yaml)): every task's mode (`off`, `shadow`, `act`), every threshold and every text. It is validated at start against [ops/supervisor.schema.json](../ops/supervisor.schema.json) (JSON Schema, draft 2020-12), which `python tools/conductor_policy.py --schema` writes from the rule registry; `python tools/conductor_policy.py [file]` checks a file.

The monitor acts only at four safe points, from a fixed action set:

| safe point | when | actions |
|-|-|-|
| SP4 (the notes' SP0) | the prompt is rendered, before prefill | `PROMPT_TAIL_NOTE`, `SET_THINK_BUDGET` |
| SP1 | a verify block is committed and yielded | `CLOSE_REASONING`, `GIVE_UP` |
| SP2 | the model wrote its end token, before it reaches the client | `CONTINUE_HIDDEN`, `GIVE_UP` |
| SP3 | `</tool_call>` closed a call, and the call is held | `REPAIR_CALL`, `REGENERATE_CALL`, `GIVE_UP` |

`PASS` is "nothing"; `NOTIFY` is out of band and has no safe point yet. What the client sees: a hidden close forwards the policy's phrase through the engine and streams only from the closer on (`</think>` in `tags`, nothing in `reasoning_content`); a hidden continue drops the end token and forwards the close phrase (inside the reasoning) or a hidden user turn with a nudge and an empty closed reasoning block (after it), and the answer follows in the same response; `GIVE_UP` is one visible sentence from the policy with finish `stop`. Caps: `acts_per_request` and `acts_per_session_hour`, and never the same hidden action twice in a row on one turn; beyond them an act becomes `GIVE_UP`. Tool calls are held (their arguments arrive whole at `</tool_call>`) only while a tool-call task can act.

SP4's splices touch the prompt's tail only, so the resident prefix and the state store stay valid: a note goes into the last message just before its turn end (the template refuses a system message that is not first), and thinking off replaces the generation prompt's open `<think>\n` with the closed empty block the template writes for `enable_thinking: false`. Neither changes the system turn or `reasoning_effort`, which render as the prompt's first tokens.

Hidden tokens are never in `usage` (`completion_tokens` counts streamed tokens, `prompt_tokens` what the client sent), never in the suffix store, and a response the monitor acted on is not stored in the response cache. The KV holds them, so the next turn's prompt resumes at the last prefill chunk line before them.

Sessions: the conversation id the client sends (`conversation_id`, `session_id` or `user` in the body, `X-Conversation-Id`, opencode's `x-session-id`); else a compaction request (no tools, one user message) joins the most recent session and the next unseen prefix joins it too; else a hash of the first `lineage_tokens` prompt tokens, joined within `join_window_s`. The state is an LRU of `max_sessions`, with tool calls and results as hashes, never text.

Every decision is one journal event and one `[conductor]` log line (task, decision, reason, counts; never a text). The journal keeps the exact hidden text in memory for the session export.

### Extending it

The interface the conductor's policy engine builds on, kept stable:

- A rule: `tier0.register(name, safe_point, params, required, doc)` on `fn(event, task, sup) -> Action | None`. `params` is the JSON Schema of the task's parameters; the policy schema picks it up. A task in the file names the rule (`rule: name`) and carries its parameters, texts included.
- The events (`server/conductor/monitor/observe.py`): `RequestEvent` (SP4, and `sup.req` at every later point: messages, tools, `tool_choice`, `n`, prompt tokens, `in_think`, `max_tokens`, the client's own budget, the client kind), `BlockEvent` (reasoning tokens, inside, the detector arm that fired, the history arm's share), `FinishEvent` (inside, the answer so far), `ToolCallEvent` (name, arguments, index).
- `Action(kind, reason, text, budget, score, detail)` from `policy.ACTIONS`. `SET_THINK_BUDGET` with `budget=0` turns thinking off; with `budget=n` the request's reasoning closes after n tokens with the action's `text`, which must contain the closer.
- `Monitor(mode, policy, tok)`: `.begin(mode, req, prompt_ids, conv_id)` gives the request's `Supervisor` (None when raw), `.reload(policy)` swaps the policy for requests that start later, `.mode_for(header, n)` lowers a mode. A supervisor's `.session` is the session's state; `.memo` is a rule's per-request scratch.
- The journal: `Journal.write(event)`; `Journal.sinks` are called with every event (the SQLite table, Prometheus, the export).
