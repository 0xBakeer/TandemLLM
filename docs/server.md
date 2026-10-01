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
| `PUT`/`POST /v1/conductor/*` | the admin token or a dashboard session, or the box itself | new config versions, restores, labels, decision-model loads |
| `POST /v1/systemone`, `/v1/predict` | the admin token or a dashboard session, or the box itself | typed decisions from the engine's own decision models (below); only with `QWEN38_DECIDER_MODELS` set |
| `GET /readyz` | anyone | whether a decision model is loaded and warm; only with `QWEN38_DECIDER_MODELS` set |

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

With `--conductor-config DIR` (off by default; `ops/start.sh` passes serve.env's `QSE_CONDUCTOR_CONFIG` to the :8000 service only), the server serves one model, `TandemLLM`: the weights with a conductor in front ([server/conductor/](../server/conductor/)). `/v1/models` lists `TandemLLM` alone. The raw engine's name (`--served-model`, `qwen38-spark-engine` on the box) is a hidden alias: it is not listed, `GET /v1/models/qwen38-spark-engine` answers with `TandemLLM`'s card, and a request that names it, or any other model, runs exactly as a `TandemLLM` request, so no client breaks. Every response carries `X-Tandem` (`raw`, `shadow` or `on`: the resolved profile's mode, which also caps the session monitor's mode below) and `X-Tandem-Profile`. The monitor runs the profile's task modes, caps and the thresholds that map onto its policy file. Every response, `/health` (with its `cache` block), the dashboard's System view and the Live stream say `TandemLLM` as the model; `/health`, the cache block and the System view give the base engine's own name as `base_model`. The usage ledger records `model_served = TandemLLM` for every request, beside the name the client sent (`model`) and the mode it ran in (`conductor_mode`).

Raw is a header, not a model name. `X-Tandem: raw` (or the locked profile `raw` or `bench` named in `X-Tandem-Profile`) serves the release's path byte for byte: no class, no note, no effort change, the monitor raw. Benchmarks and the paper's numbers use it: `tools/client_smoke.py --x-tandem raw` and `tools/session_replay.py --x-tandem raw` send it, and `tools/row3.py --no-server` refuses a server that lists `TandemLLM`, because the atlas runner sends no headers (a server row3 starts itself runs without the conductor and is the raw engine). `tools/flagoff_identity.py` runs the engine in-process and is not affected.

A request resolves to a profile: the `X-Tandem-Profile` header (an unknown name is a 400 with `param: X-Tandem-Profile`), else the profile mapped to the client (opencode gets `coding`, Open WebUI `chat`), else the default. `X-Tandem: raw` or `shadow` lowers the mode and nothing raises it. `raw` and `bench` are locked.

Before the request queues, the router decides which kind of task the turn is. A turn that only carries tool results keeps the session's class, so in an agent session only a new user message asks a model: the decider's `choice` question first (any model behind the Jev question shape: role `decider`, by default `self`, the engine's own decision models below, asked in process; `engine:<family>` names them too, and an HTTP endpoint another server's; used at 0.80), then the engine itself with `X-Tandem: raw` and thinking off and `structured_outputs.choice` (used at 0.55), else the session's class. No word list decides anything. The class sets the thinking budget, thinking off (the generation prompt's `<think>\n` becomes the template's closed empty block at the tail; the template renders `reasoning_effort` as the first system tokens and `enable_thinking: false` drops that sentence, so the router never re-renders), and a note with the effort sentence, a skill or package and rules. The note goes into the last turn before its `<|im_end|>`, and later turns replay it at the same token position, so the resident prefix keeps matching. It is never streamed, never counted in `usage`, never stored in the suffix store, and a request with a note skips the response cache. A client's `max_reasoning_tokens`, `thinking_budget` or `chat_template_kwargs.enable_thinking` always wins. The budget is a real cap per effort level (SRV-159; before it medium, high and xhigh differed by their sentence alone): low 1,024, medium 8,192, high 16,384 reasoning tokens, xhigh none below the D2 backstop, off thinking off; a profile's `thinking.budgets` overrides a level (the journal's `budget_source` is then `profile`, else `class`). When the reasoning reaches the cap, the session monitor closes it at SP1 with the vendor's phrase hidden (`texts.vendor_close`, the path the D2 backstop takes): the model's context carries the phrase, the client receives `</think>` and the answer, and the journal has `CLOSE_REASONING` with task `route` and reason `budget(N)`. Without a monitor that acts (none, shadow) or for `n` > 1 the cap is ThinkBudget's, whose phrase streams.

What a classifier reads (SRV-153): a digest of the new user message, head and tail, sized per stage (the decider's to its own `limits()`, `(context_tokens - question_tokens) x 3.5` characters, 1,120 for Laya; Qwen's to the router's `digest_chars`), followed by one context line the router writes from what it already knows: `prev=code_edit@0.93 tools=read,edit,bash files=.py,.md checker=calccheck:FAIL compaction=no` (the session's class and confidence before this turn, the last three tool names, the extensions of the files the last eight calls named, the newest checker result, whether a compaction came since the last classification; `none` for what is not known yet). The line is at most 120 characters and comes out of the tail's share, so the digest keeps its size; it never holds a path, a file name or a file's text. `classifier.context_line` in the router document switches it, off by default: with Laya alone it failed SRV-153's ship rule (2026-10-01, Laya on the box CPU: the 216 short prompts answered right at the 0.80 threshold 50.0 % without it and 32.4 % with it, and of 12 follow-ups that change the task 10 were right without it and 1 with it, because the line pulls Laya to the previous class); the Qwen stage with the line is still to be measured. The journal's `digest` is the text the stages read and `extra.context_line` the line. Before SRV-153 the router built one 2,000-character digest and the decider cut it to its limit from the front, so on a long message it never saw the tail; sized per stage, Laya's best guess on the long set was right 33 of 60 times instead of 29 (requests at the tail 14 of 20 instead of 7), and its confident wrong answers went from 3 to 0.

Effort by task size (SRV-157): a class's policy may carry `effort_by_size` (`code_edit`: small `high`, medium and large `xhigh`), and the class's effort is the ceiling (a level above it is cut to it). The size comes from four structural signals in the router document's `effort.size`: the prompt tokens the session added after its first request (from the ledger's count of each turn; opencode's own head, about 24k tokens, is not the task), the distinct files its tool calls named, the most gates one known checker's RESULT line named, and the turn index. Each is small below its first threshold (20k tokens, 3 files, 4 gates, 30 turns), medium up to its second (60k, 8, 8, 100), large above; the task's size is the largest any signal reached and never falls within a task, and a compaction starts it over. The profile's `thinking.by_size` says what it does: `off` nothing, `shadow` (the default, also for every profile stored before it existed) the class's effort with the size and the effort it would set in the journal (`extra.size`, `shadow.size`), `on` the size's level as the turn's effort (budget and sentence as before; a client's own budget still wins). The Routing screen edits the levels per class and the thresholds; the Profiles screen the mode; the test box shows the size; `tools/tandem_eval.py` reports turns per effort level per run and per suite.

The config (profiles, the router's classes and thresholds, roles, packages, settings, and the session monitor's policy as the `monitor` document, whose version 0 is the policy file below) lives in versioned JSON files under the directory, outside any repository. `PUT` makes a new version with a required note, `POST .../restore` makes an old one current again, and the next request uses it. A file that fails to parse or validate keeps the last good version and shows in `GET /v1/conductor/config`. Every decision, the router's and the monitor's, goes to one journal (below) for 30 days; `GET /v1/conductor/sessions/{key}/export?format=md` lists what the model saw that the client did not. The contract is [contract/conductor-v1](contract/conductor-v1/README.md), and the dashboard's Conductor view edits all of it.

## Decision models (stage A)

The engine can serve decision models itself (`engine/decision/`, ENG-179 and SRV-131): models that answer typed questions over a state in one forward pass with calibrated probabilities, `choice` (one of N options), `score` (an expected level) and `noul` (a probability that a statement holds). Laya's three checkpoints (`convaiinnovations/laya`, Apache-2.0) are the first: `laya-english` (ModernBERT-large, 512 tokens of state and question, 192 for the question), `laya-multilingual` (mmBERT-base, 1,024 and 256) and `laya-typed-decisions` (1,024 and 256). `QWEN38_DECIDER_MODELS` in serve.env names the ones this server serves and `QWEN38_DECIDER_DIR` where their files are; empty, nothing of it is imported and every route below is absent.

The arithmetic is the Laya SDK's, through `transformers`' ModernBERT, so the answers are the SDK's (`tools/decider_equivalence.py`: 0.00 on 22 questions): fp32 parameters, on CUDA the forward under `torch.autocast(bf16)`, on the CPU plain fp32. Nothing is shared with Qwen: each decision model's tensors live in a CUDA memory pool of its own, its forwards run on a stream of its own, and in this stage a forward runs only while no request holds the engine (the lock every completion holds from its prefill to its last token). A completion that arrives during a decision waits at most one decision forward; a question asked while a completion runs waits for it up to its deadline and is dropped after. Callers that arrive together are answered in the same idle window, each with a forward of its own, the forward the SDK runs for that call, so an answer never depends on who else asked; `QWEN38_DECIDER_COLLATE=rows` collates them into one forward instead (arbiter's way, faster under many callers, and on the GB10 a probability then moved by up to 3.4e-3 with no choice changing).

A model loads through the residency planner (the decider's claim, 8.2 GiB, against MemAvailable, floor 15 GiB and margin 6 GiB; never during a hold, never beside another load): `ops/start.sh` asks for the conductor's `deciders` document's `on_start` models once the engine is healthy and its page cache is dropped (`POST /v1/conductor/deciders/autoload`), an `on_demand` model loads when a question first needs it, and the dashboard's Conductor view, Deciders screen, loads and unloads by hand. A model whose forward fails three times in a minute is unloaded and marked failed; the engine serves on.

`POST /v1/systemone` (and `/v1/predict`, the same) takes arbiter's Jev request, `{state, model?, questions: {id: {type, instructions, criteria?}}, task?, lang?}`, and answers `{model, answers, usage, routing, latency_ms, path}`. `model` is `auto` or `jev-latest` (script and language detection picks the checkpoint), a model's name, or an alias (`en`, `multilingual`, `typed`, ...). Errors keep arbiter's body, `{"type": "error", "error": {"type", "message"}}`: 422 `invalid_request_error`, 529 `overloaded_error`, and 503 with `Retry-After` for `warming`, `loading`, `refused` (the planner said no, with its decision), `failed`, `not_loaded` and `busy_decoding` (`X-Decider-Deadline-Ms`, 30 s by default, passed while a completion held the engine). `X-Decider-Window: in_decode` is refused until the low-priority stream and its governor exist (stage C). `GET /v1/models` keeps `data` as it is, `TandemLLM` alone with the conductor on, and lists the decision models beside it under `decision_models`, so an OpenAI client still sees one chat model; `GET /v1/models/{id}` answers a decision model's card, and `GET /readyz` is 200 once one is loaded and warm. The contract is [contract/conductor-v1/systemone.schema.json](contract/conductor-v1/systemone.schema.json).

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
| SP3 | the answer ended (SP2's moment) and its calls are held | `REPAIR_CALL`, `REGENERATE_CALL`, `GIVE_UP` |

`PASS` is "nothing"; `NOTIFY` is out of band: the stall watchdog's watch sends it (below), never a request. What the client sees: a hidden close forwards the policy's phrase through the engine and streams only from the closer on (`</think>` in `tags`, nothing in `reasoning_content`); a hidden continue drops the end token and forwards the close phrase (inside the reasoning) or a hidden user turn with a nudge and an empty closed reasoning block (after it), and the answer follows in the same response; `GIVE_UP` is one visible sentence from the policy with finish `stop`. Caps: `acts_per_request` and `acts_per_session_hour`, and never the same hidden action twice in a row on one turn; beyond them an act becomes `GIVE_UP`. Tool calls are held only while a tool-call task can act (and never for a constrained request, `tool_choice` required or named: the grammar writes those calls).

### Tool-call checks (SRV-113)

With a tool-call task able to act, every call of an answer is held until the answer ends; a block that does not parse is held too, not let out as content. At the end token every call is checked by every SP3 task, before any is released:

| task | rule | what it decides | wave 1 |
|-|-|-|-|
| C4 | `malformed_call` | a block that does not parse: a deterministic fix (`REPAIR_CALL`: JSON with trailing commas or Python literals, an unquoted `name=`, a `</parameter>` that never came) or a hidden reminder with the parser's reason (`REGENERATE_CALL`, at most `regenerations` a request, then the block goes out as content as without the checks); the arguments against the tool's JSON schema (required keys, types); a call the token limit cut (`GIVE_UP`, never half a call) | act; the schema check in shadow |
| C3 | `repeated_call` | the same name and canonical arguments as the session's previous `repeats - 1` calls, whose results (from the request's tool messages) were identical: "do something different" | act |
| C1 | `risk_floor` | the floors, a list in the policy: `rm -rf` outside the project root, `git push`, a write outside the root, a download piped into a shell; the root is the working directory opencode states in its system prompt | shadow |
| C2 | `project_rules` | what text alone can decide of AGENTS.md (only when the system prompt carries it): a numbered copy of a file the session used, a path under `/tmp/`, a written file over 400 lines; a call whose every path is on `allow_paths` is exempt from all three (default `/tmp/opencode`, opencode's own scratch; a path covers everything under it, `*` `?` `[` make a glob, `~` is home, a relative entry is under the project root) | shadow |

If all pass, the calls go out whole, each in one delta with an id made at release and indexes contiguous from 0. If any act fails, none goes out: the end token is dropped, a hidden turn shaped as the calls' tool result (`templates.regen_turn`) carries the reminders, and the model writes the set again in the same response; the client receives only the new calls, at index 0 again. An answer that ended without its end token (the token limit, a stop string) cannot be written again, so there a failure gives up. A block the client's own stop string cut is not a failure: C4 lets it out as the text it was (`cut_by_stop`), as the raw engine answers; only a block the token limit cut follows C4's `cut` (`give_up`). A profile turns each check's pack on or off (`toolgate.risk`, `project`, `same_call`, `malformed`) and sets its mode and thresholds. With the monitor raw, or no tool-call task able to act, calls stream as they always did. A streamed request with `stop` strings is the one exception, with or without the monitor: its calls go out whole when their block closes, because a stop string can end the generation inside a block, and arguments already streamed cannot be taken back; such a cut block is the text it was, as in the non-streamed answer.

SP4's splices touch the prompt's tail only, so the resident prefix and the state store stay valid: a note goes into the last message just before its turn end (the template refuses a system message that is not first), and thinking off replaces the generation prompt's open `<think>\n` with the closed empty block the template writes for `enable_thinking: false`. Neither changes the system turn or `reasoning_effort`, which render as the prompt's first tokens.

Hidden tokens are never in `usage` (`completion_tokens` counts streamed tokens, `prompt_tokens` what the client sent), never in the suffix store, and a response the monitor acted on is not stored in the response cache. The KV holds them, so the next turn's prompt resumes at the last prefill chunk line before them.

Sessions: the conversation id the client sends (`conversation_id`, `session_id` or `user` in the body, `X-Conversation-Id`, opencode's `x-session-id`); else a compaction request (no tools, one user message) joins the most recent session and the next unseen prefix joins it too; else a hash of the first `lineage_tokens` prompt tokens, joined within `join_window_s`. The state is an LRU of `max_sessions`, with tool calls and results as hashes, never text.

Every decision is one journal event and one `[conductor]` log line (task, decision, reason, counts; never a text).

### Session-level tasks (SRV-121)

Three SP4 tasks look across turns. They read what a request carries about the turns before it, the model's tool calls (the assistant messages' `tool_calls`) and their results (the `tool` messages), each taken once per session by its call id, so a retried request counts nothing twice. Each acts with one `PROMPT_TAIL_NOTE` at the end of the prompt, and a request carries at most one of their notes, in the policy file's order:

| task | rule | when | the note (policy text) |
|-|-|-|-|
| S2 | `context_health` | the monitor marked opencode's compaction request (no tools, the history as one user message; a client without a conversation id carries the mark to its next session key), and the first `steps` (2) tool-call steps after it read none of `files` (`PROGRESS.yaml`, `AGENTS.md`, `SKILL.md`) through `read_tools`, nor loaded a skill through `skill_tools`. Once per compaction; a step that reads one ends the watch (a `PASS` observation names it) | "You are after a summary. Read PROGRESS.yaml and AGENTS.md before the next change." |
| S1 | `gate_stagnation` | a checker gate failed `fails` (3) times in a row and the best later reading came less than `min_improvement` (10 %) of its distance to the limit closer than the first (15.6, 12.2, 3.5 improves; 3.5, 12.3, 3.5 does not). The checkers (`pagodacheck`, `scenecheck`, `sketchcheck`, `calccheck`, `taskcheck`), the gate label and the reading are patterns in the policy; a gate is checker + target + gate, so the same metric on two screenshots is two gates. Once per gate and stage, the stage being the `next:` line the state file last showed (read, or written by an edit); a PASS of the gate starts it over. Several stuck gates: one note naming the worst (farthest from its limit) | "Gate (a) read 3.5, 12.3, 3.5. Record this stage as blocked in PROGRESS.yaml with the reason and take `next`." |
| S4 | `stall_watchdog` | requests keep coming, but for `minutes` (20) no file changed (`change`: a write or edit whose result reads as done), nothing was committed (`commit`) and no checker number got 10 % closer to its limit or passed; again only after another `minutes`, and never within `minutes` of another session note | "No progress for 20 minutes. Re-read PROGRESS.yaml and take the next stage." |

All three stay silent on a request that offers no tools and in a project whose system prompt does not match `requires` (`PROGRESS\.yaml`): a project without state files. The midnight date line in opencode's system prompt is a prompt change, not a compaction.

The client that has gone: when no request comes for S4's `minutes` after a turn ended, nothing can act in band. The watch (`Monitor.sweep`, every `--conductor-watch-s` seconds, 30 by default, on its own thread) journals one event a turn end, task S4 at `sp` `watch`, reason `client_gone(N min)`: `PASS` with `notify: off`, or `NOTIFY` when the policy's `notify.enabled` is on (in shadow `NOTIFY` is journaled as "would", never sent). A push goes to `notify.url` (`https://ntfy.khaled.de/k8s-alerts`) with the token read from `notify.token_file` (`~/.qwen38-spark-engine/conductor/ntfy.token` on the box; never in the policy, the journal or a log) and the CA in `ca_file`; its result ("sent 200", "failed: no token file") is in the event's detail. `notify.enabled` is off in `ops/supervisor.yaml`; a profile's `context.notify` sets it, on only in the `overnight` profile (Khaled, 2026-09-30). A server with the fake engine never pushes.

The state is per session (`SessionState.tasks`), hashes, numbers and file names only. `GET /v1/conductor/watch` lists it per session: each gate's readings and whether it was noted, a compaction being watched and its steps, the minutes since progress and what it was, the client's idle minutes and whether it has gone, and whether a push would go out; the dashboard's Monitor screen shows it every 5 s. A profile's `context` section sets S1's `fails` (`s1_runs`), S4's `minutes` (`s4_minutes`) and `notify.enabled` (`notify`); the task modes come from its `tasks` as for every task (`overnight`: S1, S2, S4 act; `coding` and `chat`: off).

### Observability (SRV-115)

One journal for the router and the monitor: the table `conductor_events` in the usage ledger's file (schema version 2, migrated in place at start, every old row kept; `python tools/ledger_migrate_check.py LEDGER` tries it on a copy), written from its own queue thread, never on the decode thread. A router row (`kind` route) per `TandemLLM` request; a monitor row (`kind` monitor) per decision with its task, tier, mode, rule and reason, action, the hidden text (stored once in `conductor_texts` by its hash), its tokens, its position and the reasoning tokens at a cut; the request's outcome fills its rows when it ends, and the next request's checker result (`RESULT: PASS|FAIL <- ...`) fills `next_checker`. Checks that find nothing are counted, not stored (an answer whose calls all passed is one row, `journal.tool_passes`). Rows and texts older than 30 days go nightly, and the ledger's weekly backup leaves both tables out. With the ledger off the journal lives in memory; with the conductor raw nothing is written.

The ledger's `requests` gains `conductor_acts`, `conductor_hidden_tokens` and `conductor_acted`; the dashboard's speed figures leave out the requests the conductor acted in. Prometheus (`qse_conductor_*`, [METRICS.md](../server/METRICS.md)) counts every decision, passes included, the actions, the hidden tokens, the gave-ups, the tier 1 questions and their failures, and the reasoning tokens cut. The live event (contract 1.2) carries `activity.conductor` on a watched request, with `supervising` while a hidden forward runs. `GET /v1/conductor/events` lists the monitor's decisions, and a session's timeline and export hold both halves.

### Extending it

The interface the conductor's policy engine builds on, kept stable:

- A rule: `tier0.register(name, safe_point, params, required, doc)` on `fn(event, task, sup) -> Action | None`. `params` is the JSON Schema of the task's parameters; the policy schema picks it up. A task in the file names the rule (`rule: name`) and carries its parameters, texts included.
- The events (`server/conductor/monitor/observe.py`): `RequestEvent` (SP4, and `sup.req` at every later point: messages, tools, `tool_choice`, `n`, prompt tokens, `in_think`, `max_tokens`, the client's own budget, the client kind, whether a grammar constrains the answer, the profile), `BlockEvent` (reasoning tokens, inside, the detector arm that fired, the history arm's share, the position), `FinishEvent` (inside, the answer so far, the position), `ToolCallEvent` (name, arguments, index; for a block that does not parse its text and the parser's reason; the tool's schema; `final` when nothing can be written again).
- An SP3 rule returns `REPAIR_CALL` with `detail["arguments"]` (a call) or `detail["calls"]` (a block that did not parse), `REGENERATE_CALL` whose `text` is the reminder, `GIVE_UP`, or `PASS` with a reason (an observation); `detail["mode"]` lowers what one finding may do. `Supervisor.check_answer(items)` runs them; `release_call` / `release_block` hand out what goes.
- `Action(kind, reason, text, budget, score, detail)` from `policy.ACTIONS`. `SET_THINK_BUDGET` with `budget=0` turns thinking off; with `budget=n` the request's reasoning closes after n tokens with the action's `text`, which must contain the closer.
- `Monitor(mode, policy, tok)`: `.begin(mode, req, prompt_ids, conv_id)` gives the request's `Supervisor` (None when raw), `.reload(policy)` swaps the policy for requests that start later, `.mode_for(header, n)` lowers a mode. A supervisor's `.session` is the session's state; `.memo` is a rule's per-request scratch.
- The journal: `Journal.write(event)`; `Journal.sinks` are called with every event (the SQLite journal's `monitor_sink`, Prometheus's `on_conductor_event`), and with three bookkeeping events: `OUTCOME` (the request ended), `LABEL` (a later checker result), `COUNTS` (the checks that found nothing).
