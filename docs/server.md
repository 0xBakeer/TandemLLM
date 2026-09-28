# Server and API

`server/app.py` speaks the OpenAI HTTP API, so any OpenAI client can use the engine by changing its base URL. It is built on the Python standard library (`ThreadingHTTPServer`), with no web framework. This page covers the routes and the request fields, then tool calls and structured outputs, then how the service behaves under load.

## Routes

| route | who may call it | what |
|-|-|-|
| `POST /v1/chat/completions` | anyone | chat, streamed or not |
| `POST /v1/completions` | anyone | plain text completion |
| `GET /v1/models` | anyone | the served model name |
| `GET /health` | anyone gets the status; the full body needs a token | status, queue counters, cache report, memory |
| `GET /metrics` | the metrics or admin token | Prometheus metrics ([server/METRICS.md](../server/METRICS.md)) |
| `GET /v1/cache/stats` | the admin token | the caches' counters |
| `POST /v1/cache/clear` | the admin token, as a bearer header | empty the caches |
| `GET /dashboard/` | anyone | the dashboard's static files, which hold no data |
| `/v1/dashboard/*` | the admin token or a dashboard session | the dashboard's data ([dashboard.md](dashboard.md)) |

The admin and metrics tokens live in a `secrets.env` file that `ops/make-secrets.sh` writes (mode 600, never in the repository). Without an admin token the admin routes answer 404: the dashboard API does not exist. Requests from the box itself (loopback, no proxy headers) need no token for the read-only admin routes. The watchdog and the measurement tools depend on that.

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

`server/toolcall.py` turns that into OpenAI `tool_calls` with `finish_reason: "tool_calls"`, on both the streamed and the JSON path. It reads the strict form, a lenient form and a JSON call object, and it reads only the answer, never the thinking. On a stream, a call goes out as argument deltas while the model writes it: the id and the name first, then each string value in pieces. A whole-file write therefore shows progress instead of minutes of silence.

Argument values carry their schema's types. The model writes every value as text, so `offset` arrives as the characters `150`. Our parser looks up the request's own schema, and a parameter whose schema does not allow a string is returned as JSON: `150`, `true`, a list. A typed value is held back on the stream until it is complete, then sent whole. On an 8-task agent run, a strict client that validates arguments against the schema refused 58 of 80 calls before this change, and 0 of 26 after.

`tool_choice` works as OpenAI defines it. `none` renders the prompt without tools. `required` and a named function become a constraint on the answer, the same token masks structured outputs use, so the model cannot answer without the call. `parallel_tool_calls: false` keeps one call.

## Structured outputs

`response_format` with `json_object` or `json_schema`, and `structured_outputs` with a `regex`, a `choice` list or a JSON schema, constrain the answer to a regular language. `engine/grammar.py` compiles each form to a finite automaton over bytes, because the tokenizer is byte-level and a token can end halfway through a character. A token is allowed in a state when walking its bytes never reaches the dead state. The engine computes that for the whole vocabulary at once per state and caches it, so a state costs a few milliseconds once.

Each mask applies to every verified row of a draft tree, not only to one row per step. Constrained output therefore stays exact under speculation: it is the text plain constrained decoding writes ([exactness.md](exactness.md)). JSON schemas with recursion (`$ref` cycles) are refused, and `json_object` allows a fixed nesting depth.

## Streaming

A streamed response sends one chunk per token, carries `usage` on the last chunk when the client asks for it (`stream_options.include_usage`, and by default with `--usage-default on`), then a chunk with `finish_reason` and `data: [DONE]`. A verify round accepts several tokens at once, so they reach the socket together. Half the gaps between tokens are therefore microseconds, and the other half a whole round.

`finish_reason` is one of `stop` (end of text or a stop string), `length` (`max_tokens`), `tool_calls`, `timeout` (the request's wall-clock cap), or `error`. On `error` the partial text has already been sent and an `error` object sits beside the finish reason. A client that hung up is logged as `abandoned`, not as an error.

`server/stream.py` turns tokens into characters. A character can span tokens, because the tokenizer works on bytes, and a stream that sent every decoded prefix would send a replacement character (U+FFFD) that never goes away. Our detokenizer holds back a trailing partial character until the next token completes it. Every character it sends is final.

## A long prefill, and a client that leaves

A 190,000-token prompt prefills for about 300 s. After every prefill chunk the handler checks its socket. If the client has gone, the prefill stops there, the request ends `abandoned`, and the engine is free at once. Rows already prefilled stay in the resident prefix ([caches.md](caches.md)), so a retry continues from there. A request whose client left while it waited in the queue is dropped before it runs.

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

A bad image is a 400 that names the part, for example `messages[1].content[0].image_url`: a scheme other than https or data, a download that fails, broken base64, a body in none of the five formats it reads (PNG, JPEG, WEBP, GIF, BMP), more than `--image-max-mb` (20), a header that declares more than `--image-max-decode-pixels` (64 million), more than `--max-images` (16) in a request, an image in a system message, a video part, and any image when the server runs with `--vision off`. `--image-act-gb` (6) is the admission check: an image whose encode would need more memory than that, by the tower's byte math, is refused before it queues; the largest image the processor admits needs about 3.4 GiB.

Text requests do not touch any of this: the engine's image state is empty for them and every forward is the text one, bit for bit.

## Draining

SIGTERM and SIGINT drain the server. It stops taking work, `/health` answers `draining` with a 503, new requests get a 503 with `Retry-After: 30`, the generation in flight finishes with its last chunk and `[DONE]`, and the process exits 0. `ops/stop.sh` sends SIGTERM and waits up to 60 s (its first argument) before it kills.

## Logs and privacy

Each request leaves one log line with its token counts, its finish reason and its rate. No line carries the text of a prompt or an answer. `--log-content` lets exception messages that quote a request into the log, and `--log-request-keys` logs parameter names and scalar values, never messages. A usage ledger in SQLite, one row per request with counts and timings only, feeds the dashboard.

## Sampling and penalties

Every sampling field works on every path, speculative ones included, and a seeded request reproduces its text exactly ([exactness.md](exactness.md)). Penalties apply to every row before sampling. None is set in the served profile, and a client that sends one gets it.
