# The conductor

The conductor is the layer between a client and the weights. With it on, the server serves one model, `TandemLLM`: the weights with the conductor in front ([server/conductor/](../server/conductor/)). For every request it decides which profile applies, what kind of task the turn is, how long the model may think, and which hidden text goes into the prompt. While the model writes, it watches for loops, empty answers and broken tool calls, and acts at fixed points. It does not change the model's arithmetic: the tokens come from the same greedy decoding. Every decision, acted on or not, goes to one journal.

This page is the overview. Each section links to the full rules.

## Modes and headers

`--conductor-config DIR` (`QSE_CONDUCTOR_CONFIG` in serve.env) switches the conductor on. Without it the server serves its own model name and none of this runs. With it, `/v1/models` lists `TandemLLM` alone, and a request that names any other model, the engine's own name included, runs as a `TandemLLM` request.

| mode | what runs |
|-|-|
| off | no config directory: the engine alone, under its own name |
| `raw` | the release's path, byte for byte: no class, no note, no effort change, no monitor |
| `shadow` | every rule decides and writes to the journal; nothing acts |
| `on` | the profile's decisions act, and the monitor runs each task in its policy's mode |

A request picks its profile with `X-Tandem-Profile: <name>`; an unknown name is a 400. `X-Tandem: raw` or `shadow` lowers the mode for one request. No header raises it. Every response carries `X-Tandem` (the mode it ran in) and `X-Tandem-Profile` (the profile). `--conductor raw|shadow|on` sets the session monitor's mode for the server, and a profile's mode caps it.

## Which profile a request gets

The conductor knows a client by its requests (headers, tool shapes, fixed system texts), never by a word list. A request resolves in this order:

1. the `X-Tandem-Profile` header;
2. the profile mapped to the client: opencode gets `coding`, Open WebUI and the dashboard's Playground `chat`, the Hermes agent `hermes`, the Chat tab (chatd, client kind `chat-app`) `chat-app`;
3. `tools`, when no profile maps the client and the request offers a function tool with `tool_choice` other than `none`: an OpenAI SDK loop, curl, a benchmark (SRV-621; `settings.tools_profile`, empty turns the rule off);
4. `settings.default_profile`, `general` on a new box.

The journal row says which rule picked it (`extra.profile_source`).

| profile | for | what it turns on |
|-|-|-|
| `raw`, `bench` | locked | nothing: the raw engine |
| `coding` | opencode | every routing decision, packages, the loop guards, the tool-call checks, the claim check, status lines |
| `chat-app` | the Chat tab | `coding`'s policy without status lines in the answer |
| `hermes` | the Hermes agent | `coding`'s policy, with status lines in the reasoning |
| `chat` | Open WebUI, the Playground | class-aware effort and packages, the reasoning guards, the same-call guard |
| `overnight` | unattended opencode runs, by header | `coding` plus the session tasks, higher caps, a push when the client has gone |
| `tools` | API clients that offer tools | the client's prompt, token for token; the loop guards and tool-call checks act |
| `general` | any other client | class-aware effort and packages |

The `tools` profile adds nothing to the prompt. It asks no class question: that side call made a full tool-eval-bench run take 1.23 times raw's time. Each act it takes is a journaled monitor event, so a request with no act gets the raw request's prompt.

## The router: one class per turn

Before a request queues, the router decides what kind of task the turn is. The shipped classes are `code_edit`, `debug`, `design_ui`, `drawing`, `three_d`, `writing`, `plan`, `research`, `summarize` and `chat`. Only a new user message asks a classifier. A turn that carries only tool results keeps the session's class, and so does the first message after a compaction.

In the shipped router the served model classifies its own requests: raw, thinking off, its answer limited to the class ids by a choice grammar. Every class's probability comes from the answer's first token. The answer counts at 0.45 or more. Below that, the turn keeps the session's class, or takes the profile's default class on a first turn. One exception covers a first turn whose top class builds something and leads the second by 0.1: the turn takes that class (SRV-255). On 216 labelled prompts the served model was right 0.949 of the time and a small decision model 0.782 (manual, chapter "Profiles and the router"). The classifier reads the head and the tail of the new message.

## Effort and the thinking cap

The class sets the turn's effort, and the engine enforces each level as a cap on the reasoning:

| effort | reasoning cap |
|-|-|
| off | thinking off |
| low | 1,024 tokens |
| medium | 8,192 |
| high | 16,384 |
| xhigh | none below the 32,768-token backstop (D2) |

At the cap, the monitor closes the reasoning with the model's own closing phrase. The model reads the phrase; the client receives `</think>` and the answer. The engine turns thinking off in the prompt's tail, so nothing before the tail changes. A client's own number of reasoning tokens always wins, and so does its thinking off. An effort word such as `reasoning_effort: medium` is a ceiling on the router's level. The `coding` and `overnight` profiles ignore the word, because opencode sends `medium` on every turn.

## Packages and skills

The class's plan goes into the prompt as a note: the effort sentence, the class's rules, and packages. A package is a text that tells the model how a kind of work is done well: a task skill, a test skill, a card that lists the client's tools. A package marked `general`, and the class's core skill (`threejs-scene` for `three_d`), go in on every turn of the class. A second question picks one package among the rest, or none.

Packages fit the request's tools (SRV-243). A package that needs a shell or a file write stays out when the tools cannot do it, or gives way to its fallback. On a plain Open WebUI chat a landing page request used to get the testing packages, call tools the chat did not have, and take 518 s. A package whose result a client tool delivers (send, schedule, post) can stay out when the user wants it done (SRV-615); this ships in shadow.

The library holds 44 longer skills behind a BM25 index, with no embeddings. The model searches it from its own scripts (`GET /v1/library/search?q=...`), and the router can retrieve from it under a profile's `library` switch, in shadow in `coding` and `hermes`. [packages.md](packages.md) lists the shipped packages and how each is measured.

## The note and its replay

The note goes into the last message, just before its end token. Every later turn of the session replays it at the same token position, so the resident prefix and every cache keep matching. The note is never streamed, never counted in `usage`, never stored in the suffix store, and a request with a note skips the response cache.

A later note carries only the parts no live note of the session already holds (`QWEN38_NOTE_DEDUP`, `parts`). Replayed notes have a budget of 16,000 tokens (`injection.max_replay_tokens`); over it the oldest note goes, and the turn reads everything after it again. Before the dedup, a turn in an Open WebUI agent chat read 35,442 tokens again (49 s) when the third full copy crossed the budget.

## Side requests

opencode sends title and summary requests beside the session's turns, and the Hermes agent sends a title request. The router knows them by shape: no tools, and the client's fixed instruction at the head of the system message (`structural.side_shapes`). A side request asks no classifier, gets no note and changes nothing in the session. Thinking is off, and the output is capped at 64 tokens for a title and 256 for opencode's summary. A side request of at most 8,192 prompt tokens goes ahead of every waiting turn. Before this, the router classed one title request as `code_edit` at xhigh, the model wrote 5,567 tokens for it, and the session's real turn waited 54 s behind it.

## Compaction and sessions

opencode's compaction request is the class `summarize`: thinking off, no package, nothing replayed. The first user message after it (`continue`) keeps the session's class, and the class's note goes in again at the compacted history. The engine serves the compaction as a continuation of the conversation it already holds, so the prefill reads one turn instead of the whole history ([caches.md](caches.md#compaction-as-a-continuation)).

A session is the client's conversation id (opencode's `x-session-id`, Open WebUI's `X-OpenWebUI-Chat-Id`). Without one it is a hash of the conversation's head, split further by each chat's first turn, so two chats that open with the same message stay apart (SRV-225). Before that split, a new agent chat started with 27 turns and 21,216 session tokens of earlier chats.

The notes give the model `{tandem_url}`: the request's own address plus `/s/<token>`, a random 12-character token kept 48 hours. Library reads and other calls from the model's scripts under that address are filed to their session in the journal.

## The session monitor and the loop guards

The router plans a turn before it runs. The monitor watches the turn while it runs and acts only at four safe points:

| point | when | actions |
|-|-|-|
| SP4 | the prompt is rendered, before the prefill | a hidden note at the tail, a thinking cap |
| SP1 | a verify block is committed | close the reasoning, give up |
| SP2 | the end token is written, before the client sees it | continue hidden, give up |
| SP3 | the answer ended and its tool calls are held | repair a call, have the model write it again, give up |

The shipped policy allows at most 2 acts a request and 6 a session hour, and never the same hidden action twice in a row on one turn. A give-up is one visible sentence with finish `stop`.

The tasks, by group:

- Reasoning: D1 closes a reasoning block that repeats itself, and D2 closes any block at 32,768 tokens, or at the output limit less 2,048.
- The end of a turn: E1 continues an empty answer, and sends tool calls the model wrote inside its reasoning as the answer when they parse. E2 checks a claim of done against the tool evidence since the last write.
- Tool calls: C4 repairs a call that does not parse when the fix is not a guess. C3 asks for something different when the same call comes a third time after identical results. C5 has the model write two changing calls (send, schedule, create, write, delete) one at a time; it acts in `tools` only. C1 and C2 (risk floors, project rules) run in shadow.
- Across turns: S1 (a checker gate stuck), S2 (state files not read after a compaction) and S4 (20 minutes without progress) act in `overnight`. S5 runs on every client and notes a problem stated three times, or a build turn with nothing written after three minutes.

The tool-argument loop guard stops a tool call whose arguments repeat. In one case the model wrote the same 97-character row of a tile map 12,866 times: 65,536 tokens over 327 s, and nothing was written. The guard now stops such a call within about 2,000 tokens of the loop's start. When the client sends the stopped call back, the history keeps its first 1,500 characters and one marker. After a repeat, loop escape (SRV-475) samples the next turn with the model card's settings until the session makes progress.

### The Chat tab's run monitor

chatd, the service behind the Chat tab, runs its own agent loop and its own limits ([chat.md](chat.md)). A run ends `interrupted` after 100 model calls, and the same call with the same arguments three times in a row ends it too. The stall rule (SRV-677, [chatd/stall.py](../chatd/stall.py)) catches a run that edits the wrong place. A check is a `render_page` call or a shell command the run has already run once. A check makes progress when its target is new, when its text changed (numbers are masked), or when its screenshot changed in kind (a 32 × 18 thumbnail, correlation under 0.3). The rule fires on a streak of at least 4 checks with no progress, at least max(12, 3 × the run's median gap) steps long, with at least 3 edits and 60 % of them on three files or fewer.

It acts twice. First, a note in the tool result tells the model to step back, list its hypotheses and test the cheapest one with a minimal repro. If the streak goes on as long again, a second note follows and the run ends `interrupted`; Continue goes on from there. The note rides in the tool result, so the prompt prefix stays the same. In the case behind it, a three.js page rendered near black, and 37 steps changed the lighting and the exposure in three files and rendered the page seven times. The cause was the materials' colours, and the same-call guard never fired because no call repeated.

## The journal and the dashboard

The router and the monitor write one journal: the table `conductor_events` in the usage ledger's file, kept 30 days. A request under the conductor writes one router row, and each monitor decision writes its own row. The journal stores each hidden text once, by its hash. `GET /v1/conductor/sessions/{key}/export?format=md` lists what the model saw that the client did not. Prometheus counts every decision and act (`qse_conductor_*`).

The configuration is a set of versioned JSON documents outside any repository: profiles, router, roles, packages, settings, and the monitor's policy. The dashboard's Conductor view shows the live decision feed and edits every document. The view checks a save and shows its diff; the save then becomes a new version with a note, and the next request uses it with no restart. A profile can also show the user what the conductor does, as short status lines in the reasoning or the content (`ui.states`).

## Raw mode for benchmarks

`X-Tandem: raw`, or the locked profile `raw` or `bench`, serves the release's path byte for byte. The one difference is where a plain API client's reasoning goes: to `reasoning_content` (SRV-617). Benchmarks that measure the engine itself send this header. `tools/row3.py --no-server` refuses a server that lists `TandemLLM`. [benchmarks.md](benchmarks.md) says which settings a score must name.

## Measured effect

[tool-eval-bench.md](tool-eval-bench.md) has every row; each row is one run, and run-to-run noise is about 2 points. On Qwen3.8-27B (EXL3 6.0, 2026-10-19, 69 scenarios):

| row | score | time |
|-|-|-|
| raw, `xhigh` | 97 | 12.4 min |
| conductor, `tools` | 95 | 15.0 min |
| `bench-nopk`: turn class and effort, no packages | 96 | 13.5 min |
| `bench-pk`: turn class and effort, with packages | 94 | 26.0 min |
| conductor, `chat` | 91 | 28.0 min |

- Without packages the conductor scores the same as raw. Packages double the time and add no points on this model.
- In hard mode (28 scenarios) `tools` scored 91 against 84 for raw. On the 0.12.2 build the next day, raw and the conductor without a profile header both scored 86, and `tools` 82 twice. Which part of the conductor makes the difference is not tested.
- On Ornith-1.5-122B (0.13.0, 2026-10-21), `tools` scored 96 against 93 for raw `xhigh`, and the conductor 84 against 80 in hard mode. There `bench-pk` scored 97 against 94 for `bench-nopk`, in 20.1 min against 14.1. One run each; this needs a second run before it says anything about packages.

## Where to read more

| topic | page |
|-|-|
| profiles, the router, classes, packages in the plan, side requests, compaction, sessions | [server.md](server.md#tandemllm-and-the-conductor) |
| a client's own thinking budget and effort word | [server.md](server.md#thinking-budgets-from-the-client) |
| status lines shown to the user | [server.md](server.md#tandemllms-steps-shown-to-the-user-uistates) |
| the package library and its search | [server.md](server.md#the-package-library-r1-2026-10-03), [packages.md](packages.md) |
| session addresses | [server.md](server.md#session-addresses-0323-test) |
| the session monitor, safe points, policy | [server.md](server.md#the-conductors-session-monitor) |
| tool-call checks, the claim check, the loop guard, session tasks | [server.md](server.md#tool-call-checks-srv-113), [the claim check](server.md#the-claim-check-e2-2026-10-02), [the loop guard](server.md#the-tool-argument-loop-guard-0326-test), [session tasks](server.md#session-level-tasks-srv-121) |
| the journal and metrics | [server.md](server.md#observability-srv-115), [server/METRICS.md](../server/METRICS.md) |
| compaction as a continuation | [caches.md](caches.md#compaction-as-a-continuation) |
| the Chat tab and chatd | [chat.md](chat.md) |
| memory notes | [memory.md](memory.md) |
| the dashboard's Conductor view | [dashboard.md](dashboard.md#views) |
| the `/v1/conductor/*` API | [contract/conductor-v1](contract/conductor-v1/README.md) |
| benchmarks, raw and conductor | [benchmarks.md](benchmarks.md), [tool-eval-bench.md](tool-eval-bench.md) |
| the manual's chapters: profiles and the router, packages, the session monitor | [`tandemllm-manual.pdf`](../manual/tandemllm-manual.pdf) |
