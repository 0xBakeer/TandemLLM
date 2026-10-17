# Chat tab: data and event contract (v0.2, 2026-10-11)

Owner: the Chat architect. The designer builds against this file; the backend (chatd, SRV-343 and on)
implements it. The full plan lives in Memo (ws inference-engine, note "Chat tab: plan and architecture
(2026-10-11)", epic VIS-E9). This file is the one repo copy of the contract. Change it here first,
then in Memo (note "Chat 9: architecture and API contract").

v0.2 adopts the names in `types.ts` and `backend.ts` (the designer's first pass) wherever they mean the
same as v0.1, and adds what the server needs. Section 7 lists what changed and the few edits the
client side still needs. Section 8 is for open requests from either side.

Phases: `MVP` ships in phase 1, `v2` in phases 2-3, `later` after that. Mock all of it; hide what a
phase does not ship.

## 0. Shape of the system

```
browser (Chat tab, #/chat)  -- https -->  engine server :8000  -- forwards /v1/chat-app/* -->  chatd 127.0.0.1:8016
                                           (auth: the one admin switch)                     (torch-free Python process)
                                                                                                |  SQLite ~/.tandemllm/chat/chat.sqlite3
                                                                                                |  agent loop  -- POST /v1/chat/completions --> engine (or a proxy backend)
                                                                                                |  sandboxes   -- podman API (user tandem-sbx) --> one container per chat
                                                                                                |  MCP clients -- stdio (inside a sandbox) / streamable HTTP
```

- A chat's run lives in chatd, not in the browser. Closing every tab does not stop it.
- Every client is a view: it loads the conversation (a snapshot), then tails its event stream from the
  snapshot's `seq`.
- All paths below are relative to `/v1/chat-app`. The prefix is NOT `/v1/chat/`: that namespace is the
  OpenAI API (`/v1/chat/completions`), and the forward must never touch it.
- JSON bodies, UTF-8. Ids are strings (ULIDs on the server). Times are ISO 8601 strings in UTC
  (`*_at`). Token counts are integers; byte counts are `*_bytes`.

## 1. Data model

```ts
// ---- conversations
type ConversationStatus = 'idle' | 'running' | 'waiting' | 'error';   // waiting = a tool asks for permission

interface ConversationSummary {
  id: string; title: string; created_at: string; updated_at: string;
  pinned: boolean; archived: boolean;            // archived: v0.2 addition
  folder_id: string | null;
  status: ConversationStatus;
  preview: string;                               // the last message's first line
  model: string; profile: string | null;         // profile = conductor profile, null = the server default
  turns: number;
  now?: string | null;                           // the server's words for what the run does now
  version: number;                               // v0.2: optimistic concurrency for PATCH
  unread: boolean;                               // v0.2: an answer finished since this device last looked (per device, kept in the browser)
}
interface Folder { id: string; name: string }

interface Conversation extends ConversationSummary {
  messages: Message[];                           // the active branch, root to head, in order
  head_id: string | null;                        // v0.2: the leaf of the active branch
  settings: ChatSettings;
  run: Run | null;
  context: ContextReport;
  sandbox: SandboxStatus;
  compactions: Compaction[];
  usage: ConversationUsage;
  queue: QueuedMessage[];                        // messages typed while a run is on (the steer queue)
  ui: ConversationUi;                            // v0.2: pane layout, shared across devices
  seq: string;                                   // v0.2: the last event id included in this snapshot
}
interface QueuedMessage { id: string; text: string; created_at: string; client_msg_id: string }

interface ConversationUi {                       // v0.2
  panes: PaneRef[];                              // tab order; the phone shows them as bottom-sheet tabs
  active: string | null;                         // a PaneRef id
  split: number;                                 // desktop: the thread's width fraction, 0.35..0.75
}

// ---- messages
type Role = 'user' | 'assistant' | 'system';     // system = notices (compaction divider, interrupted, cancelled)
type MessageStatus = 'done' | 'streaming' | 'stopped' | 'error' | 'interrupted';   // interrupted: v0.2 (chatd or engine restart; Continue resumes)

interface Message {
  id: string; role: Role; created_at: string;
  parent_id: string | null;                      // v0.2: the tree behind branches
  run_id: string | null;                         // v0.2: the run that wrote an assistant message
  parts: Part[];
  status: MessageStatus;
  attachments?: Attachment[];
  siblings?: { index: number; count: number };   // "2 / 3"
  stats?: TurnStats;
}

type Part = TextPart | ThinkingPart | ToolCallPart | ErrorPart | ArtifactPart;
interface TextPart { type: 'text'; text: string }
interface ThinkingPart { type: 'thinking'; text: string; tokens: number | null; done: boolean; duration_ms?: number | null }
type ToolStatus = 'asking' | 'running' | 'done' | 'failed' | 'denied' | 'cancelled';   // cancelled: v0.2
interface ToolCallPart {
  type: 'tool_call';
  id: string;                                    // the call id
  name: string;                                  // MCP tools: "mcp__<server>__<tool>"
  source?: ToolSource;                           // v0.2
  risk?: 'read' | 'write' | 'exec' | 'network';  // v0.2
  input: string;                                 // JSON text, growing while the model writes it
  output: { content: string; is_error: boolean; bytes: number; truncated?: boolean; exit_code?: number | null; images?: Attachment[];
            data?: WebSearchData | { sandbox: SandboxCallData } | null } | null;   // data: web_search's results; why the sandbox could not run the call
  status: ToolStatus;
  started_at: string | null; ended_at: string | null;
  permission?: { summary: string; suggested_rule?: { tool: string; match?: string } } | null;   // suggested_rule: v0.2
  pane?: PaneRef | null;
  harness?: 'verify';                            // SRV-532: a call the harness made itself (source {kind: 'harness'})
}
interface ErrorPart { type: 'error'; message: string; retryable: boolean }
interface VerifyPart { type: 'verify'; state: 'passed' | 'problems' | 'not_run' | 'stale'; text: string;
                       checked?: string[]; not_checked?: string[] }   // SRV-532: the end of an answer that built an app
interface ArtifactPart { type: 'artifact'; id: string; title: string; mime: string; version: number }   // v2

interface Attachment {
  id: string; name: string; kind: 'image' | 'text' | 'pdf' | 'file';
  bytes: number; mime: string;
  preview?: string | null;                       // a small image's data URL
  url?: string;                                  // v0.2: GET /attachments/{id}
  tokens?: number | null;                        // v0.2: its cost in context once read
  inline?: boolean;                              // a text file: in the prompt whole; false: one line with its path (SRV-487)
  path?: string;                                 // where the send saved it: /workspace/uploads/<name>
}

interface TurnStats {
  finish: 'stop' | 'length' | 'tool_calls' | 'stopped' | 'error' | 'timeout' | string;
  ttft_ms: number | null; decode_tps: number | null; total_ms: number | null;
  prompt_tokens: number; completion_tokens: number; reasoning_tokens: number; cached_tokens: number;
  tokens_per_round: number | null; acceptance: number | null;
  cache_source?: CacheSource | string;           // v0.2: resident, end, park, spill, disk, none ... (the engine's cache_source); `provider` for a relayed model (0.11.3)
  prefill_tps?: number | null;                   // v0.2
  cost_usd?: number | null;                      // v0.2: cloud backends only; 0.11.3: the proxy's price of every call (`: tandem-cost`)
}

// ---- the run
type RunPhase = 'queued' | 'prefill' | 'thinking' | 'writing' | 'tool' | 'permission' | 'verifying' | 'finishing';   // verifying: SRV-532
interface Run {
  id: string; conversation_id: string; message_id: string; started_at: string;
  started_by: string | null;                     // a device label when it was not this browser
  phase: RunPhase; now: string;
  step: number; max_steps: number;               // v0.2: model calls so far, the cap (default 100)
  queue_position: number | null;                 // v0.2: place in the engine's line while phase = queued
  tool: { call_id: string; name: string } | null;
  prefill: { done: number; total: number; tps?: number | null; eta_ms?: number | null } | null;
  decode_tps: number | null; tokens_per_round: number | null; acceptance: number | null;
  ttft_ms?: number | null;                       // 0.11.3: this model call's first token, once it came
  thinking: { used: number; budget: number | null };
  tokens: number;
  tps_history: (number | null)[];                // one-second decode samples, oldest first, at most 120
}

// ---- the Context panel
type ContextPartKind = 'system' | 'tools' | 'skills' | 'files' | 'images' | 'messages' | 'thinking' | 'reserved';  // images: the engine's split (VIS-205)
interface ContextPart { kind: ContextPartKind; label: string; tokens: number }
interface ContextReport {
  window: number; used: number;
  parts: ContextPart[];                          // sums to `used`
  compact_at: number;                            // v0.2: tokens; chatd compacts above this (0.8 x window)
  cache: {
    warm: number;                                // tokens the next request finds cached
    cold: number;                                // tokens it would re-read
    cold_eta_ms: number | null;                  // v0.2: the re-read estimate
    source: CacheSource | string;                // where the cached part lives
    slot: string | null;
  };
  harness: { class: string | null; effort: string | null; packages: { name: string; tokens: number | null }[];
             memory?: { mode: 'on' | 'shadow' | 'off'; why: string | null; index_notes: number; index_tokens: number;
                        bodies: number; body_tokens: number; card_tokens: number } | null } | null;   // v0.2; tokens null: not counted yet; memory: the notes' part of the turn (why it is off: profile, unset, model_fit, backend, header); an older engine sends none
  memory: {
    kv_bytes: number | null; resident_bytes: number | null; sandbox_disk_bytes: number | null;
    kv_tokens?: number | null;                   // the chat's rows the engine holds (kv_where slot or park)
    kv_where?: 'slot' | 'park' | 'nvme_spill' | 'disk_tier' | 'ram_snapshot' | 'other' | 'none' | null;   // null: no cache report
    resident_owner?: 'this' | 'other' | 'none' | null;   // whose rows the resident prefix holds
    available_bytes?: number | null;             // v0.2: MemAvailable
    floor_bytes?: number | null;                 // v0.2: 15 GiB, the engine's admission floor
    sandbox_bytes?: number | null;               // v0.2: RAM the sandboxes use
  };
  model: {
    name: string; profile: string | null; weights: string; drafter: string | null;
    max_len: number; effort: Effort | 'auto'; thinking: boolean;
    backend?: 'engine' | 'proxy' | 'cloud';      // v0.2
  };
}
interface Compaction { id: string; at: string; before_tokens: number; after_tokens: number; summary: string; duration_ms?: number; boundary_id?: string }   // boundary_id: the first message the model still reads in full; the thread's divider sits above it
interface ConversationUsage {
  turns: number; prompt_tokens: number; completion_tokens: number; reasoning_tokens: number;
  cached_tokens: number; tool_calls: number; wall_s: number; cost_usd?: number | null;   // tool_calls: every call on the branch
}

// ---- settings
type Effort = 'low' | 'medium' | 'high' | 'xhigh';
interface ChatSettings {
  model: string; profile: string | null;          // VIS-203: always null; the Chat tab runs under the conductor's chat-app profile (client kind)
  effort: Effort | 'auto';                       // v0.2: 'auto' (the default) = the harness decides per turn
  thinking: boolean;                             // false = thinking off whatever the effort
  tools: string[];                               // tool ids on for this chat
  packages: string[];                            // package/skill ids pinned on for this chat
  packages_blocked: string[];                    // v0.2: never injected in this chat
  packages_mode: 'auto' | 'manual' | 'model';    // v0.2: auto = harness picks (plus pins); manual = pins only; model = the model reads skills itself
  mcp: string[];                                 // MCP server ids on for this chat
  prompt_id: string | null;                      // the system prompt (a saved prompt), null = default
  policies: Record<string, ToolPolicy>;          // per-tool overrides for this chat
  auto_accept_edits: boolean;                    // v0.2: write tools run without asking
  max_steps: number;                             // v0.2: default 100
  sandbox_network: 'off' | 'on';                 // v0.2: 'on' = internet without the LAN; see SandboxStatus.network
  verify?: boolean;                              // SRV-532: check a page or app the run built before it ends (default true)
}
type ToolPolicy = 'ask' | 'always' | 'never';
type ToolSource = { kind: 'builtin' } | { kind: 'package'; id: string } | { kind: 'mcp'; id: string } | { kind: 'user'; id: string };
interface ToolInfo {
  id: string; name: string; description: string; source: ToolSource; enabled: boolean; policy: ToolPolicy;
  risk?: 'read' | 'write' | 'exec' | 'network';  // v0.2
  schema_tokens?: number;                        // v0.2: what its schema costs in every request
}
interface PackageInfo { id: string; name: string; description: string; kind: 'skill' | 'package'; version: string; enabled: boolean; tools: string[]; tokens?: number; user?: boolean }   // user: uploaded through /packages/upload
interface UserToolkit { id: string; slug: string; title: string; description: string; version: string; requirements: string[]; enabled: boolean; tools: string[]; valves: {name, type, default, description, secret, value, set}[]; created_at: string }
type McpStatus = 'connected' | 'starting' | 'failed' | 'stopped';
interface McpServer {
  id: string; name: string; transport: 'stdio' | 'http';
  command?: string; args?: string[]; url?: string;
  env?: Record<string, string>;                  // v0.2: values are WRITE-ONLY; GET returns every value as ""
  headers?: Record<string, string>;              // same rule
  status: McpStatus; error: string | null;
  tools: { name: string; description: string }[];
  tool_allow?: string[] | null; tool_deny?: string[];   // v0.2
  schema_tokens?: number;                        // v0.2
  enabled_global?: boolean;                      // v0.2
  updated_at: string;
}
type McpServerDraft = Omit<McpServer, 'id' | 'status' | 'error' | 'tools' | 'updated_at'>;
interface Prompt { id: string; title: string; command: string; body: string }   // body variables: {{CLIPBOARD}} {{selection}} {{input}}
interface Preset { id: string; name: string; model: string; profile: string | null; effort: Effort | 'auto'; thinking: boolean; tools: string[]; prompt_id: string | null; packages?: string[]; mcp?: string[] }
interface SlashCommand { command: string; description: string; prompt_id?: string; mcp_prompt?: { server_id: string; name: string } }
interface Capabilities {
  models: {
    id: string; name: string; weights: string; max_len: number;
    profiles: { id: string; name: string; description: string }[];                 // VIS-203: always []; the picker lists models only
    backend?: 'engine' | 'proxy' | 'cloud';                                         // v0.2
    capabilities?: { thinking: boolean; effort_levels: string[]; effort_default?: string | null; tools: boolean; vision: boolean; structured: boolean };   // v0.2, from the model card; 0.11.3: a cloud model's levels and default from its provider's model list (DeepSeek: low, high, max; high)
    loaded?: boolean;
  }[];
  efforts: (Effort | 'auto')[];
  commands: SlashCommand[];
  defaults: ChatSettings;
  session: { password_set: boolean; signed_in: boolean; sandbox: 'ready' | 'not_installed' | 'error' };   // v0.2
}

// ---- the sandbox (v2)
// a failed call's output.data when the sandbox could not run it: no start within its wait (60 s), or a process the kernel killed at the limit
interface SandboxCallData { reason: 'no_memory' | 'busy' | 'oom'; limit_gib: number; why?: string; waited_s?: number; mem_available_gib?: number; floor_gib?: number; engine_released_gib?: number }
type SandboxState = 'running' | 'stopped' | 'creating' | 'error' | 'waiting_memory';   // waiting_memory: v0.2
interface SandboxStatus {
  state: SandboxState; image: string; cwd: string;            // cwd = /workspace
  cpus: number; memory_bytes: number | null; memory_limit_bytes: number; disk_bytes: number; disk_limit_bytes: number;   // memory_bytes: in use while running, null when the runtime does not say
  network: 'off' | 'allowlist' | 'on';           // 'on' = internet, never the LAN; 'allowlist' is later (mock it disabled)
  allowlist: string[];
  uptime_s: number; error: string | null;
  stopped_reason?: 'idle' | 'memory_guard' | 'user' | null;   // v0.2
  ports?: { port: number; label?: string }[];                  // v0.2: listening ports, for the preview
}
interface SandboxFile { path: string; kind: 'file' | 'dir'; bytes: number; modified_at?: string; etag?: string }

// ---- panes
type PaneKind = 'terminal' | 'preview' | 'files' | 'editor' | 'diff' | 'markdown' | 'image' | 'pdf' | 'video' | 'audio' | 'binary' | 'context' | 'plan';   // v0.2 adds editor, pdf, context, plan; video, audio, binary: a player or a size and Download
interface PaneRef {
  id?: string;                                   // v0.2: set by the server when the pane is in the layout
  kind: PaneKind;
  ref: string;                                   // a terminal session, a file path, an artifact id, a run id (diff), "" (context)
  title?: string;
  pinned?: boolean;
}
```

## 2. Endpoints

Auth: the dashboard's one switch (`QSE_ADMIN_TOKEN`). No password set (today's default, the project's rule
of 2026-10-06): every route answers anyone on the trusted network and the UI shows no sign-in. Password
set: every route needs the `qse_dash` session cookie or the admin bearer token, and the Chat tab shows a
sign-in screen (only then). Routes that run code do so only inside the chat's sandbox (section 5); they
answer 503 `sandbox_unavailable` when the sandbox is not installed.

`{id}` is always a conversation id; the sandbox, terminals and artifacts hang under it.

| Method | Path | Body | Answer | `ChatBackend` method | Phase |
|---|---|---|---|---|---|
| GET | `/capabilities` | | `Capabilities` (also the probe: 200 means a server) | `capabilities()` | MVP |
| POST | `/session` | `{password}` | sets the cookie; 401, or 429 after 5 tries a minute | (sign-in screen) | MVP |
| DELETE | `/session` | | signs this device out | | MVP |
| GET | `/conversations` | `?q=&archived=0&folder=` | `{conversations: ConversationSummary[], folders: Folder[]}` | `listConversations()` | MVP |
| POST | `/conversations` | `{settings?: Partial<ChatSettings>, preset_id?}` | `Conversation` | `createConversation()` | MVP |
| GET | `/conversations/{id}` | | `Conversation` (the snapshot, with `seq`) | `getConversation()` | MVP |
| PATCH | `/conversations/{id}` | `{version?, title?, pinned?, archived?, folder_id?, ui?}` | `ConversationSummary`; 409 `version_conflict` with the current one | `updateConversation()` | MVP |
| DELETE | `/conversations/{id}` | | 204 (trash for 7 days, with its sandbox) | `deleteConversation()` | MVP |
| POST | `/folders` | `{name}` | `Folder` | `createFolder()` | MVP |
| GET | `/conversations/{id}/events` | `Last-Event-ID` header or `?after=seq` | SSE (section 3) | `subscribe()` | MVP |
| POST | `/conversations/{id}/messages` | `{client_msg_id, text, attachments?: string[] (ids), parent_id?}` | 202 `{message_id, run_id}`; a repeat of `client_msg_id` returns the first answer | `send()` | MVP |
| POST | `/conversations/{id}/queue` | `{client_msg_id, text}` | 202 `QueuedMessage` (goes in after the current step) | `queue()` | MVP |
| DELETE | `/conversations/{id}/queue/{qid}` | | 204 | `unqueue()` | MVP |
| POST | `/conversations/{id}/queue/{qid}/now` | | 202: stops the current generation, keeps its text, sends this one now | (v0.2: "Send now") | MVP |
| POST | `/conversations/{id}/stop` | | 202; the message keeps its text with status `stopped` | `stop()` | MVP |
| POST | `/conversations/{id}/continue` | | 202; resumes an `interrupted` run | (v0.2) | MVP |
| POST | `/conversations/{id}/messages/{mid}/regenerate` | `{model?, effort?}` | 202 | `regenerate()` | MVP |
| POST | `/conversations/{id}/messages/{mid}/edit` | `{client_msg_id, text}` | 202 | `edit()` | MVP |
| POST | `/conversations/{id}/branch` | `{message_id, index}` | `Conversation` (head moved) | `switchBranch()` | MVP |
| PATCH | `/conversations/{id}/settings` | `Partial<ChatSettings>` | `ChatSettings` | `updateSettings()` | MVP |
| POST | `/conversations/{id}/compact` | | 202 | `compact()` | MVP |
| POST | `/conversations/{id}/plan/approve` | `{message_id?, items?: string[]}` | 202 `{todos, message_id, run_id}`: the plan's steps (or `items`) become the todo list, plan mode goes off, "Carry out the approved plan." is sent | `approvePlan()` | v2 |
| GET / PUT | `/conversations/{id}/todos` | `{items: [{id?, text, state}]}` | `TodoItem[]`; every device gets `todos.changed` | | v2 |
| POST | `/conversations/{id}/rewind` | `{message_id}` (a user message on the branch) | `{conversation, text, files}`: the run that is on stops, the files go back to before the first run after that message, the head moves to its parent; nothing is deleted | `rewind()` | v2 |
| POST | `/conversations/{id}/permissions/{call_id}` | `{decision: 'once' \| 'always' \| 'deny', scope?: 'chat' \| 'all', note?}` | 204; 409 when another device answered first | `permission()` | v2 |
| GET | `/events` | `Last-Event-ID` | SSE, global: list changes, run states, permission asks in any chat | (v0.2: badges, notifications) | MVP |
| POST | `/attachments` | multipart `file` (max 50 MB) | `Attachment` | (v0.2) | MVP |
| GET | `/attachments/{att}` | | bytes | | MVP |
| GET | `/tools` / PATCH `/tools/{tool}` | `{enabled?, policy?}` | `ToolInfo[]` / `ToolInfo` | `tools()`, `setTool()` | v2 |
| POST | `/tools/import/owui` | multipart `file`: a Tools `.py` or Open WebUI's JSON export | `{toolkit: UserToolkit, toolkits: UserToolkit[], tools: ToolInfo[], valves}`; 400 `bad_tools_file` (a filter, pipe or action; no `class Tools`) | `importOwui()` | v2 |
| GET | `/tools/user` | | `UserToolkit[]` (a secret valve's value is `""`, `set` says whether one is stored) | `userToolkits()` | v2 |
| PATCH / DELETE | `/tools/user/{tid}` | `{enabled?, valves?: {NAME: value}}` (an empty secret keeps the stored one) | `UserToolkit` / 204 | `patchUserToolkit()`, `removeUserToolkit()` | v2 |
| GET | `/packages` / PATCH `/packages/{pkg}` | `{enabled}` | `PackageInfo[]` / `PackageInfo` | `packages()`, `setPackage()` | v2 |
| POST | `/packages/upload` | multipart `file`: a `.zip` with a SKILL.md folder | `PackageInfo` (`user: true`); 400 `bad_skill` with the reason, 409 when the engine has the name from elsewhere | `uploadSkill()` | v2 |
| DELETE | `/packages/{pkg}` | | 204; 404 for a package that came with the engine (switch those off instead) | `removeSkill()` | v2 |
| GET / POST | `/mcp/servers` | `McpServerDraft` | `McpServer[]` / `McpServer` | `mcpServers()`, `addMcpServer()` | v2 |
| PATCH / DELETE | `/mcp/servers/{sid}` | `Partial<McpServerDraft>` | `McpServer` / 204 | `removeMcpServer()` | v2 |
| POST | `/mcp/servers/test` | `McpServerDraft` | `{ok, message, tools}` | `testMcpServer()` | v2 |
| POST | `/mcp/servers/{sid}/restart` | | `McpServer` | `restartMcpServer()` | v2 |
| POST | `/mcp/servers/{sid}/prompts/get` | `{name, arguments, conversation_id?}` | `{text}`: the prompt's messages as one text for the composer (an MCP prompt is the slash command `mcp__<server>__<prompt>`, `SlashCommand.mcp_prompt`) | `mcpPrompt()` | v2 |
| POST | `/mcp/servers/{sid}/resources/read` | `{uri, conversation_id?}` | `Attachment` (stored; it goes into a send's attachments as it is) | `mcpResource()` | v2 |
| GET / POST / DELETE | `/prompts[/{pid}]` | `Prompt` | | `prompts()`, `savePrompt()`, `deletePrompt()` | v2 |
| GET / POST / DELETE | `/presets[/{pid}]` | `Preset` | | `presets()`, `savePreset()`, `deletePreset()` | v2 |
| GET / PUT | `/web-search` | `WebSearchPatch` (partial; `searxng_auth_value: null` clears it) | `WebSearchView`; 400 with `field`; the auth value never comes back (`searxng_auth_set`) | `webSearch()`, `saveWebSearch()` | v2 |
| POST | `/web-search/test` | `{settings?: WebSearchPatch, query?}` | `WebSearchTest` `{ok, provider, message, count, ms, results}`; nothing is stored | `testWebSearch()` | v2 |
| GET / DELETE | `/permissions/rules[/{rid}]` | | `{id, tool, match?, decision, scope, created_at}[]` | (v0.2) | v2 |
| GET | `/conversations/{id}/sandbox` | | `SandboxStatus` | `sandbox()` | v2 |
| POST | `/conversations/{id}/sandbox/reset` / `start` / `stop` | | `SandboxStatus` | `resetSandbox()` | v2 |
| GET | `/conversations/{id}/sandbox/files` | `?path=/workspace` | `SandboxFile[]` | `sandboxFiles()` | v2 |
| GET | `/conversations/{id}/sandbox/file` | `?path=` | `{content, mime, bytes, etag, kind, url}` (`content` null for a binary or a file over 2 MiB; `kind` the pane kind: file, binary, image, preview, markdown, video, audio, pdf; `url` the view route) or the whole file with `?raw=1` (streamed, byte ranges) | `readFile()` | v2 |
| GET | `/conversations/{id}/sandbox/view/<path>` | `Range: bytes=` | the file, streamed: 200 whole, 206 one range (`Content-Range`), 416 past the end; `Accept-Ranges: bytes`, `ETag`, `Last-Modified`, `If-Range`, `If-None-Match`; the preview CSP for pages, images and text, one without scripts for media, PDFs and binaries | (preview, players) | v2 |
| PUT | `/conversations/{id}/sandbox/file` | `?path=`, body, `If-Match: etag` | 204; 412 when the file changed | (v0.2) | v2 |
| POST | `/conversations/{id}/sandbox/upload` | `?dir=`, multipart | `SandboxFile[]` | (v0.2) | v2 |
| GET | `/conversations/{id}/sandbox/diff` | `?run_id=` | `{files: {path, status, additions, deletions, patch}[]}` | (v0.2, diff pane) | v2 |
| POST | `/conversations/{id}/sandbox/undo` | `{run_id}` | 202 | (v0.2) | v2 |
| GET | `/conversations/{id}/sandbox/ports/{port}/...` | | reverse proxy into the sandbox (preview pane); `Access-Control-Allow-Origin: null` for the preview's opaque origin, a page with the preview's shim (SRV-511) | (v0.2) | v2 |
| GET | `/conversations/{id}/sandbox/ports` | | `{ports: {port, addresses, command, cwd, started, http}[]}`: what listens in the sandbox, `[]` when it is stopped (SRV-511) | `sandboxPorts()` | v2 |
| POST | `/conversations/{id}/sandbox/serve` | `{dir}` | 202 `{dir, command, log}`: the project's dev server started in the background (the command from its package.json; `npm install` first without node_modules); 400 `no_dev_server` | `serve()` | v2 |
| POST | `/conversations/{id}/preview/report` | `{ref, url, title, errors, failed, elements, text, canvas}` | 204: what the preview pane's page loaded; `present` waits up to 15 s for it | `previewReport()` | v2 |
| GET | `/conversations/{id}/artifacts/{ref}` | `?version=` | `{kind, content, mime, title, url?, bytes?}`; a video, audio, PDF or binary file: `content` "", `url` the view route; `?raw=1` serves it with the preview CSP | `artifact()` | v2 |
| POST / GET | `/conversations/{id}/terminals` | | `{session, title, owner: 'user' \| 'agent', alive}` | (v0.2) | v2 |
| WS | `/conversations/{id}/terminals/{session}/ws` | | binary frames = PTY bytes; text frames = JSON `{type:'resize', cols, rows}`; on attach the server sends the scrollback (max 1 MiB) first | `terminal()` | v2 |

## 3. The event stream

SSE frames: `id: <seq>`, `event: <type>`, `data: <the ChatEvent as JSON>`. The client wraps each as
`StampedEvent {id, event}`. A connection with `Last-Event-ID` (or `?after=`) first replays what it
missed, then sends `event: current` (no id, `data: {}`), then live events. A client more than 2,000
events behind gets `event: reset` and reloads the conversation. Heartbeat: a comment `: hb` every 15 s.

Persisted events carry an `id` and are replayed. Ephemeral events (marked E) carry no `id`, are never
replayed, and the snapshot holds their latest value. Text deltas are coalesced to at most one
`part.delta` per message every 50 ms.

```ts
type ChatEvent =
  | { type: 'run.started'; run: Run }
  | { type: 'run.phase'; run_id: string; phase: RunPhase; now: string; tool?: Run['tool']; prefill?: Run['prefill']; step?: number; queue_position?: number | null }
  | { type: 'run.metrics'; run_id: string; decode_tps: number | null; tokens_per_round: number | null; acceptance: number | null; tokens: number; thinking_used: number; thinking_budget?: number | null; ttft_ms?: number | null; prefill?: Run['prefill'] }   // E, at most 4 a second
  | { type: 'run.done'; run_id: string; message_id: string; stats: TurnStats; status: MessageStatus }
  | { type: 'run.error'; run_id: string; message_id: string; message: string; retryable: boolean }
  | { type: 'message.created'; message: Message }
  | { type: 'part.added'; message_id: string; part: Part }
  | { type: 'part.delta'; message_id: string; index: number; delta: string; field?: 'text' | 'input' }
  | { type: 'part.done'; message_id: string; index: number; tokens?: number | null }
  | { type: 'tool.asking'; message_id: string; call_id: string; summary: string; risk?: string; suggested_rule?: { tool: string; match?: string } }
  | { type: 'tool.started'; message_id: string; call_id: string; pane?: PaneRef | null }
  | { type: 'tool.output'; message_id: string; call_id: string; delta: string }   // live stdout/stderr, at most 64 KiB per call in events
  | { type: 'tool.done'; message_id: string; call_id: string; status: 'done' | 'failed' | 'denied' | 'cancelled'; is_error: boolean; output?: ToolCallPart['output'] }
  | { type: 'queue.changed'; queue: QueuedMessage[] }
  | { type: 'context.changed'; context: ContextReport }   // E, at most 1 a second while running, every 10 s otherwise
  | { type: 'compaction'; compaction: Compaction }
  | { type: 'todos.changed'; todos: TodoItem[] }       // SRV-363
  | { type: 'conversation.updated'; patch: Partial<ConversationSummary> }
  | { type: 'branch.changed'; head_id: string }            // v0.2: another device switched or made a branch; reload the conversation
  | { type: 'ui.changed'; ui: ConversationUi }             // v0.2: pane layout changed (any device)
  | { type: 'pane.open'; pane: PaneRef }                   // v0.2: the agent asks to show something
  | { type: 'artifact.created'; artifact: ArtifactPart }   // v0.2, v2
  | { type: 'sandbox.changed'; sandbox: SandboxStatus }
  | { type: 'usage.changed'; usage: ConversationUsage }
  | { type: 'settings.changed'; settings: ChatSettings };
```

Rules:
- A permission prompt closes on any later `tool.*` event for its `call_id` (another device answered).
- `run.done` with status `interrupted`: the run stopped before its end and the UI offers Continue. `stats.interrupted` says why: `limit` (its limit of model calls) or `sandbox` (the same sandbox failure three calls in a row), both with the notice as the message's last part; `engine` (the engine did not answer for the retry window); `restart` (chatd restarted during the run). Absent on answers from before 0.11.2 (VIS-197).
- The global stream (`/events`) sends `conversation.updated` (with `id` in the patch), `conversation.created`
  `{conversation: ConversationSummary}`, `conversation.deleted` `{id}`, `run.started`, `run.done`,
  `tool.asking` (with `conversation_id`), each persisted with its own sequence.

## 4. The Context panel

`Conversation.context` in the snapshot; `context.changed` while it changes. Sources (for the backend):
`window` and `model.*` from the model card (`/v1/models`, SRV-347); `parts` and `harness` from the
usage frame's `tandem` object (SRV-349) plus chatd's own counts; `cache` from the engine's cache peek
(before a send) and `cache_source` (after one); live run figures from `/v1/dashboard/live`, matched by
conversation id; `memory` from `/v1/dashboard/system` (the resident report's `owner_tag` names the
chat in the slot) and the sandbox runtime's stats. The `files` part is what the file tools carried
(a write's content, an edit's old and new text, a read's result) plus the attachments, estimated,
from the first message a compaction kept. Live rates follow the dashboard's NOW rule: `null` when
idle; decode rates skip the first 50 tokens. A figure without a value says why in words ("measured
after the first turn", "not tracked for this model"), never a bare dash.

A model the proxy relays (`model.backend` `cloud` or `proxy`, 0.11.3) has no live row with counts:
chatd measures its stream itself. Live: tokens per text chunk (one token each on DeepSeek), a tool
call's arguments at 3.5 characters a token, thinking from `reasoning_content`, the first token's
wait from the send. After each call: the provider's usage (DeepSeek's `prompt_cache_hit_tokens` as
the cached tokens, `cache.source` `provider`), tok/s = (completion tokens - 1) over the stream's
span, and the proxy's price as `cost_usd`. A cloud model's `split` is chatd's estimate from the
request's characters at the chat's learned tokens per character (`estimated: true`, the skill
index apart as `skills`); the panel marks those parts "est.". The `reserved` part is what a request
asks for: the model's output limit, or the window's room when less is left.

## 5. Rules the UI can rely on

- Harness information (class, effort, injected packages) arrives in `context.harness`, never inside the
  answer text.
- The preview always uses `<iframe sandbox="allow-scripts allow-forms allow-popups allow-modals">`, never
  with `allow-same-origin`. Artifact (`?raw=1`) and port answers carry
  `Content-Security-Policy: default-src 'self' data: blob: https:; script-src 'self' 'unsafe-inline' 'unsafe-eval' https: blob:; style-src 'self' 'unsafe-inline' https:; connect-src 'self' https: wss: http: ws:; frame-ancestors 'self'` (http: and ws: because WebKit matches no 'self' for a fetch or a socket from the opaque origin: on a plain http port they were refused).
  The preview's origin is opaque, so it cannot read the dashboard cookie or call this API.
- The preview's target (SRV-511, chatd/preview.py, dashboard/src/chat/preview.ts). The artifact of
  `port:N` shows port N when it listens; else the newest server that listens, with `note` saying so;
  else `waiting` (no `url`). A page that cannot run as a file (a module script that imports a package
  by name without an import map, or a .ts/.jsx/.vue entry: a bundler project's source index.html)
  shows its project's running dev server (`note`), else `waiting` with the project's `dir` and the
  `command` that starts it; the pane offers a Start button (`POST sandbox/serve`) and asks again every
  2.5 s until a server listens. While the panes are open they ask what listens every 4 s: a port that
  starts listening opens in a Preview tab, once (not one that listened at the first look, not a port
  that does not answer HTTP, not one a preview already shows). The plus menu's Preview opens the newest
  running server, else the last page. A page on the view route resolves its absolute `/` URLs against
  its own directory (a built `dist/index.html`).
- A previewed page (port or view route) gets a script first in its head: run-time URLs that start with
  `/` or name localhost:PORT (fetch, XHR, WebSocket, EventSource, an image's src) go to the port's
  prefix, so a dev server's hot-reload socket connects; and it posts `{tandemPreview: 1, ...}` to the
  dashboard after load and on later errors. The pane shows "2 errors" and the like and passes the
  report to chatd. With an admin password set, a module request from the opaque origin carries no
  session: the preview's modules need the dashboard without a password (AU1's default).
- A video, audio file, PDF or binary from the view route carries `default-src 'none'; media-src 'self'`
  (a PDF: `script-src 'none'; object-src 'self'`): it plays or downloads, it never runs a script.
- Tool output in events and parts is capped (30,000 characters per call); the full output is a file at
  `/workspace/.tandem/outputs/<call_id>.txt`.
- MCP secrets never come back from the server (values are "").
- The pane layout (`Conversation.ui`) is shared across devices; scroll positions and editor cursors stay
  on the device.
- Nothing in this API sends anything off the box. Notifications are local (Notification API, app badge),
  and their text is the chat's title plus the event, never message text.

## 6. Mocking notes

- The in-browser mock (`mock.ts`, `mock-data.ts`) is the designer's; keep it in step with this file.
- Script at least: a plain answer with thinking; a run with 3 tool calls (one `asking`); a stopped run; a
  conversation that is `waiting` when the page loads (the resume case); a compaction; a branch (2 / 3);
  a cold cache (`cache.cold` 150,000 with `cold_eta_ms` 200,000); a sandbox stopped by the memory guard;
  an `interrupted` message with Continue; a queued message under a running one.
- Never run vite, Playwright or a browser on the Mac. Build and vitest only; browser checks run on the
  Spark runner (`~/sg-live/dashboard`).

## 7. Changes v0.1 to v0.2, and what the client side still needs

The server implements v0.2 names only; v0.1's names (`message.delta`, `permission.requested`,
`run.state`, `steer`) are gone.

Client edits needed (designer):
1. The probe and every call use `/v1/chat-app/...`, not `/v1/chat/...` (`backend.ts` `pickBackend` probes
   `/v1/chat/capabilities` today).
2. `effort: 'auto'` exists and is the default (the harness decides per turn); `EFFORTS` should offer it first.
3. The pane layout is server state: `Conversation.ui`, changed with `PATCH /conversations/{id}` `{ui}`,
   followed with `ui.changed`. Pane kinds add `editor`, `pdf`, `context`, `plan`.
4. `MessageStatus` adds `interrupted` (with a Continue action); `ToolStatus` adds `cancelled`.
5. Sends carry `client_msg_id` (a ULID made on the device) so a retry never makes two messages; the same
   for queued messages.
6. `PATCH` sends `version`; on 409 apply the returned summary and retry if still valid.
7. `branch.changed` from another device: reload the conversation.
8. A sign-in screen only when `capabilities.session.password_set && !signed_in`.
9. Sandbox `network: 'allowlist'` is later; show it disabled.
10. A "Send now" action on a queued message (`POST .../queue/{qid}/now`).

## 8. Open requests

Either side appends here with a date; the architect answers by changing the sections above.

Designer requests (2026-10-12, after building the UI against v0.2; the mock does all of these):

1. The probe. `GET /capabilities` must answer `Content-Type: application/json`; the client treats a
   200 without it as a static fallback (vite's `index.html`): the mock on the dev server only,
   elsewhere the "does not serve the chat API" state. Since VIS-179 every answer but the chat API's
   is a named state chosen by the status and `error.type` (502 `chat_unavailable`, 403 `forbidden`,
   404, 503 `setup_mode`, other, no answer; `backend.ts` `classifyProbe`); the mock loads only for
   `?chat=mock` and the dev server.
2. Pane ids. `PATCH /conversations/{id}` `{ui}` with panes that carry no `id`: the server assigns one
   and answers with `ui.changed` so every device sees the same ids. The client sends `kind:ref` as a
   provisional id when it has none; the server may keep it.
3. `ui` without a version race. A `{ui}`-only PATCH should not 409 on `version`: two devices opening
   a pane at once is normal. The client still sends `version`; treat a `ui`-only patch as last-write-wins.
4. A terminal per tool call. The client opens `WS .../terminals/{call_id}/ws` for a call's pane
   (`PaneRef.ref` = the call id) and expects the call's output as the scrollback; `shell` is the user's
   own PTY. Please list the agent's call sessions in `GET /terminals` with `owner: 'agent'`.
5. `branch.changed` after the client's own `edit` or `branch`: the client reloads once per event;
   send it to the originating device too (idempotent), or skip it there, either way is handled.
6. `run.phase` while `queued` should carry `queue_position`; the thread shows "in line, N ahead".
7. `tool.asking` should carry `risk` and `suggested_rule`; the card shows the risk word and the rule
   on the "Always allow" button.
8. `conversation.updated` patches should carry `version` so a device that missed a PATCH answer does
   not 409 on its next one.
9. Attachments before send: the client uploads each file (`POST /attachments`) and sends the ids; a
   file that fails to upload keeps the message in the composer with a notice. The thumbnails for
   images come from the device's own `preview` (a data URL) until `url` answers.
10. The sign-in. The chat reuses the setup guide's card, which posts to `/v1/dashboard/session`; the
    chat routes must accept that same `qse_dash` cookie (section 2 says they do). `POST /session` under
    `/v1/chat-app` is then unused by the dashboard.
11. The global stream (`GET /events`) is not consumed yet; the rail polls `GET /conversations` every
    10 s while the tab is visible. Badges and notifications are a follow-up.

Backend notes (2026-10-12, chatd phase 1: SRV-343, SRV-345, SRV-347, SRV-349; how each request above is met):

1. `GET /capabilities` answers `application/json` through the forward, also when a password is set
   and the device is not signed in (then `session.signed_in: false`).
2. A pane without an `id` keeps the provisional `kind:ref` as its id; `ui.changed` carries it.
3. A `{ui}`-only PATCH never answers 409 and does not move `version`.
4. Phase 2: `GET /terminals` is `[]`, the WebSocket route answers 503 `sandbox_unavailable`.
5. `branch.changed` goes to every device, the originating one included. The client applies a
   snapshot through `store.adopt` (events past the snapshot's `seq` go on top, none twice).
6. `run.phase` while `queued` carries `queue_position` when the engine's live row (or its SSE
   comment `: waiting, position N`) has one.
7. Phase 2 (no tools yet).
8. Every `conversation.updated` from a change of the summary carries `version`.
9. `POST /attachments` (multipart `file`, at most 50 MB) answers an `Attachment` with `url`; an
   image's `preview` is its `url`. A text file goes to the model as a fenced block after the message,
   an image as an `image_url` part when the model's card says `vision`.
10. The forward accepts the `qse_dash` cookie; `POST /session {password}` is the dashboard's login.
11. `GET /events` (the global stream) is served: `conversation.created/updated/deleted`,
    `run.started`, `run.done`, each with its `gseq`.

Also: the stream starts at the snapshot's `seq` (an older id replays what the snapshot already has).
`Run.started_by` is filled in a snapshot only for a device other than the one that started the run
(by its browser and system, e.g. "Chrome on macOS"); `run.started` events carry it as null.
`Run.acceptance` is null while a run is on (the engine's live row has no rate); the finished
message's `stats.acceptance` has it. `thinking.budget` null means no cap. A message sent while a run
is on is queued (`202 {queued: QueuedMessage}`). Branch switches answer 409 `run_active` while a run
is on. Stop, like the mock, lets the queue drain afterwards. Not in phase 1: compaction (501
`not_available`), tools, permissions, MCP, prompts, presets, the sandbox.


## Permission modes (frontend, 2026-10-13; the server's section lands in chat-ui-brief/community/CONTRACT.md "Permission modes (API)")

The ask: fewer allow prompts for `cd` and `ls`, and a mode that asks for nothing. A chat runs in one of three modes; `chat/permission.ts` holds the words and the mock's copy of the decision.

- `ChatSettings.permission_mode: 'ask' | 'auto' | 'yolo'`, default `auto`; absent on an older server = auto. The composer's mode pill patches it like the other settings (`PATCH /conversations/{id}/settings`); the settings page's choice is the default for new chats: `PATCH /defaults` with `Partial<ChatSettings>`, answering the new `Capabilities.defaults`.
- ask: every tool asks unless its policy says always. auto: what stays in the sandbox (risk `exec`, `read`, `write`: commands, reads, writes in /workspace) runs without asking; the internet asks (web_fetch to a domain the chat has not fetched, MCP tools not marked read-only, Open WebUI tools with network or process imports), and so does rm on /, /workspace or home (server-side; chatd/tools.py decide_mode is the authority). yolo: nothing asks. A `never` policy refuses in every mode.
- `tool.started` carries `approved: 'user' | 'rule' | 'auto' | 'yolo' | null`; `ToolCallPart.approved` keeps it and the card shows a small badge for everything but `user` (YOLO in orange).
- Server (chatd, 2026-10-13): serves exactly this shape: `ChatSettings.permission_mode` (null on chats from before 0.8.2), `approved` on `tool.started` and the tool part (`rule` = a rule, an always policy or auto-accept edits on a tool that would ask; null = the tool's own default allow), `PATCH /defaults {permission_mode}` answering `Capabilities.defaults`. It also sends `tool.auto_approved` (the event log's record of each call a mode allowed), `Run.permission_mode`, `run.mode` and the assistant message's `permission_mode`; the reducer ignores them today. Full server section: chat-ui-brief/community/CONTRACT.md "Permission modes (API)".
- Bash prompts (chatd, 2026-10-13, the 11-times "always allow cd" bug): `suggested_rule` for bash carries `matches: string[]`, one prefix per part of a chained command that no allow rule covers yet (`cd /w && node a.js > run.log 2>&1; echo $?; cat run.log` -> `["cd","node","echo","cat"]`), and `match` the same joined with ", ". "Always allow" saves all of them (once each). `matches: []` (a command with `$(...)` or backticks) means no rule can allow it: the thread hides the Always button. The button reads "Always allow `cd, node, echo, cat`".


## The verify phase (SRV-532, 2026-10-17)

A run that built or changed a web page or app checks it in a browser before it ends (chatd/verify.py is the authority).

- When the model's step would end the run and web files changed since the run's start or its last check, chatd adds a call of its own to that step: `ToolCallPart` with `name: 'render_page'`, `harness: 'verify'`, `source: {kind: 'harness'}`, `input` the target (`{path}` or `{port}`). It goes through `part.added`, `tool.started` and `tool.done` like any call; its output is what the model reads, with the 1280 px screenshot in `output.images` and an image pane. The thread shows it as a `verify` card with a `harness` badge.
- Meanwhile `run.phase` is `verifying`, `now` "Verifying the app: loading <target> in a browser (check N of 2)", `tool` the call. The model then fixes what the check found or answers; at most two checks a run.
- A completed run that built or changed an app ends with a `VerifyPart` (`part.added`): `state` passed, problems (the last check's, not fixed), not_run (no check could run: no sandbox, the step limit, the sandbox failed) or stale (files changed after the last check); `text` a lead line, then "Checked: ..." and "Not checked: ..." lines with the items of `checked` and `not_checked`. The model never reads it.
- `ChatSettings.verify` (default true; absent on an older server = on) turns it off; the composer's tools menu has the switch.


## Tasks (VIS-E15)

Agents mode (`ChatSettings.agents`, default false): the main agent talks with the user and plans; a `task` tool call gives the hands-on work to a worker, a fresh assistant that works in the chat's own sandbox and /workspace and sees nothing of the chat but its brief and the todo list. Only the task result reaches the main context; the worker's transcript is stored on its own and shown in the Task card. chatd/tasks.py is the authority; the plan is the Memo note "Chat: main agent and workers (context folding), 2026-10-17".

```ts
interface ChatSettings {                  // additions
  agents?: boolean;                       // default false; absent on an older server = off
  task_max_steps?: number;                // a worker's model calls, default 40
  task_max_per_run?: number;              // the tasks one run may give, default 8
}
type RunPhase = /* ... */ | 'task';       // the parent's phase while its worker runs

type TaskStatus = 'running' | 'done' | 'partial' | 'partial (step limit)' | 'failed' | 'cancelled' | 'interrupted';
interface Task {
  id: string; n: number;                  // n: the chat's tasks from 1 ("Task 2")
  title: string; brief: string; expect: string;
  status: TaskStatus; attempt: number;
  conversation_id: string;                // the chat
  task_conversation_id: string;           // the worker's hidden conversation (never in the list)
  run_id: string; call_id: string;        // the parent run and its task call (a ToolCallPart, name 'task')
  session: string;                        // the engine's session: chat-<chat>-task-<n>
  steps: number; max_steps: number;       // the worker's model calls so far, its limit
  prompt_tokens: number; completion_tokens: number; cached_tokens: number;   // summed over its calls
  started: string; ended: string | null; // ISO times
  report: TaskReport | null;              // what the worker reported, as chatd took it (or built it)
  report_source: 'worker' | 'worker-second' | 'chatd' | null;
  check: string | null;                   // chatd's check of the work (the column check_text)
  error: string | null;
}
interface TaskReport {                    // report.v1 (SRV-581)
  schema: 'report.v1';
  status: 'done' | 'partial' | 'failed'; status_note: string | null;   // note: 'step limit'
  summary: string;                        // at most 600 characters
  changed_files: { path: string; change: string }[];                  // at most 40
  checks: { what: string; result: string; call_id: string | null; shot: string | null }[];
  expected: ({ item: string; answer: string } | string)[];            // a line without ": " as it is
  open_problems: string[];
  cut: string[];                          // "[code block of 1,234 characters removed]", in the field where it was
  built?: string;                         // a report chatd built: why ("the worker's report missed Checks twice")
}
interface TaskDetail { task: Task; messages: Message[]; seq: string }   // the transcript; seq: the chat's log it includes
```

- `GET /conversations/{id}/tasks` -> `{tasks: Task[]}` (by `n`); `GET /conversations/{id}/tasks/{task_id}` -> `TaskDetail`: the worker's messages (its user message is the brief chatd wrote; tool calls with their arguments and results), the running one's from memory. Apply the chat's events after `seq` that carry this `task_id`.
- The task call itself is a normal `ToolCallPart` (`name: 'task'`, `risk: 'task'`) in the main agent's message; its `output.content` is the task result the main agent read. The card finds its task by `call_id`.
- Events, all on the chat's own stream:

```ts
  | { type: 'task.started'; run_id: string; task: Task }
  | { type: 'task.phase'; task_id: string; phase: RunPhase; now: string; step: number; max_steps: number; tool?: Run['tool'] }
  | { type: 'task.metrics'; task_id: string; prompt_tokens: number; completion_tokens: number; decode_tps: number | null }   // E, once a second
  | { type: 'task.done'; task_id: string; status: TaskStatus; report: TaskReport | null; check: string | null;
      stats: { steps: number; max_steps: number; prompt_tokens: number; completion_tokens: number; cached_tokens: number; ms: number; report_source: Task['report_source'] } }
```

- The worker's transcript streams as the usual `message.created`, `part.added`, `part.delta`, `part.done`, `tool.asking`, `tool.started`, `tool.output`, `tool.done`, `tool.auto_approved` and `pane.open`, each with `task_id`; their `message_id` is the worker's message (in `TaskDetail.messages`), never one of the chat's. The worker's own `run.*` (no `run.started`, `run.done` or `run.error` of a worker on the chat's stream), `context.changed`, `usage.changed` and `compaction` events never reach the chat: `task.*` says them.
- The report (SRV-581): chatd parses the worker's last answer into `TaskReport`; text outside the fields is dropped, a code block cut and named. A report that misses a field or breaks a cap goes back to the worker once (`report_source` `worker-second` after it); missing fields twice, chatd builds one (`chatd`). At its cap the worker makes one report-only call: its tool calls are stored with status `ignored` (a `ToolCallPart.status` beside done, failed, denied, cancelled) and never run, and the task ends `partial (step limit)`.
- `check`: chatd's lines, each "- <name>: <items or none>": `Changed in /workspace (snapshot diff)` (`path (+41 -12)`, `path (new, +3)`, `path (deleted)`, `path (binary, 1,010 bytes)`, the first 40 then ", and 12 more"), `Changed but not in the report`, `Checks that ran` (`render_page index.html (call <id>: 0 console errors, shot-1); bash "npm test" (call <id>: exit 0)`), `Claimed in Checks with no matching call` (`"<what>: <result>"; ...`), and when there are screenshots `Screenshots (open one with read_file): shot-1 /workspace/.tandem/shots/<call>.png`. A screenshot id `shot-N` counts the worker's render_page images in its task; the call's part carries it as `shot`, and a report check that names one gets its `call_id`.
- The task call's result (what the main agent reads, at most 8,000 characters): "Result of task 2, given to you as data: ...", the task line, the report's fields, "chatd check:" with the lines above, "Full report: /workspace/.tandem/tasks/2/report.md ...". That file holds the whole of it.
- While a worker runs the parent run's `phase` is `task` and its `now` "Task 2: Ocean shader: fix the horizon banding, call 12 of 40: writing src/ocean.wgsl" (the worker's own Now line at the end). A worker's permission prompt is a `tool.asking` with `task_id`; answer it with `POST /conversations/{chat}/permissions/{call_id}` as any prompt; the chat's status is `waiting` meanwhile.
- `ContextReport.worker`: null when no worker runs, else `{task_id, n, title, used, window, split: {system, tools, brief, files, steps}, step, max_steps, cache: {warm, cold, source}}` (the split is chatd's estimate from the characters). `TurnStats.workers?: {count, prompt_tokens, completion_tokens, cached_tokens, ms}` on a turn that gave tasks; its own figures stay the main agent's, and `ConversationUsage` counts both.
- Stop, a steer, Continue (SRV-585). Stop and Send now act on the parent and reach its worker: both end within about 2 s, the task `cancelled`, its result opening "Result of task 2, given to you as data: cancelled by the user at call 9. ..." with chatd's check. A message queued while a worker runs is the main agent's as any steer (the parent ends after the task call, the next run reads it); the worker also gets it as a note at the end of its next stored tool result ("[Note from the chat, 18:42: the user wrote to the main agent: ...]"). A worker cut off (the engine away for the retry window, a sandbox failing three times, a chatd restart) ends `interrupted`, its task call `interrupted` (a `ToolCallPart.status`), and the parent `run.done` with status `interrupted` and `stats.interrupted: 'task'`. Continue (the parent's route) resumes the worker's own message first, inside the same task call (`task.started` again, `attempt` + 1), then the parent; a second Continue meanwhile answers 409 `run_active`. A call that was running when the stop came is never run again: on Continue it ends `interrupted` and reads "this call was interrupted and its effect is unknown; check the file before you repeat it" (a write, a command, an MCP call) or "... before it returned; call it again if you need it" (for a worker, and for a main agent in Agents mode). A worker that stores its end after its parent stopped waiting keeps it; Continue delivers it without running the worker again. A worker that fails (a 4xx) gives its task `failed` and a report that says why; the main agent decides.
