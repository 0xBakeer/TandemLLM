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
  output: { content: string; is_error: boolean; bytes: number; truncated?: boolean; exit_code?: number | null; images?: Attachment[] } | null;
  status: ToolStatus;
  started_at: string | null; ended_at: string | null;
  permission?: { summary: string; suggested_rule?: { tool: string; match?: string } } | null;   // suggested_rule: v0.2
  pane?: PaneRef | null;
}
interface ErrorPart { type: 'error'; message: string; retryable: boolean }
interface ArtifactPart { type: 'artifact'; id: string; title: string; mime: string; version: number }   // v2

interface Attachment {
  id: string; name: string; kind: 'image' | 'text' | 'pdf' | 'file';
  bytes: number; mime: string;
  preview?: string | null;                       // a small image's data URL
  url?: string;                                  // v0.2: GET /attachments/{id}
  tokens?: number | null;                        // v0.2: its cost in context once read
}

interface TurnStats {
  finish: 'stop' | 'length' | 'tool_calls' | 'stopped' | 'error' | 'timeout' | string;
  ttft_ms: number | null; decode_tps: number | null; total_ms: number | null;
  prompt_tokens: number; completion_tokens: number; reasoning_tokens: number; cached_tokens: number;
  tokens_per_round: number | null; acceptance: number | null;
  cache_source?: CacheSource | string;           // v0.2: resident, end, park, spill, disk, none ... (the engine's cache_source)
  prefill_tps?: number | null;                   // v0.2
  cost_usd?: number | null;                      // v0.2: cloud backends only
}

// ---- the run
type RunPhase = 'queued' | 'prefill' | 'thinking' | 'writing' | 'tool' | 'permission' | 'finishing';
interface Run {
  id: string; conversation_id: string; message_id: string; started_at: string;
  started_by: string | null;                     // a device label when it was not this browser
  phase: RunPhase; now: string;
  step: number; max_steps: number;               // v0.2: model calls so far, the cap (default 50)
  queue_position: number | null;                 // v0.2: place in the engine's line while phase = queued
  tool: { call_id: string; name: string } | null;
  prefill: { done: number; total: number; tps?: number | null; eta_ms?: number | null } | null;
  decode_tps: number | null; tokens_per_round: number | null; acceptance: number | null;
  thinking: { used: number; budget: number | null };
  tokens: number;
  tps_history: (number | null)[];                // one-second decode samples, oldest first, at most 120
}

// ---- the Context panel
type ContextPartKind = 'system' | 'tools' | 'skills' | 'files' | 'messages' | 'thinking' | 'reserved';
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
  harness: { class: string | null; effort: string | null; packages: { name: string; tokens: number }[] } | null;   // v0.2
  memory: {
    kv_bytes: number | null; resident_bytes: number | null; sandbox_disk_bytes: number | null;
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
  cached_tokens: number; tool_calls: number; wall_s: number; cost_usd?: number | null;
}

// ---- settings
type Effort = 'low' | 'medium' | 'high' | 'xhigh';
interface ChatSettings {
  model: string; profile: string | null;
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
  max_steps: number;                             // v0.2: default 50
  sandbox_network: 'off' | 'on';                 // v0.2: 'on' = internet without the LAN; see SandboxStatus.network
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
    profiles: { id: string; name: string; description: string }[];
    backend?: 'engine' | 'proxy' | 'cloud';                                         // v0.2
    capabilities?: { thinking: boolean; effort_levels: string[]; tools: boolean; vision: boolean; structured: boolean };   // v0.2, from the model card
    loaded?: boolean;
  }[];
  efforts: (Effort | 'auto')[];
  commands: SlashCommand[];
  defaults: ChatSettings;
  session: { password_set: boolean; signed_in: boolean; sandbox: 'ready' | 'not_installed' | 'error' };   // v0.2
}

// ---- the sandbox (v2)
type SandboxState = 'running' | 'stopped' | 'creating' | 'error' | 'waiting_memory';   // waiting_memory: v0.2
interface SandboxStatus {
  state: SandboxState; image: string; cwd: string;            // cwd = /workspace
  cpus: number; memory_bytes: number; memory_limit_bytes: number; disk_bytes: number; disk_limit_bytes: number;
  network: 'off' | 'allowlist' | 'on';           // 'on' = internet, never the LAN; 'allowlist' is later (mock it disabled)
  allowlist: string[];
  uptime_s: number; error: string | null;
  stopped_reason?: 'idle' | 'memory_guard' | 'user' | null;   // v0.2
  ports?: { port: number; label?: string }[];                  // v0.2: listening ports, for the preview
}
interface SandboxFile { path: string; kind: 'file' | 'dir'; bytes: number; modified_at?: string; etag?: string }

// ---- panes
type PaneKind = 'terminal' | 'preview' | 'files' | 'editor' | 'diff' | 'markdown' | 'image' | 'pdf' | 'context' | 'plan';   // v0.2 adds editor, pdf, context, plan
interface PaneRef {
  id?: string;                                   // v0.2: set by the server when the pane is in the layout
  kind: PaneKind;
  ref: string;                                   // a terminal session, a file path, an artifact id, a run id (diff), "" (context)
  title?: string;
  pinned?: boolean;
}
```

## 2. Endpoints

Auth: the dashboard's one switch (`QSE_ADMIN_TOKEN`). No password set (today's default, Khaled's ruling
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
| GET / DELETE | `/permissions/rules[/{rid}]` | | `{id, tool, match?, decision, scope, created_at}[]` | (v0.2) | v2 |
| GET | `/conversations/{id}/sandbox` | | `SandboxStatus` | `sandbox()` | v2 |
| POST | `/conversations/{id}/sandbox/reset` / `start` / `stop` | | `SandboxStatus` | `resetSandbox()` | v2 |
| GET | `/conversations/{id}/sandbox/files` | `?path=/workspace` | `SandboxFile[]` | `sandboxFiles()` | v2 |
| GET | `/conversations/{id}/sandbox/file` | `?path=` | `{content, mime, bytes, etag}` (text) or bytes with `?raw=1` | `readFile()` | v2 |
| PUT | `/conversations/{id}/sandbox/file` | `?path=`, body, `If-Match: etag` | 204; 412 when the file changed | (v0.2) | v2 |
| POST | `/conversations/{id}/sandbox/upload` | `?dir=`, multipart | `SandboxFile[]` | (v0.2) | v2 |
| GET | `/conversations/{id}/sandbox/diff` | `?run_id=` | `{files: {path, status, additions, deletions, patch}[]}` | (v0.2, diff pane) | v2 |
| POST | `/conversations/{id}/sandbox/undo` | `{run_id}` | 202 | (v0.2) | v2 |
| GET | `/conversations/{id}/sandbox/ports/{port}/...` | | reverse proxy into the sandbox (preview pane) | (v0.2) | v2 |
| GET | `/conversations/{id}/artifacts/{ref}` | `?version=` | `{kind, content, mime, title}`; `?raw=1` serves it with the preview CSP | `artifact()` | v2 |
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
  | { type: 'run.metrics'; run_id: string; decode_tps: number | null; tokens_per_round: number | null; acceptance: number | null; tokens: number; thinking_used: number; thinking_budget?: number | null; prefill?: Run['prefill'] }   // E, at most 4 a second
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
- `run.done` with status `interrupted` means chatd or the engine restarted; the UI offers Continue.
- The global stream (`/events`) sends `conversation.updated` (with `id` in the patch), `conversation.created`
  `{conversation: ConversationSummary}`, `conversation.deleted` `{id}`, `run.started`, `run.done`,
  `tool.asking` (with `conversation_id`), each persisted with its own sequence.

## 4. The Context panel

`Conversation.context` in the snapshot; `context.changed` while it changes. Sources (for the backend):
`window` and `model.*` from the model card (`/v1/models`, SRV-347); `parts` and `harness` from the
usage frame's `tandem` object (SRV-349) plus chatd's own counts; `cache` from the engine's cache peek
(before a send) and `cache_source` (after one); live run figures from `/v1/dashboard/live`, matched by
conversation id; `memory` from `/v1/dashboard/system` and chatd's sandbox guard. Live rates follow the
dashboard's NOW rule: `null` (shown as a dash) when idle; decode rates skip the first 50 tokens.

## 5. Rules the UI can rely on

- Harness information (class, effort, injected packages) arrives in `context.harness`, never inside the
  answer text.
- The preview always uses `<iframe sandbox="allow-scripts allow-forms allow-popups allow-modals">`, never
  with `allow-same-origin`. Artifact (`?raw=1`) and port answers carry
  `Content-Security-Policy: default-src 'self' data: blob: https:; script-src 'self' 'unsafe-inline' 'unsafe-eval' https: blob:; style-src 'self' 'unsafe-inline' https:; connect-src 'self' https: wss:; frame-ancestors 'self'`.
  The preview's origin is opaque, so it cannot read the dashboard cookie or call this API.
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
   200 without it as a static fallback (vite's `index.html`) and uses the mock.
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

Khaled: "too many allow prompts for cd and ls; I want a yolo mode". A chat runs in one of three modes; `chat/permission.ts` holds the words and the mock's copy of the decision.

- `ChatSettings.permission_mode: 'ask' | 'auto' | 'yolo'`, default `auto`; absent on an older server = auto. The composer's mode pill patches it like the other settings (`PATCH /conversations/{id}/settings`); the settings page's choice is the default for new chats: `PATCH /defaults` with `Partial<ChatSettings>`, answering the new `Capabilities.defaults`.
- ask: every tool asks unless its policy says always. auto: commands and reads in the sandbox (risk `exec`, `read`) run without asking; writes and the internet ask (`auto_accept_edits` still lets writes through). yolo: nothing asks. A `never` policy refuses in every mode.
- `tool.started` carries `approved: 'user' | 'rule' | 'auto' | 'yolo' | null`; `ToolCallPart.approved` keeps it and the card shows a small badge for everything but `user` (YOLO in orange).
