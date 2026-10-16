// The Chat tab's data: conversations with branches, messages made of parts (text, thinking, tool
// calls), the run that is on and its live figures, the context report the Context panel draws,
// the settings stores (tools, packages, MCP servers, prompts, presets), the sandbox, and the
// event stream a run publishes. The shapes follow dashboard/src/chat/CONTRACT.md; where the
// contract says nothing yet, the "Designer requests" section at its end names the gap.

import type { CacheSource } from '../api/types';

// ---- conversations ----------------------------------------------------------------------------

/** idle: nothing runs; running: a turn streams; waiting: a tool asks for permission; error: the
 *  last run failed */
export type ConversationStatus = 'idle' | 'running' | 'waiting' | 'error';

export interface ConversationSummary {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  pinned: boolean;
  archived: boolean;
  folder_id: string | null;
  status: ConversationStatus;
  /** the last message's first line, for the rail */
  preview: string;
  model: string;
  profile: string | null;
  turns: number;
  /** what the run does now, in the server's words, while status is running or waiting */
  now?: string | null;
  /** optimistic concurrency for PATCH: send it back, a 409 carries the current summary */
  version: number;
  /** an answer finished since this device last looked (kept per device, in the browser) */
  unread: boolean;
}

export interface Folder {
  id: string;
  name: string;
}

export interface Conversation extends ConversationSummary {
  /** the messages on the active branch, in order */
  messages: Message[];
  /** the leaf of the active branch */
  head_id: string | null;
  settings: ChatSettings;
  run: Run | null;
  context: ContextReport;
  sandbox: SandboxStatus;
  compactions: Compaction[];
  usage: ConversationUsage;
  /** messages typed while a run was on; they go in when the current step ends */
  queue: QueuedMessage[];
  /** the pane layout, shared across devices */
  ui: ConversationUi;
  /** the last event id this snapshot includes: the stream resumes after it */
  seq: string;
  /** the task's todo list (todo_write, an approved plan); the plan pane shows it */
  todos?: TodoItem[];
}

export interface TodoItem {
  id: string;
  text: string;
  state: 'pending' | 'in_progress' | 'done';
}

export interface QueuedMessage {
  id: string;
  text: string;
  created_at: string;
  client_msg_id: string;
}

export interface ConversationUi {
  /** tab order; the phone shows them as a sheet's tabs */
  panes: PaneRef[];
  /** a PaneRef id */
  active: string | null;
  /** desktop: the thread's share of the width when the panes dock right, 0.35..0.75 */
  split: number;
}

// ---- messages ---------------------------------------------------------------------------------

/** system: notices (a compaction divider, interrupted, cancelled) */
export type Role = 'user' | 'assistant' | 'system';
/** interrupted: chatd or the engine restarted; Continue resumes */
export type MessageStatus = 'done' | 'streaming' | 'stopped' | 'error' | 'interrupted';

export interface Message {
  id: string;
  role: Role;
  created_at: string;
  /** the tree behind branches */
  parent_id: string | null;
  /** the run that wrote an assistant message */
  run_id: string | null;
  parts: Part[];
  status: MessageStatus;
  attachments?: Attachment[];
  /** set when this message has sibling versions (an edit or a regenerate made a branch) */
  siblings?: { index: number; count: number };
  stats?: TurnStats;
}

export type Part = TextPart | ThinkingPart | ToolCallPart | ErrorPart | ArtifactPart | TodosPart | VerifyPart;

/** SRV-532: the end of an answer whose run built or changed a page or app: what the harness's
 *  browser check and the model's own checks covered, and what they did not. `text` is a lead line,
 *  then "Checked: ..." and "Not checked: ..." lines (copied as it is) */
export interface VerifyPart {
  type: 'verify';
  /** passed; problems: the last check's, not fixed; not_run: no check could run (why in the text);
   *  stale: files changed after the last check */
  state: 'passed' | 'problems' | 'not_run' | 'stale';
  text: string;
  /** the items of the text's "Checked:" and "Not checked:" lines */
  checked?: string[];
  not_checked?: string[];
}

/** the open todo list as the model read it at the run's start; never drawn */
export interface TodosPart {
  type: 'todos';
  text: string;
}

export interface TextPart {
  type: 'text';
  text: string;
}
export interface ThinkingPart {
  type: 'thinking';
  text: string;
  tokens: number | null;
  done: boolean;
  duration_ms?: number | null;
}
export type ToolStatus = 'asking' | 'running' | 'done' | 'failed' | 'denied' | 'cancelled';
export type ToolRisk = 'read' | 'write' | 'exec' | 'network';
export interface ToolOutput {
  content: string;
  is_error: boolean;
  bytes: number;
  truncated?: boolean;
  exit_code?: number | null;
  images?: Attachment[];
  /** structured output for the card: web_search's results (SRV-469), or why the sandbox could
   *  not run the call (SRV-481, SRV-485) */
  data?: (WebSearchData & { sandbox?: undefined }) | { sandbox: SandboxCallData; results?: undefined } | null;
}
export interface ToolCallPart {
  type: 'tool_call';
  id: string;
  /** MCP tools: "mcp__<server>__<tool>" */
  name: string;
  source?: ToolSource | { kind: 'harness' };
  risk?: ToolRisk;
  /** JSON text, growing while the model writes it */
  input: string;
  output: ToolOutput | null;
  status: ToolStatus;
  started_at: string | null;
  ended_at: string | null;
  /** what the permission prompt should say, when the tool asks */
  permission?: { summary: string; suggested_rule?: { tool: string; match?: string; matches?: string[] } } | null;   // matches: bash, one prefix per part not yet allowed ([] = a hidden command, no always)
  /** who allowed the call: the user at the prompt, an always rule, auto mode or YOLO mode */
  approved?: ToolApproval | null;
  /** a pane that can show this call's work */
  pane?: PaneRef | null;
  /** SRV-532: a call the harness made itself, not the model (`verify`: the browser check before
   *  the run ends) */
  harness?: 'verify';
}
/** a tool call that ran without a prompt says who allowed it (the badge on its card) */
export type ToolApproval = 'user' | 'rule' | 'auto' | 'yolo';
export interface ErrorPart {
  type: 'error';
  message: string;
  retryable: boolean;
}
/** v2: an artifact the model produced, shown in a pane */
export interface ArtifactPart {
  type: 'artifact';
  id: string;
  title: string;
  mime: string;
  version: number;
}

export interface Attachment {
  id: string;
  name: string;
  kind: 'image' | 'text' | 'pdf' | 'file';
  bytes: number;
  mime: string;
  /** a small image's data URL, for the thumbnail */
  preview?: string | null;
  /** GET /attachments/{id} */
  url?: string;
  /** its cost in context once read (a markup file's: its readable text) */
  tokens?: number | null;
  /** a text file: whether the prompt carries it whole; false: one line with its path (SRV-487).
   *  Before the send, chatd's guess against its fixed cap */
  inline?: boolean;
  /** where the send saved it in the chat's workspace (uploads/) */
  path?: string;
}

export interface TurnStats {
  finish: 'stop' | 'length' | 'tool_calls' | 'stopped' | 'error' | 'timeout' | string;
  ttft_ms: number | null;
  decode_tps: number | null;
  prompt_tokens: number;
  completion_tokens: number;
  reasoning_tokens: number;
  cached_tokens: number;
  tokens_per_round: number | null;
  acceptance: number | null;
  total_ms: number | null;
  cache_source?: CacheSource | string;
  prefill_tps?: number | null;
  /** cloud backends only */
  cost_usd?: number | null;
  /** an `interrupted` answer: why (VIS-197); absent on answers from before 0.11.2 */
  interrupted?: 'limit' | 'sandbox' | 'engine' | 'restart' | string;
}

// ---- the run ----------------------------------------------------------------------------------

export type RunPhase = 'queued' | 'prefill' | 'thinking' | 'writing' | 'tool' | 'permission' | 'verifying' | 'finishing';

export interface Run {
  id: string;
  conversation_id: string;
  message_id: string;
  started_at: string;
  /** a label of the client that started the run, when it was not this browser */
  started_by: string | null;
  phase: RunPhase;
  /** the server's words for the phase, e.g. "Running bash: pytest -q" */
  now: string;
  /** model calls so far, and the cap (default 50) */
  step: number;
  max_steps: number;
  /** the place in the engine's line while phase is queued */
  queue_position: number | null;
  tool: { call_id: string; name: string } | null;
  prefill: { done: number; total: number; tps?: number | null; eta_ms?: number | null } | null;
  decode_tps: number | null;
  tokens_per_round: number | null;
  acceptance: number | null;
  /** used: the run's, every model call; call: this call's, which the budget caps */
  thinking: { used: number; budget: number | null; call?: number };
  /** the completion tokens so far, every model call of the run */
  tokens: number;
  /** one-second decode samples for the sparkline, oldest first, at most 120 */
  tps_history: (number | null)[];
}

// ---- the Context panel ------------------------------------------------------------------------

export type ContextPartKind = 'system' | 'tools' | 'skills' | 'files' | 'images' | 'messages' | 'thinking' | 'reserved';

export interface ContextPart {
  kind: ContextPartKind;
  label: string;
  tokens: number;
}

/** SRV-471: the memory notes' part of a turn (the usage frame's `tandem.harness.memory`); an
 *  engine before it sends none */
export interface HarnessMemory {
  mode: 'on' | 'shadow' | 'off' | string;
  /** why it is off: profile, model_fit, backend, header */
  why: string | null;
  index_notes: number;
  index_tokens: number;
  /** the notes sent with this message in full */
  bodies: number;
  body_tokens: number;
  card_tokens: number;
}
export interface ContextReport {
  window: number;
  used: number;
  /** sums to `used` */
  parts: ContextPart[];
  /** chatd compacts above this many tokens (0.8 of the window) */
  compact_at: number;
  cache: {
    /** tokens the next request finds cached */
    warm: number;
    /** tokens it would re-read */
    cold: number;
    /** the re-read estimate */
    cold_eta_ms: number | null;
    source: CacheSource | string;
    /** the engine slot the chat sits on */
    slot: string | null;
  };
  /** the chat's skill mode and pins (SRV-361); a pin the library lost is `missing` */
  skills?: { mode: 'auto' | 'manual' | 'model'; pinned: { name: string; tokens: number | null; missing: boolean }[]; blocked: string[] };
  /** the engine's split of the last prompt (its usage frame); null before the first frame */
  split?: { system: number; tools: number; history: number; injected: number; images: number } | null;
  /** what the harness decided for this chat; never inside the answer text */
  harness: { class: string | null; effort: string | null; packages: { name: string; tokens: number | null }[]; memory?: HarnessMemory | null } | null;
  memory: {
    kv_bytes: number | null;
    /** the chat's rows the engine holds: in its slot or parked beside it (`kv_where`) */
    kv_tokens?: number | null;
    /** slot, park; else the tier the next turn reads from (nvme_spill, disk_tier, ram_snapshot),
     *  other (another chat has the slot), none; null: the engine does not report its caches */
    kv_where?: 'slot' | 'park' | 'nvme_spill' | 'disk_tier' | 'ram_snapshot' | 'other' | 'none' | null;
    /** the resident prefix's snapshots of this chat (its anchors, end and stash) */
    resident_bytes: number | null;
    resident_owner?: 'this' | 'other' | 'none' | null;
    sandbox_disk_bytes: number | null;
    available_bytes?: number | null;
    /** the engine's admission floor */
    floor_bytes?: number | null;
    sandbox_bytes?: number | null;
  };
  model: {
    name: string;
    profile: string | null;
    weights: string;
    drafter: string | null;
    max_len: number;
    effort: Effort | 'auto';
    thinking: boolean;
    backend?: 'engine' | 'proxy' | 'cloud';
  };
}

export interface Compaction {
  id: string;
  at: string;
  before_tokens: number;
  after_tokens: number;
  summary: string;
  duration_ms?: number;
  /** the first message the model still reads in full; the thread's divider sits above it */
  boundary_id?: string;
}

export interface ConversationUsage {
  turns: number;
  prompt_tokens: number;
  completion_tokens: number;
  reasoning_tokens: number;
  cached_tokens: number;
  tool_calls: number;
  wall_s: number;
  cost_usd?: number | null;
}

// ---- settings ---------------------------------------------------------------------------------

export type Effort = 'low' | 'medium' | 'high' | 'xhigh';
/** 'auto' (the default): the harness decides per turn */
export type EffortChoice = Effort | 'auto';
export const EFFORTS: EffortChoice[] = ['auto', 'low', 'medium', 'high', 'xhigh'];

export interface ChatSettings {
  model: string;
  profile: string | null;
  effort: EffortChoice;
  /** false = thinking off whatever the effort */
  thinking: boolean;
  /** tool ids on for this chat */
  tools: string[];
  /** package and skill ids pinned on for this chat */
  packages: string[];
  /** never injected in this chat */
  packages_blocked: string[];
  /** auto: the harness picks (plus pins); manual: pins only; model: the model reads the skills itself */
  packages_mode: 'auto' | 'manual' | 'model';
  /** MCP server ids on for this chat */
  mcp: string[];
  /** a saved prompt's id, or null for the default */
  prompt_id: string | null;
  /** per-tool policy overrides for this chat ("Always allow bash" writes here) */
  policies: Record<string, ToolPolicy>;
  /** write tools run without asking */
  auto_accept_edits: boolean;
  /** model calls per turn, default 50 */
  max_steps: number;
  /** 'on' = the internet without the LAN; see SandboxStatus.network */
  sandbox_network: 'off' | 'on';
  /** SRV-363: the model reads and plans, changes nothing, until the plan is approved */
  plan_mode?: boolean;
  /** who approves a tool call (chat/permission.ts); absent on an older server = auto */
  permission_mode?: PermissionMode;
  /** SRV-469: null = the global web search settings; else this chat's own choices */
  web_search?: WebSearchOverride | null;
  /** SRV-532: a run that built or changed a page or app checks it in a browser before it ends;
   *  absent on an older server */
  verify?: boolean;
}
/** ask: every tool asks unless its policy says always; auto (the default): what stays in the
 *  sandbox runs (commands, reads, workspace writes), the internet asks; yolo: nothing asks */
export type PermissionMode = 'ask' | 'auto' | 'yolo';

export type ToolPolicy = 'ask' | 'always' | 'never';
export type ToolSource = { kind: 'builtin' } | { kind: 'package'; id: string } | { kind: 'mcp'; id: string } | { kind: 'user'; id: string };

export interface ToolInfo {
  id: string;
  name: string;
  description: string;
  source: ToolSource;
  enabled: boolean;
  policy: ToolPolicy;
  risk?: ToolRisk;
  /** what its schema costs in every request */
  schema_tokens?: number;
}

export interface PackageInfo {
  id: string;
  name: string;
  description: string;
  kind: 'skill' | 'package';
  version: string;
  enabled: boolean;
  tools: string[];
  tokens?: number;
  /** uploaded here (a SKILL.md zip): it can be removed */
  user?: boolean;
}

/** an imported Open WebUI Tools file (SRV-361): its tools are `owui__<slug>__<method>` */
export interface UserToolkit {
  id: string;
  slug: string;
  title: string;
  description: string;
  version: string;
  requirements: string[];
  enabled: boolean;
  tools: string[];
  /** a secret valve's value comes back empty; `set` says whether one is stored */
  valves: { name: string; type: string; default: unknown; description: string; secret: boolean; value: unknown; set: boolean }[];
  created_at: string;
}

export type McpStatus = 'connected' | 'starting' | 'failed' | 'stopped';
export interface McpServer {
  id: string;
  name: string;
  transport: 'stdio' | 'http';
  command?: string;
  args?: string[];
  env?: Record<string, string>;
  url?: string;
  headers?: Record<string, string>;
  status: McpStatus;
  error: string | null;
  tools: { name: string; description: string }[];
  tool_allow?: string[] | null;
  tool_deny?: string[];
  schema_tokens?: number;
  enabled_global?: boolean;
  updated_at: string;
  /** its prompts (slash commands) and resources (attachments), plan note Chat 3 */
  prompts?: { name: string; description: string; arguments: { name: string; required: boolean }[] }[];
  resources?: { uri: string; name: string; mime: string | null }[];
}
export type McpServerDraft = Omit<McpServer, 'id' | 'status' | 'error' | 'tools' | 'updated_at' | 'prompts' | 'resources'>;

/** the text after an MCP prompt's command as its arguments: `key=value` pairs, the rest to the
 *  first argument */
export function promptArgs(text: string, names: string[]): Record<string, string> {
  const out: Record<string, string> = {};
  const rest: string[] = [];
  for (const w of text.split(/\s+/).filter(Boolean)) {
    const m = /^([A-Za-z_][\w-]*)=(.*)$/.exec(w);
    if (m && names.includes(m[1])) out[m[1]] = m[2];
    else rest.push(w);
  }
  if (rest.length && names.length && !(names[0] in out)) out[names[0]] = rest.join(' ');
  return out;
}

export interface Prompt {
  id: string;
  title: string;
  /** the slash command without the slash */
  command: string;
  body: string;
}

export interface Preset {
  id: string;
  name: string;
  model: string;
  profile: string | null;
  effort: EffortChoice;
  thinking: boolean;
  tools: string[];
  prompt_id: string | null;
  packages?: string[];
  mcp?: string[];
}

export interface SlashCommand {
  command: string;
  description: string;
  /** a prompt's id, when the command inserts a saved prompt */
  prompt_id?: string;
  /** an MCP server's prompt: the text typed after the command fills its arguments */
  mcp_prompt?: { server_id: string; name: string; arguments?: { name: string; required: boolean }[] };
}

export interface ModelInfo {
  id: string;
  name: string;
  weights: string;
  max_len: number;
  profiles: { id: string; name: string; description: string }[];
  backend?: 'engine' | 'proxy' | 'cloud';
  capabilities?: { thinking: boolean; effort_levels: string[]; tools: boolean; vision: boolean; structured: boolean };
  loaded?: boolean;
}

export interface Capabilities {
  models: ModelInfo[];
  efforts: EffortChoice[];
  commands: SlashCommand[];
  /** the server's own default chat settings */
  defaults: ChatSettings;
  /** the sign-in screen shows only when a password is set and this device is not signed in */
  session: { password_set: boolean; signed_in: boolean; sandbox: 'ready' | 'not_installed' | 'error' };
  /** the global web search provider (absent on an older server) */
  web_search?: { provider: WebSearchProvider; providers: WebSearchProvider[] };
}

// ---- web search (SRV-469, VIS-157) -------------------------------------------------------------

export type WebSearchProvider = 'off' | 'duckduckgo' | 'searxng';
export type SafeSearch = 'off' | 'moderate' | 'strict';
export type TimeRange = '' | 'day' | 'week' | 'month' | 'year';
/** GET /web-search's settings: the auth value never comes back, only whether one is stored */
export interface WebSearchSettings {
  provider: WebSearchProvider;
  searxng_url: string;
  searxng_auth_header: string;
  searxng_auth_set: boolean;
  max_results: number;
  safe_search: SafeSearch;
  /** '' = any, else de or de-DE */
  language: string;
  time_range: TimeRange;
  timeout_s: number;
}
/** PUT /web-search: a partial update; searxng_auth_value null clears it, absent keeps it */
export type WebSearchPatch = Partial<Omit<WebSearchSettings, 'searxng_auth_set'>> & { searxng_auth_value?: string | null };
export interface WebSearchView {
  settings: WebSearchSettings;
  providers: WebSearchProvider[];
  safe_search: SafeSearch[];
  time_ranges: TimeRange[];
  max_results_limit: number;
}
export interface WebSearchTest {
  ok: boolean;
  provider: WebSearchProvider;
  message: string;
  count: number;
  ms: number;
  results: WebSearchResult[];
}
export type WebSearchOverride = Partial<Pick<WebSearchSettings, 'provider' | 'max_results' | 'safe_search' | 'language' | 'time_range'>>;
export interface WebSearchResult {
  title: string;
  url: string;
  snippet: string;
  date?: string;
  engine?: string;
}
export interface WebSearchData {
  provider: WebSearchProvider;
  query: string;
  results: WebSearchResult[];
}

// ---- the sandbox ------------------------------------------------------------------------------

/** a call the sandbox could not run: it did not start within its wait (short of memory, or the
 *  other chats' sandboxes busy), or the kernel killed a process at the memory limit */
export interface SandboxCallData {
  reason: 'no_memory' | 'busy' | 'oom';
  limit_gib: number;
  why?: string;
  waited_s?: number;
  mem_available_gib?: number;
  floor_gib?: number;
  engine_released_gib?: number;
}

export type SandboxState = 'running' | 'stopped' | 'creating' | 'error' | 'waiting_memory';
export interface SandboxStatus {
  state: SandboxState;
  image: string;
  cwd: string;
  cpus: number;
  /** in use (the runtime's figure, page cache left out); null when the runtime does not say */
  memory_bytes: number | null;
  memory_limit_bytes: number;
  disk_bytes: number;
  disk_limit_bytes: number;
  /** 'on' = the internet, never the LAN; 'allowlist' is later (shown disabled) */
  network: 'off' | 'allowlist' | 'on';
  allowlist: string[];
  uptime_s: number;
  error: string | null;
  stopped_reason?: 'idle' | 'memory_guard' | 'user' | null;
  /** listening ports, for the preview */
  ports?: { port: number; label?: string }[];
}

export interface SandboxFile {
  path: string;
  kind: 'file' | 'dir';
  bytes: number;
  modified_at?: string;
  etag?: string;
}
/** `GET .../sandbox/file`: the text, or null for a binary or a file over 2 MiB (VIS-159) */
export interface SandboxFileView {
  content: string | null;
  mime: string;
  bytes: number;
  etag?: string;
  /** the pane kind: file (text), binary, image, preview, markdown, video, audio, pdf */
  kind?: string;
  /** the view route, streamed with byte ranges */
  url?: string;
}

// ---- panes ------------------------------------------------------------------------------------

export type PaneKind = 'terminal' | 'preview' | 'files' | 'editor' | 'diff' | 'markdown' | 'image' | 'pdf' | 'video' | 'audio' | 'binary' | 'context' | 'plan';
export interface PaneRef {
  /** set by the server when the pane is in the layout */
  id?: string;
  kind: PaneKind;
  /** a terminal session, a file path, an artifact id, a run id (diff), "" (context) */
  ref: string;
  title?: string;
  pinned?: boolean;
}

// ---- the event stream -------------------------------------------------------------------------

/** One event on a conversation's stream (`event:` name = `type`, `id:` = a growing sequence). */
export type ChatEvent =
  | { type: 'run.started'; run: Run }
  | { type: 'run.phase'; run_id: string; phase: RunPhase; now: string; tool?: Run['tool']; prefill?: Run['prefill']; step?: number; queue_position?: number | null }
  | { type: 'run.metrics'; run_id: string; decode_tps: number | null; tokens_per_round: number | null; acceptance: number | null; tokens: number; thinking_used: number; thinking_call?: number; thinking_budget?: number | null; prefill?: Run['prefill'] }
  | { type: 'run.done'; run_id: string; message_id: string; stats: TurnStats; status: MessageStatus }
  | { type: 'run.error'; run_id: string; message_id: string; message: string; retryable: boolean }
  | { type: 'message.created'; message: Message }
  | { type: 'part.added'; message_id: string; part: Part }
  | { type: 'part.delta'; message_id: string; index: number; delta: string; field?: 'text' | 'input' }
  | { type: 'part.done'; message_id: string; index: number; tokens?: number | null }
  | { type: 'tool.asking'; message_id: string; call_id: string; summary: string; risk?: string; suggested_rule?: { tool: string; match?: string; matches?: string[] } }
  | { type: 'tool.started'; message_id: string; call_id: string; pane?: PaneRef | null; approved?: ToolApproval | null }
  | { type: 'tool.output'; message_id: string; call_id: string; delta: string }
  | { type: 'tool.done'; message_id: string; call_id: string; status: 'done' | 'failed' | 'denied' | 'cancelled'; is_error: boolean; output?: ToolOutput | null }
  | { type: 'queue.changed'; queue: QueuedMessage[] }
  | { type: 'context.changed'; context: ContextReport }
  | { type: 'compaction'; compaction: Compaction }
  | { type: 'conversation.updated'; patch: Partial<ConversationSummary> }
  /** another device switched or made a branch: reload the conversation */
  | { type: 'branch.changed'; head_id: string }
  | { type: 'ui.changed'; ui: ConversationUi }
  /** the agent asks to show something */
  | { type: 'pane.open'; pane: PaneRef }
  | { type: 'artifact.created'; artifact: ArtifactPart }
  | { type: 'sandbox.changed'; sandbox: SandboxStatus }
  | { type: 'usage.changed'; usage: ConversationUsage }
  | { type: 'settings.changed'; settings: ChatSettings }
  | { type: 'todos.changed'; todos: TodoItem[] };

export interface StampedEvent {
  id: string;
  event: ChatEvent;
}

let seq = 0;
export function localId(prefix = 'c'): string {
  seq++;
  return `${prefix}_${Date.now().toString(36)}${seq.toString(36)}`;
}

/** a client_msg_id made on the device: a send retried after a dropped request never makes two messages */
export function clientMsgId(): string {
  const r = typeof crypto !== 'undefined' && 'randomUUID' in crypto ? crypto.randomUUID().replace(/-/g, '').slice(0, 20) : Math.random().toString(36).slice(2, 12) + Date.now().toString(36);
  return `cm_${r}`;
}

/** the words for an effort choice */
export function effortWord(e: EffortChoice): string {
  return e === 'xhigh' ? 'extra high' : e === 'auto' ? 'auto effort' : e;
}

/** an MCP tool's wire name "mcp__server__tool" reads as "server: tool" */
export function toolLabel(name: string): string {
  const m = /^mcp__([^_]+(?:_[^_]+)*)__(.+)$/.exec(name);
  return m ? `${m[1]}: ${m[2]}` : name;
}

/** the typed valve values in their types (a number valve sends a number) */
export function valveValues(k: Pick<UserToolkit, 'valves'>, draft: Record<string, string>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const v of k.valves) {
    if (!(v.name in draft)) continue;
    const t = draft[v.name];
    out[v.name] = v.type === 'integer' || v.type === 'number' ? (t.trim() === '' ? null : Number(t)) : v.type === 'boolean' ? /^(true|1|yes|on)$/i.test(t.trim()) : t;
  }
  return out;
}

/** the steps of a plan answer, as the server reads them (chatd/plans.py `plan_items`): its
 *  numbered lines, else its bullets, else the whole answer as one step */
export function planItems(text: string): string[] {
  const num: string[] = [];
  const bul: string[] = [];
  for (const line of text.split('\n')) {
    const n = /^\s{0,3}\d{1,3}[.)]\s+(.+)/.exec(line);
    if (n) num.push(n[1].trim());
    else {
      const b = /^\s{0,3}[-*+]\s+(?:\[[ xX]\]\s+)?(.+)/.exec(line);
      if (b) bul.push(b[1].trim());
    }
  }
  const items = num.length ? num : bul.length ? bul : text.trim() ? [text.split(/\s+/).join(' ').slice(0, 300)] : [];
  return items.map((i) => i.replace(/\*\*(.+?)\*\*/g, '$1').slice(0, 300)).slice(0, 50);
}
