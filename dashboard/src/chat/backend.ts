// What the Chat tab needs from a server, as one interface. Two implementations: remote.ts
// speaks the contract in CONTRACT.md; mock.ts plays it in the browser for `?chat=mock` and the
// Vite dev server. pickBackend() probes the server; any other answer than the chat API's is a
// named state the tab shows (chatd not running, its key refused, no chat routes, ...), never the
// mock (VIS-179).

import { apiBase } from '../api/client';
import type { Attachment, Capabilities, ChatSettings, Conversation, ConversationSummary, ConversationUi, Folder, McpServer, McpServerDraft, PackageInfo, Preset, Prompt, SandboxFile, SandboxFileView, SandboxStatus, StampedEvent, ToolInfo, ToolPolicy, UserToolkit, WebSearchPatch, WebSearchTest, WebSearchView } from './types';
import type { StreamState } from '../lib/sse';

export interface Subscription {
  close(): void;
}

export interface StreamHandler {
  /** `replay`: the event is part of the catch-up after Last-Event-ID, not live */
  onEvent(ev: StampedEvent, replay: boolean): void;
  /** the catch-up is over: what follows is live (`event: current` on the wire) */
  onCurrent(): void;
  onState(s: StreamState, detail?: string): void;
}

export interface TerminalSession {
  write(data: string): void;
  resize(cols: number, rows: number): void;
  close(): void;
}

export interface SendOptions {
  /** made on the device (clientMsgId()); a retry with the same id makes no second message */
  client_msg_id: string;
  text: string;
  attachments?: Attachment[];
}

/** PATCH with a stale `version`: the server answers 409 with the current summary */
export class VersionConflict extends Error {
  constructor(public current: ConversationSummary) {
    super('version conflict');
  }
}

export interface ChatBackend {
  readonly kind: 'remote' | 'mock';
  capabilities(): Promise<Capabilities>;

  // conversations
  listConversations(): Promise<{ conversations: ConversationSummary[]; folders: Folder[] }>;
  getConversation(id: string): Promise<Conversation>;
  createConversation(settings?: Partial<ChatSettings>): Promise<Conversation>;
  /** `version` rides along; a stale one throws VersionConflict with the current summary */
  updateConversation(id: string, patch: { version?: number; title?: string; pinned?: boolean; archived?: boolean; folder_id?: string | null; ui?: ConversationUi }): Promise<ConversationSummary>;
  deleteConversation(id: string): Promise<void>;
  createFolder(name: string): Promise<Folder>;
  /** the live stream of one conversation; `lastEventId` resumes with a catch-up */
  subscribe(id: string, lastEventId: string | null, h: StreamHandler): Subscription;
  /** the list-level stream: run.done and tool.asking of every chat (notifications, the badge) */
  subscribeGlobal(onEvent: (ev: { type: string; conversation_id?: string; status?: string }) => void): Subscription;

  // the turn
  send(id: string, msg: SendOptions): Promise<void>;
  /** a message typed while a run is on: it goes in after the current step */
  queue(id: string, text: string, clientMsgId: string): Promise<void>;
  unqueue(id: string, queuedId: string): Promise<void>;
  /** stops the current generation, keeps its text, sends this queued message now */
  sendNow(id: string, queuedId: string): Promise<void>;
  stop(id: string): Promise<void>;
  /** resumes an `interrupted` run (chatd or the engine restarted) */
  continueRun(id: string): Promise<void>;
  regenerate(id: string, messageId: string): Promise<void>;
  /** edit a user message: a new branch from there, then a run */
  edit(id: string, messageId: string, text: string, clientMsgId: string): Promise<void>;
  switchBranch(id: string, messageId: string, index: number): Promise<Conversation>;
  permission(id: string, callId: string, decision: 'once' | 'always' | 'deny', scope?: 'chat' | 'all'): Promise<void>;
  updateSettings(id: string, patch: Partial<ChatSettings>): Promise<ChatSettings>;
  /** the server's defaults for new chats (Capabilities.defaults): the settings page writes them */
  updateDefaults(patch: Partial<ChatSettings>): Promise<ChatSettings>;
  compact(id: string): Promise<void>;
  /** the plan answer's steps (or the edited list) become the todo list; a run carries it out */
  approvePlan(id: string, items?: string[]): Promise<void>;
  /** the files back to before the runs after this user message; the head to its parent */
  rewind(id: string, messageId: string): Promise<{ conversation: Conversation; text: string; files: number }>;

  // the session (only when the server has a password set)
  signIn(password: string): Promise<void>;
  signOut(): Promise<void>;
  /** multipart upload; the answer's id goes into a send's attachments */
  uploadAttachment(file: File): Promise<Attachment>;

  // settings stores
  tools(): Promise<ToolInfo[]>;
  setTool(toolId: string, patch: { enabled?: boolean; policy?: ToolPolicy }): Promise<ToolInfo>;
  packages(): Promise<PackageInfo[]>;
  setPackage(pkgId: string, enabled: boolean): Promise<PackageInfo>;
  /** a .zip with a SKILL.md folder, installed into the engine's library */
  uploadSkill(file: File): Promise<PackageInfo>;
  removeSkill(pkgId: string): Promise<void>;
  /** an Open WebUI Tools .py (or its JSON export) */
  importOwui(file: File): Promise<{ toolkits: UserToolkit[]; tools: ToolInfo[] }>;
  userToolkits(): Promise<UserToolkit[]>;
  patchUserToolkit(id: string, patch: { enabled?: boolean; valves?: Record<string, unknown> }): Promise<UserToolkit>;
  removeUserToolkit(id: string): Promise<void>;
  mcpServers(): Promise<McpServer[]>;
  addMcpServer(draft: McpServerDraft): Promise<McpServer>;
  removeMcpServer(serverId: string): Promise<void>;
  testMcpServer(draft: McpServerDraft): Promise<{ ok: boolean; message: string; tools: { name: string; description: string }[] }>;
  restartMcpServer(serverId: string): Promise<McpServer>;
  /** an MCP prompt's text for the composer; a stdio server answers from the chat's sandbox */
  mcpPrompt(serverId: string, name: string, args: Record<string, string>, convId: string | null): Promise<string>;
  /** an MCP resource as an attachment, uploaded already */
  mcpResource(serverId: string, uri: string, convId: string | null): Promise<Attachment>;
  prompts(): Promise<Prompt[]>;
  savePrompt(p: Omit<Prompt, 'id'> & { id?: string }): Promise<Prompt>;
  deletePrompt(promptId: string): Promise<void>;
  /** SRV-469: the global web search settings (the auth value never comes back) */
  webSearch(): Promise<WebSearchView>;
  saveWebSearch(patch: WebSearchPatch): Promise<WebSearchView>;
  /** one sample search with the draft over the stored settings; nothing is stored */
  testWebSearch(draft: WebSearchPatch): Promise<WebSearchTest>;
  presets(): Promise<Preset[]>;
  savePreset(p: Omit<Preset, 'id'> & { id?: string }): Promise<Preset>;
  deletePreset(presetId: string): Promise<void>;

  // the sandbox
  sandbox(id: string): Promise<SandboxStatus>;
  resetSandbox(id: string): Promise<SandboxStatus>;
  sandboxFiles(id: string, path: string): Promise<SandboxFile[]>;
  /** `kind`: the pane kind (file, binary, image, preview, markdown, video, audio, pdf); `url`: the view route */
  readFile(id: string, path: string): Promise<SandboxFileView>;
  /** PUT with If-Match; a changed file throws */
  writeFile(id: string, path: string, content: string, etag?: string): Promise<void>;
  /** an artifact the model produced (an html preview, a diff, an image) */
  /** `url`: a same-origin address the preview loads in its sandboxed iframe (the server's files) */
  artifact(id: string, ref: string): Promise<Artifact>;
  /** SRV-511: what listens in the sandbox ([] when it is stopped) */
  sandboxPorts(id: string): Promise<ListeningPort[]>;
  /** start the dev server of the project in `dir` (the command comes from its package.json) */
  serve(id: string, dir: string): Promise<{ dir: string; command: string; log: string }>;
  /** what the preview pane loaded, for `present`'s answer to the model */
  previewReport(id: string, report: PreviewReport): Promise<void>;
  /** PTY bytes as they come (the mock sends text) */
  terminal(id: string, session: string, onData: (data: string | Uint8Array) => void): TerminalSession;
}

/** What a pane shows (chatd/panes.py `artifact`). A preview's target (SRV-511): `port` the server
 *  it shows, `note` why that one (the port asked for does not listen; a bundler project's page runs
 *  from its dev server), `waiting` when nothing can show yet (no `url` then). */
export interface Artifact {
  kind: string;
  content: string;
  mime: string;
  title: string;
  url?: string;
  bytes?: number;
  port?: number;
  server?: string;
  note?: string;
  waiting?: PreviewWaiting;
}
export interface PreviewWaiting {
  /** the port asked for, null for a page */
  port: number | null;
  /** the project whose dev server the page needs, and the command that starts it */
  dir: string | null;
  command: string | null;
  message: string;
}
/** a listening TCP port in the sandbox: its process, working directory and start (clock ticks) */
export interface ListeningPort {
  port: number;
  addresses?: string[];
  command?: string;
  cwd?: string;
  started?: number;
  /** false: it did not answer an HTTP request (a database, a language server) */
  http?: boolean;
}
/** what a previewed page posted: its errors, the files that failed, how much it drew */
export interface PreviewReport {
  ref: string;
  url: string;
  title: string;
  errors: string[];
  failed: string[];
  elements: number;
  text: number;
  canvas: number;
}

/** Why the chat API is not there, from the probe's status and `error.type` (never its text):
 *  chatd_down     502 `chat_unavailable`: the engine server answers, chatd does not
 *  chatd_refused  403 `forbidden`: chatd answers and refuses the server's forward key
 *  no_chat_routes 404: an engine build without the Chat tab's server
 *  setup_mode     503 `setup_mode`: the box is in the Setup Guide
 *  not_an_engine  200 that is not JSON: a static server or a proxy's index.html
 *  server_error   any other status (a proxy's 502 page included)
 *  offline        the fetch failed: no server reached */
export type ChatDownCause = 'chatd_down' | 'chatd_refused' | 'no_chat_routes' | 'setup_mode' | 'not_an_engine' | 'server_error' | 'offline';
export interface ChatDown {
  cause: ChatDownCause;
  /** the HTTP status, null when nothing answered */
  status: number | null;
  /** the server's own message, shown as it wrote it ('' when there is none) */
  message: string;
}

export type BackendChoice = { backend: ChatBackend; reason: 'forced' | 'probed' | 'dev'; down?: undefined } | { backend: null; reason: 'unavailable'; down: ChatDown };

/** The probe's answer as a state: null is the live chat API. */
export function classifyProbe(status: number, json: boolean, body: unknown): ChatDown | null {
  const err = body && typeof body === 'object' ? (body as { error?: { type?: unknown; message?: unknown } }).error : undefined;
  const type = typeof err?.type === 'string' ? err.type : '';
  const message = typeof err?.message === 'string' ? err.message : '';
  const at = (cause: ChatDownCause): ChatDown => ({ cause, status, message });
  if (status >= 200 && status < 300) return json ? null : at('not_an_engine');
  if (status === 502 && type === 'chat_unavailable') return at('chatd_down');
  if (status === 403 && type === 'forbidden') return at('chatd_refused');
  if (status === 404) return at('no_chat_routes');
  if (status === 503 && type === 'setup_mode') return at('setup_mode');
  return at('server_error');
}

/** A failed call of the live backend (remote.ts's ChatApiError: `status`, `body`) that says chatd
 *  went away: the tab shows its banner and probes again, instead of a toast per call. */
export function isChatDown(e: unknown): boolean {
  const x = e as { status?: unknown; body?: unknown } | null;
  return !!x && x.status === 502 && classifyProbe(502, true, x.body)?.cause === 'chatd_down';
}

/** GET /v1/chat-app/capabilities: null when the chat API answers, else why not. */
export async function probeChat(): Promise<ChatDown | null> {
  let res: Response;
  try {
    res = await fetch(apiBase() + '/v1/chat-app/capabilities', { credentials: 'same-origin', cache: 'no-store', headers: { Accept: 'application/json' } });
  } catch (e) {
    return { cause: 'offline', status: null, message: (e as Error)?.message ?? '' };
  }
  let json = (res.headers.get('Content-Type') ?? '').includes('json');
  let body: unknown = null;
  if (json) {
    try {
      body = await res.json();
    } catch {
      json = false; // a JSON type with no JSON in it is no chat API either
    }
  }
  return classifyProbe(res.status, json, body);
}

/** `?chat=mock` forces the mock; else the server's chat routes are probed. Only the Vite dev
 *  server (`dev`), which has no chat routes, falls back to the mock (with its badge); anywhere else
 *  a probe that does not find the chat API is a state the tab names. */
export async function pickBackend(params: URLSearchParams, load: { mock: () => Promise<ChatBackend>; remote: () => Promise<ChatBackend> }, dev: boolean = import.meta.env.DEV): Promise<BackendChoice> {
  if (params.get('chat') === 'mock') return { backend: await load.mock(), reason: 'forced' };
  const down = await probeChat();
  if (!down) return { backend: await load.remote(), reason: 'probed' };
  if (dev && (down.cause === 'no_chat_routes' || down.cause === 'not_an_engine')) return { backend: await load.mock(), reason: 'dev' };
  return { backend: null, reason: 'unavailable', down };
}
