// What the Chat tab needs from a server, as one interface. Two implementations: remote.ts
// speaks the contract in CONTRACT.md; mock.ts plays it in the browser so the tab runs with no
// backend. pickBackend() probes the server once and falls back to the mock with a badge.

import { apiBase } from '../api/client';
import type { Attachment, Capabilities, ChatSettings, Conversation, ConversationSummary, ConversationUi, Folder, McpServer, McpServerDraft, PackageInfo, Preset, Prompt, SandboxFile, SandboxStatus, StampedEvent, ToolInfo, ToolPolicy } from './types';
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
  compact(id: string): Promise<void>;

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
  mcpServers(): Promise<McpServer[]>;
  addMcpServer(draft: McpServerDraft): Promise<McpServer>;
  removeMcpServer(serverId: string): Promise<void>;
  testMcpServer(draft: McpServerDraft): Promise<{ ok: boolean; message: string; tools: { name: string; description: string }[] }>;
  restartMcpServer(serverId: string): Promise<McpServer>;
  prompts(): Promise<Prompt[]>;
  savePrompt(p: Omit<Prompt, 'id'> & { id?: string }): Promise<Prompt>;
  deletePrompt(promptId: string): Promise<void>;
  presets(): Promise<Preset[]>;
  savePreset(p: Omit<Preset, 'id'> & { id?: string }): Promise<Preset>;
  deletePreset(presetId: string): Promise<void>;

  // the sandbox
  sandbox(id: string): Promise<SandboxStatus>;
  resetSandbox(id: string): Promise<SandboxStatus>;
  sandboxFiles(id: string, path: string): Promise<SandboxFile[]>;
  readFile(id: string, path: string): Promise<{ content: string; mime: string; bytes: number; etag?: string }>;
  /** PUT with If-Match; a changed file throws */
  writeFile(id: string, path: string, content: string, etag?: string): Promise<void>;
  /** an artifact the model produced (an html preview, a diff, an image) */
  /** `url`: a same-origin address the preview loads in its sandboxed iframe (the server's files) */
  artifact(id: string, ref: string): Promise<{ kind: string; content: string; mime: string; title: string; url?: string }>;
  terminal(id: string, session: string, onData: (data: string) => void): TerminalSession;
}

export type BackendChoice = { backend: ChatBackend; reason: 'forced' | 'probed' | 'fallback' };

/** `?chat=mock` forces the mock; else the server's chat routes are probed once (a 404 or a
 *  network failure means the mock, with the badge). */
export async function pickBackend(params: URLSearchParams, load: { mock: () => Promise<ChatBackend>; remote: () => Promise<ChatBackend> }): Promise<BackendChoice> {
  if (params.get('chat') === 'mock') return { backend: await load.mock(), reason: 'forced' };
  try {
    const res = await fetch(apiBase() + '/v1/chat-app/capabilities', { credentials: 'same-origin', cache: 'no-store', headers: { Accept: 'application/json' } });
    // a 200 that is not JSON is a static fallback (vite's index.html), not a chat server
    if (res.ok && (res.headers.get('Content-Type') ?? '').includes('json')) return { backend: await load.remote(), reason: 'probed' };
  } catch {
    /* no server: the mock */
  }
  return { backend: await load.mock(), reason: 'fallback' };
}
