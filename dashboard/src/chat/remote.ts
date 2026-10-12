// The remote chat backend: CONTRACT.md v0.2 over /v1/chat-app. Same-origin fetches with the
// dashboard's credentials; the event stream is the resumable SSE client from lib/sse.ts with
// Last-Event-ID; the terminal is a WebSocket (binary frames = PTY bytes, text frames = resize).

import { apiBase } from '../api/client';
import { ResumableStream } from '../lib/sse';
import { VersionConflict, type ChatBackend, type SendOptions, type StreamHandler, type Subscription, type TerminalSession } from './backend';
import type { Attachment, Capabilities, ChatEvent, ChatSettings, Conversation, ConversationSummary, ConversationUi, Folder, McpServer, McpServerDraft, PackageInfo, Preset, Prompt, SandboxFile, SandboxStatus, ToolInfo, ToolPolicy, UserToolkit } from './types';

const PREFIX = '/v1/chat-app';

export class ChatApiError extends Error {
  constructor(
    public status: number,
    message: string,
    public body: unknown = null,
  ) {
    super(message);
  }
}

async function call<T>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await fetch(apiBase() + PREFIX + path, { credentials: 'same-origin', cache: 'no-store', ...init, headers: { Accept: 'application/json', ...(init.body && !(init.body instanceof FormData) ? { 'Content-Type': 'application/json' } : {}), ...(init.headers ?? {}) } });
  if (res.status === 204) return undefined as T;
  let body: unknown = null;
  const text = await res.text();
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    body = text;
  }
  if (res.status === 409 && body && typeof body === 'object' && 'current' in (body as Record<string, unknown>)) throw new VersionConflict((body as { current: ConversationSummary }).current);
  if (res.status === 401) window.dispatchEvent(new CustomEvent('qse-unauthorized'));
  if (!res.ok) {
    const msg = (body as { error?: { message?: string } } | null)?.error?.message ?? `HTTP ${res.status}`;
    throw new ChatApiError(res.status, msg, body);
  }
  return body as T;
}

const post = <T>(path: string, body?: unknown) => call<T>(path, { method: 'POST', body: body === undefined ? undefined : JSON.stringify(body) });
const patch = <T>(path: string, body: unknown) => call<T>(path, { method: 'PATCH', body: JSON.stringify(body) });
const del = (path: string) => call<void>(path, { method: 'DELETE' });

export class RemoteChatBackend implements ChatBackend {
  readonly kind = 'remote' as const;

  capabilities() {
    return call<Capabilities>('/capabilities');
  }
  async signIn(password: string) {
    await post('/session', { password });
    window.dispatchEvent(new CustomEvent('qse-signed-in'));
  }
  signOut() {
    return del('/session');
  }
  approvePlan(id: string, items?: string[]) {
    return call<void>(`/conversations/${enc(id)}/plan/approve`, { method: 'POST', body: JSON.stringify(items ? { items } : {}) });
  }
  rewind(id: string, messageId: string) {
    return call<{ conversation: Conversation; text: string; files: number }>(`/conversations/${enc(id)}/rewind`, { method: 'POST', body: JSON.stringify({ message_id: messageId }) });
  }
  uploadSkill(file: File) {
    return call<PackageInfo>('/packages/upload', { method: 'POST', body: form(file) });
  }
  removeSkill(pkgId: string) {
    return call<void>(`/packages/${enc(pkgId)}`, { method: 'DELETE' });
  }
  importOwui(file: File) {
    return call<{ toolkits: UserToolkit[]; tools: ToolInfo[] }>('/tools/import/owui', { method: 'POST', body: form(file) });
  }
  userToolkits() {
    return call<UserToolkit[]>('/tools/user');
  }
  patchUserToolkit(id: string, patch: { enabled?: boolean; valves?: Record<string, unknown> }) {
    return call<UserToolkit>(`/tools/user/${enc(id)}`, { method: 'PATCH', body: JSON.stringify(patch) });
  }
  removeUserToolkit(id: string) {
    return call<void>(`/tools/user/${enc(id)}`, { method: 'DELETE' });
  }

  async uploadAttachment(file: File) {
    return call<Attachment>('/attachments', { method: 'POST', body: form(file) });
  }

  listConversations() {
    return call<{ conversations: ConversationSummary[]; folders: Folder[] }>('/conversations?archived=0');
  }
  getConversation(id: string) {
    return call<Conversation>(`/conversations/${enc(id)}`);
  }
  createConversation(settings?: Partial<ChatSettings>) {
    return post<Conversation>('/conversations', settings ? { settings } : {});
  }
  updateConversation(id: string, p: { version?: number; title?: string; pinned?: boolean; archived?: boolean; folder_id?: string | null; ui?: ConversationUi }) {
    return patch<ConversationSummary>(`/conversations/${enc(id)}`, p);
  }
  deleteConversation(id: string) {
    return del(`/conversations/${enc(id)}`);
  }
  createFolder(name: string) {
    return post<Folder>('/folders', { name });
  }

  subscribe(id: string, lastEventId: string | null, h: StreamHandler): Subscription {
    // a connection with an id replays what it missed before `event: current`; those events reach
    // the view as replays (no pane opens, no reload for a branch change it already has)
    let replaying = false;
    const stream = new ResumableStream({
      url: (last) => `${apiBase()}${PREFIX}/conversations/${enc(id)}/events${last ? `?after=${encodeURIComponent(last)}` : ''}`,
      onEvent: (ev) => {
        if (ev.event === 'current') {
          replaying = false;
          return h.onCurrent();
        }
        if (ev.event === 'reset') return h.onState('error', 'reset');
        let data: ChatEvent;
        try {
          data = JSON.parse(ev.data) as ChatEvent;
        } catch {
          return;
        }
        const type = (ev.event && ev.event !== 'message' ? ev.event : (data as { type?: string }).type) as ChatEvent['type'];
        h.onEvent({ id: ev.id ?? '', event: { ...data, type } as ChatEvent }, replaying);
      },
      onState: (s, d) => {
        if (s === 'open') replaying = stream.lastId != null;
        h.onState(s, d);
      },
    });
    stream.lastId = lastEventId;
    stream.start();
    return { close: () => stream.close() };
  }

  async send(id: string, msg: SendOptions) {
    await post(`/conversations/${enc(id)}/messages`, { client_msg_id: msg.client_msg_id, text: msg.text, attachments: msg.attachments?.map((a) => a.id) ?? [] });
  }
  async queue(id: string, text: string, clientMsgId: string) {
    await post(`/conversations/${enc(id)}/queue`, { client_msg_id: clientMsgId, text });
  }
  unqueue(id: string, qid: string) {
    return del(`/conversations/${enc(id)}/queue/${enc(qid)}`);
  }
  async sendNow(id: string, qid: string) {
    await post(`/conversations/${enc(id)}/queue/${enc(qid)}/now`);
  }
  async stop(id: string) {
    await post(`/conversations/${enc(id)}/stop`);
  }
  async continueRun(id: string) {
    await post(`/conversations/${enc(id)}/continue`);
  }
  async regenerate(id: string, mid: string) {
    await post(`/conversations/${enc(id)}/messages/${enc(mid)}/regenerate`, {});
  }
  async edit(id: string, mid: string, text: string, clientMsgId: string) {
    await post(`/conversations/${enc(id)}/messages/${enc(mid)}/edit`, { client_msg_id: clientMsgId, text });
  }
  switchBranch(id: string, messageId: string, index: number) {
    return post<Conversation>(`/conversations/${enc(id)}/branch`, { message_id: messageId, index });
  }
  async permission(id: string, callId: string, decision: 'once' | 'always' | 'deny', scope: 'chat' | 'all' = 'chat') {
    await post(`/conversations/${enc(id)}/permissions/${enc(callId)}`, { decision, scope });
  }
  updateSettings(id: string, p: Partial<ChatSettings>) {
    return patch<ChatSettings>(`/conversations/${enc(id)}/settings`, p);
  }
  async compact(id: string) {
    await post(`/conversations/${enc(id)}/compact`);
  }

  tools() {
    return call<ToolInfo[]>('/tools');
  }
  setTool(toolId: string, p: { enabled?: boolean; policy?: ToolPolicy }) {
    return patch<ToolInfo>(`/tools/${enc(toolId)}`, p);
  }
  packages() {
    return call<PackageInfo[]>('/packages');
  }
  setPackage(pkgId: string, enabled: boolean) {
    return patch<PackageInfo>(`/packages/${enc(pkgId)}`, { enabled });
  }
  mcpServers() {
    return call<McpServer[]>('/mcp/servers');
  }
  addMcpServer(draft: McpServerDraft) {
    return post<McpServer>('/mcp/servers', draft);
  }
  removeMcpServer(sid: string) {
    return del(`/mcp/servers/${enc(sid)}`);
  }
  testMcpServer(draft: McpServerDraft) {
    return post<{ ok: boolean; message: string; tools: { name: string; description: string }[] }>('/mcp/servers/test', draft);
  }
  restartMcpServer(sid: string) {
    return post<McpServer>(`/mcp/servers/${enc(sid)}/restart`);
  }
  prompts() {
    return call<Prompt[]>('/prompts');
  }
  savePrompt(p: Omit<Prompt, 'id'> & { id?: string }) {
    return post<Prompt>(p.id ? `/prompts/${enc(p.id)}` : '/prompts', p);
  }
  deletePrompt(pid: string) {
    return del(`/prompts/${enc(pid)}`);
  }
  presets() {
    return call<Preset[]>('/presets');
  }
  savePreset(p: Omit<Preset, 'id'> & { id?: string }) {
    return post<Preset>(p.id ? `/presets/${enc(p.id)}` : '/presets', p);
  }
  deletePreset(pid: string) {
    return del(`/presets/${enc(pid)}`);
  }

  sandbox(id: string) {
    return call<SandboxStatus>(`/conversations/${enc(id)}/sandbox`);
  }
  resetSandbox(id: string) {
    return post<SandboxStatus>(`/conversations/${enc(id)}/sandbox/reset`);
  }
  sandboxFiles(id: string, path: string) {
    return call<SandboxFile[]>(`/conversations/${enc(id)}/sandbox/files?path=${encodeURIComponent(path)}`);
  }
  readFile(id: string, path: string) {
    return call<{ content: string; mime: string; bytes: number; etag?: string }>(`/conversations/${enc(id)}/sandbox/file?path=${encodeURIComponent(path)}`);
  }
  async writeFile(id: string, path: string, content: string, etag?: string) {
    await call<void>(`/conversations/${enc(id)}/sandbox/file?path=${encodeURIComponent(path)}`, { method: 'PUT', body: content, headers: { 'Content-Type': 'text/plain; charset=utf-8', ...(etag ? { 'If-Match': etag } : {}) } });
  }
  artifact(id: string, ref: string) {
    return call<{ kind: string; content: string; mime: string; title: string; url?: string }>(`/conversations/${enc(id)}/artifacts/${enc(ref)}`);
  }

  terminal(id: string, session: string, onData: (data: string | Uint8Array) => void): TerminalSession {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const base = apiBase() ? apiBase().replace(/^http/, 'ws') : `${proto}//${location.host}`;
    const ws = new WebSocket(`${base}${PREFIX}/conversations/${enc(id)}/terminals/${enc(session)}/ws`);
    ws.binaryType = 'arraybuffer';
    // bytes go to xterm as they are: a UTF-8 character split across two frames stays whole
    ws.onmessage = (e) => onData(typeof e.data === 'string' ? e.data : new Uint8Array(e.data as ArrayBuffer));
    ws.onclose = () => onData('\r\n\x1b[2m[the terminal closed]\x1b[0m\r\n');
    const enc8 = new TextEncoder();
    return {
      write: (data) => ws.readyState === WebSocket.OPEN && ws.send(enc8.encode(data)),
      resize: (cols, rows) => ws.readyState === WebSocket.OPEN && ws.send(JSON.stringify({ type: 'resize', cols, rows })),
      close: () => ws.close(),
    };
  }
}

function form(file: File): FormData {
  const fd = new FormData();
  fd.append('file', file, file.name);
  return fd;
}

function enc(s: string): string {
  return encodeURIComponent(s);
}
