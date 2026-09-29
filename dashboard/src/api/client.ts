// The dashboard's only way to the engine: same-origin fetches with the session cookie. A 401
// anywhere raises `unauthorized` on the app bus so the shell returns to the token screen; a
// network failure raises `offline`. Every error names the endpoint it came from.

import type { Bucket, Live, LogsJson, Requests, SessionInfo, Summary, SystemInfo, Usage } from './types';

export class ApiError extends Error {
  constructor(
    public endpoint: string,
    public status: number,
    message: string,
    public type: string = 'error',
  ) {
    super(message);
  }
}

type Listener = (ev: 'unauthorized' | 'offline' | 'online') => void;
const listeners = new Set<Listener>();
export function onApi(fn: Listener): () => void {
  listeners.add(fn);
  return () => listeners.delete(fn);
}
function emit(ev: 'unauthorized' | 'offline' | 'online') {
  for (const l of listeners) l(ev);
}

let base = '';
/** For tests / a different origin. Default: same origin. */
export function setApiBase(b: string): void {
  base = b.replace(/\/$/, '');
}

async function request<T>(endpoint: string, init: RequestInit = {}, parse: 'json' | 'text' | 'none' = 'json'): Promise<T> {
  let res: Response;
  try {
    res = await fetch(base + endpoint, { credentials: 'same-origin', cache: 'no-store', ...init });
  } catch (e) {
    emit('offline');
    throw new ApiError(endpoint, 0, `engine unreachable (${(e as Error).message})`, 'offline');
  }
  emit('online');
  if (res.status === 401) {
    emit('unauthorized');
    throw new ApiError(endpoint, 401, 'session expired', 'unauthorized');
  }
  if (!res.ok) {
    let msg = `HTTP ${res.status}`;
    let type = 'error';
    try {
      const body = await res.json();
      if (body?.error?.message) msg = body.error.message;
      if (body?.error?.type) type = body.error.type;
    } catch {
      /* not json */
    }
    throw new ApiError(endpoint, res.status, msg, type);
  }
  if (parse === 'none') return undefined as T;
  if (parse === 'text') return (await res.text()) as T;
  return (await res.json()) as T;
}

export const api = {
  session: {
    get: () => request<SessionInfo>('/v1/dashboard/session'),
    login: (token: string) =>
      request<void>('/v1/dashboard/session', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ token }) }, 'none'),
    logout: () => request<void>('/v1/dashboard/session', { method: 'DELETE' }, 'none'),
  },
  summary: (tz: string) => request<Summary>(`/v1/dashboard/summary?tz=${encodeURIComponent(tz)}`),
  usage: (p: { from?: string; to?: string; bucket: Bucket; tz: string; model?: string | null; client?: string | null }) => {
    const q = new URLSearchParams();
    if (p.from) q.set('from', p.from);
    if (p.to) q.set('to', p.to);
    q.set('bucket', p.bucket);
    q.set('tz', p.tz);
    if (p.model) q.set('model', p.model);
    if (p.client) q.set('client', p.client);
    return request<Usage>(`/v1/dashboard/usage?${q}`);
  },
  requests: (p: { limit?: number; before?: number | null; model?: string | null; client?: string | null; finish?: string | null } = {}) => {
    const q = new URLSearchParams();
    q.set('limit', String(p.limit ?? 50));
    if (p.before != null) q.set('before', String(p.before));
    if (p.model) q.set('model', p.model);
    if (p.client) q.set('client', p.client);
    if (p.finish) q.set('finish', p.finish);
    return request<Requests>(`/v1/dashboard/requests?${q}`);
  },
  system: () => request<SystemInfo>('/v1/dashboard/system'),
  // /metrics under the dashboard's own access rule (the same page), so it also works with the login off
  metrics: () => request<string>('/v1/dashboard/metrics', { headers: { Accept: 'text/plain' } }, 'text'),
  logsJson: (p: { level?: string; since?: number | null; grep?: string | null; backlog?: number } = {}) => {
    const q = new URLSearchParams({ follow: '0' });
    if (p.level) q.set('level', p.level);
    if (p.since != null) q.set('since', String(p.since));
    if (p.grep) q.set('grep', p.grep);
    if (p.backlog) q.set('backlog', String(p.backlog));
    return request<LogsJson>(`/v1/dashboard/logs?${q}`);
  },
  logsUrl: (p: { level?: string; since?: string | null; grep?: string | null; backlog?: number }) => {
    const q = new URLSearchParams({ follow: '1' });
    if (p.level) q.set('level', p.level);
    if (p.since) q.set('since', p.since);
    if (p.grep) q.set('grep', p.grep);
    if (p.backlog) q.set('backlog', String(p.backlog));
    return `${base}/v1/dashboard/logs?${q}`;
  },
  live: () => request<Live>('/v1/dashboard/live?follow=0'),
  liveUrl: () => `${base}/v1/dashboard/live?follow=1`,
  chatUrl: () => `${base}/v1/chat/completions`,
};

export function describeError(e: unknown): { endpoint: string; message: string; status: number } {
  if (e instanceof ApiError) return { endpoint: e.endpoint, message: e.message, status: e.status };
  return { endpoint: '', message: (e as Error)?.message ?? String(e), status: 0 };
}
