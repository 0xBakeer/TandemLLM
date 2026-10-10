// /v1/setup/* (server/setup_api.py, contract 1.0): the setup guide's routes. Same origin, same error
// handling as the rest of the dashboard (api/client.ts); the chat is a POST, so its SSE is read off
// fetch's body rather than an EventSource.

import { apiBase, ApiError, request } from '../api/client';
import type {
  SetupApplyRequest,
  SetupApplyResponse,
  SetupAssistantStatus,
  SetupChatEvent,
  SetupChatRequest,
  SetupDevice,
  SetupEstimate,
  SetupJob,
  SetupModel,
  SetupModelsList,
  SetupPlan,
  SetupResetLevels,
  SetupResetRequest,
  SetupResetResult,
  SetupSelection,
} from '../api/types';
import { SseParser } from '../lib/sse';

const post = <T>(path: string, body: unknown, signal?: AbortSignal) =>
  request<T>(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), signal });

export const setupApi = {
  models: () => request<SetupModelsList>('/v1/setup/models'),
  model: (id: string) => request<SetupModel>(`/v1/setup/models/${encodeURIComponent(id)}`),
  device: () => request<SetupDevice>('/v1/setup/device'),
  estimate: (sel: SetupSelection, signal?: AbortSignal) => post<SetupEstimate>('/v1/setup/estimate', sel, signal),
  plan: (sel: SetupSelection, signal?: AbortSignal) => post<SetupPlan>('/v1/setup/plan', { selection: sel }, signal),
  apply: (body: SetupApplyRequest) => post<SetupApplyResponse>('/v1/setup/apply', body),
  job: (id: string) => request<SetupJob>(`/v1/setup/jobs/${encodeURIComponent(id)}?format=json`),
  jobUrl: (id: string) => `${apiBase()}/v1/setup/jobs/${encodeURIComponent(id)}`,
  resetLevels: () => request<SetupResetLevels>('/v1/setup/reset'),
  reset: (body: SetupResetRequest) => post<SetupResetResult>('/v1/setup/reset', body),
  assistant: {
    status: () => request<SetupAssistantStatus>('/v1/setup/assistant/status'),
    load: () => post<SetupAssistantStatus>('/v1/setup/assistant/load', {}),
    unload: () => post<SetupAssistantStatus>('/v1/setup/assistant/unload', {}),
  },
  health: () => request<{ status: string; mode?: string }>('/health'),
};

/** POST /v1/setup/assistant/chat and hand each event to `onEvent` until the stream ends. */
export async function streamChat(body: SetupChatRequest, onEvent: (e: SetupChatEvent) => void, signal: AbortSignal): Promise<void> {
  const endpoint = '/v1/setup/assistant/chat';
  let res: Response;
  try {
    res = await fetch(apiBase() + endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
      body: JSON.stringify(body),
      credentials: 'same-origin',
      cache: 'no-store',
      signal,
    });
  } catch (e) {
    if ((e as Error).name === 'AbortError') throw e;
    throw new ApiError(endpoint, 0, (e as Error).message, 'offline');
  }
  if (!res.ok || !res.body) {
    let msg = `HTTP ${res.status}`;
    let type = 'error';
    try {
      const j = await res.json();
      msg = j?.error?.message ?? msg;
      type = j?.error?.type ?? type;
    } catch {
      /* not json */
    }
    throw new ApiError(endpoint, res.status, msg, type);
  }
  const parser = new SseParser();
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    for (const ev of parser.feed(dec.decode(value, { stream: true }))) {
      if (ev.data === '[DONE]') return;
      let parsed: SetupChatEvent;
      try {
        parsed = JSON.parse(ev.data);
      } catch {
        continue;
      }
      onEvent(parsed);
    }
  }
}
