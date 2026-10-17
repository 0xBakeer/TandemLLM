// /v1/setup/* (server/setup_api.py, contract 1.0): the setup guide's routes. Same origin, same error
// handling as the rest of the dashboard (api/client.ts); the chat is a POST, so its SSE is read off
// fetch's body rather than an EventSource. A 401 from a setup route (a box with an admin password)
// also raises `qse-setup-unauthorized` on window, and the page shows its sign-in (signin.ts).

import { apiBase, ApiError, request } from '../api/client';
import type {
  SetupApplyRequest,
  SetupApplyResponse,
  SetupAssistantStatus,
  SetupChatEvent,
  SetupChatRequest,
  CommunityBody,
  SetupCurrent,
  SetupDevice,
  SetupEstimate,
  SetupInstalledWeights,
  SetupRemoveResult,
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

export function setupUnauthorized(): void {
  window.dispatchEvent(new CustomEvent('qse-setup-unauthorized'));
}

const get = <T>(path: string): Promise<T> =>
  request<T>(path).catch((e) => {
    if (e instanceof ApiError && e.status === 401) setupUnauthorized();
    throw e;
  });
const post = <T>(path: string, body: unknown, signal?: AbortSignal): Promise<T> =>
  request<T>(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), signal }).catch((e) => {
    if (e instanceof ApiError && e.status === 401) setupUnauthorized();
    throw e;
  });

export const setupApi = {
  models: () => get<SetupModelsList>('/v1/setup/models'),
  model: (id: string) => get<SetupModel>(`/v1/setup/models/${encodeURIComponent(id)}`),
  current: (id: string) => get<SetupCurrent>(`/v1/setup/models/${encodeURIComponent(id)}/current`),
  device: () => get<SetupDevice>('/v1/setup/device'),
  estimate: (sel: SetupSelection, signal?: AbortSignal) => post<SetupEstimate>('/v1/setup/estimate', sel, signal),
  plan: (sel: SetupSelection, signal?: AbortSignal) => post<SetupPlan>('/v1/setup/plan', { selection: sel }, signal),
  apply: (body: SetupApplyRequest) => post<SetupApplyResponse>('/v1/setup/apply', body),
  job: (id: string) => get<SetupJob>(`/v1/setup/jobs/${encodeURIComponent(id)}?format=json`),
  jobUrl: (id: string) => `${apiBase()}/v1/setup/jobs/${encodeURIComponent(id)}`,
  /** SRV-560: the weights on the box, and removing some */
  weights: () => get<SetupInstalledWeights>('/v1/setup/weights'),
  remove: (profiles: string[]) => post<SetupRemoveResult>('/v1/setup/remove', { profiles, confirm: true }),
  /** VIS-213: a failed or rolled back job again, from the failed step's side (the download is kept) */
  retry: (id: string) => post<SetupApplyResponse>(`/v1/setup/jobs/${encodeURIComponent(id)}/retry`, {}),
  /** VIS-213: the job as text for a bug report (no secret) */
  diagnostics: (id: string) =>
    request<string>(`/v1/setup/jobs/${encodeURIComponent(id)}/diagnostics`, {}, 'text').catch((e) => {
      if (e instanceof ApiError && e.status === 401) setupUnauthorized();
      throw e;
    }),
  /** the community list, fetched from GitHub now (CONTRACT.md addition 3: on the weights section's
   *  open and on Refresh only; one conditional GET, 10 s); the cache or the bundled copy when GitHub
   *  does not answer, with `sync.error` */
  communitySync: (signal?: AbortSignal) => post<CommunityBody>('/v1/setup/community/sync', {}, signal),
  community: () => get<CommunityBody>('/v1/setup/community'),
  resetLevels: () => get<SetupResetLevels>('/v1/setup/reset'),
  reset: (body: SetupResetRequest) => post<SetupResetResult>('/v1/setup/reset', body),
  assistant: {
    status: () => get<SetupAssistantStatus>('/v1/setup/assistant/status'),
    load: () => post<SetupAssistantStatus>('/v1/setup/assistant/load', {}),
    unload: () => post<SetupAssistantStatus>('/v1/setup/assistant/unload', {}),
  },
  health: () => request<{ status: string; mode?: string }>('/health'),
  /** the sign-in after a 401 (signin.ts): 204 and the session cookie; 404 with the dashboard login off */
  signIn: (password: string) => request<void>('/v1/dashboard/session', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ token: password }) }, 'none'),
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
  if (res.status === 401) setupUnauthorized();
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
