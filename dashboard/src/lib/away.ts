// OPS-65: the shell's offline banner in words, from what the reverse proxy's status responder
// (ops/engine_status.py) said about an engine that is away; the old line when it said nothing.
import type { EngineAway } from '../api/client';

export function awayText(a: EngineAway | null, now: (s: number) => string = (s) => new Date(s * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })): string {
  if (!a) return 'The engine is not answering';
  const since = a.since ? ` since ${now(a.since)}` : '';
  const left = a.left_s == null ? '' : a.left_s >= 90 ? `, back in about ${Math.round(a.left_s / 60)} min` : a.left_s > 0 ? `, back in about ${Math.max(1, Math.round(a.left_s))} s` : ', back any moment';
  return `TandemLLM is ${a.words}${since}${a.reason ? ` (${a.reason})` : ''}${left}`;
}

/** OPS-131: what the service worker says right after the page loaded (`./sw-away`, pwa/sw.js): it
 *  served the cached shell because the reverse proxy answered 502, 503 or 504 for the page (status
 *  0: the network failed). null: the page came from the network, or no worker answered. */
export async function swAway(fetchImpl: typeof fetch = fetch): Promise<{ status: number; at: number } | null> {
  try {
    const r = await fetchImpl('./sw-away', { cache: 'no-store', credentials: 'same-origin' });
    if (!r.ok || !(r.headers.get('Content-Type') ?? '').includes('json')) return null;
    const b = (await r.json()) as { away?: boolean; status?: number; at?: number };
    return b?.away ? { status: Number(b.status) || 0, at: Number(b.at) || Date.now() } : null;
  } catch {
    return null;
  }
}

/** OPS-131: the banner of a page served from the cache while the engine restarts. */
export function restartText(status: number): string {
  return status
    ? `The engine is restarting (the proxy answered ${status}). This page reloads when it answers.`
    : 'The engine cannot be reached. This page reloads when it answers.';
}

/** OPS-131: ask `health` every `everyMs` until it says yes, then `onUp` once (the page reloads).
 *  The stop function ends it. */
export function whenEngineUp(health: () => Promise<boolean>, onUp: () => void, everyMs = 3000): () => void {
  let stopped = false;
  let t: ReturnType<typeof setTimeout> | null = null;
  const look = async () => {
    if (stopped) return;
    if (await health().catch(() => false)) {
      if (!stopped) onUp();
      stopped = true;
      return;
    }
    if (!stopped) t = setTimeout(() => void look(), everyMs);
  };
  void look();
  return () => {
    stopped = true;
    if (t) clearTimeout(t);
  };
}
