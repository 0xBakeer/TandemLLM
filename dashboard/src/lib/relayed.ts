// The engine's own requests beside the ones the proxy relays to another backend (a cloud model,
// 0.11.3): the Performance and Usage views draw them side by side, never pooled. A DeepSeek call
// decodes at 200-400 tok/s, the engine at 30-45: pooled, the engine's p50 jumped to 300 in an
// hour of a cloud chat and the cloud dots sat above the scatter's axis as "cold" (2026-10-17).

import type { BackendRef, RequestRow, Usage } from '../api/types';

/** the `backend=` filter's value for the engine's own rows (server/dashboard_api.py) */
export const ENGINE = 'engine';
/** at most this many relayed backends get a line of their own */
export const MAX_RELAYED = 4;

/** the backends of a usage answer other than the engine, the ones a view draws beside it */
export function relayedBackends(u: Pick<Usage, 'dimensions'> | null | undefined): BackendRef[] {
  return (u?.dimensions.backends ?? []).filter((b) => b.id !== ENGINE).slice(0, MAX_RELAYED);
}

/** a backend's name in a legend: its label (`DeepSeek (cloud)`), else its id */
export function backendName(b: Pick<BackendRef, 'id' | 'label'> | null | undefined): string {
  return b ? (b.label ?? b.id) : 'this engine';
}

/** the scatter's group of a request: the cache tier for the engine's own, `relayed` otherwise */
export function scatterGroup(r: Pick<RequestRow, 'backend'>, tier: string): string {
  return r.backend ? 'relayed' : tier;
}

export interface BackendRow {
  id: string;
  name: string;
  kind: string;
  requests: number;
  prompt: number;
  cached: number;
  completion: number;
  decodeP50: number | null;
  ttftP50: number | null;
  /** micro-dollars; null when no row had a price (the engine's own rows) */
  cost: number | null;
}

/** the comparison table: the engine first, then each relayed backend, from their usage answers */
export function compareRows(engine: Usage | null, relayed: { ref: BackendRef; usage: Usage }[]): BackendRow[] {
  const row = (id: string, name: string, kind: string, u: Usage): BackendRow => ({
    id,
    name,
    kind,
    requests: u.totals.requests,
    prompt: u.totals.prompt_tokens,
    cached: u.totals.cached_tokens,
    completion: u.totals.completion_tokens,
    decodeP50: u.totals.decode_tps_p50,
    ttftP50: u.totals.ttft_ms_p50,
    cost: u.totals.cost_micros ?? null,
  });
  const out: BackendRow[] = [];
  if (engine) out.push(row(ENGINE, 'this engine', 'engine', engine));
  for (const r of relayed) out.push(row(r.ref.id, backendName(r.ref), r.ref.kind ?? 'proxy', r.usage));
  return out;
}
