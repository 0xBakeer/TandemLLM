// Where a request's prompt came from: the ledger's and the live view's
// `cache_source`, as the engine names it, read as the tier it sits in. RAM: the rows in place
// (`resident`, a pinned head or a parked session come back as resident too), the end of the last
// generation (`end`), the state store (`prefix`, `session`); the NVMe (`disk`, and `spill`: a whole
// conversation read back from it); the response cache
// (`response`, an identical greedy request answered from memory); `none` is a cold prefill.
// A compaction served as its conversation's continuation (`continuation`, read as the
// end anchor it resumes at) or answered from a summary written ahead (`prepared`, read as the
// response cache: no prefill).

import type { CacheSource } from '../api/types';
import type { SeriesColor } from '../charts/chart-base';

export type CacheTier = 'end' | 'resident' | 'store' | 'disk' | 'response' | 'cold';

export const TIERS: CacheTier[] = ['end', 'resident', 'store', 'disk', 'cold', 'response'];

export const TIER_LABEL: Record<CacheTier, string> = {
  end: 'RAM end anchor',
  resident: 'RAM resident',
  store: 'RAM state store',
  disk: 'NVMe disk',
  response: 'response cache',
  cold: 'cold',
};

export const TIER_COLOR: Record<CacheTier, SeriesColor> = {
  end: 'aqua',
  resident: 'cobalt',
  store: 'violet',
  disk: 'yellow',
  response: 'muted',
  cold: 'orange',
};

export function cacheTier(src: CacheSource | string | null | undefined): CacheTier {
  switch (src) {
    case 'end':
    case 'continuation':
      return 'end';
    case 'resident':
    case 'pin':
    case 'park':
      return 'resident';
    case 'prefix':
    case 'session':
      return 'store';
    case 'disk':
    case 'spill':
      return 'disk';
    case 'response':
    case 'prepared':
      return 'response';
    default:
      return 'cold';
  }
}

/** Answered from memory with nothing decoded: the response cache and a summary written ahead.
 * A replay's "decode rate" is the speed of a copy, so no speed view plots or averages it. */
export function isReplay(src: CacheSource | string | null | undefined): boolean {
  return cacheTier(src) === 'response';
}

export interface Reuse {
  reused: number | null;
  forwarded: number | null;
}

/** Reused and forwarded prompt tokens; `forwarded` from the server when it sends it. */
export function reuse(prompt: number | null | undefined, cached: number | null | undefined, forwarded?: number | null): Reuse {
  if (prompt == null) return { reused: null, forwarded: null };
  const r = Math.max(0, Math.min(prompt, cached ?? 0));
  return { reused: r, forwarded: forwarded ?? prompt - r };
}

export interface TierRow {
  tier: CacheTier;
  label: string;
  requests: number;
  reused: number;
  forwarded: number;
  /** TTFT p50 of the tier's requests, ms; null when none reported one */
  ttftP50: number | null;
}

export interface CacheSummary {
  rows: TierRow[];
  requests: number;
  reused: number;
  forwarded: number;
  /** the share of prompt tokens that came out of a cache */
  reusedShare: number | null;
}

interface SummaryInput {
  ts: string;
  prompt_tokens: number | null;
  cached_tokens: number | null;
  ttft_ms: number | null;
  cache_source: CacheSource | string | null;
}

function median(xs: number[]): number | null {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

/** The requests that ended in the last `windowMs` (by their ledger time), per tier, in TIERS order;
 * a tier no request used is left out. Requests without a prompt (a refusal) are not counted. */
export function cacheSummary(rows: SummaryInput[], now: number, windowMs = 3_600_000): CacheSummary {
  const by = new Map<CacheTier, { n: number; reused: number; forwarded: number; ttft: number[] }>();
  for (const r of rows) {
    if (r.prompt_tokens == null) continue;
    const t = Date.parse(r.ts);
    if (!Number.isFinite(t) || now - t > windowMs || t > now + 60_000) continue;
    const tier = cacheTier(r.cache_source);
    const u = reuse(r.prompt_tokens, r.cached_tokens);
    const acc = by.get(tier) ?? { n: 0, reused: 0, forwarded: 0, ttft: [] };
    acc.n += 1;
    acc.reused += u.reused ?? 0;
    acc.forwarded += u.forwarded ?? 0;
    if (r.ttft_ms != null) acc.ttft.push(r.ttft_ms);
    by.set(tier, acc);
  }
  const out: TierRow[] = [];
  for (const tier of TIERS) {
    const a = by.get(tier);
    if (a) out.push({ tier, label: TIER_LABEL[tier], requests: a.n, reused: a.reused, forwarded: a.forwarded, ttftP50: median(a.ttft) });
  }
  const reused = out.reduce((s, x) => s + x.reused, 0);
  const forwarded = out.reduce((s, x) => s + x.forwarded, 0);
  return {
    rows: out,
    requests: out.reduce((s, x) => s + x.requests, 0),
    reused,
    forwarded,
    reusedShare: reused + forwarded > 0 ? reused / (reused + forwarded) : null,
  };
}
