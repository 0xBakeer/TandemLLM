// The System view's memory plan (OPS-41): the engine's split as segments of one bar, and the
// residency planner's words. Pure functions; GiB throughout, as the planner and the box's
// operators count (the floor is 15 GiB of MemAvailable).

import type { MemoryBudget, MemoryEvent, MemoryLimitRow, MemorySplit, ResidencyDoc, ResidencyReason, ResidencyRow, SessionSwitch } from '../api/types';

const GIB = 1024 ** 3;

/** "16.35 GiB"; "—" for nothing. */
export function gib(bytes: number | null | undefined, d = 2): string {
  if (bytes == null || !Number.isFinite(bytes)) return '—';
  return `${(bytes / GIB).toFixed(d)} GiB`;
}

/** A GiB figure the planner already gives in GiB: "22.8 GiB". */
export function gibOf(v: number | null | undefined, d = 1): string {
  return v == null || !Number.isFinite(v) ? '—' : `${v.toFixed(d)} GiB`;
}

export interface SplitSegment {
  key: string;
  label: string;
  bytes: number;
  share: number; // of the engine's reserved total
  color: string; // a series token
}

/** The parts in the order they load, then the allocator's own share. */
const PARTS: [string, string, string][] = [
  ['weights', 'weights', 'cobalt'],
  ['kv_pool', 'KV pool', 'orange'],
  ['gdn_state', 'GDN state', 'aqua'],
  ['drafter_kv', 'drafter caches', 'violet'],
  ['vision', 'vision tower', 'yellow'],
  ['resident_prefix', 'resident prefix', 'red'],
  ['state_store', 'state store', 'aqua'],
  ['deciders', 'decision models', 'yellow'],
];

/**
 * One bar for the engine: each named part, then `other` (the allocator's scratch, graph pools and
 * small buffers beyond the named parts) and `reserve` (taken from the driver, no tensor in it),
 * as shares of the reserved total. Parts of zero bytes are left out.
 */
export function splitSegments(s: MemorySplit | null | undefined): SplitSegment[] {
  if (!s) return [];
  const segs: SplitSegment[] = [];
  for (const [key, label, color] of PARTS) {
    const b = s.parts[key];
    if (b) segs.push({ key, label, bytes: b, share: 0, color });
  }
  for (const [key, b] of Object.entries(s.parts)) if (b && !PARTS.some(([k]) => k === key)) segs.push({ key, label: key.replace(/_/g, ' '), bytes: b, share: 0, color: 'muted' });
  if (s.other != null && s.other > 0) segs.push({ key: 'other', label: 'other allocations', bytes: s.other, share: 0, color: 'muted' });
  if (s.reserve != null && s.reserve > 0) segs.push({ key: 'reserve', label: 'allocator reserve', bytes: s.reserve, share: 0, color: 'rule' });
  const total = segs.reduce((a, x) => a + x.bytes, 0) || 1;
  for (const x of segs) x.share = x.bytes / total;
  return segs;
}

export const REASON_WORDS: Record<ResidencyReason, string> = {
  refused_hold: 'refused: a hold owns the board',
  refused_load_in_progress: 'refused: another load is running',
  refused_memory: 'refused: it would break the floor',
  refused_unknown_role: 'refused: no claim for this role',
};

/** A plan row in words: "fits, 22.8 GiB left" / "refused: it would break the floor (6.0 GiB left, 21.0 needed)".
 *  The left figure already counts the engine's growth to its peak (OPS-47). */
export function rowWords(r: ResidencyRow): string {
  if (r.allowed) return `fits, ${gibOf(r.after_gib)} left`;
  const why = r.reason ? REASON_WORDS[r.reason] : 'refused';
  return r.reason === 'refused_memory' && r.after_gib != null ? `${why} (${gibOf(r.after_gib)} left, ${gibOf(r.need_gib)} needed)` : why;
}

// ---- OPS-47: the engine's memory limits ---------------------------------------------------------

/** The split as one bar on an absolute scale, so the ceiling and the peak can be drawn over it:
 *  each segment's share of `scale` (the larger of the peak and what the engine reserves), and
 *  where the two lines fall on it. Without a budget: the old bar, shares of the reserved total. */
export function scaledBar(s: MemorySplit | null | undefined, b: MemoryBudget | null | undefined): { segs: SplitSegment[]; rest: number; ceiling: number | null; peak: number | null } {
  const segs = splitSegments(s);
  const total = segs.reduce((a, x) => a + x.bytes, 0);
  if (!b || !b.peak || !total) return { segs, rest: 0, ceiling: null, peak: null };
  const scale = Math.max(b.peak, total) * 1.03;
  for (const x of segs) x.share = x.bytes / scale;
  return { segs, rest: Math.max(0, 1 - total / scale), ceiling: b.ceiling / scale, peak: b.peak / scale };
}

/** "70 GiB", "8,192 tokens", "on"; "—" for none. */
export function limitValue(r: Pick<MemoryLimitRow, 'unit'>, v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return '—';
  if (r.unit === 'on/off') return v ? 'on' : 'off';
  if (r.unit === 'tokens') return `${Math.round(v).toLocaleString('en-US')} tokens`;
  return `${Number(v.toFixed(2))} ${r.unit}`;
}

/** What a limit holds now against its cap, as a share (0..1, null where it has no figure). */
export function usageShare(r: Pick<MemoryLimitRow, 'used' | 'cap'>): number | null {
  if (r.used == null || r.cap == null || r.cap <= 0) return null;
  return Math.max(0, Math.min(1, r.used / r.cap));
}

/** "3.85 of 4.00 GiB"; "—" where nothing is read. */
export function usageWords(r: Pick<MemoryLimitRow, 'unit' | 'used' | 'cap'>): string {
  if (r.used == null) return '—';
  if (r.unit === 'GiB' || r.unit === 'MiB') {
    const u = r.unit === 'GiB' ? 1024 ** 3 : 1024 ** 2;
    const f = (x: number) => (x / u).toFixed(2);
    return r.cap != null ? `${f(r.used)} of ${f(r.cap)} ${r.unit}` : `${f(r.used)} ${r.unit}`;
  }
  return r.cap != null ? `${r.used} of ${r.cap}` : String(r.used);
}

/** When a change to this limit takes effect. */
export function appliesWords(r: Pick<MemoryLimitRow, 'applies' | 'locked'>): string {
  if (r.applies === 'locked') return r.locked ?? 'locked';
  return r.applies === 'restart' ? 'needs a restart' : 'applies at the next request';
}

/** A restart key set to something the engine is not running yet. */
export function pendingWords(r: MemoryLimitRow): string | null {
  if (!r.pending_restart) return null;
  return `set to ${limitValue(r, r.value)}; the engine runs ${limitValue(r, r.running)} until it restarts`;
}

/** "81,199 tokens re-read in 125 s (opencode-a → opencode-b, from zero)". */
export function switchWords(sw: SessionSwitch | null | undefined): string | null {
  if (!sw) return null;
  const via: Record<string, string> = { park: 'the parked rows swapped back', pin: 'from the pinned head', resident: 'from the shared head', none: 'from zero', prefix: 'from a store entry', session: 'from a store entry' };
  const who = sw.from || sw.to ? ` (${sw.from ?? 'no id'} → ${sw.to ?? 'no id'}, ${via[sw.via] ?? sw.via})` : ` (${via[sw.via] ?? sw.via})`;
  return `${sw.reread.toLocaleString('en-US')} tokens re-read in ${(sw.ms / 1000).toFixed(1)} s, ${sw.reused.toLocaleString('en-US')} reused${who}`;
}

/** One line of the shed log. */
export function eventWords(e: MemoryEvent): string {
  const g = (x?: number) => (x == null ? '—' : gib(x));
  switch (e.kind) {
    case 'shed':
      return `gave back the ${e.level === 'store' ? 'state store' : e.level === 'anchors' ? 'body anchors' : e.level === 'stash' ? 'guest stash' : e.level === 'park' ? 'parked session' : 'pinned head'}: ${g(e.freed)} (${e.why === 'ceiling' ? 'over the ceiling' : e.why === 'peak' ? 'for a prefill under the peak' : `for ${String(e.why ?? '').replace('admit:', '')}`})`;
    case 'refused':
      return `refused a ${(e.n ?? 0).toLocaleString('en-US')}-token prefill with a 503: it needed ${g(e.need)}, the peak is ${g(e.peak)}`;
    case 'empty':
      return `reserve back to the driver: ${g(e.freed)} in ${e.ms ?? '—'} ms`;
    case 'estimate_raised':
      return `workspace estimate raised: ${(e.n ?? 0).toLocaleString('en-US')} tokens took ${g(e.workspace)}`;
    default:
      return `a ${e.level ?? 'cache'} allocation of ${g(e.nbytes)} was refused`;
  }
}

/** The residency document with one engine limit set, or removed (back to serve.env). */
export function withEngineLimit(doc: ResidencyDoc, key: string, value: number | null): ResidencyDoc {
  const engine = { ...(doc.engine ?? {}) };
  if (value == null || !Number.isFinite(value)) delete engine[key];
  else engine[key] = value;
  const out: ResidencyDoc = { ...doc, engine };
  if (!Object.keys(engine).length) delete out.engine;
  return out;
}
