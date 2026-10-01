// The System view's memory plan (OPS-41): the engine's split as segments of one bar, and the
// residency planner's words. Pure functions; GiB throughout, as the planner and the box's
// operators count (the floor is 15 GiB of MemAvailable).

import type { MemorySplit, ResidencyReason, ResidencyRow } from '../api/types';

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

/** A plan row in words: "fits, 22.8 GiB left" / "refused: it would break the floor (6.0 GiB left, 21.0 needed)". */
export function rowWords(r: ResidencyRow): string {
  if (r.allowed) return `fits, ${gibOf(r.after_gib)} left`;
  const why = r.reason ? REASON_WORDS[r.reason] : 'refused';
  return r.reason === 'refused_memory' && r.after_gib != null ? `${why} (${gibOf(r.after_gib)} left, ${gibOf(r.need_gib)} needed)` : why;
}
