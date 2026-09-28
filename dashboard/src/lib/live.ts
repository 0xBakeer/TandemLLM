// The live panel's arithmetic: a five-minute ring of one-second samples seeded by the
// first event's `history`, gaps by timestamp (the sampler sleeps when nothing is in flight and
// nobody watches), the row ordering, and the words for a phase. Pure functions, no DOM.

import type { LivePhase, LiveRequest, LiveSample } from '../api/types';

export const WINDOW_S = 300;

export interface Ring {
  samples: LiveSample[]; // by time, newest last, at most WINDOW_S seconds deep
}

export function emptyRing(): Ring {
  return { samples: [] };
}

/** Seed from `history` (the first event); older than the window or out of order is dropped. */
export function seedRing(history: LiveSample[] | undefined, now = Date.now() / 1000): Ring {
  const ring = emptyRing();
  for (const s of history ?? []) push(ring, s, now);
  return ring;
}

/** Append one sample; a repeat of the newest timestamp is ignored (the SSE resends the last sample on reconnect). */
export function push(ring: Ring, s: LiveSample | null, now = Date.now() / 1000): Ring {
  if (!s) return ring;
  const last = ring.samples[ring.samples.length - 1];
  if (last && s.t <= last.t) return ring;
  ring.samples.push(s);
  const cut = now - WINDOW_S;
  while (ring.samples.length && ring.samples[0].t <= cut) ring.samples.shift();
  return ring;
}

/**
 * The window as fixed one-second slots ending now: `null` where no sample exists (a gap is a gap,
 * not a zero and not a line across). Two series come out: decode tok/s and prefill tok/s.
 */
export function slots(ring: Ring, now = Date.now() / 1000, seconds = WINDOW_S): { t: number[]; decode: (number | null)[]; prefill: (number | null)[] } {
  const end = Math.floor(now);
  const start = end - seconds + 1;
  const t: number[] = [];
  const decode: (number | null)[] = new Array(seconds).fill(null);
  const prefill: (number | null)[] = new Array(seconds).fill(null);
  for (let i = 0; i < seconds; i++) t.push(start + i);
  for (const s of ring.samples) {
    const i = Math.round(s.t) - start;
    if (i < 0 || i >= seconds) continue;
    decode[i] = s.decode_tps;
    if (s.prefill_tps != null) prefill[i] = s.prefill_tps;
  }
  return { t, decode, prefill };
}

/** Peak of the last `seconds` of a series, for a "peak 5 min" caption. */
export function peak(values: (number | null)[]): number | null {
  let m: number | null = null;
  for (const v of values) if (v != null && (m == null || v > m)) m = v;
  return m;
}

const ORDER: Record<LivePhase, number> = { decode: 0, prefill: 1, queued: 2, done: 3 };

/** Decoding first, then prefilling, queued, then the finished ones newest first (the server's order, kept stable here). */
export function orderRows(rows: LiveRequest[]): LiveRequest[] {
  return [...rows].sort((a, b) => ORDER[a.phase] - ORDER[b.phase] || (a.phase === 'done' ? (a.ended_ms_ago ?? 0) - (b.ended_ms_ago ?? 0) : 0));
}

export interface PhaseWords {
  glyph: string;
  word: string;
  cls: 'decode' | 'prefill' | 'queued' | 'done' | 'failed';
}

/** The glyph and the word for a row's phase; a finish that went wrong reads as failed. */
export function phaseWords(r: Pick<LiveRequest, 'phase' | 'finish_reason' | 'status'>): PhaseWords {
  switch (r.phase) {
    case 'decode':
      return { glyph: '●', word: 'decoding', cls: 'decode' };
    case 'prefill':
      return { glyph: '◐', word: 'prefilling', cls: 'prefill' };
    case 'queued':
      return { glyph: '○', word: 'queued', cls: 'queued' };
    default: {
      const bad = r.finish_reason === 'error' || r.finish_reason === 'refused' || r.finish_reason === 'abandoned' || r.status >= 400;
      const word = r.finish_reason ?? (r.status >= 400 ? `HTTP ${r.status}` : 'done');
      return bad ? { glyph: '✕', word, cls: 'failed' } : { glyph: '✓', word, cls: 'done' };
    }
  }
}

/** Sum of the rows' 2 s rates where the server gives none (older servers, or before two samples). */
export function sumDecodeNow(rows: LiveRequest[]): number | null {
  let sum = 0;
  let n = 0;
  for (const r of rows) {
    if (r.phase === 'decode' && r.decode_tps_now != null) {
      sum += r.decode_tps_now;
      n++;
    }
  }
  return n ? Math.round(sum * 10) / 10 : null;
}

/** "12 s ago" for a finished row, from `ended_ms_ago`. */
export function agoShort(ms: number | null): string {
  if (ms == null) return '';
  const s = Math.max(0, Math.round(ms / 1000));
  return s < 60 ? `${s} s ago` : `${Math.round(s / 60)} m ago`;
}

/** Elapsed in a compact form: "0.8 s", "12.4 s", "1 m 12 s". */
export function elapsed(ms: number | null): string {
  if (ms == null || !Number.isFinite(ms)) return '—';
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)} s`;
  const m = Math.floor(ms / 60_000);
  return `${m} m ${Math.round((ms - m * 60_000) / 1000)} s`;
}
