// The live panel's arithmetic: a five-minute ring of one-second samples seeded by the
// first event's `history`, gaps by timestamp (the sampler sleeps when nothing is in flight and
// nobody watches), the row ordering, and the words for a phase. Pure functions, no DOM.

import type { Live, LivePhase, LiveRequest, LiveSample, LiveWindows } from '../api/types';
import { exact } from './format';

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
 * not a zero and not a line across). Four series come out: decode and prefill tok/s as contract
 * 1.0 has them (all tokens; one prefill dot when a request finishes), and the live ones of VIS-40
 * (decode from each request's warm-up anchor on; the running prefills' now-rate every second).
 */
export function slots(ring: Ring, now = Date.now() / 1000, seconds = WINDOW_S): { t: number[]; decode: (number | null)[]; prefill: (number | null)[]; decodeLive: (number | null)[]; prefillLive: (number | null)[] } {
  const end = Math.floor(now);
  const start = end - seconds + 1;
  const t: number[] = [];
  const decode: (number | null)[] = new Array(seconds).fill(null);
  const prefill: (number | null)[] = new Array(seconds).fill(null);
  const decodeLive: (number | null)[] = new Array(seconds).fill(null);
  const prefillLive: (number | null)[] = new Array(seconds).fill(null);
  for (let i = 0; i < seconds; i++) t.push(start + i);
  for (const s of ring.samples) {
    const i = Math.round(s.t) - start;
    if (i < 0 || i >= seconds) continue;
    decode[i] = s.decode_tps;
    if (s.prefill_tps != null) prefill[i] = s.prefill_tps;
    decodeLive[i] = s.decode_live_tps ?? null;
    prefillLive[i] = s.prefill_live_tps ?? null;
  }
  return { t, decode, prefill, decodeLive, prefillLive };
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

/** The voice's line for the Live card (narrator v3): the line of the request the Now line is
 *  about, else the newest line of a request in flight, else of one that just ended. `undefined`
 *  when the server sends no lines at all (an older server: no slot is reserved). */
export function voiceLine(rows: LiveRequest[], requestId?: string | null): { text: string; kind: string; id: string } | null | undefined {
  if (!rows.some((r) => r.line !== undefined)) return undefined;
  const pick = (r?: LiveRequest) => (r && r.line ? { text: r.line.text, kind: r.line.kind, id: r.request_id } : null);
  const own = pick(rows.find((r) => r.request_id === requestId));
  if (own) return own;
  for (const r of orderRows(rows)) {
    const p = pick(r);
    if (p) return p;
  }
  return null;
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

// ---- VIS-40: the live rates without the warm-up ---------------------------------------------------

/** The server sends the live rates (it reports their `windows`); an older one gets 1.0's figures. */
export function hasLiveRates(l: Pick<Live, 'windows'> | null | undefined): l is { windows: LiveWindows } {
  return !!l?.windows;
}

/** A rate as the tile and the banner both print it: "743", "1,253", "43.2"; "—" for none. */
export function rateText(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return '—';
  return v >= 100 ? exact(Math.round(v)) : v.toFixed(1);
}

/** "prefilling 14,336 of 107,604 · 13 %", or "prefilling 4,000 tokens · 12.0 s" without progress. */
export function prefillingWords(rows: LiveRequest[]): string {
  const p = rows.find((r) => r.phase === 'prefill');
  if (!p) return 'prefilling';
  const pf = p.activity?.prefill;
  if (pf && pf.done != null && pf.pct != null && pf.progress !== 'single_call') return `prefilling ${exact(pf.done)} of ${exact(pf.total)} · ${Math.round(pf.pct)} %`;
  return p.prompt_tokens != null ? `prefilling ${exact(p.prompt_tokens)} tokens · ${elapsed(p.elapsed_ms)}` : 'prefilling';
}

export interface LiveFigure {
  value: number | null; // the headline; null reads "—"
  sub: string;
  caption: string; // the window, under the label: "from token 50", "after the first chunk"
  captionTitle: string;
}

function warmChunks(n: number): string {
  return n === 0 ? 'from the start' : n === 1 ? 'after the first chunk' : `after the first ${n} chunks`;
}

/** "mostly cached (2,571 new of 195,083)" for a row the server marks so; null otherwise. */
export function mostlyCachedWords(r: LiveRequest): string | null {
  if (!r.mostly_cached || r.prompt_tokens == null) return null;
  const cached = r.phase === 'prefill' ? (r.activity?.prefill?.cached ?? null) : r.cached_tokens;
  if (cached == null) return 'mostly cached';
  return `mostly cached (${exact(r.prompt_tokens - cached)} new of ${exact(r.prompt_tokens)})`;
}

/**
 * The prefill tile of a server with the live rates: the running prefills' now-rate (summed over
 * several), "—" while none has one. Never the last finished request's rate: that stays in its row.
 */
export function prefillFigure(l: Live & { windows: LiveWindows }, rows: LiveRequest[]): LiveFigure {
  const w = l.windows;
  const value = l.now.prefill_live_tps ?? null;
  const pre = rows.filter((r) => r.phase === 'prefill');
  let sub: string;
  if (!pre.length) {
    const replay = rows.find((r) => r.phase !== 'done' && r.cache_source === 'response');
    sub = replay && replay.prompt_tokens != null ? `fully cached, no prefill (${exact(replay.prompt_tokens)} tokens)` : 'nothing prefilling';
  } else if (pre.length > 1) {
    const rates = pre.map((r) => r.activity?.prefill?.tps_live).filter((v): v is number => v != null);
    sub = `${pre.length} prefilling · ${rates.length ? rates.map(rateText).join(' + ') : 'warming up'}`;
  } else {
    const p = pre[0];
    const words = prefillingWords(pre);
    const warming = value == null && p.activity?.prefill?.progress !== 'single_call';
    sub = mostlyCachedWords(p) ?? (warming ? `warming up · ${words}` : words);
  }
  const caption = warmChunks(w.prefill_warmup_chunks);
  return {
    value,
    sub,
    caption,
    captionTitle: `The prefill running now: new tokens over its last ${w.now_window_s} s of chunks, ${caption === 'from the start' ? 'from the start of the loop' : `${caption} left out (the cache restore, allocation and warm-up)`}. A finished request keeps its end-to-end rate (timings) in its row.`,
  };
}

/** The decode tile of a server with the live rates: every request's rate from its warm-up anchor on. */
export function decodeFigure(l: Live & { windows: LiveWindows }, rows: LiveRequest[]): LiveFigure {
  const w = l.windows;
  const value = l.now.decode_live_tps ?? null;
  const c = l.counts;
  let sub: string;
  if (!c.decoding) sub = 'nothing decoding';
  else if (value == null && (l.now.decode_warming ?? 0) > 0) {
    const warm = rows.filter((r) => r.phase === 'decode' && r.decode_warmup && !r.decode_warmup.reached);
    sub = warm.length === 1 ? warmingWords(warm[0], w) : `${warm.length || l.now.decode_warming} warming up`;
  } else sub = `${c.decoding} decoding · last ${w.now_window_s} s`;
  return {
    value,
    sub,
    caption: `from token ${w.decode_warmup_tokens}`,
    captionTitle: `The speed now: tokens over the last ${w.now_window_s} s, each request counted from its ${w.decode_warmup_tokens}th decoded token on (its warm-up left out). A finished request keeps its end-to-end rate (timings) in its row.`,
  };
}

/** "warming up (23 of 50 tokens)" for a decoding row before its anchor. */
export function warmingWords(r: LiveRequest, w: Pick<LiveWindows, 'decode_warmup_tokens'>): string {
  const t = Math.min(r.decode_warmup?.tokens ?? 0, w.decode_warmup_tokens);
  return `warming up (${exact(t)} of ${exact(w.decode_warmup_tokens)} tokens)`;
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

/** VIS-58: "15,957 cached · +4,625 hidden": the cached share of the client's prompt (never above it), and TandemLLM's own notes as their own figure. */
export function cachedShare(r: Pick<LiveRequest, 'cached_tokens' | 'hidden_tokens'>): string | null {
  const parts: string[] = [];
  if (r.cached_tokens) parts.push(`${exact(r.cached_tokens)} cached`);
  if (r.hidden_tokens) parts.push(`+${exact(r.hidden_tokens)} hidden`);
  return parts.length ? parts.join(' · ') : null;
}
