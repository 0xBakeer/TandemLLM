// The Live panel's words for contract 1.1: the "Now" line for the engine, the activity
// cell and the timeline of a request, the tone of a stop, and the health of the stream itself.
// Pure functions over the message, no DOM. The server's `label` and `sentence` are shown as sent
// (server/activity.py); this file adds numbers and formatting only. Every state has a glyph and
// a word, so colour is never the only signal.

import type { ActivityState, LiveActivity, LiveEngine, LiveRequest, LiveStop, LiveTimelineEntry, StopReason } from '../api/types';
import { bytes, exact, fixed, tps } from './format';
import { phaseWords } from './live';
import type { StreamState } from './sse';

/** The colour family of a state; the CSS maps a tone to the palette (cobalt for prefill, orange for now). */
export type Tone = 'idle' | 'queued' | 'prefill' | 'replay' | 'think' | 'write' | 'tool' | 'finish' | 'wait' | 'ok' | 'bad' | 'warn';

export const GLYPH: Record<ActivityState, string> = {
  queued: '○',
  prefilling: '◐',
  replaying: '↻',
  thinking: '◍',
  closing_reasoning: '◍',
  writing: '●',
  tool_call: '⚒',
  finishing: '…',
  done: '✓',
};

export const TONE: Record<ActivityState, Tone> = {
  queued: 'queued',
  prefilling: 'prefill',
  replaying: 'replay',
  thinking: 'think',
  closing_reasoning: 'think',
  writing: 'write',
  tool_call: 'tool',
  finishing: 'finish',
  done: 'ok',
};

/** The state as a word: `closing_reasoning` reads "closing reasoning". */
export function stateWord(s: ActivityState | string): string {
  return s === 'tool_call' ? 'tool call' : s.replace(/_/g, ' ');
}

/** States where tokens move (the glyph pulses; nothing pulses under reduced motion). */
export function isMoving(s: ActivityState | null | undefined): boolean {
  return s === 'prefilling' || s === 'replaying' || s === 'thinking' || s === 'closing_reasoning' || s === 'writing' || s === 'tool_call';
}

const RANK: Record<ActivityState, number> = { tool_call: 0, writing: 1, thinking: 2, closing_reasoning: 3, finishing: 4, replaying: 5, prefilling: 6, queued: 7, done: 8 };

/** The request the Now line speaks for: the one furthest along, never a finished one. */
export function pickRunning(rows: LiveRequest[]): LiveRequest | null {
  let best: LiveRequest | null = null;
  for (const r of rows) {
    if (r.phase === 'done' || !r.activity || r.activity.state === 'done') continue;
    if (!best || RANK[r.activity.state] < RANK[best.activity!.state]) best = r;
  }
  return best;
}

export function ordinal(n: number): string {
  const m = n % 100;
  const suf = m >= 10 && m <= 20 ? 'th' : ({ 1: 'st', 2: 'nd', 3: 'rd' } as Record<number, string>)[n % 10] ?? 'th';
  return `${n}${suf}`;
}

/** "about 5 s left" / "about 1 m 10 s left"; null under half a second. */
export function etaWord(ms: number | null | undefined): string | null {
  if (ms == null || !Number.isFinite(ms) || ms < 500) return null;
  const s = Math.round(ms / 1000);
  if (s < 60) return `about ${s} s left`;
  const m = Math.floor(s / 60);
  return `about ${m} m ${s - m * 60} s left`;
}

/** Seconds in a short form for "in state": "0.8 s", "34 s", "2 m 05 s". */
export function secs(ms: number | null | undefined): string {
  if (ms == null || !Number.isFinite(ms)) return '';
  const s = ms / 1000;
  if (s < 10) return `${s.toFixed(1)} s`;
  if (s < 60) return `${Math.round(s)} s`;
  const m = Math.floor(s / 60);
  return `${m} m ${String(Math.round(s - m * 60)).padStart(2, '0')} s`;
}

/** "611 thinking · 380 content" for the tokens cell; null when the split is empty. */
export function tokensSplit(d: LiveActivity['decode'] | null | undefined): string | null {
  if (!d || (d.thinking_tokens === 0 && d.content_tokens === 0 && d.tool_tokens === 0)) return null;
  const parts = [`${exact(d.thinking_tokens)} thinking`, `${exact(d.content_tokens)} content`];
  if (d.tool_tokens > 0) parts.push(`${exact(d.tool_tokens)} tool`);
  return parts.join(' · ');
}

// ---- the Now line -------------------------------------------------------------------------------

export interface NowLine {
  glyph: string;
  tone: Tone;
  headline: string; // the server's label, e.g. "Prefilling 36,864 of 48,210 (76 %)"
  numbers: string[]; // the live numbers beside it, e.g. ["2,310 tok/s", "12,288 cached", "about 5 s left"]
  sinceMs: number | null; // time in this state
  client: string | null; // the client kind, e.g. "opencode"
  requestId: string | null;
  progress: number | null; // 0..1 while a chunked prefill runs
  moving: boolean; // tokens move: the glyph may pulse
  warning: string | null; // "client disconnected 12 s ago"
}

/** One glyph and one sentence for the engine, from `engine.state` and the running request's activity. */
export function nowLine(engine: LiveEngine | null | undefined, rows: LiveRequest[]): NowLine | null {
  if (!engine) return null; // a 1.0 server: the panel, no Now line
  const line = (glyph: string, tone: Tone, headline: string, numbers: string[] = [], more: Partial<NowLine> = {}): NowLine => ({ glyph, tone, headline, numbers, sinceMs: engine.since_ms, client: null, requestId: null, progress: null, moving: false, warning: null, ...more });
  const w = engine.waiting_for_client;
  if (engine.state === 'idle') return line('○', 'idle', 'Idle', ['nothing in flight']);
  if (engine.state === 'starting') return line('◌', 'warn', 'Starting', ['the model is loading']);
  if (engine.state === 'draining') return line('◑', 'warn', 'Draining', ['finishing what runs, taking nothing new']);
  if (engine.state === 'waiting_for_client') return line('◌', 'wait', engine.label || 'Waiting for client', [], { sinceMs: w?.since_ms ?? engine.since_ms, client: w?.client_kind ?? null, requestId: w?.request_id ?? null });
  const r = pickRunning(rows);
  if (!r?.activity) {
    // busy, but the activity is off (`--live-activity off`): the engine's own label, nothing more
    const p = rows.find((x) => x.phase !== 'done');
    return line(p ? phaseWords(p).glyph : '●', p?.phase === 'prefill' ? 'prefill' : p?.phase === 'queued' ? 'queued' : 'write', engine.label || 'Busy', [], { client: p?.client.kind ?? null, requestId: p?.request_id ?? null, moving: p?.phase === 'decode' });
  }
  const a = r.activity;
  const d = a.decode;
  const p = a.prefill;
  const t = a.tool;
  const q = a.queue;
  const rate = d?.tps_now ?? d?.tps_avg ?? r.decode_tps_now ?? r.decode_tps;
  const rateWord = rate == null ? null : tps(rate);
  const st = a.state;
  const n: (string | null | false)[] = [];
  let progress: number | null = null;
  if (st === 'queued') n.push(q?.wait_ms != null && `waiting ${secs(q.wait_ms)}`, q?.timeout_s != null && `gives up after ${q.timeout_s} s`);
  else if (st === 'prefilling') {
    const ptps = p?.tps_now ?? p?.tps_avg;
    n.push(ptps != null && tps(ptps), !!p?.cached && `${exact(p!.cached)} cached`, etaWord(p?.eta_ms) ?? null);
    if (p && p.pct != null && p.progress !== 'single_call') progress = Math.max(0, Math.min(1, p.pct / 100));
  } else if (st === 'replaying') n.push(`${exact(r.tokens)} tokens replayed, no model run`);
  else if (st === 'thinking') n.push(rateWord, d && `${exact(d.thinking_tokens)} thinking tokens`);
  else if (st === 'closing_reasoning') n.push(!!a.reasoning?.closed_by && `closed by the ${a.reasoning!.closed_by}`, d && `${exact(d.thinking_tokens)} thinking tokens`);
  else if (st === 'writing') n.push(rateWord, d && `${exact(d.content_tokens)} content tokens`);
  else if (st === 'tool_call') n.push(t?.arg_bytes != null && `${bytes(t.arg_bytes, 1)} of arguments`, !!t?.calls_done && `${t!.calls_done} call${t!.calls_done === 1 ? '' : 's'} done`, rateWord);
  else if (st === 'finishing') n.push(d && `${exact(d.tokens)} tokens`);
  n.push(!!a.constrained && constrainedWord(a.constrained));
  return line(GLYPH[st], TONE[st], a.label, n.filter((x): x is string => !!x), { sinceMs: a.since_ms, client: r.client.kind, requestId: r.request_id, progress, moving: isMoving(st), warning: clientFlag(a) });
}

export function constrainedWord(c: LiveActivity['constrained']): string {
  return c === 'response_format' ? 'JSON schema' : c === 'tool_choice' ? 'tool_choice' : '';
}

/** "client disconnected 12 s ago" while the engine still works for a client that left; null otherwise. */
export function clientFlag(a: LiveActivity | null | undefined): string | null {
  if (!a || a.client.connected !== false) return null;
  const ago = a.client.gone_ms ?? a.client.silent_ms;
  return ago != null && ago >= 500 ? `client disconnected ${secs(ago)} ago` : 'client disconnected';
}

/** "continues chatcmpl-77aa after 8.4 s (client ran bash)", with "(inferred)" when the link is a guess. */
export function continuesText(c: LiveActivity['continues']): string | null {
  if (!c) return null;
  let s = `continues ${c.request_id.slice(0, 18)}`;
  if (c.gap_ms != null) s += ` after ${secs(c.gap_ms)}`;
  if (c.tool_names.length) s += ` (client ran ${c.tool_names.join(', ')})`;
  if (c.inferred) s += ' (inferred)';
  return s;
}

// ---- the activity cell ----------------------------------------------------------------------------

export interface Cell {
  glyph: string;
  word: string;
  detail: string | null;
  tone: Tone;
  moving: boolean;
}

/** Glyph, word and detail for a row: `◐ prefilling 76 %`, `⚒ write_file 18 KB`, `✗ abandoned`. */
export function activityCell(a: LiveActivity | null | undefined, r?: Pick<LiveRequest, 'phase' | 'finish_reason' | 'status' | 'tokens'>): Cell {
  if (!a) {
    // a 1.0 server or the kill switch: the phase words
    const p = phaseWords(r ?? { phase: 'queued', finish_reason: null, status: 200 });
    const tone: Tone = p.cls === 'decode' ? 'write' : p.cls === 'prefill' ? 'prefill' : p.cls === 'queued' ? 'queued' : p.cls === 'failed' ? 'bad' : 'ok';
    return { glyph: p.glyph, word: p.word, detail: null, tone, moving: p.cls === 'decode' };
  }
  const st = a.state;
  const d = a.decode;
  const p = a.prefill;
  const q = a.queue;
  let word = stateWord(st);
  let detail: string | null = null;
  if (st === 'queued') detail = [q?.place ? `${q.place_is_estimate ? 'about ' : ''}${ordinal(q.place)}` : '', q?.wait_ms != null ? secs(q.wait_ms) : ''].filter(Boolean).join(' · ') || null;
  else if (st === 'prefilling') detail = p && p.pct != null && p.progress !== 'single_call' ? `${Math.round(p.pct)} %` : p ? `${exact(p.total)} tokens` : null;
  else if (st === 'replaying') detail = r ? `${exact(r.tokens)} tok` : null;
  else if (st === 'thinking') detail = d ? `${exact(d.thinking_tokens)} tok` : null;
  else if (st === 'closing_reasoning') detail = a.reasoning?.closed_by ?? null;
  else if (st === 'writing') detail = d?.content_tokens ? `${exact(d.content_tokens)} tok` : null;
  else if (st === 'tool_call') {
    word = a.tool?.name ?? 'tool call';
    detail = a.tool?.arg_bytes != null ? bytes(a.tool.arg_bytes, 1) : null;
  } else if (st === 'finishing') detail = a.step ? { flush: 'flushing', saving_state: 'saving state', final_chunk: 'final chunk' }[a.step] : null;
  else {
    const t = stopTone(a.stop);
    return { glyph: t.glyph, word: t.word, detail: null, tone: t.tone, moving: false };
  }
  return { glyph: GLYPH[st], word, detail, tone: TONE[st], moving: isMoving(st) };
}

// ---- stops ----------------------------------------------------------------------------------------

const BAD: StopReason[] = ['abandoned', 'error', 'timeout', 'cancelled'];
const WARN: StopReason[] = ['refused', 'rejected'];

/** A red ✗ for abandoned, error, timeout and cancelled; an amber ! for refused and rejected; ✓ otherwise. */
export function stopTone(stop: Pick<LiveStop, 'reason'> | null | undefined): { glyph: string; word: string; tone: 'ok' | 'bad' | 'warn' } {
  if (!stop) return { glyph: '✓', word: 'done', tone: 'ok' };
  const word = stop.reason === 'tool_calls' ? 'tool call' : stop.reason;
  if (BAD.includes(stop.reason)) return { glyph: '✗', word, tone: 'bad' };
  if (WARN.includes(stop.reason)) return { glyph: '!', word, tone: 'warn' };
  return { glyph: '✓', word, tone: 'ok' };
}

// ---- the timeline ---------------------------------------------------------------------------------

export interface Segment {
  state: ActivityState;
  word: string;
  tone: Tone;
  startMs: number;
  endMs: number;
  share: number; // 0..1 of the whole strip
  showLabel: boolean; // wide enough for the word
  detail?: string;
}

/**
 * One segment per state, width by time: a segment ends where the next begins, the last one at
 * `elapsedMs` (the request is still running) or at the last entry. A `done` entry has no width
 * and is left out; the stop mark says how it ended. A segment shows its word when its share of a
 * `stripPx`-wide strip fits the word at 10 px (about 5.2 px a character plus padding).
 */
export function timelineSegments(timeline: LiveTimelineEntry[] | null | undefined, elapsedMs: number | null | undefined, stripPx = 160): Segment[] {
  const entries = (timeline ?? []).filter((e) => Number.isFinite(e.t_ms));
  if (!entries.length) return [];
  const first = entries[0].t_ms;
  const lastT = entries[entries.length - 1].t_ms;
  const end = Math.max(elapsedMs ?? lastT, lastT);
  const total = Math.max(0, end - first);
  const out: Segment[] = [];
  for (let i = 0; i < entries.length; i++) {
    const e = entries[i];
    if (e.state === 'done') continue;
    const next = entries[i + 1]?.t_ms ?? end;
    const dur = Math.max(0, next - e.t_ms);
    out.push({ state: e.state, word: stateWord(e.state), tone: TONE[e.state], startMs: e.t_ms - first, endMs: next - first, share: 0, showLabel: false, detail: e.detail });
    out[out.length - 1].share = total > 0 ? dur / total : 0;
  }
  if (total <= 0 && out.length) for (const s of out) s.share = 1 / out.length; // no time yet: equal shares
  for (const s of out) s.showLabel = s.share * stripPx >= s.word.length * 5.2 + 8;
  return out;
}

/** The last `n` transitions as words: "+19.9 s thinking" (the phone's list, and the screen-reader list). */
export function timelineList(timeline: LiveTimelineEntry[] | null | undefined, n = 6): { at: string; word: string; detail?: string }[] {
  const entries = (timeline ?? []).slice(-n);
  return entries.map((e) => ({ at: `+${secs(e.t_ms) || '0.0 s'}`, word: stateWord(e.state), detail: e.detail }));
}

/** A recent row's `path` as segments of equal width, no words (no times are kept for it). */
export function pathSegments(path: ActivityState[]): Segment[] {
  const states = path.filter((s) => s !== 'done');
  return timelineSegments([...states.map((state, i) => ({ t_ms: i, state })), { t_ms: states.length, state: 'done' }], null, 0);
}

// ---- the stream's own health --------------------------------------------------------------------

export interface Health {
  word: string; // "streaming", "reconnecting since 4 s", "no update for 7 s"
  cadence: string | null; // "4/s"
  tone: 'ok' | 'warn' | 'bad' | 'muted';
  stale: boolean; // the numbers on the page are old: dim them
  ageMs: number | null;
}

/**
 * The connection state, the cadence, and the age of the last event. A silent stream is never
 * mistaken for an idle engine: past `2 × interval + 1.5 s` without an event the word turns to
 * "no update for N s" and the page dims; past 6 s it is red.
 */
export function streamHealth(state: StreamState, lastEventAt: number | null, stateSince: number | null, now: number, intervalS = 1): Health {
  const ageMs = lastEventAt == null ? null : Math.max(0, now - lastEventAt);
  const since = stateSince == null ? null : Math.max(0, now - stateSince);
  const h: Health = { word: 'paused', cadence: null, tone: 'muted', stale: true, ageMs };
  if (state === 'open') {
    const late = ageMs != null && ageMs > 2 * intervalS * 1000 + 1500;
    Object.assign(h, { word: late ? `no update for ${secs(ageMs!)}` : 'streaming', cadence: intervalS >= 1 ? `${Math.round(intervalS)} s` : `${Math.round(1 / intervalS)}/s`, tone: !late ? 'ok' : ageMs! > 6000 ? 'bad' : 'warn', stale: late });
  } else if (state === 'connecting') Object.assign(h, { word: 'connecting', tone: 'warn', stale: lastEventAt != null });
  else if (state === 'reconnecting' || state === 'error') Object.assign(h, { word: since != null && since >= 500 ? `reconnecting since ${secs(since)}` : 'reconnecting', tone: since != null && since > 6000 ? 'bad' : 'warn' });
  else if (state === 'unauthorized') Object.assign(h, { word: 'signed out', tone: 'bad' });
  return h;
}

/** "4.4 tokens per round · 90.9 ms a round" under the decode figure, from the decoding row's activity. */
export function roundWords(rows: LiveRequest[]): string | null {
  const r = rows.find((x) => x.phase === 'decode' && x.activity?.decode);
  const d = r?.activity?.decode;
  if (!d) return null;
  const tpr = d.tokens_per_round_now ?? d.tokens_per_round;
  const mpr = d.ms_per_round_now ?? d.ms_per_round;
  if (tpr == null && mpr == null) return null;
  const parts: string[] = [];
  if (tpr != null) parts.push(`${fixed(tpr, 1)} tokens per round`);
  if (mpr != null) parts.push(`${fixed(mpr, 1)} ms a round`);
  return parts.join(' · ') + (d.tokens_per_round_now != null || d.ms_per_round_now != null ? ' (last second)' : '');
}
