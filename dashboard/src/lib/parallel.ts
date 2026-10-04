// The parallel view's data (`GET /v1/dashboard/parallel`, server/parallel.py): the types, the
// lanes' slots (a request keeps its row from arrival until its "done" fades, and a new request
// takes the first free row, so rows never jump), the backends' colours, the series merge and the
// stack the chart draws. No DOM here: the view (views/parallel.ts) renders, this file decides.

import type { SeriesColor } from '../charts/chart-base';

export type LaneState = 'queued' | 'prefilling' | 'waiting' | 'replaying' | 'thinking' | 'closing_reasoning' | 'tool_call' | 'writing' | 'finishing' | 'done';

export interface Lane {
  id: string;
  backend: string;
  kind: string;
  model: string | null;
  client: { kind: string; user: string | null; session: string | null };
  state: LaneState;
  label: string | null;
  tool: string | null;
  elapsed_ms: number | null;
  ttft_ms: number | null;
  waiting_ms: number | null;
  tokens: number | null;
  tokens_source: 'engine' | 'frames' | 'usage' | null;
  frames: number | null;
  tool_frames?: number | null;
  tool_chars?: number | null;
  thinking_tokens: number | null;
  decode_now: number | null;
  decode_avg: number | null;
  prefill: { done: number | null; total: number | null; pct: number | null; cached?: number | null; tps: number | null; tps_avg: number | null; eta_ms: number | null } | null;
  prompt_tokens: number | null;
  cached_tokens: number | null;
  cache_hit: number | null;
  cost_micros: number | null;
  cost_partial: boolean;
  max_tokens: number | null;
  finish?: string | null;
  status?: number;
  ended_ms?: number;
  relay_id?: string | null;
  via?: 'engine' | null;
}

export interface BackendRow {
  id: string;
  kind: string;
  models: string[];
  in_flight: number;
  decoding: number;
  waiting: number;
  decode_tps: number | null;
  decode_unreported: number;
  ttft_p50: number | null;
  ttft_p90: number | null;
  cache_hit: number | null;
  cache_reported: number;
  finished: number;
  errors: number;
  cost_per_min: number | null;
  cost_window: number | null;
  tokens_window: number;
  frames: { verified: boolean; tools_verified?: boolean; models: number; checked: number; passed: number } | null;
}

export interface Totals {
  in_flight: number;
  decoding: number;
  waiting: number;
  decode_tps: number | null;
  decode_reported: number;
  decode_unreported: number;
  prefill_tps: number | null;
  ttft_p50: number | null;
  ttft_p90: number | null;
  ttft_n: number;
  cache_hit: number | null;
  cache_reported: number;
  prompt_tokens_window: number | null;
  cached_tokens_window: number | null;
  cost_per_min: number | null;
  cost_window: number | null;
  cost_in_flight: number | null;
  finished: number;
  errors: number;
  refused: number;
  tokens_window: number;
  backends: number;
}

export interface Sample {
  t: number;
  by: Record<string, number>;
  in_flight: number;
}

export interface Parallel {
  contract_version: string;
  at: number;
  windows: { now_s: number; done_keep_s: number; stats_s: number; cost_min_s: number };
  lanes: Lane[];
  recent: Lane[];
  backends: BackendRow[];
  totals: Totals;
  series: Sample[];
  build_us?: { last: number; max: number };
}

// ---- colours ---------------------------------------------------------------------------------
/** The palette order a backend takes its colour from (the engine's own always cobalt). */
export const PALETTE: SeriesColor[] = ['cobalt', 'aqua', 'orange', 'violet', 'yellow', 'red', 'muted'];

/** Gives every backend a colour once and keeps it for the page's life (a backend that goes idle
 *  and comes back keeps its colour; the legend never reshuffles). */
export class Colors {
  private map = new Map<string, SeriesColor>();
  of(id: string): SeriesColor {
    let c = this.map.get(id);
    if (!c) {
      if (id === 'engine') c = 'cobalt';
      else {
        const used = new Set(this.map.values());
        c = PALETTE.find((p) => p !== 'cobalt' && !used.has(p)) ?? PALETTE[(this.map.size % (PALETTE.length - 1)) + 1];
      }
      this.map.set(id, c);
    }
    return c;
  }
  /** Assigns in a stable order: the engine first, then the ids sorted. */
  seed(ids: string[]): void {
    for (const id of [...ids].sort((a, b) => (a === 'engine' ? -1 : b === 'engine' ? 1 : a.localeCompare(b)))) this.of(id);
  }
}

// ---- slots ------------------------------------------------------------------------------------
export interface Slot {
  lane: Lane | null;
  /** when the slot became free (ms), for trimming the trailing empty rows */
  freeSince: number | null;
}

/** Rows that stay put. A lane keeps its slot from the first snapshot that shows it until its
 *  "done" row leaves `recent`; a new lane takes the lowest free slot; trailing free slots are
 *  dropped only after `trimMs` (so a wave of finishing requests does not shrink the list at
 *  once and grow it again a second later). */
export class Slots {
  slots: Slot[] = [];
  private where = new Map<string, number>();
  /** `holdMs`: how long a finished lane keeps its row (the server keeps it longer; short agent
   *  turns would otherwise fill the list with "done" rows) */
  constructor(
    private trimMs = 12_000,
    private holdMs = 3_000,
  ) {}

  update(lanes: Lane[], recent: Lane[], now: number): Slot[] {
    const all = new Map<string, Lane>();
    for (const l of recent) if ((l.ended_ms ?? 0) <= this.holdMs) all.set(l.id, l);
    for (const l of lanes) all.set(l.id, l); // in flight wins over a stale "done"
    // free the slots whose lane is gone
    for (const [id, i] of [...this.where]) {
      if (!all.has(id)) {
        this.where.delete(id);
        this.slots[i] = { lane: null, freeSince: now };
      }
    }
    // update the known ones in place
    for (const [id, i] of this.where) this.slots[i] = { lane: all.get(id) as Lane, freeSince: null };
    // place the new ones: oldest first, into the lowest free slot
    const fresh = [...all.values()].filter((l) => !this.where.has(l.id) && l.state !== 'done');
    fresh.sort((a, b) => (b.elapsed_ms ?? 0) - (a.elapsed_ms ?? 0));
    for (const l of fresh) {
      let i = this.slots.findIndex((s) => s.lane == null);
      if (i < 0) {
        i = this.slots.length;
        this.slots.push({ lane: null, freeSince: null });
      }
      this.slots[i] = { lane: l, freeSince: null };
      this.where.set(l.id, i);
    }
    // trim the free tail once it has been free long enough
    while (this.slots.length) {
      const last = this.slots[this.slots.length - 1];
      if (last.lane == null && last.freeSince != null && now - last.freeSince >= this.trimMs) this.slots.pop();
      else break;
    }
    return this.slots;
  }
}

// ---- the series ------------------------------------------------------------------------------
/** Appends new samples (a poll with `since` returns only those), keeps `keepS` seconds. */
export function mergeSeries(old: Sample[], fresh: Sample[], keepS = 300): Sample[] {
  const last = old.length ? old[old.length - 1].t : -Infinity;
  const out = old.concat(fresh.filter((s) => s.t > last));
  const end = out.length ? out[out.length - 1].t : 0;
  let i = 0;
  while (i < out.length && end - out[i].t > keepS) i++;
  return i ? out.slice(i) : out;
}

export interface StackLayer {
  id: string;
  /** per sample: [lower, upper] */
  bands: [number, number][];
}

/** The stacked layers, in `order` (bottom first). A backend absent from a sample adds 0 there:
 *  a sample lists the backends that decoded in that second. */
export function stack(series: Sample[], order: string[]): StackLayer[] {
  const base = series.map(() => 0);
  return order.map((id) => {
    const bands = series.map((s, i) => {
      const v = s.by[id] ?? 0;
      const lo = base[i];
      base[i] = lo + v;
      return [lo, lo + v] as [number, number];
    });
    return { id, bands };
  });
}

/** Every backend that appears in the series or the rows, the engine first. */
export function backendOrder(series: Sample[], rows: BackendRow[]): string[] {
  const ids = new Set<string>(rows.map((b) => b.id));
  for (const s of series) for (const k of Object.keys(s.by)) ids.add(k);
  return [...ids].sort((a, b) => (a === 'engine' ? -1 : b === 'engine' ? 1 : a.localeCompare(b)));
}

// ---- words --------------------------------------------------------------------------------------
export const STATE_WORD: Record<LaneState, string> = {
  queued: 'Queued',
  prefilling: 'Prefilling',
  waiting: 'Waiting for the first token',
  replaying: 'Replaying',
  thinking: 'Thinking',
  closing_reasoning: 'Closing the reasoning',
  tool_call: 'Tool call',
  writing: 'Writing',
  finishing: 'Finishing',
  done: 'Done',
};

/** The state's group, for its colour: wait (grey), prefill (cobalt), think (light cobalt),
 *  tool (violet), write (orange), done (green, or red for an error). */
export function stateTone(l: Lane): 'wait' | 'prefill' | 'think' | 'tool' | 'write' | 'done' | 'error' {
  switch (l.state) {
    case 'queued':
    case 'waiting':
      return 'wait';
    case 'prefilling':
    case 'replaying':
      return 'prefill';
    case 'thinking':
    case 'closing_reasoning':
      return 'think';
    case 'tool_call':
      return 'tool';
    case 'done':
      return l.finish === 'error' || l.finish === 'timeout' || (l.status ?? 200) >= 500 ? 'error' : 'done';
    default:
      return 'write';
  }
}

/** The state's line: "Tool call: read_file", "Prefilling 36 %", "Waiting for the first token",
 *  "Done: tool_calls". */
export function stateLine(l: Lane): string {
  if (l.state === 'tool_call') return l.tool ? `Tool call: ${l.tool}` : 'Tool call';
  if (l.state === 'prefilling' && l.prefill?.pct != null) return `Prefilling ${Math.round(l.prefill.pct)} %`;
  if (l.state === 'done') return l.finish && l.finish !== 'stop' ? `Done: ${l.finish.replace('_', ' ')}` : 'Done';
  return STATE_WORD[l.state] ?? l.state;
}

/** The kind's label: "TandemLLM", "vLLM", "SGLang", "DeepSeek", "OpenAI", "OpenAI-compatible". */
export function kindLabel(kind: string): string {
  return ({ tandemllm: 'TandemLLM', vllm: 'vLLM', sglang: 'SGLang', deepseek: 'DeepSeek', openai: 'OpenAI', generic: 'OpenAI-compatible' } as Record<string, string>)[kind] ?? kind;
}

/** The token bar's parts as fractions of `scale`: thinking, the rest (answer and calls). Null
 *  parts stay 0; a lane without a live count has no bar. */
export function tokenBar(l: Lane, scale: number): { think: number; out: number } | null {
  if (l.tokens == null || scale <= 0) return null;
  const th = Math.min(l.thinking_tokens ?? 0, l.tokens);
  return { think: Math.min(1, th / scale), out: Math.min(1, (l.tokens - th) / scale) };
}

/** The scale every token bar shares: the largest lane's tokens, rounded up to a nice number, at
 *  least 256 (a bar that rescales every second would look like it shrinks). */
export function barScale(lanes: Lane[], prev = 0): number {
  const top = Math.max(256, ...lanes.map((l) => l.tokens ?? 0));
  if (prev >= top && prev <= top * 4) return prev;
  const p = Math.pow(2, Math.ceil(Math.log2(top)));
  return Math.max(256, p);
}

/** Where a lane's live count comes from, in words (the tooltip of its token count). */
export function sourceWord(l: Lane): string {
  switch (l.tokens_source) {
    case 'engine':
      return 'counted by the engine';
    case 'frames':
      return 'stream frames, checked to carry one token each on this backend';
    case 'usage':
      return "the backend's usage";
    default:
      return l.state === 'done' ? 'the backend sent no usage' : 'not reported live: this backend\'s frames are not checked as one token each yet';
  }
}
