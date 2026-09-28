// The mock's live registry: what `server/live.py` + `server/activity.py` do,
// over the mock engine's own requests -- the tick simulator's request every ~15 s, the
// Playground's chats, the "busy" scenario (one decoding, one prefilling, one queued, two done),
// and the contract 1.1 scenarios: `agent-turn` (prefilling with progress, thinking, writing, a
// tool call with growing arguments, finishing, waiting for the client, the next request continuing
// it), `abandoned` (the client leaves mid-prefill), `stops` (one of each stop reason in `recent`)
// and `constrained`. `contract: '1.0'` makes the snapshot a 1.0 message for the fallback test.
// The words of `label` and `sentence` mirror server/activity.py. Never in dist/.

import type { ActivityState, EngineState, FinishingStep, Live, LiveActivity, LiveCounts, LiveEngine, LivePhase, LiveRecent, LiveRequest, LiveSample, LiveStop, LiveTimelineEntry, StopReason } from '../src/api/types.ts';
import type { LedgerRow } from './generate.ts';

export interface ActSegment {
  state: ActivityState;
  ms: number; // how long the state lasts
  detail?: string;
  step?: FinishingStep;
}

/** A scripted 1.1 request: the states it walks through and the numbers it shows on the way. */
export interface ActScript {
  segs: ActSegment[];
  total: number; // prompt tokens
  cached: number;
  tps: number; // decode tok/s
  prefillTps: number;
  progress?: 'chunked' | 'single_call';
  tool?: { name: string; argBytes: number };
  finish: StopReason;
  finishDetail?: string | null;
  constrained?: 'response_format' | 'tool_choice' | null;
  clientGoneAt?: number; // ms from arrival when the client leaves
  continues?: LiveActivity['continues'];
  queuePlace?: number;
}

export interface MockLiveRequest {
  row: LedgerRow;
  startedAt: number; // ms
  lockAt: number | null;
  firstAt: number | null;
  endedAt: number | null;
  tokens: number;
  blocks: number;
  marks: { t: number; n: number }[]; // the last three ticks
  scripted?: { tps: number; prefillMs: number; queueMs: number; total: number; advancedAt?: number }; // the busy scenario
  act?: ActScript; // a 1.1 scenario
  actAdvancedAt?: number;
  actIndex?: number; // the last script segment entered
  timeline: LiveTimelineEntry[];
  stop: LiveStop | null;
  clientGoneAt: number | null;
  toolNames: string[];
  state: ActivityState; // the current activity state (scripted or derived)
  stateSince: number;
  tokensIn: { thinking: number; content: number; tool: number }; // where the tokens went
}

const HISTORY = 300;
const KEEP_MS = 30_000;
const RECENT_MS = 15 * 60_000;
const WAIT_MS = 10 * 60_000;

const n = (x: number) => Math.round(x).toLocaleString('en-US');
const dur = (ms: number | null) => {
  if (ms == null) return '?';
  const s = ms / 1e3;
  return s < 60 ? `${s.toFixed(1)} s` : s < 3600 ? `${s.toFixed(0)} s` : `${Math.floor(s / 3600)} h ${Math.floor((s % 3600) / 60)} min`;
};
const ordinal = (k: number) => `${k}${k % 100 >= 10 && k % 100 <= 20 ? 'th' : ({ 1: 'st', 2: 'nd', 3: 'rd' } as Record<number, string>)[k % 10] ?? 'th'}`;

const SILENT: Record<string, string> = { queued: 'in the queue', prefilling: 'of silent prefill', replaying: 'while replaying', thinking: 'while thinking', closing_reasoning: 'while closing the reasoning', writing: 'while writing', tool_call: 'while writing a tool call', finishing: 'while finishing' };

/** server/activity.py::sentence, word for word. */
export function sentence(stop: Omit<LiveStop, 'sentence'>, r: { elapsed: number | null; lockMs: number | null; toolNames: string[]; maxTokens: number | null }): string {
  const sent = stop.tokens_sent || 0;
  const tokens = `${n(sent)} token${sent === 1 ? '' : 's'} sent`;
  switch (stop.reason) {
    case 'abandoned':
      if (stop.state === 'queued') return `abandoned by the client after ${dur(r.elapsed)} in the queue, ${tokens}`;
      if (stop.state === 'prefilling') return `abandoned by the client after ${dur(r.lockMs ?? r.elapsed)} of silent prefill, ${tokens}`;
      return `abandoned by the client ${SILENT[stop.state] ?? 'while running'} after ${dur(r.elapsed)}, ${tokens}`;
    case 'tool_calls': {
      const k = r.toolNames.length;
      return `tool call: ${k ? r.toolNames.join(', ') : 'a tool'}${k > 1 ? ` (${k} calls)` : ''} after ${dur(r.elapsed)}`;
    }
    case 'length':
      return r.maxTokens ? `stopped at the length limit (${n(r.maxTokens)} tokens)` : 'stopped at the length limit';
    case 'timeout':
      return `timed out after ${dur(r.elapsed)}`;
    case 'refused':
      return stop.detail === 'queue_full' ? 'refused: queue full' : stop.detail === 'queue_timeout' ? `refused: timed out in the queue after ${dur(r.elapsed)}` : 'refused: the engine is busy';
    case 'rejected':
      return stop.detail === 'prompt_too_long' ? 'rejected: the prompt is longer than the context window' : 'rejected: a bad request (400)';
    case 'cancelled':
      return 'cancelled: the server was shutting down';
    case 'error':
      return `error: ${stop.detail || 'internal error'} after ${n(sent)} tokens`;
    default: {
      const how = stop.detail === 'stop_string' ? 'at a stop string' : stop.detail === 'pattern_guard' ? 'by the repetition guard' : 'at the end of the answer';
      return `finished ${how} after ${dur(r.elapsed)}, ${tokens}`;
    }
  }
}

/** server/activity.py::label, word for word. */
export function label(a: Pick<LiveActivity, 'state' | 'queue' | 'prefill' | 'reasoning' | 'tool' | 'step' | 'stop'>): string {
  switch (a.state) {
    case 'queued':
      return a.queue?.place ? `Queued, about ${ordinal(a.queue.place)} in line` : 'Queued';
    case 'prefilling': {
      const p = a.prefill;
      if (p && p.done != null && p.total && p.pct != null) return `Prefilling ${n(p.done)} of ${n(p.total)} (${Math.round(p.pct)} %)`;
      return p?.total ? `Prefilling ${n(p.total)} tokens` : 'Prefilling';
    }
    case 'replaying':
      return 'Replaying a cached answer';
    case 'thinking':
      return 'Thinking';
    case 'closing_reasoning':
      return a.reasoning?.closed_by ? `Closing the reasoning (${a.reasoning.closed_by})` : 'Closing the reasoning';
    case 'writing':
      return 'Writing';
    case 'tool_call':
      return a.tool?.name ? `Calling tool ${a.tool.name}` : 'Calling a tool';
    case 'finishing':
      return a.step === 'flush' ? 'Finishing: flushing the last text' : a.step === 'saving_state' ? 'Finishing: saving the session state' : a.step === 'final_chunk' ? 'Finishing: sending the final chunk' : 'Finishing';
    default:
      return a.stop?.sentence ?? 'Done';
  }
}

export class MockLive {
  reqs = new Map<string, MockLiveRequest>();
  samples: LiveSample[] = [];
  served = 0;
  errors = 0;
  refused = 0;
  ends: number[] = [];
  lastPrefill: { at: number; tps: number } | null = null;
  recent: LiveRecent[] = [];
  /** `'1.0'` answers like a server (no engine, no activity, no recent). */
  contract: '1.0' | '1.1' = '1.1';
  /** `--live-activity off`: 1.1 with the new fields null. */
  activity = true;
  draining = false;
  seq = 0;
  private pending: number[] = [];
  private prev: { t: number; tokens: number; decoding: boolean } | null = null;
  private finishedTokens = 0;
  private credited = new Set<string>();
  private lastToolFinish: { at: number; request_id: string; tool_names: string[]; client_kind: string } | null = null;
  private engine: { state: EngineState; since: number; key: string } = { state: 'idle', since: Date.now(), key: 'idle' };

  constructor(
    private rng: () => number,
    served = 0,
  ) {
    this.served = served;
  }

  /** A request arrived: queued until `lock()`. */
  begin(row: LedgerRow, now = Date.now()): MockLiveRequest {
    const r: MockLiveRequest = { row, startedAt: now, lockAt: null, firstAt: null, endedAt: null, tokens: 0, blocks: 0, marks: [], timeline: [{ t_ms: 0, state: 'queued' }], stop: null, clientGoneAt: null, toolNames: [], state: 'queued', stateSince: now, tokensIn: { thinking: 0, content: 0, tool: 0 } };
    this.reqs.set(row.request_id, r);
    return r;
  }
  lock(r: MockLiveRequest, now = Date.now()): void {
    r.lockAt = now;
    this.enter(r, 'prefilling', now, r.row.cached_tokens ? `${r.row.cached_tokens} cached` : undefined);
  }
  first(r: MockLiveRequest, now = Date.now()): void {
    r.firstAt = now;
    r.tokens = Math.max(1, r.tokens);
    this.enter(r, r.row.thinking ? 'thinking' : 'writing', now);
  }
  token(r: MockLiveRequest, k = 1): void {
    r.tokens += k;
    r.blocks = Math.max(1, Math.ceil((r.tokens - 1) / 4.3));
    r.tokensIn[r.state === 'thinking' || r.state === 'closing_reasoning' ? 'thinking' : r.state === 'tool_call' ? 'tool' : 'content'] += k;
    // a derived (unscripted) request: thinking for the row's reasoning share, then writing, then the tool call
    if (!r.act && r.firstAt != null && r.endedAt == null) {
      const reasoning = r.row.thinking ? (r.row.reasoning_tokens ?? Math.round((r.row.completion_tokens ?? 400) * 0.35)) : 0;
      const total = r.row.completion_tokens ?? Infinity;
      const toolFrom = r.row.tool_calls ? Math.max(reasoning + 1, total - 40) : Infinity;
      const want: ActivityState = r.tokens >= toolFrom ? 'tool_call' : r.tokens > reasoning ? 'writing' : 'thinking';
      if (want !== r.state) this.enter(r, want, Date.now(), want === 'tool_call' ? 'get_weather' : undefined);
    }
  }
  /** The client's socket closed while the engine still works (the sampler sees it within a tick). */
  clientGone(r: MockLiveRequest, now = Date.now()): void {
    if (r.clientGoneAt == null) r.clientGoneAt = now;
  }
  finish(r: MockLiveRequest, finish: string | null, status = 200, now = Date.now(), tokens?: number, detail: string | null = null): void {
    if (tokens != null) r.tokens = tokens; // the final count is the response's own completion_tokens
    r.row.finish_reason = finish as LedgerRow['finish_reason'];
    r.row.status = status;
    r.endedAt = now;
    r.marks = [];
    this.ends.push(now);
    this.finishedTokens += r.tokens;
    if (finish === 'refused') this.refused++;
    else {
      this.served++;
      if (finish === 'error') this.errors++;
    }
    const tps = this.prefillTps(r);
    if (tps != null && !this.credited.has(r.row.request_id)) {
      this.credited.add(r.row.request_id);
      this.pending.push(tps);
      this.lastPrefill = { at: r.firstAt ?? now, tps };
    }
    // the stop block and its sentence
    const reason = (finish ?? (status >= 400 ? (status === 400 ? 'rejected' : 'refused') : 'stop')) as StopReason;
    const inState = r.state === 'done' ? 'writing' : r.state;
    if (reason === 'tool_calls' && !r.toolNames.length) r.toolNames = [r.act?.tool?.name ?? 'get_weather'];
    const silent = r.firstAt != null ? Math.min(now - (r.actAdvancedAt ?? r.firstAt), 900) : r.lockAt != null ? now - r.lockAt : now - r.startedAt;
    const partial: Omit<LiveStop, 'sentence'> = {
      reason,
      detail: detail ?? (reason === 'stop' ? 'eos' : reason === 'abandoned' ? (inState === 'prefilling' ? 'left_during_prefill' : inState === 'queued' ? 'left_while_queued' : 'write_failed') : reason === 'rejected' ? 'prompt_too_long' : reason === 'error' ? (r.row.error_type ?? 'internal error') : null),
      state: inState,
      tokens_sent: reason === 'abandoned' && inState === 'prefilling' ? 0 : r.tokens,
      silent_ms: Math.round(silent * 10) / 10,
      client_gone_ms: r.clientGoneAt != null ? Math.round((now - r.clientGoneAt) * 10) / 10 : null,
    };
    r.stop = { ...partial, sentence: sentence(partial, { elapsed: now - r.startedAt, lockMs: r.lockAt != null ? now - r.lockAt : null, toolNames: r.toolNames, maxTokens: r.row.max_tokens }) };
    this.enter(r, 'done', now);
    this.recent.unshift({
      request_id: r.row.request_id,
      ended_at: new Date(now).toISOString().replace(/\.\d{3}Z$/, 'Z'),
      client_kind: r.row.client_kind,
      path: r.timeline.map((e) => e.state).filter((s, i, a) => s !== 'done' && a.indexOf(s) === i),
      tokens: r.tokens,
      elapsed_ms: Math.round((now - r.startedAt) * 10) / 10,
      ttft_ms: r.firstAt != null && r.lockAt != null ? Math.round((r.firstAt - r.startedAt) * 100) / 100 : null,
      decode_tps: this.decodeTps(r, now),
      tool_names: [...r.toolNames],
      stop: r.stop,
    });
    if (this.recent.length > 20) this.recent.length = 20;
    if (reason === 'tool_calls') this.lastToolFinish = { at: now, request_id: r.row.request_id, tool_names: [...r.toolNames], client_kind: r.row.client_kind };
    else if (this.lastToolFinish) this.lastToolFinish = null;
  }

  private enter(r: MockLiveRequest, state: ActivityState, now: number, detail?: string): void {
    if (r.state === state) return;
    r.state = state;
    r.stateSince = now;
    const e: LiveTimelineEntry = { t_ms: Math.round((now - r.startedAt) * 10) / 10, state };
    if (detail) e.detail = detail;
    r.timeline.push(e);
    if (r.timeline.length > 16) r.timeline.splice(0, r.timeline.length - 16);
  }

  private prefillTps(r: MockLiveRequest): number | null {
    if (r.firstAt == null || r.lockAt == null || r.row.cache_source === 'response') return null;
    const ms = r.firstAt - r.lockAt;
    const fwd = (r.row.prompt_tokens ?? 0) - (r.row.cached_tokens ?? 0);
    return fwd && ms > 0 ? Math.round((fwd / ms) * 1000 * 100) / 100 : null;
  }

  private decodeTps(r: MockLiveRequest, now: number): number | null {
    const end = r.endedAt ?? now;
    const decodeMs = r.firstAt != null ? end - r.firstAt : null;
    return decodeMs && r.tokens > 1 ? Math.round(((r.tokens - 1) / decodeMs) * 1000 * 100) / 100 : null;
  }

  /** One sample a second, like the server's sampler. Also advances the scripted requests. */
  tick(now = Date.now()): void {
    this.advanceAll(now);
    let total = this.finishedTokens;
    let decoding = false;
    let running = 0;
    let waiting = 0;
    for (const r of this.reqs.values()) {
      if (r.endedAt != null) continue;
      total += r.tokens;
      if (r.lockAt == null) waiting++;
      if (r.firstAt == null) continue;
      running++;
      decoding = true;
      r.marks.push({ t: now, n: r.tokens });
      if (r.marks.length > 3) r.marks.shift();
      if (!this.credited.has(r.row.request_id)) {
        const tps = this.prefillTps(r);
        if (tps != null) {
          this.credited.add(r.row.request_id);
          this.pending.push(tps);
          this.lastPrefill = { at: r.firstAt, tps };
        }
      }
    }
    let decode: number | null = null;
    if (this.prev) {
      const dt = (now - this.prev.t) / 1000;
      if (dt > 0 && (decoding || this.prev.decoding)) decode = Math.round(((total - this.prev.tokens) / dt) * 100) / 100;
    }
    const prefill = this.pending.length ? this.pending[this.pending.length - 1] : null;
    this.pending = [];
    this.samples.push({ t: Math.round(now) / 1000, decode_tps: decode, prefill_tps: prefill, running, waiting, tokens: total });
    if (this.samples.length > HISTORY) this.samples.splice(0, this.samples.length - HISTORY);
    this.prev = { t: now, tokens: total, decoding };
    this.prune(now);
  }

  /** The scripted requests move between the one-second ticks too (the 4 Hz stream calls this). */
  advanceAll(now = Date.now()): void {
    for (const r of this.reqs.values()) {
      if (r.endedAt != null) continue;
      if (r.act) this.advanceAct(r, now);
      else if (r.scripted) this.advance(r, now);
    }
  }

  private advance(r: MockLiveRequest, now: number): void {
    const s = r.scripted!;
    const since = now - r.startedAt;
    if (r.lockAt == null && since >= s.queueMs) this.lock(r, r.startedAt + s.queueMs);
    if (r.lockAt != null && r.firstAt == null && now - r.lockAt >= s.prefillMs) {
      this.first(r, r.lockAt + s.prefillMs);
      r.tokens = 1;
    }
    if (r.firstAt != null) {
      // tokens accrue per elapsed second with a little jitter, so the 2 s rate moves like a real one
      const from = s.advancedAt ?? r.firstAt;
      const dt = Math.max(0, now - from) / 1000;
      const add = Math.floor(dt * s.tps * (0.9 + this.rng() * 0.2));
      if (add > 0) {
        this.token(r, add);
        s.advancedAt = now;
      }
      if (r.tokens >= s.total) this.finish(r, 'stop', 200, now);
    }
  }

  /** Walk a 1.1 script: the state at `now`, the lock and first-token stamps, the tokens, the stop.
   *  Each segment is entered once (`actIndex` remembers the last), so the timeline keeps its order. */
  private advanceAct(r: MockLiveRequest, now: number): void {
    const a = r.act!;
    const el = now - r.startedAt;
    if (a.clientGoneAt != null && el >= a.clientGoneAt) this.clientGone(r, r.startedAt + a.clientGoneAt);
    const starts: number[] = [];
    let t = 0;
    for (const seg of a.segs) {
      starts.push(t);
      t += seg.ms;
    }
    let idx = a.segs.findIndex((seg, k) => el < starts[k] + seg.ms);
    if (idx < 0) idx = a.segs.length; // past the end: every segment done, then the stop
    for (let k = (r.actIndex ?? -1) + 1; k < Math.min(idx + 1, a.segs.length); k++) this.enterScripted(r, a.segs[k], r.startedAt + starts[k]);
    r.actIndex = Math.min(idx, a.segs.length - 1);
    const ended = idx >= a.segs.length;
    const upto = ended ? r.startedAt + t : now;
    if (r.firstAt != null) {
      // tokens since the last advance, credited to the state they were made in (a jump with `at`
      // crosses several segments at once, or the whole script)
      const from = r.actAdvancedAt ?? r.firstAt;
      let added = 0;
      a.segs.forEach((seg, k) => {
        if (seg.state !== 'thinking' && seg.state !== 'writing' && seg.state !== 'tool_call' && seg.state !== 'closing_reasoning') return;
        const lo = Math.max(from, r.startedAt + starts[k]);
        const hi = Math.min(upto, r.startedAt + starts[k] + seg.ms);
        const k2 = Math.floor((Math.max(0, hi - lo) / 1000) * a.tps);
        if (k2 > 0) {
          r.tokensIn[seg.state === 'tool_call' ? 'tool' : seg.state === 'writing' ? 'content' : 'thinking'] += k2;
          added += k2;
        }
      });
      if (added > 0) {
        r.tokens += added;
        r.blocks = Math.max(1, Math.ceil((r.tokens - 1) / 4.3));
        r.actAdvancedAt = upto;
      }
    }
    if (ended) this.finish(r, a.finish, a.finish === 'rejected' ? 400 : a.finish === 'refused' ? 503 : 200, upto, undefined, a.finishDetail ?? null);
  }

  /** After a scenario seeds tokens in one go: no delta across the seam, in the samples or the next tick. */
  private rebase(): void {
    let total = this.finishedTokens;
    for (const r of this.reqs.values()) if (r.endedAt == null) total += r.tokens;
    for (const smp of this.samples) smp.tokens = total;
    this.prev = null;
  }

  private enterScripted(r: MockLiveRequest, seg: ActSegment, at: number): void {
    if (seg.state === 'prefilling' && r.lockAt == null) r.lockAt = at;
    if ((seg.state === 'thinking' || seg.state === 'writing' || seg.state === 'tool_call' || seg.state === 'replaying') && r.firstAt == null) {
      r.firstAt = at;
      r.tokens = 1;
    }
    if (seg.state === 'tool_call' && seg.detail && !r.toolNames.includes(seg.detail)) r.toolNames.push(seg.detail);
    this.enter(r, seg.state, at, seg.detail);
  }

  private prune(now: number): void {
    for (const [id, r] of this.reqs) if (r.endedAt != null && now - r.endedAt > KEEP_MS) this.reqs.delete(id);
    this.ends = this.ends.filter((t) => now - t <= 60_000);
    this.recent = this.recent.filter((x) => now - Date.parse(x.ended_at) <= RECENT_MS);
  }

  private rowOf(r: MockLiveRequest, now: number): LiveRequest {
    const row = r.row;
    const done = r.endedAt != null;
    const phase: LivePhase = done ? 'done' : r.lockAt == null ? 'queued' : r.firstAt == null ? 'prefill' : 'decode';
    const ran = r.lockAt != null && row.prompt_tokens != null;
    const end = done ? r.endedAt! : now;
    const decodeMs = r.firstAt != null ? end - r.firstAt : null;
    const decodeTps = this.decodeTps(r, now);
    let nowTps: number | null = null;
    if (!done && r.marks.length >= 2) {
      const a = r.marks[0];
      const b = r.marks[r.marks.length - 1];
      nowTps = b.t > a.t ? Math.round(((b.n - a.n) / (b.t - a.t)) * 1000 * 100) / 100 : null;
    }
    const queueMs = r.lockAt != null ? r.lockAt - r.startedAt : null;
    const promptMs = r.firstAt != null && r.lockAt != null ? r.firstAt - r.lockAt : null;
    const out: LiveRequest = {
      request_id: row.request_id,
      phase,
      finish_reason: done ? row.finish_reason : null,
      status: row.status,
      model: row.model,
      client: { id: row.client_id, kind: row.client_kind },
      endpoint: row.endpoint,
      stream: row.stream,
      thinking: row.thinking,
      temperature: r.lockAt != null ? 0 : null,
      prompt_tokens: ran ? row.prompt_tokens : null,
      cached_tokens: ran ? (row.cached_tokens ?? 0) : null,
      forwarded_tokens: ran ? (row.prompt_tokens ?? 0) - (row.cached_tokens ?? 0) : null,
      tokens: r.tokens,
      blocks: r.blocks || null,
      tokens_per_block: r.blocks && r.tokens > 1 ? Math.round(((r.tokens - 1) / r.blocks) * 100) / 100 : null,
      elapsed_ms: Math.round((end - r.startedAt) * 100) / 100,
      queue_ms: queueMs,
      prompt_ms: promptMs,
      ttft_ms: queueMs != null && promptMs != null ? queueMs + promptMs : null,
      decode_ms: decodeMs != null ? Math.round(decodeMs * 100) / 100 : null,
      prefill_tps: this.prefillTps(r),
      decode_tps: decodeTps,
      decode_tps_now: nowTps,
      cache_source: ran ? row.cache_source : null,
      max_tokens: row.max_tokens,
      ended_ms_ago: done ? Math.round((now - r.endedAt!) * 10) / 10 : null,
    };
    if (this.contract === '1.1') {
      out.activity = this.activity ? this.activityOf(r, out, now, nowTps ?? decodeTps) : null;
      out.timeline = this.activity ? [...r.timeline] : null;
    }
    return out;
  }

  private activityOf(r: MockLiveRequest, row: LiveRequest, now: number, rate: number | null): LiveActivity {
    const a = r.act;
    const done = r.endedAt != null;
    const end = done ? r.endedAt! : now;
    const state = r.state;
    const total = row.prompt_tokens ?? a?.total ?? 0;
    const cached = row.cached_tokens ?? a?.cached ?? 0;
    // prefill progress: scripted requests count chunks of 2,048 at the script's rate; derived ones show the total only
    let prefill: LiveActivity['prefill'] = null;
    if (r.lockAt != null && total) {
      const single = a?.progress === 'single_call';
      const ptps = a?.prefillTps ?? row.prefill_tps ?? null;
      let doneTok: number | null = null;
      let pct: number | null = null;
      let eta: number | null = null;
      if (r.firstAt != null || done) {
        doneTok = total;
        pct = 100;
      } else if (a && !single && ptps) {
        const fwd = Math.min(total - cached, Math.floor(((end - r.lockAt) / 1000) * ptps));
        doneTok = cached + Math.floor(fwd / 2048) * 2048;
        pct = Math.round((doneTok / total) * 1000) / 10;
        eta = Math.max(0, Math.round(((total - doneTok) / ptps) * 1000));
      }
      prefill = { done: single && r.firstAt == null ? null : doneTok, total, cached, pct: single && r.firstAt == null ? null : pct, tps_now: r.firstAt == null && doneTok != null && doneTok > cached ? ptps : null, tps_avg: row.prefill_tps ?? (r.firstAt != null ? ptps : null), eta_ms: eta, progress: single ? 'single_call' : 'chunked', at_ms: doneTok != null ? Math.round((end - r.startedAt) * 10) / 10 : null };
    }
    const thinkingTokens = Math.min(r.tokens, r.tokensIn.thinking);
    const toolTokens = Math.min(r.tokens - thinkingTokens, r.tokensIn.tool);
    const rounds = r.blocks || null;
    const decode: LiveActivity['decode'] = r.firstAt != null ? { tokens: r.tokens, thinking_tokens: thinkingTokens, content_tokens: Math.max(0, r.tokens - thinkingTokens - toolTokens), tool_tokens: toolTokens, tps_now: done ? null : rate, tps_avg: row.decode_tps, rounds, tokens_per_round: row.tokens_per_block, tokens_per_round_now: done ? null : rate != null ? Math.round((4.1 + this.rng() * 0.6) * 100) / 100 : null, ms_per_round: rate ? Math.round((1000 / rate) * 4.2 * 10) / 10 : null, ms_per_round_now: done || !rate ? null : Math.round((1000 / rate) * 4.2 * 10) / 10, accept_mean: 3.05 } : null;
    let tool: LiveActivity['tool'] = null;
    if (state === 'tool_call' || (done && r.toolNames.length)) {
      const bytesTotal = a?.tool?.argBytes ?? 640;
      tool = { index: 0, name: r.toolNames[r.toolNames.length - 1] ?? a?.tool?.name ?? null, arg_bytes: Math.round(bytesTotal * this.toolShare(r, now)), calls_done: done ? r.toolNames.length : 0 };
    }
    const queue: LiveActivity['queue'] = state === 'queued' ? { place: a?.queuePlace ?? this.placeOf(r), wait_ms: Math.round((end - r.startedAt) * 10) / 10, timeout_s: 120, place_is_estimate: true } : null;
    const stop = r.stop;
    const partial = {
      state,
      queue,
      prefill,
      reasoning: r.row.thinking ? { closed_by: state === 'thinking' ? null : ('model' as const), tokens: thinkingTokens } : null,
      tool,
      step: state === 'finishing' ? (a?.segs.find((s) => s.state === 'finishing')?.step ?? 'saving_state') : null,
      stop,
    };
    return {
      ...partial,
      label: label(partial),
      since_ms: Math.round((end - r.stateSince) * 10) / 10,
      constrained: a?.constrained ?? null,
      decode,
      client: { connected: r.clientGoneAt == null, silent_ms: Math.round((r.firstAt != null && !done ? Math.min(end - (r.actAdvancedAt ?? r.firstAt), 900) : end - (r.lockAt ?? r.startedAt)) * 10) / 10, gone_ms: r.clientGoneAt != null ? Math.round((end - r.clientGoneAt) * 10) / 10 : null },
      continues: a?.continues ?? null,
    };
  }

  private toolShare(r: MockLiveRequest, now: number): number {
    if (r.endedAt != null) return 1;
    const a = r.act;
    const seg = a?.segs.find((s) => s.state === 'tool_call');
    if (!a || !seg || r.state !== 'tool_call') return 0.4;
    return Math.min(1, Math.max(0.02, (now - r.stateSince) / seg.ms));
  }

  private placeOf(r: MockLiveRequest): number {
    let k = 1;
    for (const o of this.reqs.values()) if (o !== r && o.endedAt == null && o.lockAt == null && o.startedAt < r.startedAt) k++;
    return k;
  }

  private engineOf(rows: LiveRequest[], now: number): LiveEngine {
    const running = rows.filter((x) => x.phase !== 'done');
    let state: EngineState = 'idle';
    let lbl = 'Idle';
    let waiting: LiveEngine['waiting_for_client'] = null;
    if (this.draining) {
      state = 'draining';
      lbl = 'Draining';
    } else if (running.length) {
      state = 'busy';
      const rank: Record<string, number> = { tool_call: 0, writing: 1, thinking: 2, closing_reasoning: 3, finishing: 4, replaying: 5, prefilling: 6, queued: 7 };
      const top = [...running].sort((x, y) => (rank[x.activity?.state ?? 'queued'] ?? 9) - (rank[y.activity?.state ?? 'queued'] ?? 9))[0];
      lbl = top.activity?.label ?? (top.phase === 'prefill' ? 'Prefilling' : top.phase === 'queued' ? 'Queued' : 'Decoding');
    } else if (this.activity && this.lastToolFinish && now - this.lastToolFinish.at < WAIT_MS) {
      state = 'waiting_for_client';
      const w = this.lastToolFinish;
      waiting = { request_id: w.request_id, tool_names: w.tool_names, since_ms: Math.round((now - w.at) * 10) / 10, client_kind: w.client_kind };
      lbl = w.tool_names.length ? `Waiting for client: running tool ${w.tool_names.join(', ')}` : 'Waiting for client';
    }
    const key = state === 'busy' ? `busy:${running.map((x) => x.request_id).join(',')}` : state;
    if (key !== this.engine.key) this.engine = { state, since: now, key };
    const kv = running.reduce((m, x) => Math.max(m, (x.prompt_tokens ?? 0) + x.tokens), 0);
    return {
      state,
      label: lbl,
      since_ms: Math.round((now - this.engine.since) * 10) / 10,
      model: 'qwen38-spark-engine',
      version: '0.1.0-rc8',
      draining: this.draining,
      kv: { length: kv || 0, max_len: 262144 },
      memory: { mem_available_gib: Math.round((39.7 - kv / 60000) * 10) / 10, rss_gib: 71.4 },
      store: { entries: 31, bytes_gib: 9.8 },
      waiting_for_client: this.activity ? waiting : null,
    };
  }

  snapshot(now = Date.now(), history = true): Live {
    this.admit(now);
    this.prune(now);
    this.advanceAll(now);
    const live = [...this.reqs.values()].filter((r) => r.endedAt == null);
    const done = [...this.reqs.values()].filter((r) => r.endedAt != null).sort((a, b) => b.endedAt! - a.endedAt!);
    const order: Record<string, number> = { decode: 0, prefill: 1, queued: 2 };
    const rows = live.map((r) => this.rowOf(r, now)).sort((a, b) => order[a.phase] - order[b.phase]);
    rows.push(...done.map((r) => this.rowOf(r, now)));
    const counts: LiveCounts = {
      in_flight: live.length,
      queued: live.filter((r) => r.lockAt == null).length,
      prefilling: live.filter((r) => r.lockAt != null && r.firstAt == null).length,
      decoding: live.filter((r) => r.firstAt != null).length,
      completed_1m: this.ends.length,
      served: this.served,
      errors: this.errors,
      refused: this.refused,
    };
    const last3 = this.samples.slice(-3);
    let nowDecode: number | null = null;
    if (last3.length === 3 && last3.some((s) => s.running)) {
      const dt = last3[2].t - last3[0].t;
      nowDecode = dt > 0 ? Math.round(((last3[2].tokens - last3[0].tokens) / dt) * 100) / 100 : null;
    }
    const dec = live.find((r) => r.firstAt != null);
    const busy = live.length > 0;
    const out: Live = {
      contract_version: this.contract,
      generated_at: new Date(now).toISOString().replace(/\.\d{3}Z$/, 'Z'),
      interval_s: this.contract === '1.1' && this.activity && busy ? 0.25 : 1.0,
      counts,
      now: {
        decode_tps: nowDecode,
        prefill_tps: this.lastPrefill?.tps ?? null,
        prefilling: counts.prefilling > 0,
        tokens_per_block: dec && dec.blocks && dec.tokens > 1 ? Math.round(((dec.tokens - 1) / dec.blocks) * 100) / 100 : null,
        last_prefill_ms_ago: this.lastPrefill ? Math.round((now - this.lastPrefill.at) * 10) / 10 : null,
      },
      requests: rows,
      sample: this.samples.length ? this.samples[this.samples.length - 1] : null,
    };
    if (this.contract === '1.1') {
      out.seq = ++this.seq;
      out.engine = this.engineOf(rows, now);
      out.recent = this.activity ? [...this.recent] : null;
      out.sampler = { ticks: this.seq, encodes: this.seq, tick_us_last: 410, tick_us_mean: 399, tick_us_max: 1830, tick_cpu_us_mean: 258.6, tick_cpu_us_max: 1834.6 };
    }
    if (history) out.history = [...this.samples];
    return out;
  }

  /** The busy scenario: one decoding, one prefilling, one queued, two done (a stop and an error),
   *  and four minutes of history (two decode runs with a gap between them, three prefills). */
  scenario(makeRow: (forced: Partial<LedgerRow>) => LedgerRow, now = Date.now()): void {
    const add = (forced: Partial<LedgerRow>, script: MockLiveRequest['scripted'], startedAgo: number) => {
      const r = this.begin(makeRow(forced), now - startedAgo);
      r.scripted = script;
      this.advance(r, now);
      return r;
    };
    add({ client_id: 'k:3f9a1c0e2b7d', client_kind: 'open-webui', endpoint: 'chat', stream: true, thinking: true, prompt_tokens: 1959, cached_tokens: 1536, cache_source: 'prefix', max_tokens: 32768, reasoning_tokens: 180, completion_tokens: 4000, tool_calls: 0 }, { tps: 43, prefillMs: 2240, queueMs: 0.4, total: 4000 }, 8000);
    add({ client_id: 'anon', client_kind: 'curl', endpoint: 'chat', stream: true, thinking: false, prompt_tokens: 8192, cached_tokens: 0, cache_source: 'none', max_tokens: 4096, tool_calls: 0 }, { tps: 38, prefillMs: 9000, queueMs: 900, total: 3000 }, 2200);
    add({ client_id: 'k:0b7e55c3d1f2', client_kind: 'dashboard', endpoint: 'completions', stream: false, thinking: false, prompt_tokens: 640, cached_tokens: 0, cache_source: 'none', max_tokens: 512, tool_calls: 0 }, { tps: 41, prefillMs: 700, queueMs: 20000, total: 2000 }, 800);
    // two finished ones, with final numbers
    const okRow = makeRow({ client_id: 'k:0b7e55c3d1f2', client_kind: 'dashboard', endpoint: 'chat', stream: true, thinking: true, prompt_tokens: 412, cached_tokens: 0, cache_source: 'none', max_tokens: 32768, reasoning_tokens: 140, completion_tokens: 388, tool_calls: 0 });
    const ok = this.begin(okRow, now - 22_000);
    this.lock(ok, ok.startedAt + 0.3);
    this.first(ok, ok.lockAt! + 705.1);
    this.token(ok, 387);
    this.finish(ok, 'stop', 200, ok.firstAt! + 8901.2);
    const errRow = makeRow({ client_id: 'anon', client_kind: 'curl', endpoint: 'chat', stream: true, thinking: false, prompt_tokens: 64, cached_tokens: 0, cache_source: 'none', max_tokens: 200, error_type: 'RuntimeError', tool_calls: 0 });
    const err = this.begin(errRow, now - 25_600);
    this.lock(err, err.startedAt + 0.2);
    this.first(err, err.lockAt! + 120.4);
    this.token(err, 4);
    this.finish(err, 'error', 200, err.firstAt! + 210);
    this.seedHistory(now);
  }

  private seedHistory(now: number): void {
    // four minutes of history, counted back from the total as it stands now, so the running
    // totals are continuous across the seam and the 2 s figure has no jump to explain
    let total = this.finishedTokens;
    for (const r of this.reqs.values()) if (r.endedAt == null) total += r.tokens;
    // the history's tokens count back from the total; a fresh registry has too few to count
    // back from, so the finished total is lifted first (a mock detail: the server's total only grows)
    const need = 240 * 45 - total;
    if (need > 0) {
      this.finishedTokens += need;
      total += need;
    }
    const seeded: LiveSample[] = [];
    for (let i = 1; i <= 240; i++) {
      const t = now - i * 1000;
      const inRun = (i > 150 && i <= 235) || (i > 20 && i <= 95);
      const asleep = i > 95 && i <= 150 && i !== 120; // the sampler slept: a gap, with one lone prefill dot
      if (asleep) continue;
      const rate = inRun ? 38 + Math.sin(i / 9) * 4 + (this.rng() - 0.5) * 5 : null;
      const prefill = i === 235 ? 1180.4 : i === 120 ? 2410.5 : i === 95 ? 640.2 : null;
      seeded.push({ t: Math.round(t) / 1000, decode_tps: rate == null ? null : Math.round(rate * 100) / 100, prefill_tps: prefill, running: inRun ? 1 : 0, waiting: 0, tokens: total });
      if (rate != null) total -= Math.round(rate);
    }
    this.samples = seeded.reverse();
    this.prev = null; // the next tick has no delta to a time before the scenario existed
  }

  /** Everything gone: requests, recent stops, the waiting-for-client memory. */
  clear(): void {
    this.reqs.clear();
    this.hidden = [];
    this.recent = [];
    this.lastToolFinish = null;
    this.draining = false;
  }

  // ---- the 1.1 scenarios --------------------------------------------------------------

  /**
   * `agent-turn`: an opencode-shaped turn. Queued 0.4 s, a chunked prefill of 48,210 tokens
   * (12,288 cached) at 2,310 tok/s, thinking 12 s, writing 6 s, a `write_file` call whose
   * arguments grow to 18 KB over 6 s, finishing, `tool_calls`; then the engine waits for the
   * client for 6 s and the next request arrives, continuing the first. `at` (ms) starts the
   * scenario that far in, so a screenshot can pick a moment; the second request gets the same
   * offset. The whole thing plays once and then leaves its stops in `recent`.
   */
  agentTurn(makeRow: (forced: Partial<LedgerRow>) => LedgerRow, now = Date.now(), at = 0): void {
    const firstId = 'chatcmpl-77aa1c0e2b7d4f5a9c3b';
    const row1 = makeRow({ request_id: firstId, client_id: 'k:9a1b2c3d4e5f', client_kind: 'opencode', endpoint: 'chat', stream: true, thinking: true, prompt_tokens: 48210, cached_tokens: 12288, cache_source: 'session', max_tokens: 32000, completion_tokens: 1422, reasoning_tokens: 611, tool_calls: 1, error_type: null });
    const r1 = this.begin(row1, now - at);
    r1.act = {
      segs: [
        { state: 'queued', ms: 400 },
        { state: 'prefilling', ms: 15_600, detail: '12288 cached' },
        { state: 'thinking', ms: 12_000 },
        { state: 'writing', ms: 6000 },
        { state: 'tool_call', ms: 6000, detail: 'write_file' },
        { state: 'finishing', ms: 400, step: 'saving_state' },
      ],
      total: 48210,
      cached: 12288,
      tps: 43,
      prefillTps: 2310,
      progress: 'chunked',
      tool: { name: 'write_file', argBytes: 18234 },
      finish: 'tool_calls',
    };
    const secondId = 'chatcmpl-3f1c8d2e4b6a7c9d1e2f';
    const row2 = makeRow({ request_id: secondId, client_id: 'k:9a1b2c3d4e5f', client_kind: 'opencode', endpoint: 'chat', stream: true, thinking: true, prompt_tokens: 48930, cached_tokens: 48210, cache_source: 'session', max_tokens: 32000, completion_tokens: 380, reasoning_tokens: 120, tool_calls: 0, error_type: null });
    const gap = 6000;
    const r2 = this.begin(row2, now - at + 40_400 + gap);
    r2.act = {
      segs: [
        { state: 'queued', ms: 300 },
        { state: 'prefilling', ms: 2000, detail: '48210 cached' },
        { state: 'thinking', ms: 4000 },
        { state: 'writing', ms: 8000 },
        { state: 'finishing', ms: 300, step: 'final_chunk' },
      ],
      total: 48930,
      cached: 48210,
      tps: 44,
      prefillTps: 2410,
      progress: 'chunked',
      finish: 'stop',
      finishDetail: 'eos',
      continues: { request_id: firstId, gap_ms: gap + 400, tool_names: ['write_file'], inferred: false },
    };
    this.advanceAll(now);
    this.rebase();
    // the second request is not "in the registry" before it arrives: hide it until then
    if (r2.startedAt > now) this.hidden.push(r2);
    this.reqs.delete(secondId);
    if (r2.startedAt <= now) this.reqs.set(secondId, r2);
  }
  private hidden: MockLiveRequest[] = [];

  /** `abandoned`: a 60,014-token prefill whose client leaves after 3.6 s; the handler notices at the next chunk. */
  abandoned(makeRow: (forced: Partial<LedgerRow>) => LedgerRow, now = Date.now(), at = 0): void {
    const row = makeRow({ request_id: 'chatcmpl-d9220271b5524d5daf07', client_id: 'k:9a1b2c3d4e5f', client_kind: 'opencode', endpoint: 'chat', stream: true, thinking: true, prompt_tokens: 60014, cached_tokens: 0, cache_source: 'none', max_tokens: 32000, completion_tokens: 0, reasoning_tokens: 0, tool_calls: 0, error_type: null });
    const r = this.begin(row, now - at);
    r.act = {
      segs: [
        { state: 'queued', ms: 200 },
        { state: 'prefilling', ms: 8000 },
      ],
      total: 60014,
      cached: 0,
      tps: 0,
      prefillTps: 2200,
      progress: 'chunked',
      finish: 'abandoned',
      finishDetail: 'left_during_prefill',
      clientGoneAt: 3600,
    };
    this.advanceAll(now);
    this.rebase();
  }

  /** `stops`: one finished request per stop reason in `recent`, newest first. */
  stops(makeRow: (forced: Partial<LedgerRow>) => LedgerRow, now = Date.now()): void {
    const cases: { reason: StopReason; detail: string | null; state: ActivityState; tokens: number; tools?: string[]; gone?: boolean; kind: LedgerRow['client_kind']; secsAgo: number }[] = [
      { reason: 'stop', detail: 'eos', state: 'writing', tokens: 812, kind: 'open-webui', secsAgo: 20 },
      { reason: 'length', detail: null, state: 'writing', tokens: 32000, kind: 'opencode', secsAgo: 65 },
      { reason: 'tool_calls', detail: null, state: 'tool_call', tokens: 412, tools: ['write_file', 'bash'], kind: 'opencode', secsAgo: 110 },
      { reason: 'timeout', detail: null, state: 'thinking', tokens: 2900, kind: 'openai-sdk', secsAgo: 170 },
      { reason: 'abandoned', detail: 'left_during_prefill', state: 'prefilling', tokens: 0, gone: true, kind: 'opencode', secsAgo: 230 },
      { reason: 'error', detail: 'RuntimeError', state: 'writing', tokens: 5, kind: 'curl', secsAgo: 300 },
      { reason: 'refused', detail: 'queue_full', state: 'queued', tokens: 0, kind: 'curl', secsAgo: 360 },
      { reason: 'rejected', detail: 'prompt_too_long', state: 'queued', tokens: 0, kind: 'openai-sdk', secsAgo: 420 },
      { reason: 'cancelled', detail: 'shutting_down', state: 'writing', tokens: 140, kind: 'dashboard', secsAgo: 480 },
      { reason: 'stop', detail: 'stop_string', state: 'writing', tokens: 3, kind: 'opencode', secsAgo: 540 },
    ];
    for (const c of cases.reverse()) {
      const end = now - c.secsAgo * 1000;
      const row = makeRow({ client_kind: c.kind, endpoint: 'chat', stream: true, thinking: c.state === 'thinking' || c.reason === 'timeout', prompt_tokens: c.reason === 'rejected' ? 300_000 : 8192, cached_tokens: 0, cache_source: 'none', max_tokens: 32000, completion_tokens: c.tokens, reasoning_tokens: 0, tool_calls: c.tools?.length ?? 0, error_type: c.reason === 'error' ? c.detail : null });
      const r = this.begin(row, end - (c.reason === 'abandoned' ? 73_400 : 41_230));
      if (c.state !== 'queued') this.lock(r, r.startedAt + 2.1);
      if (c.state !== 'queued' && c.state !== 'prefilling') {
        this.first(r, r.lockAt! + 2100);
        this.enter(r, c.state === 'tool_call' ? 'writing' : c.state, r.firstAt! + 1);
        if (c.state === 'tool_call') this.enter(r, 'tool_call', r.firstAt! + 30_000, c.tools![0]);
        r.tokens = Math.max(1, c.tokens);
      }
      if (c.tools) r.toolNames = [...c.tools];
      if (c.gone) this.clientGone(r, end - 218);
      this.finish(r, c.reason, c.reason === 'rejected' ? 400 : c.reason === 'refused' ? 503 : 200, end, c.tokens, c.detail);
      this.reqs.delete(row.request_id); // older than the 30 s tail: only `recent` keeps it
    }
    this.lastToolFinish = null;
    this.ends = []; // these ended minutes ago
    this.rebase();
  }

  /** `constrained`: a JSON-schema answer being written, and a forced tool call. */
  constrained(makeRow: (forced: Partial<LedgerRow>) => LedgerRow, now = Date.now(), at = 3000): void {
    const a = this.begin(makeRow({ request_id: 'chatcmpl-c0n5tra1ned0000000001', client_kind: 'openai-sdk', endpoint: 'chat', stream: true, thinking: false, prompt_tokens: 1200, cached_tokens: 0, cache_source: 'none', max_tokens: 2048, completion_tokens: 600, reasoning_tokens: 0, tool_calls: 0, error_type: null }), now - at);
    a.act = { segs: [{ state: 'queued', ms: 100 }, { state: 'prefilling', ms: 600 }, { state: 'writing', ms: 20_000 }], total: 1200, cached: 0, tps: 40, prefillTps: 2000, finish: 'stop', constrained: 'response_format' };
    const b = this.begin(makeRow({ request_id: 'chatcmpl-c0n5tra1ned0000000002', client_kind: 'opencode', endpoint: 'chat', stream: true, thinking: false, prompt_tokens: 9000, cached_tokens: 8192, cache_source: 'prefix', max_tokens: 2048, completion_tokens: 200, reasoning_tokens: 0, tool_calls: 1, error_type: null }), now - at + 2500);
    b.act = { segs: [{ state: 'queued', ms: 2500 }, { state: 'prefilling', ms: 500 }, { state: 'tool_call', ms: 15_000, detail: 'write_file' }], total: 9000, cached: 8192, tps: 42, prefillTps: 2300, tool: { name: 'write_file', argBytes: 4200 }, finish: 'tool_calls', constrained: 'tool_choice', queuePlace: 2 };
    this.advanceAll(now);
    this.rebase();
  }

  /**
   * `loop`: an opencode tool loop over a ~68k-token session, the shape the layout-shift test
   * streams at 4 events a second. Seven finished turns in the 30 s tail, each continuing
   * the one before it after the client ran a tool (long tool names, so the "continues" note is
   * long), and one turn in flight that walks a single-call prefill, thinking, writing and a
   * `todowrite` call: the cells, the two figures and the Now line change every event.
   */
  loop(makeRow: (forced: Partial<LedgerRow>) => LedgerRow, now = Date.now(), at = 0): void {
    const tools = ['grep', 'read', 'edit', 'chrome-devtools_navigate_page', 'chrome-devtools_evaluate_script', 'chrome-devtools_list_console_messages', 'edit'];
    const id = (k: number) => `chatcmpl-${(0x3fedcdc96 + k * 0x1b2c3d).toString(16).padStart(9, '0')}e1a2b3c4d5`;
    let prev: string | null = null;
    let prevTool: string | null = null;
    const turns = tools.length;
    for (let k = 0; k <= turns; k++) {
      const live = k === turns;
      const rid = id(k);
      const prompt = 66_973 + k * 220;
      const row = makeRow({ request_id: rid, client_id: 'k:9a1b2c3d4e5f', client_kind: 'opencode', endpoint: 'chat', stream: true, thinking: true, prompt_tokens: prompt, cached_tokens: live ? 0 : prompt - 900 - k * 40, cache_source: live ? 'none' : 'session', max_tokens: 32000, completion_tokens: 60 + k * 30, reasoning_tokens: 8 + k * 6, tool_calls: 1, error_type: null });
      // finished turns end 2.5 s apart, the newest 2 s ago; the live one starts `at` ms ago
      const start = live ? now - at : now - 2000 - (turns - 1 - k) * 2500 - 3400;
      const r = this.begin(row, start);
      r.act = live
        ? {
            segs: [
              { state: 'queued', ms: 100 },
              { state: 'prefilling', ms: 3000 },
              { state: 'thinking', ms: 2000 },
              { state: 'writing', ms: 1500 },
              { state: 'tool_call', ms: 20_000, detail: 'todowrite' },
              { state: 'finishing', ms: 200, step: 'saving_state' },
            ],
            total: prompt,
            cached: 0,
            tps: 64 + k,
            prefillTps: 480,
            progress: 'single_call',
            tool: { name: 'todowrite', argBytes: 410 },
            finish: 'tool_calls',
            continues: prev ? { request_id: prev, gap_ms: 100, tool_names: [prevTool!], inferred: false } : null,
          }
        : {
            segs: [
              { state: 'queued', ms: 50 },
              { state: 'prefilling', ms: 1500, detail: `${prompt - 900} cached` },
              { state: 'thinking', ms: 300 },
              { state: 'writing', ms: 400 },
              { state: 'tool_call', ms: 1100, detail: tools[k] },
              { state: 'finishing', ms: 50, step: 'saving_state' },
            ],
            total: prompt,
            cached: prompt - 900,
            tps: 50 + k * 6,
            prefillTps: 400 + k * 15,
            progress: 'single_call',
            tool: { name: tools[k], argBytes: 300 + k * 90 },
            finish: 'tool_calls',
            continues: prev ? { request_id: prev, gap_ms: 100 + k * 30, tool_names: [prevTool!], inferred: false } : null,
          };
      prev = rid;
      prevTool = tools[Math.min(k, turns - 1)];
    }
    this.advanceAll(now);
    this.rebase();
  }

  /** Called once a tick by the server: a scripted request whose arrival time has come joins the registry. */
  admit(now = Date.now()): void {
    this.hidden = this.hidden.filter((r) => {
      if (r.startedAt > now) return true;
      this.reqs.set(r.row.request_id, r);
      return false;
    });
  }
}
