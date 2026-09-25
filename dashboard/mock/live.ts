// The mock's live registry (VIS-23): what `server/live.py` does, over the mock engine's own
// requests -- the tick simulator's request every ~15 s, the Playground's chats, and the "busy"
// scenario (one decoding, one prefilling, one queued, two done) for the e2e and the screenshots.
// Never in dist/.

import type { Live, LiveCounts, LivePhase, LiveRequest, LiveSample } from '../src/api/types.ts';
import type { LedgerRow } from './generate.ts';

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
}

const HISTORY = 300;
const KEEP_MS = 30_000;

export class MockLive {
  reqs = new Map<string, MockLiveRequest>();
  samples: LiveSample[] = [];
  served = 0;
  errors = 0;
  refused = 0;
  ends: number[] = [];
  lastPrefill: { at: number; tps: number } | null = null;
  private pending: number[] = [];
  private prev: { t: number; tokens: number; decoding: boolean } | null = null;
  private finishedTokens = 0;
  private credited = new Set<string>();

  constructor(
    private rng: () => number,
    served = 0,
  ) {
    this.served = served;
  }

  /** A request arrived: queued until `lock()`. */
  begin(row: LedgerRow, now = Date.now()): MockLiveRequest {
    const r: MockLiveRequest = { row, startedAt: now, lockAt: null, firstAt: null, endedAt: null, tokens: 0, blocks: 0, marks: [] };
    this.reqs.set(row.request_id, r);
    return r;
  }
  lock(r: MockLiveRequest, now = Date.now()): void {
    r.lockAt = now;
  }
  first(r: MockLiveRequest, now = Date.now()): void {
    r.firstAt = now;
    r.tokens = Math.max(1, r.tokens);
  }
  token(r: MockLiveRequest, n = 1): void {
    r.tokens += n;
    r.blocks = Math.max(1, Math.ceil((r.tokens - 1) / 4.3));
  }
  finish(r: MockLiveRequest, finish: string | null, status = 200, now = Date.now()): void {
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
  }

  private prefillTps(r: MockLiveRequest): number | null {
    if (r.firstAt == null || r.lockAt == null || r.row.cache_source === 'response') return null;
    const ms = r.firstAt - r.lockAt;
    const fwd = (r.row.prompt_tokens ?? 0) - (r.row.cached_tokens ?? 0);
    return fwd && ms > 0 ? Math.round((fwd / ms) * 1000 * 100) / 100 : null;
  }

  /** One sample a second, like the server's sampler. Also advances the scripted requests. */
  tick(now = Date.now()): void {
    for (const r of this.reqs.values()) {
      if (r.scripted && r.endedAt == null) this.advance(r, now);
    }
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

  private advance(r: MockLiveRequest, now: number): void {
    const s = r.scripted!;
    const since = now - r.startedAt;
    if (r.lockAt == null && since >= s.queueMs) r.lockAt = r.startedAt + s.queueMs;
    if (r.lockAt != null && r.firstAt == null && now - r.lockAt >= s.prefillMs) {
      r.firstAt = r.lockAt + s.prefillMs;
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

  private prune(now: number): void {
    for (const [id, r] of this.reqs) if (r.endedAt != null && now - r.endedAt > KEEP_MS) this.reqs.delete(id);
    this.ends = this.ends.filter((t) => now - t <= 60_000);
  }

  private rowOf(r: MockLiveRequest, now: number): LiveRequest {
    const row = r.row;
    const done = r.endedAt != null;
    const phase: LivePhase = done ? 'done' : r.lockAt == null ? 'queued' : r.firstAt == null ? 'prefill' : 'decode';
    const ran = r.lockAt != null && row.prompt_tokens != null;
    const end = done ? r.endedAt! : now;
    const decodeMs = r.firstAt != null ? end - r.firstAt : null;
    const decodeTps = decodeMs && r.tokens > 1 ? Math.round(((r.tokens - 1) / decodeMs) * 1000 * 100) / 100 : null;
    let nowTps: number | null = null;
    if (!done && r.marks.length >= 2) {
      const a = r.marks[0];
      const b = r.marks[r.marks.length - 1];
      nowTps = b.t > a.t ? Math.round(((b.n - a.n) / (b.t - a.t)) * 1000 * 100) / 100 : null;
    }
    const queueMs = r.lockAt != null ? r.lockAt - r.startedAt : null;
    const promptMs = r.firstAt != null && r.lockAt != null ? r.firstAt - r.lockAt : null;
    return {
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
  }

  snapshot(now = Date.now(), history = true): Live {
    this.prune(now);
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
    const out: Live = {
      contract_version: '1.0',
      generated_at: new Date(now).toISOString().replace(/\.\d{3}Z$/, 'Z'),
      interval_s: 1.0,
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
    if (history) out.history = [...this.samples];
    return out;
  }

  /** The busy scenario: one decoding, one prefilling, one queued, two done (a stop and an error),
   *  and four minutes of history (two decode runs with a gap between them, three prefills). */
  scenario(makeRow: (forced: Partial<LedgerRow>) => LedgerRow, now = Date.now()): void {
    this.samples = [];
    let tokens = this.finishedTokens;
    for (let i = 240; i >= 1; i--) {
      const t = now - i * 1000;
      const inRun = (i > 150 && i <= 235) || (i > 20 && i <= 95);
      const asleep = i > 95 && i <= 150 && i !== 120; // the sampler slept: a gap, with one lone prefill dot
      if (asleep) continue;
      const rate = inRun ? 38 + Math.sin(i / 9) * 4 + (this.rng() - 0.5) * 5 : null;
      if (rate != null) tokens += Math.round(rate);
      const prefill = i === 235 ? 1180.4 : i === 120 ? 2410.5 : i === 95 ? 640.2 : null;
      this.samples.push({ t: Math.round(t) / 1000, decode_tps: rate == null ? null : Math.round(rate * 100) / 100, prefill_tps: prefill, running: inRun ? 1 : 0, waiting: 0, tokens });
    }
    this.finishedTokens = tokens;
    this.prev = null;
    const add = (forced: Partial<LedgerRow>, script: MockLiveRequest['scripted'], startedAgo: number) => {
      const r = this.begin(makeRow(forced), now - startedAgo);
      r.scripted = script;
      this.advance(r, now);
      return r;
    };
    add({ client_id: 'k:3f9a1c0e2b7d', client_kind: 'open-webui', endpoint: 'chat', stream: true, thinking: true, prompt_tokens: 1959, cached_tokens: 1536, cache_source: 'prefix', max_tokens: 32768 }, { tps: 43, prefillMs: 2240, queueMs: 0.4, total: 4000 }, 8000);
    add({ client_id: 'anon', client_kind: 'curl', endpoint: 'chat', stream: true, thinking: false, prompt_tokens: 8192, cached_tokens: 0, cache_source: 'none', max_tokens: 4096 }, { tps: 38, prefillMs: 9000, queueMs: 900, total: 3000 }, 2200);
    add({ client_id: 'k:0b7e55c3d1f2', client_kind: 'dashboard', endpoint: 'completions', stream: false, thinking: false, prompt_tokens: 640, cached_tokens: 0, cache_source: 'none', max_tokens: 512 }, { tps: 41, prefillMs: 700, queueMs: 20000, total: 2000 }, 800);
    // two finished ones, with final numbers
    const okRow = makeRow({ client_id: 'k:0b7e55c3d1f2', client_kind: 'dashboard', endpoint: 'chat', stream: true, thinking: true, prompt_tokens: 412, cached_tokens: 0, cache_source: 'none', max_tokens: 32768 });
    const ok = this.begin(okRow, now - 22_000);
    ok.lockAt = ok.startedAt + 0.3;
    ok.firstAt = ok.lockAt + 705.1;
    this.token(ok, 388);
    this.finish(ok, 'stop', 200, ok.firstAt + 8901.2);
    const errRow = makeRow({ client_id: 'anon', client_kind: 'curl', endpoint: 'chat', stream: true, thinking: false, prompt_tokens: 64, cached_tokens: 0, cache_source: 'none', max_tokens: 200 });
    const err = this.begin(errRow, now - 25_600);
    err.lockAt = err.startedAt + 0.2;
    err.firstAt = err.lockAt + 120.4;
    this.token(err, 5);
    this.finish(err, 'error', 200, err.firstAt + 210);
    this.prev = null; // the next tick has no delta to a time before the scenario existed
  }
}
