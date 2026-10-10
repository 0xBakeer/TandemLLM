// The parallel view in the mock (`GET /v1/dashboard/parallel`): a simulator that keeps about
// `streams` requests in flight over six backends -- the engine (one at a time, with a prefill),
// a vLLM and an SGLang (frames checked as tokens), DeepSeek (text frames checked, tool-call
// frames not: its call phase has no live count), an OpenAI-compatible one whose frames were never
// checked (no live count at all) and the fake backend (many, at different speeds). It moves with
// the wall clock, so a page that polls once a second sees lanes arrive, think, call tools, write
// and finish. `POST /__mock/parallel {streams, reset, engine_slots}` sets the load (0: idle) and how
// many requests the engine serves at once (its sequence slots, server --parallel N; 1: one at a time).

type J = Record<string, unknown>;

interface Sim {
  id: string;
  backend: string;
  kind: string;
  model: string;
  session: string;
  client: string;
  t0: number; // arrival, s
  ttft: number; // s
  think: number; // thinking tokens
  out: number; // answer tokens
  tool: number; // tool-call tokens (0: none)
  tps: number;
  prompt: number;
  cached: number;
  price: number | null; // output micro-dollars per token
  inPrice: number | null;
  frames: 'all' | 'text' | 'none' | 'engine';
  slot?: number; // the engine's sequence slot, with engine_slots > 1
}

const BACKENDS: { id: string; kind: string; models: string[]; tps: [number, number]; ttft: [number, number]; share: number; frames: Sim['frames']; price: number | null; inPrice: number | null }[] = [
  { id: 'engine', kind: 'tandemllm', models: ['TandemLLM'], tps: [46, 58], ttft: [0.6, 2.4], share: 0, frames: 'engine', price: null, inPrice: null },
  { id: 'vllm1', kind: 'vllm', models: ['Qwen3.5-4B'], tps: [64, 90], ttft: [0.12, 0.5], share: 3, frames: 'all', price: null, inPrice: null },
  { id: 'sglang1', kind: 'sglang', models: ['Qwen3.5-2B'], tps: [80, 110], ttft: [0.1, 0.4], share: 2, frames: 'all', price: null, inPrice: null },
  { id: 'ds', kind: 'deepseek', models: ['deepseek-flash', 'deepseek-v4-pro'], tps: [38, 62], ttft: [0.5, 1.6], share: 4, frames: 'text', price: 0.6, inPrice: 0.15 },
  { id: 'oa', kind: 'generic', models: ['gpt-oss-120b'], tps: [30, 45], ttft: [0.4, 1.2], share: 1, frames: 'none', price: 0.3, inPrice: 0.1 },
  { id: 'fake', kind: 'generic', models: ['fake-a'], tps: [25, 120], ttft: [0.05, 1.2], share: 5, frames: 'all', price: null, inPrice: null },
];

export class MockParallel {
  streams = 14;
  /** the engine's sequence slots (server --parallel N); 1 = one request at a time */
  engineSlots = 1;
  private joint: Record<string, number> = {};
  private live: Sim[] = [];
  private done: { at: number; lane: J; stat: J }[] = [];
  private series: { t: number; by: Record<string, number>; in_flight: number }[] = [];
  private last: Map<string, number> = new Map();
  private lastTick = 0;
  private seq = 0;
  private engineBusyUntil = 0;
  private rr = 0; // the first spawns go round the relayed backends once, so each one shows
  private rnd: () => number;

  constructor(seed = 7) {
    let s = seed >>> 0 || 1;
    this.rnd = () => {
      s = (s * 1664525 + 1013904223) >>> 0;
      return s / 2 ** 32;
    };
  }

  reset(streams?: number): void {
    if (streams != null) this.streams = streams;
    this.live = [];
    this.done = [];
    this.series = [];
    this.last.clear();
    this.lastTick = 0;
    this.engineBusyUntil = 0;
    this.rr = 0;
    this.joint = {};
  }

  private engineLive(): Sim[] {
    return this.live.filter((l) => l.backend === 'engine');
  }

  private pick<T>(xs: T[]): T {
    return xs[Math.floor(this.rnd() * xs.length)];
  }
  private between([a, b]: [number, number]): number {
    return a + (b - a) * this.rnd();
  }

  private spawn(now: number): void {
    const engineFree = this.engineSlots > 1 ? this.engineLive().length < this.engineSlots : now >= this.engineBusyUntil && !this.live.some((l) => l.backend === 'engine');
    const pool = BACKENDS.filter((b) => b.share > 0);
    const total = pool.reduce((a, b) => a + b.share, 0);
    let b = pool[0];
    if (this.rr < pool.length) b = pool[this.rr++];
    else if (engineFree && this.rnd() < (this.engineSlots > 1 ? 0.5 : 0.25)) b = BACKENDS[0];
    else {
      let r = this.rnd() * total;
      for (const x of pool) {
        r -= x.share;
        if (r <= 0) {
          b = x;
          break;
        }
      }
    }
    let slot: number | undefined;
    if (b.id === 'engine' && this.engineSlots > 1) {
      const used = new Set(this.engineLive().map((l) => l.slot));
      slot = [...Array(this.engineSlots).keys()].find((i) => !used.has(i));
    } else if (b.id === 'engine') this.engineBusyUntil = now + 1e9;
    const think = this.rnd() < 0.55 ? Math.round(80 + this.rnd() * 900) : 0;
    const prompt = Math.round(b.id === 'engine' ? 6000 + this.rnd() * 30000 : 800 + this.rnd() * 14000);
    this.live.push({
      id: `mock-${++this.seq}`,
      backend: b.id,
      kind: b.kind,
      model: this.pick(b.models),
      session: `agent-${String(1 + Math.floor(this.rnd() * 24)).padStart(2, '0')}`,
      client: this.pick(['opencode', 'other', 'hermes', 'openai-sdk']),
      t0: now,
      ttft: this.between(b.ttft) + (b.id === 'engine' ? prompt / 9000 : 0),
      think,
      out: Math.round(40 + this.rnd() * 520),
      tool: this.rnd() < 0.4 ? Math.round(30 + this.rnd() * 120) : 0,
      tps: this.between(b.tps),
      prompt,
      cached: Math.round(prompt * (this.rnd() < 0.7 ? 0.3 + this.rnd() * 0.65 : 0)),
      price: b.price,
      inPrice: b.inPrice,
      frames: b.frames,
      ...(slot != null ? { slot } : {}),
    });
  }

  private tokensAt(s: Sim, now: number): number {
    const dt = now - s.t0 - s.ttft;
    if (dt < 0) return 0;
    return Math.min(s.think + s.out + s.tool, 1 + Math.floor(dt * s.tps));
  }

  private lane(s: Sim, now: number, ended: boolean): J {
    const n = this.tokensAt(s, now);
    const total = s.think + s.out + s.tool;
    const started = now - s.t0 >= s.ttft;
    let state = 'waiting';
    if (s.backend === 'engine' && !started) state = 'prefilling';
    else if (started) state = n <= s.think ? 'thinking' : n > s.think + s.out && s.tool ? 'tool_call' : 'writing';
    if (ended) state = 'done';
    const inTool = state === 'tool_call';
    const counted = s.frames === 'engine' || s.frames === 'all' || (s.frames === 'text' && !inTool && !(ended && s.tool));
    const live = !ended;
    const tokens = ended ? total : started ? (counted ? n : null) : 0;
    const dec = started ? Math.max(0.001, now - s.t0 - s.ttft) : 0;
    const avg = tokens != null && tokens > 1 && dec > 0 ? (Math.min(n, total) - 1) / Math.min(dec, total / s.tps) : null;
    const jitter = 1 + (this.rnd() - 0.5) * 0.08;
    const relayed = s.backend !== 'engine';
    const pct = !started ? Math.min(99, (100 * (now - s.t0)) / s.ttft) : 100;
    return {
      id: s.id,
      backend: s.backend,
      kind: s.kind,
      model: s.model,
      client: { kind: s.client, user: null, session: s.session },
      state,
      label: s.backend === 'engine' && live ? (state === 'prefilling' ? `Prefilling ${Math.round((s.prompt * pct) / 100).toLocaleString('en-US')} of ${s.prompt.toLocaleString('en-US')} (${Math.round(pct)} %)` : state === 'tool_call' ? 'Calling tool edit' : state === 'thinking' ? 'Thinking' : 'Writing') : null,
      tool: inTool ? this.pick(['read_file', 'run_tests', 'edit']) : null,
      elapsed_ms: Math.round((now - s.t0) * 1000),
      ttft_ms: started ? Math.round(s.ttft * 10000) / 10 : null,
      waiting_ms: started ? null : Math.round((now - s.t0) * 1000),
      tokens,
      tokens_source: ended ? (relayed ? 'usage' : 'engine') : tokens == null ? null : relayed ? 'frames' : 'engine',
      frames: relayed ? (s.frames === 'text' ? Math.min(n, s.think + s.out) + (n > s.think + s.out ? Math.ceil((n - s.think - s.out) / 6) : 0) : n) : null,
      tool_frames: relayed && n > s.think + s.out ? Math.ceil((n - s.think - s.out) / (s.frames === 'text' ? 6 : 1)) : 0,
      tool_chars: relayed && n > s.think + s.out ? (n - s.think - s.out) * 3 : 0,
      thinking_tokens: tokens == null ? null : Math.min(s.think, ended ? s.think : n),
      decode_now: live && started && tokens != null ? Math.round(s.tps * jitter * 10) / 10 : null,
      decode_avg: avg == null ? null : Math.round(avg * 10) / 10,
      prefill:
        s.backend === 'engine'
          ? { done: Math.round((s.prompt * pct) / 100), total: s.prompt, pct: Math.round(pct * 10) / 10, cached: s.cached, tps: !started ? 9000 + Math.round(this.rnd() * 600) : null, tps_avg: 9100, eta_ms: null }
          : null,
      prompt_tokens: relayed && live ? null : s.prompt,
      cached_tokens: relayed && live ? null : s.frames === 'none' && ended ? null : s.cached,
      cache_hit: relayed && live ? null : s.frames === 'none' && ended ? null : Math.round((s.cached / s.prompt) * 1e4) / 1e4,
      cost_micros: s.price == null ? null : ended ? Math.round(total * s.price + (s.prompt - s.cached) * (s.inPrice ?? 0)) : tokens ? Math.round(tokens * s.price) : null,
      cost_partial: s.price != null && !ended && !!tokens,
      max_tokens: 4096,
      ...(s.slot != null ? { slot: s.slot } : {}),
    };
  }

  tick(nowMs: number): void {
    const now = nowMs / 1000;
    // finish the ones that are through
    const still: Sim[] = [];
    for (const s of this.live) {
      const total = s.think + s.out + s.tool;
      if (this.tokensAt(s, now) >= total && now - s.t0 > s.ttft + total / s.tps) {
        const lane = this.lane(s, now, true);
        lane.finish = s.tool ? 'tool_calls' : 'stop';
        lane.status = 200;
        this.done.push({ at: now, lane, stat: { backend: s.backend, kind: s.kind, ttft: s.ttft * 1000, prompt: s.prompt, cached: s.frames === 'none' ? null : s.cached, cost: lane.cost_micros, tokens: total } });
        if (s.backend === 'engine') this.engineBusyUntil = now + 2;
      } else still.push(s);
    }
    this.live = still;
    while (this.live.length < this.streams) this.spawn(now - this.rnd() * 0.4);
    this.done = this.done.filter((d) => now - d.at <= 300);
    // one series sample a second
    if (!this.lastTick || now - this.lastTick >= 1) {
      const by: Record<string, number> = {};
      const dt = this.lastTick ? now - this.lastTick : 1;
      for (const s of this.live) {
        const lane = this.lane(s, now, false);
        if (lane.tokens == null) continue;
        const n = lane.tokens as number;
        const prev = this.last.get(s.id) ?? 0;
        by[s.backend] = (by[s.backend] ?? 0) + Math.max(0, n - prev) / dt;
        this.last.set(s.id, n);
      }
      for (const k of Object.keys(by)) by[k] = Math.round(by[k] * 10) / 10;
      this.series.push({ t: Math.round(now * 1000) / 1000, by, in_flight: this.live.length });
      if (this.series.length > 300) this.series.shift();
      this.lastTick = now;
    }
  }

  /** VIS-76: `POST /__mock/parallel {fixture: "stall"}` answers the 0.6.x shape of 2026-10-10 09:52
   *  instead of the simulator: four engine slots, slot 0 prefilling 18,432 of 44,558 (one 8,192-row
   *  window a mark), slots 1-3 decoding with no token over the last 2 s (decode_now 0.0), a fifth
   *  request held for memory (SRV-305), and a series whose decode drops to 0 between bursts. */
  fixture: 'stall' | null = null;

  private stallDoc(nowMs: number, since: number | null): J {
    const now = nowMs / 1000;
    const lane = (i: number, state: string, extra: J): J => ({
      id: `chatcmpl-sta11${i}000000000000000000`, backend: 'engine', kind: 'tandemllm', model: 'TandemLLM',
      client: { kind: 'open-webui', user: null, session: `c:s${i}` }, state, label: null, tool: null,
      elapsed_ms: 60_000 - i * 9000, ttft_ms: 900, waiting_ms: null, tokens: 400 + i * 150, tokens_source: 'engine', frames: null,
      thinking_tokens: 300, decode_now: 0, decode_avg: 18.5, prefill: null, prompt_tokens: 20_000, cached_tokens: 18_000, cache_hit: 0.9,
      cost_micros: null, cost_partial: false, max_tokens: 32_768, relay_id: null, via: null, slot: i, ...extra,
    });
    const lanes = [
      lane(0, 'prefilling', { label: 'Prefilling 18,432 of 44,558 (41 %)', elapsed_ms: 19_000, ttft_ms: null, waiting_ms: 19_000, tokens: 0, thinking_tokens: 0, decode_now: null, decode_avg: null, prefill: { done: 18_432, total: 44_558, pct: 41.4, cached: 10_240, tps: 774.8, tps_avg: 774.8, eta_ms: 33_700 }, prompt_tokens: 34_487, cached_tokens: 10_240 }),
      lane(1, 'thinking', { label: 'Thinking' }),
      lane(2, 'tool_call', { label: 'Calling tool write_file', tool: 'write_file' }),
      lane(3, 'writing', { label: 'Writing' }),
      { ...lane(4, 'waiting_memory', { label: 'waiting for memory', ttft_ms: null, waiting_ms: 6000, tokens: 0, thinking_tokens: 0, decode_now: null, decode_avg: null, elapsed_ms: 6000 }), slot: undefined },
    ];
    const t0 = Math.floor(now) - 120;
    const series = Array.from({ length: 121 }, (_, k) => {
      const ph = k % 24; // a burst of decode between two prefill windows, then nothing for ~11 s
      return { t: t0 + k, by: ph < 12 ? { engine: 40 + ((k * 37) % 40) } : {}, in_flight: 5 };
    });
    return {
      contract_version: '1.0', at: now, windows: { now_s: 2, done_keep_s: 4, stats_s: 300, cost_min_s: 60 }, lanes, recent: [],
      backends: [{ id: 'engine', kind: 'tandemllm', models: ['TandemLLM'], in_flight: 5, decoding: 3, waiting: 2, decode_tps: 0, decode_unreported: 0, ttft_p50: 1580, ttft_p90: 9920, cache_hit: 0.565, cache_reported: 10, finished: 9, errors: 0, cost_per_min: null, cost_window: null, tokens_window: 18_000, frames: null, engine_view: false }],
      totals: { in_flight: 5, decoding: 3, waiting: 2, waiting_memory: 1, decode_tps: 0, decode_reported: 3, decode_unreported: 0, prefill_tps: 774.8, ttft_p50: 1580, ttft_p90: 9920, ttft_n: 9, cache_hit: 0.565, cache_reported: 10, prompt_tokens_window: 18_100, cached_tokens_window: 10_200, cost_per_min: null, cost_window: null, cost_in_flight: null, finished: 9, errors: 0, refused: 0, tokens_window: 18_000, backends: 1 },
      series: series.filter((x) => since == null || x.t > since),
      build_us: { last: 260, max: 900 },
      engine_slots: { n: 4, running: 4, decoding: 3, slots: [0, 1, 2, 3].map((i) => ({ index: i, window: 262_144, busy: true, phase: i ? 'decode' : '', request: null })), rounds: 10_078, joint_rounds: 4231, joint_hist: { 2: 2312, 3: 1338, 4: 581 }, timeouts: 456, admission: { floor_gib: 15, in_flight: 4, reserved_gib: 7.2, waits: 1, refused: 0 } },
    };
  }

  snapshot(nowMs: number, since: number | null): J {
    if (this.fixture === 'stall') return this.stallDoc(nowMs, since);
    this.tick(nowMs);
    const now = nowMs / 1000;
    const lanes = this.live.map((s) => this.lane(s, now, false)).sort((a, b) => (b.elapsed_ms as number) - (a.elapsed_ms as number));
    const recent = this.done.filter((d) => now - d.at <= 8).map((d) => ({ ...d.lane, ended_ms: Math.round((now - d.at) * 1000) }));
    const stats = this.done.map((d) => d.stat);
    const lastMin = this.done.filter((d) => now - d.at <= 60).map((d) => d.stat);
    const pctl = (xs: number[], q: number) => {
      const v = [...xs].sort((a, b) => a - b);
      return v.length ? Math.round(v[Math.min(v.length - 1, Math.round(q * (v.length - 1)))] * 10) / 10 : null;
    };
    const ids = [...new Set([...lanes.map((l) => l.backend as string), ...stats.map((s) => s.backend as string)])];
    const backends = BACKENDS.filter((b) => ids.includes(b.id)).map((b) => {
      const mine = lanes.filter((l) => l.backend === b.id);
      const st = stats.filter((s) => s.backend === b.id);
      const dec = mine.filter((l) => !['waiting', 'prefilling', 'queued'].includes(l.state as string));
      const rates = dec.map((l) => l.decode_now).filter((x): x is number => x != null);
      const rep = st.filter((s) => s.cached != null);
      const costs = st.map((s) => s.cost).filter((x): x is number => x != null);
      const lm = lastMin.filter((s) => s.backend === b.id).map((s) => s.cost).filter((x): x is number => x != null);
      return {
        id: b.id,
        kind: b.kind,
        models: [...new Set(mine.map((l) => l.model as string))],
        in_flight: mine.length,
        decoding: dec.length,
        waiting: mine.length - dec.length,
        decode_tps: rates.length ? Math.round(rates.reduce((a, c) => a + c, 0) * 10) / 10 : null,
        decode_unreported: dec.length - rates.length,
        ttft_p50: pctl(st.map((s) => s.ttft as number), 0.5),
        ttft_p90: pctl(st.map((s) => s.ttft as number), 0.9),
        cache_hit: rep.length ? Math.round((rep.reduce((a, s) => a + (s.cached as number), 0) / rep.reduce((a, s) => a + (s.prompt as number), 0)) * 1e4) / 1e4 : null,
        cache_reported: rep.length,
        finished: st.length,
        errors: 0,
        cost_per_min: lm.length ? lm.reduce((a, c) => a + c, 0) : null,
        cost_window: costs.length ? costs.reduce((a, c) => a + c, 0) : null,
        tokens_window: st.reduce((a, s) => a + (s.tokens as number), 0),
        engine_view: false,
        frames: b.id === 'engine' ? null : b.frames === 'none' ? { verified: false, tools_verified: false, models: 1, checked: 6, passed: 1 } : { verified: true, tools_verified: b.frames === 'all', models: b.models.length, checked: 8, passed: b.frames === 'all' ? 8 : 5 },
      };
    });
    const dec = lanes.filter((l) => !['waiting', 'prefilling', 'queued'].includes(l.state as string));
    const rates = dec.map((l) => l.decode_now).filter((x): x is number => x != null);
    const rep = stats.filter((s) => s.cached != null);
    const costs = stats.map((s) => s.cost).filter((x): x is number => x != null);
    const lm = lastMin.map((s) => s.cost).filter((x): x is number => x != null);
    const partial = lanes.map((l) => l.cost_micros).filter((x): x is number => x != null);
    const pre = lanes.filter((l) => l.state === 'prefilling').map((l) => (l.prefill as J)?.tps as number | null).filter((x): x is number => x != null);
    return {
      contract_version: '1.0',
      at: Math.round(now * 1000) / 1000,
      windows: { now_s: 2, done_keep_s: 8, stats_s: 300, cost_min_s: 60 },
      lanes,
      recent,
      backends,
      totals: {
        in_flight: lanes.length,
        decoding: dec.length,
        waiting: lanes.length - dec.length,
        decode_tps: rates.length ? Math.round(rates.reduce((a, c) => a + c, 0) * 10) / 10 : null,
        decode_reported: rates.length,
        decode_unreported: dec.length - rates.length,
        prefill_tps: pre.length ? pre.reduce((a, c) => a + c, 0) : null,
        ttft_p50: pctl(stats.map((s) => s.ttft as number), 0.5),
        ttft_p90: pctl(stats.map((s) => s.ttft as number), 0.9),
        ttft_n: stats.length,
        cache_hit: rep.length ? Math.round((rep.reduce((a, s) => a + (s.cached as number), 0) / rep.reduce((a, s) => a + (s.prompt as number), 0)) * 1e4) / 1e4 : null,
        cache_reported: rep.length,
        prompt_tokens_window: rep.length ? rep.reduce((a, s) => a + (s.prompt as number), 0) : null,
        cached_tokens_window: rep.length ? rep.reduce((a, s) => a + (s.cached as number), 0) : null,
        cost_per_min: lm.length ? lm.reduce((a, c) => a + c, 0) : null,
        cost_window: costs.length ? costs.reduce((a, c) => a + c, 0) : null,
        cost_in_flight: partial.length ? partial.reduce((a, c) => a + c, 0) : null,
        finished: stats.length,
        errors: 0,
        refused: 0,
        tokens_window: stats.reduce((a, s) => a + (s.tokens as number), 0),
        backends: backends.length,
      },
      series: this.series.filter((s) => since == null || s.t > since),
      build_us: { last: 240 + Math.round(this.rnd() * 120), max: 910 },
      ...(this.engineSlots > 1 ? { engine_slots: this.engineSlotsDoc(lanes) } : {}),
    };
  }

  /** What server/app.py `_par_brief` answers: the slots, their windows, what they do, and the
   *  verify passes they shared (counted up while two or more engine lanes decode). */
  private engineSlotsDoc(lanes: J[]): J {
    const eng = lanes.filter((l) => l.backend === 'engine');
    const decoding = eng.filter((l) => ['thinking', 'writing', 'tool_call'].includes(l.state as string));
    if (decoding.length > 1) {
      const k = String(decoding.length);
      this.joint[k] = (this.joint[k] ?? 0) + Math.round(8 + this.rnd() * 4);
    }
    const slots = [...Array(this.engineSlots).keys()].map((i) => {
      const l = eng.find((x) => x.slot === i);
      const dec = l != null && decoding.includes(l);
      return { index: i, window: i === 0 ? 262144 : 65536, busy: l != null, phase: dec ? 'decode' : '', request: i && l ? (l.id as string) : null };
    });
    const joint = Object.values(this.joint).reduce((a, c) => a + c, 0);
    return { n: this.engineSlots, running: eng.length, decoding: decoding.length, slots, rounds: joint * 2, joint_rounds: joint, joint_hist: { ...this.joint }, timeouts: 0, admission: { floor_gib: 15, in_flight: eng.length, reserved_gib: eng.length * 1.4, waits: 0, refused: 0, checks: 0, admitted: 0, min_after: null } };
  }
}
