// The mock engine: a Connect middleware for Vite dev/preview (`npm run dev:mock`,
// `npm run preview:mock`) answering every contract endpoint from the seeded year, a live SSE log,
// moving /metrics, a streaming /v1/chat/completions (mock/chat.ts) with the SRV-27 finish chunk, and the failure
// switches. Never part of dist/ (dynamic import in vite.config.ts; tests/build.test.ts checks).
//
// Failure switches: open the app as /dashboard/?mock=<mode> (the middleware sets a cookie), or
// POST /__mock/mode {"mode": "..."}. Modes: ok | 401 | expire | 500 | empty | slow | offline |
// drop[:seconds] | busy | nospec.

import type { IncomingMessage, ServerResponse } from 'node:http';
import { randomBytes } from 'node:crypto';
import type { LogLevel, LogLine, SystemInfo } from '../src/api/types.ts';
import { createChatHandler } from './chat.ts';
import { dayKey } from '../src/lib/time.ts';
import { generateYear, makeRow, rng, type LedgerRow, ENGINE_VERSION, CODE_SHA } from './generate.ts';
import { requests as aggRequests, summary as aggSummary, usage as aggUsage } from './aggregate.ts';
import { LogRing } from './logs.ts';
import { applyRow, initialState, renderMetrics, type LiveEngineState } from './metrics.ts';
import { MockLive, type MockLiveRequest } from './live.ts';

type Next = (err?: unknown) => void;
type Middleware = (req: IncomingMessage, res: ServerResponse, next: Next) => void;

const TZ_DEFAULT = 'Europe/Berlin';
const TOKEN = process.env.QSE_MOCK_TOKEN ?? 'mock';
const METRICS_TOKEN = process.env.QSE_MOCK_METRICS_TOKEN ?? 'mock-metrics';

export interface MockOptions {
  seed?: number;
  /** disable the live simulator (tests that want a still engine) */
  live?: boolean;
  tz?: string;
}

interface Session {
  expires: number;
}

export function createMockMiddleware(opts: MockOptions = {}): Middleware {
  const tz = opts.tz ?? TZ_DEFAULT;
  const seed = opts.seed ?? 42;
  const now0 = Date.now();
  const full = generateYear({ seed, tz, now: now0 });
  const emptyLedger = generateYear({ seed: seed + 1, tz, now: now0, days: 3 });
  const liveRng = rng(seed + 7);

  let mode = 'ok';
  let dropEveryMs = 30_000;
  const sessions = new Map<string, Session>();
  const startedAt = now0 - 6 * 3600 * 1000 - 1677 * 1000;
  const state: LiveEngineState = initialState(full.rows, startedAt);
  const logs = new LogRing(10_000);
  logs.seed(full.rows, rng(seed + 3));
  // VIS-23: the live registry the /v1/dashboard/live endpoint reads; the simulator and the chat feed it
  const live = new MockLive(rng(seed + 11), full.rows.filter((r) => r.ts_ms >= startedAt && r.finish_reason !== 'refused').length);
  let draining = false;
  const memHistory: number[] = [];
  void memHistory;

  const rowsFor = () => (mode === 'empty' ? emptyLedger : full);

  // ---- the live simulator: a request every ~10-25 s, flavour lines in between --------------
  let inflight: { row: LedgerRow; startedAt: number; streamed: number; done: number; live: MockLiveRequest } | null = null;
  let simulate = opts.live !== false; // `POST /__mock/live {simulator: false}` stops new simulated requests (the scenario tests)
  const tick = () => {
    const now = Date.now();
    if (inflight) {
      const el = now - inflight.startedAt;
      const total = Math.min(inflight.row.total_ms ?? 1000, 20_000);
      const ttft = Math.min(inflight.row.ttft_ms ?? 300, total * 0.3);
      const frac = Math.min(1, Math.max(0, el - ttft) / Math.max(1, total - ttft));
      const target = Math.floor((inflight.row.completion_tokens ?? 0) * frac);
      state.generationTokens += target - inflight.streamed;
      inflight.streamed = target;
      if (el >= ttft && inflight.live.firstAt == null) live.first(inflight.live, inflight.startedAt + ttft);
      if (inflight.live.firstAt != null && target > inflight.live.tokens) live.token(inflight.live, target - inflight.live.tokens);
      if (frac >= 1) {
        state.running = 0;
        state.generationTokens -= inflight.streamed;
        const row = inflight.row;
        row.ts_ms = now;
        row.id = full.rows.length ? full.rows[full.rows.length - 1].id + 1 : 1;
        full.rows.push(row);
        applyRow(state, row);
        logs.pushRequest(row);
        live.finish(inflight.live, row.finish_reason, row.status, now);
        inflight = null;
      }
    } else if (simulate && liveRng() < 1 / 14) {
      const row = makeRow(liveRng, now);
      if (row.finish_reason === 'refused') {
        row.ts_ms = now;
        row.id = full.rows[full.rows.length - 1].id + 1;
        full.rows.push(row);
        applyRow(state, row);
        logs.pushRequest(row);
        const lr = live.begin(row, now);
        live.finish(lr, 'refused', row.status, now);
      } else {
        const lr = live.begin(row, now);
        live.lock(lr, now + (row.queue_ms ?? 0));
        inflight = { row, startedAt: now, streamed: 0, done: 0, live: lr };
        state.running = 1;
        logs.push('debug', 'http', `127.0.0.1 - "POST /v1/${row.endpoint === 'chat' ? 'chat/completions' : 'completions'} HTTP/1.1" 200`, null);
        if (row.cache_source && row.cache_source !== 'none') {
          logs.push('info', 'cache', `[cache] ${row.cache_source} hit reused=${row.cached_tokens} forwarded=${(row.prompt_tokens ?? 0) - (row.cached_tokens ?? 0)}`, row.request_id);
        }
      }
    }
    if (liveRng() < 0.45) logs.flavour(liveRng);
    live.tick(now);
    // memory drift: allocated breathes with requests, reserved only climbs (fragmentation)
    state.gpuAllocated = 61.2e9 + (inflight ? 1.6e9 : 0) + (liveRng() - 0.5) * 2e8;
    state.gpuReserved = Math.max(state.gpuReserved, state.gpuAllocated + 2.4e9);
    state.unifiedFree = 42.1e9 - (state.gpuReserved - 63.9e9) + (liveRng() - 0.5) * 3e8;
    state.rss = 4.4e9 + (liveRng() - 0.5) * 5e7;
  };
  if (opts.live !== false) {
    const t = setInterval(tick, 1000);
    if (typeof t === 'object' && 'unref' in t) t.unref();
  }

  // ---- helpers ----------------------------------------------------------------------------
  const json = (res: ServerResponse, code: number, body: unknown, headers: Record<string, string> = {}) => {
    const s = JSON.stringify(body);
    res.writeHead(code, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store', ...headers });
    res.end(s);
  };
  const error = (res: ServerResponse, code: number, type: string, message: string, headers: Record<string, string> = {}) =>
    json(res, code, { error: { type, message } }, headers);

  const cookies = (req: IncomingMessage): Record<string, string> => {
    const out: Record<string, string> = {};
    for (const part of (req.headers.cookie ?? '').split(';')) {
      const i = part.indexOf('=');
      if (i > 0) out[part.slice(0, i).trim()] = decodeURIComponent(part.slice(i + 1).trim());
    }
    return out;
  };

  const authed = (req: IncomingMessage): boolean => {
    const auth = req.headers.authorization ?? '';
    if (auth === `Bearer ${TOKEN}`) return true;
    const c = cookies(req).qse_dash;
    if (!c) return false;
    const s = sessions.get(c);
    if (!s) return false;
    if (s.expires < Date.now()) {
      sessions.delete(c);
      return false;
    }
    return true;
  };

  const readBody = (req: IncomingMessage): Promise<string> =>
    new Promise((resolve) => {
      let data = '';
      req.on('data', (c) => (data += c));
      req.on('end', () => resolve(data));
    });

  const delay = (ms: number) => new Promise((r) => setTimeout(r, ms));

  const liveStatus = (): SystemInfo['engine']['status'] => (draining ? 'draining' : state.waiting > 0 || state.running > 0 ? 'busy' : 'ok');

  const sysSnapshot = (): SystemInfo => {
    const now = Date.now();
    const rows = rowsFor().rows;
    return {
      contract_version: '1.0',
      generated_at: new Date(now).toISOString().replace(/\.\d{3}Z$/, 'Z'),
      engine: {
        version: ENGINE_VERSION,
        git_sha: CODE_SHA,
        code_sha256: '4c1f0e9a7b2d5e8f3a6c9d0b1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f',
        started_at: new Date(startedAt).toISOString().replace(/\.\d{3}Z$/, 'Z'),
        uptime_s: Math.round((now - startedAt) / 100) / 10,
        pid: 123456,
        model: 'qwen38-spark-engine',
        max_len: 262144,
        drafter: 'LengthRouter',
        tree: true,
        reasoning_format: 'tags',
        reasoning_effort: 'medium',
        status: liveStatus(),
      },
      flags: {
        args: {
          model: '~/qwen3.8-27b',
          served_model: 'qwen38-spark-engine',
          host: '0.0.0.0',
          port: 8000,
          max_len: 262144,
          default_max_tokens: 32768,
          reasoning_format: 'tags',
          reasoning_effort: 'medium',
          rep_penalty: 1.0,
          presence_penalty: 0.0,
          frequency_penalty: 0.0,
          temperature: 0.0,
          top_p: 1.0,
          top_k: 0,
          no_repeat_ngram: 0,
          pattern_stop: '',
          cache_budget_gb: 8.0,
          session_cache: true,
          prefix_cache: true,
          prefix_chunk: 512,
          response_cache: true,
          request_timeout: 3600.0,
          max_queue: 8,
          queue_timeout: 240.0,
          usage_ledger: '~/.qwen38-spark-engine/usage/ledger.sqlite3',
          usage_retention_days: 400,
          trust_loopback: true,
          verbose: false,
        },
        env: {
          QWEN38_DEEP: '32',
          QWEN38_DEEP_AFTER: '2',
          QWEN38_TREE_NODES: '24',
          QWEN38_TREE_NODES_NARROW: '16',
          QWEN38_TREE_WIDE_AFTER: '32',
          QWEN38_VERIFY_ROWS: '32',
          QWEN38_DF2_TREE_MODE: 'nodes',
          QWEN38_TREE_ALIAS_STATE: '0',
          QWEN38_DRAFT_TEMP: '0.3',
          QWEN38_GDNV_ONE_WARP: '1',
          QWEN38_FUSED_GDNPREFILL: '1',
          QWEN38_DECODE_ATTN: '1',
          QWEN38_DRAFT_NVFP4: '1',
          QWEN38_NVFP4_PREFILL_V2: '1',
          QWEN38_NVFP4_SKINNY: '1',
          QWEN38_SKINNY_TILES: '~/qwen38-spark-engine/ops/skinny-tiles.json',
          QWEN38_SKINNY_TILES_WIDE: '1',
          QWEN38_FUSED_COMMIT: '1',
          QWEN38_FUSED_GDNVERIFY: '1',
          QWEN38_DRAFT_FC_NVFP4: '1',
          QWEN38_FUSED_ADDNORM: '1',
          QWEN38_TREE_HOST_DEPTH: '1',
          QSE_USAGE_LEDGER: '~/.qwen38-spark-engine/usage/ledger.sqlite3',
          QSE_ADMIN_TOKEN: '<redacted>',
          QSE_METRICS_TOKEN: '<redacted>',
        },
      },
      memory: {
        gpu_allocated_bytes: Math.round(state.gpuAllocated),
        gpu_reserved_bytes: Math.round(state.gpuReserved),
        gpu_max_allocated_bytes: Math.round(Math.max(state.gpuAllocated, 64.1e9)),
        unified_total_bytes: 128_000_000_000,
        unified_available_bytes: Math.round(state.unifiedFree),
        process_rss_bytes: Math.round(state.rss),
      },
      gpu:
        mode === 'nogpu'
          ? { name: null, temperature_c: null, power_w: null, sm_clock_mhz: null, utilization: null, source: null, sampled_at: null }
          : {
              name: 'NVIDIA GB10',
              temperature_c: Math.round((inflight ? 58 : 41) + (liveRng() - 0.5) * 3),
              power_w: Math.round(((inflight ? 96 : 27.4) + (liveRng() - 0.5) * 4) * 10) / 10,
              sm_clock_mhz: inflight ? 2405 : 1980,
              utilization: inflight ? Math.round((0.86 + liveRng() * 0.1) * 100) / 100 : 0.0,
              source: 'nvidia-smi',
              sampled_at: new Date(now - 1200).toISOString().replace(/\.\d{3}Z$/, 'Z'),
            },
      caches: {
        model: 'qwen38-spark-engine',
        session_cache: true,
        prefix_cache: true,
        prefix_chunk: 512,
        last_prefill: { reused: 1536, forwarded: 423, source: 'prefix' },
        state_store: {
          entries: state.cacheEntries,
          bytes: Math.round(state.cacheBytes),
          budget: 8_000_000_000,
          puts: state.cacheHits['state,session'] + state.cacheHits['state,prefix'] + state.cacheMisses.state,
          hits: state.cacheHits['state,session'] + state.cacheHits['state,prefix'],
          misses: state.cacheMisses.state,
          evictions: state.cacheEvictions,
          rejected_collisions: 0,
          declined_big: 2,
          declined_short: 11,
          boundaries: [512, 1024, 2048, 4096, 8192],
          chunk: 512,
        },
        response_cache: {
          entries: 40,
          bytes: 1_200_000,
          budget: 64_000_000,
          ttl_s: 3600,
          hits: state.cacheHits.response,
          misses: state.cacheMisses.response,
          expired: 3,
          evictions: 0,
          puts: 771,
        },
        suffix_store: { path: '~/.qwen38-spark-engine/suffix.bin', tokens: 1_284_211, pending: 0, max_tokens: 4_000_000, indexed: true },
        snapshot_cost: { recurrent_bytes: 150_000_000, conv_bytes: 8_400_000, kv_bytes_per_token: 65536 },
      },
      queue: { running: state.running, waiting: mode === 'busy' ? 7 : state.waiting, max_queue: 8, queue_timeout_s: 240, request_timeout_s: 3600 },
      inflight: {
        served: Object.values(state.requestsByFinish).reduce((a, b) => a + b, 0),
        refused: Object.values(state.refusedByReason).reduce((a, b) => a + b, 0),
        errors: state.requestsByFinish.error ?? 0,
        timeouts: state.requestsByFinish.timeout ?? 0,
        abandoned: state.requestsByFinish.abandoned ?? 0,
      },
      ledger: {
        enabled: true,
        rows: rows.length,
        bytes: rows.length * 300 + 32768,
        oldest: rows.length ? new Date(rows[0].ts_ms).toISOString().replace(/\.\d{3}Z$/, 'Z') : null,
        queue: 0,
        dropped: mode === 'busy' ? 3 : 0,
      },
      disk: { state_dir_free_bytes: mode === 'busy' ? 15_000_000_000 : 119_185_342_464 },
    };
  };

  // ---- SSE log stream ----------------------------------------------------------------------
  let subscribers = 0;
  const streamLogs = (req: IncomingMessage, res: ServerResponse, q: URLSearchParams) => {
    if (subscribers >= 4) return error(res, 429, 'too_many', 'at most 4 log subscribers');
    const level = (q.get('level') as LogLevel) || 'info';
    const grep = q.get('grep');
    const headerId = req.headers['last-event-id'];
    const since = headerId ? Number(headerId) : q.get('since') ? Number(q.get('since')) : null;
    const backlogN = q.get('backlog') ? Number(q.get('backlog')) : 500;
    res.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache, no-transform',
      Connection: 'keep-alive',
      'X-Accel-Buffering': 'no',
    });
    subscribers++;
    state.logSubscribers = subscribers;
    const send = (l: LogLine) => {
      res.write(`event: log\nid: ${l.seq}\ndata: ${JSON.stringify(l)}\n\n`);
    };
    for (const l of logs.backlog({ since, level, grep, limit: backlogN })) send(l);
    let liveCount = 0;
    const unsub = logs.subscribe((l) => {
      if (l.level === 'debug' && level !== 'debug') return;
      if (level === 'warning' && l.level === 'info') return;
      if (level === 'error' && l.level !== 'error') return;
      if (grep && !l.msg.toLowerCase().includes(grep.toLowerCase())) return;
      liveCount++;
      if (liveCount % 400 === 0) res.write(`event: gap\ndata: ${JSON.stringify({ dropped: 3 })}\n\n`);
      send(l);
    });
    const ping = setInterval(() => res.write(': ping\n\n'), 15_000);
    let dropTimer: NodeJS.Timeout | null = null;
    if (mode.startsWith('drop')) dropTimer = setTimeout(() => res.end(), dropEveryMs);
    const cleanup = () => {
      unsub();
      clearInterval(ping);
      if (dropTimer) clearTimeout(dropTimer);
      subscribers--;
      state.logSubscribers = subscribers;
    };
    req.on('close', cleanup);
    res.on('close', cleanup);
  };

  // ---- SSE live stream (VIS-23): one event a second, the first with the history --------------
  let liveSubs = 0;
  const streamLive = (req: IncomingMessage, res: ServerResponse) => {
    if (liveSubs >= 4) return error(res, 429, 'too_many', 'at most 4 live streams at once');
    res.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache, no-transform',
      Connection: 'keep-alive',
      'X-Accel-Buffering': 'no',
    });
    liveSubs++;
    let first = true;
    let n = 0;
    const send = () => {
      const snap = live.snapshot(Date.now(), first);
      res.write(`${snap.seq != null ? `id: ${snap.seq}\n` : ''}event: live\ndata: ${JSON.stringify(snap)}\n\n`);
      first = false;
    };
    send();
    // contract 1.1 (SRV-39): four events a second while a request is in flight, one a second otherwise
    const timer = setInterval(() => {
      n++;
      const busy = live.contract === '1.1' && live.activity && [...live.reqs.values()].some((r) => r.endedAt == null);
      if (busy || n % 4 === 0) send();
    }, 250);
    let dropTimer: NodeJS.Timeout | null = null;
    if (mode.startsWith('drop')) dropTimer = setTimeout(() => res.end(), dropEveryMs);
    const cleanup = () => {
      clearInterval(timer);
      if (dropTimer) clearTimeout(dropTimer);
      liveSubs--;
    };
    req.on('close', cleanup);
    res.on('close', cleanup);
  };

  // ---- streaming chat completion (mock/chat.ts: thinking, formats, tools, stops, budgets) ---
  const chat = createChatHandler({
    mode: () => mode,
    draining: () => draining,
    rng: liveRng,
    state,
    live,
    readBody,
    finishRow: (row) => {
      row.ts_ms = Date.now();
      row.id = full.rows[full.rows.length - 1].id + 1;
      full.rows.push(row);
      applyRow(state, row);
      logs.pushRequest(row);
    },
  });

  // ---- the middleware ----------------------------------------------------------------------
  return (req, res, next) => {
    void (async () => {
      const url = new URL(req.url ?? '/', 'http://mock');
      const path = url.pathname.replace(/\/+$/, '') || '/';
      const q = url.searchParams;

      // The page itself: pick up ?mock=<mode> and remember it in a cookie.
      if ((path === '/dashboard' || path === '/' || path === '/index.html') && q.has('mock')) {
        const m = q.get('mock') || 'ok';
        mode = m.startsWith('drop') ? 'drop' : m;
        if (m.startsWith('drop:')) dropEveryMs = Number(m.slice(5)) * 1000;
        if (mode === '401') sessions.clear();
        res.setHeader('Set-Cookie', `qse_mock=${encodeURIComponent(m)}; Path=/; SameSite=Strict`);
        if (path === '/dashboard') req.url = '/' + url.search;
        return next();
      }
      // The engine serves the app at /dashboard/; the dev/preview server serves it at /.
      if (url.pathname === '/dashboard' && !q.has('mock')) {
        res.writeHead(302, { Location: '/dashboard/' + (url.search || '') });
        return res.end();
      }
      if (url.pathname.startsWith('/dashboard/')) {
        req.url = url.pathname.slice('/dashboard'.length) + url.search;
        return next();
      }
      if (path === '/__mock/mode') {
        if (req.method === 'POST') {
          const b = JSON.parse((await readBody(req)) || '{}');
          mode = String(b.mode ?? 'ok');
          if (mode.startsWith('drop:')) {
            dropEveryMs = Number(mode.slice(5)) * 1000;
            mode = 'drop';
          }
          if (mode === '401') sessions.clear();
          if (typeof b.draining === 'boolean') draining = b.draining;
          if (typeof b.waiting === 'number') state.waiting = b.waiting;
        }
        return json(res, 200, { mode, draining, waiting: state.waiting, rows: full.rows.length, lastSeq: logs.lastSeq });
      }
      if (path === '/__mock/log' && req.method === 'POST') {
        // tests: inject N lines
        const b = JSON.parse((await readBody(req)) || '{}');
        const n = Number(b.count ?? 1);
        for (let i = 0; i < n; i++) logs.push((b.level as LogLevel) ?? 'info', 'server', b.msg ?? `[server] injected line ${i + 1}`, null);
        return json(res, 200, { lastSeq: logs.lastSeq });
      }
      if (path === '/__mock/live' && req.method === 'POST') {
        // tests + screenshots (VIS-23, VIS-24): `busy` (one decoding, one prefilling, one queued, two
        // done), `agent-turn`, `abandoned`, `stops`, `constrained`, `clear`; `at` (ms) starts a
        // scenario that far in; `contract: "1.0"` answers like a SRV-34 server; `activity: false`
        // is the kill switch (1.1 with the new fields null); `draining: true` drains; `simulator: false`
        // stops the mock engine's own requests so a scenario owns the Now line.
        const b = JSON.parse((await readBody(req)) || '{}');
        const mk = (forced: Partial<LedgerRow>) => makeRow(liveRng, Date.now(), false, forced);
        const at = Number(b.at ?? 0) || 0;
        if (b.scenario === 'busy') live.scenario(mk);
        if (b.scenario === 'agent-turn') live.agentTurn(mk, Date.now(), at);
        if (b.scenario === 'abandoned') live.abandoned(mk, Date.now(), at);
        if (b.scenario === 'stops') live.stops(mk);
        if (b.scenario === 'constrained') live.constrained(mk, Date.now(), at || 3000);
        if (b.scenario === 'clear') live.clear();
        if (b.contract === '1.0' || b.contract === '1.1') live.contract = b.contract;
        if (typeof b.activity === 'boolean') live.activity = b.activity;
        if (typeof b.draining === 'boolean') live.draining = b.draining;
        if (typeof b.simulator === 'boolean') simulate = b.simulator;
        return json(res, 200, { ok: true, requests: live.reqs.size, recent: live.recent.length, contract: live.contract, activity: live.activity });
      }
      if (path === '/__mock/request' && req.method === 'POST') {
        // tests: finish a request now (a [req] line + a row)
        const row = makeRow(liveRng, Date.now(), false, { finish_reason: 'stop', status: 200 });
        row.id = full.rows[full.rows.length - 1].id + 1;
        full.rows.push(row);
        applyRow(state, row);
        logs.pushRequest(row);
        return json(res, 200, { request_id: row.request_id, id: row.id });
      }

      const isData = path.startsWith('/v1/dashboard/') && path !== '/v1/dashboard/session';
      const isMetrics = path === '/metrics';

      if (mode === 'offline' && (isData || isMetrics || path === '/health' || path === '/v1/dashboard/session')) {
        req.socket.destroy();
        return;
      }

      // ---- session
      if (path === '/v1/dashboard/session') {
        if (req.method === 'POST') {
          const b = JSON.parse((await readBody(req)) || '{}');
          if (typeof b.token !== 'string' || b.token !== TOKEN) {
            return error(res, 401, 'unauthorized', 'token refused');
          }
          const sid = randomBytes(16).toString('hex');
          sessions.set(sid, { expires: Date.now() + 43_200_000 });
          if (mode === '401') mode = 'ok';
          res.writeHead(204, { 'Set-Cookie': `qse_dash=${sid}; HttpOnly; SameSite=Strict; Path=/; Max-Age=43200` });
          return res.end();
        }
        if (req.method === 'DELETE') {
          const c = cookies(req).qse_dash;
          if (c) sessions.delete(c);
          res.writeHead(204, { 'Set-Cookie': 'qse_dash=; HttpOnly; SameSite=Strict; Path=/; Max-Age=0' });
          return res.end();
        }
        if (req.method === 'GET') {
          if (mode === '401' || !authed(req)) return error(res, 401, 'unauthorized', 'no session');
          const c = cookies(req).qse_dash;
          const s = c ? sessions.get(c) : null;
          return json(res, 200, { authenticated: true, expires_at: new Date(s?.expires ?? Date.now() + 43_200_000).toISOString().replace(/\.\d{3}Z$/, 'Z') });
        }
      }

      if (isData || isMetrics) {
        if (mode === 'expire') {
          mode = 'ok';
          sessions.clear();
          return error(res, 401, 'unauthorized', 'session expired');
        }
        if (mode === '401') return error(res, 401, 'unauthorized', 'session expired');
        const metricsOk = isMetrics && req.headers.authorization === `Bearer ${METRICS_TOKEN}`;
        if (!metricsOk && !authed(req)) return error(res, 401, 'unauthorized', 'a session or an admin token is required');
        if (mode === 'slow') await delay(3000);
      }

      if (isMetrics) {
        res.writeHead(200, { 'Content-Type': 'text/plain; version=0.0.4; charset=utf-8', 'Cache-Control': 'no-store' });
        return res.end(renderMetrics(state, Date.now(), { spec: mode !== 'nospec' }));
      }

      if (path === '/v1/dashboard/summary') {
        if (mode === '500') return json(res, 500, { error: { type: 'internal_error', message: 'ledger query failed: database is locked' } });
        const data = rowsFor();
        const now = Date.now();
        const last = data.rows.length ? data.rows[data.rows.length - 1] : null;
        return json(
          res,
          200,
          aggSummary({
            rows: data.rows,
            tz: q.get('tz') || tz,
            now,
            since: data.since,
            live: {
              status: liveStatus(),
              running: state.running,
              waiting: state.waiting,
              uptime_s: Math.round((now - startedAt) / 100) / 10,
              last_request_at: last ? new Date(last.ts_ms).toISOString().replace(/\.\d{3}Z$/, 'Z') : null,
            },
          }),
        );
      }
      if (path === '/v1/dashboard/usage') {
        const data = rowsFor();
        const ztz = q.get('tz') || tz;
        const bucket = (q.get('bucket') as 'day' | 'hour') || 'day';
        const today = dayKey(Date.now(), ztz);
        const to = (q.get('to') || today).slice(0, 10);
        const defFrom = shiftDay(to, bucket === 'day' ? -364 : -2);
        const from = (q.get('from') || defFrom).slice(0, 10);
        if (!/^\d{4}-\d{2}-\d{2}$/.test(from) || !/^\d{4}-\d{2}-\d{2}$/.test(to)) return error(res, 400, 'bad_request', 'from/to must be YYYY-MM-DD');
        if (bucket !== 'day' && bucket !== 'hour') return error(res, 400, 'bad_request', 'bucket must be day or hour');
        const span = (Date.parse(to) - Date.parse(from)) / 86400000 + 1;
        if (bucket === 'hour' && span > 31) return error(res, 400, 'bad_request', 'bucket=hour allows at most 31 days');
        if (span > 400) return error(res, 400, 'bad_request', 'at most 400 days');
        return json(res, 200, aggUsage(data.rows, { from, to, bucket, tz: ztz, model: q.get('model'), client: q.get('client') }));
      }
      if (path === '/v1/dashboard/requests') {
        const data = rowsFor();
        const limit = Number(q.get('limit') || 50);
        if (!Number.isFinite(limit) || limit < 1 || limit > 500) return error(res, 400, 'bad_request', 'limit must be 1..500');
        const before = q.get('before') ? Number(q.get('before')) : null;
        return json(res, 200, { contract_version: '1.0', ...aggRequests(data.rows, { limit, before, model: q.get('model'), client: q.get('client'), finish: q.get('finish') }) });
      }
      if (path === '/v1/dashboard/system') {
        return json(res, 200, sysSnapshot());
      }
      if (path === '/v1/dashboard/live') {
        if (q.get('follow') === '0' || q.get('follow') === 'false') return json(res, 200, live.snapshot(Date.now(), true));
        return streamLive(req, res);
      }
      if (path === '/v1/dashboard/logs') {
        if (q.get('follow') === '0') {
          const level = (q.get('level') as LogLevel) || 'info';
          const since = q.get('since') ? Number(q.get('since')) : null;
          const lines = logs.backlog({ since, level, grep: q.get('grep'), limit: q.get('backlog') ? Number(q.get('backlog')) : 500 });
          return json(res, 200, { contract_version: '1.0', lines, last_seq: logs.lastSeq });
        }
        return streamLogs(req, res, q);
      }
      if (path.startsWith('/v1/dashboard/')) return error(res, 404, 'not_found', `no route ${path}`);

      if (path === '/v1/chat/completions' && req.method === 'POST') return chat(req, res);
      if (path === '/v1/models') {
        return json(res, 200, { object: 'list', data: [{ id: 'qwen38-spark-engine', object: 'model', created: Math.floor(startedAt / 1000), owned_by: 'local' }] });
      }
      if (path === '/health') return json(res, 200, { status: liveStatus() });
      next();
    })().catch((e) => {
      if (!res.headersSent) json(res, 500, { error: { type: 'internal_error', message: String(e) } });
      else res.end();
    });
  };
}

function shiftDay(key: string, n: number): string {
  const t = Date.parse(key + 'T00:00:00Z') + n * 86400000;
  return new Date(t).toISOString().slice(0, 10);
}
