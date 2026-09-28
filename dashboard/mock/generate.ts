// Seeded generator of a year of synthetic usage — one row per request, the ledger's own columns
//. Same seed → same rows. Diurnal + weekly rhythm, quiet weeks, three very heavy days,
// a client mix, both endpoints, every finish reason, tool calls, thinking on/off, every cache
// source, speeds around today's live numbers (decode p50 ~60 tok/s, tail to ~140; TTFT from
// 5 ms replays to 10+ s long prompts).

import { addDays, dayKey, localToUtc } from '../src/lib/time.ts';
import type { CacheSource, ClientKind, Endpoint, FinishReason } from '../src/api/types.ts';

export interface LedgerRow {
  id: number;
  ts_ms: number;
  request_id: string;
  model: string;
  client_id: string;
  client_kind: ClientKind;
  endpoint: Endpoint;
  stream: boolean;
  status: number;
  finish_reason: FinishReason | null;
  prompt_tokens: number | null;
  cached_tokens: number | null;
  completion_tokens: number | null;
  reasoning_tokens: number | null;
  queue_ms: number | null;
  prompt_ms: number | null;
  ttft_ms: number | null;
  decode_ms: number | null;
  total_ms: number | null;
  decode_tps: number | null;
  prefill_tps: number | null;
  blocks: number | null;
  draft_tokens: number | null;
  draft_accepted: number | null;
  tool_calls: number;
  thinking: boolean;
  cache_source: CacheSource | null;
  max_tokens: number | null;
  error_type: string | null;
  engine_version: string;
  code_sha: string;
}

export interface ClientDef {
  id: string;
  label: string | null;
  kind: ClientKind;
  weight: number;
}

export const CLIENTS: ClientDef[] = [
  { id: 'k:3f9a1c0e2b7d', label: 'open-webui', kind: 'open-webui', weight: 0.58 },
  { id: 'k:8c21d4e0aa19', label: 'agent-loop', kind: 'openai-sdk', weight: 0.2 },
  { id: 'anon', label: null, kind: 'curl', weight: 0.1 },
  { id: 'k:0b7e55c3d1f2', label: 'dashboard', kind: 'dashboard', weight: 0.05 },
  { id: 'k:aa90e1c47b03', label: null, kind: 'other', weight: 0.07 },
];

export const MODELS = ['qwen38-spark-engine'];
export const ENGINE_VERSION = '0.1.0-rc4';
export const CODE_SHA = '7d0b176';

/** mulberry32 — small, fast, deterministic. */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gauss(r: () => number): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = r();
  while (v === 0) v = r();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function logNormal(r: () => number, median: number, sigma: number): number {
  return median * Math.exp(sigma * gauss(r));
}

function pick<T>(r: () => number, items: T[], weights: number[]): T {
  const total = weights.reduce((a, b) => a + b, 0);
  let x = r() * total;
  for (let i = 0; i < items.length; i++) {
    x -= weights[i];
    if (x <= 0) return items[i];
  }
  return items[items.length - 1];
}

function hex(r: () => number, n: number): string {
  let s = '';
  for (let i = 0; i < n; i++) s += Math.floor(r() * 16).toString(16);
  return s;
}

// Hour-of-day weights (local): quiet nights, morning, afternoon and evening peaks.
const HOURS = [0.4, 0.2, 0.1, 0.05, 0.05, 0.1, 0.3, 0.8, 1.4, 1.8, 1.9, 1.6, 1.0, 1.3, 1.7, 1.8, 1.6, 1.3, 0.9, 0.9, 1.2, 1.5, 1.3, 0.8];
// Weekday weights, Monday first.
const WEEKDAYS = [1.0, 1.1, 1.15, 1.1, 0.95, 0.5, 0.45];

export interface GenerateOptions {
  seed?: number;
  days?: number; // how many days of history ending on `today`
  today?: string; // local day key of the last day
  tz?: string;
  /** clock of "now" (ms) — rows after this are not generated */
  now?: number;
}

export interface Generated {
  rows: LedgerRow[];
  since: string; // first day key
  tz: string;
  seed: number;
}

export function generateYear(opts: GenerateOptions = {}): Generated {
  const tz = opts.tz ?? 'Europe/Berlin';
  const seed = opts.seed ?? 42;
  const now = opts.now ?? Date.now();
  const today = opts.today ?? dayKey(now, tz);
  const days = opts.days ?? 400;
  const r = rng(seed);
  const first = addDays(today, -(days - 1));

  // Structural choices fixed by the seed: quiet weeks, heavy days, a growth trend.
  const quietStarts = new Set<number>();
  while (quietStarts.size < 3) quietStarts.add(Math.floor(r() * (days - 14)));
  const heavyDays = new Set<number>();
  while (heavyDays.size < 3) heavyDays.add(Math.floor(days * 0.3 + r() * days * 0.68));
  const baseline = 28 + r() * 20; // requests/day at the start

  const rows: LedgerRow[] = [];
  let id = 0;
  for (let d = 0; d < days; d++) {
    const key = addDays(first, d);
    const [y, m, dd] = key.split('-').map(Number);
    const weekday = (new Date(Date.UTC(y, m - 1, dd)).getUTCDay() + 6) % 7;
    const intensity = WEEKDAYS[weekday] * (1 + (d / days) * 1.4); // grows through the year
    let quiet = false;
    for (const q of quietStarts) if (d >= q && d < q + 9) quiet = true;
    const heavy = heavyDays.has(d);
    // Quiet weeks (holidays): most days have no requests at all, a few have one to three.
    const nReq = heavy ? 220 + Math.floor(r() * 160) : quiet ? (r() < 0.7 ? 0 : 1 + Math.floor(r() * 3)) : Math.max(0, Math.round(baseline * intensity * (0.6 + r() * 0.8)));
    for (let i = 0; i < nReq; i++) {
      const hour = pick(r, [...HOURS.keys()], HOURS);
      const ms = localToUtc(y, m, dd, hour, Math.floor(r() * 60), tz) + Math.floor(r() * 60000);
      if (ms > now) continue;
      rows.push(makeRow(r, ms, heavy));
    }
  }
  rows.sort((a, b) => a.ts_ms - b.ts_ms);
  for (const row of rows) row.id = ++id;
  return { rows, since: first, tz, seed };
}

/** One synthetic request at instant `ms`. Exposed so the mock server can append live rows. */
export function makeRow(r: () => number, ms: number, heavy = false, forced?: Partial<LedgerRow>): LedgerRow {
  const client = pick(r, CLIENTS, CLIENTS.map((c) => c.weight));
  const endpoint: Endpoint = r() < 0.92 ? 'chat' : 'completions';
  const stream = r() < 0.9;
  const thinking = endpoint === 'chat' && r() < 0.55;
  const wantsTools = endpoint === 'chat' && r() < 0.12;
  const cacheSource = pick<CacheSource>(r, ['response', 'session', 'prefix', 'none'], [0.05, 0.22, 0.33, 0.4]);
  const finish = pick<FinishReason>(
    r,
    ['stop', 'length', 'tool_calls', 'timeout', 'abandoned', 'error', 'refused'],
    [0.83, 0.06, wantsTools ? 0.6 : 0, 0.004, 0.02, 0.012, 0.01],
  );
  const model = MODELS[0];
  const base: LedgerRow = {
    id: 0,
    ts_ms: ms,
    request_id: (endpoint === 'chat' ? 'chatcmpl-' : 'cmpl-') + hex(r, 16),
    model,
    client_id: client.id,
    client_kind: client.kind,
    endpoint,
    stream,
    status: 200,
    finish_reason: finish,
    prompt_tokens: null,
    cached_tokens: null,
    completion_tokens: null,
    reasoning_tokens: null,
    queue_ms: null,
    prompt_ms: null,
    ttft_ms: null,
    decode_ms: null,
    total_ms: null,
    decode_tps: null,
    prefill_tps: null,
    blocks: null,
    draft_tokens: null,
    draft_accepted: null,
    tool_calls: 0,
    thinking,
    cache_source: null,
    max_tokens: pick(r, [256, 1024, 4096, 8192, 16384], [0.1, 0.2, 0.3, 0.35, 0.05]),
    error_type: null,
    engine_version: ENGINE_VERSION,
    code_sha: CODE_SHA,
  };
  if (finish === 'refused') {
    base.status = r() < 0.7 ? 503 : 429;
    base.queue_ms = 0;
    base.total_ms = 1 + r() * 3;
    return Object.assign(base, forced);
  }

  const prompt = Math.round(Math.min(120_000, logNormal(r, heavy ? 9000 : 1500, heavy ? 0.9 : 1.1)));
  let cached = 0;
  if (cacheSource === 'response') cached = prompt;
  else if (cacheSource === 'session') cached = Math.round(prompt * (0.6 + r() * 0.38));
  else if (cacheSource === 'prefix') cached = Math.round(prompt * (0.2 + r() * 0.6));
  const uncached = prompt - cached;
  let completion = Math.round(Math.min(base.max_tokens ?? 8192, logNormal(r, thinking ? 700 : 320, 0.8)));
  if (finish === 'length') completion = base.max_tokens ?? completion;
  if (finish === 'abandoned' || finish === 'timeout') completion = Math.round(completion * (0.1 + r() * 0.5));
  if (finish === 'error') completion = Math.round(completion * r() * 0.3);
  completion = Math.max(1, completion);
  const reasoning = thinking ? Math.min(completion - 1, Math.round(completion * (0.3 + r() * 0.45))) : 0;

  const queueMs = r() < 0.85 ? 0.2 + r() * 1.5 : 200 + r() * 6000;
  const prefillTps = Math.max(150, logNormal(r, 2100, 0.35));
  const promptMs = cacheSource === 'response' ? 3 + r() * 4 : 35 + (uncached / prefillTps) * 1000;
  const ttft = queueMs + promptMs;
  const tokensPerBlock = cacheSource === 'response' ? null : Math.max(1.2, 4.1 + gauss(r) * 0.6);
  // A response-cache replay has no decode; its "speed" is the honest replay wall clock.
  const replayMs = 2 + completion * 0.02;
  const decodeTps = cacheSource === 'response' ? (completion - 1) / (replayMs / 1000) : Math.min(148, Math.max(12, logNormal(r, 60, 0.38)));
  const decodeMs = cacheSource === 'response' ? replayMs : ((completion - 1) / decodeTps) * 1000;
  const blocks = tokensPerBlock ? Math.max(1, Math.ceil((completion - 1) / tokensPerBlock)) : null;
  const acceptance = 0.18 + r() * 0.2;
  const draftTokens = blocks ? blocks * 15 : null;
  const draftAccepted = draftTokens ? Math.round(draftTokens * acceptance) : null;

  Object.assign(base, {
    prompt_tokens: prompt,
    cached_tokens: cached,
    completion_tokens: completion,
    reasoning_tokens: reasoning,
    queue_ms: round2(queueMs),
    prompt_ms: round2(promptMs),
    ttft_ms: round2(ttft),
    decode_ms: round2(decodeMs),
    total_ms: round2(ttft + decodeMs),
    decode_tps: round2(decodeTps),
    prefill_tps: cacheSource === 'response' ? null : round2(prefillTps),
    blocks,
    draft_tokens: draftTokens,
    draft_accepted: draftAccepted,
    tool_calls: finish === 'tool_calls' ? 1 + Math.floor(r() * 3) : 0,
    cache_source: cacheSource,
    error_type: finish === 'error' ? pick(r, ['RuntimeError', 'ValueError', 'TimeoutError'], [0.6, 0.3, 0.1]) : null,
  });
  if (finish === 'error') base.status = 500;
  return Object.assign(base, forced);
}

function round2(x: number): number {
  return Math.round(x * 100) / 100;
}
