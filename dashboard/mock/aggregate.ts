// Aggregation of ledger rows into the contract's shapes, following its rules exactly: dense
// buckets (zero-filled, percentiles null), local days/hours in `tz`, exact percentiles over the
// rows in range, requests without decode (errors, refused, response-cache replays) excluded from
// the speed percentiles and counted in the token sums. The backend implements the same
// rules in SQL; these functions are the executable reading of the contract.

import type { Bucket, ClientDim, RequestRow, Summary, Totals, TopDay, Usage, UsageBucket, LiveStatus } from '../src/api/types.ts';
import { addDays, dayKey, dayRange, dayStart, isoWithOffset } from '../src/lib/time.ts';
import { streaks } from '../src/lib/heatmap.ts';
import { CLIENTS, type LedgerRow } from './generate.ts';

export interface Filters {
  model?: string | null;
  client?: string | null;
}

function matches(row: LedgerRow, f: Filters): boolean {
  if (f.model && row.model !== f.model) return false;
  if (f.client && row.client_id !== f.client) return false;
  return true;
}

/** Rows that carry a decode speed: served, not a replay, not an error. */
function hasDecode(row: LedgerRow): boolean {
  return (
    row.status === 200 &&
    row.finish_reason !== 'error' &&
    row.finish_reason !== 'refused' &&
    row.cache_source !== 'response' &&
    row.decode_tps != null
  );
}

/** Exact percentile, nearest-rank, over a sorted copy. Null on empty. */
export function percentile(values: number[], p: number): number | null {
  if (values.length === 0) return null;
  const s = [...values].sort((a, b) => a - b);
  const rank = Math.max(1, Math.ceil(p * s.length));
  return round2(s[rank - 1]);
}

function mean(values: number[]): number | null {
  if (values.length === 0) return null;
  return round2(values.reduce((a, b) => a + b, 0) / values.length);
}

function round2(x: number): number {
  return Math.round(x * 100) / 100;
}

export function totals(rows: LedgerRow[], tz: string): Totals {
  let requests = 0;
  let errors = 0;
  let refused = 0;
  let prompt = 0;
  let cached = 0;
  let completion = 0;
  let reasoning = 0;
  let tool = 0;
  let think = 0;
  const decode: number[] = [];
  const ttft: number[] = [];
  const prefill: number[] = [];
  const tpb: number[] = [];
  let draftT = 0;
  let draftA = 0;
  const days = new Set<string>();
  for (const r of rows) {
    requests++;
    days.add(dayKey(r.ts_ms, tz));
    if (r.finish_reason === 'refused') refused++;
    if (r.finish_reason === 'error') errors++;
    prompt += r.prompt_tokens ?? 0;
    cached += r.cached_tokens ?? 0;
    completion += r.completion_tokens ?? 0;
    reasoning += r.reasoning_tokens ?? 0;
    if (r.tool_calls > 0) tool++;
    if (r.thinking) think++;
    if (hasDecode(r)) {
      decode.push(r.decode_tps as number);
      if (r.ttft_ms != null) ttft.push(r.ttft_ms);
      if (r.prefill_tps != null) prefill.push(r.prefill_tps);
      if (r.blocks && r.completion_tokens) tpb.push((r.completion_tokens - 1) / r.blocks);
      draftT += r.draft_tokens ?? 0;
      draftA += r.draft_accepted ?? 0;
    }
  }
  return {
    requests,
    errors,
    refused,
    prompt_tokens: prompt,
    cached_tokens: cached,
    completion_tokens: completion,
    reasoning_tokens: reasoning,
    total_tokens: prompt + completion,
    tool_call_requests: tool,
    thinking_requests: think,
    decode_tps_p50: percentile(decode, 0.5),
    decode_tps_p90: percentile(decode, 0.9),
    ttft_ms_p50: percentile(ttft, 0.5),
    ttft_ms_p90: percentile(ttft, 0.9),
    prefill_tps_p50: percentile(prefill, 0.5),
    tokens_per_block_mean: mean(tpb),
    draft_acceptance: draftT > 0 ? round2(draftA / draftT) : null,
    active_days: days.size,
  };
}

function bucketOf(rows: LedgerRow[], start: string): UsageBucket {
  const t = totals(rows, 'UTC');
  return {
    start,
    requests: t.requests,
    errors: t.errors,
    prompt_tokens: t.prompt_tokens,
    cached_tokens: t.cached_tokens,
    completion_tokens: t.completion_tokens,
    reasoning_tokens: t.reasoning_tokens,
    total_tokens: t.total_tokens,
    tool_call_requests: t.tool_call_requests,
    decode_tps_p50: t.decode_tps_p50,
    decode_tps_p90: t.decode_tps_p90,
    ttft_ms_p50: t.ttft_ms_p50,
    ttft_ms_p90: t.ttft_ms_p90,
    prefill_tps_p50: t.prefill_tps_p50,
    tokens_per_block_mean: t.tokens_per_block_mean,
    draft_acceptance: t.draft_acceptance,
  };
}

export function clientDims(): ClientDim[] {
  return CLIENTS.map((c) => ({ id: c.id, label: c.label, kind: c.kind }));
}

export function clientDim(id: string): ClientDim {
  const c = CLIENTS.find((x) => x.id === id);
  return c ? { id: c.id, label: c.label, kind: c.kind } : { id, label: null, kind: 'other' };
}

export interface UsageQuery extends Filters {
  from: string; // day key
  to: string; // day key
  bucket: Bucket;
  tz: string;
}

export function usage(all: LedgerRow[], q: UsageQuery): Usage {
  const rows = all.filter((r) => matches(r, q));
  const startMs = dayStart(q.from, q.tz);
  const endMs = dayStart(addDays(q.to, 1), q.tz);
  const inRange = rows.filter((r) => r.ts_ms >= startMs && r.ts_ms < endMs);

  const buckets: UsageBucket[] = [];
  if (q.bucket === 'day') {
    const byDay = new Map<string, LedgerRow[]>();
    for (const r of inRange) {
      const k = dayKey(r.ts_ms, q.tz);
      (byDay.get(k) ?? byDay.set(k, []).get(k)!).push(r);
    }
    for (const day of dayRange(q.from, q.to)) {
      buckets.push(bucketOf(byDay.get(day) ?? [], isoWithOffset(dayStart(day, q.tz), q.tz)));
    }
  } else {
    // Hour buckets are consecutive hours of instants from each local day's start: the DST days
    // get 23 or 25 of them, and the repeated fall-back hour is two buckets with two offsets.
    for (const day of dayRange(q.from, q.to)) {
      const dStart = dayStart(day, q.tz);
      const dEnd = dayStart(addDays(day, 1), q.tz);
      const byHour = new Map<number, LedgerRow[]>();
      for (const r of inRange) {
        if (r.ts_ms < dStart || r.ts_ms >= dEnd) continue;
        const k = dStart + Math.floor((r.ts_ms - dStart) / 3600000) * 3600000;
        (byHour.get(k) ?? byHour.set(k, []).get(k)!).push(r);
      }
      for (let ms = dStart; ms < dEnd; ms += 3600000) buckets.push(bucketOf(byHour.get(ms) ?? [], isoWithOffset(ms, q.tz)));
    }
  }

  // Top days over the whole (filtered) range, day buckets only.
  const topDays: TopDay[] = [];
  if (q.bucket === 'day') {
    const perDay = new Map<string, { tokens: number; requests: number; byClient: Map<string, number> }>();
    for (const r of inRange) {
      const k = dayKey(r.ts_ms, q.tz);
      const e = perDay.get(k) ?? { tokens: 0, requests: 0, byClient: new Map() };
      const t = (r.prompt_tokens ?? 0) + (r.completion_tokens ?? 0);
      e.tokens += t;
      e.requests++;
      e.byClient.set(r.client_id, (e.byClient.get(r.client_id) ?? 0) + t);
      perDay.set(k, e);
    }
    for (const [date, e] of [...perDay.entries()].sort((a, b) => b[1].tokens - a[1].tokens).slice(0, 10)) {
      let top: string | null = null;
      let best = -1;
      for (const [c, t] of e.byClient) if (t > best) ((best = t), (top = c));
      topDays.push({ date, total_tokens: e.tokens, requests: e.requests, top_client: top });
    }
  }

  return {
    contract_version: '1.0',
    from: q.from,
    to: q.to,
    bucket: q.bucket,
    tz: q.tz,
    filters: { model: q.model ?? null, client: q.client ?? null },
    buckets,
    totals: totals(inRange, q.tz),
    top_days: topDays,
    dimensions: { models: [...new Set(all.map((r) => r.model))], clients: clientDims() },
  };
}

export interface SummaryInput {
  rows: LedgerRow[];
  tz: string;
  now: number;
  since: string; // ledger since day key (or null when disabled)
  live: { status: LiveStatus; running: number; waiting: number; uptime_s: number; last_request_at: string | null };
  ledgerBytes?: number;
}

export function summary(inp: SummaryInput): Summary {
  const today = dayKey(inp.now, inp.tz);
  const win = (days: number) => {
    const from = dayStart(addDays(today, -(days - 1)), inp.tz);
    return inp.rows.filter((r) => r.ts_ms >= from && r.ts_ms <= inp.now);
  };
  const yearRows = win(365);
  const dayBuckets = usage(inp.rows, { from: addDays(today, -364), to: today, bucket: 'day', tz: inp.tz }).buckets;
  const st = streaks(dayBuckets, today);
  void yearRows;
  return {
    contract_version: '1.0',
    generated_at: new Date(inp.now).toISOString().replace(/\.\d{3}Z$/, 'Z'),
    tz: inp.tz,
    windows: {
      today: totals(win(1), inp.tz),
      '7d': totals(win(7), inp.tz),
      '30d': totals(win(30), inp.tz),
      '365d': totals(win(365), inp.tz),
      all: totals(inp.rows, inp.tz),
    },
    streak: { current_days: st.current, longest_days: st.longest },
    ledger: {
      enabled: true,
      since: new Date(dayStart(inp.since, inp.tz)).toISOString().replace(/\.\d{3}Z$/, 'Z'),
      rows: inp.rows.length,
      bytes: inp.ledgerBytes ?? inp.rows.length * 300 + 32768,
      retention_days: 400,
    },
    live: inp.live,
  };
}

export function toRequestRow(r: LedgerRow): RequestRow {
  return {
    id: r.id,
    request_id: r.request_id,
    ts: new Date(r.ts_ms).toISOString(),
    model: r.model,
    client: clientDim(r.client_id),
    endpoint: r.endpoint,
    stream: r.stream,
    status: r.status,
    finish_reason: r.finish_reason,
    prompt_tokens: r.prompt_tokens,
    cached_tokens: r.cached_tokens,
    completion_tokens: r.completion_tokens,
    reasoning_tokens: r.reasoning_tokens,
    queue_ms: r.queue_ms,
    prompt_ms: r.prompt_ms,
    ttft_ms: r.ttft_ms,
    decode_ms: r.decode_ms,
    total_ms: r.total_ms,
    decode_tps: r.decode_tps,
    prefill_tps: r.prefill_tps,
    blocks: r.blocks,
    tokens_per_block: r.blocks && r.completion_tokens ? round2((r.completion_tokens - 1) / r.blocks) : null,
    draft_tokens: r.draft_tokens,
    draft_accepted: r.draft_accepted,
    tool_calls: r.tool_calls,
    thinking: r.thinking,
    cache_source: r.cache_source,
    error_type: r.error_type,
  };
}

export interface RequestsQuery extends Filters {
  limit: number;
  before?: number | null;
  finish?: string | null;
}

export function requests(all: LedgerRow[], q: RequestsQuery): { next_before: number | null; requests: RequestRow[] } {
  const limit = Math.max(1, Math.min(500, q.limit));
  const out: RequestRow[] = [];
  for (let i = all.length - 1; i >= 0 && out.length < limit; i--) {
    const r = all[i];
    if (q.before != null && r.id >= q.before) continue;
    if (!matches(r, q)) continue;
    if (q.finish && r.finish_reason !== q.finish) continue;
    out.push(toRequestRow(r));
  }
  const last = out[out.length - 1];
  return { next_before: out.length === limit && last ? last.id : null, requests: out };
}
