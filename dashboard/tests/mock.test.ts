// The mock generator and its aggregation, held to the contract's rules.
import { describe, expect, it } from 'vitest';
import { generateYear } from '../mock/generate.ts';
import { percentile, requests, summary, totals, usage } from '../mock/aggregate.ts';
import { addDays, dayKey } from '../src/lib/time.ts';

const TZ = 'Europe/Berlin';
const NOW = Date.parse('2026-09-24T14:00:00Z');

describe('seeded generator', () => {
  it('same seed → identical rows; another seed → different rows', () => {
    const a = generateYear({ seed: 42, now: NOW, tz: TZ });
    const b = generateYear({ seed: 42, now: NOW, tz: TZ });
    expect(a.rows.length).toBeGreaterThan(5000);
    expect(JSON.stringify(a.rows.slice(0, 200))).toBe(JSON.stringify(b.rows.slice(0, 200)));
    expect(a.rows.length).toBe(b.rows.length);
    const c = generateYear({ seed: 7, now: NOW, tz: TZ });
    expect(c.rows.length).not.toBe(a.rows.length);
  });

  it('covers 400 days, sorted, ids sequential, nothing in the future', () => {
    const g = generateYear({ seed: 42, now: NOW, tz: TZ });
    expect(g.since).toBe(addDays(dayKey(NOW, TZ), -399));
    for (let i = 1; i < g.rows.length; i++) {
      expect(g.rows[i].ts_ms).toBeGreaterThanOrEqual(g.rows[i - 1].ts_ms);
      expect(g.rows[i].id).toBe(g.rows[i - 1].id + 1);
    }
    expect(g.rows[g.rows.length - 1].ts_ms).toBeLessThanOrEqual(NOW);
  });

  it('has every finish reason, every cache source, tool calls, thinking on and off, both endpoints, all client kinds', () => {
    const g = generateYear({ seed: 42, now: NOW, tz: TZ });
    const set = (f: (r: (typeof g.rows)[number]) => unknown) => new Set(g.rows.map(f));
    expect(set((r) => r.finish_reason)).toEqual(new Set(['stop', 'length', 'tool_calls', 'timeout', 'abandoned', 'error', 'refused']));
    expect(set((r) => r.cache_source)).toEqual(new Set(['response', 'session', 'prefix', 'none', null]));
    expect(set((r) => r.thinking)).toEqual(new Set([true, false]));
    expect(set((r) => r.endpoint)).toEqual(new Set(['chat', 'completions']));
    expect(set((r) => r.client_kind)).toEqual(new Set(['open-webui', 'openai-sdk', 'curl', 'dashboard', 'other']));
    expect(g.rows.some((r) => r.tool_calls > 0)).toBe(true);
    const refused = g.rows.filter((r) => r.finish_reason === 'refused');
    expect(refused.every((r) => r.prompt_tokens === null && (r.status === 503 || r.status === 429))).toBe(true);
  });

  it('speeds sit around the live numbers: decode p50 ~60 with a tail to ~140; three days > 1M tokens', () => {
    const g = generateYear({ seed: 42, now: NOW, tz: TZ });
    const t = totals(g.rows, TZ);
    expect(t.decode_tps_p50).toBeGreaterThan(45);
    expect(t.decode_tps_p50).toBeLessThan(80);
    expect(t.decode_tps_p90).toBeGreaterThan(t.decode_tps_p50!);
    expect(Math.max(...g.rows.filter((r) => r.cache_source !== 'response').map((r) => r.decode_tps ?? 0))).toBeLessThanOrEqual(148);
    // a replay carries its honest wall-clock speed, far above any decode
    expect(Math.min(...g.rows.filter((r) => r.cache_source === 'response').map((r) => r.decode_tps ?? 0))).toBeGreaterThan(1000);
    const u = usage(g.rows, { from: g.since, to: dayKey(NOW, TZ), bucket: 'day', tz: TZ });
    expect(u.buckets.filter((b) => b.total_tokens > 1_000_000).length).toBeGreaterThanOrEqual(3);
  });
});

describe('aggregation rules', () => {
  const g = generateYear({ seed: 42, now: NOW, tz: TZ });
  const today = dayKey(NOW, TZ);

  it('percentiles are exact (nearest rank) and null on empty', () => {
    expect(percentile([], 0.5)).toBeNull();
    expect(percentile([5, 1, 3], 0.5)).toBe(3);
    expect(percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.9)).toBe(9);
  });

  it('day buckets are dense: exactly 365 for a year, zero days with null percentiles', () => {
    const u = usage(g.rows, { from: addDays(today, -364), to: today, bucket: 'day', tz: TZ });
    expect(u.buckets).toHaveLength(365);
    expect(u.buckets[0].start.slice(0, 10)).toBe(addDays(today, -364));
    expect(u.buckets[364].start.slice(0, 10)).toBe(today);
    const zero = u.buckets.find((b) => b.requests === 0);
    expect(zero).toBeDefined();
    expect(zero!.decode_tps_p50).toBeNull();
    expect(zero!.total_tokens).toBe(0);
    // every bucket's start carries the zone's own offset
    expect(u.buckets.every((b) => /[+-]\d{2}:\d{2}$/.test(b.start))).toBe(true);
  });

  it('hour buckets: the DST days have 23 and 25 buckets', () => {
    const spring = usage(g.rows, { from: '2026-03-29', to: '2026-03-29', bucket: 'hour', tz: TZ });
    expect(spring.buckets).toHaveLength(23);
    const fall = usage(g.rows, { from: '2025-10-26', to: '2025-10-26', bucket: 'hour', tz: TZ });
    expect(fall.buckets).toHaveLength(25);
    const plain = usage(g.rows, { from: '2026-09-20', to: '2026-09-21', bucket: 'hour', tz: TZ });
    expect(plain.buckets).toHaveLength(48);
  });

  it('speed percentiles exclude replays, errors and refused; token sums include them', () => {
    const rows = g.rows.filter((r) => r.ts_ms >= Date.parse('2026-09-01T00:00:00Z'));
    const t = totals(rows, TZ);
    const decodeRows = rows.filter((r) => r.status === 200 && r.cache_source !== 'response' && r.finish_reason !== 'error' && r.decode_tps != null);
    expect(t.decode_tps_p50).toBe(percentile(decodeRows.map((r) => r.decode_tps as number), 0.5));
    const replayTokens = rows.filter((r) => r.cache_source === 'response').reduce((a, r) => a + (r.prompt_tokens ?? 0) + (r.completion_tokens ?? 0), 0);
    expect(replayTokens).toBeGreaterThan(0);
    expect(t.total_tokens).toBe(rows.reduce((a, r) => a + (r.prompt_tokens ?? 0) + (r.completion_tokens ?? 0), 0));
    expect(t.requests).toBe(rows.length);
    expect(t.refused).toBe(rows.filter((r) => r.finish_reason === 'refused').length);
  });

  it('filters by client and model exactly; dimensions list every client', () => {
    const u = usage(g.rows, { from: addDays(today, -29), to: today, bucket: 'day', tz: TZ, client: 'k:3f9a1c0e2b7d' });
    const expected = g.rows.filter((r) => r.client_id === 'k:3f9a1c0e2b7d' && r.ts_ms >= Date.parse(addDays(today, -29) + 'T00:00:00+02:00'));
    expect(u.totals.requests).toBe(expected.length);
    expect(u.filters.client).toBe('k:3f9a1c0e2b7d');
    expect(u.dimensions.clients.length).toBe(5);
    expect(u.dimensions.models).toEqual(['qwen38-spark-engine']);
    const none = usage(g.rows, { from: addDays(today, -29), to: today, bucket: 'day', tz: TZ, model: 'other-model' });
    expect(none.totals.requests).toBe(0);
    expect(none.buckets).toHaveLength(30);
  });

  it('top days are the ten heaviest, descending, with a top client', () => {
    const u = usage(g.rows, { from: addDays(today, -364), to: today, bucket: 'day', tz: TZ });
    expect(u.top_days).toHaveLength(10);
    for (let i = 1; i < 10; i++) expect(u.top_days[i].total_tokens).toBeLessThanOrEqual(u.top_days[i - 1].total_tokens);
    expect(u.top_days[0].top_client).toBeTruthy();
    const hour = usage(g.rows, { from: today, to: today, bucket: 'hour', tz: TZ });
    expect(hour.top_days).toEqual([]);
  });

  it('summary windows nest: today ⊆ 7d ⊆ 30d ⊆ 365d ⊆ all; streak fields present', () => {
    const s = summary({ rows: g.rows, tz: TZ, now: NOW, since: g.since, live: { status: 'ok', running: 0, waiting: 0, uptime_s: 10, last_request_at: null } });
    const w = s.windows;
    expect(w.today.requests).toBeLessThanOrEqual(w['7d'].requests);
    expect(w['7d'].requests).toBeLessThanOrEqual(w['30d'].requests);
    expect(w['30d'].requests).toBeLessThanOrEqual(w['365d'].requests);
    expect(w['365d'].requests).toBeLessThanOrEqual(w.all.requests);
    expect(w.today.active_days).toBeLessThanOrEqual(1);
    expect(s.streak.longest_days).toBeGreaterThanOrEqual(s.streak.current_days);
    expect(s.ledger.rows).toBe(g.rows.length);
    expect(s.contract_version).toBe('1.0');
  });

  it('requests page newest first with keyset paging and no overlap', () => {
    const p1 = requests(g.rows, { limit: 50 });
    expect(p1.requests).toHaveLength(50);
    expect(p1.requests[0].id).toBeGreaterThan(p1.requests[49].id);
    expect(p1.next_before).toBe(p1.requests[49].id);
    const p2 = requests(g.rows, { limit: 50, before: p1.next_before });
    expect(p2.requests[0].id).toBeLessThan(p1.requests[49].id);
    const f = requests(g.rows, { limit: 20, finish: 'refused' });
    expect(f.requests.every((r) => r.finish_reason === 'refused')).toBe(true);
    expect(requests(g.rows, { limit: 5000 }).requests).toHaveLength(500);
  });
});
