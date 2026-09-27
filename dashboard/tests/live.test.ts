import { describe, expect, it } from 'vitest';
import { agoShort, elapsed, emptyRing, orderRows, peak, phaseWords, push, seedRing, slots, sumDecodeNow, WINDOW_S } from '../src/lib/live';
import type { LiveRequest, LiveSample } from '../src/api/types';

const s = (t: number, decode: number | null, prefill: number | null = null, running = 1): LiveSample => ({ t, decode_tps: decode, prefill_tps: prefill, running, waiting: 0, tokens: 0 });

describe('the five-minute ring', () => {
  it('seeds from history and drops what is older than the window', () => {
    const now = 10_000;
    const ring = seedRing([s(now - 400, 10), s(now - 299, 20), s(now - 1, 30), s(now, 40)], now);
    expect(ring.samples.map((x) => x.t)).toEqual([now - 299, now - 1, now]);
  });
  it('appends in order and ignores a resent sample', () => {
    const ring = emptyRing();
    push(ring, s(1, 1), 1);
    push(ring, s(2, 2), 2);
    push(ring, s(2, 99), 2);
    push(ring, s(1, 99), 2);
    push(ring, null, 2);
    expect(ring.samples.map((x) => x.decode_tps)).toEqual([1, 2]);
  });
  it('holds at most the window', () => {
    const ring = emptyRing();
    for (let t = 0; t < 400; t++) push(ring, s(t, t), t);
    expect(ring.samples.length).toBe(WINDOW_S);
    expect(ring.samples[0].t).toBe(100);
  });
});

describe('slots', () => {
  it('a gap is a gap: null slots where the sampler slept, values where it ran', () => {
    const now = 1000;
    const ring = seedRing([s(now - 40, 41), s(now - 39, 42, 2400), s(now - 10, 44), s(now, 45)], now);
    const out = slots(ring, now, 60);
    expect(out.t.length).toBe(60);
    expect(out.t[0]).toBe(now - 59);
    expect(out.decode[59]).toBe(45);
    expect(out.decode[49]).toBe(44);
    expect(out.decode[20]).toBe(42);
    expect(out.decode[19]).toBe(41);
    expect(out.decode.slice(21, 49).every((v) => v === null)).toBe(true);
    expect(out.prefill[20]).toBe(2400);
    expect(out.prefill.filter((v) => v != null)).toEqual([2400]);
  });
  it('fractional timestamps round to their second', () => {
    const ring = seedRing([s(99.6, 7)], 100);
    expect(slots(ring, 100, 5).decode).toEqual([null, null, null, null, 7]);
  });
  it('peak ignores nulls', () => {
    expect(peak([null, 3, null, 9, 4])).toBe(9);
    expect(peak([null, null])).toBeNull();
  });
});

const row = (p: Partial<LiveRequest>): LiveRequest => ({
  request_id: 'x',
  phase: 'decode',
  finish_reason: null,
  status: 200,
  model: 'm',
  client: { id: 'anon', kind: 'curl' },
  endpoint: 'chat',
  stream: true,
  thinking: false,
  temperature: 0,
  prompt_tokens: 10,
  cached_tokens: 0,
  forwarded_tokens: 10,
  tokens: 5,
  blocks: 1,
  tokens_per_block: 4,
  elapsed_ms: 100,
  queue_ms: 0,
  prompt_ms: 50,
  ttft_ms: 50,
  decode_ms: 50,
  prefill_tps: 200,
  decode_tps: 80,
  decode_tps_now: null,
  cache_source: 'none',
  max_tokens: 100,
  ended_ms_ago: null,
  ...p,
});

describe('rows', () => {
  it('orders decoding, prefilling, queued, then the finished newest first', () => {
    const rows = orderRows([row({ request_id: 'd2', phase: 'done', ended_ms_ago: 5000 }), row({ request_id: 'q', phase: 'queued' }), row({ request_id: 'd1', phase: 'done', ended_ms_ago: 500 }), row({ request_id: 'p', phase: 'prefill' }), row({ request_id: 'a', phase: 'decode' })]);
    expect(rows.map((r) => r.request_id)).toEqual(['a', 'p', 'q', 'd1', 'd2']);
  });
  it('phase words: never colour alone, a bad finish reads as failed', () => {
    expect(phaseWords({ phase: 'decode', finish_reason: null, status: 200 })).toEqual({ glyph: '●', word: 'decoding', cls: 'decode' });
    expect(phaseWords({ phase: 'prefill', finish_reason: null, status: 200 }).word).toBe('prefilling');
    expect(phaseWords({ phase: 'queued', finish_reason: null, status: 200 }).cls).toBe('queued');
    expect(phaseWords({ phase: 'done', finish_reason: 'stop', status: 200 })).toEqual({ glyph: '✓', word: 'stop', cls: 'done' });
    expect(phaseWords({ phase: 'done', finish_reason: 'error', status: 200 }).cls).toBe('failed');
    expect(phaseWords({ phase: 'done', finish_reason: 'refused', status: 503 }).word).toBe('refused');
    expect(phaseWords({ phase: 'done', finish_reason: null, status: 400 })).toEqual({ glyph: '✕', word: 'HTTP 400', cls: 'failed' });
  });
  it('the sum of the rows is the all-together number: 20 + 25 = 45', () => {
    expect(sumDecodeNow([row({ decode_tps_now: 20 }), row({ decode_tps_now: 25 }), row({ phase: 'done', decode_tps_now: 99 })])).toBe(45);
    expect(sumDecodeNow([row({ decode_tps_now: null })])).toBeNull();
  });
  it('formats ago and elapsed', () => {
    expect(agoShort(12_040)).toBe('12 s ago');
    expect(agoShort(125_000)).toBe('2 m ago');
    expect(agoShort(null)).toBe('');
    expect(elapsed(812)).toBe('0.8 s');
    expect(elapsed(12_400)).toBe('12.4 s');
    expect(elapsed(72_000)).toBe('1 m 12 s');
    expect(elapsed(null)).toBe('—');
  });
});
