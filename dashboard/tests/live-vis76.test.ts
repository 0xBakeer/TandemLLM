// VIS-76: the Perf tab's live readings against the 0.6.x feed (2026-10-10, Khaled on a phone).
// A prefill that reports one mark per 8,192-row window shows its progress between marks as an
// estimate, said to be one; the tile and the card count the same total on the same clock; the
// five-minute charts have no holes from the sampler's 1.007 s second; a decode that stands still
// beside a prefill reads as a stall; the Parallel view's words match its legend and know the
// lanes held for memory. Pure functions, the mock's registry and the mock's parallel fixture.
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import Ajv2020 from 'ajv/dist/2020.js';
import { nowLine } from '../src/lib/activity';
import { clockSkew, decodeFigure, emptyFoot, emptyRing, hasLiveRates, prefillEstimate, prefillFigure, push, rateText, slots } from '../src/lib/live';
import { inFlightWords, stallLine, stateLine, stateTone, type Lane, type Parallel } from '../src/lib/parallel';
import type { Live, LiveEngine, LiveRequest, LiveSample } from '../src/api/types';
import { MockLive, WINDOWS } from '../mock/live.ts';
import { MockParallel } from '../mock/parallel.ts';
import { makeRow, rng } from '../mock/generate.ts';

const DIR = join(__dirname, '..', '..', 'docs', 'contract', 'dashboard-v1');
const ajv = new Ajv2020({ allErrors: true, strict: true });
const liveSchema = ajv.compile(JSON.parse(readFileSync(join(DIR, 'live.schema.json'), 'utf8')));
const parSchema = ajv.compile(JSON.parse(readFileSync(join(DIR, 'parallel.schema.json'), 'utf8')));

const NOW = Date.parse('2026-10-10T08:30:36Z');

function windowed(at: number): Live {
  const r = rng(76);
  const live = new MockLive(r, 812);
  live.rates((forced) => makeRow(r, NOW, false, forced), 'exl3-window', NOW, at);
  const l = live.snapshot(NOW);
  if (!liveSchema(l)) throw new Error(ajv.errorsText(liveSchema.errors, { separator: '\n' }));
  return l;
}

function shown(l: Live) {
  const rows = l.requests;
  const p = rows.find((x) => x.phase === 'prefill')!;
  return { row: p, tile: hasLiveRates(l) ? prefillFigure(l, rows) : null, now: nowLine(l.engine, rows, l.windows)! };
}

describe('a prefill with one mark a window (EXL3, 8,192 rows ~10.6 s apart)', () => {
  it('before the first mark: the restored rows fill the bar, the tile counts the served total on the card\'s clock', () => {
    const { row, tile, now } = shown(windowed(7700));
    expect(row.prompt_tokens).toBe(34_487);
    expect(row.hidden_tokens).toBe(10_071);
    expect(row.activity!.prefill!.total).toBe(44_558);
    expect(now.headline).toBe('Prefilling 44,558 tokens');
    // the bar: 10,240 restored of 44,558, measured; no estimate without a mark
    expect(now.progress).toBeCloseTo(10_240 / 44_558, 4);
    expect(now.estimate).toBeNull();
    expect(now.numbers).toEqual(['warming up', '10,240 cached']);
    // the tile: the card's total (not the client's 34,487) and the time in the state (not since arrival)
    expect(tile!.sub).toBe('warming up · prefilling 44,558 tokens · 5.9 s');
  });

  it('just after the first mark: measured figures, no estimate yet', () => {
    const { now, row } = shown(windowed(12_400 + 500));
    expect(now.headline).toBe('Prefilling 18,432 of 44,558 (41 %)');
    expect(prefillEstimate(row).stage).toBe('measuring');
    expect(now.estimate).toBeNull();
    expect(now.progress).toBeCloseTo(18_432 / 44_558, 4);
  });

  it('between two marks: the estimate moves at the measured rate, said to be one; the ETA counts down', () => {
    const a = shown(windowed(12_000 + 7000)); // 19 s in: 6.6 s since the mark at 12.4 s
    const b = shown(windowed(12_000 + 9000)); // 21 s in
    // the headline stays the measured one, the bar's solid part too
    expect(a.now.headline).toBe('Prefilling 18,432 of 44,558 (41 %)');
    expect(b.now.headline).toBe(a.now.headline);
    expect(a.now.progress).toBeCloseTo(b.now.progress!, 6);
    // 18,432 + 772.8 tok/s x 6.6 s = 23,532 of 44,558 = 53 %; two seconds later 56 %
    expect(a.now.estimate).toBeCloseTo((18_432 + ((18_432 - 10_240) / 10.6) * 6.6) / 44_558, 2);
    expect(b.now.estimate!).toBeGreaterThan(a.now.estimate!);
    expect(a.now.numbers).toEqual(['≈ 53 % est.', 'about 27 s left', '10,240 cached']);
    expect(b.now.numbers).toEqual(['≈ 56 % est.', 'about 25 s left', '10,240 cached']);
    expect(a.now.sinceMarkMs).toBeCloseTo(6600, -1);
    expect(a.tile!.sub).toBe('measuring · prefilling 18,432 of 44,558 · 41 % (≈ 53 % est.)');
  });

  it('the estimate never goes past one more window, nor past the total', () => {
    const { row } = shown(windowed(12_400 + 10_000)); // 10 s past the mark, the window ends at 10.57 s
    const e = prefillEstimate(row);
    expect(e.estPct!).toBeLessThanOrEqual((18_432 + 8192) / 44_558 + 1e-9);
    const late = { ...row, elapsed_ms: row.elapsed_ms! + 120_000 };
    expect(prefillEstimate(late).estPct!).toBeLessThanOrEqual((18_432 + 8192) / 44_558 + 1e-9);
    const near = { ...row, activity: { ...row.activity!, prefill: { ...row.activity!.prefill!, done: 43_008, pct: 96.5 } } } as LiveRequest;
    expect(prefillEstimate({ ...near, elapsed_ms: near.elapsed_ms! + 60_000 }).estPct!).toBeLessThanOrEqual(1);
  });

  it('two marks: the live rate is there, the tile and the card say it', () => {
    const { now, tile } = shown(windowed(12_400 + 10_570 + 300));
    expect(rateText(tile!.value)).toBe('775');
    expect(now.numbers[0]).toBe('775 tok/s');
  });

  it('marks a second apart (chunk-major) get no estimate', () => {
    const p = { done: 36_864, total: 48_210, cached: 12_288, pct: 76.5, tps_now: 2310, tps_avg: 1822.6, eta_ms: 4900, progress: 'chunked' as const, at_ms: 4000, tps_live: 2310, tps_live_avg: 2000, warmup: { chunks: 24, reached: true } };
    const r = { phase: 'prefill', elapsed_ms: 5000, activity: { prefill: p } } as unknown as LiveRequest;
    expect(prefillEstimate(r).estPct).toBeNull();
    expect(prefillEstimate(r).etaMs).toBe(3900);
  });
});

const sample = (t: number, d: number | null): LiveSample => ({ t, decode_tps: d, prefill_tps: null, running: 1, waiting: 0, tokens: 0, decode_live_tps: d, prefill_live_tps: null });

describe('the five-minute charts', () => {
  it('a sampler second of 1.007 s leaves no hole (rounding skipped a slot every ~140 s)', () => {
    const now = 1_791_621_700.4;
    const ring = emptyRing();
    for (let k = 299; k >= 0; k--) push(ring, sample(now - k * 1.007, 50), now);
    const s = slots(ring, now);
    const filled = s.decodeLive.map((v) => v != null);
    const first = filled.indexOf(true);
    expect(filled.slice(first).every(Boolean)).toBe(true);
    expect(s.decodeLive[s.decodeLive.length - 1]).toBe(50);
  });
  it('a real pause stays a hole, and the newest sample is at the right edge', () => {
    const now = 1000.2;
    const ring = emptyRing();
    for (const t of [980.1, 981.1, 982.1, 990.2, 991.2, 1000.2]) push(ring, sample(t, t), now);
    const s = slots(ring, now);
    const at = (t: number) => s.t.indexOf(t);
    expect(s.decodeLive[at(1000)]).toBe(1000.2);
    expect(s.decodeLive[at(991)]).toBe(991.2);
    expect(s.decodeLive[at(982)]).toBe(982.1);
    expect(s.decodeLive.slice(at(983), at(990)).every((v) => v == null)).toBe(true);
    expect(s.decodeLive.slice(at(992), at(1000)).every((v) => v == null)).toBe(true);
  });
  it('a server clock a little ahead still puts its newest sample on the chart', () => {
    const ring = emptyRing();
    push(ring, sample(1001.4, 7), 1001.4);
    expect(slots(ring, 1000.2).decodeLive.at(-1)).toBe(7);
    expect(clockSkew('2026-10-10T08:30:37.500Z', Date.parse('2026-10-10T08:30:36.000Z'))).toBeCloseTo(1.5, 6);
    expect(clockSkew(undefined)).toBe(0);
  });
  it('the empty foot: "no samples yet" only without samples; with samples and no value, none in 5 min', () => {
    const ring = emptyRing();
    expect(emptyFoot(ring, 1000)).toBe('no samples yet');
    push(ring, sample(990, null), 1000);
    expect(emptyFoot(ring, 1000)).toBe('none in 5 min');
    expect(emptyFoot(ring, 1400)).toBe('no samples yet');
  });
});

const engine: LiveEngine = { state: 'busy', label: 'Thinking', since_ms: 1000, model: 'TandemLLM', version: '0.6.1', draining: false, kv: null, memory: null, store: null, waiting_for_client: null } as unknown as LiveEngine;
function dec(id: string, now: number | null, state: 'thinking' | 'writing' = 'thinking'): LiveRequest {
  return {
    request_id: id, phase: 'decode', client: { id: 'k', kind: 'open-webui' }, tokens: 703, decode_live_now_tps: now, decode_live_avg_tps: 6, decode_warmup: { tokens: 702, reached: true },
    activity: { state, label: state === 'thinking' ? 'Thinking' : 'Writing', since_ms: 1000, client: { connected: true, silent_ms: 0, gone_ms: null }, decode: { tokens: 703, thinking_tokens: 703, content_tokens: 0, tool_tokens: 0, tps_now: now, tps_avg: 6 } },
  } as unknown as LiveRequest;
}
const prefilling = { request_id: 'p', phase: 'prefill', client: { id: 'k', kind: 'open-webui' }, tokens: 0, activity: { state: 'prefilling', label: 'Prefilling 44,558 tokens', since_ms: 500, prefill: { done: 10_240, total: 44_558, cached: 10_240, pct: null, progress: 'chunked', at_ms: null, eta_ms: null, tps_live: null, warmup: { chunks: 0, reached: false } } } } as unknown as LiveRequest;
const windows = { ...WINDOWS };

describe('several requests at once', () => {
  it('the Now line\'s rate is one request\'s share of the tile\'s sum, said so', () => {
    const n = nowLine(engine, [dec('a', 4.9), dec('b', 8.1, 'thinking'), dec('c', 4.2, 'thinking')], windows)!;
    expect(n.numbers[0]).toBe('4.9 of 17.2 tok/s');
    const alone = nowLine(engine, [dec('a', 4.9)], windows)!;
    expect(alone.numbers[0]).toBe('4.9 tok/s');
  });
  it('a request whose tokens stand still beside a prefill is stalled behind it', () => {
    const n = nowLine(engine, [dec('a', 0), dec('b', 0), prefilling], windows)!;
    expect(n.numbers[0]).toBe('stalled behind a prefill');
    expect(nowLine(engine, [dec('a', 0)], windows)!.numbers[0]).toBe('stalled, no token in the last 2 s');
  });
  it('the decode tile says the stall instead of a silent 0', () => {
    const l = { counts: { decoding: 3, prefilling: 1 }, now: { decode_live_tps: 0, decode_warming: 0 }, windows } as unknown as Live & { windows: typeof windows };
    expect(decodeFigure(l, []).sub).toBe('3 decoding · stalled behind a prefill');
    const ok = { ...l, now: { decode_live_tps: 17.2, decode_warming: 0 } } as typeof l;
    expect(decodeFigure(ok, []).sub).toBe('3 decoding · last 2 s');
  });
});

describe('the Parallel view on the 0.6.x shape (mock fixture "stall")', () => {
  const mp = new MockParallel();
  mp.fixture = 'stall';
  const d = mp.snapshot(NOW, null) as unknown as Parallel;
  it('the fixture validates against the contract', () => {
    if (!parSchema(d)) throw new Error(ajv.errorsText(parSchema.errors, { separator: '\n' }));
  });
  it('a lane held for memory is waiting, not writing', () => {
    const m = d.lanes.find((l) => l.state === 'waiting_memory')!;
    expect(stateTone(m)).toBe('wait');
    expect(stateLine(m)).toBe('Waiting for memory');
  });
  it('the In flight tile speaks the legend\'s words (the server\'s waiting counts the prefill too)', () => {
    expect(d.totals.waiting).toBe(2);
    expect(inFlightWords(d.lanes, d.totals.decoding)).toBe('1 waiting · 1 prefilling · 3 decoding');
  });
  it('an aggregate 0 with lanes decoding is a stall, said with the prefill it waits behind', () => {
    expect(stallLine(d.totals, d.lanes)).toBe('3 decoding, stalled behind a prefill (41 % of 44,558)');
    const noPre = d.lanes.filter((l) => l.state !== 'prefilling');
    expect(stallLine(d.totals, noPre)).toBe('3 decoding, no token in the last 2 s');
    expect(stallLine({ ...d.totals, decode_tps: 17.2 }, d.lanes)).toBeNull();
    expect(stallLine({ ...d.totals, decode_tps: null }, d.lanes)).toBeNull();
    expect(stallLine({ decode_tps: 0, decoding: 0 }, [] as Lane[])).toBeNull();
  });
});
