// The parallel view's logic (lib/parallel.ts): lanes keep their rows, backends keep their colours,
// the series merges and stacks, the mock's document validates against the contract, and the
// view's words for what a backend does not report.

import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import Ajv2020 from 'ajv/dist/2020';
import { describe, expect, it } from 'vitest';
import { MockParallel } from '../mock/parallel';
import { backendOrder, barScale, Colors, jointWords, mergeSeries, Slots, slotWord, stack, stateLine, stateTone, tokenBar, windowWord, type Lane, type Parallel } from '../src/lib/parallel';

const DIR = join(__dirname, '..', '..', 'docs', 'contract', 'dashboard-v1');
const schema = JSON.parse(readFileSync(join(DIR, 'parallel.schema.json'), 'utf8'));
const validate = new Ajv2020({ allErrors: true, strict: true }).compile(schema);
const example = JSON.parse(readFileSync(join(DIR, 'parallel.example.json'), 'utf8')) as Parallel;

function lane(id: string, extra: Partial<Lane> = {}): Lane {
  return { ...example.lanes[0], id, state: 'writing', elapsed_ms: 1000, ...extra };
}

describe('slots', () => {
  it('a lane keeps its row, a new one takes the first free row, the free tail is trimmed late', () => {
    const s = new Slots(5000, 3000);
    s.update([lane('a', { elapsed_ms: 3000 }), lane('b', { elapsed_ms: 2000 }), lane('c', { elapsed_ms: 1000 })], [], 0);
    expect(s.slots.map((x) => x.lane?.id)).toEqual(['a', 'b', 'c']);
    // b finishes: its "done" stays in row 1
    s.update([lane('a'), lane('c')], [lane('b', { state: 'done', ended_ms: 500 })], 1000);
    expect(s.slots.map((x) => x.lane?.id)).toEqual(['a', 'b', 'c']);
    expect(s.slots[1].lane?.state).toBe('done');
    // b's done leaves; d arrives into row 1, a and c do not move
    s.update([lane('a'), lane('c'), lane('d')], [], 2000);
    expect(s.slots.map((x) => x.lane?.id)).toEqual(['a', 'd', 'c']);
    // c leaves: row 2 stays as a free row until the trim time has passed
    s.update([lane('a'), lane('d')], [], 3000);
    expect(s.slots.length).toBe(3);
    expect(s.slots[2].lane).toBeNull();
    s.update([lane('a'), lane('d')], [], 7999);
    expect(s.slots.length).toBe(3);
    s.update([lane('a'), lane('d')], [], 8001);
    expect(s.slots.length).toBe(2);
  });
  it('a finished lane leaves its row after the hold time, though the server still lists it', () => {
    const s = new Slots(5000, 3000);
    s.update([lane('a')], [], 0);
    s.update([], [lane('a', { state: 'done', ended_ms: 1000 })], 1000);
    expect(s.slots[0].lane?.state).toBe('done');
    s.update([], [lane('a', { state: 'done', ended_ms: 3500 })], 3500);
    expect(s.slots[0].lane).toBeNull();
  });
  it('a done lane never takes a new row', () => {
    const s = new Slots();
    s.update([], [lane('x', { state: 'done' })], 0);
    expect(s.slots).toHaveLength(0);
  });
});

describe('colours, series, stack', () => {
  it('the engine is cobalt, the others keep the colour they got first', () => {
    const c = new Colors();
    c.seed(['vllm1', 'engine', 'ds']);
    expect(c.of('engine')).toBe('cobalt');
    const ds = c.of('ds');
    c.seed(['zz', 'ds', 'aa']);
    expect(c.of('ds')).toBe(ds);
    expect(new Set([c.of('ds'), c.of('vllm1'), c.of('zz'), c.of('aa')]).size).toBe(4);
  });
  it('merges new samples only and keeps five minutes', () => {
    const a = [{ t: 100, by: { x: 1 }, in_flight: 1 }];
    const m = mergeSeries(a, [{ t: 100, by: { x: 9 }, in_flight: 1 }, { t: 101, by: { x: 2 }, in_flight: 1 }]);
    expect(m.map((s) => s.by.x)).toEqual([1, 2]);
    expect(mergeSeries(m, [{ t: 500, by: {}, in_flight: 0 }]).map((s) => s.t)).toEqual([500]);
  });
  it('stacks in order, a missing backend adds 0', () => {
    const L = stack([{ t: 1, by: { a: 10, b: 5 }, in_flight: 2 }, { t: 2, by: { b: 7 }, in_flight: 1 }], ['a', 'b']);
    expect(L[0].bands).toEqual([[0, 10], [0, 0]]);
    expect(L[1].bands).toEqual([[10, 15], [0, 7]]);
    expect(backendOrder([{ t: 1, by: { z: 1, engine: 2 }, in_flight: 1 }], [])).toEqual(['engine', 'z']);
  });
});

describe('words and bars', () => {
  it('tones and lines', () => {
    expect(stateTone(lane('a', { state: 'waiting' }))).toBe('wait');
    expect(stateTone(lane('a', { state: 'tool_call' }))).toBe('tool');
    expect(stateTone(lane('a', { state: 'done', finish: 'error' }))).toBe('error');
    expect(stateLine(lane('a', { state: 'tool_call', tool: 'read_file' }))).toBe('Tool call: read_file');
    expect(stateLine(lane('a', { state: 'done', finish: 'tool_calls' }))).toBe('Done: tool calls');
  });
  it('a lane without a live count has no bar; the scale only grows in powers of two', () => {
    expect(tokenBar(lane('a', { tokens: null }), 512)).toBeNull();
    expect(tokenBar(lane('a', { tokens: 256, thinking_tokens: 128 }), 512)).toEqual({ think: 0.25, out: 0.25 });
    expect(barScale([lane('a', { tokens: 300 })])).toBe(512);
    expect(barScale([lane('a', { tokens: 100 })], 512)).toBe(512);
    expect(barScale([lane('a', { tokens: 3000 })], 512)).toBe(4096);
  });
});

describe('contract', () => {
  it("the server's example validates (written by tests/test_parallel.py from the real board)", () => {
    expect(validate(example), JSON.stringify(validate.errors)).toBe(true);
  });
  it("the mock's document validates, under load and idle", () => {
    const m = new MockParallel(3);
    const t0 = Date.parse('2026-10-05T00:00:00Z');
    let doc: unknown = null;
    let unreported = false;
    for (let i = 0; i < 60; i++) {
      doc = m.snapshot(t0 + i * 1000, null);
      unreported ||= (doc as Parallel).lanes.some((l) => l.tokens == null && l.state !== 'waiting' && l.state !== 'prefilling');
    }
    expect(validate(doc), JSON.stringify(validate.errors)).toBe(true);
    const d = doc as Parallel;
    expect(d.totals.in_flight).toBe(14);
    expect(unreported).toBe(true); // the unchecked backend, DeepSeek's call phase
    m.reset(0);
    const idle = m.snapshot(t0 + 60_000, null) as unknown as Parallel;
    expect(validate(idle)).toBe(true);
    expect(idle.lanes).toHaveLength(0);
  });
});

describe("the engine's sequence slots (server --parallel N)", () => {
  it('the mock with three engine slots validates, its engine lanes carry their slot, the slots doc is there', () => {
    const m = new MockParallel(5);
    m.engineSlots = 3;
    m.reset(10);
    const t0 = Date.parse('2026-10-05T00:00:00Z');
    let doc: Parallel | null = null;
    let most = 0;
    for (let i = 0; i < 90; i++) {
      doc = m.snapshot(t0 + i * 1000, null) as unknown as Parallel;
      expect(validate(doc), JSON.stringify(validate.errors)).toBe(true);
      const eng = doc.lanes.filter((l) => l.backend === 'engine');
      most = Math.max(most, eng.length);
      for (const l of eng) expect(l.slot).toBeGreaterThanOrEqual(0);
      expect(new Set(eng.map((l) => l.slot)).size).toBe(eng.length); // one request a slot
    }
    expect(most).toBeGreaterThan(1);
    expect(doc!.engine_slots!.n).toBe(3);
    expect(doc!.engine_slots!.slots.map((s) => s.window)).toEqual([262144, 65536, 65536]);
  });
  it('without slots the document has neither key (the board a single-sequence server sends)', () => {
    const d = new MockParallel(2).snapshot(Date.parse('2026-10-05T00:00:00Z'), null) as unknown as Parallel;
    expect('engine_slots' in d).toBe(false);
    expect(d.lanes.every((l) => !('slot' in l))).toBe(true);
  });
  it('the words', () => {
    expect(slotWord({ index: 1, window: 65536, busy: false, phase: '' })).toBe('free');
    expect(slotWord({ index: 1, window: 65536, busy: true, phase: 'decode' })).toBe('decoding');
    expect(slotWord({ index: 1, window: 65536, busy: true, phase: '' })).toBe('busy');
    expect(windowWord(262144)).toBe('256k');
    expect(windowWord(512)).toBe('512');
    expect(jointWords({ '4': 3, '2': 10, '3': 0 })).toBe('2-way 10 · 4-way 3');
    expect(jointWords(null)).toBe('none yet');
  });
});
