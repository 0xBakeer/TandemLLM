// VIS-40 / VIS-41: the live prefill and decode rates without the warm-up. The tile and the banner
// read the same number, "—" when nothing runs, the finished rows keep their end-to-end numbers,
// and an older server gets the figures it always had. Pure functions and the mock's registry.
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import Ajv2020 from 'ajv/dist/2020.js';
import { nowLine } from '../src/lib/activity';
import { decodeFigure, hasLiveRates, mostlyCachedWords, peak, prefillFigure, push, rateText, seedRing, slots, warmingWords } from '../src/lib/live';
import type { Live, LiveActivity, LiveRequest, LiveWindows } from '../src/api/types';
import { MockLive, mostlyCached, prefillLive, WINDOWS } from '../mock/live.ts';
import { makeRow, rng } from '../mock/generate.ts';

const DIR = join(__dirname, '..', '..', 'docs', 'contract', 'dashboard-v1');
const ajv = new Ajv2020({ allErrors: true, strict: true });
const schema = ajv.compile(JSON.parse(readFileSync(join(DIR, 'live.schema.json'), 'utf8')));
function valid(l: Live): Live {
  if (!schema(l)) throw new Error(ajv.errorsText(schema.errors, { separator: '\n' }));
  return l;
}

const NOW = Date.parse('2026-09-30T15:02:45Z');
const W: LiveWindows = { ...WINDOWS };

function mock(kind: string, at: number, opts: { liveRates?: boolean } = {}): MockLive {
  const r = rng(40);
  const live = new MockLive(r, 812);
  if (opts.liveRates === false) live.liveRates = false;
  live.rates((forced) => makeRow(r, NOW, false, forced), kind, NOW, at);
  return live;
}

/** What the page shows: the tile's number and the banner's rate, as text. */
function shown(l: Live) {
  const rows = l.requests;
  const pf = hasLiveRates(l) ? prefillFigure(l, rows) : null;
  const df = hasLiveRates(l) ? decodeFigure(l, rows) : null;
  const n = nowLine(l.engine, rows, l.windows);
  return { prefill: pf, decode: df, tile: pf ? rateText(pf.value) : null, banner: n?.numbers ?? [], headline: n?.headline ?? '' };
}

const pre = (l: Live, id: string) => l.requests.find((x) => x.request_id.startsWith(id))!;

describe('scenario 1 + 2: the running prefill drives the headline, and the tile equals the banner', () => {
  it('B prefills at 743 tok/s; A finished at 433 end to end: the tile says 743', () => {
    const l = valid(mock('prefill-now', 19_000).snapshot(NOW, true));
    const s = shown(l);
    expect(l.now.prefill_tps).toBeCloseTo(433.19, 2); // 1.0: the last finished request, unchanged
    expect(l.now.prefill_live_tps).toBe(743);
    expect(s.tile).toBe('743');
    expect(s.banner[0]).toBe('743 tok/s');
    expect(s.prefill!.sub).toMatch(/^prefilling 12,288 of 107,604 · 11 %$/);
    const a = pre(l, 'chatcmpl-1f2552d48');
    expect(a.phase).toBe('done');
    expect(a.prefill_tps).toBeCloseTo(433.19, 2);
    expect(a.mostly_cached).toBe(true);
  });
  it('20 consecutive events at four a second: the tile never differs from the banner, never 433', () => {
    const live = mock('prefill-now', 19_000);
    for (let i = 0; i < 20; i++) {
      const t = NOW + i * 250;
      if (i % 4 === 0) live.tick(t);
      const s = shown(valid(live.snapshot(t, i === 0)));
      expect(`${s.tile} tok/s`).toBe(s.banner[0]);
      expect(s.tile).not.toBe('433');
    }
  });
});

describe('scenario 3 + 4 + 5 + 15: the prefill now-rate', () => {
  it('3: chunks at 900, then at 600: the headline is about 600', () => {
    const l = mock('now-rate', 200 + 3000 + 10 * (1024 / 900) * 1000 + 6 * (1024 / 600) * 1000 + 50).snapshot(NOW);
    expect(l.now.prefill_live_tps).toBeCloseTo(600, 0);
  });
  it('4: the 3.0 s warm-up chunk is left out: "—" while it runs, 731 from chunks 2 and 3', () => {
    let s = shown(mock('warm-chunk', 200 + 2000).snapshot(NOW));
    expect(s.tile).toBe('—');
    expect(s.prefill!.sub).toMatch(/^warming up · /);
    expect(s.banner[0]).toBe('warming up');
    s = shown(mock('warm-chunk', 200 + 3000 + 2 * 1400 + 50).snapshot(NOW));
    expect(s.tile).toBe('731');
    expect(s.banner[0]).toBe('731 tok/s');
  });
  it('5: 900 new tokens is one chunk: never a live rate, the finished row keeps timings', () => {
    const live = mock('one-chunk', 0);
    for (let t = NOW; t < NOW + 4000; t += 500) expect(shown(live.snapshot(t)).tile).toBe('—');
    const done = live.snapshot(NOW + 12_000);
    const c = pre(done, 'chatcmpl-c0000');
    expect(c.phase).toBe('done');
    expect(c.prefill_tps).toBeCloseTo(900 / 4, 1);
  });
  it('15: a single call shows "—" with the tokens and the elapsed time', () => {
    const s = shown(mock('single-call', 12_100).snapshot(NOW));
    expect(s.tile).toBe('—');
    // VIS-76: the time in the state, the Now card's clock (12.1 s since arrival, 0.1 s of it queued)
    expect(s.prefill!.sub).toBe('prefilling 4,000 tokens · 12.0 s');
  });
});

describe('scenario 6 + 7 + 10: decode from token 50', () => {
  it('6: 23 tokens = warming up (23 of 50 tokens), then the rate from the anchor', () => {
    const live = mock('decode-warmup', 1750);
    let l = live.snapshot(NOW);
    let s = shown(l);
    expect(s.decode!.value).toBeNull();
    expect(s.decode!.sub).toBe('warming up (23 of 50 tokens)');
    expect(s.banner[0]).toBe('warming up (23 of 50 tokens)');
    for (let i = 1; i <= 4 * 5; i++) {
      const t = NOW + i * 250;
      if (i % 4 === 0) live.tick(t);
      l = live.snapshot(t);
    }
    s = shown(valid(l));
    const d = pre(l, 'chatcmpl-d0000');
    expect(d.decode_warmup!.reached).toBe(true);
    expect(s.decode!.value).toBeCloseTo(20, -1);
    expect(s.decode!.sub).toBe('1 decoding · last 2 s');
    expect(s.banner[0]).toMatch(/^\d+\.\d tok\/s$/);
  });
  it('7: a 30-token reply never shows a live decode rate; its row keeps the end-to-end rate', () => {
    const live = mock('short-reply', 0);
    for (let i = 0; i <= 4 * 4; i++) {
      const t = NOW + i * 250;
      if (i % 4 === 0) live.tick(t);
      const l = live.snapshot(t);
      expect(l.now.decode_live_tps ?? null).toBeNull();
    }
    const e = pre(live.snapshot(NOW + 4000), 'chatcmpl-e0000');
    expect(e.phase).toBe('done');
    expect(e.decode_tps).not.toBeNull();
    expect(e.decode_live_avg_tps).toBeNull();
    expect(e.decode_warmup).toEqual({ tokens: 30, reached: false });
  });
  it('10: nothing decoding', () => {
    const live = new MockLive(rng(1), 0);
    const l = live.snapshot(NOW);
    expect(decodeFigure(l as Live & { windows: LiveWindows }, l.requests)).toMatchObject({ value: null, sub: 'nothing decoding', caption: 'from token 50' });
  });
});

describe('scenario 8: the end-to-end numbers stay where they were', () => {
  it('the finished row keeps prefill_tps and decode_tps, the windowed ones are extra fields', () => {
    const l = mock('prefill-now', 19_000).snapshot(NOW);
    const a = pre(l, 'chatcmpl-1f2552d48');
    expect(a.prefill_tps).toBeCloseTo(2571 / 5.935, 1);
    expect(a.activity!.prefill!.tps_live_avg).toBeCloseTo((195_083 - 193_536) / 1.935, 0);
    expect(a.decode_tps).not.toBeNull();
  });
});

describe('scenario 9 + 11 + 12 + 13 + 14', () => {
  it('9: nothing prefilling: "—" and "nothing prefilling", no greyed last value', () => {
    const live = mock('prefill-now', 0);
    live.clear();
    const l = live.snapshot(NOW + 30_000);
    const s = shown(l);
    expect(s.tile).toBe('—');
    expect(s.prefill!.sub).toBe('nothing prefilling');
  });
  it('11: one live sample a second during the prefill, and the peak comes from them', () => {
    const live = mock('prefill-now', 4000);
    for (let i = 1; i <= 20; i++) live.tick(NOW + i * 1000);
    const l = live.snapshot(NOW + 20_000, true);
    const ring = seedRing(l.history, NOW / 1000 + 20);
    const sl = slots(ring, NOW / 1000 + 20);
    const got = sl.prefillLive.filter((v) => v != null);
    expect(got.length).toBeGreaterThanOrEqual(18);
    expect(peak(sl.prefillLive)).toBe(743);
    expect(sl.prefill.filter((v) => v != null).length).toBeLessThanOrEqual(1); // 1.0: one dot per finish
  });
  it('12: two prefills at once: 743 + 510 = 1,253 in the tile, both in the subtitle', () => {
    const l = valid(mock('two-prefills', 12_000).snapshot(NOW));
    const s = shown(l);
    expect(l.now.prefill_live_count).toBe(2);
    expect(s.tile).toBe('1,253');
    expect(s.prefill!.sub).toBe('2 prefilling · 743 + 510');
  });
  it('13: a fully cached prompt: no rate, no sample, "fully cached, no prefill"', () => {
    const live = mock('fully-cached', 2000);
    live.tick(NOW);
    live.tick(NOW + 1000);
    const l = valid(live.snapshot(NOW + 1000, true));
    expect(shown(l).prefill!.sub).toBe('fully cached, no prefill (195,083 tokens)');
    expect(l.now.prefill_live_tps).toBeNull();
    expect(l.history!.every((x) => x.prefill_live_tps == null && x.prefill_tps == null)).toBe(true);
  });
  it('14: mostly cached is labelled and left out of the samples, so out of the peak', () => {
    const live = mock('mostly-cached', 100 + 4000 + 1200 + 300);
    live.tick(NOW);
    live.tick(NOW + 1000);
    const l = valid(live.snapshot(NOW + 1000, true));
    const i = pre(l, 'chatcmpl-10000');
    expect(i.mostly_cached).toBe(true);
    expect(mostlyCachedWords(i)).toBe('mostly cached (2,571 new of 195,083)');
    expect(shown(l).prefill!.sub).toBe('mostly cached (2,571 new of 195,083)');
    expect(l.now.prefill_live_tps).toBeCloseTo(853.33, 1); // the headline shows it: it is the speed now
    expect(l.history!.every((x) => x.prefill_live_tps == null)).toBe(true);
    expect([mostlyCached(900, 0), mostlyCached(100_000, 80_000), mostlyCached(20_000, 17_000), mostlyCached(10, 10)]).toEqual([false, false, true, false]);
  });
});

describe('scenario 16: an older server', () => {
  it('a 1.1 server without the live fields: the 1.0 figures, no captions, no error', () => {
    const l = valid(mock('prefill-now', 19_000, { liveRates: false }).snapshot(NOW));
    expect(hasLiveRates(l)).toBe(false);
    expect(l.now.prefill_live_tps).toBeUndefined();
    const n = nowLine(l.engine, l.requests, l.windows)!;
    expect(n.numbers[0]).toMatch(/tok\/s$/); // tps_now of the last chunk, as before
  });
  it('the recorded box messages (from before VIS-40) have no windows and keep their Now line', () => {
    const fx = JSON.parse(readFileSync(join(__dirname, '..', '..', 'tests', 'fixtures', 'live-1.1-box.json'), 'utf8')) as { messages: Record<string, Live> };
    for (const m of Object.values(fx.messages)) {
      expect(hasLiveRates(m)).toBe(false);
      expect(nowLine(m.engine, m.requests, m.windows)).toEqual(nowLine(m.engine, m.requests));
    }
  });
});

describe('the pieces', () => {
  it('rateText prints what the tile and the banner print', () => {
    expect([rateText(743.02), rateText(1253), rateText(43.21), rateText(null), rateText(99.95)]).toEqual(['743', '1,253', '43.2', '—', '100.0']);
  });
  it('prefillLive mirrors server/activity.py: the warm-up mark, the 2 s window, the last two marks', () => {
    const m = [
      { done: 1024, t: 3000 },
      { done: 2048, t: 4400 },
      { done: 3072, t: 5800 },
    ];
    expect(prefillLive(m.slice(0, 1), 1, 2, null)).toEqual({ tps_live: null, tps_live_avg: null, reached: true });
    expect(prefillLive(m, 1, 2, null).tps_live).toBe(731.43);
    expect(prefillLive(m, 1, 5, null).tps_live).toBe(731.43);
    expect(prefillLive([], 1, 2, null).reached).toBe(false);
    expect(prefillLive(m, 0, 10, { done: 0, t: 0 }).tps_live).toBeCloseTo(3072 / 5.8, 1);
  });
  it('warming words cap at the window', () => {
    const r = { decode_warmup: { tokens: 52, reached: false } } as LiveRequest;
    expect(warmingWords(r, W)).toBe('warming up (50 of 50 tokens)');
  });
  it('the ring carries the live series; an older sample reads as a gap', () => {
    const ring = seedRing([], 100);
    push(ring, { t: 98, decode_tps: 40, prefill_tps: null, running: 1, waiting: 0, tokens: 10, decode_live_tps: 41, prefill_live_tps: 700 }, 100);
    push(ring, { t: 99, decode_tps: 40, prefill_tps: 433, running: 1, waiting: 0, tokens: 50 }, 100);
    const sl = slots(ring, 100, 3);
    expect(sl.decodeLive).toEqual([41, null, null]);
    expect(sl.prefillLive).toEqual([700, null, null]);
    expect(sl.prefill).toEqual([null, 433, null]);
  });
  it('the Now line: the prefill banner reads tps_live on a new server, tps_now on an old one', () => {
    const base = pre(mock('prefill-now', 19_000).snapshot(NOW), 'chatcmpl-6294');
    const p = base.activity!.prefill!;
    const eng = { state: 'busy' as const, label: 'x', since_ms: 1, model: null, version: null, draining: false, kv: null, memory: null, store: null, waiting_for_client: null };
    expect(nowLine(eng, [base], W)!.numbers[0]).toBe('743 tok/s');
    const old = { ...base, activity: { ...base.activity!, prefill: { ...p, tps_live: undefined, tps_now: 810.2 } } as LiveActivity };
    expect(nowLine(eng, [old], null)!.numbers[0]).toBe('810 tok/s');
  });
});
