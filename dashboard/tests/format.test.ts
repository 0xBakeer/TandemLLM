import { describe, expect, it } from 'vitest';
import { compact, exact, ms, duration, bytes, tps, pct, ago } from '../src/lib/format';

describe('number formatting', () => {
  it('compact notation keeps three significant figures', () => {
    expect(compact(0)).toBe('0');
    expect(compact(999)).toBe('999');
    expect(compact(1234)).toBe('1.2K');
    expect(compact(12345)).toBe('12.3K');
    expect(compact(123456)).toBe('123K');
    expect(compact(1_234_567)).toBe('1.2M');
    expect(compact(148_223_019, 2)).toBe('148M');
    expect(compact(4_493_007, 2)).toBe('4.49M');
    expect(compact(null)).toBe('—');
  });
  it('exact values are thousands-separated', () => {
    expect(exact(4493007)).toBe('4,493,007');
  });
  it('durations carry their unit', () => {
    expect(ms(702)).toBe('702 ms');
    expect(ms(2240.1)).toBe('2.24 s');
    expect(ms(5.3)).toBe('5.3 ms');
    expect(ms(72_000)).toBe('1 m 12 s');
    expect(duration(3811.4)).toBe('1 h 03 m');
    expect(duration(2 * 86400 + 4 * 3600)).toBe('2 d 4 h');
    expect(duration(48)).toBe('48 s');
  });
  it('bytes, tok/s, percent', () => {
    expect(bytes(7340032)).toBe('7.34 MB');
    expect(bytes(61.2e9)).toBe('61.20 GB');
    expect(tps(68.26)).toBe('68.3 tok/s');
    expect(tps(2105)).toBe('2105 tok/s');
    expect(pct(0.2385)).toBe('23.9 %');
  });
  it('relative time', () => {
    const now = Date.parse('2026-09-24T16:40:00Z');
    expect(ago('2026-09-24T16:39:48Z', now)).toBe('12 s ago');
    expect(ago('2026-09-24T16:10:00Z', now)).toBe('30 m ago');
    expect(ago('2026-09-23T16:40:00Z', now)).toBe('yesterday');
    expect(ago(null)).toBe('never');
  });
});
