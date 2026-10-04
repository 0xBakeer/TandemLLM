import { describe, expect, it } from 'vitest';
import { isReplay } from '../src/lib/cache-source';
import { laneTitle, laneWord } from '../src/lib/lane';
import { robustLimit } from '../src/charts/scatter';

describe('replays (0.3.26-test)', () => {
  it('the response cache and a summary written ahead are replays, every prefill is not', () => {
    expect(isReplay('response')).toBe(true);
    expect(isReplay('prepared')).toBe(true);
    for (const s of ['resident', 'end', 'continuation', 'prefix', 'session', 'disk', 'spill', 'none', null, undefined]) {
      expect(isReplay(s as string | null | undefined)).toBe(false);
    }
  });
});

describe('the robust y axis', () => {
  it('a lone replay-sized outlier does not set the scale', () => {
    // 2026-10-04 13:08: one 160,500 tok/s point among real decodes of 30-60 tok/s
    const ys = [...Array.from({ length: 60 }, (_, i) => 30 + (i % 30)), 160_500];
    const lim = robustLimit(ys);
    expect(lim).toBeLessThan(200);
    expect(ys.filter((v) => v > lim)).toEqual([160_500]);
  });
  it('an ordinary spread keeps every point on the axis', () => {
    const ys = Array.from({ length: 40 }, (_, i) => 20 + i * 2);
    expect(ys.every((v) => v <= robustLimit(ys))).toBe(true);
  });
  it('a handful of points is never clipped', () => {
    expect(robustLimit([10, 20, 160_500])).toBe(Infinity);
  });
});

describe('the side lane words', () => {
  it('reads as served beside a turn, never as a refusal', () => {
    expect(laneWord('decode')).toBe('side lane');
    expect(laneWord('pending')).toBe('side lane, waiting');
    expect(laneTitle('decode')).toContain('went on unchanged');
    expect(laneTitle('self')).toContain('no turn running');
  });
});
