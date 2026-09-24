import { describe, expect, it } from 'vitest';
import { buildHeatGrid, quantileThresholds, levelFor, streaks, yearWindow, bucketDayKey } from '../src/lib/heatmap';
import { addDays, dayKey, dayRange, dayStart, diffDays, hourKey, isLeapYear, isoLocalDayStart, localToUtc, tzOffsetMinutes, weekdayMon0 } from '../src/lib/time';
import type { UsageBucket } from '../src/api/types';

const TZ = 'Europe/Berlin';

function bucket(date: string, over: Partial<UsageBucket> = {}): UsageBucket {
  return {
    start: isoLocalDayStart(date, TZ),
    requests: 0,
    errors: 0,
    prompt_tokens: 0,
    cached_tokens: 0,
    completion_tokens: 0,
    reasoning_tokens: 0,
    total_tokens: 0,
    tool_call_requests: 0,
    decode_tps_p50: null,
    decode_tps_p90: null,
    ttft_ms_p50: null,
    ttft_ms_p90: null,
    prefill_tps_p50: null,
    tokens_per_block_mean: null,
    draft_acceptance: null,
    ...over,
  };
}

describe('local time in a zone', () => {
  it('day keys follow the zone, not UTC', () => {
    // 2026-09-24 22:30 UTC is 2026-09-25 00:30 in Berlin (CEST)
    const ms = Date.UTC(2026, 8, 24, 22, 30);
    expect(dayKey(ms, TZ)).toBe('2026-09-25');
    expect(dayKey(ms, 'UTC')).toBe('2026-09-24');
    expect(hourKey(ms, TZ)).toBe('2026-09-25T00');
  });

  it('offsets change across DST: +02:00 in summer, +01:00 in winter', () => {
    expect(tzOffsetMinutes(Date.UTC(2026, 6, 1, 12), TZ)).toBe(120);
    expect(tzOffsetMinutes(Date.UTC(2026, 0, 1, 12), TZ)).toBe(60);
  });

  it('local day start is DST-safe: the spring-forward day is 23 hours long', () => {
    // Europe/Berlin springs forward on 2026-03-29
    const a = dayStart('2026-03-29', TZ);
    const b = dayStart('2026-03-30', TZ);
    expect((b - a) / 3600000).toBe(23);
    // and falls back on 2026-10-25: 25 hours
    const c = dayStart('2026-10-25', TZ);
    const d = dayStart('2026-10-26', TZ);
    expect((d - c) / 3600000).toBe(25);
    expect(isoLocalDayStart('2026-03-29', TZ)).toBe('2026-03-29T00:00:00+01:00');
    expect(isoLocalDayStart('2026-03-30', TZ)).toBe('2026-03-30T00:00:00+02:00');
  });

  it('a wall-clock hour that does not exist resolves forward, one that repeats resolves once', () => {
    // 02:30 on the spring-forward day does not exist; the instant is 01:30 UTC either way
    const ms = localToUtc(2026, 3, 29, 2, 30, TZ);
    expect(dayKey(ms, TZ)).toBe('2026-03-29');
    expect(new Date(ms).toISOString()).toBe('2026-03-29T01:30:00.000Z');
  });

  it('calendar arithmetic ignores DST', () => {
    expect(addDays('2026-03-28', 2)).toBe('2026-03-30');
    expect(diffDays('2026-03-28', '2026-03-30')).toBe(2);
    expect(dayRange('2026-12-30', '2027-01-02')).toEqual(['2026-12-30', '2026-12-31', '2027-01-01', '2027-01-02']);
  });

  it('leap year 2028: 29 February exists and the year has 366 days', () => {
    expect(isLeapYear(2028)).toBe(true);
    expect(isLeapYear(2100)).toBe(false);
    expect(addDays('2028-02-28', 1)).toBe('2028-02-29');
    expect(dayRange('2028-01-01', '2028-12-31')).toHaveLength(366);
  });

  it('weekday Monday=0', () => {
    expect(weekdayMon0('2026-09-21')).toBe(0); // Monday
    expect(weekdayMon0('2026-09-27')).toBe(6); // Sunday
  });
});

describe('quantile colour scale', () => {
  it('five steps over four orders of magnitude, zeros empty', () => {
    const values = [0, 0, 1, 10, 100, 1000, 10000, 5, 50, 500, 5000, 0];
    const t = quantileThresholds(values);
    expect(t).toHaveLength(4);
    expect(levelFor(0, t)).toBe(0);
    expect(levelFor(1, t)).toBe(1);
    expect(levelFor(10000, t)).toBe(5);
    const levels = new Set(values.filter((v) => v > 0).map((v) => levelFor(v, t)));
    expect(levels).toEqual(new Set([1, 2, 3, 4, 5]));
  });

  it('no non-zero days: every cell is level 0', () => {
    const t = quantileThresholds([0, 0, 0]);
    expect(levelFor(0, t)).toBe(0);
  });
});

describe('the year grid', () => {
  const today = '2026-09-24'; // a Thursday
  const { from, to } = yearWindow(today);

  it('covers 365 days ending today, Monday first, ≤ 53 columns, and marks today', () => {
    expect(diffDays(from, to)).toBe(364);
    const g = buildHeatGrid([bucket('2026-09-18', { total_tokens: 1000, requests: 3 })], { from, to, metric: 'total_tokens', today });
    const inRange = g.cells.filter((c) => c.inRange);
    expect(inRange).toHaveLength(365);
    expect(g.cols).toBeLessThanOrEqual(53);
    expect(g.cols).toBeGreaterThanOrEqual(52);
    expect(inRange[0].date).toBe(from);
    expect(inRange[0].row).toBe(weekdayMon0(from));
    expect(g.cells.filter((c) => c.isToday)).toHaveLength(1);
    expect(g.cells.find((c) => c.isToday)!.date).toBe(today);
    expect(g.cells.find((c) => c.isToday)!.row).toBe(3); // Thursday
  });

  it('missing days are zero cells with no bucket; one day of use is level 5 of 5', () => {
    const g = buildHeatGrid([bucket('2026-09-18', { total_tokens: 1000, requests: 3 })], { from, to, metric: 'total_tokens', today });
    const d = g.cells.find((c) => c.date === '2026-09-18')!;
    expect(d.level).toBe(5);
    expect(d.value).toBe(1000);
    const z = g.cells.find((c) => c.date === '2026-09-17')!;
    expect(z.level).toBe(0);
    expect(z.bucket).toBeNull();
  });

  it('dense zero buckets from the API render as level 0 too', () => {
    const buckets = dayRange(from, to).map((d) => bucket(d));
    const g = buildHeatGrid(buckets, { from, to, metric: 'requests', today });
    expect(g.cells.filter((c) => c.inRange && c.level > 0)).toHaveLength(0);
  });

  it("the bucket start's first ten characters are the local day, across DST", () => {
    expect(bucketDayKey(bucket('2026-03-29'))).toBe('2026-03-29');
    expect(bucketDayKey(bucket('2026-10-25'))).toBe('2026-10-25');
  });

  it('metric switch recolours by requests', () => {
    const b = [bucket('2026-09-18', { total_tokens: 1, requests: 100 }), bucket('2026-09-19', { total_tokens: 1000, requests: 1 })];
    const byTok = buildHeatGrid(b, { from, to, metric: 'total_tokens', today });
    const byReq = buildHeatGrid(b, { from, to, metric: 'requests', today });
    expect(byTok.cells.find((c) => c.date === '2026-09-19')!.level).toBeGreaterThan(byTok.cells.find((c) => c.date === '2026-09-18')!.level);
    expect(byReq.cells.find((c) => c.date === '2026-09-18')!.level).toBeGreaterThan(byReq.cells.find((c) => c.date === '2026-09-19')!.level);
  });

  it('month labels start at the first column of each month', () => {
    const g = buildHeatGrid([], { from, to, metric: 'total_tokens', today });
    expect(g.months.length).toBeGreaterThanOrEqual(11);
    expect(g.months[g.months.length - 1].label).toBe('Sep');
  });

  it('a leap-year window (2028) has 366 in-range cells when asked for a full year', () => {
    const g = buildHeatGrid([], { from: '2028-01-01', to: '2028-12-31', metric: 'total_tokens', today: '2028-12-31' });
    expect(g.cells.filter((c) => c.inRange)).toHaveLength(366);
  });
});

describe('streaks', () => {
  const today = '2026-09-24';
  it('current streak counts back from today, longest is the longest run', () => {
    const active = ['2026-09-19', '2026-09-20', '2026-09-21', '2026-09-22', '2026-09-23', '2026-09-24', '2026-09-01', '2026-09-02', '2026-09-03', '2026-09-04', '2026-09-05', '2026-09-06', '2026-09-07'];
    const s = streaks(active.map((d) => bucket(d, { requests: 1 })), today);
    expect(s.current).toBe(6);
    expect(s.longest).toBe(7);
    expect(s.activeDays).toBe(13);
  });
  it('a quiet today keeps yesterday’s streak alive', () => {
    const s = streaks(['2026-09-22', '2026-09-23'].map((d) => bucket(d, { requests: 1 })), today);
    expect(s.current).toBe(2);
  });
  it('a gap resets the current streak', () => {
    const s = streaks(['2026-09-20', '2026-09-21'].map((d) => bucket(d, { requests: 1 })), today);
    expect(s.current).toBe(0);
    expect(s.longest).toBe(2);
  });
});
