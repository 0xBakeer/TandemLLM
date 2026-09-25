// The year heatmap's data shaping: 53 week columns × 7 rows (Monday first), dense day cells
// including zero days, a 5-step quantile scale over the non-zero days, month labels, streaks.
// Pure functions — unit-tested for zero days, DST weeks, leap years and the tz rule.

import type { UsageBucket } from '../api/types.ts';
import { addDays, dayKey, dayRange, diffDays, weekdayMon0 } from './time.ts';

export type HeatMetric = 'total_tokens' | 'completion_tokens' | 'requests';

export interface HeatCell {
  date: string; // 'YYYY-MM-DD'
  col: number; // week column
  row: number; // 0 = Monday … 6 = Sunday
  value: number; // metric value (0 for empty days)
  level: 0 | 1 | 2 | 3 | 4 | 5; // 0 = zero day, 1..5 = quantile step
  bucket: UsageBucket | null;
  isToday: boolean;
  inRange: boolean; // false for padding cells before `from` / after `to`
}

export interface HeatGrid {
  cells: HeatCell[];
  cols: number;
  months: { label: string; col: number }[];
  thresholds: number[]; // upper bounds of levels 1..4 (level 5 above the last)
  from: string;
  to: string;
  todayCol: number;
}

/** Local day key of a bucket's `start` (RFC 3339 with offset). The offset is the zone's own, so the
 * first ten characters are already the local date — no re-conversion, no DST surprise. */
export function bucketDayKey(b: UsageBucket): string {
  return b.start.slice(0, 10);
}

export function metricOf(b: UsageBucket | null, metric: HeatMetric): number {
  if (!b) return 0;
  return b[metric] ?? 0;
}

/**
 * Quantile thresholds over the non-zero values: five steps, computed at the 20/40/60/80 %
 * points of the sorted non-zero values. Values spread over orders of magnitude still fill every
 * step because the steps are rank-based, not linear.
 */
export function quantileThresholds(values: number[]): number[] {
  const nz = values.filter((v) => v > 0).sort((a, b) => a - b);
  if (nz.length === 0) return [0, 0, 0, 0];
  const q = (p: number) => nz[Math.min(nz.length - 1, Math.max(0, Math.ceil(p * nz.length) - 1))];
  return [q(0.2), q(0.4), q(0.6), q(0.8)];
}

export function levelFor(value: number, thresholds: number[]): HeatCell['level'] {
  if (value <= 0) return 0;
  if (value < thresholds[0]) return 1;
  if (value < thresholds[1]) return 2;
  if (value < thresholds[2]) return 3;
  if (value < thresholds[3]) return 4;
  return 5;
}

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

/**
 * Build the grid. `from`/`to` are local day keys (inclusive). Columns start on the Monday of
 * the week containing `from`; cells before `from` or after `to` are padding (inRange=false).
 * `buckets` may be sparse or dense — missing days are zero cells.
 */
export function buildHeatGrid(
  buckets: UsageBucket[],
  opts: { from: string; to: string; metric: HeatMetric; today: string },
): HeatGrid {
  const byDay = new Map<string, UsageBucket>();
  for (const b of buckets) byDay.set(bucketDayKey(b), b);

  const days = dayRange(opts.from, opts.to);
  const values = days.map((d) => metricOf(byDay.get(d) ?? null, opts.metric));
  const thresholds = quantileThresholds(values);

  const gridStart = addDays(opts.from, -weekdayMon0(opts.from));
  const gridEnd = addDays(opts.to, 6 - weekdayMon0(opts.to));
  const all = dayRange(gridStart, gridEnd);
  const cols = Math.ceil(all.length / 7);

  const cells: HeatCell[] = all.map((date, i) => {
    const inRange = date >= opts.from && date <= opts.to;
    const bucket = inRange ? (byDay.get(date) ?? null) : null;
    const value = inRange ? metricOf(bucket, opts.metric) : 0;
    return {
      date,
      col: Math.floor(i / 7),
      row: i % 7,
      value,
      level: inRange ? levelFor(value, thresholds) : 0,
      bucket,
      isToday: date === opts.today,
      inRange,
    };
  });

  // A month label at the first column whose Monday-row cell falls in a new month.
  const months: { label: string; col: number }[] = [];
  let lastMonth = '';
  for (let c = 0; c < cols; c++) {
    const monday = all[c * 7];
    const m = monday.slice(0, 7);
    if (m !== lastMonth) {
      if (monday >= opts.from || c === 0) {
        // skip a label that would collide with the previous one
        if (months.length === 0 || c - months[months.length - 1].col >= 3) {
          months.push({ label: MONTHS[Number(monday.slice(5, 7)) - 1], col: c });
        }
      }
      lastMonth = m;
    }
  }

  const todayIdx = all.indexOf(opts.today);
  return {
    cells,
    cols,
    months,
    thresholds,
    from: opts.from,
    to: opts.to,
    todayCol: todayIdx >= 0 ? Math.floor(todayIdx / 7) : cols - 1,
  };
}

/** Streaks over dense day buckets: current (ending today or yesterday) and longest run of days with requests > 0. */
export function streaks(buckets: UsageBucket[], today: string): { current: number; longest: number; activeDays: number } {
  const active = new Set<string>();
  for (const b of buckets) if (b.requests > 0) active.add(bucketDayKey(b));
  let longest = 0;
  let run = 0;
  let prev: string | null = null;
  for (const d of [...active].sort()) {
    run = prev && diffDays(prev, d) === 1 ? run + 1 : 1;
    longest = Math.max(longest, run);
    prev = d;
  }
  // Current: count back from today (a day without usage yet today keeps yesterday's streak alive).
  let current = 0;
  let cursor = active.has(today) ? today : addDays(today, -1);
  while (active.has(cursor)) {
    current++;
    cursor = addDays(cursor, -1);
  }
  return { current, longest, activeDays: active.size };
}

/** The 365-day window ending today (local), as day keys. */
export function yearWindow(today: string): { from: string; to: string } {
  return { from: addDays(today, -364), to: today };
}

/** Convenience: the local day key for an instant in a zone (re-exported for the views). */
export const localDay = dayKey;
