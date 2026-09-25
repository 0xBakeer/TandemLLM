// Local-day arithmetic in an IANA time zone, DST-safe. Day keys are 'YYYY-MM-DD' local dates.
// Both the app (heatmap, ranges) and the mock aggregation use these, so both sides agree on
// what a "day" is — the contract's rule: day buckets are local days in `tz`.

const partsCache = new Map<string, Intl.DateTimeFormat>();

function fmt(tz: string): Intl.DateTimeFormat {
  let f = partsCache.get(tz);
  if (!f) {
    f = new Intl.DateTimeFormat('en-US', {
      timeZone: tz,
      hourCycle: 'h23',
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
    partsCache.set(tz, f);
  }
  return f;
}

export interface LocalParts {
  year: number;
  month: number; // 1-12
  day: number;
  hour: number;
  minute: number;
  second: number;
}

export function localParts(ms: number, tz: string): LocalParts {
  const p: Record<string, number> = {};
  for (const part of fmt(tz).formatToParts(new Date(ms))) {
    if (part.type !== 'literal') p[part.type] = Number(part.value);
  }
  return { year: p.year, month: p.month, day: p.day, hour: p.hour === 24 ? 0 : p.hour, minute: p.minute, second: p.second };
}

/** 'YYYY-MM-DD' of the local day that contains `ms`. */
export function dayKey(ms: number, tz: string): string {
  const p = localParts(ms, tz);
  return `${p.year}-${pad(p.month)}-${pad(p.day)}`;
}

/** 'YYYY-MM-DDTHH' local hour key. */
export function hourKey(ms: number, tz: string): string {
  const p = localParts(ms, tz);
  return `${p.year}-${pad(p.month)}-${pad(p.day)}T${pad(p.hour)}`;
}

export function pad(n: number): string {
  return String(n).padStart(2, '0');
}

/** Offset (minutes east of UTC) of `tz` at the instant `ms`. */
export function tzOffsetMinutes(ms: number, tz: string): number {
  const p = localParts(ms, tz);
  const asUtc = Date.UTC(p.year, p.month - 1, p.day, p.hour, p.minute, p.second);
  return Math.round((asUtc - Math.floor(ms / 1000) * 1000) / 60000);
}

/** The instant (ms) at which local wall-clock `y-m-d h:mi` starts in `tz`. DST-safe: a time
 * inside a spring-forward gap resolves forward (02:30 → 03:30), a repeated fall-back time
 * resolves to its second occurrence. */
export function localToUtc(y: number, m: number, d: number, h = 0, mi = 0, tz = 'UTC'): number {
  const wall = Date.UTC(y, m - 1, d, h, mi, 0);
  const c1 = wall - tzOffsetMinutes(wall, tz) * 60000;
  const off1 = tzOffsetMinutes(c1, tz);
  const c2 = wall - off1 * 60000;
  if (c2 === c1) return c2;
  const off2 = tzOffsetMinutes(c2, tz);
  const c3 = wall - off2 * 60000;
  if (c3 === c2) return c3;
  // A gap: the wall-clock time does not exist. Use the offset from before the switch, which
  // moves the time forward by the size of the gap.
  return wall - Math.min(off1, off2) * 60000;
}

/** Start instant of a local day key. */
export function dayStart(key: string, tz: string): number {
  const [y, m, d] = key.split('-').map(Number);
  return localToUtc(y, m, d, 0, 0, tz);
}

/** Shift a day key by n days (calendar arithmetic, independent of DST). */
export function addDays(key: string, n: number): string {
  const [y, m, d] = key.split('-').map(Number);
  const t = Date.UTC(y, m - 1, d) + n * 86400000;
  const x = new Date(t);
  return `${x.getUTCFullYear()}-${pad(x.getUTCMonth() + 1)}-${pad(x.getUTCDate())}`;
}

/** Days between two keys (b - a). */
export function diffDays(a: string, b: string): number {
  const [ay, am, ad] = a.split('-').map(Number);
  const [by, bm, bd] = b.split('-').map(Number);
  return Math.round((Date.UTC(by, bm - 1, bd) - Date.UTC(ay, am - 1, ad)) / 86400000);
}

/** 0 = Monday … 6 = Sunday for a day key. */
export function weekdayMon0(key: string): number {
  const [y, m, d] = key.split('-').map(Number);
  const js = new Date(Date.UTC(y, m - 1, d)).getUTCDay(); // 0 = Sunday
  return (js + 6) % 7;
}

/** Every day key from `from` to `to` inclusive. */
export function dayRange(from: string, to: string): string[] {
  const n = diffDays(from, to);
  const out: string[] = [];
  for (let i = 0; i <= n; i++) out.push(addDays(from, i));
  return out;
}

/** RFC 3339 with the zone's offset for a local day start, e.g. 2026-09-24T00:00:00+02:00. */
export function isoLocalDayStart(key: string, tz: string): string {
  const ms = dayStart(key, tz);
  return isoWithOffset(ms, tz);
}

export function isoWithOffset(ms: number, tz: string): string {
  const p = localParts(ms, tz);
  const off = tzOffsetMinutes(ms, tz);
  const sign = off >= 0 ? '+' : '-';
  const a = Math.abs(off);
  return `${p.year}-${pad(p.month)}-${pad(p.day)}T${pad(p.hour)}:${pad(p.minute)}:${pad(p.second)}${sign}${pad(Math.floor(a / 60))}:${pad(a % 60)}`;
}

export function isLeapYear(y: number): boolean {
  return (y % 4 === 0 && y % 100 !== 0) || y % 400 === 0;
}

export function browserTz(): string {
  try {
    return Intl.DateTimeFormat().resolvedOptions().timeZone || 'Europe/Berlin';
  } catch {
    return 'Europe/Berlin';
  }
}

/** The day key of "now" in the browser's zone. */
export function todayKey(tz = browserTz(), now = Date.now()): string {
  return dayKey(now, tz);
}
