// Number and unit formatting. Every figure on the dashboard carries its unit; compact notation
// for the big numbers, exact values in tooltips.

const nf0 = new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 });

export function compact(n: number | null | undefined, digits = 1): string {
  if (n == null || !Number.isFinite(n)) return '—';
  const abs = Math.abs(n);
  if (abs < 1000) return nf0.format(n);
  const units: [number, string][] = [
    [1e12, 'T'],
    [1e9, 'B'],
    [1e6, 'M'],
    [1e3, 'K'],
  ];
  for (const [v, s] of units) {
    if (abs >= v) {
      const x = n / v;
      // 1.2M, 12.3K, 123K — keep three significant figures at most.
      const d = Math.abs(x) >= 100 ? 0 : Math.abs(x) >= 10 ? Math.min(digits, 1) : digits;
      return trimZero(x.toFixed(d)) + s;
    }
  }
  return nf0.format(n);
}

function trimZero(s: string): string {
  return s.includes('.') ? s.replace(/\.?0+$/, '') : s;
}

export function exact(n: number | null | undefined): string {
  if (n == null || !Number.isFinite(n)) return '—';
  return nf0.format(n);
}

export function fixed(n: number | null | undefined, d = 1): string {
  if (n == null || !Number.isFinite(n)) return '—';
  return n.toFixed(d);
}

export function pct(x: number | null | undefined, d = 1): string {
  if (x == null || !Number.isFinite(x)) return '—';
  return Number((x * 100).toPrecision(12)).toFixed(d) + ' %';
}

/** Milliseconds → "702 ms" / "2.24 s" / "1 m 12 s". */
export function ms(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return '—';
  if (v < 1000) return `${v < 10 ? v.toFixed(1) : Math.round(v)} ms`;
  if (v < 60_000) return `${(v / 1000).toFixed(2)} s`;
  const m = Math.floor(v / 60_000);
  const s = Math.round((v - m * 60_000) / 1000);
  return `${m} m ${s} s`;
}

/** Seconds of uptime → "2 d 4 h" / "1 h 03 m" / "48 s". */
export function duration(sec: number | null | undefined): string {
  if (sec == null || !Number.isFinite(sec)) return '—';
  const s = Math.max(0, Math.floor(sec));
  const d = Math.floor(s / 86400);
  const h = Math.floor((s % 86400) / 3600);
  const m = Math.floor((s % 3600) / 60);
  if (d > 0) return `${d} d ${h} h`;
  if (h > 0) return `${h} h ${String(m).padStart(2, '0')} m`;
  if (m > 0) return `${m} m ${String(s % 60).padStart(2, '0')} s`;
  return `${s} s`;
}

export function bytes(b: number | null | undefined, d = 2): string {
  if (b == null || !Number.isFinite(b)) return '—';
  const abs = Math.abs(b);
  if (abs >= 1e9) return `${(b / 1e9).toFixed(d)} GB`;
  if (abs >= 1e6) return `${(b / 1e6).toFixed(d)} MB`;
  if (abs >= 1e3) return `${(b / 1e3).toFixed(d)} KB`;
  return `${b} B`;
}

export function tps(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return '—';
  return `${v >= 100 ? Math.round(v) : v.toFixed(1)} tok/s`;
}

export function shortHash(h: string | null | undefined, n = 7): string {
  if (!h) return '—';
  return h.slice(0, n);
}

/** "2026-09-18" → "18 Sep 2026" */
export function dateLong(iso: string): string {
  const d = new Date(iso.length === 10 ? iso + 'T00:00:00' : iso);
  return d.toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' });
}

export function timeShort(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

export function dateTimeShort(iso: string): string {
  const d = new Date(iso);
  return (
    d.toLocaleDateString('en-GB', { day: '2-digit', month: 'short' }) +
    ' ' +
    d.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' })
  );
}

/** Relative "12 s ago" / "3 m ago" / "2 h ago" / "yesterday". */
export function ago(iso: string | null | undefined, now = Date.now()): string {
  if (!iso) return 'never';
  const t = new Date(iso).getTime();
  const s = Math.max(0, Math.round((now - t) / 1000));
  if (s < 60) return `${s} s ago`;
  const m = Math.round(s / 60);
  if (m < 60) return `${m} m ago`;
  const h = Math.round(m / 60);
  if (h < 24) return `${h} h ago`;
  const d = Math.round(h / 24);
  return d === 1 ? 'yesterday' : `${d} d ago`;
}
