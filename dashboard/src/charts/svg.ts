// Scales, ticks and path builders for the hand-drawn SVG charts. Pure functions, no DOM.

export interface Scale {
  (v: number): number;
  domain: [number, number];
  range: [number, number];
  invert(px: number): number;
}

export function linear(domain: [number, number], range: [number, number]): Scale {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const span = d1 - d0 || 1;
  const f = ((v: number) => r0 + ((v - d0) / span) * (r1 - r0)) as Scale;
  f.domain = domain;
  f.range = range;
  f.invert = (px) => d0 + ((px - r0) / (r1 - r0 || 1)) * span;
  return f;
}

export function log10(domain: [number, number], range: [number, number]): Scale {
  const lo = Math.log10(Math.max(1e-9, domain[0]));
  const hi = Math.log10(Math.max(1e-9, domain[1]));
  const inner = linear([lo, hi], range);
  const f = ((v: number) => inner(Math.log10(Math.max(1e-9, v)))) as Scale;
  f.domain = domain;
  f.range = range;
  f.invert = (px) => Math.pow(10, inner.invert(px));
  return f;
}

/** "Nice" linear ticks — 3..6 round values covering [min, max]. */
export function niceTicks(min: number, max: number, count = 4): number[] {
  if (!(max > min)) max = min + 1;
  const raw = (max - min) / count;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm >= 5 ? 5 : norm >= 2 ? 2 : 1) * mag;
  const start = Math.ceil(min / step) * step;
  const out: number[] = [];
  for (let v = start; v <= max + 1e-9; v += step) out.push(Number(v.toFixed(10)));
  return out;
}

/** Log ticks at powers of ten (and 2×, 5× when the span is short). */
export function logTicks(min: number, max: number): number[] {
  const lo = Math.floor(Math.log10(Math.max(1e-9, min)));
  const hi = Math.ceil(Math.log10(Math.max(1e-9, max)));
  const out: number[] = [];
  const decades = hi - lo;
  for (let e = lo; e <= hi; e++) {
    const base = Math.pow(10, e);
    for (const m of decades <= 2 ? [1, 2, 5] : [1]) {
      const v = base * m;
      if (v >= min && v <= max) out.push(v);
    }
  }
  return out;
}

/** Nice upper bound for a y axis (zero-based). */
export function niceMax(max: number): number {
  if (!(max > 0)) return 1;
  const t = niceTicks(0, max, 4);
  const last = t[t.length - 1];
  return last >= max ? last : last + (t[1] - t[0]);
}

export interface Pt {
  x: number;
  y: number;
}

/** Monotone cubic interpolation (Fritsch–Carlson): smooth without overshoot. Null gaps break the path. */
export function monotonePath(points: (Pt | null)[]): string {
  const segs: Pt[][] = [];
  let cur: Pt[] = [];
  for (const p of points) {
    if (p == null || !Number.isFinite(p.y)) {
      if (cur.length) segs.push(cur);
      cur = [];
    } else cur.push(p);
  }
  if (cur.length) segs.push(cur);
  return segs.map(segmentPath).join(' ');
}

function segmentPath(p: Pt[]): string {
  const n = p.length;
  if (n === 0) return '';
  if (n === 1) return `M${f(p[0].x)},${f(p[0].y)} h0.01`;
  if (n === 2) return `M${f(p[0].x)},${f(p[0].y)} L${f(p[1].x)},${f(p[1].y)}`;
  const dx: number[] = [];
  const dy: number[] = [];
  const m: number[] = [];
  for (let i = 0; i < n - 1; i++) {
    dx.push(p[i + 1].x - p[i].x);
    dy.push(p[i + 1].y - p[i].y);
    m.push(dx[i] === 0 ? 0 : dy[i] / dx[i]);
  }
  const t: number[] = [m[0]];
  for (let i = 1; i < n - 1; i++) {
    if (m[i - 1] * m[i] <= 0) t.push(0);
    else t.push((m[i - 1] + m[i]) / 2);
  }
  t.push(m[n - 2]);
  for (let i = 0; i < n - 1; i++) {
    if (m[i] === 0) {
      t[i] = 0;
      t[i + 1] = 0;
      continue;
    }
    const a = t[i] / m[i];
    const b = t[i + 1] / m[i];
    const s = a * a + b * b;
    if (s > 9) {
      const tau = 3 / Math.sqrt(s);
      t[i] = tau * a * m[i];
      t[i + 1] = tau * b * m[i];
    }
  }
  let d = `M${f(p[0].x)},${f(p[0].y)}`;
  for (let i = 0; i < n - 1; i++) {
    const h = dx[i] / 3;
    d += ` C${f(p[i].x + h)},${f(p[i].y + t[i] * h)} ${f(p[i + 1].x - h)},${f(p[i + 1].y - t[i + 1] * h)} ${f(p[i + 1].x)},${f(p[i + 1].y)}`;
  }
  return d;
}

/** Area between two monotone curves (upper then lower reversed). Null gaps break it into pieces. */
export function bandPath(upper: (Pt | null)[], lower: (Pt | null)[]): string {
  const pieces: string[] = [];
  let u: Pt[] = [];
  let l: Pt[] = [];
  const flush = () => {
    if (u.length >= 2) {
      const top = segmentPath(u);
      const bottom = segmentPath([...l].reverse());
      pieces.push(`${top} L${f(l[l.length - 1].x)},${f(l[l.length - 1].y)} ${bottom.replace(/^M[^ ]+/, '')} Z`);
    }
    u = [];
    l = [];
  };
  for (let i = 0; i < upper.length; i++) {
    const a = upper[i];
    const b = lower[i];
    if (!a || !b || !Number.isFinite(a.y) || !Number.isFinite(b.y)) flush();
    else {
      u.push(a);
      l.push(b);
    }
  }
  flush();
  return pieces.join(' ');
}

function f(n: number): string {
  return Number.isInteger(n) ? String(n) : n.toFixed(2);
}

/** Pick ≤ n evenly spaced indices for x labels (always the first and the last). */
export function labelIndices(count: number, n: number): number[] {
  if (count <= n) return Array.from({ length: count }, (_, i) => i);
  const out: number[] = [];
  const step = (count - 1) / (n - 1);
  for (let i = 0; i < n; i++) out.push(Math.round(i * step));
  return [...new Set(out)];
}

/** Nearest index to px on an evenly spaced x axis. */
export function nearestIndex(px: number, x0: number, x1: number, count: number): number {
  if (count <= 1) return 0;
  const t = (px - x0) / (x1 - x0);
  return Math.max(0, Math.min(count - 1, Math.round(t * (count - 1))));
}
