// Microbenchmark of the qse-spark ribbon render per event, old vs new, on this box.
// Real monotonePath from src/charts/svg.ts. 300 slots, a fresh identical array a
// fresh second, 4 events per second for 12 s (the activity.spec.ts soak shape).
import { linear, monotonePath } from '../src/charts/svg.ts';

function ptsFor(values, W, H) {
  const vals = values.filter((v) => v != null && Number.isFinite(v));
  const max = vals.length ? Math.max(...vals) : 1;
  const y = linear([0, max > 0 ? max * 1.1 : 1], [H - 3, 4]);
  const x = linear([0, Math.max(1, values.length - 1)], [3, W - 3]);
  return values.map((v, i) => (v == null || !Number.isFinite(v) ? null : { x: x(i), y: y(v) }));
}

function sameValues(a, b) {
  if (a === b) return true;
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

// the week's window: 300 slots, ~40% gaps, a fresh copy per event (as lit re-passes it)
const base = Array.from({ length: 300 }, (_, i) => (i % 5 === 4 || i < 12 ? null : 20 + 60 * Math.abs(Math.sin(i / 9))));

function oldRender(values, W, H) {              // v0.11.0: the spline, twice
  const pts = ptsFor(values, W, H);
  return monotonePath(pts).length + monotonePath(pts).length;
}

let cache = null;
function newRender(values, W, H) {              // now: once, and kept while unchanged
  const c = cache;
  let d;
  if (c && c.w === W && c.h === H && sameValues(c.vals, values)) d = c.d;
  else d = monotonePath(ptsFor(values, W, H));
  cache = { vals: values, w: W, h: H, d };
  return d.length;
}

const seconds = 12, events = 4;
const oldMs = [], newMs = [];
for (let run = 0; run < 2; run++) {
  for (const [name, step] of [['old', oldRender], ['new', newRender]]) {
    for (let s = 0; s < seconds; s++) {
      const values = base.map((v, i) => (i === 299 - s ? 77.7 : v));   // one new point a second
      for (let e = 0; e < events; e++) {
        const t = process.hrtime.bigint();
        step([...values], 560, 44);                                     // a fresh array per event
        const ns = Number(process.hrtime.bigint() - t);
        (name === 'old' ? oldMs : newMs).push(ns / 1e6);
      }
    }
  }
}
const sum = (a) => a.reduce((x, v) => x + v, 0);
console.log(JSON.stringify({
  runs: 2, seconds, events_per_second: events,
  old_total_ms: +sum(oldMs).toFixed(2),
  new_total_ms: +sum(newMs).toFixed(2),
  old_mean_ms_per_frame: +(sum(oldMs) / oldMs.length).toFixed(3),
  new_mean_ms_per_frame: +(sum(newMs) / newMs.length).toFixed(3),
  speedup: +(sum(oldMs) / sum(newMs)).toFixed(2),
}));
