// A small Prometheus text-format parser and the delta arithmetic the live strip needs.
// Handles counters, gauges, histograms (cumulative buckets, +Inf, _sum/_count), labels with
// escapes, HELP/TYPE lines, and missing families (= feature off → null, never 0).

export type SampleLabels = Record<string, string>;

export interface Sample {
  name: string;
  labels: SampleLabels;
  value: number;
}

export interface Family {
  name: string;
  type: 'counter' | 'gauge' | 'histogram' | 'summary' | 'untyped';
  help: string;
  samples: Sample[];
}

export interface Scrape {
  at: number; // ms
  families: Map<string, Family>;
}

const LABEL_RE = /([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"/g;

function unescapeLabel(v: string): string {
  return v.replace(/\\(["\\n])/g, (_, c) => (c === 'n' ? '\n' : c));
}

function parseValue(s: string): number {
  const t = s.trim();
  if (t === '+Inf' || t === 'Inf') return Infinity;
  if (t === '-Inf') return -Infinity;
  if (t === 'NaN') return NaN;
  return Number(t);
}

/** Family name for a sample name: strips histogram/summary suffixes. */
function familyOf(sampleName: string, types: Map<string, Family['type']>): string {
  for (const suffix of ['_bucket', '_sum', '_count']) {
    if (sampleName.endsWith(suffix)) {
      const base = sampleName.slice(0, -suffix.length);
      const t = types.get(base);
      if (t === 'histogram' || t === 'summary') return base;
    }
  }
  return sampleName;
}

export function parsePrometheus(text: string, at = Date.now()): Scrape {
  const families = new Map<string, Family>();
  const types = new Map<string, Family['type']>();
  const ensure = (name: string): Family => {
    let f = families.get(name);
    if (!f) {
      f = { name, type: types.get(name) ?? 'untyped', help: '', samples: [] };
      families.set(name, f);
    }
    return f;
  };
  for (const raw of text.split('\n')) {
    const line = raw.trim();
    if (!line) continue;
    if (line.startsWith('#')) {
      const m = /^#\s+(HELP|TYPE)\s+([a-zA-Z_:][a-zA-Z0-9_:]*)\s*(.*)$/.exec(line);
      if (!m) continue;
      if (m[1] === 'TYPE') {
        const t = m[3].trim() as Family['type'];
        types.set(m[2], t);
        ensure(m[2]).type = t;
      } else {
        ensure(m[2]).help = m[3];
      }
      continue;
    }
    // name{labels} value [timestamp]
    let name: string;
    let labels: SampleLabels = {};
    let rest: string;
    const brace = line.indexOf('{');
    if (brace >= 0) {
      name = line.slice(0, brace).trim();
      const close = line.indexOf('}', brace);
      if (close < 0) continue;
      const body = line.slice(brace + 1, close);
      LABEL_RE.lastIndex = 0;
      let lm: RegExpExecArray | null;
      while ((lm = LABEL_RE.exec(body))) labels[lm[1]] = unescapeLabel(lm[2]);
      rest = line.slice(close + 1).trim();
    } else {
      const sp = line.search(/\s/);
      if (sp < 0) continue;
      name = line.slice(0, sp);
      rest = line.slice(sp + 1).trim();
    }
    const value = parseValue(rest.split(/\s+/)[0]);
    if (Number.isNaN(value) && rest.split(/\s+/)[0] !== 'NaN') continue;
    ensure(familyOf(name, types)).samples.push({ name, labels, value });
  }
  return { at, families };
}

function labelsMatch(have: SampleLabels, want?: SampleLabels): boolean {
  if (!want) return true;
  for (const k of Object.keys(want)) if (have[k] !== want[k]) return false;
  return true;
}

/** Sum of a plain sample (counter/gauge) across the given label filter; null when the family is absent. */
export function value(s: Scrape | null, name: string, labels?: SampleLabels): number | null {
  if (!s) return null;
  const f = s.families.get(name);
  if (!f || f.samples.length === 0) return null;
  let sum = 0;
  let n = 0;
  for (const x of f.samples) {
    if (x.name === name && labelsMatch(x.labels, labels)) {
      sum += x.value;
      n++;
    }
  }
  return n ? sum : null;
}

export interface Histogram {
  buckets: { le: number; count: number }[]; // cumulative, sorted, +Inf last
  sum: number;
  count: number;
}

export function histogram(s: Scrape | null, name: string, labels?: SampleLabels): Histogram | null {
  if (!s) return null;
  const f = s.families.get(name);
  if (!f) return null;
  const buckets = new Map<number, number>();
  let sum = 0;
  let count = 0;
  let seen = false;
  for (const x of f.samples) {
    if (!labelsMatch(x.labels, labels)) continue;
    if (x.name === name + '_bucket') {
      const le = parseValue(x.labels.le ?? '+Inf');
      buckets.set(le, (buckets.get(le) ?? 0) + x.value);
      seen = true;
    } else if (x.name === name + '_sum') {
      sum += x.value;
      seen = true;
    } else if (x.name === name + '_count') {
      count += x.value;
      seen = true;
    }
  }
  if (!seen) return null;
  return {
    buckets: [...buckets.entries()].map(([le, c]) => ({ le, count: c })).sort((a, b) => a.le - b.le),
    sum,
    count,
  };
}

/** Bucket-wise delta b - a (same bucket bounds); null when either side is missing. */
export function histogramDelta(a: Histogram | null, b: Histogram | null): Histogram | null {
  if (!a || !b) return null;
  const am = new Map(a.buckets.map((x) => [x.le, x.count]));
  const buckets = b.buckets.map((x) => ({ le: x.le, count: Math.max(0, x.count - (am.get(x.le) ?? 0)) }));
  return { buckets, sum: Math.max(0, b.sum - a.sum), count: Math.max(0, b.count - a.count) };
}

/**
 * Quantile estimate from cumulative buckets by linear interpolation inside the bucket
 * (Prometheus' histogram_quantile). Null when the histogram has no observations.
 */
export function histogramQuantile(h: Histogram | null, q: number): number | null {
  if (!h || h.count <= 0 || h.buckets.length === 0) return null;
  const total = h.buckets[h.buckets.length - 1].count;
  if (total <= 0) return null;
  const rank = q * total;
  let prevLe = 0;
  let prevCount = 0;
  for (const b of h.buckets) {
    if (b.count >= rank) {
      if (!Number.isFinite(b.le)) return prevLe; // +Inf bucket: the last finite bound
      const inBucket = b.count - prevCount;
      if (inBucket <= 0) return b.le;
      return prevLe + ((b.le - prevLe) * (rank - prevCount)) / inBucket;
    }
    prevLe = b.le;
    prevCount = b.count;
  }
  return prevLe;
}

/** Counter delta per second between two scrapes; null when a family is missing or time did not move. */
export function rate(a: Scrape | null, b: Scrape | null, name: string, labels?: SampleLabels): number | null {
  if (!a || !b) return null;
  const va = value(a, name, labels);
  const vb = value(b, name, labels);
  if (va == null || vb == null) return null;
  const dt = (b.at - a.at) / 1000;
  if (dt <= 0) return null;
  return Math.max(0, vb - va) / dt;
}

export function delta(a: Scrape | null, b: Scrape | null, name: string, labels?: SampleLabels): number | null {
  const va = value(a, name, labels);
  const vb = value(b, name, labels);
  if (va == null || vb == null) return null;
  return Math.max(0, vb - va);
}

/** The live numbers the Performance strip shows, from two successive scrapes. */
export interface LiveNumbers {
  decodeTps: number | null; // tokens generated per second between the scrapes
  generating: boolean; // a request was running at the second scrape
  ttftP50: number | null; // seconds, over the observations between the scrapes (or the whole histogram when nothing new)
  tokensPerBlock: number | null;
  acceptance: number | null; // 0..1
  running: number | null;
  waiting: number | null;
  specReported: boolean; // the speculation families exist at all
}

export function liveNumbers(prev: Scrape | null, cur: Scrape | null): LiveNumbers {
  const running = value(cur, 'qse_requests_running');
  const waiting = value(cur, 'qse_requests_waiting');
  const genRate = rate(prev, cur, 'qse_generation_tokens_total');
  const ttftCur = histogram(cur, 'qse_time_to_first_token_seconds');
  const ttftD = histogramDelta(histogram(prev, 'qse_time_to_first_token_seconds'), ttftCur);
  const ttftP50 = ttftD && ttftD.count > 0 ? histogramQuantile(ttftD, 0.5) : histogramQuantile(ttftCur, 0.5);
  const apbCur = histogram(cur, 'qse_spec_accept_per_block');
  const apbD = histogramDelta(histogram(prev, 'qse_spec_accept_per_block'), apbCur);
  const apbSrc = apbD && apbD.count > 0 ? apbD : apbCur;
  const tokensPerBlock = apbSrc && apbSrc.count > 0 ? apbSrc.sum / apbSrc.count : null;
  const accD = delta(prev, cur, 'qse_spec_decode_num_accepted_tokens_total');
  const drD = delta(prev, cur, 'qse_spec_decode_num_draft_tokens_total');
  let acceptance: number | null = null;
  if (accD != null && drD != null && drD > 0) acceptance = accD / drD;
  else {
    const acc = value(cur, 'qse_spec_decode_num_accepted_tokens_total');
    const dr = value(cur, 'qse_spec_decode_num_draft_tokens_total');
    if (acc != null && dr != null && dr > 0) acceptance = acc / dr;
  }
  const specReported = !!cur && (cur.families.has('qse_spec_accept_per_block') || cur.families.has('qse_spec_decode_num_draft_tokens_total'));
  return {
    decodeTps: genRate,
    generating: (running ?? 0) > 0,
    ttftP50,
    tokensPerBlock,
    acceptance,
    running,
    waiting,
    specReported,
  };
}
