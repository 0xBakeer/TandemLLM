// A small Prometheus text-format parser and the delta arithmetic the live strip needs.
// Handles counters, gauges, histograms (cumulative buckets, +Inf, _sum/_count), labels with
// escapes, HELP/TYPE lines, and missing families (= feature off → null, never 0).

import type { EngineFigures, PhaseSplit } from '../api/types';

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
  finished: number | null; // requests that ended between the scrapes (qse_requests_total, every finish reason)
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
    finished: delta(prev, cur, 'qse_requests_total'),
  };
}

// ---- P2, P3: the vLLM and SGLang backends the proxy reads (qse_backend_*, proxy/engines.py) ----


export interface BackendEngine {
  id: string;
  kind: string;
  version: string | null;
}

/** The backends whose own /metrics page the proxy reads, from `qse_backend_info`. */
export function backendEngines(s: Scrape | null): BackendEngine[] {
  const f = s?.families.get('qse_backend_info');
  if (!f) return [];
  return f.samples
    .filter((x) => x.name === 'qse_backend_info' && x.labels.backend)
    .map((x) => ({ id: x.labels.backend, kind: x.labels.kind ?? '', version: x.labels.version || null }))
    .sort((a, b) => a.id.localeCompare(b.id));
}

const r4 = (x: number | null): number | null => (x == null ? null : Math.round(x * 1e4) / 1e4);
const r2 = (x: number | null): number | null => (x == null ? null : Math.round(x * 100) / 100);
const ratio = (n: number | null, d: number | null): number | null => (n == null || !d ? null : n / d);

/**
 * One backend's figures from two scrapes of the proxy's page, the first and the newest of the
 * window: the same numbers proxy/engines.py `figures()` derives from the engine's own pages, so the
 * Performance view and the System view agree. Null where the page has no number, never 0.
 */
export function backendFigures(first: Scrape | null, last: Scrape | null, id: string, kind: string): EngineFigures {
  const L = { backend: id };
  const n = (name: string, s: Scrape | null = last, extra: SampleLabels = {}) => value(s, 'qse_backend_' + name, { ...L, ...extra });
  const dt = first && last ? (last.at - first.at) / 1000 : 0;
  const win = !!first && dt > 0;
  const d = (name: string, extra: SampleLabels = {}) => (win ? delta(first, last, 'qse_backend_' + name, { ...L, ...extra }) : null);
  const h = (name: string) => (win ? histogramDelta(histogram(first, 'qse_backend_' + name, L), histogram(last, 'qse_backend_' + name, L)) : null);
  const split = (name: string): PhaseSplit | null => {
    const x = h(name);
    if (!x || x.count <= 0) return null;
    return { p50: r4(histogramQuantile(x, 0.5)), p90: r4(histogramQuantile(x, 0.9)), count: Math.round(x.count) };
  };
  const gen = d('generation_tokens_total');
  const prompt = d('prompt_tokens_total');
  const cached = d('prompt_tokens_cached_total');
  const ttft = h('ttft_seconds');
  const awake = n('awake');
  const out: EngineFigures = {
    running: n('requests_running'),
    waiting: n('requests_waiting'),
    kv_usage: r4(n('kv_usage')),
    kv_tokens: n('kv_tokens'),
    kv_capacity_tokens: n('kv_capacity_tokens'),
    awake: awake == null ? null : awake >= 0.5,
    decode_tps: gen == null ? null : r2(gen / dt),
    prompt_tps: prompt == null ? null : r2(prompt / dt),
    engine_gen_tps: r2(n('gen_tps')),
    cached_share: r4(ratio(cached, prompt)),
    prefix_hit_rate: null,
    engine_cache_hit_rate: r4(n('cache_hit_rate')),
    ttft_p50_s: ttft && ttft.count > 0 ? r4(histogramQuantile(ttft, 0.5)) : null,
    queue_s: split('queue_seconds'),
    prefill_s: split('prefill_seconds'),
    decode_s: split('decode_seconds'),
    e2e_s: split('e2e_seconds'),
    preemptions: d('preemptions_total'),
    finished: d('requests_finished_total'),
    window_s: win ? Math.round(dt * 10) / 10 : null,
    spec: null,
  };
  if (kind === 'vllm' && win) out.prefix_hit_rate = r4(ratio(d('prefix_cache_hits_total'), d('prefix_cache_queries_total')));
  if (kind === 'sglang' && win) out.prefix_hit_rate = out.cached_share ?? null;
  const has = (name: string) => n(name) != null;
  if (has('spec_draft_tokens_total') || has('spec_accepted_tokens_total')) {
    const acc = d('spec_accepted_tokens_total');
    const drafted = d('spec_draft_tokens_total');
    const drafts = d('spec_drafts_total');
    let perPos: (number | null)[] | null = null;
    const fam = last?.families.get('qse_backend_spec_accepted_tokens_per_pos_total');
    if (fam && win && drafts) {
      const positions = [...new Set(fam.samples.filter((x) => x.labels.backend === id).map((x) => x.labels.position))].sort((a, b) => Number(a) - Number(b));
      perPos = positions.map((p) => r4(ratio(d('spec_accepted_tokens_per_pos_total', { position: p }), drafts)));
    }
    out.spec = { source: 'counters', acceptance: r4(ratio(acc, drafted)), tokens_per_step: drafts && acc != null ? r4(acc / drafts + 1) : null, per_position: perPos, drafts };
  } else if (has('spec_accept_rate') || has('spec_accept_length')) {
    out.spec = { source: 'engine_window', acceptance: r4(n('spec_accept_rate')), tokens_per_step: r4(n('spec_accept_length')), per_position: null, drafts: null };
  }
  return out;
}
