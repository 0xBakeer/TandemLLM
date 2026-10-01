// The Conductor view's rules, kept out of the components so the tests read them directly:
// the promotion rule, when two eval arms are comparable, the speed-guard line, the colours of the
// classes, the ribbon series, and immutable edits of a config document.

import type { SeriesColor } from '../charts/chart-base';
import type { RibbonSeries } from '../charts/ribbon';
import type { DecisionRow, EvalReport, Stats } from '../api/conductor';

/** Act is allowed at precision ≥ 0.9 over ≥ 30 labelled decisions (TandemLLM 4, Compare). */
export const PROMOTE_MIN = 30;
export const PROMOTE_PRECISION = 0.9;
export const PROMOTE_RULE = `on is allowed at precision ≥ ${PROMOTE_PRECISION} over ≥ ${PROMOTE_MIN} labelled decisions`;

export function promotion(p: { labelled: number; precision: number | null }): { allowed: boolean; why: string } {
  if (p.labelled < PROMOTE_MIN) return { allowed: false, why: `${p.labelled} of ${PROMOTE_MIN} labelled decisions` };
  if (p.precision == null || p.precision < PROMOTE_PRECISION) return { allowed: false, why: `precision ${fmtP(p.precision)} is below ${PROMOTE_PRECISION}` };
  return { allowed: true, why: `precision ${fmtP(p.precision)} over ${p.labelled}` };
}

const fmtP = (x: number | null) => (x == null ? '—' : x.toFixed(2));

export interface SuiteRow {
  suite: string;
  arm: string;
  gates_passed?: number | null;
  gates_total?: number | null;
  minutes_to_pass?: number | null;
  tokens?: number | null;
  interventions?: number | null;
  build?: string | null;
  night?: string | null;
  n?: number | null;
}

/** Both arms must come from the same build and the same night; anything else is "not comparable". */
export function comparable(a: SuiteRow, b: SuiteRow): { ok: boolean; why: string | null } {
  if (!a.build || !b.build || a.build !== b.build) return { ok: false, why: `builds ${a.build ?? '?'} and ${b.build ?? '?'}` };
  if (!a.night || !b.night || a.night !== b.night) return { ok: false, why: `nights ${a.night ?? '?'} and ${b.night ?? '?'}` };
  return { ok: true, why: null };
}

/** Pair each suite's raw arm with its TandemLLM arm. */
export function suitePairs(rows: SuiteRow[]): { suite: string; raw: SuiteRow | null; tandem: SuiteRow | null; cmp: { ok: boolean; why: string | null } }[] {
  const by = new Map<string, { raw: SuiteRow | null; tandem: SuiteRow | null }>();
  for (const r of rows) {
    const e = by.get(r.suite) ?? { raw: null, tandem: null };
    if (r.arm === 'raw') e.raw = r;
    else e.tandem = r;
    by.set(r.suite, e);
  }
  return [...by.entries()].map(([suite, e]) => ({
    suite,
    ...e,
    cmp: e.raw && e.tandem ? comparable(e.raw, e.tandem) : { ok: false, why: e.raw ? 'no TandemLLM arm' : 'no raw arm' },
  }));
}

/** The orange header line when no never-worse gate ran for the build that serves now. */
export function speedGuard(reports: EvalReport[], engine: { version: string | null; code_sha: string | null }): string | null {
  const hit = reports.some((r) => {
    if (r.kind !== 'speed_guard') return false;
    const b = (r.build ?? {}) as { version?: string; code_sha?: string };
    return (engine.code_sha && b.code_sha === engine.code_sha) || (!engine.code_sha && engine.version && b.version === engine.version);
  });
  return hit ? null : `speed guard not run for build ${engine.version ?? 'unknown'}`;
}

const PALETTE: SeriesColor[] = ['cobalt', 'orange', 'aqua', 'violet', 'yellow', 'red'];

/** A stable colour per class id: its index in the router's list. */
export function classColor(id: string | null, ids: string[]): SeriesColor {
  const i = id == null ? -1 : ids.indexOf(id);
  return i < 0 ? 'muted' : PALETTE[i % PALETTE.length];
}

/** The 24 h ribbon: decisions an hour for the busiest classes, the rest folded into "other". */
export function hourlySeries(stats: Stats, ids: string[], top = 4): RibbonSeries[] {
  const totals = Object.entries(stats.by_class).sort((a, b) => b[1] - a[1]);
  const keep = totals.slice(0, top).map(([k]) => k);
  const series: RibbonSeries[] = keep.map((k) => ({ name: k, color: classColor(k, ids), values: stats.hourly.map((h) => h.by_class[k] ?? 0) }));
  if (totals.length > top) {
    series.push({ name: 'other', color: 'muted', values: stats.hourly.map((h) => Object.entries(h.by_class).filter(([k]) => !keep.includes(k)).reduce((s, [, v]) => s + v, 0)) });
  }
  return series;
}

export function clone<T>(x: T): T {
  return JSON.parse(JSON.stringify(x)) as T;
}

/** `obj` with `value` at `path`, every object on the path copied; the input is untouched. */
export function setIn<T>(obj: T, path: (string | number)[], value: unknown): T {
  if (!path.length) return value as T;
  const [head, ...rest] = path;
  const src = (obj ?? (typeof head === 'number' ? [] : {})) as Record<string | number, unknown>;
  const copy: Record<string | number, unknown> = Array.isArray(src) ? ([...src] as unknown as Record<number, unknown>) : { ...src };
  copy[head] = setIn(src[head] as unknown, rest, value);
  return copy as T;
}

/** A path like `$.classes[code_edit].policy.effort` for people. */
export function readablePath(p: string): string {
  return p.replace(/^\$\.?/, '').replace(/\[/g, ' › ').replace(/\]/g, '').replace(/\./g, ' › ') || '(document)';
}

export function short(v: unknown, n = 80): string {
  if (v == null) return '—';
  const s = typeof v === 'string' ? v : JSON.stringify(v);
  return s.length > n ? s.slice(0, n - 1) + '…' : s;
}

/** What a decision did, in words for the feed: "xhigh, ui-design (1,204 tokens)". */
export function didWhat(r: DecisionRow): string {
  const parts: string[] = [];
  if (r.effort) parts.push(r.thinking && r.thinking.startsWith('off') ? 'thinking off' : r.effort);
  for (const n of r.injected ?? []) {
    const names = n.parts.filter((p) => p.kind !== 'effort').map((p) => p.name);
    parts.push(`${names.length ? names.join(' + ') : 'note'} (${n.tokens.toLocaleString('en-US')} tokens)`);
  }
  if (r.replayed?.count) parts.push(`replayed ${r.replayed.count}`);
  if (r.mode === 'shadow') parts.push('shadow');
  return parts.join(' · ') || '—';
}

export const SOURCE_WORDS: Record<string, string> = {
  decider: 'decider',
  qwen: 'Qwen',
  sticky: 'kept (same task)',
  structural: 'request shape',
  fallback: 'fail-open',
  profile: 'profile default',
};

/** The checker readings of a session as series: one per gate id, a value per turn that had one. */
export function gateSeries(turns: DecisionRow[]): { series: RibbonSeries[]; labels: string[] } {
  const gates = new Map<string, (number | null)[]>();
  turns.forEach((t, i) => {
    for (const r of t.next_checker ?? []) {
      for (const rd of r.readings) {
        const id = `${r.checker} ${rd.gate}`;
        if (!gates.has(id)) gates.set(id, Array(turns.length).fill(null));
        gates.get(id)![i] = rd.value;
      }
    }
  });
  const colors = ['cobalt', 'orange', 'aqua', 'violet', 'yellow', 'red'] as const;
  return {
    series: [...gates.entries()].slice(0, 6).map(([name, values], i) => ({ name, color: colors[i % colors.length], values })),
    labels: turns.map((_, i) => `turn ${i + 1}`),
  };
}
