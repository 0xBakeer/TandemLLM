// The Conductor view's rules, kept out of the components so the tests read them directly:
// the promotion rule, when two eval arms are comparable, the speed-guard line, the colours of the
// classes, the ribbon series, and immutable edits of a config document.

import type { SeriesColor } from '../charts/chart-base';
import type { RibbonSeries } from '../charts/ribbon';
import type { DecisionRow, EvalReport, JsonSchema, MonitorDoc, MonitorStats, Stats, WatchSession } from '../api/conductor';

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

/** One suite on one arm (`raw` or `tandem`), a row of a suite report (tools/tandem_eval.py). */
export interface SuiteRow {
  suite: string;
  suite_id?: string;
  arm: string;
  gates_passed?: number | null;
  gates_total?: number | null;
  minutes_to_pass?: number | null;
  tokens?: number | null;
  interventions?: number | null;
  build?: string | null;
  night?: string | null;
  n?: number | null;
  runs_passed?: number;
  pass_rate?: number;
  wall_minutes?: number | null;
  reasoning_tokens?: number | null;
  decode_tps?: number | null;
  gen_tps?: number | null;
  cache_hit?: number | null;
  empty_stops?: number;
  continuations?: number;
  gave_ups?: number;
  pristine?: boolean;
  indicative?: boolean;
}

/** n under 10 per arm is indicative only (OPS-43: 3 reps is noise on this rig). */
export const INDICATIVE_N = 10;
export function indicative(rows: (SuiteRow | null)[]): boolean {
  return rows.some((r) => r != null && (r.indicative ?? (r.n ?? 0) < INDICATIVE_N));
}

/** The arm as people read it: the composite is TandemLLM. */
export function armName(arm: string): string {
  return arm === 'raw' ? 'raw' : 'TandemLLM';
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

/** The speed guard for the build that serves now, as the header line and the Compare screen say
 * it. `warn`: no report for this build, a report that resolved worse, or one recorded for the
 * same version but other code. `note`: a short gate of one run a side, whose noise is unknown
 * (Khaled's one-pair rule): the deltas as measured, and that they decide nothing. `ok`: the
 * never-worse gate passed with a noise estimate. */
export interface GuardState {
  tone: 'warn' | 'note' | 'ok';
  status: 'missing' | 'pass' | 'worse' | 'no_noise_estimate' | 'other_code';
  line: string;
  detail: string | null;
}

interface GuardStat { stat: string; base: number; cand: number; delta_pct: number; noise_pct: number | null }

const GUARD_STAT: Record<string, [string, string, number]> = {
  mean: ['mean', 'tok/s', 2],
  ttft_p50_ms: ['TTFT', 'ms', 1],
  ms_blk: ['ms a block', 'ms', 2],
};

function guardStats(stats: GuardStat[]): string {
  return stats
    .filter((s) => s.stat in GUARD_STAT)
    .map((s) => {
      const [name, unit, d] = GUARD_STAT[s.stat];
      const sign = s.delta_pct >= 0 ? '+' : '−';
      return `${name} ${s.base.toFixed(d)} → ${s.cand.toFixed(d)} ${unit} (${sign}${Math.abs(s.delta_pct).toFixed(1)} %${s.noise_pct != null ? `, noise ${s.noise_pct.toFixed(1)} %` : ''})`;
    })
    .join(', ');
}

export function speedGuard(reports: EvalReport[], engine: { version: string | null; code_sha: string | null }): GuardState {
  const guards = reports.filter((r) => r.kind === 'speed_guard');
  const b = (r: EvalReport) => (r.build ?? {}) as { version?: string; code_sha?: string };
  const exact = guards.find((r) => (engine.code_sha && b(r).code_sha === engine.code_sha) || (!engine.code_sha && engine.version && b(r).version === engine.version));
  const same = exact ?? guards.find((r) => engine.version && b(r).version === engine.version);
  const build = engine.version ?? 'unknown';
  if (!same) return { tone: 'warn', status: 'missing', line: `speed guard not run for build ${build}`, detail: "the raw engine's never-worse gate (both views, same hour) has no result for the code that serves now." };
  const stats = (same.stats ?? []) as GuardStat[];
  const runs = same.runs_each as number | undefined;
  const nums = guardStats(stats);
  const v = String(same.verdict ?? '');
  const other = exact ? '' : ` It was recorded for code ${String(b(same).code_sha ?? '?')}; the server runs ${engine.code_sha ?? '?'}.`;
  if (v === 'no_noise_estimate') {
    return {
      tone: exact ? 'note' : 'warn',
      status: exact ? 'no_noise_estimate' : 'other_code',
      line: `speed guard for build ${build}: one row pair, ${runs ?? 1} run a side${nums ? ` — ${nums}` : ''}`,
      detail: `One run gives no noise estimate, so this row cannot say whether a difference is real; it is no pass and no failure.${other}`,
    };
  }
  if (v === 'worse') {
    return { tone: 'warn', status: 'worse', line: `speed guard for build ${build}: ${String(same.why ?? 'resolved worse')}`, detail: `${nums}.${other}` };
  }
  if (v === 'pass' && exact) {
    return { tone: 'ok', status: 'pass', line: `speed guard passed for build ${build}${runs ? ` (${runs} runs a side)` : ''}`, detail: nums || null };
  }
  if (v === 'pass') return { tone: 'warn', status: 'other_code', line: `speed guard passed for build ${build} on other code`, detail: `${nums}${nums ? '.' : ''}${other}`.trim() };
  return { tone: 'warn', status: 'missing', line: `speed guard for build ${build}: verdict ${v || 'unknown'}`, detail: (same.why as string | undefined) ?? null };
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

// ---------------------------------------------------------------- the session monitor (SRV-115)
// the action words, `isAct` and `eventWhat` live in lib/supervisor.ts: the Live view and the Dev
// tab say the same things without loading this chunk (VIS-44)
export { ACTION_WORDS, eventWhat, isAct } from './supervisor';
import { eventWhat, isAct, isShadow, type EventLike } from './supervisor';

export const FAMILY_COLOR: Record<string, SeriesColor> = {
  thinking: 'cobalt',
  'end of turn': 'orange',
  'tool calls': 'aqua',
  session: 'violet',
  other: 'muted',
};

/** The 24 h ribbon of the monitor's acts an hour, one series per task family. */
export function monitorSeries(m: MonitorStats): RibbonSeries[] {
  const fams = [...new Set(m.hourly.flatMap((h) => Object.keys(h.by_family)))];
  const order = Object.keys(FAMILY_COLOR);
  fams.sort((a, b) => order.indexOf(a) - order.indexOf(b));
  return fams.map((f) => ({ name: f, color: FAMILY_COLOR[f] ?? 'muted', values: m.hourly.map((h) => h.by_family[f] ?? 0) }));
}

/** The JSON Schema of one monitor task's parameters: the branch of the policy schema whose
 * `rule` constant is the task's rule (rule and mode left out: the editor renders those). */
export function ruleSchema(monitor: JsonSchema | undefined, rule: string): JsonSchema | null {
  const branches = monitor?.properties?.tasks?.additionalProperties;
  const list = typeof branches === 'object' && branches ? branches.oneOf ?? [] : [];
  const hit = list.find((b) => b.properties?.rule?.const === rule);
  if (!hit?.properties) return null;
  const { rule: _r, mode: _m, ...rest } = hit.properties;
  return { ...hit, properties: rest, required: (hit.required ?? []).filter((k) => k !== 'rule' && k !== 'mode') };
}

/** The tool-call tasks (C1 to C4) first, then the rest, in the document's order. */
export function monitorTaskOrder(doc: MonitorDoc): string[] {
  const ids = Object.keys(doc.tasks);
  const tool = ids.filter((t) => /^C\d/.test(t)).sort();
  return [...tool, ...ids.filter((t) => !tool.includes(t))];
}

/** The session-level tasks (S1, S2, S4: SRV-121), in the document's order: the order they get
 * to add the one note a request may carry. */
export function isSessionTask(id: string): boolean {
  return /^S\d/.test(id);
}

/** A watched session's state in one line: "gate (a) 3.5, 12.3, 3.5 · no progress 21 min". */
export function watchLine(w: WatchSession): string {
  const parts: string[] = [];
  for (const g of w.gates) parts.push(`${g.checker} (${g.gate}) ${g.values.map((v) => fmtNum(v)).join(', ')}${g.noted ? ' · noted' : ''}`);
  if (w.compaction?.watching) parts.push(`after compaction ${w.compaction.count}: ${w.compaction.steps} step${w.compaction.steps === 1 ? '' : 's'}, no re-read yet`);
  if (w.progress) parts.push(w.progress.last ? `last progress ${fmtNum(w.progress.minutes)} min ago (${w.progress.last})` : `no progress for ${fmtNum(w.progress.minutes)} min`);
  if (w.client_gone) parts.push(`client gone after ${fmtNum(w.idle_minutes ?? 0)} min`);
  else if (w.idle_minutes != null) parts.push(`idle ${fmtNum(w.idle_minutes)} min`);
  return parts.join(' · ') || 'nothing to watch yet';
}

const fmtNum = (v: number) => (Number.isInteger(v) ? String(v) : String(Math.round(v * 100) / 100));

/** Lines of a textarea as a list, empty lines dropped; and back. */
export function linesToList(text: string): string[] {
  return text.split('\n').map((x) => x.trim()).filter(Boolean);
}

// ---------------------------------------------------------------- one session on one time axis (VIS-44)
export interface SessionAxis {
  start: number; // ms
  end: number;
  turns: { id: string; x0: number; x1: number; color: SeriesColor; compaction: boolean; title: string }[];
  acts: { id: string; x: number; shadow: boolean; task: string; title: string }[];
  notes: { id: string; x: number; shadow: boolean; title: string }[];
  compactions: { id: string; x: number; title: string }[];
  checkers: { id: string; x: number; pass: boolean; title: string }[];
  ticks: { x: number; label: string }[];
}

/** A turn that was opencode's compaction request: the router classed it by its shape. */
export function isCompaction(t: Pick<DecisionRow, 'source' | 'shadow'>): boolean {
  return t.source === 'structural' || (t.shadow as { source?: string } | null)?.source === 'structural';
}

/**
 * The session's night on one axis, wall-clock time from its first turn to the end of its last:
 * each turn a bar from its start to its end (the gaps are the client's time), the monitor's
 * actions above it (hollow: shadow), the router's hidden notes, the compactions, and each checker
 * RESULT where it arrived (with the next request, so at that request's start; the last at its
 * turn's end), with the gate numbers in its title. Positions are 0..1 of the width.
 */
export function sessionAxis(turns: DecisionRow[], events: EventLike[], ids: string[], fmt: (ms: number) => string, nTicks = 5): SessionAxis {
  const ts = (t: DecisionRow) => t.ts_ms;
  const endOf = (t: DecisionRow) => t.ts_ms + Math.max(0, t.outcome?.total_ms ?? 0);
  const marks = events.filter((e) => e.ts_ms != null && (isAct(e) || isShadow(e)));
  const times = [...turns.map(ts), ...turns.map(endOf), ...marks.map((e) => e.ts_ms as number)];
  const start = times.length ? Math.min(...times) : 0;
  const end = times.length ? Math.max(...times) : 1;
  const span = Math.max(1, end - start);
  const x = (t: number) => Math.min(1, Math.max(0, (t - start) / span));
  const order = [...turns].sort((a, b) => a.ts_ms - b.ts_ms);
  const out: SessionAxis = { start, end, turns: [], acts: [], notes: [], compactions: [], checkers: [], ticks: [] };
  order.forEach((t, i) => {
    const comp = isCompaction(t);
    out.turns.push({ id: t.id, x0: x(ts(t)), x1: x(endOf(t)), color: classColor(t.class_id, ids), compaction: comp, title: `${fmt(t.ts_ms)} · ${t.class_id ?? '—'} · ${t.outcome?.finish_reason ?? 'running'}` });
    if (comp) out.compactions.push({ id: t.id, x: x(ts(t)), title: `${fmt(t.ts_ms)} compaction (the client summarised the session)` });
    for (const n of t.injected ?? []) out.notes.push({ id: `${t.id}:${n.pos}`, x: x(ts(t)), shadow: false, title: `${fmt(t.ts_ms)} router note, ${n.tokens} tokens (${n.reason})` });
    if (!t.injected?.length && t.shadow && (t.shadow as { notes?: unknown[] }).notes?.length) out.notes.push({ id: `${t.id}:shadow`, x: x(ts(t)), shadow: true, title: `${fmt(t.ts_ms)} router note in shadow` });
    if (t.next_checker?.length) {
      const at = order[i + 1] ? ts(order[i + 1]) : endOf(t);
      const pass = t.next_checker.every((r) => r.verdict === 'PASS');
      const words = t.next_checker.map((r) => `${r.checker} ${r.verdict}${r.readings.length ? `: ${r.readings.map((g) => `(${g.gate}) ${g.value} ${g.op} ${g.limit}`).join(', ')}` : ''}`).join('; ');
      out.checkers.push({ id: t.id, x: x(at), pass, title: `${fmt(at)} ${words}` });
    }
  });
  for (const e of marks) {
    out.acts.push({ id: e.id ?? `${e.ts_ms}`, x: x(e.ts_ms as number), shadow: !isAct(e), task: e.task ?? '—', title: `${fmt(e.ts_ms as number)} ${eventWhat({ task: e.task, decision: e.decision, mode: e.mode, trigger: e.trigger ?? null, cap: e.cap ?? null })}` });
  }
  for (let k = 0; k <= nTicks; k++) out.ticks.push({ x: k / nTicks, label: fmt(start + (span * k) / nTicks) });
  return out;
}

/** Tier 1's shadow agreement in a session: its shadow decisions, how many were labelled, and how
 *  many of those the outcome confirmed (`right`); null while Tier 1 has decided nothing here. */
export function tier1Agreement(events: (EventLike & { label?: string | null })[]): { decisions: number; labelled: number; agree: number } | null {
  const t1 = events.filter((e) => e.tier === 1 && isShadow(e));
  if (!t1.length) return null;
  const lab = t1.filter((e) => e.label === 'right' || e.label === 'wrong');
  return { decisions: t1.length, labelled: lab.length, agree: lab.filter((e) => e.label === 'right').length };
}
