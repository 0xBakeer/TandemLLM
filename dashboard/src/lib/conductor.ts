// The Conductor view's rules, kept out of the components so the tests read them directly:
// the promotion rule, when two eval arms are comparable, the speed-guard line, the colours of the
// classes, the ribbon series, and immutable edits of a config document.

import type { SeriesColor } from '../charts/chart-base';
import type { RibbonSeries } from '../charts/ribbon';
import type { ContextStage, DecisionRow, Effort, EvalReport, JsonSchema, LibraryHit, Mode3, MonitorDoc, MonitorStats, PackagesDoc, Policy, ProfileDoc, RoleContract, RoleKind, RoleWindow, RoleWrites, RouterDoc, SharingForm, Size, SizeConfig, SizeDecision, SideShape, SizeSignal, Stats, ToolTurnEffort, UiProgress, UiStates, WatchSession } from '../api/conductor';

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
  // SRV-157: turns per effort level over the runs, and the profile's by_size they ran under
  efforts?: Record<string, number> | null;
  size_mode?: string | null;
  // OPS-63: the median turns of a run (opencode's assistant messages)
  turns?: number | null;
}

/** A kept report that a later one replaced: the box marks it (`superseded_by`), or another
 *  report names it in its `supersedes`. */
export function supersededBy(rep: EvalReport | undefined, all: EvalReport[]): { file: string; why: string | null } | null {
  if (!rep) return null;
  const own = rep.superseded_by as { file?: string; why?: string | null } | null | undefined;
  if (own?.file) return { file: own.file, why: own.why ?? null };
  const by = all.find((r) => (r.supersedes as { file?: string } | null | undefined)?.file === rep.file && r.file);
  return by ? { file: String(by.file), why: (by.supersedes as { why?: string | null }).why ?? null } : null;
}

/** The suite report to show: the picked one, else the newest that nothing superseded, else the
 *  newest (the box lists them newest first). */
export function pickSuite(reports: EvalReport[], file: string | null): EvalReport | undefined {
  const suite = reports.filter((r) => r.kind === 'suite');
  return suite.find((r) => r.file === file) ?? suite.find((r) => !supersededBy(r, suite)) ?? suite[0];
}

/** "high 4 · xhigh 10 (size on)": turns per effort level, low to high (SRV-157). */
export function effortsCell(r: Pick<SuiteRow, 'efforts' | 'size_mode'>): string {
  const e = r.efforts;
  if (!e || !Object.keys(e).length) return '—';
  const order = ['off', 'low', 'medium', 'high', 'xhigh', 'unchanged'];
  const rank = (k: string) => (order.includes(k) ? order.indexOf(k) : 99);
  const s = Object.keys(e).sort((a, b) => rank(a) - rank(b) || a.localeCompare(b)).map((k) => `${k} ${e[k]}`).join(' · ');
  return r.size_mode ? `${s} (size ${r.size_mode})` : s;
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
 * (the owner's one-pair rule): the deltas as measured, and that they decide nothing. `ok`: the
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
/** S1: the built-in side-request shapes (server/conductor/signals.py SIDE_SHAPES), what a router
 *  without `structural.side_shapes` uses. */
export const SIDE_SHAPES: SideShape[] = [
  { id: 'opencode_title', client: 'opencode', kind: 'title', max_tokens: 64, match: 'You are a title generator. You output ONLY a thread title. Nothing else.' },
  { id: 'opencode_summary', client: 'opencode', kind: 'summary', max_tokens: 256, match: 'Summarize what was done in this conversation. Write like a pull request description.' },
  { id: 'hermes_title', client: 'Hermes', kind: 'title', max_tokens: 64, match: "You name chat sessions. Given the user's opening message, write a title that lets them find this conversation again in a list." },
];

/** S1: what the router did with a client's side request (its journal row's `extra.side`). */
export interface SideInfo { shape: string; client: string; kind: string; max_tokens: number; thinking: string; first?: boolean }
export function sideOf(r: Pick<DecisionRow, 'kind' | 'extra'>): SideInfo | null {
  return r.kind === 'side' ? ((r.extra?.side as SideInfo | undefined) ?? null) : null;
}

export function didWhat(r: DecisionRow): string {
  const side = sideOf(r);
  if (r.kind === 'side') {
    return side ? `${side.client}'s ${side.kind} request · at most ${side.max_tokens.toLocaleString('en-US')} tokens · thinking ${side.thinking === 'off' ? 'off' : "the client's"}${r.mode === 'shadow' || r.shadow ? ' · shadow' : ''}` : 'side request';
  }
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

// SRV-157: the size signals as the Routing screen names them, with their unit
export const SIZE_SIGNALS: { id: SizeSignal; label: string; unit: string; hint: string }[] = [
  { id: 'session_tokens', label: 'prompt tokens the session added', unit: 'tokens', hint: 'after its first request: opencode\'s own head (about 24k tokens) is not the task' },
  { id: 'files', label: 'files the tool calls named', unit: 'files', hint: 'distinct files read, edited or written' },
  { id: 'checks', label: 'gates a checker named', unit: 'gates', hint: 'the most one RESULT line of a known checker listed; no checker yet: it does not vote' },
  { id: 'turns', label: 'turns', unit: 'turns', hint: 'requests in the session' },
];
export const BY_SIZE_DEFAULT: Mode3 = 'shadow';
// the built-in thresholds (server/conductor/schema.py ROUTER.effort.size), for a router version
// stored before SRV-157; tests/conductor.test.ts holds it to seeds.json
export const SIZE_SEED: SizeConfig = {
  signals: {
    session_tokens: { enabled: true, small: 20000, medium: 60000 },
    files: { enabled: true, small: 3, medium: 8 },
    checks: { enabled: true, small: 4, medium: 8 },
    turns: { enabled: true, small: 30, medium: 100 },
  },
  reset_on_compaction: true,
};

/** "small (12,000 tokens, 2 files): high under xhigh, on" — what the size did on one turn. */
export function sizeLine(s: SizeDecision | null | undefined): string {
  if (!s) return '—';
  const sig = Object.entries(s.signals ?? {})
    .map(([k, v]) => {
      const n = v?.value ?? 0;
      const unit = SIZE_SIGNALS.find((x) => x.id === k)?.unit ?? k;
      return `${n.toLocaleString('en-US')} ${n === 1 ? unit.replace(/s$/, '') : unit}`;
    })
    .join(', ');
  const head = `${s.size}${sig ? ` (${sig})` : ''}`;
  if (!s.effort) return `${head}, ${s.mode}; this class has no size map`;
  const verb = s.applied ? (s.effort === s.ceiling ? 'stays' : 'thinks at') : 'would think at';
  return `${head}: ${verb} ${s.effort} under ${s.ceiling}${s.applied ? '' : ` (${s.mode})`}`;
}

/** The effort a size gets from a class's map: its own level, never above the class's effort. */
export function sizeEffort(policy: Policy, size: Size): Effort {
  const order: Effort[] = ['off', 'low', 'medium', 'high', 'xhigh'];
  const ceiling = policy.effort ?? 'medium';
  const want = policy.effort_by_size?.[size] ?? ceiling;
  return order.indexOf(want) <= order.indexOf(ceiling) ? want : ceiling;
}

export type BudgetLevel = Exclude<Effort, 'off'>;
export const BUDGET_LEVELS: BudgetLevel[] = ['low', 'medium', 'high', 'xhigh'];
// the schema's maximum: the context, 256k (a hard rule)
export const BUDGET_MAX = 262144;

/** A level's reasoning cap under a profile (SRV-159): the profile's own, else the router's;
 *  `cap` 0 is no cap below the backstop. */
export function levelBudget(level: BudgetLevel, router: Pick<RouterDoc, 'effort'> | null, profile: Pick<ProfileDoc, 'thinking'> | null): { cap: number | null; source: 'profile' | 'router' | null } {
  const own = profile?.thinking.budgets?.[level];
  if (own != null) return { cap: own, source: 'profile' };
  const lv = router?.effort.levels[level];
  return lv ? { cap: lv.budget, source: 'router' } : { cap: null, source: null };
}

/** "16,384 tokens", "no cap", "—" */
export function budgetText(cap: number | null): string {
  if (cap == null) return '—';
  return cap > 0 ? `${cap.toLocaleString('en-US')} tokens` : 'no cap';
}

/** The profile's `thinking.budgets` with one level set, or dropped (null or an empty field):
 *  the key goes away when no level is left, so a stored profile stays as it was. */
export function withBudget(thinking: ProfileDoc['thinking'], level: BudgetLevel, value: number | null): ProfileDoc['thinking'] {
  const budgets = { ...(thinking.budgets ?? {}) };
  if (value == null || !Number.isFinite(value)) delete budgets[level];
  else budgets[level] = Math.min(BUDGET_MAX, Math.max(0, Math.round(value)));
  const out: ProfileDoc['thinking'] = { ...thinking, budgets };
  if (!Object.keys(budgets).length) delete out.budgets;
  return out;
}

// F2: the levels a tool-result turn may think at; `inherit` is the class's (the server's
// TOOL_TURN_EFFORTS, held to seeds.json by tests/conductor.test.ts)
export const TOOL_TURN_EFFORTS: ToolTurnEffort[] = ['inherit', 'off', 'low', 'medium', 'high', 'xhigh'];

/** A tool-result turn's effort under a profile (server/conductor/schema.py `tool_turn_effort`):
 *  the profile's when it names one (its `inherit` is the class's), else the router's, else
 *  inherit. */
export function toolTurnEffort(router: Pick<RouterDoc, 'structural'> | null, profile: Pick<ProfileDoc, 'thinking'> | null): { level: ToolTurnEffort; source: 'profile' | 'router' | 'default' } {
  const own = profile?.thinking.tool_turn_effort;
  if (own) return { level: own, source: 'profile' };
  const r = router?.structural.tool_turn_effort;
  return r ? { level: r, source: 'router' } : { level: 'inherit', source: 'default' };
}

/** "the class's effort", "low (1,024 tokens)", "thinking off" — what a tool turn gets. */
export function toolTurnText(level: ToolTurnEffort, router: Pick<RouterDoc, 'effort'> | null, profile: Pick<ProfileDoc, 'thinking'> | null): string {
  if (level === 'inherit') return "the class's effort";
  if (level === 'off') return 'thinking off';
  return `${level} (${budgetText(levelBudget(level, router, profile).cap)})`;
}

/** A classifier report's digest variants as table rows (SRV-153): one per variant and set. */
export function variantRows(rep: EvalReport): { variant: string; set: string; n: number; decider: number | null; cascade: number | null; router: number | null; p50: number | null }[] {
  const v = (rep as unknown as { variants?: Record<string, Record<string, { n: number; stages: Record<string, { accuracy: number; p50_ms: number | null }> }>> }).variants;
  if (!v) return [];
  const out = [];
  for (const [variant, sets] of Object.entries(v)) {
    for (const [set, r] of Object.entries(sets)) {
      out.push({ variant, set: set.split('/').pop() ?? set, n: r.n, decider: r.stages.decider?.accuracy ?? null, cascade: r.stages.cascade?.accuracy ?? null, router: r.stages.router?.accuracy ?? null, p50: r.stages.decider?.p50_ms ?? null });
    }
  }
  return out;
}

/** A classifier stage's name on the screens. */
export function stageName(stage: string): string {
  return stage === 'decider' ? 'decider' : stage === 'qwen_context' ? 'Qwen in context' : 'Qwen';
}

/** The in-context stage's built-in settings: what a router stored before it gets (off). */
export const CONTEXT_STAGE_SEED: Required<Omit<ContextStage, 'question'>> = { mode: 'off', min_tokens: 8192, followups: true, accept: 0.55 };
export function contextStage(router: Pick<RouterDoc, 'classifier'> | null): Required<Omit<ContextStage, 'question'>> & { question?: string } {
  return { ...CONTEXT_STAGE_SEED, ...(router?.classifier.context_stage ?? {}) };
}

/** S1: where a profile's state lines go; a profile without the key keeps them in the reasoning. */
export type StatesChannel = 'reasoning' | 'content' | 'inline';
export function statesChannel(ui: Pick<UiStates, 'states_channel'> | null | undefined): StatesChannel {
  return ui?.states_channel ?? 'reasoning';
}
/** S1: the progress lines' settings with their defaults (server/conductor/states.py PROGRESS_DEFAULT). */
export const PROGRESS_DEFAULT: Required<UiProgress> = { mode: 'on', prefill_every_s: 10, thinking_every_s: 120, thinking_every_tokens: 4096 };
export function progressOf(ui: Pick<UiStates, 'progress'> | null | undefined): Required<UiProgress> {
  return { ...PROGRESS_DEFAULT, ...(ui?.progress ?? {}) };
}

/** What a profile's state lines say, one per kind (server/conductor/states.py TEXTS). */
export const STATE_KINDS: { id: string; example: string }[] = [
  { id: 'route', example: 'deciding: code_edit, high effort' },
  { id: 'compaction', example: 'compaction seen' },
  { id: 'package', example: 'retrieving: skill photo-to-3d' },
  { id: 'memory', example: 'reading memory: 2 notes' },
  { id: 'steer', example: 'steering: gate progress' },
  { id: 'cap', example: 'thinking closed at the 16,384 cap' },
  { id: 'backstop', example: 'thinking closed at the 32,768 backstop' },
  { id: 'loop', example: 'loop detected, steering' },
  { id: 'close', example: 'closing reasoning' },
  { id: 'continue', example: 'empty answer, continuing' },
  { id: 'check', example: 'checking the tool call' },
  { id: 'repair', example: 'repairing the tool call' },
  { id: 'regenerate', example: 'asking for the tool call again' },
  { id: 'give_up', example: 'giving up' },
  // S1: the progress lines, templates with numbers (never the narrator's)
  { id: 'queue', example: "waiting: opencode's title request runs first, position 1" },
  { id: 'prefill', example: 'reading 35,189 tokens at ~750 tok/s, about 47 s left' },
  { id: 'thinking', example: 'thinking 4 min, 9,800 tokens' },
  // N2: a thought that ended; with the narrator on thoughts its summary, "Thought 1m 43s: …"
  { id: 'thought', example: 'thought 1 min, 4,096 tokens' },
  // v3 (narrator live): a tool call's action from its structure, a plan in the answer
  { id: 'tool', example: 'writing mario/render.js: the vector art for the player' },
  { id: 'plan', example: 'plan: level, entities, game and render, then a browser test' },
];

export const SOURCE_WORDS: Record<string, string> = {
  decider: 'decider',
  qwen: 'Qwen',
  qwen_context: 'Qwen in the conversation',
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

// ---------------------------------------------------------------- the role contract (SRV-133)
// The server's rules (server/conductor/schema.py role_rules), so the roles form can say why an
// option is off before the save is refused.
export const ROLE_KINDS: RoleKind[] = ['reasoner', 'decider', 'probe', 'clerk', 'librarian', 'tool', 'generative'];
export const ROLE_WINDOWS: RoleWindow[] = ['idle', 'in_decode', 'any', 'between_rounds'];
export const ROLE_WRITES: RoleWrites[] = ['nothing', 'hidden_notes', 'memory_files', 'client_visible', 'state_lines'];
/** The longest a role between Qwen's rounds may hold Qwen per call (the server's BETWEEN_ROUNDS_MAX_MS). */
export const BETWEEN_ROUNDS_MAX_MS = 1000;
export const SHARING_FORMS: { form: SharingForm; label: string }[] = [
  { form: 'text', label: 'text' },
  { form: 'tokens', label: 'token ids' },
  { form: 'states', label: "Qwen's states" },
];
export const WINDOW_WORDS: Record<RoleWindow, string> = {
  idle: 'between requests, when the engine lock is free',
  in_decode: 'inside the round, beside a decoding Qwen (deciders and probes only)',
  any: 'any time: it is not on the GPU, or it is Qwen through its own lock',
  between_rounds: 'between two of Qwen’s rounds on the request’s thread, Qwen waiting, within its budget (a generative role only)',
};

/** Why `window` is not allowed for `kind`, or null when it is. */
export function windowRefusal(kind: RoleKind, window: RoleWindow): string | null {
  if (window === 'in_decode' && kind !== 'decider' && kind !== 'probe') return 'in decode the GPU belongs to Qwen; only a decider or a probe may run there';
  if (window === 'between_rounds' && kind !== 'generative') return 'between Qwen’s rounds is for a generative role only (a small model inside the engine that Qwen waits for)';
  return null;
}

/** Why `writes` is not allowed for `kind`, or null when it is. */
export function writesRefusal(kind: RoleKind, writes: RoleWrites): string | null {
  if (writes === 'client_visible' && kind !== 'reasoner') return 'only the reasoner writes tokens the client sees (the honesty rule)';
  if (writes === 'state_lines' && kind !== 'generative') return 'state lines are written by a generative role only';
  return null;
}

/** Each form of sharing with whether this role may declare it, and why not: text always, token
 *  ids only with the engine's own tokenizer hash, Qwen's states only for a probe. */
export function sharingOptions(r: Pick<RoleContract, 'kind' | 'tokenizer_sha256'>, engineSha: string | null): { form: SharingForm; label: string; allowed: boolean; why: string | null }[] {
  return SHARING_FORMS.map(({ form, label }) => {
    let why: string | null = null;
    if (form === 'tokens') {
      if (r.tokenizer_sha256 === 'none') why = 'needs the role’s tokenizer hash';
      else if (!engineSha) why = 'the engine recorded no tokenizer hash';
      else if (r.tokenizer_sha256 !== engineSha) why = 'a different tokenizer shares text only';
    }
    if (form === 'states' && r.kind !== 'probe') why = 'a probe inside the engine only';
    return { form, label, allowed: why === null, why };
  });
}

/** The fields a stored version was loaded with, by role, as the server reports them:
 *  ["decider.window=idle", ...] -> { decider: ["window=idle", ...] }. */
export function defaultsByRole(applied: string[] | undefined): Record<string, string[]> {
  const out: Record<string, string[]> = {};
  for (const a of applied ?? []) {
    const i = a.indexOf('.');
    if (i < 0) continue;
    (out[a.slice(0, i)] ??= []).push(a.slice(i + 1));
  }
  return out;
}

// ---------------------------------------------------------------- the package library (R1)
/** The packages a class may list (the library's skills are searched, not listed). */
export function classPackages(doc: Pick<PackagesDoc, 'packages'> | null): string[] {
  return Object.entries(doc?.packages ?? {}).filter(([, p]) => !p.library).map(([n]) => n);
}
/** The library's skills in the document, by name. */
export function libraryNames(doc: Pick<PackagesDoc, 'packages'> | null): string[] {
  return Object.entries(doc?.packages ?? {}).filter(([, p]) => p.library).map(([n]) => n).sort();
}
/** The library's skills a search answered, in its order, then (no query) every one by name;
 *  a name the document does not hold (an older search) is left out. */
export function libraryRows(doc: Pick<PackagesDoc, 'packages'> | null, hits: LibraryHit[] | null): { name: string; score: number | null }[] {
  const names = new Set(libraryNames(doc));
  if (hits) return hits.filter((h) => names.has(h.name)).map((h) => ({ name: h.name, score: h.score }));
  return [...names].map((name) => ({ name, score: null }));
}
