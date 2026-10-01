// The Conductor view's rules (src/lib/conductor.ts), VIS-43's unit rows: the promotion rule
// (41 decisions at 0.93 may act, 12 may not), "not comparable" unless both arms share a build
// and a night, the speed-guard line when no gate ran for the serving build, immutable edits,
// the ribbon's series, and the gate series of a session.
import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import type { DecisionRow, MonitorDoc, MonitorStats, ProfileDoc, RouterDoc, SizeDecision, Stats, WatchSession } from '../src/api/conductor';
import { armName, BUDGET_LEVELS, BUDGET_MAX, budgetText, BY_SIZE_DEFAULT, comparable, levelBudget, withBudget, didWhat, effortsCell, eventWhat, gateSeries, hourlySeries, indicative, isAct, isSessionTask, linesToList, monitorSeries, monitorTaskOrder, promotion, PROMOTE_MIN, readablePath, ruleSchema, setIn, SIZE_SEED, SIZE_SIGNALS, sizeEffort, sizeLine, speedGuard, suitePairs, variantRows, watchLine } from '../src/lib/conductor';

describe('promotion', () => {
  it('is allowed at precision 0.93 over 41 labelled decisions', () => {
    expect(promotion({ labelled: 41, precision: 0.93 })).toEqual({ allowed: true, why: 'precision 0.93 over 41' });
  });
  it('is refused with 12 decisions, and says the rule', () => {
    const p = promotion({ labelled: 12, precision: 0.92 });
    expect(p.allowed).toBe(false);
    expect(p.why).toBe(`12 of ${PROMOTE_MIN} labelled decisions`);
  });
  it('is refused below 0.9 however many', () => {
    expect(promotion({ labelled: 300, precision: 0.89 }).allowed).toBe(false);
    expect(promotion({ labelled: 30, precision: null }).allowed).toBe(false);
    expect(promotion({ labelled: 30, precision: 0.9 }).allowed).toBe(true);
  });
});

describe('eval tables', () => {
  const raw = { suite: 'pagoda', arm: 'raw', build: 'A', night: '2026-10-01' };
  it('marks arms from two builds or two nights not comparable', () => {
    expect(comparable(raw, { ...raw, arm: 'TandemLLM', build: 'B' })).toEqual({ ok: false, why: 'builds A and B' });
    expect(comparable(raw, { ...raw, arm: 'TandemLLM', night: '2026-09-30' }).ok).toBe(false);
    expect(comparable(raw, { ...raw, arm: 'TandemLLM' })).toEqual({ ok: true, why: null });
  });
  it('pairs arms per suite and flags a missing arm', () => {
    const pairs = suitePairs([raw, { ...raw, arm: 'TandemLLM' }, { suite: 'spiral', arm: 'TandemLLM', build: 'A', night: 'x' }]);
    expect(pairs.map((p) => [p.suite, p.cmp.ok, p.cmp.why])).toEqual([
      ['pagoda', true, null],
      ['spiral', false, 'no raw arm'],
    ]);
  });
  it('reads the harness rows: arms raw and tandem, n under 10 indicative', () => {
    const rows = [{ ...raw, arm: 'raw', n: 2 }, { ...raw, arm: 'tandem', n: 2, indicative: true }];
    const p = suitePairs(rows);
    expect(p[0].raw?.arm).toBe('raw');
    expect(p[0].tandem?.arm).toBe('tandem');
    expect(armName('tandem')).toBe('TandemLLM');
    expect(indicative(rows)).toBe(true);
    expect(indicative([{ ...raw, n: 12 }, { ...raw, arm: 'tandem', n: 10 }])).toBe(false);
    expect(indicative([{ ...raw, n: 12, indicative: true }])).toBe(true);
  });
  it('the contract example is a report the screen can show', () => {
    const ex = JSON.parse(readFileSync(join(__dirname, '..', '..', 'docs', 'contract', 'dashboard-v1', 'tandem-eval.example.json'), 'utf8'));
    const p = suitePairs(ex.rows);
    expect(p.map((x) => [x.suite, x.cmp.ok, x.raw?.n, x.tandem?.n])).toEqual([
      ['minicalc (short coding)', true, 2, 2],
      ['spiral portrait', true, 2, 2],
    ]);
    expect(ex.verdict.indicative).toBe(true);
    expect(ex.comparison.map((c: { verdict: string }) => c.verdict)).toEqual(['better', 'better']);
  });
  it('shows the guard line when no speed guard ran for the serving build', () => {
    const engine = { version: '0.3.1-test', code_sha: 'abc' };
    expect(speedGuard([], engine)).toMatchObject({ tone: 'warn', status: 'missing', line: 'speed guard not run for build 0.3.1-test' });
    expect(speedGuard([{ kind: 'speed_guard', build: { code_sha: 'old' } }], engine).status).toBe('missing');
    expect(speedGuard([{ kind: 'speed_guard', build: { code_sha: 'abc' }, verdict: 'pass', runs_each: 3, stats: [] }], engine)).toMatchObject({ tone: 'ok', status: 'pass' });
  });
  it('says what a one-run short gate measured and that it decides nothing (0.3.4-test)', () => {
    const engine = { version: '0.3.4-test', code_sha: '3c64a688569d2967' };
    const rep = {
      kind: 'speed_guard' as const, build: engine, verdict: 'no_noise_estimate', runs_each: 1,
      stats: [
        { stat: 'mean', base: 49.98, cand: 50.54, delta_pct: 1.12, noise_pct: null },
        { stat: 'ttft_p50_ms', base: 424.9, cand: 425.6, delta_pct: 0.16, noise_pct: null },
        { stat: 'p90', base: 68, cand: 68.2, delta_pct: 0.3, noise_pct: null },
      ],
    };
    const g = speedGuard([rep], engine);
    expect(g.tone).toBe('note');
    expect(g.status).toBe('no_noise_estimate');
    expect(g.line).toBe('speed guard for build 0.3.4-test: one row pair, 1 run a side — mean 49.98 → 50.54 tok/s (+1.1 %), TTFT 424.9 → 425.6 ms (+0.2 %)');
    expect(g.detail).toContain('One run gives no noise estimate');
    expect(g.line).not.toContain('not run');
    // the same version on other code is a warning, and says which code
    const other = speedGuard([{ ...rep, build: { version: '0.3.4-test', code_sha: 'zzz' } }], engine);
    expect(other.tone).toBe('warn');
    expect(other.status).toBe('other_code');
    expect(other.detail).toContain('recorded for code zzz');
  });
  it('a resolved-worse guard is a warning with its numbers', () => {
    const engine = { version: '0.3.5-test', code_sha: 'abc' };
    const g = speedGuard([{ kind: 'speed_guard', build: engine, verdict: 'worse', why: 'resolved worse: mean', runs_each: 3, stats: [{ stat: 'mean', base: 50, cand: 45, delta_pct: -10, noise_pct: 2 }] }], engine);
    expect(g).toMatchObject({ tone: 'warn', status: 'worse', line: 'speed guard for build 0.3.5-test: resolved worse: mean' });
    expect(g.detail).toContain('mean 50.00 → 45.00 tok/s (−10.0 %, noise 2.0 %)');
  });
});

describe('editing', () => {
  it('setIn copies the path and leaves the input alone', () => {
    const doc = { classes: [{ id: 'a', policy: { effort: 'low' } }], x: { y: 1 } };
    const next = setIn(doc, ['classes', 0, 'policy', 'effort'], 'xhigh');
    expect(next.classes[0].policy.effort).toBe('xhigh');
    expect(doc.classes[0].policy.effort).toBe('low');
    expect(next.x).toBe(doc.x);
    expect(Array.isArray(next.classes)).toBe(true);
  });
  it('reads a diff path', () => {
    expect(readablePath('$.classes[code_edit].policy.effort')).toBe('classes › code_edit › policy › effort');
    expect(readablePath('$')).toBe('(document)');
  });
});

describe('series', () => {
  it('keeps the busiest classes and folds the rest into other', () => {
    const stats = { by_class: { a: 9, b: 5, c: 3, d: 2, e: 1 }, hourly: [{ by_class: { a: 1, e: 1 } }, { by_class: { d: 2 } }] } as unknown as Stats;
    const s = hourlySeries(stats, ['a', 'b', 'c', 'd', 'e'], 3);
    expect(s.map((x) => x.name)).toEqual(['a', 'b', 'c', 'other']);
    expect(s[3].values).toEqual([1, 2]);
    expect(s[0].color).toBe('cobalt');
  });
  it('turns checker readings into one line per gate', () => {
    const turns = [
      { next_checker: [{ checker: 'pagodacheck', readings: [{ gate: 'a', value: 15.6 }] }] },
      { next_checker: null },
      { next_checker: [{ checker: 'pagodacheck', readings: [{ gate: 'a', value: 3.5 }, { gate: 'b', value: 0.011 }] }] },
    ] as unknown as DecisionRow[];
    const g = gateSeries(turns);
    expect(g.series.map((s) => s.name)).toEqual(['pagodacheck a', 'pagodacheck b']);
    expect(g.series[0].values).toEqual([15.6, null, 3.5]);
  });
  it('says what a decision did', () => {
    const r = { effort: 'xhigh', thinking: 'on', injected: [{ tokens: 1204, parts: [{ kind: 'effort', name: 'xhigh' }, { kind: 'package', name: 'ui-design' }] }], replayed: null, mode: 'on', shadow: null } as unknown as DecisionRow;
    expect(didWhat(r)).toBe('xhigh · ui-design (1,204 tokens)');
    expect(didWhat({ ...r, effort: 'off', thinking: 'off_template', injected: [] })).toBe('thinking off');
  });
});

// SRV-113/115 in the view: the monitor's rules render from the policy's own schema, and its
// decisions read the same in every list.
describe('the session monitor', () => {
  const seeds = JSON.parse(readFileSync(join(__dirname, '..', '..', 'docs', 'contract', 'conductor-v1', 'seeds.json'), 'utf8'));
  const mon = seeds.monitor as MonitorDoc;
  const schema = seeds.meta.schemas.monitor;
  it('finds each rule\'s parameters in the policy schema, rule and mode left out', () => {
    const c1 = ruleSchema(schema, 'risk_floor')!;
    expect(Object.keys(c1.properties!).sort()).toEqual(['floors', 'home', 'root', 'text']);
    expect(c1.properties!.floors.items!.properties!.kind.enum).toEqual(['pattern', 'rm_outside_root', 'write_outside_root']);
    expect(c1.required).not.toContain('rule');
    expect(ruleSchema(schema, 'malformed_call')!.properties!.cut.enum).toEqual(['content', 'give_up']);
    expect(ruleSchema(schema, 'no_such_rule')).toBeNull();
  });
  it('puts the tool checks first, as the policy file lists the rest', () => {
    expect(monitorTaskOrder(mon)).toEqual(['C1', 'C2', 'C3', 'C4', 'D1', 'D2', 'E1', 'S2', 'S1', 'S4']);
    expect(monitorTaskOrder(mon).filter(isSessionTask)).toEqual(['S2', 'S1', 'S4']);
    expect(mon.tasks.C4.mode).toBe('act');
    expect(mon.tasks.C1.mode).toBe('shadow');
  });
  it('says what a decision did, and what a shadow one would have', () => {
    expect(eventWhat({ task: 'C4', decision: 'REGENERATE_CALL', mode: 'act', trigger: 'unreadable(x)', cap: null })).toBe('C4 had the tool calls written again (unreadable(x))');
    expect(eventWhat({ task: 'C1', decision: 'REGENERATE_CALL', mode: 'shadow', trigger: 'floor(git_push)', cap: null })).toBe('C1 would have had the tool calls written again (floor(git_push))');
    expect(eventWhat({ task: 'E1', decision: 'GIVE_UP', mode: 'act', trigger: null, cap: 'per_request' })).toBe('E1 gave up the turn · cap per_request');
    expect(isAct({ mode: 'act', decision: 'PASS' })).toBe(false);
    expect(isAct({ mode: 'shadow', decision: 'CLOSE_REASONING' })).toBe(false);
    expect(isAct({ mode: 'act', decision: 'REPAIR_CALL' })).toBe(true);
  });
  it('draws the acts an hour by task family', () => {
    const m = { hourly: [{ by_family: { 'tool calls': 2 } }, { by_family: { thinking: 1, 'tool calls': 1 } }] } as unknown as MonitorStats;
    const s = monitorSeries(m);
    expect(s.map((x) => [x.name, x.color, x.values])).toEqual([['thinking', 'cobalt', [0, 1]], ['tool calls', 'aqua', [2, 1]]]);
  });
  it('reads a list one item a line', () => {
    expect(linesToList(' /tmp/ \n\n/var/tmp/\n')).toEqual(['/tmp/', '/var/tmp/']);
  });
});

// SRV-121 in the view: what the session tasks hold per session, in one line
describe('the session tasks\' watch', () => {
  const base: WatchSession = { session: 'c:s', requests: 3, last_seen: 0, ended_at: null, running: true, note_at: null, idle_minutes: null, client_gone: false, watch: null, gates: [], compaction: null, progress: null };
  it('names the gate readings, a compaction being watched and the minutes since progress', () => {
    expect(watchLine({ ...base, gates: [{ task: 'S1', checker: 'pagodacheck', gate: 'a', op: '>', limit: 1, values: [3.5, 12.3, 3.5], noted: true }], compaction: { task: 'S2', count: 2, watching: true, steps: 1 }, progress: { task: 'S4', last: null, minutes: 21.25 } })).toBe(
      'pagodacheck (a) 3.5, 12.3, 3.5 · noted · after compaction 2: 1 step, no re-read yet · no progress for 21.25 min',
    );
  });
  it('says when the client has gone, and an idle one', () => {
    expect(watchLine({ ...base, running: false, idle_minutes: 24.8, client_gone: true })).toBe('client gone after 24.8 min');
    expect(watchLine({ ...base, running: false, idle_minutes: 3, progress: { task: 'S4', last: 'edit: changed', minutes: 4 } })).toBe('last progress 4 min ago (edit: changed) · idle 3 min');
    expect(watchLine(base)).toBe('nothing to watch yet');
  });
});

// SRV-157 / SRV-153 in the view: what the size did on a turn, the class's levels under its
// ceiling, the built-in thresholds equal the server's seed, and the report columns.
describe('task size and the classifier digest', () => {
  const seeds = JSON.parse(readFileSync(join(__dirname, '..', '..', 'docs', 'contract', 'conductor-v1', 'seeds.json'), 'utf8'));
  it('says what the size did', () => {
    const on = { mode: 'on', size: 'small', score: 0, now: 'small', signals: { session_tokens: { value: 12000, size: 'small' }, files: { value: 2, size: 'small' } }, ceiling: 'xhigh', effort: 'high', applied: true } as SizeDecision;
    expect(sizeLine(on)).toBe('small (12,000 tokens, 2 files): thinks at high under xhigh');
    expect(sizeLine({ ...on, mode: 'shadow', applied: false })).toBe('small (12,000 tokens, 2 files): would think at high under xhigh (shadow)');
    expect(sizeLine({ ...on, size: 'large', effort: 'xhigh' })).toBe('large (12,000 tokens, 2 files): stays xhigh under xhigh');
    expect(sizeLine({ ...on, effort: undefined, ceiling: undefined })).toBe('small (12,000 tokens, 2 files), on; this class has no size map');
    expect(sizeLine(null)).toBe('—');
  });
  it('keeps a size level under the class effort', () => {
    expect(sizeEffort({ effort: 'xhigh', effort_by_size: { small: 'high' } }, 'small')).toBe('high');
    expect(sizeEffort({ effort: 'xhigh', effort_by_size: { small: 'high' } }, 'large')).toBe('xhigh');
    expect(sizeEffort({ effort: 'medium', effort_by_size: { small: 'xhigh' } }, 'small')).toBe('medium');
  });
  it('holds the built-in thresholds to the server seed', () => {
    expect(SIZE_SEED).toEqual(seeds.router.effort.size);
    expect(SIZE_SIGNALS.map((s) => s.id)).toEqual(seeds.meta.size_signals);
    expect(seeds.router.classes[0].policy.effort_by_size).toEqual({ small: 'high', medium: 'xhigh', large: 'xhigh' });
    expect(seeds.profiles.coding.thinking.by_size).toBe(BY_SIZE_DEFAULT);
    expect(typeof seeds.router.classifier.context_line).toBe('boolean');
  });
  it('reads the digest variants and the efforts column', () => {
    const rep = { kind: 'classifier' as const, variants: { today: { 'tests/fixtures/conductor/followups.jsonl': { n: 66, stages: { decider: { accuracy: 0.1, p50_ms: 300 }, router: { accuracy: 0.8, p50_ms: 300 } } } }, line: { 'tests/fixtures/conductor/followups.jsonl': { n: 66, stages: { decider: { accuracy: 0.9, p50_ms: 320 }, router: { accuracy: 0.9, p50_ms: 320 } } } } } };
    expect(variantRows(rep)).toEqual([
      { variant: 'today', set: 'followups.jsonl', n: 66, decider: 0.1, cascade: null, router: 0.8, p50: 300 },
      { variant: 'line', set: 'followups.jsonl', n: 66, decider: 0.9, cascade: null, router: 0.9, p50: 320 },
    ]);
    expect(variantRows({ kind: 'classifier' })).toEqual([]);
    expect(effortsCell({ efforts: { xhigh: 10, high: 4 }, size_mode: 'on' })).toBe('high 4 · xhigh 10 (size on)');
    expect(effortsCell({ efforts: null })).toBe('—');
  });
});

// SRV-159: real thinking budgets. The levels' caps equal the server's seed, a profile overrides a
// level and an empty field gives it back to Routing, and the schema bounds are the view's.
describe('thinking budgets', () => {
  const seeds = JSON.parse(readFileSync(join(__dirname, '..', '..', 'docs', 'contract', 'conductor-v1', 'seeds.json'), 'utf8'));
  const router = seeds.router as RouterDoc;
  const coding = seeds.profiles.coding as ProfileDoc;
  it('holds the level caps to the server seed', () => {
    const caps = Object.fromEntries(Object.entries(router.effort.levels).map(([k, v]) => [k, v.budget]));
    expect(caps).toEqual({ off: 0, low: 1024, medium: 8192, high: 16384, xhigh: 0 });
    const schema = seeds.meta.schemas.profiles.properties.thinking.properties.budgets;
    expect(Object.keys(schema.properties)).toEqual(BUDGET_LEVELS);
    expect(schema.properties.high.maximum).toBe(BUDGET_MAX);
    expect(coding.thinking.budgets).toBeUndefined();
  });
  it('reads a level from the profile, else from Routing', () => {
    expect(levelBudget('high', router, coding)).toEqual({ cap: 16384, source: 'router' });
    expect(levelBudget('xhigh', router, coding)).toEqual({ cap: 0, source: 'router' });
    const own = { thinking: { ...coding.thinking, budgets: { high: 4096, medium: 0 } } };
    expect(levelBudget('high', router, own)).toEqual({ cap: 4096, source: 'profile' });
    expect(levelBudget('medium', router, own)).toEqual({ cap: 0, source: 'profile' });
    expect(levelBudget('low', router, own)).toEqual({ cap: 1024, source: 'router' });
    expect(levelBudget('low', null, null)).toEqual({ cap: null, source: null });
  });
  it('says the cap', () => {
    expect(budgetText(16384)).toBe('16,384 tokens');
    expect(budgetText(0)).toBe('no cap');
    expect(budgetText(null)).toBe('—');
  });
  it('sets and drops an override without touching the rest', () => {
    const t = coding.thinking;
    const a = withBudget(t, 'high', 4096);
    expect(a.budgets).toEqual({ high: 4096 });
    expect(a.default).toBe(t.default);
    expect(t.budgets).toBeUndefined();
    expect(withBudget(a, 'medium', 999999).budgets).toEqual({ high: 4096, medium: BUDGET_MAX });
    expect(withBudget(a, 'low', -3).budgets).toEqual({ high: 4096, low: 0 });
    expect(withBudget(a, 'high', 4100.6).budgets).toEqual({ high: 4101 });
    const gone = withBudget(a, 'high', null);
    expect('budgets' in gone).toBe(false);
    expect(withBudget(a, 'high', Number.NaN)).not.toHaveProperty('budgets');
  });
});
