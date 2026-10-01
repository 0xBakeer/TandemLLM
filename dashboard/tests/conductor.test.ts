// The Conductor view's rules (src/lib/conductor.ts), VIS-43's unit rows: the promotion rule
// (41 decisions at 0.93 may act, 12 may not), "not comparable" unless both arms share a build
// and a night, the speed-guard line when no gate ran for the serving build, immutable edits,
// the ribbon's series, and the gate series of a session.
import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import type { DecisionRow, MonitorDoc, MonitorStats, Stats, WatchSession } from '../src/api/conductor';
import { armName, comparable, didWhat, eventWhat, gateSeries, hourlySeries, indicative, isAct, isSessionTask, linesToList, monitorSeries, monitorTaskOrder, promotion, PROMOTE_MIN, readablePath, ruleSchema, setIn, speedGuard, suitePairs, watchLine } from '../src/lib/conductor';

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
    expect(speedGuard([], engine)).toBe('speed guard not run for build 0.3.1-test');
    expect(speedGuard([{ kind: 'speed_guard', build: { code_sha: 'old' } }], engine)).not.toBeNull();
    expect(speedGuard([{ kind: 'speed_guard', build: { code_sha: 'abc' } }], engine)).toBeNull();
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
