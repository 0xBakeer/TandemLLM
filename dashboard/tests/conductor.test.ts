// The Conductor view's rules (src/lib/conductor.ts), VIS-43's unit rows: the promotion rule
// (41 decisions at 0.93 may act, 12 may not), "not comparable" unless both arms share a build
// and a night, the speed-guard line when no gate ran for the serving build, immutable edits,
// the ribbon's series, and the gate series of a session.
import { describe, expect, it } from 'vitest';
import type { DecisionRow, Stats } from '../src/api/conductor';
import { comparable, didWhat, gateSeries, hourlySeries, promotion, PROMOTE_MIN, readablePath, setIn, speedGuard, suitePairs } from '../src/lib/conductor';

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
