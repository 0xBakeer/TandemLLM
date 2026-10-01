// VIS-45's vitest rows for VIS-44 (src/lib/supervisor.ts and the session axis in
// src/lib/conductor.ts): the banner's line (1), the badge and the drawer (2), the shadow style
// (3), the timeline's layout (4), an older server rendering nothing (6), and the header's mode.
// The same scenarios run in the browser in e2e/supervisor.spec.ts.
import { describe, expect, it } from 'vitest';
import type { DecisionRow } from '../src/api/conductor';
import type { LiveConductor, RequestConductor } from '../src/api/types';
import { isCompaction, sessionAxis, tier1Agreement } from '../src/lib/conductor';
import { badge, eventLines, eventWhat, modeWord, supervisorLine, type EventLike } from '../src/lib/supervisor';
import { MockConductor } from '../mock/conductor';

const block = (over: Partial<LiveConductor> = {}): LiveConductor => ({ mode: 'on', acts: 0, shadow: 0, hidden_tokens: 0, gave_up: false, supervising: false, action: null, last: null, last_action: null, ...over });
const d1 = { task: 'D1', decision: 'CLOSE_REASONING', mode: 'act', reason: 'repeat16(share=0.93)', ago_ms: 400, score: 0.93, tier: 0, at_tokens: 2880 };

describe('1: the banner line', () => {
  it('shows the action, its task, its score and where it was taken', () => {
    const l = supervisorLine(block({ acts: 1, last: d1, last_action: d1 }));
    expect(l?.text).toBe('Supervisor: closed the reasoning at 2,880 tokens (D1, score 0.93)');
    expect(l?.shadow).toBe(false);
    expect(l?.title).toContain('repeat16(share=0.93)');
    expect(l?.title).toContain('0.4 s ago');
  });
  it('keeps the last action when a later decision was a pass', () => {
    const pass = { ...d1, decision: 'PASS', score: 0.4, at_tokens: 4096 };
    expect(supervisorLine(block({ acts: 1, last: pass, last_action: d1 }))?.text).toContain('at 2,880 tokens');
  });
  it('falls back to the reason when a rule has no score', () => {
    const e1 = { task: 'E1', decision: 'CONTINUE_HIDDEN', mode: 'act', reason: 'empty_turn(case=B)', ago_ms: 10, score: null, tier: 0, at_tokens: null };
    expect(supervisorLine(block({ acts: 1, last: e1, last_action: e1 }))?.text).toBe('Supervisor: continued an empty turn (E1, empty_turn(case=B))');
  });
  it('is null while the monitor has decided nothing, and with no block', () => {
    expect(supervisorLine(block())).toBeNull();
    expect(supervisorLine(null)).toBeNull();
    expect(supervisorLine(undefined)).toBeNull();
  });
});

describe('3: shadow decisions look different', () => {
  it('a shadow decision with no action says what it would have done', () => {
    const sh = { ...d1, mode: 'shadow', score: 0.71, at_tokens: 2400 };
    const l = supervisorLine(block({ mode: 'shadow', shadow: 1, last: sh }));
    expect(l?.text).toBe('Supervisor (shadow): would have closed the reasoning at 2,400 tokens (D1, score 0.71)');
    expect(l?.shadow).toBe(true);
    expect(badge(block({ shadow: 1 }))).toEqual({ n: 1, kind: 'shadow', label: '1 shadow decision' });
  });
  it('a shadow pass is no decision to show', () => {
    expect(supervisorLine(block({ last: { ...d1, mode: 'shadow', decision: 'PASS' } }))).toBeNull();
  });
});

describe('2: the badge and the drawer', () => {
  it('counts the actions, filled, before any shadow decision', () => {
    expect(badge(block({ acts: 2, shadow: 3 }))).toEqual({ n: 2, kind: 'act', label: '2 supervisor actions' });
    const row: RequestConductor = { acts: 2, shadow: 0, mode: 'on', hidden_tokens: 53, acted: true };
    expect(badge(row)?.n).toBe(2);
    expect(badge(block())).toBeNull();
  });
  it('lists each decision with time, task, tier, mode, score, action, hidden tokens and outcome', () => {
    const evs: EventLike[] = [
      { id: 'b', ts_ms: 2000, task: 'E1', tier: 0, mode: 'act', decision: 'CONTINUE_HIDDEN', score: null, hidden_tokens: 31, outcome: { kind: 'tool_call', finish: 'tool_calls' } },
      { id: 'p', ts_ms: 1500, task: 'SP3', tier: 0, mode: 'act', decision: 'PASS' },
      { id: 'a', ts_ms: 1000, task: 'D1', tier: 0, mode: 'act', decision: 'CLOSE_REASONING', score: 0.91, hidden_tokens: 22, tokens_at_cut: 2304, outcome: { kind: 'tool_call', finish: 'tool_calls' } },
      { id: 's', ts_ms: 3000, task: 'C2', tier: 0, mode: 'shadow', decision: 'REGENERATE_CALL', outcome: null },
    ];
    const lines = eventLines(evs, (ms) => `t${ms}`);
    expect(lines.map((l) => l.id)).toEqual(['a', 'b', 's']);
    expect(lines[0]).toEqual({ id: 'a', time: 't1000', task: 'D1', tier: 'tier 0', mode: 'act', score: '0.91', action: 'closed the reasoning at 2,304 tokens', hidden: '22 tok', outcome: 'tool call · tool_calls' });
    expect(lines[2]).toMatchObject({ mode: 'shadow', action: 'would have had the tool calls written again', hidden: '—', outcome: 'pending' });
  });
  it('keeps the Conductor view\'s words', () => {
    expect(eventWhat({ task: 'C1', decision: 'REGENERATE_CALL', mode: 'shadow', trigger: null, cap: null })).toBe('C1 would have had the tool calls written again');
  });
});

describe('the header', () => {
  it('names the monitor\'s mode, and nothing without one', () => {
    expect(modeWord({ mode: 'shadow' })).toBe('monitor shadow');
    expect(modeWord({ mode: 'off' })).toBe('monitor off');
    expect(modeWord(undefined)).toBeNull();
  });
});

describe('4: the night on one axis', () => {
  const m = new MockConductor(99);
  m.night140();
  const r = m.handle('GET', '/v1/conductor/sessions/c%3Ases_night_140', new URLSearchParams(), {}, 'test');
  const body = r.body as { turns: DecisionRow[]; events: EventLike[] };
  const a = sessionAxis(body.turns, body.events, ['three_d', 'summarize'], (ms) => String(ms));
  it('places 140 turns, 3 actions, 1 compaction and 9 checker results', () => {
    expect(a.turns.length).toBe(140);
    expect(a.acts.filter((x) => !x.shadow).length).toBe(3);
    expect(a.compactions.length).toBe(1);
    expect(a.checkers.length).toBe(9);
    expect(a.checkers.filter((c) => c.pass).length).toBe(1);
  });
  it('keeps every mark on the axis, in time order, with the gate numbers in the titles', () => {
    for (const x of [...a.acts.map((v) => v.x), ...a.compactions.map((v) => v.x), ...a.checkers.map((v) => v.x)]) {
      expect(x).toBeGreaterThanOrEqual(0);
      expect(x).toBeLessThanOrEqual(1);
    }
    const x0 = a.turns.map((t) => t.x0);
    expect([...x0].sort((p, q) => p - q)).toEqual(x0);
    expect(a.turns[0].x0).toBe(0);
    expect(a.checkers[0].title).toMatch(/pagodacheck FAIL: \(a\) [\d.]+ > 1/);
    expect(a.ticks.length).toBe(6);
  });
  it('puts a checker result where it arrived: the next turn\'s start', () => {
    const i = body.turns.findIndex((t) => t.next_checker?.length);
    const next = body.turns[i + 1];
    expect(a.checkers[0].x).toBeCloseTo((next.ts_ms - a.start) / (a.end - a.start), 6);
  });
  it('knows a compaction by its structural class, in shadow too', () => {
    expect(isCompaction({ source: 'structural', shadow: null })).toBe(true);
    expect(isCompaction({ source: 'profile', shadow: { source: 'structural' } })).toBe(true);
    expect(isCompaction({ source: 'sticky', shadow: null })).toBe(false);
  });
  it('says nothing about Tier 1 until it decided something, then its agreement', () => {
    expect(tier1Agreement(body.events)).toBeNull();
    const t1 = [
      { task: 'D3', tier: 1, mode: 'shadow' as const, decision: 'CLOSE_REASONING', label: 'right' },
      { task: 'D3', tier: 1, mode: 'shadow' as const, decision: 'CLOSE_REASONING', label: 'wrong' },
      { task: 'D3', tier: 1, mode: 'shadow' as const, decision: 'CLOSE_REASONING', label: null },
    ];
    expect(tier1Agreement(t1)).toEqual({ decisions: 3, labelled: 2, agree: 1 });
  });
});
