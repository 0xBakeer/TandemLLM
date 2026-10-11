// VIS-80 (QA VIS-81): Wren's tool calls are validated in the browser; the page state carries no
// secret; the figure's state machine.
import { describe, expect, it } from 'vitest';
import model from '../mock/setup/qwen3.8-27b.json';
import type { SetupModel } from '../src/api/types';
import { buildPageState } from '../src/setup/checks';
import { estimate } from '../src/setup/estimate';
import { foldFigure, nextFigureState, FIGURE_STATES } from '../src/setup/figure-state';
import { defaults, emptyState, setupReducer } from '../src/setup/state';
import { fieldKey, planToolCall, TOOL_NAMES, type ToolContext } from '../src/setup/tools';

const M = model as unknown as SetupModel;
const ctx = (over: Partial<ToolContext> = {}): ToolContext => ({ model: M, selection: defaults(M), step: 'components', ...over });

describe('planToolCall', () => {
  it('toggle(vision, false) becomes the same action a click makes, with an undo', () => {
    const p = planToolCall({ name: 'toggle', args: { component: 'vision', on: false } }, ctx());
    expect(p).toMatchObject({ ok: true, kind: 'action', actions: [{ type: 'component', id: 'vision', on: false }], target: { step: 'components', field: 'vision' }, summary: 'Wren turned off Vision' });
    if (!p.ok || p.kind !== 'action') return;
    let st = setupReducer(emptyState(), { type: 'open', model: M }, M);
    for (const a of p.actions) st = setupReducer(st, a, M);
    expect(st.selection!.components.vision).toBe(false);
    for (const a of p.undo) st = setupReducer(st, a, M);
    expect(st.selection!.components.vision).toBe(true);
  });
  it('undo of a cascade puts every dependent back', () => {
    const p = planToolCall({ name: 'toggle', args: { component: 'conductor', on: false } }, ctx());
    if (!p.ok || p.kind !== 'action') throw new Error('expected an action');
    let st = setupReducer(emptyState(), { type: 'open', model: M }, M);
    for (const a of p.actions) st = setupReducer(st, a, M);
    expect(st.selection!.components.narrator).toBe(false);
    for (const a of p.undo) st = setupReducer(st, a, M);
    expect(st.selection!.components).toMatchObject({ conductor: true, narrator: true, laya: true, shadow: true });
  });
  it('set_field(parallelism, slots, 2) and string numbers from a small model', () => {
    expect(planToolCall({ name: 'set_field', args: { step: 'parallelism', field: 'slots', value: 2 } }, ctx({ step: 'parallelism' }))).toMatchObject({ ok: true, actions: [{ type: 'parallel', key: 'slots', value: 2 }] });
    expect(planToolCall({ name: 'set_field', args: '{"step": 4, "field": "slots", "value": "3"}' }, ctx())).toMatchObject({ ok: true, actions: [{ type: 'parallel', key: 'slots', value: 3 }] });
  });
  it('a value outside the range or the enum is rejected with the reason', () => {
    expect(planToolCall({ name: 'set_field', args: { step: 'parallelism', field: 'slots', value: 9 } }, ctx())).toEqual({ ok: false, error: 'The arguments are wrong: slots: at most 4.' });
    expect(planToolCall({ name: 'set_field', args: { step: 'methods', field: 'reasoning_effort', value: 'max' } }, ctx())).toMatchObject({ ok: false, error: expect.stringMatching(/expected one of low, medium, xhigh/) });
  });
  it('an unknown component is rejected', () => {
    expect(planToolCall({ name: 'toggle', args: { component: 'flux', on: true } }, ctx())).toEqual({ ok: false, error: 'This setup has no tool called flux.' });
  });
  it('finish, apply, reset and the like are unknown tools: no handler exists', () => {
    for (const name of ['finish', 'apply', 'reset', 'install', 'unload', 'confirm_reset']) {
      const p = planToolCall({ name, args: {} }, ctx());
      expect(p.ok, name).toBe(false);
      if (!p.ok) expect(p.error).toMatch(/Wren has no tool called/);
    }
    expect(TOOL_NAMES).not.toContain('finish' as never);
  });
  it('Wren cannot open the install step or press Next on Review', () => {
    expect(planToolCall({ name: 'go_to_step', args: { n: 7 } }, ctx())).toMatchObject({ ok: false, error: S_install() });
    expect(planToolCall({ name: 'next', args: {} }, ctx({ step: 'review' }))).toMatchObject({ ok: false });
  });
  it('every way to the install step is refused, and no tool call ever starts a job: Finish is the user\'s click', () => {
    for (const args of [{ n: 7 }, { n: '7' }, { n: 'install' }, { n: 8 }]) {
      const p = planToolCall({ name: 'go_to_step', args }, ctx({ step: 'review' }));
      expect(p.ok, JSON.stringify(args)).toBe(false);
    }
    expect(planToolCall({ name: 'next', args: {} }, ctx({ step: 'review' }))).toMatchObject({ ok: false });
    expect(planToolCall({ name: 'set_field', args: { step: 'install', field: 'x', value: 1 } }, ctx())).toMatchObject({ ok: false });
    // the reducer: Next stops at Review, and even a step change to install starts nothing
    let st = setupReducer(emptyState(), { type: 'open', model: M, step: 'review' }, M);
    st = setupReducer(st, { type: 'next' }, M);
    expect(st.step).toBe('review');
    st = setupReducer(st, { type: 'go', step: 'install' }, M);
    expect(st.job).toBeNull();
    // every action a valid call plans is one of the page's own clicks, never a job
    const calls = [
      { name: 'toggle', args: { component: 'vision', on: false } },
      { name: 'select_profile', args: { id: 'exl3-4.0' } },
      { name: 'set_field', args: { step: 'parallelism', field: 'slots', value: 2 } },
      { name: 'go_to_step', args: { n: 6 } },
      { name: 'next', args: {} },
      { name: 'previous', args: {} },
    ];
    for (const c of calls) {
      const p = planToolCall(c, ctx({ step: 'methods' }));
      if (p.ok && p.kind === 'action') for (const a of [...p.actions, ...p.undo]) expect(a.type, c.name).not.toBe('job');
    }
  });
  it('the page posts /v1/setup/apply in one place, finish(), which only the confirm dialog\'s button calls', async () => {
    const { readFileSync } = await import('node:fs');
    const { fileURLToPath } = await import('node:url');
    const view = readFileSync(fileURLToPath(new URL('../src/setup/view.ts', import.meta.url)), 'utf8');
    const wrenSrc = readFileSync(fileURLToPath(new URL('../src/setup/wren.ts', import.meta.url)), 'utf8');
    expect(view.match(/setupApi\.apply\(/g)).toHaveLength(1);
    expect(view.match(/this\.finish\(\)/g)).toEqual(['this.finish()']);
    expect(view).toMatch(/class="btn btn-primary setup-confirm-go"[^>]*@click=\$\{\(\) => void this\.finish\(\)\}/);
    expect(wrenSrc).not.toMatch(/setupApi\.apply|finish\(\)|confirmOpen/);
  });
  it('a component disabled by a dependency cannot be switched on: the reason goes back', () => {
    const sel = defaults(M);
    sel.components = { ...sel.components, conductor: false, narrator: false, laya: false, shadow: false };
    expect(planToolCall({ name: 'toggle', args: { component: 'shadow', on: true } }, ctx({ selection: sel }))).toEqual({ ok: false, error: 'Needs the conductor' });
  });
  it('security fields are the user’s only', () => {
    expect(planToolCall({ name: 'set_field', args: { step: 'methods', field: 'admin_password', value: 'generate' } }, ctx())).toMatchObject({ ok: false, error: expect.stringMatching(/security setting/) });
  });
  it('a field the user changed after asking is stale: the user wins', () => {
    const touchedAt = new Map([[fieldKey('components', 'vision'), 2000]]);
    expect(planToolCall({ name: 'toggle', args: { component: 'vision', on: false } }, ctx({ touchedAt, turnStartedAt: 1000 }))).toMatchObject({ ok: false, stale: true });
    expect(planToolCall({ name: 'toggle', args: { component: 'vision', on: false } }, ctx({ touchedAt, turnStartedAt: 3000 })).ok).toBe(true);
  });
  it('explain reads the registry; read_device is handed to the panel', () => {
    expect(planToolCall({ name: 'explain', args: { field: 'narrator' } }, ctx())).toMatchObject({ ok: true, kind: 'explain', result: expect.stringMatching(/^Narrator: A small model/) });
    expect(planToolCall({ name: 'read_device', args: {} }, ctx())).toEqual({ ok: true, kind: 'read_device' });
  });
  it('broken arguments and no model', () => {
    expect(planToolCall({ name: 'toggle', args: '{oops' }, ctx())).toMatchObject({ ok: false, error: expect.stringMatching(/JSON object/) });
    expect(planToolCall({ name: 'toggle', args: { component: 'vision', on: false } }, ctx({ model: null, selection: null }))).toMatchObject({ ok: false });
  });
});

function S_install() {
  return 'The install starts only when the user presses Finish.';
}

describe('buildPageState', () => {
  it('is compact, carries the step fields and never a secret', () => {
    const sel = defaults(M);
    sel.advanced = { ...sel.advanced, admin_password: 'generate', api_key: 'sk-secret-1' };
    const page = buildPageState(M, 'methods', sel, estimate(M, defaults(M)));
    const s = JSON.stringify(page);
    expect(s.length).toBeLessThan(6 * 1024);
    expect(s).not.toContain('sk-secret-1');
    expect(page.step_index).toBe(5);
    const pw = (page.visible!.fields as { field: string; value: unknown }[]).find((f) => f.field === 'admin_password');
    expect(pw!.value).toBe('generate');
    expect((page.selection.advanced as Record<string, unknown>).api_key).toBe('<set>');
  });
  it('lists the components with their values on the components step', () => {
    const page = buildPageState(M, 'components', defaults(M), null);
    const vision = (page.visible!.fields as { field: string; value: unknown }[]).find((f) => f.field === 'vision');
    expect(vision).toMatchObject({ type: 'bool', value: true });
  });
});

describe('nextFigureState', () => {
  it('focus, input, send, first token, delta, done: listening, thinking, talking, idle', () => {
    expect(foldFigure(['focus', 'input', 'send', 'first-token', 'delta', 'done'])).toEqual(['listening', 'thinking', 'talking', 'idle']);
  });
  it('a tool call acts, then thinks again', () => {
    expect(foldFigure(['send', 'tool-call', 'acted', 'delta', 'done'])).toEqual(['thinking', 'acting', 'thinking', 'talking', 'idle']);
  });
  it('error and unload win from any state', () => {
    for (const s of FIGURE_STATES) {
      expect(nextFigureState(s, 'error')).toBe('error');
      expect(nextFigureState(s, 'unload')).toBe('sleeping');
    }
  });
  it('a sleeping wren only wakes by loading', () => {
    expect(nextFigureState('sleeping', 'focus')).toBe('sleeping');
    expect(foldFigure(['load-start', 'load-progress', 'loaded'], 'sleeping')).toEqual(['loading', 'idle']);
  });
  it('typing does not interrupt an answer', () => {
    expect(nextFigureState('talking', 'input')).toBe('talking');
    expect(nextFigureState('thinking', 'focus')).toBe('thinking');
  });
});
