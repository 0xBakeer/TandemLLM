// VIS-78 (QA VIS-79): the wizard reducer, the dependency rules and the localStorage store.
import { describe, expect, it } from 'vitest';
import model from '../mock/setup/qwen3.8-27b.json';
import type { SetupModel } from '../src/api/types';
import { availability, dependentsOf, normaliseComponents, toggleComponent } from '../src/setup/deps';
import { estimate } from '../src/setup/estimate';
import { howToFix, overSentence, stepIssues, tokenBlocked } from '../src/setup/checks';
import { capChat, clearSaved, defaults, emptyState, readSaved, restoreSelection, setupReducer, STORAGE_KEY, toSaved, writeSaved, type SetupAction, type WizardState } from '../src/setup/state';

const M = model as unknown as SetupModel;

class MemStore implements Storage {
  data = new Map<string, string>();
  get length() {
    return this.data.size;
  }
  clear() {
    this.data.clear();
  }
  getItem(k: string) {
    return this.data.has(k) ? this.data.get(k)! : null;
  }
  key(i: number) {
    return [...this.data.keys()][i] ?? null;
  }
  removeItem(k: string) {
    this.data.delete(k);
  }
  setItem(k: string, v: string) {
    this.data.set(k, v);
  }
}

const run = (actions: SetupAction[], st: WizardState = emptyState()) => actions.reduce((s, a) => setupReducer(s, a, M), st);

describe('defaults', () => {
  it('pre-selects the recommended profile and every default component', () => {
    const d = defaults(M);
    expect(d.profile).toBe('exl3-4.5');
    for (const c of M.components) expect(d.components[c.id]).toBe(c.required || c.default);
    expect(d.slots).toBe(4);
    expect(d.window).toBe(262_144);
    expect(d.kv_pool).toBe(true);
    expect(d.methods.sampled_tree).toBe('det');
    expect(d.advanced.admin_password).toBe('keep');
  });
  it('the recommended setup fits the DGX Spark', () => {
    expect(estimate(M, defaults(M)).fits).toBe(true);
  });
});

describe('setupReducer', () => {
  it('opens a model on the overview and walks Next and Previous', () => {
    let s = run([{ type: 'open', model: M }]);
    expect(s.step).toBe('overview');
    s = run([{ type: 'go', step: 'weights' }, { type: 'next' }, { type: 'next' }, { type: 'next' }, { type: 'previous' }], s);
    expect(s.step).toBe('parallelism');
    expect(s.visited).toEqual(['models', 'overview', 'weights', 'components', 'parallelism', 'methods']);
  });
  it('Next stops at Review: the install is Finish, never Next', () => {
    const s = run([{ type: 'open', model: M, step: 'review' }, { type: 'next' }]);
    expect(s.step).toBe('review');
  });
  it('Previous stops at the tiles', () => {
    const s = run([{ type: 'open', model: M }, { type: 'previous' }, { type: 'previous' }]);
    expect(s.step).toBe('models');
  });
  it('keeps the selection when the same model opens again (deep link)', () => {
    const s = run([{ type: 'open', model: M }, { type: 'profile', id: 'exl3-4.0' }, { type: 'tiles' }, { type: 'open', model: M, step: 'review' }]);
    expect(s.selection!.profile).toBe('exl3-4.0');
    expect(s.step).toBe('review');
  });
  it('ignores an unknown profile', () => {
    const s = run([{ type: 'open', model: M }, { type: 'profile', id: 'exl3-9.9' }]);
    expect(s.selection!.profile).toBe('exl3-4.5');
  });
  it('snaps and clamps the window, with a note', () => {
    let s = run([{ type: 'open', model: M }, { type: 'parallel', key: 'window', value: 100_000 }]);
    expect(s.selection!.window).toBe(98_304);
    expect(s.notice).toMatch(/98,304/);
    s = run([{ type: 'parallel', key: 'window', value: 999_999 }], s);
    expect(s.selection!.window).toBe(262_144);
    expect(s.selection!.side_window).toBeLessThanOrEqual(262_144);
  });
  it('the side window follows the main window down', () => {
    const s = run([{ type: 'open', model: M }, { type: 'parallel', key: 'window', value: 65_536 }]);
    expect(s.selection!.side_window).toBe(65_536);
  });
  it('clamps slots and floor to the registry range', () => {
    const s = run([{ type: 'open', model: M }, { type: 'parallel', key: 'slots', value: 9 }, { type: 'parallel', key: 'floor_gb', value: 2 }]);
    expect(s.selection!.slots).toBe(4);
    expect(s.selection!.floor_gb).toBe(10);
  });
  it('sets a method and resets to the recommended values', () => {
    let s = run([{ type: 'open', model: M }, { type: 'field', group: 'methods', id: 'reasoning_effort', value: 'xhigh' }]);
    expect(s.selection!.methods.reasoning_effort).toBe('xhigh');
    s = run([{ type: 'recommended', scope: 'methods' }], s);
    expect(s.selection!.methods.reasoning_effort).toBe('medium');
  });
  it('clear goes back to an empty wizard', () => {
    const s = run([{ type: 'open', model: M }, { type: 'clear' }]);
    expect(s).toMatchObject({ model: null, step: 'models', selection: null, chat: [] });
  });
});

describe('dependencies', () => {
  const sel = defaults(M).components;
  it('turning the conductor off turns the narrator, Laya and the shadow reader off with a reason', () => {
    const r = toggleComponent(M, sel, 'conductor', false);
    expect(r.ok).toBe(true);
    if (!r.ok) return;
    expect(r.components).toMatchObject({ conductor: false, narrator: false, laya: false, shadow: false });
    expect(r.cascaded.map((c) => c.id).sort()).toEqual(['laya', 'narrator', 'shadow']);
    expect(r.notice).toMatch(/need the conductor, so they are off too/);
    const av = availability(M, r.components);
    expect(av.narrator).toMatchObject({ disabled: true, reason: 'Needs the conductor' });
  });
  it('a dependent cannot be switched on while its dependency is off', () => {
    const off = { ...sel, disk_tier: false, nvme_spill: false };
    const r = toggleComponent(M, off, 'nvme_spill', true);
    expect(r).toEqual({ ok: false, error: 'Needs the disk cache' });
  });
  it('the drafters are required: always on, never off', () => {
    expect(availability(M, sel).drafters).toMatchObject({ disabled: true, required: true });
    expect(toggleComponent(M, sel, 'drafters', false).ok).toBe(false);
  });
  it('dependents are found through chains', () => {
    expect(dependentsOf(M, 'disk_tier').map((c) => c.id)).toEqual(['nvme_spill']);
  });
  it('normalising drops a component whose dependency is off', () => {
    expect(normaliseComponents(M, { conductor: false, narrator: true }).narrator).toBe(false);
  });
  it('a conflict disables the other side with a reason', () => {
    const m = { ...M, components: M.components.map((c) => (c.id === 'response_cache' ? { ...c, conflicts: ['lookup'] } : c)) };
    const av = availability(m, { ...sel, lookup: true, response_cache: false });
    expect(av.response_cache).toMatchObject({ disabled: true, reason: 'Cannot run with the lookup store' });
  });
});

describe('restoreSelection', () => {
  it('falls back for a profile or a component that no longer exists, with notes', () => {
    const { selection, notices } = restoreSelection(M, { ...defaults(M), profile: 'exl3-9.9', components: { flux: true, vision: false } });
    expect(selection.profile).toBe('exl3-4.5');
    expect(selection.components.vision).toBe(false);
    expect(notices.join(' ')).toMatch(/exl3-9.9/);
    expect(notices.join(' ')).toMatch(/flux/);
  });
  it('rejects out-of-range numbers and bad enum values', () => {
    const { selection, notices } = restoreSelection(M, { ...defaults(M), slots: 12, methods: { reasoning_effort: 'max' } });
    expect(selection.slots).toBe(4);
    expect(selection.methods.reasoning_effort).toBe('medium');
    expect(notices).toHaveLength(2);
  });
});

describe('localStorage tandem.setup.v1', () => {
  it('saves and restores step, selection and chat', () => {
    const store = new MemStore();
    const s = run([{ type: 'open', model: M }, { type: 'profile', id: 'exl3-4.0' }, { type: 'component', id: 'vision', on: false }, { type: 'go', step: 'parallelism' }, { type: 'chat', chat: [{ role: 'user', content: 'hi' }] }]);
    expect(writeSaved(store, s)).toBe(true);
    const back = readSaved(store);
    expect(back.problem).toBeNull();
    expect(back.state.step).toBe('parallelism');
    expect(back.state.selection!.profile).toBe('exl3-4.0');
    expect(back.state.selection!.components.vision).toBe(false);
    expect(back.state.chat).toEqual([{ role: 'user', content: 'hi' }]);
    expect(JSON.parse(store.getItem(STORAGE_KEY)!).v).toBe(1);
  });
  it('corrupt JSON and an unknown version are discarded with the reason', () => {
    const store = new MemStore();
    store.setItem(STORAGE_KEY, '{not json');
    expect(readSaved(store)).toMatchObject({ problem: 'corrupt', state: { model: null } });
    store.setItem(STORAGE_KEY, JSON.stringify({ v: 0, model: 'qwen3.8-27b' }));
    expect(readSaved(store)).toMatchObject({ problem: 'version', state: { model: null } });
  });
  it('storage that throws is reported as unavailable and writes fail quietly', () => {
    const bad = { getItem: () => { throw new Error('denied'); }, setItem: () => { throw new Error('quota'); }, removeItem: () => { throw new Error('denied'); } } as unknown as Storage;
    expect(readSaved(bad).problem).toBe('unavailable');
    expect(readSaved(null).problem).toBe('unavailable');
    expect(writeSaved(bad, emptyState())).toBe(false);
    expect(clearSaved(bad)).toBe(false);
  });
  it('never stores a typed secret, and the install step resumes at Review', () => {
    const s = run([{ type: 'open', model: M }]);
    const st: WizardState = { ...s, step: 'install', selection: { ...s.selection!, advanced: { ...s.selection!.advanced, admin_password: 'hunter2hunter2', api_key: 'sk-1' } } };
    const saved = toSaved(st);
    expect(JSON.stringify(saved)).not.toMatch(/hunter2|sk-1/);
    expect(saved.step).toBe('review');
  });
  it('the chat is capped at 40 messages and 64 KB, oldest first, never starting on a tool answer', () => {
    const many = Array.from({ length: 60 }, (_, i) => ({ role: (i % 2 ? 'assistant' : 'user') as 'user' | 'assistant', content: `m${i}` }));
    expect(capChat(many)).toHaveLength(40);
    expect(capChat(many)[0].content).toBe('m20');
    const big = Array.from({ length: 10 }, (_, i) => ({ role: 'user' as const, content: 'x'.repeat(10_000) + i }));
    const capped = capChat(big);
    expect(new TextEncoder().encode(JSON.stringify(capped)).length).toBeLessThanOrEqual(64 * 1024);
    expect(capped[capped.length - 1].content.endsWith('9')).toBe(true);
    expect(capChat([{ role: 'tool', content: 'ok', tool_call_id: 'a' }, { role: 'user', content: 'x' }])[0].role).toBe('user');
  });
});

describe('the memory check on Review', () => {
  it('exl3-6.0, every tool, 4 slots of 262,144 without the pool does not fit; How to fix names the pool first', () => {
    const all = Object.fromEntries(M.components.map((c) => [c.id, true]));
    const sel = { ...defaults(M), profile: 'exl3-6.0', components: normaliseComponents(M, all), kv_pool: false };
    const e = estimate(M, sel);
    expect(e.fits).toBe(false);
    expect(stepIssues(M, sel, null, e).review?.[0]).toMatch(/does not fit/);
    const fixes = howToFix(M, sel);
    expect(fixes[0].text).toBe('Turn on the shared memory pool');
    expect(fixes.some((f) => /Lower the longest conversation/.test(f.text))).toBe(true);
    expect(overSentence(M, sel, e)).toMatch(/^Over by \d+\.\d GiB\. Turn on the shared memory pool, or use 1 parallel chat\.$/);
    // lowering the window until it fits
    expect(estimate(M, { ...sel, window: 65_536, side_window: 65_536 }).fits).toBe(true);
  });
  it('a private profile that is not on the box and no token blocks the weights step', () => {
    const sel = defaults(M);
    const dev = { profiles_on_disk: { 'exl3-6.0': true }, hf_token_present: false } as never;
    expect(tokenBlocked(M, sel, dev)).toBe(true);
    expect(tokenBlocked(M, { ...sel, profile: 'exl3-6.0' }, dev)).toBe(false);
    expect(tokenBlocked(M, { ...sel, profile: 'nvfp4' }, dev)).toBe(false);
  });
});
