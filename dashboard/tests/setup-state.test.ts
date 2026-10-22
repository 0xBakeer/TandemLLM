// VIS-78 (QA VIS-79): the wizard reducer, the dependency rules and the localStorage store.
import { describe, expect, it } from 'vitest';
import model from '../mock/setup/qwen3.8-27b.json';
import flashNext from '../mock/setup/qwen3.8-flash-next.json';
import type { SetupModel } from '../src/api/types';
import { availability, dependentsOf, normaliseComponents, toggleComponent } from '../src/setup/deps';
import { estimate } from '../src/setup/estimate';
import { howToFix, overSentence, stepIssues, tokenBlocked } from '../src/setup/checks';
import { estimateLoose } from '../src/setup/estimate';
import { simpleDefaultsActions } from '../src/setup/simple';
import { capChat, clearSaved, defaults, emptyState, readSaved, restoreSelection, setupReducer, stepPos, STORAGE_KEY, toSaved, windowMax, writeSaved, type SetupAction, type WizardState } from '../src/setup/state';

const M = model as unknown as SetupModel;
const FN = flashNext as unknown as SetupModel;

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
  it('the install job belongs to its model — a different model opens clean, the same one resumes', () => {
    // Back-navigation to #/setup/<other>/install must not replay the first model's job
    // (and its Apply-and-restart) inside the other wizard: the job waits on its own model.
    let s = setupReducer(emptyState(), { type: 'open', model: M }, M);
    s = setupReducer(s, { type: 'job', id: 'job-a' }, M);
    expect(s.job).toBe('job-a');
    expect(s.job_model).toBe(M.id);
    s = setupReducer(s, { type: 'open', model: FN }, FN);
    // the job belongs to its model: it waits there, it does not follow the browser
    expect(s.job).toBe('job-a');
    expect(s.job_model).toBe(M.id);
    s = setupReducer(s, { type: 'job', id: 'job-b' }, FN);
    s = setupReducer(s, { type: 'open', model: FN }, FN);
    expect(s.job).toBe('job-b'); // same model, mid-install: a reload must still resume it
  });
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
  it('a new model starts from what the box runs (the seed), checked, with a note', () => {
    const seed = { ...defaults(M), profile: 'exl3-6.0', slots: 2, components: { ...defaults(M).components, vision: false } };
    const s = run([{ type: 'open', model: M, seed }]);
    expect(s.selection).toMatchObject({ profile: 'exl3-6.0', slots: 2 });
    expect(s.selection!.components.vision).toBe(false);
    expect(s.notice).toMatch(/what this box runs now/);
    // a box that runs the recommended setup: no note
    expect(run([{ type: 'open', model: M, seed: defaults(M) }]).notice).toBeNull();
    // a bad seed value falls back to the recommended one
    const bad = run([{ type: 'open', model: M, seed: { ...defaults(M), profile: 'exl3-9.9' } }]);
    expect(bad.selection!.profile).toBe('exl3-4.5');
  });
  it("this browser's progress wins over the seed", () => {
    const s = run([{ type: 'open', model: M }, { type: 'profile', id: 'exl3-4.0' }, { type: 'open', model: M, seed: { ...defaults(M), profile: 'exl3-6.0' } }]);
    expect(s.selection!.profile).toBe('exl3-4.0');
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
  it("a profile's largest window caps the window, and a switch to it takes the window down (SRV-337)", () => {
    // the registry's packs all reach 262,144 on the Spark today: a copy with a shorter one
    const fn = JSON.parse(JSON.stringify(FN)) as SetupModel;
    fn.profiles.find((p) => p.id === 'flashnext-4.05')!.max_window = 131_072;
    expect(windowMax(fn, 'flashnext-3.05')).toBe(262_144);
    expect(windowMax(fn, 'flashnext-4.05')).toBe(131_072);
    expect(windowMax(fn, 'flashnext-6.05')).toBe(262_144);      // fits none: the estimate says so
    const go = (actions: SetupAction[], st: WizardState = emptyState()) => actions.reduce((s, a) => setupReducer(s, a, fn), st);
    let s = go([{ type: 'open', model: fn }]);
    expect(s.selection!.profile).toBe('flashnext-3.05');
    expect(s.selection!.window).toBe(262_144);
    s = go([{ type: 'profile', id: 'flashnext-4.05' }], s);
    expect(s.selection!.window).toBe(131_072);
    expect(s.selection!.side_window).toBe(131_072);
    expect(s.notice).toMatch(/131,072/);
    s = go([{ type: 'parallel', key: 'window', value: 262_144 }], s);
    expect(s.selection!.window).toBe(131_072);
    expect(estimate(fn, s.selection!).fits).toBe(true);
    // a saved window above the profile's largest falls back inside it
    const r = restoreSelection(fn, { ...s.selection!, window: 262_144 });
    expect(r.selection.window).toBe(131_072);
    expect(r.notices.length).toBe(1);
  });
  it('Flash-Next: the recommended pack fits the Spark at 262,144 with the small models off; 6.05 fits none', () => {
    const d = defaults(FN);
    expect(d.profile).toBe('flashnext-3.05');
    expect(d.slots).toBe(1);
    expect(estimate(FN, d).fits).toBe(true);
    const big = estimate(FN, { ...d, profile: 'flashnext-6.05' });
    expect(big.fits).toBe(false);
    expect(big.context_tokens_fit).toBe(0);
  });
  it('the side window follows the main window down', () => {
    const s = run([{ type: 'open', model: M }, { type: 'parallel', key: 'window', value: 65_536 }]);
    expect(s.selection!.side_window).toBe(65_536);
  });
  it('clamps slots to the engine limit (not the card) and floor to the registry range', () => {
    const s = run([{ type: 'open', model: M }, { type: 'parallel', key: 'slots', value: 40 }, { type: 'parallel', key: 'floor_gb', value: 2 }]);
    expect(s.selection!.slots).toBe(16);
    expect(run([{ type: 'open', model: M }, { type: 'parallel', key: 'slots', value: 9 }]).selection!.slots).toBe(9);
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

describe('simple mode: resume, mode switch, counters', () => {
  const runSimple = (actions: SetupAction[], st: WizardState = emptyState()) =>
    actions.reduce((s, a) => setupReducer(s, a, M, true), st);

  it('a model card click after an abandonment starts over at the quant cards, not the abandoned page', () => {
    // the wizard must not keep the saved 'review' (or leak a route-synced 'overview') and
    // strand a returning visitor away from the quant cards they came for
    let s = runSimple([{ type: 'open', model: M }, { type: 'next' }]);
    expect(s.step).toBe('review');                                   // page 3, then the abandonment
    s = runSimple([{ type: 'tiles' }], s);
    expect(s.step).toBe('models');
    s = runSimple([{ type: 'go', step: 'overview' }], s);            // the card click's route sync
    expect(s.step).toBe('weights');                                  // page 2 — the start, not page 3
  });

  it('opens on the quant cards and walks the three pages, never leaving the mode', () => {
    let s = runSimple([{ type: 'open', model: M }]);
    expect(s.step).toBe('weights');
    s = runSimple([{ type: 'next' }], s);
    expect(s.step).toBe('review');
    s = runSimple([{ type: 'next' }], s);                             // review is the last page: Finish owns it
    expect(s.step).toBe('review');
    s = runSimple([{ type: 'previous' }, { type: 'previous' }, { type: 'previous' }], s);
    expect(s.step).toBe('models');                                    // page 2 is walkable in both directions
  });

  it('a step saved under the standard wizard is re-derived when simple mode opens the model', () => {
    let s = run([{ type: 'open', model: M }, { type: 'go', step: 'parallelism' }]);
    s = setupReducer(s, { type: 'open', model: M }, M, true);
    expect(s.step).toBe('review');                                    // the fold that owns parallelism
    s = setupReducer(s, { type: 'go', step: 'components' }, M, true);
    expect(s.step).toBe('review');
  });

  it('the mode switch re-derives the step through the entered mode', () => {
    let s = run([{ type: 'open', model: M }, { type: 'go', step: 'parallelism' }]);
    s = setupReducer(s, { type: 'mode', simple: true }, M, false);
    expect(s.step).toBe('review');                                    // folded away
    s = setupReducer(s, { type: 'mode', simple: false }, M, true);
    expect(s.step).toBe('review');                                    // a page of both modes
    let t = run([{ type: 'open', model: M, step: 'weights' }]);
    t = setupReducer(t, { type: 'mode', simple: true }, M, false);
    expect(t.step).toBe('weights');
    let u = run([{ type: 'open', model: M }, { type: 'go', step: 'overview' }]);
    u = setupReducer(u, { type: 'mode', simple: true }, M, false);
    expect(u.step).toBe('weights');                                   // the intro's twin is the quant cards
  });

  it('the counter counts the active mode\'s pages, never the other mode\'s', () => {
    expect(stepPos('weights', true)).toEqual({ n: 2, of: 3 });
    expect(stepPos('review', true)).toEqual({ n: 3, of: 3 });
    expect(stepPos('models', true)).toEqual({ n: 1, of: 3 });
    expect(stepPos('install', true)).toEqual({ n: 3, of: 3 });        // the job sits on the folded page
    expect(stepPos('parallelism', true)).toEqual({ n: 3, of: 3 });    // a standard-mode step still counts under the simple pages
    expect(stepPos('weights', false)).toEqual({ n: 3, of: 8 });
    expect(stepPos('parallelism', false)).toEqual({ n: 5, of: 8 });
  });
});

describe('dependencies', () => {
  const sel = defaults(M).components;
  it('turning the conductor off turns the narrator, Laya and the shadow reader off with a reason', () => {
    // the narrator switched on first: off by default since 0.11.3 (SRV-534)
    const r = toggleComponent(M, { ...sel, narrator: true }, 'conductor', false);
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
    const { selection, notices } = restoreSelection(M, { ...defaults(M), slots: 17, methods: { reasoning_effort: 'max' } });
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
    // OPS-95 made every published profile public, so the private ones are marked here, as the
    // mock's privateProfile option does for the e2e case
    const sel = defaults(M);
    const P = { ...M, profiles: M.profiles.map((p) => (p.id === sel.profile || p.id === 'exl3-6.0' ? { ...p, hf_private: true } : p)) };
    const dev = { profiles_on_disk: { 'exl3-6.0': true }, hf_token_present: false } as never;
    expect(tokenBlocked(P, sel, dev)).toBe(true);
    expect(tokenBlocked(P, sel, { ...(dev as object), hf_token_present: true } as never)).toBe(false);
    expect(tokenBlocked(P, { ...sel, profile: 'exl3-6.0' }, dev)).toBe(false);
    expect(tokenBlocked(P, { ...sel, profile: 'nvfp4' }, dev)).toBe(false);
    expect(tokenBlocked(M, sel, dev)).toBe(false);
  });
});

describe('the simple answers', () => {
  const FLASH = flashNext as unknown as SetupModel;
  const sel = () => ({ ...defaults(FLASH), slots: 1, kv_pool: false });
  const capped = (max: number): SetupModel => {
    const c = JSON.parse(JSON.stringify(FLASH));
    c.parallelism.slots = { min: 1, max, default: Math.min(max, c.parallelism.slots.default) };
    return c as SetupModel;
  };
  const valueOf = (acts: SetupAction[], key: string) => {
    const a = acts.find((x) => x.type === 'parallel' && x.key === key) as { value?: unknown } | undefined;
    return a?.value;
  };
  it('asks for no more chats than the card measured', () => {
    for (const max of [1, 2, 3]) {
      const n = valueOf(simpleDefaultsActions(capped(max), sel()), 'slots') as number | undefined;
      expect(n ?? 1).toBeLessThanOrEqual(max);
    }
  });
  const pooledCard = (kvb: number): SetupModel => {
    // a four-chat card of the Flash-Next shape whose KV costs kvb bytes a token: the
    // plain reservation commits every chat's WHOLE window at load, the pool commits the
    // typical fill as chats grow -- at 44,000 the plain reservation cannot stretch past one
    // chat of the 262,144 window, at 16,000 it carries all four and the pool earns nothing
    const c = JSON.parse(JSON.stringify(FLASH));
    c.parallelism.slots = { min: 1, max: 4, default: 4 };
    c.parallelism.kv_bytes_per_token = kvb;
    for (const pr of c.profiles) delete pr.kv_bytes_per_token;   // the card, not a profile, prices it
    return c as SetupModel;
  };
  it('turns the pool on when it buys chats the plain reservation cannot carry', () => {
    const buy = pooledCard(44_000);
    const acts = simpleDefaultsActions(buy, sel());
    const n = valueOf(acts, 'slots') as number;
    expect(n).toBe(4);
    expect(valueOf(acts, 'kv_pool')).toBe(true);
    // the witness that it buys: at that very chat count the plain reservation does not fit
    const w = windowMax(buy, sel().profile);
    expect(estimateLoose(buy, { ...sel(), window: w, kv_pool: false, slots: n })?.fits).toBe(false);
  });
  it('leaves the pool as the box left it when the plain reservation already carries the chats', () => {
    const free = pooledCard(16_000);
    const acts = simpleDefaultsActions(free, sel());
    expect(valueOf(acts, 'kv_pool')).toBeUndefined();   // no gain, no switch: as the box left it
    expect(valueOf(acts, 'slots')).toBe(4);             // and the four chats came without it
  });
});
