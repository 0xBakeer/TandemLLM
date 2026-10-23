// The wizard's whole state (VIS-78): one reducer, `setupReducer`, every change goes through it (the
// user's clicks and Wren's tool calls alike, so the rules and the memory bar react the same way).
// Saved to localStorage `tandem.setup.v1` as {v: 1, model, step, selection, visited, updatedAt, chat};
// every access is in try/catch (private mode, quota). A typed secret is never part of the state.

import type { SetupChatMessage, SetupModel, SetupSelection, SetupStepId, SetupValue } from '../api/types';
import { packVisionOff } from './community';
import { normaliseComponents, toggleComponent } from './deps';
import { fieldError, isStepId, registryField, slotsMax, STEP_IDS, stepIndex } from './fields';
import { S } from './strings';

export const STORAGE_KEY = 'tandem.setup.v1';
export const STORAGE_VERSION = 1;
export const CHAT_MAX_MESSAGES = 40;
export const CHAT_MAX_BYTES = 64 * 1024;

export interface WizardState {
  model: string | null;
  step: SetupStepId;
  selection: SetupSelection | null;
  visited: SetupStepId[];
  chat: SetupChatMessage[];
  updatedAt: number;
  /** the install job this browser started (a reload during the install resumes it) */
  job: string | null;
  /** the model that job was started for: the progress page belongs to it and only it */
  job_model: string | null;
  /** a one-line note for the user (a cascade, a clamp); not saved */
  notice: string | null;
}

export function emptyState(): WizardState {
  return { model: null, step: 'models', selection: null, visited: ['models'], chat: [], updatedAt: 0, job: null, job_model: null, notice: null };
}

/** The recommended setup of a model: what the wizard shows on a first visit. */
export function defaults(m: SetupModel): SetupSelection {
  const rec = m.profiles.find((p) => p.recommended) ?? m.profiles[0];
  const par = m.parallelism;
  const methods: Record<string, SetupValue> = {};
  for (const f of m.methods) methods[f.id] = f.recommended;
  const advanced: Record<string, SetupValue> = {};
  for (const f of m.advanced) advanced[f.id] = f.recommended;
  return {
    model: m.id,
    profile: rec.id,
    components: normaliseComponents(m, {}),
    slots: par.slots.default,
    window: par.window.default,
    side_window: par.side_window.default,
    kv_pool: par.kv_pool.default,
    floor_gb: par.floor_gb.default,
    methods,
    advanced,
    device: m.devices[0]?.id,
  };
}

/** The box's own setup as the start, checked like a saved one; the note only when it is not the
 *  recommended setup anyway. */
function seeded(m: SetupModel, seed: SetupSelection): { selection: SetupSelection; notices: string[] } {
  const selection = restoreSelection(m, seed).selection;
  const same = JSON.stringify(selection) === JSON.stringify(defaults(m));
  return { selection, notices: same ? [] : [S.seededFromBox] };
}

/** Check a saved selection against the registry as it is now; bad parts fall back with a note. */
export function restoreSelection(m: SetupModel, saved: unknown): { selection: SetupSelection; notices: string[] } {
  const d = defaults(m);
  const notices: string[] = [];
  if (!saved || typeof saved !== 'object') return { selection: d, notices };
  const s = saved as Partial<SetupSelection>;
  const out: SetupSelection = { ...d, methods: { ...d.methods }, advanced: { ...d.advanced } };
  if (typeof s.profile === 'string') {
    if (m.profiles.some((p) => p.id === s.profile)) out.profile = s.profile;
    else notices.push(S.savedProfileGone(s.profile));
  }
  if (s.components && typeof s.components === 'object') {
    const comps: Record<string, boolean> = {};
    for (const [k, v] of Object.entries(s.components)) {
      if (!m.components.some((c) => c.id === k)) notices.push(S.savedComponentGone(k));
      else if (typeof v === 'boolean') comps[k] = v;
    }
    out.components = normaliseComponents(m, comps);
  }
  const par = m.parallelism;
  const num = (v: unknown, lo: number, hi: number, int: boolean, label: string, fb: number): number => {
    if (v === undefined) return fb;
    if (typeof v === 'number' && Number.isFinite(v) && v >= lo && v <= hi && (!int || Number.isInteger(v))) return v;
    notices.push(S.savedValueBad(label));
    return fb;
  };
  out.slots = num(s.slots, par.slots.min, slotsMax(m), true, S.slots, Math.min(d.slots, slotsMax(m)));
  const wmax = windowMax(m, out.profile);
  out.window = num(s.window, par.window.min, wmax, true, S.window, Math.min(d.window, wmax));
  out.side_window = num(s.side_window, par.side_window.min, par.side_window.max, true, S.sideWindow, d.side_window ?? d.window);
  out.floor_gb = num(s.floor_gb, par.floor_gb.min, par.floor_gb.max, false, S.floor, d.floor_gb);
  if (typeof s.kv_pool === 'boolean') out.kv_pool = s.kv_pool;
  for (const group of ['methods', 'advanced'] as const) {
    const vals = s[group];
    if (!vals || typeof vals !== 'object') continue;
    for (const f of m[group]) {
      if (!(f.id in vals)) continue;
      const v = (vals as Record<string, unknown>)[f.id];
      if (fieldError(f, v) == null) out[group][f.id] = v as SetupValue;
      else notices.push(S.savedValueBad(f.label));
    }
  }
  if (typeof s.device === 'string' && m.devices.some((x) => x.id === s.device)) out.device = s.device;
  if (packVisionOff(m, out)) out.components = { ...out.components, vision: false };
  return { selection: out, notices };
}

export type ParallelKey = 'slots' | 'window' | 'side_window' | 'kv_pool' | 'floor_gb';

// The active mode's page list, in one place: the reducer walks it, the counter reads it, and a
// saved or routed step is normalised through it. Simple mode's pages are the model cards (the
// tiles, page 1), the quant cards, and the one folded settings page (simple.ts's design); the
// install job lives on the folded page too. Before this, each view branch carried its own
// ternary over 'weights'/3, and a step that belonged to the other mode (a resumed
// 'parallelism', a route-synced 'overview') rendered simple pages under a standard counter and
// let Previous/Next walk out of the mode; the one list in one place closes that.

const SIMPLE_FLOW: readonly SetupStepId[] = ['models', 'weights', 'review'];

/** The page list of the active mode. */
export function activeSteps(simple: boolean): readonly SetupStepId[] {
  return simple ? SIMPLE_FLOW : STEP_IDS;
}

/** The step as the active mode understands it. The install job is outside both walks (Finish
 *  owns it); everything the simple fold holds collapses onto the folded page, and the tiles'
 *  twin — the standard intro — opens on the quant cards, the first in-model simple page. */
export function normalizeStep(step: SetupStepId, simple: boolean): SetupStepId {
  if (!simple || step === 'models' || step === 'weights' || step === 'review' || step === 'install') return step;
  return step === 'overview' ? 'weights' : 'review';
}

/** Where a step counts in the active mode's pages, 1-based — the single source of the "n of m"
 *  counter. An install under simple mode counts as the folded page. */
export function stepPos(step: SetupStepId, simple: boolean): { n: number; of: number } {
  const list = activeSteps(simple);
  const s = step === 'install' && simple ? 'review' : normalizeStep(step, simple);
  const i = list.indexOf(s);
  return { n: (i < 0 ? 0 : i) + 1, of: list.length };
}

export type SetupAction =
  /** seed: what the box runs now (GET .../current), used when this browser has no progress for
   *  the model; checked like a saved selection */
  | { type: 'open'; model: SetupModel; step?: SetupStepId; seed?: SetupSelection | null }
  | { type: 'go'; step: SetupStepId }
  | { type: 'mode'; simple: boolean }
  | { type: 'next' }
  | { type: 'previous' }
  | { type: 'tiles' }
  | { type: 'profile'; id: string }
  | { type: 'component'; id: string; on: boolean }
  | { type: 'components'; set: Record<string, boolean> }
  | { type: 'parallel'; key: ParallelKey; value: number | boolean }
  | { type: 'field'; group: 'methods' | 'advanced'; id: string; value: SetupValue }
  | { type: 'recommended'; scope: 'components' | 'parallelism' | 'methods' }
  | { type: 'chat'; chat: SetupChatMessage[] }
  | { type: 'job'; id: string | null }
  | { type: 'dismiss' }
  | { type: 'clear' };

function visit(v: SetupStepId[], step: SetupStepId): SetupStepId[] {
  return v.includes(step) ? v : [...v, step];
}

/** The longest window a profile takes: the model's, or the profile's max_window where that is shorter
 * (SRV-337: a Flash-Next pack on the Spark). A profile that fits no window (null) keeps the model's
 * range; its estimate says it does not fit. */
export function windowMax(m: SetupModel, profile: string): number {
  const mw = m.profiles.find((p) => p.id === profile)?.max_window;
  return typeof mw === 'number' ? Math.min(m.parallelism.window.max, mw) : m.parallelism.window.max;
}

/** Snap a window to the model's step and range (and `max`, the profile's largest); the note says when it moved. */
export function clampWindow(m: SetupModel, key: 'window' | 'side_window', n: number, max = m.parallelism[key].max): { value: number; clamped: boolean } {
  const r = m.parallelism[key];
  let v = Math.round(n / r.step) * r.step;
  v = Math.min(r.max, max, Math.max(r.min, v));
  return { value: v, clamped: v !== n };
}

/** The pure reducer. `m` is the open model (null on the tiles); `simple` is the browser's mode
 *  (simple.ts's localStorage bit) — every step-bearing action is normalised against it, so a
 *  resumed or routed step can never belong to the other mode. */
export function setupReducer(st: WizardState, a: SetupAction, m: SetupModel | null, simple = false): WizardState {
  const now = Date.now();
  const done = (next: Partial<WizardState>): WizardState => ({ ...st, notice: null, ...next, updatedAt: now });
  switch (a.type) {
    case 'open': {
      const same = st.model === a.model.id && st.selection;
      const { selection, notices } = same
        ? restoreSelection(a.model, st.selection)
        : a.seed
          ? seeded(a.model, a.seed)
          : { selection: defaults(a.model), notices: [] as string[] };
      const step = normalizeStep(a.step && a.step !== 'models' ? a.step : same && st.step !== 'models' ? st.step : 'overview', simple);
      // the install job belongs to the model it was started for: opening another model keeps
      // the job where it is (the progress page shows for its own model only), and opening the
      // same model again resumes it untouched.
      return done({ model: a.model.id, selection, step, visited: visit(same ? st.visited : ['models'], step), notice: notices.length ? notices.join(' ') : null });
    }
    case 'go': {
      if (a.step !== 'models' && !st.selection) return st;
      const step = normalizeStep(a.step, simple);
      return done({ step, visited: visit(st.visited, step) });
    }
    case 'mode': {
      // the Simple|Advanced switch: the step is re-derived from the mode being entered — a
      // standard step the fold holds lands on the folded page, and the other way on the
      // quant cards; the tiles and the running install are pages of both modes. `a.simple` is
      // the mode being entered; the reducer's `simple` is the one being left.
      const step = normalizeStep(st.step, a.simple);
      return done({ step, visited: visit(st.visited, step) });
    }
    case 'next': {
      const list = activeSteps(simple);
      const i = list.indexOf(normalizeStep(st.step, simple));
      // Next never starts the install: that is Finish, the user's own button
      if (!st.selection || i < 0 || i >= list.indexOf('review')) return st;
      const step = list[i + 1];
      return done({ step, visited: visit(st.visited, step) });
    }
    case 'previous': {
      const list = activeSteps(simple);
      const i = list.indexOf(normalizeStep(st.step, simple));
      if (i <= 0) return st;
      const step = list[i - 1];
      return done({ step, visited: visit(st.visited, step) });
    }
    case 'tiles':
      return done({ step: 'models' });
    case 'clear':
      return emptyState();
    case 'dismiss':
      return { ...st, notice: null };
    case 'chat':
      return { ...st, chat: capChat(a.chat), updatedAt: now };
    case 'job':
      return a.id
        ? done({ job: a.id, job_model: st.model, step: 'install', visited: visit(st.visited, 'install') })
        : done({ job: null, job_model: null, step: st.step === 'install' ? 'review' : st.step });
  }
  if (!m || !st.selection) return st;
  const sel = st.selection;
  switch (a.type) {
    case 'profile': {
      if (!m.profiles.some((p) => p.id === a.id)) return st;
      // a profile with a shorter largest window takes the window down with it, and the side window
      const max = windowMax(m, a.id);
      const notices: string[] = [];
      let next: SetupSelection = { ...sel, profile: a.id };
      if (sel.window > max) {
        next = { ...next, window: max, side_window: Math.min(sel.side_window ?? sel.window, max) };
        notices.push(S.windowClamped(max));
      }
      // a community pack the engine cannot run images from: the Vision tool goes off with it
      if (packVisionOff(m, next)) {
        const r = toggleComponent(m, next.components, 'vision', false);
        next = { ...next, components: r.ok ? r.components : { ...next.components, vision: false } };
        notices.push(S.community.visionOff);
      }
      return done({ selection: next, notice: notices.length ? notices.join(' ') : null });
    }
    case 'component': {
      const r = toggleComponent(m, sel.components, a.id, a.on);
      if (!r.ok) return { ...st, notice: r.error };
      return done({ selection: { ...sel, components: r.components }, notice: r.notice });
    }
    case 'components':
      return done({ selection: { ...sel, components: normaliseComponents(m, { ...sel.components, ...a.set }) } });
    case 'parallel': {
      const p = m.parallelism;
      if (a.key === 'kv_pool') return typeof a.value === 'boolean' ? done({ selection: { ...sel, kv_pool: a.value } }) : st;
      if (typeof a.value !== 'number' || !Number.isFinite(a.value)) return st;
      if (a.key === 'window' || a.key === 'side_window') {
        const { value, clamped } = clampWindow(m, a.key, a.value, a.key === 'window' ? windowMax(m, sel.profile) : undefined);
        const next = { ...sel, [a.key]: value };
        // a side window longer than slot 0's makes no sense: it follows the main window down
        if (a.key === 'window' && (next.side_window ?? value) > value) next.side_window = value;
        return done({ selection: next, notice: clamped ? S.windowClamped(value) : null });
      }
      const r = p[a.key];
      // slots: the card's max is how far it was measured, the engine's limit is the cap (VIS-199);
      // the stepper refuses a typed value outside it before it gets here
      const max = a.key === 'slots' ? slotsMax(m) : r.max;
      const v = Math.min(max, Math.max(r.min, a.key === 'slots' ? Math.round(a.value) : a.value));
      return done({ selection: { ...sel, [a.key]: v } });
    }
    case 'field': {
      const f = registryField(m, a.id);
      if (!f || f.group !== a.group) return st;
      return done({ selection: { ...sel, [a.group]: { ...sel[a.group], [a.id]: a.value } } });
    }
    case 'recommended': {
      const d = defaults(m);
      if (a.scope === 'components') return done({ selection: { ...sel, components: d.components } });
      if (a.scope === 'parallelism') return done({ selection: { ...sel, slots: d.slots, window: d.window, side_window: d.side_window, kv_pool: d.kv_pool, floor_gb: d.floor_gb } });
      return done({ selection: { ...sel, methods: d.methods, advanced: d.advanced } });
    }
  }
  return st;
}

// ---- persistence ------------------------------------------------------------------------------

export interface Saved {
  v: 1;
  model: string | null;
  step: SetupStepId;
  selection: SetupSelection | null;
  visited: SetupStepId[];
  updatedAt: number;
  chat: SetupChatMessage[];
  job?: string | null;
  job_model?: string | null;
}

/** localStorage, or null where the browser refuses it (private mode, blocked storage). */
export function safeStorage(): Storage | null {
  try {
    const s = globalThis.localStorage;
    if (!s) return null;
    s.getItem(STORAGE_KEY);
    return s;
  } catch {
    return null;
  }
}

export type LoadProblem = 'unavailable' | 'corrupt' | 'version' | null;

/** The saved progress, or why there is none to use. */
export function readSaved(store: Storage | null): { state: WizardState; problem: LoadProblem } {
  if (!store) return { state: emptyState(), problem: 'unavailable' };
  let raw: string | null;
  try {
    raw = store.getItem(STORAGE_KEY);
  } catch {
    return { state: emptyState(), problem: 'unavailable' };
  }
  if (raw == null) return { state: emptyState(), problem: null };
  let data: Partial<Saved>;
  try {
    data = JSON.parse(raw);
  } catch {
    return { state: emptyState(), problem: 'corrupt' };
  }
  if (!data || typeof data !== 'object') return { state: emptyState(), problem: 'corrupt' };
  if (data.v !== STORAGE_VERSION) return { state: emptyState(), problem: 'version' };
  const st = emptyState();
  st.model = typeof data.model === 'string' ? data.model : null;
  st.step = isStepId(data.step) ? data.step : 'models';
  st.selection = data.selection && typeof data.selection === 'object' ? (data.selection as SetupSelection) : null;
  st.visited = Array.isArray(data.visited) ? data.visited.filter(isStepId) : ['models'];
  st.chat = Array.isArray(data.chat) ? capChat(data.chat.filter(isChatMessage)) : [];
  st.updatedAt = typeof data.updatedAt === 'number' ? data.updatedAt : 0;
  st.job = typeof data.job === 'string' ? data.job : null;
  st.job_model = typeof data.job_model === 'string' ? data.job_model : st.job ? st.model : null;
  if (st.step === 'install' && (!st.job || st.job_model !== st.model)) st.step = 'review';
  if (!st.model || !st.selection) {
    st.model = null;
    st.selection = null;
    st.step = 'models';
  }
  return { state: st, problem: null };
}

const SECRET = /password|secret|token|api_?key/i;

/** The saved form of a state: no notice, and nothing that looks like a typed secret. */
export function toSaved(st: WizardState): Saved {
  let selection = st.selection;
  if (selection) {
    const clean = (o: Record<string, SetupValue>) => Object.fromEntries(Object.entries(o).filter(([k, v]) => !SECRET.test(k) || v === 'keep' || v === 'generate'));
    selection = { ...selection, methods: clean(selection.methods), advanced: clean(selection.advanced) };
  }
  return { v: 1, model: st.model, step: st.step === 'install' && (!st.job || st.job_model !== st.model) ? 'review' : st.step, selection, visited: st.visited, updatedAt: st.updatedAt, chat: capChat(st.chat), job: st.job, job_model: st.job_model };
}

export function writeSaved(store: Storage | null, st: WizardState): boolean {
  if (!store) return false;
  try {
    store.setItem(STORAGE_KEY, JSON.stringify(toSaved(st)));
    return true;
  } catch {
    return false;
  }
}

export function clearSaved(store: Storage | null): boolean {
  if (!store) return false;
  try {
    store.removeItem(STORAGE_KEY);
    return true;
  } catch {
    return false;
  }
}

function isChatMessage(x: unknown): x is SetupChatMessage {
  const m = x as SetupChatMessage;
  return !!m && typeof m === 'object' && (m.role === 'user' || m.role === 'assistant' || m.role === 'tool') && typeof m.content === 'string';
}

/** At most 40 messages and 64 KB, oldest dropped first; never starts on a tool answer. */
export function capChat(msgs: SetupChatMessage[]): SetupChatMessage[] {
  let out = msgs.slice(-CHAT_MAX_MESSAGES);
  const size = (xs: SetupChatMessage[]) => new TextEncoder().encode(JSON.stringify(xs)).length;
  while (out.length && size(out) > CHAT_MAX_BYTES) out = out.slice(1);
  while (out.length && out[0].role === 'tool') out = out.slice(1);
  return out;
}
