// What is wrong with a step (the rail's valid/invalid marks, Next stays usable, Finish does not),
// the How-to-fix list for a setup that does not fit, the step that fixes a plan error, what a failed
// sign-in says, and the compact page state Wren gets each turn.

import type { SetupDevice, SetupEstimate, SetupModel, SetupPageState, SetupSelection, SetupStepId } from '../api/types';
import { isOn, lowerLabel } from './deps';
import { estimateLoose, overBy, profileOf } from './estimate';
import { fieldError, stepIndex, stepsOf } from './fields';
import { S, STEP_TITLE } from './strings';

export type StepIssues = Partial<Record<SetupStepId, string[]>>;

/** The step that fixes a plan error, by its code (SetupPlan.error_codes, one per error, in order);
 *  null: the fix is on the box itself (ops/serve.env's overlay lines, its REPO). */
const PLAN_FIX: Record<string, SetupStepId> = {
  over_budget: 'parallelism',
  missing_token: 'weights',
  disk_full: 'weights',
  missing_assets: 'components',
  secrets_exist: 'methods',
  port_in_use: 'methods',
};
export function planFixStep(code: string | undefined): SetupStepId | null {
  return (code && PLAN_FIX[code]) || null;
}

/** What a failed sign-in says, by the status of POST /v1/dashboard/session (404: the server's
 *  dashboard login is off, so there is no session to make). */
export function signInError(status: number, message: string): string {
  if (status === 401) return S.signIn.refused;
  if (status === 429) return S.signIn.tooMany;
  if (status === 404) return S.signIn.loginOff;
  if (status === 0) return S.signIn.offline;
  return `${S.signIn.failed} ${message}`;
}

export function stepIssues(m: SetupModel, sel: SetupSelection, device: SetupDevice | null, est: SetupEstimate | null): StepIssues {
  const out: StepIssues = {};
  const add = (s: SetupStepId, msg: string) => (out[s] ??= []).push(msg);
  const p = m.profiles.find((x) => x.id === sel.profile);
  if (!p) add('weights', S.tool.noProfile(sel.profile));
  else if (tokenBlocked(m, sel, device)) add('weights', S.privateNoToken);
  for (const f of [...m.methods, ...m.advanced]) {
    const group = m.methods.includes(f) ? sel.methods : sel.advanced;
    const why = fieldError(f, group[f.id] ?? f.recommended);
    if (why) add('methods', `${f.label}: ${why}`);
  }
  const par = m.parallelism;
  if (sel.slots < par.slots.min || sel.slots > par.slots.max) add('parallelism', `${S.slots}: ${sel.slots}`);
  if (est && !est.fits) add('review', S.reviewOver(overBy(est)));
  return out;
}

/** The weights must be downloaded from a private repo and the box has no token. */
export function tokenBlocked(m: SetupModel, sel: SetupSelection, device: SetupDevice | null): boolean {
  const p = m.profiles.find((x) => x.id === sel.profile);
  if (!p || !p.hf_private || !device) return false;
  if (device.profiles_on_disk?.[p.id]) return false;
  return device.hf_token_present === false;
}

export interface Fix {
  text: string;
  step: SetupStepId;
  /** the memory it brings back, GiB */
  saves: number;
}

const WINDOWS = [262_144, 196_608, 131_072, 65_536, 32_768, 16_384, 8_192];

/** Ways to make an over-budget selection fit, the ones that fit on their own first. */
export function howToFix(m: SetupModel, sel: SetupSelection): Fix[] {
  const base = estimateLoose(m, sel);
  if (!base || base.fits) return [];
  const fixes: Fix[] = [];
  const NO = { fits: false, total_gb: base.total_gb } as SetupEstimate;
  const try_ = (s: SetupSelection) => estimateLoose(m, s) ?? NO;
  if (!sel.kv_pool) {
    const e = try_({ ...sel, kv_pool: true });
    fixes.push({ text: S.fixPool, step: 'parallelism', saves: base.total_gb - e.total_gb });
  }
  for (const w of WINDOWS.filter((x) => x < sel.window && x >= m.parallelism.window.min)) {
    const e = try_({ ...sel, window: w, side_window: Math.min(sel.side_window ?? sel.window, w) });
    if (e.fits) {
      fixes.push({ text: S.fixWindow(w), step: 'parallelism', saves: base.total_gb - e.total_gb });
      break;
    }
  }
  for (let n = sel.slots - 1; n >= m.parallelism.slots.min; n--) {
    const e = try_({ ...sel, slots: n });
    if (e.fits || n === m.parallelism.slots.min) {
      fixes.push({ text: S.fixSlots(n), step: 'parallelism', saves: base.total_gb - e.total_gb });
      break;
    }
  }
  const cur = profileOf(m, sel.profile);
  const smaller = m.profiles.filter((p) => p.weights_gib < cur.weights_gib).sort((a, b) => b.weights_gib - a.weights_gib);
  for (const p of smaller) {
    const e = try_({ ...sel, profile: p.id });
    if (e.fits || p === smaller[smaller.length - 1]) {
      fixes.push({ text: S.fixProfile(p.label), step: 'weights', saves: base.total_gb - e.total_gb });
      break;
    }
  }
  const big = m.components.filter((c) => !c.required && isOn(c, sel.components) && c.memory_gib >= 0.5).sort((a, b) => b.memory_gib - a.memory_gib).slice(0, 2);
  for (const c of big) fixes.push({ text: S.fixComponent(lowerLabel(c.label), c.memory_gib), step: 'components', saves: c.memory_gib });
  const over = overBy(base);
  // the ones that fix it alone first, then by what they bring back
  return fixes.sort((a, b) => Number(b.saves >= over) - Number(a.saves >= over) || b.saves - a.saves);
}

/** "Over by 6.2 GiB. Try 2 slots or the 4.0 weights." */
export function overSentence(m: SetupModel, sel: SetupSelection, est: SetupEstimate): string {
  const fixes = howToFix(m, sel);
  const hint = fixes.length ? S.overHint(fixes[0].text, fixes[1]?.text ?? null) : null;
  return S.overSentence(overBy(est), hint);
}

const PAGE_STATE_MAX = 6 * 1024;
const SECRET = /password|secret|token|api_?key/i;

/** What Wren is told about the page each turn: compact, and never a secret. */
export function buildPageState(m: SetupModel | null, step: SetupStepId, sel: SetupSelection | null, est: SetupEstimate | null, extra: { errors?: string[] } = {}): SetupPageState {
  const page: SetupPageState = { model: m?.id ?? null, step, step_index: stepIndex(step), selection: {} };
  if (!m || !sel) {
    page.visible = { title: STEP_TITLE[step] };
    return page;
  }
  // nothing the page holds is a secret (the admin password is keep or generate); a value under a
  // secret-looking name that is anything else is still never sent
  const redact = (o: Record<string, unknown>) => Object.fromEntries(Object.entries(o).map(([k, v]) => [k, SECRET.test(k) && v !== 'keep' && v !== 'generate' ? '<set>' : v])) as SetupSelection['advanced'];
  page.selection = { model: sel.model, profile: sel.profile, components: sel.components, slots: sel.slots, window: sel.window, side_window: sel.side_window, kv_pool: sel.kv_pool, floor_gb: sel.floor_gb, methods: redact(sel.methods), advanced: redact(sel.advanced) };
  const schema = stepsOf(m).find((s) => s.id === step);
  const valueOf = (field: string): unknown => {
    if (step === 'weights') return sel.profile;
    if (step === 'components') return sel.components[field];
    if (step === 'parallelism') return (sel as unknown as Record<string, unknown>)[field];
    if (step === 'methods') {
      if (field === 'admin_password') return sel.advanced[field] === 'generate' ? 'generate' : 'keep';
      return sel.methods[field] ?? sel.advanced[field];
    }
    return undefined;
  };
  const fields = (schema?.fields ?? []).map((f) => {
    const o: Record<string, unknown> = { field: f.field, label: f.label, type: f.type, value: valueOf(f.field) };
    if (f.options) o.options = f.options;
    if (f.min != null || f.max != null) o.range = [f.min ?? null, f.max ?? null];
    return o;
  });
  page.visible = { title: STEP_TITLE[step], fields, ...(extra.errors?.length ? { errors: extra.errors } : {}) };
  if (est) {
    const top = [...est.parts].sort((a, b) => b.gb - a.gb).slice(0, 3).map((p) => ({ id: p.id, gb: p.gb }));
    page.estimate = {
      total_gb: est.total_gb,
      budget_gb: est.budget_gb,
      device_gb: est.device_gb,
      available_gb: est.available_gb,
      floor_gb: est.floor_gb,
      fits: est.fits,
      fits_worst: est.fits_worst,
      kv_worst_gb: est.kv_worst_gb,
      context_tokens_fit: est.context_tokens_fit,
      parts: top as never,
    };
  }
  // keep it under the cap: the field list goes first (options are the long part)
  let s = JSON.stringify(page);
  if (s.length > PAGE_STATE_MAX && page.visible) {
    page.visible = { title: STEP_TITLE[step], fields: fields.map(({ options: _o, ...rest }) => rest) };
    s = JSON.stringify(page);
    if (s.length > PAGE_STATE_MAX) page.visible = { title: STEP_TITLE[step] };
  }
  return page;
}
