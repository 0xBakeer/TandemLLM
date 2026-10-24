// Wren's tools, executed by the browser only (D6, VIS-80). A tool call from the stream is checked
// against the step schema (fields.ts) and the component rules (deps.ts) and turned into the same
// reducer action a click makes, or rejected with an error that goes back to the model as the tool
// message. There is no finish, apply, reset, load or unload tool: those names are rejected as
// unknown, and no handler for them exists here.

import type { SetupModel, SetupSelection, SetupStepId, SetupToolCall, SetupValue } from '../api/types';
import { availability } from './deps';
import { fieldError, isStepId, registryField, SECURITY_FIELDS, STEP_IDS, stepIndex, stepsOf } from './fields';
import type { SetupAction } from './state';
import { S, STEP_TITLE } from './strings';

export const TOOL_NAMES = ['set_field', 'toggle', 'select_profile', 'go_to_step', 'next', 'previous', 'explain', 'read_device'] as const;
export type ToolName = (typeof TOOL_NAMES)[number];

/** What the field a tool touched is called on the page (`data-field`), for the highlight. */
export interface ToolTarget {
  step: SetupStepId;
  field: string | null;
}

export interface ToolContext {
  model: SetupModel | null;
  selection: SetupSelection | null;
  step: SetupStepId;
  /** field key (`step.field`) -> when the user last changed it; a tool call older than that is stale */
  touchedAt?: Map<string, number>;
  /** when the user sent the message this turn answers */
  turnStartedAt?: number;
}

export type ToolPlan =
  | { ok: true; kind: 'action'; actions: SetupAction[]; undo: SetupAction[]; target: ToolTarget; summary: string; result: string }
  | { ok: true; kind: 'explain'; target: ToolTarget; result: string }
  | { ok: true; kind: 'read_device' }
  | { ok: false; error: string; stale?: boolean };

const fail = (error: string, stale = false): ToolPlan => ({ ok: false, error, ...(stale ? { stale } : {}) });

export function fieldKey(step: SetupStepId, field: string): string {
  return `${step}.${field}`;
}

/** Arguments as the model sent them: an object, or a JSON string of one. */
export function parseArgs(args: unknown): Record<string, unknown> | null {
  if (args == null) return {};
  if (typeof args === 'string') {
    if (!args.trim()) return {};
    try {
      const v = JSON.parse(args);
      return v && typeof v === 'object' && !Array.isArray(v) ? v : null;
    } catch {
      return null;
    }
  }
  return typeof args === 'object' && !Array.isArray(args) ? (args as Record<string, unknown>) : null;
}

function stepRef(x: unknown): SetupStepId | null {
  if (typeof x === 'number' && Number.isInteger(x) && x >= 0 && x < STEP_IDS.length) return STEP_IDS[x];
  if (typeof x === 'string') {
    if (/^\d+$/.test(x)) return stepRef(Number(x));
    if (isStepId(x)) return x;
  }
  return null;
}

/** Numbers and booleans a small model sends as strings ("2", "true") are taken as what they say. */
function coerce(type: string, v: unknown): unknown {
  if ((type === 'int' || type === 'number') && typeof v === 'string' && v.trim() !== '' && Number.isFinite(Number(v))) return Number(v);
  if (type === 'bool' && (v === 'true' || v === 'false')) return v === 'true';
  return v;
}

function display(v: SetupValue): string {
  if (typeof v === 'boolean') return v ? 'on' : 'off';
  if (typeof v === 'number') return v.toLocaleString('en-US');
  return v;
}

/** Check one call and say what it would do. Nothing changes until the caller dispatches. */
/** A call as the model sent it: any name (unknown ones are rejected), arguments as an object or JSON text. */
export interface RawToolCall {
  name: string;
  args: unknown;
}

export function planToolCall(call: RawToolCall | Pick<SetupToolCall, 'name' | 'args'>, ctx: ToolContext): ToolPlan {
  const name = String(call.name ?? '');
  if (!(TOOL_NAMES as readonly string[]).includes(name)) {
    if (/finish|apply|install|reset|delete|unload|load|confirm|submit/i.test(name)) return fail(S.tool.unknown(name) + ' ' + S.tool.userOnly(name));
    return fail(S.tool.unknown(name));
  }
  const args = parseArgs(call.args);
  if (!args) return fail(S.tool.badArgs('expected a JSON object'));
  if (name === 'read_device') return { ok: true, kind: 'read_device' };

  const m = ctx.model;
  const sel = ctx.selection;
  const stale = (step: SetupStepId, field: string): boolean => {
    const t = ctx.touchedAt?.get(fieldKey(step, field));
    return t != null && ctx.turnStartedAt != null && t > ctx.turnStartedAt;
  };

  switch (name as ToolName) {
    case 'go_to_step': {
      const raw = args.n ?? args.step ?? args.index;
      const step = stepRef(raw);
      if (!step) return fail(S.tool.noStep(String(raw)));
      if (step === 'install') return fail(S.tool.installStep);
      if (step !== 'models' && !m) return fail(S.tool.noModel);
      return { ok: true, kind: 'action', actions: [{ type: step === 'models' ? 'tiles' : 'go', step } as SetupAction], undo: [{ type: 'go', step: ctx.step }], target: { step, field: null }, summary: S.toolDone.step(STEP_TITLE[step]), result: S.tool.ok };
    }
    case 'next': {
      if (!m) return fail(S.tool.noModel);
      const i = stepIndex(ctx.step);
      if (i >= stepIndex('review')) return fail(S.tool.atEnd);
      const step = STEP_IDS[i + 1];
      return { ok: true, kind: 'action', actions: [{ type: 'next' }], undo: [{ type: 'go', step: ctx.step }], target: { step, field: null }, summary: S.toolDone.step(STEP_TITLE[step]), result: S.tool.ok };
    }
    case 'previous': {
      const i = stepIndex(ctx.step);
      if (i <= 0) return fail(S.tool.atStart);
      const step = STEP_IDS[i - 1];
      return { ok: true, kind: 'action', actions: [{ type: 'previous' }], undo: [{ type: 'go', step: ctx.step }], target: { step, field: null }, summary: S.toolDone.step(STEP_TITLE[step]), result: S.tool.ok };
    }
  }

  if (!m || !sel) return fail(S.tool.noModel);

  if (name === 'explain') {
    const id = String(args.field ?? args.id ?? '');
    const c = m.components.find((x) => x.id === id);
    if (c) return { ok: true, kind: 'explain', target: { step: 'components', field: c.id }, result: `${c.label}: ${c.what}${c.note ? ' ' + c.note : ''}` };
    const f = registryField(m, id);
    if (f) return { ok: true, kind: 'explain', target: { step: 'methods', field: f.field.id }, result: `${f.field.label}: ${f.field.what}` };
    const p = m.profiles.find((x) => x.id === id);
    if (p) return { ok: true, kind: 'explain', target: { step: 'weights', field: 'profile' }, result: `${p.label}: KL ${p.quality.kl_vs_bf16 ?? '?'}, ${p.weights_gib} GiB in memory${p.note ? '. ' + p.note : ''}` };
    const par: Record<string, string> = { slots: S.slotsWhat, window: S.windowWhat, kv_pool: S.kvPoolWhat, floor_gb: S.floorWhat };
    if (par[id]) return { ok: true, kind: 'explain', target: { step: 'parallelism', field: id }, result: par[id] };
    return fail(S.tool.noField('any', id));
  }

  if (name === 'select_profile') {
    const id = String(args.id ?? args.profile ?? '');
    const p = m.profiles.find((x) => x.id === id);
    if (!p) return fail(S.tool.noProfile(id));
    if (stale('weights', 'profile')) return fail(S.tool.stale, true);
    if (sel.profile === id) return { ok: true, kind: 'action', actions: [], undo: [], target: { step: 'weights', field: 'profile' }, summary: S.toolDone.profile(p.label), result: S.tool.same };
    return { ok: true, kind: 'action', actions: [{ type: 'profile', id }], undo: [{ type: 'profile', id: sel.profile }], target: { step: 'weights', field: 'profile' }, summary: S.toolDone.profile(p.label), result: S.tool.ok };
  }

  if (name === 'toggle') {
    const id = String(args.component ?? args.id ?? '');
    const on = coerce('bool', args.on);
    if (typeof on !== 'boolean') return fail(S.tool.badArgs('on must be true or false'));
    return planToggle(m, sel, id, on, stale('components', id));
  }

  // set_field(step, field, value)
  const step = stepRef(args.step);
  if (!step) return fail(S.tool.noStep(String(args.step)));
  const fieldName = String(args.field ?? '');
  const schema = stepsOf(m).find((s) => s.id === step);
  const f = schema?.fields.find((x) => x.field === fieldName);
  if (!f) return fail(S.tool.noField(step, fieldName));
  if (SECURITY_FIELDS.has(fieldName)) return fail(S.tool.secret(f.label));
  const value = coerce(f.type, args.value);
  const why = fieldError(f, value);
  if (why) return fail(S.tool.badArgs(`${fieldName}: ${why}`));
  if (step === 'components') return planToggle(m, sel, fieldName, value as boolean, stale(step, fieldName));
  if (stale(step, fieldName)) return fail(S.tool.stale, true);
  const target = { step, field: fieldName };
  const summary = S.toolDone.field(f.label, display(value as SetupValue));
  if (step === 'weights') {
    return planToolCall({ name: 'select_profile', args: { id: value } }, ctx);
  }
  if (step === 'parallelism') {
    const key = fieldName as 'slots' | 'window' | 'kv_pool' | 'floor_gb';
    const prev = sel[key] as number | boolean;
    if (prev === value) return { ok: true, kind: 'action', actions: [], undo: [], target, summary, result: S.tool.same };
    return { ok: true, kind: 'action', actions: [{ type: 'parallel', key, value: value as number | boolean }], undo: [{ type: 'parallel', key, value: prev }], target, summary, result: S.tool.ok };
  }
  if (step === 'methods') {
    const rf = registryField(m, fieldName);
    if (!rf) return fail(S.tool.noField(step, fieldName));
    const prev = sel[rf.group][fieldName] ?? rf.field.recommended;
    if (prev === value) return { ok: true, kind: 'action', actions: [], undo: [], target, summary, result: S.tool.same };
    return { ok: true, kind: 'action', actions: [{ type: 'field', group: rf.group, id: fieldName, value: value as SetupValue }], undo: [{ type: 'field', group: rf.group, id: fieldName, value: prev }], target, summary, result: S.tool.ok };
  }
  return fail(S.tool.noField(step, fieldName));
}

function planToggle(m: SetupModel, sel: SetupSelection, id: string, on: boolean, isStale: boolean): ToolPlan {
  const c = m.components.find((x) => x.id === id);
  if (!c) return fail(S.tool.noComponent(id));
  if (c.required && !on) return fail(S.tool.required(c.label));
  const av = availability(m, sel.components)[id];
  if (on && av.disabled && !av.required) return fail(av.reason ?? S.tool.noComponent(id));
  if (isStale) return fail(S.tool.stale, true);
  const target = { step: 'components' as SetupStepId, field: id };
  const summary = on ? S.toolDone.toggleOn(c.label) : S.toolDone.toggleOff(c.label);
  const cur = c.required || (sel.components[id] ?? c.default);
  if (cur === on) return { ok: true, kind: 'action', actions: [], undo: [], target, summary, result: S.tool.same };
  // Undo puts back every component the toggle touched (a cascade switches dependents off too)
  return { ok: true, kind: 'action', actions: [{ type: 'component', id, on }], undo: [{ type: 'components', set: { ...sel.components } }], target, summary, result: S.tool.ok };
}

/** The OpenAI tool definitions, for a server that does not send its own (and for the mock). */
export function toolDefinitions(): { type: 'function'; function: { name: ToolName; description: string; parameters: Record<string, unknown> } }[] {
  const obj = (props: Record<string, unknown>, required: string[] = Object.keys(props)) => ({ type: 'object', properties: props, required, additionalProperties: false });
  return [
    { type: 'function', function: { name: 'set_field', description: 'Set one field of a wizard step.', parameters: obj({ step: { type: ['string', 'integer'] }, field: { type: 'string' }, value: { type: ['string', 'number', 'boolean'] } }) } },
    { type: 'function', function: { name: 'toggle', description: 'Turn a tool that runs beside the model on or off.', parameters: obj({ component: { type: 'string' }, on: { type: 'boolean' } }) } },
    { type: 'function', function: { name: 'select_profile', description: 'Choose the weight profile.', parameters: obj({ id: { type: 'string' } }) } },
    { type: 'function', function: { name: 'go_to_step', description: 'Open a step (0 to 6).', parameters: obj({ n: { type: 'integer', minimum: 0, maximum: 6 } }) } },
    { type: 'function', function: { name: 'next', description: 'Go to the next step.', parameters: obj({}) } },
    { type: 'function', function: { name: 'previous', description: 'Go to the previous step.', parameters: obj({}) } },
    { type: 'function', function: { name: 'explain', description: 'Read what a field, a tool or a weight profile does.', parameters: obj({ field: { type: 'string' } }) } },
    { type: 'function', function: { name: 'read_device', description: "Read the box's memory, GPU, disk and engine state.", parameters: obj({}) } },
  ];
}
