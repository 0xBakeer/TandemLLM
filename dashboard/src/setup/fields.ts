// The step schema: which fields each wizard step has, their types and ranges. The forms render it
// and Wren's tool calls are checked against it (tools.ts). The server may send it as the model's
// `wizard_fields`; without that the page derives the same schema from the registry entry.
// `fieldError` is server/setup/registry.py's field_error, line for line.

import type { SetupField, SetupModel, SetupStep, SetupStepField, SetupStepId } from '../api/types';
import { STEP_TITLE } from './strings';

/** server/setup/registry.py ENGINE_MAX_SLOTS (VIS-199): the most parallel chats, whatever a card
 *  measured. A joint verify pass holds at most 32 rows (server/slots.py) and a decoding chat keeps 2
 *  of them at least, so 16 verify in one pass. The card's slots.max is how far it was measured. */
export const ENGINE_MAX_SLOTS = 16;

export const STEP_IDS: SetupStepId[] = ['models', 'overview', 'weights', 'components', 'parallelism', 'methods', 'review', 'install'];

export function stepIndex(id: SetupStepId): number {
  return STEP_IDS.indexOf(id);
}

export function isStepId(x: unknown): x is SetupStepId {
  return typeof x === 'string' && (STEP_IDS as string[]).includes(x);
}

type FieldLike = Pick<SetupStepField, 'type' | 'min' | 'max' | 'pattern'> & { options?: (string | { value: string })[] };

/** Why `value` is not valid for the field, or null. */
export function fieldError(f: FieldLike, value: unknown): string | null {
  const t = f.type;
  if (t === 'bool') return typeof value === 'boolean' ? null : 'expected true or false';
  if (t === 'enum') {
    const opts = (f.options ?? []).map((o) => (typeof o === 'string' ? o : o.value));
    return typeof value === 'string' && opts.includes(value) ? null : `expected one of ${opts.join(', ')}`;
  }
  if (t === 'int' || t === 'number') {
    if (typeof value !== 'number' || !Number.isFinite(value) || (t === 'int' && !Number.isInteger(value))) return t === 'int' ? 'expected an integer' : 'expected a number';
    if (f.min != null && value < f.min) return `at least ${fmt(f.min)}`;
    if (f.max != null && value > f.max) return `at most ${fmt(f.max)}`;
    return null;
  }
  const re = new RegExp(f.pattern ? f.pattern : '^[A-Za-z0-9._:-]{1,64}$');
  if (typeof value !== 'string' || !fullMatch(re, value)) return `letters, digits and ._:- only (${f.pattern || 'at most 64'})`;
  return null;
}

function fullMatch(re: RegExp, s: string): boolean {
  const m = s.match(re);
  return m != null && m.index === 0 && m[0].length === s.length;
}

function fmt(n: number): string {
  return Number.isInteger(n) ? String(n) : String(+n.toPrecision(6));
}

function fromField(f: SetupField): SetupStepField {
  const out: SetupStepField = { field: f.id, label: f.label, type: f.type };
  if (f.options) out.options = f.options.map((o) => o.value);
  if (f.min != null) out.min = f.min;
  if (f.max != null) out.max = f.max;
  if (f.pattern) out.pattern = f.pattern;
  return out;
}

/** The schema from the registry entry: what `wizard_fields` holds when the server sends it. */
export function deriveSteps(m: SetupModel): SetupStep[] {
  const p = m.parallelism;
  const fields: Record<SetupStepId, SetupStepField[]> = {
    models: [],
    overview: [],
    weights: [{ field: 'profile', label: 'Weights', type: 'enum', options: m.profiles.map((x) => x.id) }],
    components: m.components.map((c) => ({ field: c.id, label: c.label, type: 'bool' as const })),
    parallelism: [
      { field: 'slots', label: 'Parallel chats', type: 'int', min: p.slots.min, max: ENGINE_MAX_SLOTS },
      { field: 'window', label: 'Longest conversation', type: 'int', min: p.window.min, max: p.window.max },
      { field: 'side_window', label: 'Longest conversation in the extra chats', type: 'int', min: p.side_window.min, max: p.side_window.max },
      { field: 'kv_pool', label: 'Shared memory pool', type: 'bool' },
      { field: 'floor_gb', label: 'Memory kept free', type: 'number', min: p.floor_gb.min, max: p.floor_gb.max },
    ],
    methods: [...m.methods, ...m.advanced].map(fromField),
    review: [],
    install: [],
  };
  return STEP_IDS.map((id, index) => ({ index, id, label: STEP_TITLE[id], fields: fields[id] }));
}

export function stepsOf(m: SetupModel): SetupStep[] {
  return m.wizard_fields && m.wizard_fields.length ? m.wizard_fields : deriveSteps(m);
}

/** The model's methods or advanced field by id, and which group it is in. */
export function registryField(m: SetupModel, id: string): { group: 'methods' | 'advanced'; field: SetupField } | null {
  const a = m.methods.find((f) => f.id === id);
  if (a) return { group: 'methods', field: a };
  const b = m.advanced.find((f) => f.id === id);
  return b ? { group: 'advanced', field: b } : null;
}

/** Security settings: only the user changes them (Wren cannot, and the page never stores a typed secret). */
export const SECURITY_FIELDS = new Set(['admin_password', 'dashboard_login', 'host']);
