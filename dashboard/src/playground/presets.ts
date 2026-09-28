// Presets and the draft: named setups in localStorage, wrapped so a blocked or full
// store degrades to "nothing remembered" and never to an error. Import/export as JSON files.

import type { Message, ParamValues, Preset, Setup, ToolChoice } from './types';
import { emptySetup } from './types';
import { PARAMS } from './params';

export const PRESETS_KEY = 'qse.playground.presets';
export const DRAFT_KEY = 'qse.playground.draft';

export interface Store {
  getItem(k: string): string | null;
  setItem(k: string, v: string): void;
  removeItem(k: string): void;
}

function store(): Store | null {
  try {
    return typeof localStorage !== 'undefined' ? localStorage : null;
  } catch {
    return null;
  }
}

function read<T>(key: string, s: Store | null = store()): T | null {
  try {
    const raw = s?.getItem(key);
    return raw ? (JSON.parse(raw) as T) : null;
  } catch {
    return null;
  }
}

/** true when written; false when the store is unavailable or full */
function write(key: string, value: unknown, s: Store | null = store()): boolean {
  try {
    if (!s) return false;
    s.setItem(key, JSON.stringify(value));
    return true;
  } catch {
    return false;
  }
}

export function loadPresets(s?: Store | null): Preset[] {
  const list = read<unknown>(PRESETS_KEY, s === undefined ? store() : s);
  const { presets } = parsePresets(list);
  return presets;
}

export function savePresets(list: Preset[], s?: Store | null): boolean {
  return write(PRESETS_KEY, list, s === undefined ? store() : s);
}

/** Save or overwrite by name; returns the new list and whether it was persisted. */
export function upsertPreset(list: Preset[], p: Preset, s?: Store | null): { list: Preset[]; persisted: boolean } {
  const next = list.filter((x) => x.name !== p.name);
  next.push(p);
  next.sort((a, b) => a.name.localeCompare(b.name));
  return { list: next, persisted: savePresets(next, s) };
}

export function deletePreset(list: Preset[], name: string, s?: Store | null): { list: Preset[]; persisted: boolean } {
  const next = list.filter((x) => x.name !== name);
  return { list: next, persisted: savePresets(next, s) };
}

export function renamePreset(list: Preset[], from: string, to: string, s?: Store | null): { list: Preset[]; persisted: boolean } {
  const next = list.filter((x) => x.name !== to).map((x) => (x.name === from ? { ...x, name: to } : x));
  next.sort((a, b) => a.name.localeCompare(b.name));
  return { list: next, persisted: savePresets(next, s) };
}

export function presetFrom(name: string, setup: Setup, now = new Date()): Preset {
  return { version: 1, name, savedAt: now.toISOString(), system: setup.system, params: { ...setup.params }, toolsText: setup.toolsText, tool_choice: setup.tool_choice };
}

export function setupFrom(p: Preset): Setup {
  return { system: p.system, params: { ...p.params }, toolsText: p.toolsText, tool_choice: p.tool_choice };
}

/** Does the setup differ from the preset it was loaded from? */
export function setupEquals(a: Setup, b: Setup): boolean {
  return a.system === b.system && a.toolsText === b.toolsText && JSON.stringify(a.tool_choice) === JSON.stringify(b.tool_choice) && JSON.stringify(cleanParams(a.params)) === JSON.stringify(cleanParams(b.params));
}

function cleanParams(p: ParamValues): ParamValues {
  const out: ParamValues = {};
  for (const d of PARAMS) {
    const v = p[d.key];
    if (v !== undefined) (out as Record<string, unknown>)[d.key] = v;
  }
  return out;
}

const PARAM_KEYS = new Set(PARAMS.map((p) => p.key as string));

/** Validate a parsed presets document; unknown fields are dropped, bad entries named. */
export function parsePresets(doc: unknown): { presets: Preset[]; errors: string[] } {
  const errors: string[] = [];
  const presets: Preset[] = [];
  const list = Array.isArray(doc) ? doc : doc && typeof doc === 'object' && Array.isArray((doc as { presets?: unknown }).presets) ? (doc as { presets: unknown[] }).presets : null;
  if (!list) return { presets, errors: ['expected a JSON array of presets (or {"presets": [...]})'] };
  list.forEach((item, i) => {
    if (!item || typeof item !== 'object') return errors.push(`preset ${i}: not an object`);
    const o = item as Record<string, unknown>;
    if (typeof o.name !== 'string' || !o.name.trim()) return errors.push(`preset ${i}: needs a name`);
    const params: ParamValues = {};
    if (o.params !== undefined) {
      if (!o.params || typeof o.params !== 'object' || Array.isArray(o.params)) return errors.push(`preset ${i} (${o.name}): params must be an object`);
      for (const [k, v] of Object.entries(o.params as Record<string, unknown>)) {
        if (PARAM_KEYS.has(k) && v !== undefined && v !== null) (params as Record<string, unknown>)[k] = v;
      }
    }
    const tc = o.tool_choice;
    const toolChoice: ToolChoice = tc === 'none' || tc === 'required' ? tc : tc && typeof tc === 'object' && (tc as { function?: { name?: unknown } }).function?.name ? { type: 'function', function: { name: String((tc as { function: { name: unknown } }).function.name) } } : 'auto';
    presets.push({
      version: 1,
      name: o.name.trim(),
      savedAt: typeof o.savedAt === 'string' ? o.savedAt : new Date(0).toISOString(),
      system: typeof o.system === 'string' ? o.system : '',
      params,
      toolsText: typeof o.toolsText === 'string' ? o.toolsText : Array.isArray(o.tools) ? JSON.stringify(o.tools, null, 2) : '',
      tool_choice: toolChoice,
    });
  });
  return { presets, errors };
}

export function exportPresets(list: Preset[]): string {
  return JSON.stringify({ format: 'qse-playground-presets', version: 1, exportedAt: new Date().toISOString(), presets: list }, null, 2);
}

/** Merge an imported document into the list by name (imported wins). */
export function importPresets(list: Preset[], text: string, s?: Store | null): { list: Preset[]; added: number; errors: string[]; persisted: boolean } {
  let doc: unknown;
  try {
    doc = JSON.parse(text);
  } catch (e) {
    return { list, added: 0, errors: [`not valid JSON: ${(e as Error).message}`], persisted: false };
  }
  const { presets, errors } = parsePresets(doc);
  if (!presets.length) return { list, added: 0, errors: errors.length ? errors : ['the file holds no presets'], persisted: false };
  const byName = new Map(list.map((p) => [p.name, p]));
  for (const p of presets) byName.set(p.name, p);
  const next = [...byName.values()].sort((a, b) => a.name.localeCompare(b.name));
  return { list: next, added: presets.length, errors, persisted: savePresets(next, s) };
}

// ---- the draft: the current setup + transcript, so a reload keeps the bench ------------------
export interface Draft {
  version: 1;
  setup: Setup;
  messages: Message[];
  presetName: string | null;
}

export function loadDraft(s?: Store | null): Draft | null {
  const d = read<Draft>(DRAFT_KEY, s === undefined ? store() : s);
  if (!d || d.version !== 1 || !d.setup || !Array.isArray(d.messages)) return null;
  return { version: 1, setup: { ...emptySetup(), ...d.setup, params: { ...(d.setup.params ?? {}) } }, messages: d.messages, presetName: d.presetName ?? null };
}

export function saveDraft(d: Draft, s?: Store | null): boolean {
  return write(DRAFT_KEY, d, s === undefined ? store() : s);
}

export function clearDraft(s?: Store | null): void {
  try {
    (s === undefined ? store() : s)?.removeItem(DRAFT_KEY);
  } catch {
    /* ignore */
  }
}
