// presets: round trip, a throwing store, a refused import, the draft.
import { describe, expect, it } from 'vitest';
import { clearDraft, deletePreset, exportPresets, importPresets, loadDraft, loadPresets, parsePresets, presetFrom, renamePreset, saveDraft, setupEquals, setupFrom, upsertPreset, type Store } from '../src/playground/presets';
import { emptySetup } from '../src/playground/types';

function memStore(): Store & { data: Map<string, string> } {
  const data = new Map<string, string>();
  return { data, getItem: (k) => data.get(k) ?? null, setItem: (k, v) => void data.set(k, v), removeItem: (k) => void data.delete(k) };
}
const throwing: Store = {
  getItem: () => {
    throw new Error('blocked');
  },
  setItem: () => {
    throw new Error('QuotaExceededError');
  },
  removeItem: () => {
    throw new Error('blocked');
  },
};

describe('presets', () => {
  it('save / load / rename / delete round trip through the store', () => {
    const s = memStore();
    const setup = { ...emptySetup(), system: 'Be terse.', params: { temperature: 0.7, stop: ['###'] }, toolsText: '[]' };
    const r = upsertPreset([], presetFrom('terse', setup, new Date('2026-09-25T10:00:00Z')), s);
    expect(r.persisted).toBe(true);
    const loaded = loadPresets(s);
    expect(loaded).toHaveLength(1);
    expect(loaded[0]).toMatchObject({ version: 1, name: 'terse', system: 'Be terse.', params: { temperature: 0.7, stop: ['###'] }, savedAt: '2026-09-25T10:00:00.000Z' });
    expect(setupEquals(setupFrom(loaded[0]), setup)).toBe(true);
    // overwrite by name keeps one entry
    const r2 = upsertPreset(loaded, presetFrom('terse', { ...setup, system: 'v2' }), s);
    expect(r2.list).toHaveLength(1);
    expect(loadPresets(s)[0].system).toBe('v2');
    const r3 = renamePreset(r2.list, 'terse', 'brief', s);
    expect(loadPresets(s).map((p) => p.name)).toEqual(['brief']);
    const r4 = deletePreset(r3.list, 'brief', s);
    expect(r4.list).toEqual([]);
    expect(loadPresets(s)).toEqual([]);
  });

  it('a throwing store: nothing persists, nothing throws', () => {
    expect(loadPresets(throwing)).toEqual([]);
    const r = upsertPreset([], presetFrom('x', emptySetup()), throwing);
    expect(r.persisted).toBe(false);
    expect(r.list).toHaveLength(1);
    expect(saveDraft({ version: 1, setup: emptySetup(), messages: [], presetName: null }, throwing)).toBe(false);
    expect(loadDraft(throwing)).toBeNull();
    expect(() => clearDraft(throwing)).not.toThrow();
  });

  it('setupEquals ignores unset params and compares tool_choice by value', () => {
    const a = { ...emptySetup(), params: { temperature: 0.5, top_p: undefined } };
    const b = { ...emptySetup(), params: { temperature: 0.5 } };
    expect(setupEquals(a, b)).toBe(true);
    expect(setupEquals(a, { ...b, tool_choice: 'required' })).toBe(false);
  });
});

describe('import / export', () => {
  it('exports a document and imports it back, merging by name', () => {
    const s = memStore();
    const list = upsertPreset([], presetFrom('a', { ...emptySetup(), system: 'A' }), s).list;
    const doc = exportPresets(list);
    expect(JSON.parse(doc)).toMatchObject({ format: 'qse-playground-presets', version: 1, presets: [{ name: 'a' }] });
    const other = upsertPreset([], presetFrom('b', emptySetup()), memStore()).list;
    const r = importPresets(other, doc, s);
    expect(r.added).toBe(1);
    expect(r.errors).toEqual([]);
    expect(r.list.map((p) => p.name)).toEqual(['a', 'b']);
    expect(loadPresets(s).map((p) => p.name)).toEqual(['a', 'b']);
  });

  it('refuses a file that is not a preset array and says why; unknown params are dropped', () => {
    expect(importPresets([], '{"hello": 1}', memStore())).toMatchObject({ added: 0, errors: [expect.stringMatching(/expected a JSON array/)] });
    expect(importPresets([], 'nope', memStore())).toMatchObject({ added: 0, errors: [expect.stringMatching(/not valid JSON/)] });
    const r = parsePresets([{ name: 'ok', params: { temperature: 1, min_p: 0.1, junk: true }, tool_choice: { type: 'function', function: { name: 'f' } }, tools: [{ type: 'function', function: { name: 'f' } }] }, { params: {} }, 'x']);
    expect(r.presets).toHaveLength(1);
    expect(r.presets[0].params).toEqual({ temperature: 1 });
    expect(r.presets[0].tool_choice).toEqual({ type: 'function', function: { name: 'f' } });
    expect(JSON.parse(r.presets[0].toolsText)).toHaveLength(1);
    expect(r.errors).toEqual(['preset 1: needs a name', 'preset 2: not an object']);
  });
});

describe('draft', () => {
  it('round-trips the setup and the transcript; a broken draft is ignored', () => {
    const s = memStore();
    const d = { version: 1 as const, setup: { ...emptySetup(), system: 'S' }, messages: [{ uid: 'u', role: 'user' as const, content: 'hi' }], presetName: 'p' };
    expect(saveDraft(d, s)).toBe(true);
    expect(loadDraft(s)).toEqual(d);
    s.setItem('qse.playground.draft', '{"version": 2}');
    expect(loadDraft(s)).toBeNull();
    clearDraft(s);
    expect(s.data.has('qse.playground.draft')).toBe(false);
  });
});
