// VIS-203 (QA VIS-204): the composer's model picker listed the conductor's profiles (each client's
// harness: opencode, Open WebUI, Hermes, A/B tests, header-only ones) as the only buttons, so a model
// without profiles (DeepSeek through the proxy) was a heading nobody could pick (2026-10-16).
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { QseChatComposer, modelDesc, modelPatch } from '../src/chat/composer';
import type { Capabilities, ChatSettings, ModelInfo } from '../src/chat/types';
import { CAPABILITIES, DEFAULT_SETTINGS } from '../src/chat/mock-data';

/** A template's text with its values in place (nested templates and lists flattened). */
function text(v: unknown): string {
  if (Array.isArray(v)) return v.map(text).join('');
  if (v && typeof v === 'object' && 'strings' in v && 'values' in v) {
    const t = v as { strings: readonly string[]; values: unknown[] };
    return t.strings.map((s, i) => s + (i < t.values.length ? text(t.values[i]) : '')).join('');
  }
  return typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean' ? String(v) : '';
}

const ENGINE: ModelInfo = { id: 'TandemLLM', name: 'Nemotron 3.5 Super VL 120B-A12B', weights: 'nemotron-super-vl-exl3-4.0', max_len: 262_144, profiles: [], backend: 'engine' };
const FLASH: ModelInfo = { id: 'deepseek-flash', name: 'DeepSeek V4 Flash (cloud)', weights: '', max_len: 1_048_576, profiles: [], backend: 'cloud' };
const PRO: ModelInfo = { id: 'deepseek-v4-pro', name: 'DeepSeek V4 Pro (cloud)', weights: '', max_len: 1_048_576, profiles: [], backend: 'cloud' };
const CAPS: Capabilities = { ...CAPABILITIES, models: [ENGINE, FLASH, PRO] };

function pop(settings: ChatSettings) {
  const patched: Partial<ChatSettings>[] = [];
  const el = Object.create(QseChatComposer.prototype) as QseChatComposer & { patch: (p: Partial<ChatSettings>) => void; pop: string };
  Object.defineProperty(el, 'patch', { value: (p: Partial<ChatSettings>) => patched.push(p) });
  Object.defineProperty(el, 'pop', { value: 'model', writable: true });
  const tpl = (el as unknown as { renderModelPop(c: Capabilities, s: ChatSettings): { strings: readonly string[]; values: unknown[] } }).renderModelPop(CAPS, settings);
  const rows = tpl.values[0] as { strings: readonly string[]; values: unknown[] }[];
  return { tpl, rows, patched };
}

describe('the model picker', () => {
  it('lists the models only, one row each, every row a button', () => {
    const { tpl, rows } = pop({ ...DEFAULT_SETTINGS, model: 'TandemLLM' });
    expect(rows).toHaveLength(3);
    for (const r of rows) expect(r.strings[0]).toContain('<button');
    const out = text(tpl);
    expect(out).toContain('Nemotron 3.5 Super VL 120B-A12B');
    expect(out).toContain('DeepSeek V4 Flash (cloud)');
    expect(out).toContain('DeepSeek V4 Pro (cloud)');
    expect(out).not.toContain('ch-pop-title');
  });

  it('a model without profiles is picked like any other: the model, no profile', () => {
    const { rows, patched } = pop({ ...DEFAULT_SETTINGS, model: 'TandemLLM' });
    const click = rows[1].values.find((v) => typeof v === 'function') as () => void;
    click();
    expect(patched).toEqual([{ model: 'deepseek-flash', profile: null }]);
    expect(modelPatch(ENGINE)).toEqual({ model: 'TandemLLM', profile: null });
  });

  it("marks the chat's model, the first one when the chat names none it knows", () => {
    const on = (s: ChatSettings) => pop(s).rows.map((r) => r.values.includes(true));
    expect(on({ ...DEFAULT_SETTINGS, model: 'deepseek-v4-pro' })).toEqual([false, false, true]);
    expect(on({ ...DEFAULT_SETTINGS, model: 'gone' })).toEqual([true, false, false]);
  });

  it('a stored profile shows nowhere in the popover', () => {
    const out = text(pop({ ...DEFAULT_SETTINGS, model: 'TandemLLM', profile: 'ui-check' }).tpl);
    expect(out).not.toContain('ui-check');
  });

  it('the line under a name leaves out what the backend does not report', () => {
    expect(modelDesc(FLASH)).toBe('1024k window');
    expect(modelDesc(ENGINE)).toBe('nemotron-super-vl-exl3-4.0, 256k window');
    expect(modelDesc({ ...FLASH, max_len: 0 })).toBe('');
  });

  it('on a phone the popover scrolls inside itself, under the top bar', () => {
    const css = readFileSync(new URL('../src/styles/chat.css', import.meta.url), 'utf8');
    const base = css.slice(css.indexOf('.ch-pop {'), css.indexOf('}', css.indexOf('.ch-pop {')));
    expect(base).toContain('overflow: auto');
    expect(base).toContain('max-height: 60vh');
    expect(css).toMatch(/max-height: min\(60vh, calc\(100dvh - var\(--topbar-h\) - 150px\)\)/);
  });
});
