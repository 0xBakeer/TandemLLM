// The Backends screen's helpers and the mock proxy (contract 1.3): money, lists, locks, the
// write-only key, a user's key shown once, the env rule that cannot change.

import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { money, NOT_REPORTED } from '../src/lib/format';
import { listOf, lockOf, modelPatch, writerConfig, writerPatch } from '../src/views/backends';
import { MockBackends } from '../mock/backends';
import { WRITER_DEFAULTS, type BackendConfig } from '../src/api/config';

describe('money', () => {
  it('formats micro-dollars by size', () => {
    expect(money(null)).toBe('—');
    expect(money(0)).toBe('$0');
    expect(money(812)).toBe('$0.0008');
    expect(money(41_230)).toBe('$0.04');
    expect(money(1_200_000)).toBe('$1.20');
    expect(money(1_234_000_000)).toBe('$1,234');
  });
  it('names what the proxy cannot see', () => {
    expect(NOT_REPORTED).toBe('not reported by this backend');
  });
});

describe('form helpers', () => {
  it('splits a comma or line list', () => {
    expect(listOf(' a, b ,,\nc ')).toEqual(['a', 'b', 'c']);
    expect(listOf('')).toEqual([]);
  });
  it('reads a lock by field', () => {
    const b = { locks: { url: 'TANDEM_BACKEND_X_URL' } } as unknown as BackendConfig;
    expect(lockOf(b, 'url')).toBe('TANDEM_BACKEND_X_URL');
    expect(lockOf(b, 'label')).toBeNull();
    expect(lockOf(null, 'url')).toBeNull();
  });
});

describe('the mock proxy', () => {
  it('locks env fields with a 422 that names the variable', () => {
    const m = new MockBackends();
    const [st, body] = m.handle('PATCH', '/v1/config/backends/deepseek', { url: 'https://x/v1' });
    expect(st).toBe(422);
    expect((body as { error: { variable: string } }).error.variable).toBe('TANDEM_BACKEND_DEEPSEEK_URL');
    const [ok] = m.handle('PATCH', '/v1/config/backends/deepseek', { label: 'DS' });
    expect(ok).toBe(200);
  });
  it('never returns a key, and a user key once', () => {
    const m = new MockBackends();
    const [st, b] = m.handle('POST', '/v1/config/backends', { id: 'oa', kind: 'openai', key: 'sk-fixture-ABCDEFGH1234' });
    expect(st).toBe(201);
    expect(JSON.stringify(b)).not.toContain('sk-fixture-ABCDEFGH1234');
    expect(JSON.stringify(m.overview())).not.toContain('sk-fixture-ABCDEFGH1234');
    const [, u] = m.handle('POST', '/v1/config/users', { label: 'laptop' });
    const key = (u as { key: string }).key;
    expect(key.startsWith('tdm-')).toBe(true);
    expect(JSON.stringify(m.overview())).not.toContain(key);
  });
  it('keeps the env rule and shows Setup when empty', () => {
    const m = new MockBackends();
    const [st] = m.handle('PUT', '/v1/config/rules', { rules: [] });
    expect(st).toBe(422);
    m.empty();
    expect((m.overview() as { setup_needed: boolean }).setup_needed).toBe(true);
  });
});

describe('the memory writer switches (0.3.36)', () => {
  const dr = (o: Record<string, unknown> = {}) => ({ writer: 'auto' as const, writer_usd_day: '0.25', writer_nocap: false, writer_runs_day: '48', writer_prompt_tokens: '32000', ...o });
  it('resolves from the API, else the harness document over the defaults', () => {
    expect(writerConfig(null)).toEqual(WRITER_DEFAULTS);
    const h = { harness: { mode: 'on', profile: null, writer: 'shadow', writer_usd_day: null } } as unknown as BackendConfig;
    expect(writerConfig(h)).toEqual({ ...WRITER_DEFAULTS, writer: 'shadow', writer_usd_day: null });
    const v = { harness: { mode: 'off', profile: null }, writer: { config: { ...WRITER_DEFAULTS, writer_runs_day: 5 }, today: { runs: 0, cost_micros: 0 } } } as unknown as BackendConfig;
    expect(writerConfig(v).writer_runs_day).toBe(5);
  });
  it('sends only what changed, never a locked field, and null for no cap', () => {
    expect(writerPatch(dr(), WRITER_DEFAULTS)).toEqual({ patch: {}, error: null });
    expect(writerPatch(dr({ writer: 'shadow', writer_runs_day: '5' }), WRITER_DEFAULTS).patch).toEqual({ writer: 'shadow', writer_runs_day: 5 });
    expect(writerPatch(dr({ writer_nocap: true }), WRITER_DEFAULTS).patch).toEqual({ writer_usd_day: null });
    const locked = writerPatch(dr({ writer_usd_day: '1', writer: 'on' }), WRITER_DEFAULTS, { 'harness.writer_usd_day': 'TANDEM_BACKEND_X_WRITER_USD_DAY' });
    expect(locked.patch).toEqual({ writer: 'on' });
  });
  it('refuses out-of-range values and names the field', () => {
    expect(writerPatch(dr({ writer_runs_day: '-1' }), WRITER_DEFAULTS).error?.field).toBe('harness.writer_runs_day');
    expect(writerPatch(dr({ writer_runs_day: '2.5' }), WRITER_DEFAULTS).error?.field).toBe('harness.writer_runs_day');
    expect(writerPatch(dr({ writer_prompt_tokens: '999' }), WRITER_DEFAULTS).error?.field).toBe('harness.writer_prompt_tokens');
    expect(writerPatch(dr({ writer_usd_day: '' }), WRITER_DEFAULTS).error?.field).toBe('harness.writer_usd_day');
    expect(writerPatch(dr({ writer_usd_day: '2000' }), WRITER_DEFAULTS).error?.field).toBe('harness.writer_usd_day');
  });
  it('the mock resolves auto per model, merges the harness and keeps the env lock', () => {
    const m = new MockBackends();
    type V = { id: string; writer: { config: Record<string, unknown>; settings_mode: string; shadow_log: string }; model_rows: { model: string; writer: { mode: string; why: string } }[]; harness: Record<string, unknown> };
    const ov = m.overview() as unknown as { backends: V[] };
    const ds = ov.backends.find((b) => b.id === 'deepseek')!;
    expect(ds.writer.config.writer_usd_day).toBe(0.1);
    expect(ds.model_rows.find((r) => r.model === 'deepseek-chat')!.writer).toEqual({ mode: 'off', why: 'paid_cloud' });
    expect(ov.backends.find((b) => b.id === 'vllm1')!.model_rows[0].writer).toEqual({ mode: 'on', why: 'auto' });
    expect(ds.writer.shadow_log).toContain('/shadow/writer-');
    const [st, nb] = m.handle('PATCH', '/v1/config/backends/vllm1', { harness: { writer: 'shadow' } });
    expect(st).toBe(200);
    const v = nb as unknown as V;
    expect(v.harness.mode).toBe('on');
    expect(v.harness.profile).toBe('coding');
    expect(v.model_rows[0].writer).toEqual({ mode: 'shadow', why: 'backend' });
    const [bad, body] = m.handle('PATCH', '/v1/config/backends/deepseek', { harness: { writer_usd_day: 1 } });
    expect(bad).toBe(422);
    expect((body as { error: { variable: string } }).error.variable).toBe('TANDEM_BACKEND_DEEPSEEK_WRITER_USD_DAY');
    expect(m.handle('PATCH', '/v1/config/backends/deepseek', { harness: { writer_usd_day: 0.1, writer: 'on' } })[0]).toBe(200);
    expect(m.handle('PATCH', '/v1/config/backends/vllm1', { harness: { writer_runs_day: -3 } })[0]).toBe(422);
  });
});

describe("a model row's output limit (VIS-227)", () => {
  it('sends a whole number, null for an empty field, nothing when untouched', () => {
    expect(modelPatch({ public_id: 'qwen' })).toEqual({ patch: { public_id: 'qwen' }, error: null });
    expect(modelPatch({ output: ' 8192 ' }).patch).toEqual({ max_output: 8192 });
    expect(modelPatch({ output: '' }).patch).toEqual({ max_output: null });
    expect(modelPatch({ output: '4096', enabled: false, price: null }).patch).toEqual({ enabled: false, max_output: 4096, price: null });
  });
  it('refuses anything else and names the field', () => {
    // the field is text: what was typed arrives as typed
    for (const bad of ['0', '-5', '2.5', 'many', '1e400', '16k', '8,192', '0x10', '1e3', '99999999999999999999']) {
      const r = modelPatch({ output: bad, public_id: 'x' });
      expect(r.error?.field, bad).toBe('max_output');
      expect(r.patch).toEqual({});
    }
  });
  it("the mock stores it as the operator's, apart from the provider's, and refuses a bad one", () => {
    const m = new MockBackends();
    const [st, row] = m.handle('PUT', '/v1/config/backends/vllm1/models/Qwen/Qwen3.5-4B', { max_output: 8192 });
    expect(st).toBe(200);
    expect(row).toMatchObject({ max_output_set: 8192, max_output: null });
    expect(m.handle('PUT', '/v1/config/backends/vllm1/models/Qwen/Qwen3.5-4B', { max_output: null })[1]).toMatchObject({ max_output_set: null });
    const [bad, body] = m.handle('PUT', '/v1/config/backends/vllm1/models/Qwen/Qwen3.5-4B', { max_output: 0 });
    expect(bad).toBe(422);
    expect((body as { error: { field: string } }).error.field).toBe('max_output');
  });
  it('the row editor has the field beside the public id and saves through modelPatch', () => {
    const src = readFileSync(new URL('../src/views/backends.ts', import.meta.url), 'utf8');
    const row = src.slice(src.indexOf('private renderModelRow'), src.indexOf('private async saveModel'));
    expect(row.indexOf('public id`}')).toBeLessThan(row.indexOf('max output`}'));
    expect(row).toContain('output: (e.target as HTMLInputElement).value');
    expect(row).toContain('m.max_output_set');
    expect(row).toContain('class="bk-out" type="text" inputmode="numeric"');
    expect(row).toContain('placeholder=${m.max_output_default ?');
    expect(row).toContain('colspan="9"');
    const save = src.slice(src.indexOf('private async saveModel'), src.indexOf('private async saveModel') + 400);
    expect(save).toContain('modelPatch(this.modelDrafts[key] ?? {})');
  });
});
