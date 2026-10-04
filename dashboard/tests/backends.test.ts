// The Backends screen's helpers and the mock proxy (contract 1.3): money, lists, locks, the
// write-only key, a user's key shown once, the env rule that cannot change.

import { describe, expect, it } from 'vitest';
import { money, NOT_REPORTED } from '../src/lib/format';
import { listOf, lockOf } from '../src/views/backends';
import { MockBackends } from '../mock/backends';
import type { BackendConfig } from '../src/api/config';

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
