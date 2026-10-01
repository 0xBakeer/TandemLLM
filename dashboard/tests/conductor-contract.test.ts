// The conductor's contract (docs/contract/conductor-v1): every response schema compiles strictly,
// the built-in documents validate against the config schemas, and every mock response validates,
// writes included. The server's own tests (tests/test_conductor_e2e.py) hold the real responses
// to the same files.
import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import Ajv2020 from 'ajv/dist/2020.js';
import { MockConductor } from '../mock/conductor.ts';

const DIR = join(__dirname, '..', '..', 'docs', 'contract', 'conductor-v1');
const ajv = new Ajv2020({ allErrors: true, strict: true });
const schemas: Record<string, ReturnType<typeof ajv.compile>> = {};
for (const f of readdirSync(DIR).filter((f) => f.endsWith('.schema.json'))) {
  const s = JSON.parse(readFileSync(join(DIR, f), 'utf8'));
  ajv.addSchema(s, f);
}
for (const f of readdirSync(DIR).filter((f) => f.endsWith('.schema.json'))) schemas[f.replace('.schema.json', '')] = ajv.getSchema(f)!;

function check(name: string, data: unknown) {
  const v = schemas[name];
  expect(v, `schema ${name}`).toBeDefined();
  if (!v(data)) throw new Error(`${name}: ${ajv.errorsText(v.errors, { separator: '\n' })}`);
}

describe('conductor contract files', () => {
  it('has the response schemas and the five config schemas', () => {
    const names = Object.keys(schemas).sort();
    for (const n of ['classify', 'config', 'decision', 'decisions', 'diff', 'document', 'error', 'evals', 'export', 'meta', 'profiles', 'roles', 'role-test', 'scan', 'session', 'sessions', 'stats', 'versions', 'write']) expect(names).toContain(n);
    for (const k of ['settings', 'router', 'roles', 'packages', 'profiles']) expect(names).toContain(`${k}.config`);
  });
  it('the built-in documents validate', () => {
    const seeds = JSON.parse(readFileSync(join(DIR, 'seeds.json'), 'utf8'));
    for (const k of ['settings', 'router', 'roles', 'packages']) check(`${k}.config`, seeds[k]);
    for (const p of Object.values(seeds.profiles)) check('profiles.config', p);
  });
  it('is strict', () => {
    expect(schemas.write({ version: 1 })).toBe(false);
    expect(schemas['profiles.config']({ mode: 'loud' })).toBe(false);
  });
});

describe('mock responses validate', () => {
  const m = new MockConductor(7);
  const q = (s = '') => new URLSearchParams(s);
  const get = (path: string, qs = '') => m.handle('GET', path, q(qs), {}, 'test');
  it('reads', () => {
    check('config', get('/v1/conductor/config').body);
    check('meta', get('/v1/conductor/meta').body);
    check('profiles', get('/v1/conductor/profiles').body);
    for (const p of ['/v1/conductor/router', '/v1/conductor/packages', '/v1/conductor/settings', '/v1/conductor/profiles/coding', '/v1/conductor/profiles/bench']) check('document', get(p).body);
    check('roles', get('/v1/conductor/roles').body);
    check('versions', get('/v1/conductor/profiles/coding/versions').body);
    check('diff', get('/v1/conductor/profiles/coding/diff', 'from=0&to=1').body);
    check('decisions', get('/v1/conductor/decisions', 'limit=50').body);
    const id = (get('/v1/conductor/decisions').body as { decisions: { id: string }[] }).decisions[0].id;
    check('decision', get(`/v1/conductor/decisions/${id}`).body);
    check('stats', get('/v1/conductor/stats', 'hours=24').body);
    check('sessions', get('/v1/conductor/sessions').body);
    check('session', get('/v1/conductor/sessions/c%3Ases_night_pagoda').body);
    check('export', get('/v1/conductor/sessions/c%3Ases_night_pagoda/export').body);
    check('evals', get('/v1/conductor/evals').body);
    check('scan', get('/v1/conductor/packages/scan').body);
    check('classify', m.handle('POST', '/v1/conductor/classify', q(), { text: 'restyle the page, design_ui', profile: 'coding' }, 't').body);
    check('role-test', m.handle('POST', '/v1/conductor/roles/decider/test', q(), {}, 't').body);
  });
  it('writes and their refusals', () => {
    const cur = get('/v1/conductor/profiles/chat').body as { version: number; doc: Record<string, unknown> };
    const w = m.handle('PUT', '/v1/conductor/profiles/chat', q(), { doc: cur.doc, note: 'same', base_version: cur.version }, 't');
    expect(w.status).toBe(200);
    check('write', w.body);
    check('write', m.handle('POST', '/v1/conductor/profiles/chat/mode', q(), { mode: 'shadow' }, 't').body);
    check('write', m.handle('POST', '/v1/conductor/profiles/chat/restore', q(), { version: 0 }, 't').body);
    const locked = m.handle('PUT', '/v1/conductor/profiles/raw', q(), { doc: cur.doc, note: 'x' }, 't');
    expect(locked.status).toBe(409);
    check('error', locked.body);
    const bad = m.handle('PUT', '/v1/conductor/profiles/chat', q(), { doc: { ...cur.doc, mode: 'loud' }, note: 'x' }, 't');
    expect(bad.status).toBe(422);
    check('error', bad.body);
    const stale = m.handle('PUT', '/v1/conductor/profiles/chat', q(), { doc: cur.doc, note: 'x', base_version: 0 }, 't');
    expect(stale.status).toBe(409);
  });
  it('seeds the promotion cases VIS-42 names', () => {
    const s = get('/v1/conductor/stats', 'hours=24').body as { precision: { profile: string; decision: string; labelled: number; precision: number }[]; monitor: { precision: { task: string; labelled: number; precision: number }[] } };
    const d1 = s.monitor.precision.find((p) => p.task === 'D1')!;
    const e2 = s.monitor.precision.find((p) => p.task === 'E2')!;
    expect([d1.labelled, Math.round(d1.precision * 100) / 100]).toEqual([41, 0.93]);
    expect(e2.labelled).toBe(12);
    expect(s.precision.some((p) => p.profile === 'coding' && p.labelled >= 30 && p.precision >= 0.9)).toBe(true);
    expect(s.precision.some((p) => p.profile === 'chat' && p.labelled < 30)).toBe(true);
  });
});
