// npm run test:contract — every example file and every mock response validates against its
// schema in docs/contract/dashboard-v1/. The backend (SRV-29) validates its real responses
// against the same files.
import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import Ajv2020 from 'ajv/dist/2020.js';
import { generateYear } from '../mock/generate.ts';
import { requests, summary, usage, toRequestRow } from '../mock/aggregate.ts';
import { LogRing, reqLine } from '../mock/logs.ts';
import { rng } from '../mock/generate.ts';
import { addDays, dayKey } from '../src/lib/time.ts';

const DIR = join(__dirname, '..', '..', 'docs', 'contract', 'dashboard-v1');
const ajv = new Ajv2020({ allErrors: true, strict: true });
const schemas: Record<string, ReturnType<typeof ajv.compile>> = {};
for (const f of readdirSync(DIR)) {
  if (f.endsWith('.schema.json')) schemas[f.replace('.schema.json', '')] = ajv.compile(JSON.parse(readFileSync(join(DIR, f), 'utf8')));
}

function check(name: string, data: unknown) {
  const v = schemas[name];
  expect(v, `schema ${name}`).toBeDefined();
  const ok = v(data);
  if (!ok) throw new Error(`${name}: ${ajv.errorsText(v.errors, { separator: '\n' })}`);
}

describe('contract files', () => {
  it('there are eight schemas and eight examples', () => {
    const files = readdirSync(DIR);
    expect(files.filter((f) => f.endsWith('.schema.json'))).toHaveLength(8);
    expect(files.filter((f) => f.endsWith('.example.json'))).toHaveLength(8);
    expect(files).toContain('README.md');
  });
  for (const name of ['summary', 'usage', 'requests', 'system', 'logs-line', 'logs-json', 'session', 'error']) {
    it(`${name}.example.json validates`, () => {
      check(name, JSON.parse(readFileSync(join(DIR, `${name}.example.json`), 'utf8')));
    });
  }
  it('a wrong example is rejected (the schemas are strict)', () => {
    expect(schemas.error({ error: { type: 'weird', message: 'x' } })).toBe(false);
    expect(schemas.session({ authenticated: true })).toBe(false);
    expect(schemas.summary({ contract_version: '2.0' })).toBe(false);
  });
});

describe('mock responses validate', () => {
  const TZ = 'Europe/Berlin';
  const NOW = Date.parse('2026-09-24T14:00:00Z');
  const g = generateYear({ seed: 42, now: NOW, tz: TZ });
  const today = dayKey(NOW, TZ);

  it('summary', () => {
    check('summary', summary({ rows: g.rows, tz: TZ, now: NOW, since: g.since, live: { status: 'ok', running: 0, waiting: 0, uptime_s: 12.5, last_request_at: new Date(NOW).toISOString() } }));
  });
  it('usage day and hour', () => {
    check('usage', usage(g.rows, { from: addDays(today, -364), to: today, bucket: 'day', tz: TZ }));
    check('usage', usage(g.rows, { from: addDays(today, -2), to: today, bucket: 'hour', tz: TZ, client: 'anon' }));
  });
  it('requests (500 rows, every finish reason)', () => {
    check('requests', { contract_version: '1.0', ...requests(g.rows, { limit: 500 }) });
    for (const r of g.rows.slice(-2000)) check('requests', { contract_version: '1.0', next_before: null, requests: [toRequestRow(r)] });
  });
  it('log lines', () => {
    const ring = new LogRing(200);
    ring.seed(g.rows, rng(3), 150);
    const lines = ring.backlog({ level: 'debug', limit: 200 });
    expect(lines.length).toBeGreaterThan(50);
    for (const l of lines) check('logs-line', l);
    check('logs-json', { contract_version: '1.0', lines, last_seq: ring.lastSeq });
    expect(reqLine(g.rows[g.rows.length - 1])).toMatch(/^\[req\] /);
  });
  it('session and error bodies', () => {
    check('session', { authenticated: true, expires_at: '2026-09-25T04:40:00Z' });
    check('error', { error: { type: 'too_many', message: 'at most 4 log subscribers' } });
  });
});
