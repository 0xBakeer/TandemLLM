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
import { MockLive } from '../mock/live.ts';
import { makeRow } from '../mock/generate.ts';
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
  it('there are ten schemas and ten examples', () => {
    const files = readdirSync(DIR);
    expect(files.filter((f) => f.endsWith('.schema.json'))).toHaveLength(10);
    expect(files.filter((f) => f.endsWith('.example.json'))).toHaveLength(10);
    expect(files).toContain('README.md');
  });
  for (const name of ['summary', 'usage', 'requests', 'system', 'logs-line', 'logs-json', 'gap', 'session', 'error', 'live']) {
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
  it('live 1.1: the agent turn, at four events a second, every message validates and walks the states in order', () => {
    const r = rng(12);
    const live = new MockLive(r, 812);
    live.agentTurn((forced) => makeRow(r, NOW, false, forced), NOW);
    const seen: string[] = [];
    let engineStates = new Set<string>();
    for (let i = 0; i <= 62 * 4; i++) {
      const t = NOW + i * 250;
      if (i % 4 === 0) live.tick(t);
      const snap = live.snapshot(t, i === 0);
      check('live', snap);
      expect(snap.contract_version).toBe('1.1');
      expect(snap.seq).toBe(i + 1);
      engineStates.add(snap.engine!.state);
      const running = snap.requests.find((x) => x.phase !== 'done');
      const st = running?.activity?.state ?? (snap.engine!.state === 'waiting_for_client' ? 'waiting' : 'idle');
      if (seen[seen.length - 1] !== st) seen.push(st);
      if (running?.activity?.state === 'prefilling' && running.activity.prefill?.pct != null) {
        expect(running.activity.label).toMatch(/^Prefilling [\d,]+ of (48,210|48,930) \(\d+ %\)$/);
        expect(running.activity.prefill.done).toBeLessThanOrEqual(running.activity.prefill.total);
      }
      if (running?.activity?.state === 'tool_call') {
        expect(running.activity.label).toBe('Calling tool write_file');
        expect(running.activity.tool!.arg_bytes).toBeGreaterThan(0);
      }
      // the sparkline sample is one a second, whatever the event rate
      expect(snap.sample!.t).toBe(Math.round(NOW + Math.floor(i / 4) * 4 * 250) / 1000);
    }
    expect(seen.join(' > ')).toBe(['queued', 'prefilling', 'thinking', 'writing', 'tool_call', 'finishing', 'waiting', 'queued', 'prefilling', 'thinking', 'writing', 'finishing', 'idle'].join(' > '));
    expect([...engineStates].sort()).toEqual(['busy', 'idle', 'waiting_for_client']);
    const last = live.snapshot(NOW + 63_000, true);
    expect(last.recent!.map((x) => x.stop.reason)).toEqual(['stop', 'tool_calls']);
    expect(last.recent![1].stop.sentence).toMatch(/^tool call: write_file after \d+\.\d s$/);
    expect(last.recent![1].path).toEqual(['queued', 'prefilling', 'thinking', 'writing', 'tool_call', 'finishing']);
    const second = last.requests.find((x) => x.request_id.startsWith('chatcmpl-3f1c'))!;
    expect(second.activity!.continues).toMatchObject({ request_id: 'chatcmpl-77aa1c0e2b7d4f5a9c3b', tool_names: ['write_file'], inferred: false });
  });

  it('live 1.1: abandoned, stops, constrained, the kill switch and the 1.0 fallback', () => {
    const r = rng(13);
    const live = new MockLive(r, 812);
    live.abandoned((forced) => makeRow(r, NOW, false, forced), NOW);
    let sawGone = false;
    for (let i = 0; i <= 40; i++) {
      const t = NOW + i * 250;
      if (i % 4 === 0) live.tick(t);
      const snap = live.snapshot(t, i === 0);
      check('live', snap);
      const row = snap.requests[0];
      if (row.phase === 'prefill' && row.activity!.client.connected === false) sawGone = true;
    }
    expect(sawGone).toBe(true);
    const ab = live.snapshot(NOW + 10_000, true);
    check('live', ab);
    expect(ab.recent![0].stop).toMatchObject({ reason: 'abandoned', state: 'prefilling', tokens_sent: 0, detail: 'left_during_prefill' });
    expect(ab.recent![0].stop.sentence).toMatch(/^abandoned by the client after \d+\.\d s of silent prefill, 0 tokens sent$/);
    expect(ab.recent![0].stop.client_gone_ms).toBeGreaterThan(0);

    live.clear();
    live.stops((forced) => makeRow(r, NOW, false, forced), NOW);
    const st = live.snapshot(NOW, true);
    check('live', st);
    expect(st.recent!.map((x) => x.stop.reason)).toEqual(['stop', 'length', 'tool_calls', 'timeout', 'abandoned', 'error', 'refused', 'rejected', 'cancelled', 'stop']);
    expect(st.recent!.map((x) => x.stop.sentence)).toEqual([
      expect.stringMatching(/^finished at the end of the answer after 41\.2 s, 812 tokens sent$/),
      'stopped at the length limit (32,000 tokens)',
      'tool call: write_file, bash (2 calls) after 41.2 s',
      'timed out after 41.2 s',
      'abandoned by the client after 73 s of silent prefill, 0 tokens sent',
      'error: RuntimeError after 5 tokens',
      'refused: queue full',
      'rejected: the prompt is longer than the context window',
      'cancelled: the server was shutting down',
      'finished at a stop string after 41.2 s, 3 tokens sent',
    ]);
    expect(st.engine!.state).toBe('idle');
    expect(st.requests).toEqual([]);

    live.clear();
    live.constrained((forced) => makeRow(r, NOW, false, forced), NOW, 6000);
    const c = live.snapshot(NOW, true);
    check('live', c);
    expect(c.requests.map((x) => x.activity!.constrained)).toEqual(['response_format', 'tool_choice']);
    expect(c.requests[1].activity!.label).toBe('Calling tool write_file');

    live.activity = false;
    const off = live.snapshot(NOW + 100, true);
    check('live', off);
    expect(off.contract_version).toBe('1.1');
    expect(off.requests.every((x) => x.activity === null && x.timeline === null)).toBe(true);
    expect(off.recent).toBeNull();
    expect(off.engine!.waiting_for_client).toBeNull();

    live.activity = true;
    live.contract = '1.0';
    const v10 = live.snapshot(NOW + 200, true); // the 1.0 shape of SRV-34: not the 1.1 schema, by design
    expect(v10.contract_version).toBe('1.0');
    expect(v10.engine).toBeUndefined();
    expect(v10.recent).toBeUndefined();
    expect(v10.seq).toBeUndefined();
    expect(v10.requests.every((x) => !('activity' in x))).toBe(true);
  });

  it('live: idle, the busy scenario, and every second of a scripted run', () => {
    const r = rng(11);
    const live = new MockLive(r, 812);
    check('live', live.snapshot(NOW, true));
    live.scenario((forced) => makeRow(r, NOW, false, forced), NOW);
    for (let i = 0; i < 40; i++) {
      live.tick(NOW + i * 1000);
      const snap = live.snapshot(NOW + i * 1000, i === 0);
      check('live', snap);
      if (i === 0) {
        expect(snap.counts).toMatchObject({ in_flight: 3, queued: 1, prefilling: 1, decoding: 1, completed_1m: 2 });
        expect(snap.requests.map((x) => x.phase)).toEqual(['decode', 'prefill', 'queued', 'done', 'done']);
        expect(snap.requests[3].finish_reason).toBe('stop');
        expect(snap.requests[4].finish_reason).toBe('error');
      }
    }
    const late = live.snapshot(NOW + 39_000, true);
    expect(late.history!.length).toBeGreaterThanOrEqual(40); // four minutes of seeded history plus the run
    expect(late.now.decode_tps).toBeGreaterThan(30);
    expect(late.requests.find((x) => x.phase === 'decode')!.decode_tps_now).toBeGreaterThan(30);
    // the queued one took the lock after 20 s and is prefilling now; the done ones fade after 30 s
    expect(late.counts.queued).toBe(0);
    expect(late.requests.filter((x) => x.phase === 'done').length).toBeLessThan(2);
  });
  it('session and error bodies', () => {
    check('session', { authenticated: true, expires_at: '2026-09-25T04:40:00Z' });
    check('error', { error: { type: 'too_many', message: 'at most 4 log subscribers' } });
  });
});
