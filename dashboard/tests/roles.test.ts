// SRV-134's vitest rows for the role contract (SRV-133): the roles form's rules (the window and
// writes a kind allows, the forms of sharing its tokenizer hash allows, the defaults a stored
// version was loaded with), the Deciders screen's contract line, and the mock engine's roles
// route against docs/contract/conductor-v1 -- a complete role saves, a missing claim or window is
// refused naming the field, a clerk that writes client-visible tokens is refused. The server's
// own tests (tests/test_role_contract.py) hold the real responses to the same files.
import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import Ajv2020 from 'ajv/dist/2020.js';
import { ENGINE_SHA, MockConductor } from '../mock/conductor.ts';
import type { RoleDoc } from '../src/api/conductor';
import { defaultsByRole, ROLE_KINDS, sharingOptions, windowRefusal, writesRefusal } from '../src/lib/conductor';
import { roleWords } from '../src/lib/deciders';

const DIR = join(__dirname, '..', '..', 'docs', 'contract', 'conductor-v1');
const ajv = new Ajv2020({ allErrors: true, strict: true });
for (const f of readdirSync(DIR).filter((f) => f.endsWith('.schema.json'))) ajv.addSchema(JSON.parse(readFileSync(join(DIR, f), 'utf8')), f);
function check(name: string, data: unknown) {
  const v = ajv.getSchema(`${name}.schema.json`);
  expect(v, `schema ${name}`).toBeDefined();
  if (!v!(data)) throw new Error(`${name}: ${ajv.errorsText(v!.errors, { separator: '\n' })}`);
}
const q = () => new URLSearchParams();

const clerk: RoleDoc = {
  endpoint: 'http://127.0.0.1:8031', model: 'gemma-4-e4b', contract: 'openai-chat', timeout_ms: 20000, memory_claim_gib: 11, mode: 'shadow',
  kind: 'clerk', window: 'idle', preemptible: true, latency_budget_ms: 1500, fallback: 'skip', tokenizer_sha256: 'none', writes: 'hidden_notes', sharing: ['text'],
};

describe('the roles form’s rules', () => {
  it('in decode is for a decider or a probe only', () => {
    for (const k of ROLE_KINDS) expect(windowRefusal(k, 'in_decode') === null, k).toBe(k === 'decider' || k === 'probe');
    expect(windowRefusal('clerk', 'idle')).toBeNull();
    expect(windowRefusal('librarian', 'any')).toBeNull();
  });
  it('between Qwen’s rounds and the state lines are for a generative role only (P1, the narrator)', () => {
    for (const k of ROLE_KINDS) {
      expect(windowRefusal(k, 'between_rounds') === null, k).toBe(k === 'generative');
      expect(writesRefusal(k, 'state_lines') === null, k).toBe(k === 'generative');
    }
    expect(writesRefusal('generative', 'client_visible')).toMatch(/honesty rule/);
  });
  it('only the reasoner writes client-visible tokens (scenario 3)', () => {
    for (const k of ROLE_KINDS) expect(writesRefusal(k, 'client_visible') === null, k).toBe(k === 'reasoner');
    expect(writesRefusal('clerk', 'client_visible')).toMatch(/honesty rule/);
    expect(writesRefusal('clerk', 'memory_files')).toBeNull();
  });
  it('the tokenizer hash decides the forms of sharing', () => {
    const allowed = (r: Pick<RoleDoc, 'kind' | 'tokenizer_sha256'>, sha: string | null) => sharingOptions(r, sha).filter((o) => o.allowed).map((o) => o.form);
    expect(allowed({ kind: 'clerk', tokenizer_sha256: ENGINE_SHA }, ENGINE_SHA)).toEqual(['text', 'tokens']);
    expect(allowed({ kind: 'clerk', tokenizer_sha256: 'cd'.repeat(32) }, ENGINE_SHA)).toEqual(['text']);
    expect(allowed({ kind: 'clerk', tokenizer_sha256: ENGINE_SHA }, null)).toEqual(['text']);
    expect(allowed({ kind: 'probe', tokenizer_sha256: 'none' }, ENGINE_SHA)).toEqual(['text', 'states']);
    const why = Object.fromEntries(sharingOptions({ kind: 'clerk', tokenizer_sha256: 'none' }, ENGINE_SHA).map((o) => [o.form, o.why]));
    expect(why).toEqual({ text: null, tokens: 'needs the role’s tokenizer hash', states: 'a probe inside the engine only' });
  });
  it('groups the defaults a stored version was loaded with by role (scenario 4)', () => {
    expect(defaultsByRole(['decider.memory_claim_gib=5.1 (memory_gb)', 'decider.window=idle', 'reader.kind=clerk', 'junk'])).toEqual({
      decider: ['memory_claim_gib=5.1 (memory_gb)', 'window=idle'],
      reader: ['kind=clerk'],
    });
    expect(defaultsByRole(undefined)).toEqual({});
  });
  it('the Deciders screen says the contract a model signs in one line', () => {
    expect(roleWords({ role: { kind: 'decider', memory_claim_gib: 4.4, window: 'idle', preemptible: true, latency_budget_ms: 100, fallback: 'PASS', tokenizer_sha256: 'none', writes: 'nothing', sharing: ['text'] } })).toBe(
      'decider · idle · preemptible · 100 ms then PASS · writes nothing · shares text · claims 4.4 GiB',
    );
    expect(roleWords({ role: undefined })).toBeNull();
  });
});

describe('the roles route on the mock engine', () => {
  const put = (m: MockConductor, doc: unknown, note = 'x') => m.handle('PUT', '/v1/conductor/roles', q(), { doc, note }, 't');
  const roles = (m: MockConductor) => m.handle('GET', '/v1/conductor/roles', q(), {}, 't').body as { version: number; doc: Record<string, RoleDoc>; contract: { engine_tokenizer_sha256: string; roles: Record<string, { sharing_allowed: string[]; tokenizer_match: boolean | null }> } };
  it('the built-in roles sign the contract and the route says what each may share', () => {
    const m = new MockConductor(7);
    const r = roles(m);
    check('roles', r);
    for (const [n, x] of Object.entries(r.doc)) expect(Object.keys(x), n).toEqual(expect.arrayContaining(['kind', 'memory_claim_gib', 'window', 'preemptible', 'latency_budget_ms', 'fallback', 'tokenizer_sha256', 'writes', 'sharing']));
    expect(r.contract.engine_tokenizer_sha256).toBe(ENGINE_SHA);
    expect(r.contract.roles.decider).toEqual({ sharing_allowed: ['text'], tokenizer_match: null });
    const meta = m.handle('GET', '/v1/conductor/meta', q(), {}, 't').body as { roles: { windows: string[] } };
    check('meta', meta);
    expect(meta.roles.windows).toEqual(['idle', 'in_decode', 'any', 'between_rounds']);
  });
  it('a complete role saves and reads back whole (scenario 1)', () => {
    const m = new MockConductor(7);
    const r = roles(m);
    const w = put(m, { ...r.doc, clerk }, 'a clerk in shadow');
    expect(w.status).toBe(200);
    check('write', w.body);
    const after = roles(m);
    expect(after.version).toBe(r.version + 1);
    expect(after.doc.clerk).toEqual(clerk);
  });
  it('a missing claim or window is refused naming the field (scenario 2)', () => {
    const m = new MockConductor(7);
    const r = roles(m);
    for (const f of ['memory_claim_gib', 'window'] as const) {
      const { [f]: _gone, ...rest } = clerk;
      const w = put(m, { ...r.doc, clerk: rest });
      expect(w.status, f).toBe(422);
      check('error', w.body);
      expect(JSON.stringify(w.body), f).toContain(f);
      expect(roles(m).version).toBe(r.version);
    }
  });
  it('a clerk that writes client-visible tokens is refused, a reasoner is not (scenario 3)', () => {
    const m = new MockConductor(7);
    const r = roles(m);
    const w = put(m, { ...r.doc, clerk: { ...clerk, writes: 'client_visible' } });
    expect(w.status).toBe(422);
    expect(JSON.stringify(w.body)).toMatch(/honesty rule/);
    expect(put(m, { ...r.doc, writer: { ...clerk, kind: 'reasoner', window: 'any', writes: 'client_visible' } }).status).toBe(200);
  });
  it('token sharing needs the engine’s own tokenizer hash', () => {
    const m = new MockConductor(7);
    const r = roles(m);
    expect(put(m, { ...r.doc, gemma: { ...clerk, tokenizer_sha256: 'cd'.repeat(32), sharing: ['text', 'tokens'] } }).status).toBe(422);
    expect(put(m, { ...r.doc, small_qwen: { ...clerk, tokenizer_sha256: ENGINE_SHA, sharing: ['text', 'tokens'] } }).status).toBe(200);
    expect(roles(m).contract.roles.small_qwen).toEqual({ sharing_allowed: ['text', 'tokens'], tokenizer_match: true });
  });
  it('the decision models carry the contract they sign', () => {
    const m = new MockConductor(7);
    const s = m.handle('GET', '/v1/conductor/deciders/status', q(), {}, 't').body as { models: { role: { kind: string; memory_claim_gib: number } }[] };
    check('deciders-status', s);
    expect(s.models.every((x) => x.role.kind === 'decider' && x.role.memory_claim_gib === 4.4)).toBe(true);
  });
});
