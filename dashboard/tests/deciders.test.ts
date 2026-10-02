// VIS-47's vitest rows for the Deciders screen (VIS-46): the contract files of the new routes, the
// mock engine's answers against them (status, load, a refusal, unload, the test box, the
// `deciders` document), and the screen's words (src/lib/deciders.ts). The server's own tests
// (tests/test_decider_api.py) hold the real responses to the same files.
import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { afterEach, describe, expect, it, vi } from 'vitest';
import Ajv2020 from 'ajv/dist/2020.js';
import { MockConductor } from '../mock/conductor.ts';
import { choiceQuestion, picksWords, plannerWords, probsOf, refusalWords, stateBadge } from '../src/lib/deciders';
import { splitSegments } from '../src/lib/residency';

const DIR = join(__dirname, '..', '..', 'docs', 'contract', 'conductor-v1');
const ajv = new Ajv2020({ allErrors: true, strict: true });
for (const f of readdirSync(DIR).filter((f) => f.endsWith('.schema.json'))) ajv.addSchema(JSON.parse(readFileSync(join(DIR, f), 'utf8')), f);
function check(name: string, data: unknown) {
  const v = ajv.getSchema(`${name}.schema.json`);
  expect(v, `schema ${name}`).toBeDefined();
  if (!v!(data)) throw new Error(`${name}: ${ajv.errorsText(v!.errors, { separator: '\n' })}`);
}
const q = () => new URLSearchParams();

afterEach(() => vi.useRealTimers());

describe('the contract of the decision-model routes', () => {
  it('has the files and the built-in deciders document validates', () => {
    for (const n of ['deciders-status', 'decider-action', 'systemone', 'deciders.config']) expect(ajv.getSchema(`${n}.schema.json`), n).toBeDefined();
    const seeds = JSON.parse(readFileSync(join(DIR, 'seeds.json'), 'utf8'));
    check('deciders.config', seeds.deciders);
    expect(seeds.roles.decider.endpoint).toBe('self');
  });
  it('status: what runs is visible (scenario 1)', () => {
    const m = new MockConductor(7);
    const s = m.handle('GET', '/v1/conductor/deciders/status', q(), {}, 't').body as { models: { name: string; state: string; limits: { context_tokens: number } }[]; planner: { allowed: boolean; claim_gib: number; need_gib: number } };
    check('deciders-status', s);
    expect(s.models.map((x) => [x.name, x.state])).toEqual([['laya-english', 'loaded'], ['laya-multilingual', 'not_loaded'], ['laya-typed-decisions', 'not_loaded']]);
    expect(s.models[1].limits.context_tokens).toBe(1024);
    expect([s.planner.allowed, s.planner.claim_gib, s.planner.need_gib]).toEqual([true, 8.2, 21]);
    check('document', m.handle('GET', '/v1/conductor/deciders', q(), {}, 't').body);
  });
  it('L1: the Laya route for scripts: its settings and counts in the status, its responses in the contract', () => {
    const m = new MockConductor(7);
    const s = m.handle('GET', '/v1/conductor/deciders/status', q(), {}, 't').body as { api: { mode: string; rate_per_min: number; stats: { calls: number } } };
    check('deciders-status', s);
    expect([s.api.mode, s.api.rate_per_min, s.api.stats.calls]).toEqual(['on', 240, 12]);
    const seeds = JSON.parse(readFileSync(join(DIR, 'seeds.json'), 'utf8'));
    expect(seeds.deciders.api.mode).toBe('on');
    expect(seeds.deciders.questions.laya_api).toBe('auto');
    expect(seeds.roles.tester.kind).toBe('decider');
    check('laya-ask', { label: 'failure', p: 0.91, sure: true, probs: { failure: 0.91, clean: 0.09 }, chars: 40, read_chars: 40, clipped: false, model: 'laya-english', ms: 31.2 });
    check('laya-ask-texts', { answers: [{ label: 'no', p: 0.9, sure: true, yes: 0.1, probs: { yes: 0.1, no: 0.9 }, chars: 9, read_chars: 9, clipped: false }], model: 'laya-english', ms: 40 });
  });
  it('load goes loading, warming, loaded (scenario 2) and unload ends it (scenario 4)', () => {
    vi.useFakeTimers({ now: new Date('2026-10-01T13:00:00Z') });
    const m = new MockConductor(7);
    const r = m.handle('POST', '/v1/conductor/deciders/models/laya-multilingual/load', q(), {}, 't');
    expect(r.status).toBe(202);
    check('decider-action', r.body);
    const state = () => (m.decidersStatus() as { models: { name: string; state: string }[] }).models.find((x) => x.name === 'laya-multilingual')!.state;
    expect(state()).toBe('loading');
    vi.advanceTimersByTime(1400);
    expect(state()).toBe('warming');
    vi.advanceTimersByTime(800);
    expect(state()).toBe('loaded');
    const u = m.handle('POST', '/v1/conductor/deciders/models/laya-multilingual/unload', q(), {}, 't');
    expect(u.status).toBe(200);
    check('decider-action', u.body);
    expect(state()).toBe('not_loaded');
    expect(m.handle('POST', '/v1/conductor/deciders/models/laya-multilingual/unload', q(), {}, 't').status).toBe(409);
  });
  it('a refused load says why with the numbers, and nothing loads (scenario 3)', () => {
    const m = new MockConductor(7);
    m.deciderMem = 28;
    const r = m.handle('POST', '/v1/conductor/deciders/models/laya-typed-decisions/load', q(), {}, 't');
    expect(r.status).toBe(409);
    check('decider-action', r.body);
    const d = (r.body as { decision: { reason: string; after_gib: number } }).decision;
    // OPS-47: 28 - 7 (the engine's growth to its 77 GiB peak) - 8.2
    expect([d.reason, d.after_gib]).toEqual(['refused_memory', 12.8]);
    const s = m.decidersStatus() as { models: { name: string; state: string; refusal: unknown }[]; last_refusal: unknown };
    check('deciders-status', s);
    const row = s.models.find((x) => x.name === 'laya-typed-decisions')!;
    expect(row.state).toBe('not_loaded');
    expect(refusalWords(row.refusal as never)).toBe('refused: it would break the floor (MemAvailable 28.0 GiB, engine growth 7.0 GiB, claim 8.2 GiB, 12.8 GiB left, 21.0 GiB needed)');
  });
  it('the mapping is a versioned document (scenario 5)', () => {
    const m = new MockConductor(7);
    const cur = m.handle('GET', '/v1/conductor/deciders', q(), {}, 't').body as { version: number; doc: { models: { name: string; load: string }[]; questions: Record<string, string> } };
    const doc = { ...cur.doc, questions: { ...cur.doc.questions, classify: 'laya-english' } };
    const w = m.handle('PUT', '/v1/conductor/deciders', q(), { doc, note: 'classify by the English checkpoint' }, 't');
    expect(w.status).toBe(200);
    check('write', w.body);
    expect((m.handle('GET', '/v1/conductor/deciders/versions', q(), {}, 't').body as { versions: { version: number; note: string }[] }).versions[0]).toMatchObject({ version: 1, note: 'classify by the English checkpoint' });
    const twice = { ...doc, models: [...doc.models, { name: 'laya-english', load: 'off' }] };
    expect(m.handle('PUT', '/v1/conductor/deciders', q(), { doc: twice, note: 'x' }, 't').status).toBe(422);
    const bad = { ...doc, models: [{ name: 'laya-english', load: 'sometimes' }] };
    expect(m.handle('PUT', '/v1/conductor/deciders', q(), { doc: bad, note: 'x' }, 't').status).toBe(422);
  });
  it('the test box answers in the Jev shape (scenario 6)', () => {
    const m = new MockConductor(7);
    const r = m.handle('POST', '/v1/conductor/deciders/test', q(), { state: 'please debug the parser', questions: { q: { type: 'choice', instructions: 'What kind?', criteria: { plan: null, debug: null, chat: null } } }, model: 'auto' }, 't');
    expect(r.status).toBe(200);
    check('systemone', r.body);
    const out = r.body as { model: string; answers: { q: { choice: string } }; routing: { reason: string } };
    expect([out.model, out.answers.q.choice, out.routing.reason]).toEqual(['laya-english', 'debug', 'English Latin text']);
    const ru = m.handle('POST', '/v1/conductor/deciders/test', q(), { state: 'Привет, помогите', questions: { q: { type: 'choice', instructions: 'What kind?', criteria: { plan: null, chat: null } } } }, 't').body as { model: string; routing: { requested: string } };
    expect([ru.model, ru.routing.requested]).toEqual(['laya-english', 'laya-multilingual']);
  });
  it('off: the status says why and no model is listed (scenario 7)', () => {
    const m = new MockConductor(7);
    m.decidersOff = true;
    const s = m.handle('GET', '/v1/conductor/deciders/status', q(), {}, 't').body as { enabled: boolean; why: string; models: unknown[] };
    check('deciders-status', s);
    expect([s.enabled, s.models.length]).toEqual([false, 0]);
    expect(s.why).toContain('QWEN38_DECIDER_MODELS is empty');
    expect(m.handle('POST', '/v1/conductor/deciders/models/laya-english/load', q(), {}, 't').status).toBe(404);
  });
});

describe('the screen’s words', () => {
  it('plans and refusals', () => {
    expect(plannerWords({ allowed: true, reason: null, after_gib: 22.8 })).toBe('fits, 22.8 GiB left after the claim');
    expect(plannerWords(null)).toBe('—');
    expect(refusalWords({ allowed: false, reason: 'refused_hold' })).toBe('refused: a hold owns the board');
    expect(refusalWords({ allowed: true, reason: null })).toBeNull();
  });
  it('the test box question', () => {
    expect(choiceQuestion('Which?', 'a, b\nc')).toEqual({ type: 'choice', instructions: 'Which?', criteria: { a: null, b: null, c: null } });
    expect(choiceQuestion('Which?', 'a')).toBe('Give at least two options.');
    expect(choiceQuestion(' ', 'a, b')).toBe('Write the question.');
    expect(choiceQuestion('Which?', 'a, a')).toBe('Two options have the same name.');
    expect(typeof choiceQuestion('Which?', Array.from({ length: 21 }, (_, i) => `o${i}`).join(','))).toBe('string');
  });
  it('answers, picks and states', () => {
    expect(probsOf({ type: 'choice', confidence: 0.6, probabilities: { a: 0.2, b: 0.8 } })).toEqual([['b', 0.8], ['a', 0.2]]);
    expect(probsOf({ type: 'noul', confidence: 0.9, noul: 0.9 })[0]).toEqual(['true', 0.9]);
    expect(picksWords({ 'laya-multilingual': 3, 'laya-english': 12 })).toBe('laya-english 12 · laya-multilingual 3');
    expect(picksWords({})).toBe('no call yet');
    expect([stateBadge('loaded'), stateBadge('failed'), stateBadge('loading'), stateBadge('not_loaded')]).toEqual(['badge-ok', 'badge-warn', 'badge-shadow', '']);
  });
  it('the memory plan shows the decision models as their own part', () => {
    const G = 1024 ** 3;
    const segs = splitSegments({ allocated: 60 * G, reserved: 61 * G, parts: { weights: 16 * G, deciders: 2 * G }, other: 42 * G, reserve: G, mem_available: 31 * G, max_len: 262144, drafter_caches: 2 } as never);
    expect(segs.find((x) => x.key === 'deciders')?.label).toBe('decision models');
  });
});
