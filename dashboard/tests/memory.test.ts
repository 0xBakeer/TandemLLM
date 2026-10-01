// VIS-49's vitest rows for the Memory screen (VIS-48): the block renderer over the server's own
// golden blocks (tests/fixtures/transcripts/*.md, written by server/conductor/transcripts.py), the
// runtime lines with the hidden notes and their journal rows, a session's one-line summary, the
// transcript settings over their defaults, and the mock engine's memory routes against
// docs/contract/conductor-v1 (the list, a page of 200, keep, the exports).
import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import Ajv2020 from 'ajv/dist/2020.js';
import { MockConductor } from '../mock/conductor.ts';
import { longSession } from '../mock/memory.ts';
import { parseBlock, runtimeLines, sessionLine, shortKey, TRANSCRIPT_DEFAULTS, transcriptSettings } from '../src/lib/memory';

const DIR = join(__dirname, '..', '..', 'docs', 'contract', 'conductor-v1');
const GOLDEN = join(__dirname, '..', '..', 'tests', 'fixtures', 'transcripts');
const ajv = new Ajv2020({ allErrors: true, strict: true });
for (const f of readdirSync(DIR).filter((f) => f.endsWith('.schema.json'))) ajv.addSchema(JSON.parse(readFileSync(join(DIR, f), 'utf8')), f);
function check(name: string, data: unknown) {
  const v = ajv.getSchema(`${name}.schema.json`);
  expect(v, `schema ${name}`).toBeDefined();
  if (!v!(data)) throw new Error(`${name}: ${ajv.errorsText(v!.errors, { separator: '\n' })}`);
}
const q = (s = '') => new URLSearchParams(s);

describe('the block renderer', () => {
  it('cuts the server’s golden block into its sections and runtime lines', () => {
    const p = parseBlock(readFileSync(join(GOLDEN, 'block-tools.md'), 'utf8'));
    expect(p.head).toMatch(/^## turn 1 · 2026-09-21 16:13:20 · request chatcmpl-0123456789abcdef01234567 · class code_edit · effort xhigh/);
    expect(p.sections.map((s) => [s.kind, s.name, s.clipped])).toEqual([
      ['tool_result', 'bash', true],
      ['assistant', null, false],
      ['tool_call', 'edit', false],
      ['reasoning', null, false],
    ]);
    expect(p.sections[0].body.startsWith('calccheck minicalc\nRESULT: FAIL')).toBe(true);
    expect(p.sections[1].body).toBe('I will add the operator.');
    const hidden = p.runtime.filter((r) => r.kind === 'hidden');
    expect(hidden.map((r) => r.journal)).toEqual(['9f3c2a1b0c4d5e6f', 'ab12cd34ef56ab12']);
    expect(p.runtime.map((r) => r.kind)).toEqual(['hidden', 'hidden', 'replayed', 'acts', 'checker', 'finish']);
    expect(p.runtime.find((r) => r.kind === 'checker')!.verdict).toBe('FAIL');
  });
  it('a compaction block keeps its line and its summary, with the summary’s own headings inside it', () => {
    const p = parseBlock(readFileSync(join(GOLDEN, 'block-compaction.md'), 'utf8'));
    expect(p.compaction).toBe('compaction: 120,200 tokens in, 4,100 tokens out');
    expect(p.sections.map((s) => s.kind)).toEqual(['summary']);
    expect(p.sections[0].body).toBe('## Goal\nA calculator with a power operator.\n## Done\nParser, tests.');
  });
  it('an escaped heading inside a message stays in the message', () => {
    const p = parseBlock('## turn 2 · 2026-10-01 02:00:00 · request r\n\n### assistant\nSee:\n\\### user\nquoted\n\n### runtime\nfinish: stop · prompt 1 tokens · completion 1 tokens');
    expect(p.sections.map((s) => s.kind)).toEqual(['assistant']);
    expect(p.sections[0].body).toBe('See:\n\\### user\nquoted');
  });
  it('reads the runtime lines', () => {
    const r = runtimeLines('hidden: D1 CLOSE_REASONING, 24 tokens · journal ab12cd34ef56ab12\nchecker: pagodacheck summer PASS\nredactions: 2 (bearer 1, jwt 1)\nsomething new');
    expect(r.map((x) => [x.kind, x.journal, x.verdict])).toEqual([
      ['hidden', 'ab12cd34ef56ab12', null],
      ['checker', null, 'PASS'],
      ['redactions', null, null],
      ['other', null, null],
    ]);
  });
  it('says a session in one line', () => {
    const counts = { requests: 9, blocks: 6, untranscribed: 3, compactions: 1, hidden_notes: 2, acts: 1, redactions: 1, clipped: 0, reasoning_tokens: 9 };
    expect(sessionLine({ counts, checkers: { last: 'pagodacheck summer PASS' } })).toBe('6 turns · 3 without tools (counted only) · 1 act · 1 compaction · 1 redacted · pagodacheck summer PASS');
    expect(sessionLine({ counts: null })).toBe('counts unknown (no meta.json)');
    expect(shortKey('c:ses_ab', 'c-ses_ab')).toBe('ses_ab');
    expect(shortKey(null, 'h-0f3a')).toBe('h-0f3a');
  });
  it('the transcript settings fall back to their defaults', () => {
    expect(transcriptSettings({})).toEqual(TRANSCRIPT_DEFAULTS);
    expect(transcriptSettings({ memory: { transcripts: { scope: 'all' } } })).toEqual({ ...TRANSCRIPT_DEFAULTS, scope: 'all' });
    const seeds = JSON.parse(readFileSync(join(DIR, 'seeds.json'), 'utf8'));
    expect(seeds.settings.memory.transcripts).toEqual(TRANSCRIPT_DEFAULTS);
  });
});

describe('the memory routes on the mock engine', () => {
  it('lists five sessions with their counts (scenario 1)', () => {
    const m = new MockConductor(7);
    const r = m.handle('GET', '/v1/conductor/memory/sessions', q(), {}, 't').body as { sessions: { session: string | null; counts: unknown; keep: boolean }[] };
    check('memory-sessions', r);
    expect(r.sessions).toHaveLength(5);
    expect(r.sessions[0].session).toBe('c:ses_minicalc');
    expect(r.sessions.find((s) => s.session === null)!.counts).toBeNull();
  });
  it('a transcript reads with its blocks; its hidden notes name rows of the journal (scenario 2)', () => {
    const m = new MockConductor(7);
    const r = m.handle('GET', '/v1/conductor/memory/sessions/c%3Ases_pagoda', q(), {}, 't').body as { blocks: { text: string; compaction: boolean }[] };
    check('memory-session', r);
    expect(r.blocks.map((b) => b.compaction)).toEqual([false, false, false, true, false, false]);
    const ids = r.blocks.flatMap((b) => parseBlock(b.text).runtime.filter((x) => x.journal).map((x) => x.journal));
    expect(ids).toHaveLength(2);
    expect(m.handle('GET', `/v1/conductor/decisions/${ids[0]}`, q(), {}, 't').status).toBe(200);
    expect(m.handle('GET', `/v1/conductor/events/${ids[1]}`, q(), {}, 't').status).toBe(200);
  });
  it('pages 3,000 blocks at 200, the newest page first', () => {
    const m = new MockConductor(7);
    m.memory.set('c:ses_long', longSession(3000, Date.now()));
    const last = m.handle('GET', '/v1/conductor/memory/sessions/c%3Ases_long', q(), {}, 't').body as { page: number; pages: number; blocks: { turn: number }[] };
    check('memory-session', last);
    expect([last.page, last.pages, last.blocks.length, last.blocks.at(-1)!.turn]).toEqual([15, 15, 200, 3000]);
    const first = m.handle('GET', '/v1/conductor/memory/sessions/c%3Ases_long', q('page=1'), {}, 't').body as { blocks: { turn: number }[] };
    expect(first.blocks[0].turn).toBe(1);
  });
  it('keep is written and read back (scenario 3); the exports answer', () => {
    const m = new MockConductor(7);
    const k = m.handle('PATCH', '/v1/conductor/memory/sessions/c%3Ases_minicalc/keep', q(), { keep: true }, 't');
    expect(k.status).toBe(200);
    check('memory-keep', k.body);
    const r = m.handle('GET', '/v1/conductor/memory/sessions', q(), {}, 't').body as { sessions: { session: string; keep: boolean }[] };
    expect(r.sessions.find((s) => s.session === 'c:ses_minicalc')!.keep).toBe(true);
    expect(m.handle('PATCH', '/v1/conductor/memory/sessions/c%3Ases_minicalc/keep', q(), { keep: 'yes' }, 't').status).toBe(400);
    const md = m.handle('GET', '/v1/conductor/memory/sessions/c%3Ases_minicalc/export', q('format=md'), {}, 't');
    expect(md.raw!.data.startsWith('## turn 1 · ')).toBe(true);
    check('memory-session', m.handle('GET', '/v1/conductor/memory/sessions/c%3Ases_minicalc/export', q('format=json'), {}, 't').body);
    expect(m.handle('GET', '/v1/conductor/memory/sessions/c%3Anobody', q(), {}, 't').status).toBe(404);
  });
});
