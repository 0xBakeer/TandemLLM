// The Memory screen's Notes tab (2026-10-05): the list's filters, the tree's labels, the line diff of
// a note's history, the write rows in words, the settings over their defaults, and the mock engine's
// notes routes (the server's shapes: server/conductor/routes.py `_notes`).
import { describe, expect, it } from 'vitest';
import { notesRoute, notesSeed } from '../mock/notes.ts';
import type { NoteMeta, NoteWriteRow } from '../src/api/conductor';
import { filterNotes, lineDiff, memoryWhere, NO_FILTER, nodeLabel, NOTES_DEFAULTS, notesSettings, usesLine, writeLine } from '../src/lib/notes';

const note = (o: Partial<NoteMeta>): NoteMeta => ({ id: '0123456789ab', title: 't', kind: 'gotcha', scope: 'user', project: null, session: null, status: 'provisional', writer: 'qwen-inline', confidence: null, tags: [], created: '2026-10-05T01:00:00Z', updated: '2026-10-05T01:00:00Z', verify_by: null, pinned: false, sources: [], revisions: 0, node: 'user', ...o });

describe('the list', () => {
  const ns = [note({ id: 'a', title: 'Short answers', kind: 'rule', status: 'ratified', writer: 'human', tags: ['style'] }), note({ id: 'b', title: 'Summer scene gate', node: 'projects/pagoda-1', scope: 'project' }), note({ id: 'c', title: 'Seasons state', kind: 'program_state', stale: true, node: 'projects/pagoda-1', scope: 'project' })];
  it('filters by node, kind, status, writer, stale and words', () => {
    expect(filterNotes(ns, NO_FILTER).map((n) => n.id)).toEqual(['a', 'b', 'c']);
    expect(filterNotes(ns, { ...NO_FILTER, node: 'projects/pagoda-1' }).map((n) => n.id)).toEqual(['b', 'c']);
    expect(filterNotes(ns, { ...NO_FILTER, kind: 'rule' }).map((n) => n.id)).toEqual(['a']);
    expect(filterNotes(ns, { ...NO_FILTER, status: 'provisional' }).map((n) => n.id)).toEqual(['b', 'c']);
    expect(filterNotes(ns, { ...NO_FILTER, writer: 'human' }).map((n) => n.id)).toEqual(['a']);
    expect(filterNotes(ns, { ...NO_FILTER, stale: true }).map((n) => n.id)).toEqual(['c']);
    expect(filterNotes(ns, { ...NO_FILTER, q: 'STYLE' }).map((n) => n.id)).toEqual(['a']);
  });
  it('names the tree nodes in words', () => {
    expect(nodeLabel({ node: 'user', scope: 'user', label: 'user' })).toBe('About you');
    expect(nodeLabel({ node: 'projects/x-1', scope: 'project', label: '/work/x' })).toBe('/work/x');
    expect(nodeLabel({ node: 'sessions/c-ses_1', scope: 'session', label: 'c:ses_1' })).toBe('session c:ses_1');
  });
});

describe('the history', () => {
  it('diffs two texts by lines, unchanged lines kept', () => {
    expect(lineDiff('a\nb\nc', 'a\nB\nc\nd')).toEqual([{ op: '=', text: 'a' }, { op: '-', text: 'b' }, { op: '+', text: 'B' }, { op: '=', text: 'c' }, { op: '+', text: 'd' }]);
    expect(lineDiff('same', 'same')).toEqual([{ op: '=', text: 'same' }]);
  });
  it('says the uses in words', () => {
    expect(usesLine({}, '2026-10-05')).toBe('not injected yet');
    expect(usesLine({ '2026-10-04': 2, '2026-10-05': 1 }, '2026-10-05')).toBe('3 injections (1 today)');
  });
});

describe('the writes', () => {
  const row = (o: Partial<NoteWriteRow>): NoteWriteRow => ({ id: 'w', ts_ms: 0, task: 'write', decision: 'add', detail: {}, request_id: null, session_key: null, ...o });
  it('says what was written and what was refused', () => {
    expect(writeLine(row({ detail: { op: 'add', writer: 'qwen-inline', kind: 'gotcha', scope: 'project', id: '0a1b' } }))).toBe('added gotcha (project) by the model, mid-answer · 0a1b');
    expect(writeLine(row({ detail: { op: 'add', revised: true, kind: 'rule', scope: 'user', id: '0a1b' } }))).toBe('revised rule (user) by the model, mid-answer · 0a1b');
    expect(writeLine(row({ decision: 'cap', detail: { op: 'add', why: 'cap', writer: 'qwen-inline' } }))).toBe('refused (cap): add by the model, mid-answer');
    expect(writeLine(row({ task: 'nightly', decision: 'done', detail: { deleted: 2, duplicates: 1, stale: 0 } }))).toBe('nightly: 2 archives deleted, 1 duplicates, 0 stale');
  });
  it('reads the settings over the defaults', () => {
    expect(notesSettings(null)).toEqual(NOTES_DEFAULTS);
    expect(notesSettings({ memory: { notes: { intent: 'on' } } })).toEqual({ ...NOTES_DEFAULTS, intent: 'on' });
  });
});

describe('the mock routes', () => {
  const q = (s = '') => new URLSearchParams(s);
  it('lists the tree, the notes and the archive; one note with its revision', () => {
    const m = notesSeed(Date.now());
    const r = notesRoute(m, 'GET', [], q(), {}, NOTES_DEFAULTS as unknown as Record<string, unknown>);
    const b = r.body as { notes: NoteMeta[]; tree: { scope: string; count: number }[]; archived: unknown[] };
    expect(b.notes).toHaveLength(6);
    expect(b.tree.map((t) => t.scope)).toEqual(['user', 'project', 'session']);
    expect(b.archived).toHaveLength(1);
    expect(b.notes.some((n) => n.stale)).toBe(true);
    const one = notesRoute(m, 'GET', ['6700a1b20003'], q(), {}, {}).body as { revisions: unknown[]; note: { body: string } };
    expect(one.revisions).toHaveLength(1);
    expect(one.note.body).toContain('offerings.js');
  });
  it('edits as a ratified revision, pins only a ratified rule, archives and deletes', () => {
    const m = notesSeed(Date.now());
    const ed = notesRoute(m, 'PUT', ['6700a1b20002'], q(), { body: 'Personal notes in German.' }, {}).body as { note: { status: string; writer: string } };
    expect(ed.note).toMatchObject({ status: 'ratified', writer: 'human' });
    expect(notesRoute(m, 'PATCH', ['6700a1b20003'], q(), { pinned: true }, {}).status).toBe(422);
    expect(notesRoute(m, 'PATCH', ['6700a1b20002'], q(), { pinned: true }, {}).status).toBe(200);
    expect(notesRoute(m, 'POST', ['6700a1b20005', 'archive'], q(), { reason: 'done' }, {}).status).toBe(200);
    expect(notesRoute(m, 'DELETE', ['6700a1b20005'], q(), {}, {}).status).toBe(200);
    expect(notesRoute(m, 'DELETE', ['6700a1b20005'], q(), {}, {}).status).toBe(404);
    expect(notesRoute(m, 'DELETE', ['users', 'owner'], q(), {}, {}).status).toBe(400);
    expect(notesRoute(m, 'DELETE', ['users', 'owner'], q(), { confirm: 'owner' }, {}).status).toBe(200);
  });
  it('searches with scores', () => {
    const m = notesSeed(Date.now());
    const s = notesRoute(m, 'GET', ['search'], q('q=summer+scene'), {}, {}).body as { hits: { id: string; score: number }[] };
    expect(s.hits[0].id).toBe('6700a1b20003');
    expect(s.hits[0].score).toBeGreaterThan(0);
  });
});

describe('the writer runs', () => {
  it('reads a run and a skipped run from their journal rows', async () => {
    const { writerRun } = await import('../src/lib/notes');
    const r = writerRun({ id: 'w', ts_ms: 1, task: 'writer', decision: 'wrote', detail: { trigger: 'compaction', gate: { answer: 'yes' }, ms: 11840, tokens: 412, ops: [{ op: 'add' }, { op: 'add', revised: true }, { op: 'update' }], refused: ['cap'] }, request_id: null, session_key: null });
    expect(r).toMatchObject({ trigger: 'compaction', gate: 'yes', outcome: 'wrote', seconds: 11.8, tokens: 412, ops: '1 added, 1 revised, 1 updated', refused: ['cap'] });
    const s = writerRun({ id: 'w', ts_ms: 1, task: 'writer', decision: 'gate_no', detail: { trigger: 'idle', skipped: 'gate_no', gate: { answer: 'no' }, ops: [] }, request_id: null, session_key: null });
    expect(s).toMatchObject({ outcome: 'gate_no', gate: 'no', ops: '—' });
    const sh = writerRun({ id: 'w', ts_ms: 1, task: 'writer', decision: 'wrote', detail: { trigger: 'idle', ops: [{ op: 'add', stored: false }] }, request_id: null, session_key: null });
    expect(sh.ops).toBe('1 added (shadow: not stored)');
  });
});

describe('the package proposals (mock routes)', async () => {
  const { proposalsRoute, proposalsSeed } = await import('../mock/notes.ts');
  it('shows the diff data, applies through a new version, refuses a stale replacement, rejects', () => {
    let text: string | null = '# UI design\n\nOld rules.';
    let version = 4;
    const live = () => ({ text, version });
    const apply = (_name: string, t: string) => {
      text = t;
      version += 1;
      return { status: 200, body: { version } };
    };
    const list = proposalsSeed(Date.now(), text!, version);
    const one = proposalsRoute(list, 'GET', ['ui-design', '1'], {}, live, apply).body as { proposal: { applies: boolean; proposed: string; evidence_word: string } };
    expect(one.proposal.applies).toBe(true);
    expect(one.proposal.proposed.startsWith('# UI design\n\nOld rules.\n\n## Sources')).toBe(true);
    expect(one.proposal.evidence_word).toBe('2 outcome rows');
    const r = proposalsRoute(list, 'POST', ['ui-design', '1', 'apply'], {}, live, apply);
    expect(r.body).toMatchObject({ version: 5, previous: 4 });
    expect(proposalsRoute(list, 'POST', ['ui-design', '1', 'apply'], {}, live, apply).status).toBe(409);
    // the text changed: the replacement (#2) is stale
    expect((proposalsRoute(list, 'POST', ['ui-design', '2', 'apply'], {}, live, apply).body as { error: { type: string } }).error.type).toBe('stale');
    expect(proposalsRoute(list, 'POST', ['ui-design', '2', 'reject'], { reason: 'no' }, live, apply).status).toBe(200);
    expect(list.find((p) => p.n === 2)!.status).toBe('rejected');
  });
});

// SRV-703 (QA SRV-704): the Notes tab says where memory is on, the served model first
describe('where memory is on', () => {
  const profiles = [
    { name: 'chat', memory: 'on' as const, set: true, mode: 'on', clients: ['open-webui', 'dashboard'], default: false },
    { name: 'general', memory: 'none' as const, set: true, mode: 'on', clients: [], default: true },
    { name: 'chat-app', memory: 'off' as const, set: true, mode: 'on', clients: ['chat-app'], default: false },
    { name: 'bench-pk', memory: 'none' as const, set: false, mode: 'on', clients: [], default: false },
    { name: 'raw', memory: 'none' as const, set: true, mode: 'raw', clients: [], default: false },
  ];
  it('names each profile with memory off, and the clients it serves', () => {
    const w = memoryWhere({ model: { memory: true, why: null, words: null }, profiles })!;
    expect(w.model).toBeNull();
    expect(w.lines.map((l) => [l.name, l.on, l.state, l.who])).toEqual([
      ['chat', true, 'on', 'open-webui, dashboard'],
      ['bench-pk', false, 'off (not set)', ''],
      ['chat-app', false, 'off, and the model is told', 'chat-app'],
      ['general', false, 'off', 'any other client'],
    ]);
  });
  it('says when the served model reads no note, on every profile', () => {
    const w = memoryWhere({ model: { memory: false, why: 'model_notes', words: "the model's prompt has no place for the conductor's notes" }, profiles })!;
    expect(w.model).toBe("Memory is off for the served model: the model's prompt has no place for the conductor's notes. No profile reads or writes notes while it serves.");
    expect(w.lines.find((l) => l.name === 'chat')).toEqual({ name: 'chat', on: false, state: 'on in the profile, off for this model', who: 'open-webui, dashboard' });
    expect(w.lines.every((l) => !l.on)).toBe(true);
  });
  it('shows nothing for an engine before it', () => {
    expect(memoryWhere(undefined)).toBeNull();
  });
  it('the mock engine answers with the shipped switches', () => {
    const r = notesRoute(notesSeed(Date.now()), 'GET', [], new URLSearchParams(), {}, {});
    const where = (r.body as { where: Parameters<typeof memoryWhere>[0] }).where;
    const w = memoryWhere(where)!;
    expect(w.lines.filter((l) => l.on).map((l) => l.name)).toEqual(['chat', 'chat-app', 'coding', 'hermes']);
  });
});
