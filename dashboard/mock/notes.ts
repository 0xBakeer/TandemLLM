// The mock engine's memory notes (2026-10-05): six synthetic notes of the owner in the three scopes
// -- a pinned ratified rule, a provisional rule, a project gotcha with an earlier revision, a stale
// program_state, a session note, an archived one -- and the routes over them as
// server/conductor/routes.py `_notes` answers (list, one, edit, status, archive, delete, search,
// injected, writes, export, the user's delete). Synthetic text only.

type Note = Record<string, unknown> & { id: string; title: string; body: string; kind: string; scope: string; status: string; writer: string; tags: string[]; pinned: boolean; changelog: string[]; updated: string; created: string; project: string | null; session: string | null; verify_by: string | null; sources: string[]; revisions: number; archived?: string; reason?: string };
export interface MockNotes {
  users: Map<string, { live: Map<string, Note>; archived: Map<string, Note>; revs: Map<string, Note[]> }>;
  injected: { id: string; ts_ms: number; request_id: string; session_key: string; ids: string[]; index: boolean; user: string }[];
  writes: { id: string; ts_ms: number; task: string; decision: string; detail: Record<string, unknown>; request_id: string | null; session_key: string | null }[];
  stats: { blocks: number; taken: number; failed: number; index_notes: number; body_notes: number; intent_yes: number };
}

const KIND_WORDS: Record<string, string> = { rule: 'how I want things done', ruling: 'a decision that was taken', gotcha: 'what went wrong once', runbook: 'how to do a thing', program_state: 'where the work stands', reference: 'a fact to look up', registry: 'a list of things', archive: 'kept for the record' };
const iso = (ms: number) => new Date(ms).toISOString().replace(/\.\d+Z$/, 'Z');
const PROJECT = '/Users/demo/demos/pagoda-seasons';
const node = (n: Note) => (n.scope === 'user' ? 'user' : n.scope === 'project' ? 'projects/pagoda-seasons-3f2a9c1d' : `sessions/${String(n.session).replace(':', '-')}`);
const meta = (n: Note) => {
  const { body: _b, changelog: _c, ...m } = n;
  return m;
};

export function notesSeed(now: number): MockNotes {
  const mk = (id: string, o: Partial<Note> & { title: string; body: string; kind: string; scope: string }, agoH: number): Note => ({
    id, status: 'provisional', writer: 'qwen-inline', tags: [], pinned: false, project: o.scope === 'project' ? PROJECT : null, session: o.scope === 'session' ? 'c:ses_pagoda' : null, verify_by: null, sources: ['chatcmpl-00000000000000000000beef'], revisions: 0, confidence: null,
    created: iso(now - agoH * 3600_000), updated: iso(now - agoH * 3600_000), changelog: [`${iso(now - agoH * 3600_000).slice(0, 16).replace('T', ' ')} created by ${o.writer ?? 'qwen-inline'}`], ...o,
  } as Note);
  const live = new Map<string, Note>();
  const add = (n: Note) => live.set(n.id, n);
  add(mk('6700a1b20001', { title: 'Short answers: at most five sentences unless asked', body: 'The user wants answers of at most five sentences. Longer only when they ask for detail or a plan.', kind: 'rule', scope: 'user', status: 'ratified', writer: 'human', pinned: true, tags: ['style'] }, 50));
  add(mk('6700a1b20002', { title: 'Personal notes are written in German', body: 'Diary and personal notes go in German; code, commits and documentation stay in English.', kind: 'rule', scope: 'user', tags: ['language'] }, 20));
  const g = mk('6700a1b20003', { title: 'The summer scene differs because offerings recolour by season', body: 'Gate (a) wants the summer scene within 1.0 of night 1. With every season file disabled the run still reached 3.5: the rest comes from `applyOfferings(state.season)` in js/creatures/offerings.js.', kind: 'gotcha', scope: 'project', tags: ['pagoda', 'checker'], revisions: 1 }, 9);
  g.changelog.push(`${iso(now - 8 * 3600_000).slice(0, 16).replace('T', ' ')} revised (same fact written again) by qwen-inline`);
  add(g);
  add(mk('6700a1b20004', { title: 'Seasons: spring and summer pass, autumn next', body: 'The pagoda checker passes spring and summer; autumn is the next stage.', kind: 'program_state', scope: 'project', verify_by: iso(now - 2 * 86400_000).slice(0, 10) }, 70));
  add(mk('6700a1b20005', { title: 'Do not touch dist/ in this session', body: 'The user said dist/ is built by CI; never edit it by hand in this task.', kind: 'rule', scope: 'session', tags: ['build'] }, 3));
  add(mk('6700a1b20006', { title: 'Run the checker with node tools/pagodacheck.js <season>', body: '```\nnode tools/pagodacheck.js summer\n```\nThe RESULT line names the failed gates.', kind: 'runbook', scope: 'project', writer: 'api' }, 6));
  const archived = new Map<string, Note>();
  const a = mk('6700a1b20007', { title: 'The island uses 32 voxels a side', body: 'Was 32; the user changed it to 48.', kind: 'reference', scope: 'project' }, 100);
  a.archived = iso(now - 30 * 3600_000);
  a.reason = 'no longer true';
  archived.set(a.id, a);
  const revs = new Map<string, Note[]>([[g.id, [{ ...g, body: 'Gate (a) wants the summer scene within 1.0 of night 1; the run reached 3.5.', updated: g.created, revisions: 0 }]]]);
  const injected = [
    { id: 'j1', ts_ms: now - 2 * 3600_000, request_id: 'chatcmpl-000000000000000000000a01', session_key: 'c:ses_pagoda', ids: ['6700a1b20001', '6700a1b20003', '6700a1b20005'], index: true, user: 'owner' },
    { id: 'j2', ts_ms: now - 1 * 3600_000, request_id: 'chatcmpl-000000000000000000000a02', session_key: 'c:ses_pagoda', ids: ['6700a1b20003'], index: false, user: 'owner' },
  ];
  const writes = [
    { id: 'w1', ts_ms: now - 3 * 3600_000, task: 'write', decision: 'add', detail: { op: 'add', why: null, writer: 'qwen-inline', id: '6700a1b20005', scope: 'session', kind: 'rule' }, request_id: 'chatcmpl-000000000000000000000a00', session_key: 'c:ses_pagoda' },
    { id: 'w2', ts_ms: now - 50 * 60_000, task: 'write', decision: 'cap', detail: { op: 'add', why: 'cap', writer: 'qwen-inline' }, request_id: 'chatcmpl-000000000000000000000a03', session_key: 'c:ses_pagoda' },
    { id: 'w3', ts_ms: now - 40 * 60_000, task: 'writer', decision: 'wrote', detail: { trigger: 'compaction', mode: 'on', gate: { answer: 'yes', how: 'decider', ms: 21.4 }, ms: 11840, prefill_ms: 610, tokens: 412, record: 61234, tail: 388, existing: 2, ops: [{ op: 'add', id: '6700a1b20004', kind: 'program_state', scope: 'project' }, { op: 'add', id: '6700a1b20003', kind: 'gotcha', scope: 'project', revised: true }], refused: [] }, request_id: null, session_key: 'c:ses_pagoda' },
    { id: 'w4', ts_ms: now - 10 * 60_000, task: 'writer', decision: 'gate_no', detail: { trigger: 'idle', skipped: 'gate_no', gate: { answer: 'no', how: 'decider', ms: 19.0 }, ops: [], refused: [] }, request_id: null, session_key: 'c:ses_minicalc' },
  ];
  return { users: new Map([['owner', { live, archived, revs }]]), injected, writes, stats: { blocks: 4, taken: 3, failed: 1, index_notes: 3, body_notes: 1, intent_yes: 1 } };
}

type R = { status: number; body?: unknown; raw?: { type: string; data: string } };
const err = (status: number, type: string, message: string): R => ({ status, body: { error: { type, message } } });

// SRV-703: the shipped profiles' switches, the served model with a place for the notes
const MOCK_WHERE = {
  model: { memory: true, why: null, words: null },
  profiles: [
    { name: 'chat', memory: 'on', set: true, mode: 'on', clients: ['open-webui', 'dashboard'], default: false },
    { name: 'chat-app', memory: 'on', set: true, mode: 'on', clients: ['chat-app'], default: false },
    { name: 'coding', memory: 'on', set: true, mode: 'on', clients: ['opencode'], default: false },
    { name: 'general', memory: 'none', set: true, mode: 'on', clients: [], default: true },
    { name: 'hermes', memory: 'on', set: true, mode: 'on', clients: ['hermes'], default: false },
    { name: 'overnight', memory: 'none', set: true, mode: 'on', clients: [], default: false },
    { name: 'tools', memory: 'none', set: true, mode: 'on', clients: [], default: false },
  ],
};

export function notesRoute(m: MockNotes, method: string, rest: string[], q: URLSearchParams, body: Record<string, unknown>, settings: Record<string, unknown>): R {
  const user = q.get('user') || 'owner';
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/.test(user)) return err(400, 'bad_request', 'user: letters, digits, dot, dash, underscore');
  const u: { live: Map<string, Note>; archived: Map<string, Note>; revs: Map<string, Note[]> } = m.users.get(user) ?? { live: new Map(), archived: new Map(), revs: new Map() };
  const today = new Date().toISOString().slice(0, 10);
  const stale = (n: Note) => n.kind === 'program_state' && Boolean(n.verify_by) && String(n.verify_by) < today;
  if (rest.length === 0 && method === 'GET') {
    const notes = [...u.live.values()].map((n) => ({ ...meta(n), node: node(n), stale: stale(n), uses_7d: m.injected.filter((r) => r.ids.includes(n.id)).length })).sort((a, b) => String(b.updated).localeCompare(String(a.updated)));
    const tree = new Map<string, { node: string; scope: string; label: string | null; count: number; provisional: number; stale: number }>();
    for (const n of u.live.values()) {
      const k = node(n);
      const t = tree.get(k) ?? { node: k, scope: n.scope, label: n.scope === 'user' ? 'user' : n.scope === 'project' ? n.project : n.session, count: 0, provisional: 0, stale: 0 };
      t.count++;
      t.provisional += n.status === 'provisional' ? 1 : 0;
      t.stale += stale(n) ? 1 : 0;
      tree.set(k, t);
    }
    const order = ['user', 'project', 'session'];
    return { status: 200, body: { contract_version: '1.0', user, users: [...new Set([...m.users.keys(), user])], tree: [...tree.values()].sort((a, b) => order.indexOf(a.scope) - order.indexOf(b.scope) || a.node.localeCompare(b.node)), notes, archived: [...u.archived.values()].map(meta), duplicates: [], kinds: KIND_WORDS, store: { root: '~/.qwen38-spark-engine/memory/notes', refused: null, users: [...m.users.keys()], writes: 5, refused_writes: 1, revised: 1, archived: 1, redactions: 0, searches: 12 }, memory: m.stats, settings, where: MOCK_WHERE } };
  }
  if (rest[0] === 'search' && method === 'GET') {
    const words = (q.get('q') ?? '').toLowerCase().split(/\W+/).filter((w) => w.length > 2);
    if (!words.length) return err(400, 'bad_request', 'q: the words to search for');
    const hits = [...u.live.values()].map((n) => {
      const text = `${n.title} ${n.title} ${n.title} ${n.tags.join(' ')} ${n.body}`.toLowerCase();
      const score = words.reduce((s, w) => s + (text.split(w).length - 1) * 0.7, 0);
      return { ...meta(n), node: node(n), score: Math.round(score * 1000) / 1000 };
    }).filter((h) => h.score > 0).sort((a, b) => b.score - a.score).slice(0, Number(q.get('k') ?? 10));
    return { status: 200, body: { contract_version: '1.0', q_words: words.length, hits } };
  }
  if (rest[0] === 'injected' && method === 'GET') return { status: 200, body: { contract_version: '1.0', rows: m.injected.filter((r) => r.user === user).slice().reverse() } };
  if (rest[0] === 'writes' && method === 'GET') return { status: 200, body: { contract_version: '1.0', rows: m.writes.slice().reverse() } };
  if (rest[0] === 'export' && method === 'GET') {
    if (!m.users.has(user)) return err(404, 'not_found', `no memory for ${user}`);
    return { status: 200, raw: { type: 'application/zip', data: 'PK\u0003\u0004mock' } };
  }
  if (rest[0] === 'users' && rest.length === 2 && method === 'DELETE') {
    if (body.confirm !== rest[1]) return err(400, 'bad_request', "send {confirm: <the user's name>} to delete every note of the user");
    if (!m.users.delete(rest[1])) return err(404, 'not_found', `no memory for ${rest[1]}`);
    return { status: 200, body: { contract_version: '1.0', deleted: rest[1] } };
  }
  const id = rest[0];
  const n = u.live.get(id);
  const now = iso(Date.now());
  const log = (what: string) => n && n.changelog.push(`${now.slice(0, 16).replace('T', ' ')} ${what}`);
  if (rest.length === 1 && method === 'GET') {
    const x = n ?? u.archived.get(id);
    if (!x) return err(404, 'not_found', `no note ${id}`);
    const uses: Record<string, number> = {};
    for (const r of m.injected) if (r.ids.includes(id)) uses[new Date(r.ts_ms).toISOString().slice(0, 10)] = (uses[new Date(r.ts_ms).toISOString().slice(0, 10)] ?? 0) + 1;
    return { status: 200, body: { contract_version: '1.0', note: { ...x, node: node(x), stale: stale(x) }, revisions: (u.revs.get(id) ?? []).map((r, i) => ({ n: i + 1, title: r.title, body: r.body, kind: r.kind, status: r.status, writer: r.writer, updated: r.updated })), uses } };
  }
  if (rest.length === 1 && method === 'DELETE') {
    const gone = u.live.delete(id) || u.archived.delete(id);
    if (!gone) return err(404, 'not_found', `no note ${id}`);
    return { status: 200, body: { contract_version: '1.0', deleted: id } };
  }
  if (!n) return err(404, 'unknown_id', `no note ${id}`);
  if (rest.length === 1 && method === 'PUT') {
    const title = String(body.title ?? n.title).trim();
    const text = String(body.body ?? n.body).trim();
    if (title.length > 120 || !text || text.split(/\s+/).length > 400) return err(422, 'cap', 'a title of at most 120 characters, a body of 1 to 400 words');
    u.revs.set(id, [...(u.revs.get(id) ?? []), { ...n }]);
    Object.assign(n, { title, body: text, kind: String(body.kind ?? n.kind), tags: Array.isArray(body.tags) ? body.tags.map(String) : n.tags, verify_by: 'verify_by' in body ? (body.verify_by as string | null) : n.verify_by, status: 'ratified', writer: 'human', updated: now, revisions: n.revisions + 1 });
    log('edited by human');
    m.writes.push({ id: `w${m.writes.length + 1}`, ts_ms: Date.now(), task: 'human', decision: 'edit', detail: { id, user }, request_id: null, session_key: null });
    return { status: 200, body: { contract_version: '1.0', note: n } };
  }
  if (rest.length === 1 && method === 'PATCH') {
    if (body.status !== undefined) {
      if (body.status !== 'ratified' && body.status !== 'provisional') return err(422, 'schema', 'status: one of provisional, ratified');
      n.status = body.status;
      log(`${body.status} by human`);
    }
    if (body.pinned !== undefined) {
      if (body.pinned && !(n.status === 'ratified' && (n.kind === 'rule' || n.kind === 'ruling'))) return err(422, 'schema', 'only a ratified rule or ruling can be pinned');
      n.pinned = Boolean(body.pinned);
      log(`${n.pinned ? 'pinned' : 'unpinned'} by human`);
    }
    n.updated = now;
    return { status: 200, body: { contract_version: '1.0', note: n } };
  }
  if (rest.length === 2 && rest[1] === 'archive' && method === 'POST') {
    u.live.delete(id);
    n.archived = now;
    n.reason = String(body.reason ?? 'archived in the dashboard');
    log('archived by human');
    u.archived.set(id, n);
    return { status: 200, body: { contract_version: '1.0', id, archived: n.archived, reason: n.reason } };
  }
  return err(404, 'not_found', `no route ${method} /v1/conductor/memory/notes/${rest.join('/')}`);
}

// ---------------------------------------------------------------- P-M3: the package proposals
export interface MockProposal {
  n: number; name: string; kind: string; mode: 'replace' | 'append'; title: string; reason: string; status: 'open' | 'applied' | 'rejected'; writer: string; created: string; updated: string; session: string; request_id: string; base_version: number; base_text: string; evidence: string[]; body: string; applied_version?: number; rejected_reason?: string;
}

/** Two open proposals for the seed's inline `ui-design` (an appended Sources section with evidence,
 *  a replacement without), and a rejected one. */
export function proposalsSeed(now: number, liveText: string, version: number): MockProposal[] {
  const t = (h: number) => iso(now - h * 3600_000);
  const base = { kind: 'package', writer: 'qwen-inline', session: 'c:ses_pagoda', request_id: 'chatcmpl-000000000000000000000b01', base_version: version, base_text: liveText };
  return [
    { ...base, n: 1, name: 'ui-design', mode: 'append', title: 'a Sources section and what to do without search', reason: 'the last design session searched for palette references the package does not name', status: 'open', created: t(2), updated: t(2), evidence: ['oc1', 'oc2'], body: '## Sources\n- Search "material design color system" and "refactoring ui spacing" for references.\n- No search allowed: use the palette and spacing scale above and say that no reference was checked.' },
    { ...base, n: 2, name: 'ui-design', mode: 'replace', title: 'a shorter package', reason: 'the package costs 400 tokens on every design turn', status: 'open', created: t(1), updated: t(1), evidence: [], body: '# UI design\n\nBuild interfaces that look designed.\n\n1. One spacing scale.\n2. One accent colour.' },
    { ...base, n: 3, name: 'ui-design', mode: 'append', title: 'a dark mode rule', reason: 'one session asked for dark mode', status: 'rejected', rejected_reason: 'already covered', created: t(30), updated: t(29), evidence: [], body: '## Dark mode\nDefine tokens for both themes.' },
  ];
}

export function proposalsRoute(list: MockProposal[], method: string, rest: string[], body: Record<string, unknown>, live: () => { text: string | null; version: number }, apply: (name: string, text: string, note: string) => { status: number; body?: unknown }): { status: number; body?: unknown } {
  const meta = (p: MockProposal) => {
    const { body: _b, base_text: _t, ...m } = p;
    return m;
  };
  if (rest.length === 0 && method === 'GET') return { status: 200, body: { contract_version: '1.0', proposals: list.map(meta) } };
  const p = list.find((x) => x.name === rest[0] && x.n === Number(rest[1]));
  if (!p) return err(404, 'not_found', `no proposal ${rest[0]}#${rest[1]}`);
  const now = live();
  const stale = now.text !== null && now.text !== p.base_text;
  const proposed = p.mode === 'replace' ? p.body : `${(now.text ?? p.base_text).trimEnd()}\n\n${p.body}`;
  if (rest.length === 2 && method === 'GET') {
    return { status: 200, body: { contract_version: '1.0', proposal: { ...meta(p), proposed, body: p.body, live: now.text, stale, applies: p.status === 'open' && now.text !== null && (!stale || p.mode === 'append'), evidence_word: p.evidence.length ? `${p.evidence.length} outcome row${p.evidence.length === 1 ? '' : 's'}` : 'no evidence' } } };
  }
  if (rest[2] === 'apply' && method === 'POST') {
    if (p.status !== 'open') return err(409, 'closed', `proposal ${p.name}#${p.n} is ${p.status}`);
    if (stale && p.mode !== 'append') return err(409, 'stale', `${p.name} changed since the proposal; reject it or let the model propose again`);
    const r = apply(p.name, proposed, `proposal ${p.name}#${p.n}: ${p.title}`);
    if (r.status !== 200) return r;
    const version = (r.body as { version: number }).version;
    Object.assign(p, { status: 'applied', applied_version: version, updated: iso(Date.now()) });
    return { status: 200, body: { contract_version: '1.0', name: p.name, n: p.n, version, previous: now.version } };
  }
  if (rest[2] === 'reject' && method === 'POST') {
    if (p.status !== 'open') return err(409, 'closed', `proposal ${p.name}#${p.n} is ${p.status}`);
    Object.assign(p, { status: 'rejected', rejected_reason: String(body.reason ?? 'rejected in the dashboard'), updated: iso(Date.now()) });
    return { status: 200, body: { contract_version: '1.0', name: p.name, n: p.n, status: 'rejected' } };
  }
  return err(404, 'not_found', `no route ${method} /v1/conductor/memory/proposals/${rest.join('/')}`);
}
