// The mock engine's session transcripts (VIS-48): five synthetic sessions in the server's block
// format (server/conductor/transcripts.py) -- a long overnight one with hidden notes whose journal
// rows exist in the mock's journal, a compaction, checker results and a redacted secret; a short
// coding one; a kept one; a chat counted only; and a directory without meta.json -- and the
// memory routes over them, as server/conductor/routes.py `_memory` answers. Synthetic text only.

const PER_PAGE = 200;
type Row = Record<string, unknown> & { session: string | null; dir: string; keep: boolean; counts: Record<string, number> | null; last_ms: number | null };
interface Mem {
  row: Row;
  blocks: string[];
}

const at = (ms: number) => {
  const d = new Date(ms);
  const p = (n: number) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
};
const n = (x: number) => x.toLocaleString('en-US');

function block(turn: number, ms: number, o: { cls?: string; effort?: string; reasoning?: number; result?: [string, string]; user?: string; answer?: string; call?: [string, string]; hidden?: string[]; acts?: string; checker?: string; finish?: string; redactions?: string; compaction?: [number, number, string] }): string {
  const rid = `chatcmpl-${(turn * 7919).toString(16).padStart(24, '0')}`;
  if (o.compaction) {
    const [i, out, summary] = o.compaction;
    return `## turn ${turn} · ${at(ms)} · request ${rid} · compaction\n\ncompaction: ${n(i)} tokens in, ${n(out)} tokens out\n\n### summary\n${summary}\n\n### runtime\nfinish: stop · prompt ${n(i)} tokens · completion ${n(out)} tokens\n\n`;
  }
  const head = [`## turn ${turn}`, at(ms), `request ${rid}`, ...(o.cls ? [`class ${o.cls}`] : []), ...(o.effort ? [`effort ${o.effort}`] : []), ...(o.reasoning ? [`reasoning ${n(o.reasoning)} tokens`] : [])].join(' · ');
  const parts = [head];
  if (o.user) parts.push(`### user\n${o.user}`);
  if (o.result) parts.push(`### tool result (${o.result[0]}, ${n(o.result[1].length)} chars)\n${o.result[1]}`);
  if (o.answer) parts.push(`### assistant\n${o.answer}`);
  if (o.call) parts.push(`### tool call (${o.call[0]})\n${o.call[1]}`);
  const rt = [...(o.hidden ?? []), ...(o.acts ? [o.acts] : []), ...(o.checker ? [o.checker] : []), `finish: ${o.finish ?? 'tool_calls'} · prompt ${n(30000 + turn * 1200)} tokens · completion ${n(200 + turn * 7)} tokens`, ...(o.redactions ? [o.redactions] : [])];
  parts.push(`### runtime\n${rt.join('\n')}`);
  return parts.join('\n\n') + '\n\n';
}

/** The five sessions; `rowId` and `eventId` are journal rows the mock serves (the drawer). */
export function memorySeed(now: number, rowId: string | null, eventId: string | null): Map<string, Mem> {
  const out = new Map<string, Mem>();
  const night = now - 9 * 3600_000;
  const blocks: string[] = [];
  const project = '/Users/demo/demos/pagoda-seasons';
  blocks.push(block(1, night, { cls: 'three_d', effort: 'high', reasoning: 2140, user: 'Give the voxel island four seasons and run the pagoda checker after each stage.', call: ['read', '{"filePath": "js/seasons.js"}'], hidden: [`hidden: router note (effort high, package voxel-seasons), 2,410 tokens at token 24,112${rowId ? ` · journal ${rowId}` : ''}`] }));
  blocks.push(block(2, night + 40_000, { cls: 'three_d', effort: 'high', reasoning: 880, result: ['read', 'export const SEASONS = ["spring", "summer", "autumn", "winter"];\n// palette per season\n'], call: ['edit', '{"filePath": "js/seasons.js", "oldString": "spring", "newString": "spring "}'] }));
  blocks.push(block(3, night + 95_000, { cls: 'three_d', effort: 'high', reasoning: 3010, result: ['bash', 'pagodacheck summer\nRESULT: FAIL <- (a) 3.5 > 1.0'], call: ['bash', '{"command": "node tools/pagodacheck.js summer"}'], checker: 'checker: pagodacheck summer FAIL <- (a) 3.5 > 1.0', hidden: [`hidden: D1 CLOSE_REASONING, 24 tokens${eventId ? ` · journal ${eventId}` : ''}`], acts: 'acts: 1 (D1)' }));
  blocks.push(block(4, night + 3_600_000, { compaction: [120200, 4100, '## Goal\nFour seasons on the voxel island.\n## Done\nSpring and summer palettes; the pagoda checker passes spring.\n## Open\nGate (a) of summer reads 3.5 against 1.0.'] }));
  blocks.push(block(5, night + 3_700_000, { cls: 'three_d', effort: 'xhigh', reasoning: 5200, result: ['bash', 'curl -H "Authorization: Bearer [redacted:bearer]" https://api.example.test\n200 OK'], call: ['edit', '{"filePath": "js/creatures/offerings.js", "oldString": "applyOfferings(state.season)", "newString": "applyOfferings(\'night\')"}'], redactions: 'redactions: 1 (bearer 1)' }));
  blocks.push(block(6, night + 3_800_000, { cls: 'three_d', effort: 'xhigh', reasoning: 640, result: ['bash', 'pagodacheck summer\nRESULT: PASS'], answer: 'Summer now matches night 1 within the gate. The offerings were recoloured by season; they keep their night colours now.', checker: 'checker: pagodacheck summer PASS', finish: 'stop' }));
  out.set('c:ses_pagoda', { row: { session: 'c:ses_pagoda', dir: 'c-ses_pagoda', month: '', client: 'opencode', profile: 'overnight', project, first_ms: night, last_ms: night + 3_800_000, keep: false, counts: { requests: 6, blocks: 6, untranscribed: 0, compactions: 1, hidden_notes: 2, acts: 1, redactions: 1, clipped: 0, reasoning_tokens: 11870 }, checkers: { PASS: 1, FAIL: 1, last: 'pagodacheck summer PASS' }, has_meta: true }, blocks });
  const t2 = now - 2 * 3600_000;
  out.set('c:ses_minicalc', { row: { session: 'c:ses_minicalc', dir: 'c-ses_minicalc', month: '', client: 'opencode', profile: 'coding', project: '/work/minicalc', first_ms: t2, last_ms: t2 + 60_000, keep: false, counts: { requests: 2, blocks: 2, untranscribed: 0, compactions: 0, hidden_notes: 0, acts: 0, redactions: 0, clipped: 1, reasoning_tokens: 1500 }, checkers: { FAIL: 1, last: 'calccheck minicalc FAIL' }, has_meta: true },
    blocks: [block(1, t2, { cls: 'code_edit', effort: 'xhigh', reasoning: 1200, user: 'Add a power operator to the calculator and run the checks.', call: ['bash', '{"command": "python check.py"}'] }), block(2, t2 + 60_000, { cls: 'code_edit', effort: 'xhigh', reasoning: 300, result: ['bash', 'calccheck minicalc\nRESULT: FAIL <- (a) 3 of 5 cases pass'], call: ['edit', '{"filePath": "calc.py"}'], checker: 'checker: calccheck minicalc FAIL <- (a) 3 of 5 cases pass' })] });
  const t3 = now - 26 * 86400_000;
  out.set('c:ses_kept', { row: { session: 'c:ses_kept', dir: 'c-ses_kept', month: '', client: 'opencode', profile: 'coding', project: '/work/csvq', first_ms: t3, last_ms: t3 + 30_000, keep: true, counts: { requests: 1, blocks: 1, untranscribed: 0, compactions: 0, hidden_notes: 0, acts: 0, redactions: 0, clipped: 0, reasoning_tokens: 400 }, checkers: {}, has_meta: true },
    blocks: [block(1, t3, { cls: 'code_edit', effort: 'high', reasoning: 400, user: 'Explain the csv quoting rule we chose.', answer: 'Fields with a comma or a quote are quoted, and a quote inside is doubled.', finish: 'stop' })] });
  const t4 = now - 5 * 3600_000;
  out.set('c:ses_chat', { row: { session: 'c:ses_chat', dir: 'c-ses_chat', month: '', client: 'open-webui', profile: 'chat', project: null, first_ms: t4, last_ms: t4 + 120_000, keep: false, counts: { requests: 3, blocks: 0, untranscribed: 3, compactions: 0, hidden_notes: 0, acts: 0, redactions: 0, clipped: 0, reasoning_tokens: 0 }, checkers: {}, has_meta: true }, blocks: [] });
  const t5 = now - 3 * 86400_000;
  out.set('h:0f3a9c', { row: { session: null, dir: 'h-0f3a9c', month: '', client: null, profile: null, project: null, first_ms: null, last_ms: t5, keep: false, counts: null, checkers: {}, has_meta: false }, blocks: [block(1, t5, { user: 'hello', answer: 'hi', finish: 'stop' })] });
  for (const m of out.values()) {
    const d = new Date(m.row.last_ms ?? now);
    m.row.month = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}`;
    m.row.bytes = m.blocks.reduce((s, b) => s + new TextEncoder().encode(b).length, 0);
  }
  return out;
}

function parsed(text: string, turn: number) {
  const m = text.split('\n', 1)[0].match(/^## turn (\d+) · (\S+ \S+) · request (\S+)(?: · (compaction))?(?: · class (\S+))?(?: · effort (\S+))?(?: · reasoning ([\d,]+) tokens)?$/);
  return { turn: m ? Number(m[1]) : turn, time: m ? m[2] : null, request_id: m ? m[3] : null, compaction: !!m?.[4], class_id: m?.[5] ?? null, effort: m?.[6] ?? null, reasoning_tokens: m?.[7] ? Number(m[7].replace(/,/g, '')) : null, text: text.replace(/\n+$/, '') };
}

/** `/v1/conductor/memory/...` on the mock, as the server answers. */
export function memoryRoute(mem: Map<string, Mem>, method: string, rest: string[], q: URLSearchParams, body: Record<string, unknown>, settings: Record<string, unknown>): { status: number; body?: unknown; raw?: { type: string; data: string } } {
  const nf = (key: string) => ({ status: 404, body: { error: { type: 'not_found', message: `no transcript for session ${key}` } } });
  if (rest.length === 1 && rest[0] === 'sessions' && method === 'GET') {
    const rows = [...mem.values()].map((m) => ({ ...m.row })).sort((a, b) => (b.last_ms ?? 0) - (a.last_ms ?? 0));
    const blocks = [...mem.values()].reduce((s, m) => s + m.blocks.length, 0);
    return { status: 200, body: { contract_version: '1.0', sessions: rows, writer: { root: '~/.qwen38-spark-engine/memory', refused: null, queue: 0, blocks, counters: 3, dropped: 0, failed: 0, redactions: 1, pruned: 0, bytes: rows.reduce((s, r) => s + Number(r.bytes ?? 0), 0), settings } } };
  }
  if (rest[0] !== 'sessions' || rest.length < 2) return { status: 404, body: { error: { type: 'not_found', message: `no route ${method} /v1/conductor/memory/${rest.join('/')}` } } };
  const key = rest[1];
  const m = mem.get(key) ?? [...mem.values()].find((x) => x.row.dir === key);
  if (!m) return nf(key);
  if (rest.length === 2 && method === 'GET') {
    const all = m.blocks.map((b, i) => parsed(b, i + 1));
    const pages = Math.max(1, Math.ceil(all.length / PER_PAGE));
    const page = Math.min(pages, Math.max(1, Number(q.get('page') ?? pages) || pages));
    return { status: 200, body: { contract_version: '1.0', session: { ...m.row }, blocks: all.slice((page - 1) * PER_PAGE, page * PER_PAGE), page, pages, per_page: PER_PAGE, total: all.length } };
  }
  if (rest.length === 3 && rest[2] === 'keep' && (method === 'PATCH' || method === 'POST')) {
    if (typeof body.keep !== 'boolean') return { status: 400, body: { error: { type: 'bad_request', message: 'send {keep: true|false}' } } };
    m.row.keep = body.keep;
    return { status: 200, body: { contract_version: '1.0', session: key, keep: body.keep } };
  }
  if (rest.length === 3 && rest[2] === 'export' && method === 'GET') {
    if (q.get('format') === 'json') return { status: 200, body: { contract_version: '1.0', session: { ...m.row }, blocks: m.blocks.map((b, i) => parsed(b, i + 1)) } };
    return { status: 200, raw: { type: 'text/markdown; charset=utf-8', data: m.blocks.join('') } };
  }
  return { status: 404, body: { error: { type: 'not_found', message: `no route ${method} /v1/conductor/memory/${rest.join('/')}` } } };
}

/** A session with `n` blocks, for the paging tests. */
export function longSession(n: number, now: number): Mem {
  const blocks = Array.from({ length: n }, (_, i) => block(i + 1, now - (n - i) * 60_000, { cls: 'code_edit', effort: 'high', user: `step ${i + 1}`, answer: `done ${i + 1}`, finish: 'stop' }));
  return { row: { session: 'c:ses_long', dir: 'c-ses_long', month: '2026-10', client: 'opencode', profile: 'overnight', project: '/work/long', first_ms: now - n * 60_000, last_ms: now, keep: false, counts: { requests: n, blocks: n, untranscribed: 0, compactions: 0, hidden_notes: 0, acts: 0, redactions: 0, clipped: 0, reasoning_tokens: 0 }, checkers: {}, has_meta: true, bytes: 0 }, blocks };
}
