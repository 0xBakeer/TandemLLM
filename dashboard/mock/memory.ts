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

function block(turn: number, ms: number, o: { cls?: string; effort?: string; reasoning?: number; result?: [string, string]; user?: string; answer?: string; call?: [string, string]; hidden?: string[]; acts?: string; checker?: string; snapshot?: string; finish?: string; redactions?: string; compaction?: [number, number, string] }): string {
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
  const rt = [...(o.hidden ?? []), ...(o.acts ? [o.acts] : []), ...(o.checker ? [o.checker] : []), ...(o.snapshot ? [o.snapshot] : []), `finish: ${o.finish ?? 'tool_calls'} · prompt ${n(30000 + turn * 1200)} tokens · completion ${n(200 + turn * 7)} tokens`, ...(o.redactions ? [o.redactions] : [])];
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
  blocks.push(block(3, night + 95_000, { cls: 'three_d', effort: 'high', reasoning: 3010, result: ['bash', 'pagodacheck summer\nRESULT: FAIL <- (a) 3.5 > 1.0'], call: ['bash', '{"command": "node tools/pagodacheck.js summer"}'], checker: 'checker: pagodacheck summer FAIL <- (a) 3.5 > 1.0', snapshot: 'snapshot: snapshots/turn-3.json · outcome low_quality', hidden: [`hidden: D1 CLOSE_REASONING, 24 tokens${eventId ? ` · journal ${eventId}` : ''}`], acts: 'acts: 1 (D1)' }));
  blocks.push(block(4, night + 3_600_000, { compaction: [120200, 4100, '## Goal\nFour seasons on the voxel island.\n## Done\nSpring and summer palettes; the pagoda checker passes spring.\n## Open\nGate (a) of summer reads 3.5 against 1.0.'] }));
  blocks.push(block(5, night + 3_700_000, { cls: 'three_d', effort: 'xhigh', reasoning: 5200, result: ['bash', 'curl -H "Authorization: Bearer [redacted:bearer]" https://api.example.test\n200 OK'], call: ['edit', '{"filePath": "js/creatures/offerings.js", "oldString": "applyOfferings(state.season)", "newString": "applyOfferings(\'night\')"}'], redactions: 'redactions: 1 (bearer 1)' }));
  blocks.push(block(6, night + 3_800_000, { cls: 'three_d', effort: 'xhigh', reasoning: 640, result: ['bash', 'pagodacheck summer\nRESULT: PASS'], answer: 'Summer now matches night 1 within the gate. The offerings were recoloured by season; they keep their night colours now.', checker: 'checker: pagodacheck summer PASS', finish: 'stop' }));
  out.set('c:ses_pagoda', { row: { session: 'c:ses_pagoda', dir: 'c-ses_pagoda', month: '', client: 'opencode', profile: 'overnight', project, first_ms: night, last_ms: night + 3_800_000, keep: false, counts: { requests: 6, blocks: 6, untranscribed: 0, compactions: 1, hidden_notes: 2, acts: 1, redactions: 1, clipped: 0, reasoning_tokens: 11870, snapshots: 5 }, checkers: { PASS: 1, FAIL: 1, last: 'pagodacheck summer PASS' }, has_meta: true }, blocks });
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

// ---------------------------------------------------------------- the disk tier (C1)
/** A running disk tier (server/diskcache.py `view`) with six entries -- two opencode heads with their
 *  previews, a package, a session head, a memory file and a repeated prefix without one -- and its
 *  last events. Synthetic numbers on the 27B's layout: 106,496 bytes a token and a 147 MiB anchor. */
export interface MockDisk {
  running: boolean;
  report: Record<string, unknown>;
}

const ROW = 106_496;
const ANCHOR = 153_944_064;
const reads = (tokens: number) => tokens * ROW + ANCHOR;

export function diskSeed(now: number): MockDisk {
  const s = now / 1000;
  const top = [
    { tokens: 27_648, kind: 'opencode-head', label: null, hits: 14, used: s - 1_800, created: s - 6 * 86_400, complete: true, reads_bytes: reads(27_648), value: 4.1e-5, preview: 'You are opencode, an interactive CLI tool that helps users with software engineering tasks.' },
    { tokens: 26_624, kind: 'opencode-head', label: null, hits: 3, used: s - 86_400, created: s - 3 * 86_400, complete: true, reads_bytes: reads(26_624), value: 9e-6, preview: 'You are opencode, an interactive CLI tool that helps users with software engineering tasks.' },
    { tokens: 31_744, kind: 'package', label: 'voxel-seasons', hits: 2, used: s - 7_200, created: s - 2 * 86_400, complete: true, reads_bytes: reads(31_744), value: 6e-6, preview: 'The runtime remembers: # Voxel seasons. Four palettes, one per season, applied by stage.' },
    { tokens: 12_288, kind: 'head', label: null, hits: 1, used: s - 40_000, created: s - 40_000, complete: true, reads_bytes: reads(12_288), value: 5e-6, preview: null },
    { tokens: 9_216, kind: 'memory', label: null, hits: 0, used: s - 90_000, created: s - 90_000, complete: true, reads_bytes: reads(9_216), value: 1e-6, preview: null },
    { tokens: 61_440, kind: 'prefix', label: null, hits: 1, used: s - 20_000, created: s - 30_000, complete: true, reads_bytes: reads(61_440), value: 2e-6, preview: null },
  ];
  const chunks = 61 + 4 + 12 + 9 + 1;
  const bytes = chunks * 1024 * ROW + top.length * ANCHOR;
  return {
    running: true,
    report: {
      on: true, root: '~/.qwen38-spark-engine/cache/disk', chunk: 1024, fingerprint: '9c41e0d2a7b3f155', entries: top.length, chunks, bytes, budget: 50e9, floor: 50e9,
      free_bytes: 276e9, staging_bytes: 0, staging_planned: 526_385_152, head_anchors: 4, min_tokens: 4096, hit_rate: 0.6522, read_bps: 5.4e9, counted_lines: 1_840, sessions_seen: 23, open_jobs: 0,
      top, by_kind: { 'opencode-head': { entries: 2, hits: 17 }, package: { entries: 1, hits: 2 }, head: { entries: 1, hits: 1 }, memory: { entries: 1, hits: 0 }, prefix: { entries: 1, hits: 1 } },
      events: [
        { kind: 'write', ts: s - 6_000, tokens: 27_648, anchors: 2, chunks: 27, bytes: 3_250_000_000, kinds: ['opencode-head'], dropped: null },
        { kind: 'hit', ts: s - 1_800, tokens: 27_648, read: 27_648, bytes: 3_098_000_000, ms: 612.4, entry: 'opencode-head' },
        { kind: 'drop', ts: s - 900, why: 'window', tokens: 40_960, what: 'chunk' },
        { kind: 'evict', ts: s - 600, why: 'ttl', tokens: 18_432, chunks: 6, bytes: 808_000_000 },
      ],
      lookups: 23, hits: 15, misses: 8, declined_slow: 1, rejected_collisions: 0, incomplete: 0, tokens_restored: 412_672, bytes_read: 46_100_000_000, read_ms: 8_400, last_read_gbs: 5.06, last_restore_ms: 612.4,
      jobs: 9, writes_chunk: 120, writes_anchor: 9, bytes_written: 14_200_000_000, write_ms: 3_300, last_write_gbs: 4.31, evicted: { ttl: 1 }, dropped: { window: 1 }, corrupt: 0, failed_restores: 0,
      swept_tmp: 0, swept_fingerprint: 0, cleared: 0, direct: true, admitted: { 'opencode-head': 4, package: 1, head: 1, memory: 1, prefix: 2 },
    },
  };
}

/** `/v1/conductor/memory/disk[/clear]` on the mock, as server/conductor/routes.py answers. */
export function diskRoute(disk: MockDisk, method: string, rest: string[], settings: Record<string, unknown>, version: number): { status: number; body: unknown } {
  const view = () => {
    const base = { contract_version: '1.0', running: disk.running, settings_version: version, settings, flag: disk.running ? '/home/demo/.qwen38-spark-engine/cache/disk' : 'off', floor_gb: 50 };
    if (!disk.running) return { ...base, root: null, free_bytes: 276e9 };
    return { ...base, ...disk.report, settings: { ...settings }, contract_version: '1.0' };
  };
  if (rest.length === 1 && method === 'GET') return { status: 200, body: view() };
  if (rest.length === 2 && rest[1] === 'clear' && method === 'POST') {
    if (!disk.running) return { status: 409, body: { contract_version: '1.0', error: { type: 'not_running', message: 'the disk tier is not running on this server' } } };
    const r = disk.report;
    const before = { entries: r.entries as number, chunks: r.chunks as number, bytes: r.bytes as number };
    Object.assign(r, { entries: 0, chunks: 0, bytes: 0, top: [], by_kind: {}, cleared: (r.cleared as number) + 1, counted_lines: 0, sessions_seen: 0 });
    (r.events as unknown[]).push({ kind: 'clear', ts: Date.now() / 1000, entries: before.entries, chunks: before.chunks, bytes: before.bytes });
    return { status: 200, body: { contract_version: '1.0', cleared: true, ...before, disk: view() } };
  }
  return { status: 404, body: { error: { type: 'not_found', message: `no route ${method} /v1/conductor/memory/${rest.join('/')}` } } };
}
