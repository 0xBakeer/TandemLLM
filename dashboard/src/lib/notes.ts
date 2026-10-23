// The Memory screen's Notes tab (2026-10-05): the rules kept out of the component so the tests read
// them directly -- the kinds in plain words, the list's filters, the tree's labels, a line diff for a
// note's history, the journal's write rows in words, and the notes settings over their defaults.

import type { MemoryWhere, NoteKind, NoteMeta, NoteNode, NotesSettings, NoteWriteRow, WriterSettings } from '../api/conductor';

export const WRITER_DEFAULTS: WriterSettings = { mode: 'off', idle_s: 180, due_idle_s: 3, min_new_tokens: 1500, max_tokens: 600, max_blocks: 4, gate: true, after_compaction: 'idle', turn_deadline_s: 30, existing: 5 };

export const NOTES_DEFAULTS: NotesSettings = {
  index_tokens: 1500,
  body_tokens: 600,
  pinned_tokens: 300,
  bodies: 3,
  min_score: 0,
  min_words: 2,
  dedupe_threshold: 0.6,
  intent: 'shadow',
  provisional: true,
  digest_chars: 2000,
  writer: WRITER_DEFAULTS,
};

export const KIND_WORDS: Record<NoteKind, string> = {
  rule: 'how I want things done',
  ruling: 'a decision that was taken',
  gotcha: 'what went wrong once',
  runbook: 'how to do a thing',
  program_state: 'where the work stands',
  reference: 'a fact to look up',
  registry: 'a list of things',
  archive: 'kept for the record',
};
export const KINDS = Object.keys(KIND_WORDS) as NoteKind[];

export const WRITER_WORDS: Record<string, string> = {
  'qwen-inline': 'the model, mid-answer',
  'qwen-writer': 'the model, in the background',
  api: "the model's script",
  human: 'you',
};

export interface NoteFilter {
  node: string | null;
  kind: NoteKind | '';
  status: 'provisional' | 'ratified' | '';
  writer: string;
  stale: boolean;
  q: string;
}
export const NO_FILTER: NoteFilter = { node: null, kind: '', status: '', writer: '', stale: false, q: '' };

export function filterNotes(notes: NoteMeta[], f: NoteFilter): NoteMeta[] {
  const q = f.q.trim().toLowerCase();
  return notes.filter(
    (n) =>
      (!f.node || n.node === f.node) &&
      (!f.kind || n.kind === f.kind) &&
      (!f.status || n.status === f.status) &&
      (!f.writer || n.writer === f.writer) &&
      (!f.stale || Boolean(n.stale)) &&
      (!q || `${n.title} ${n.tags.join(' ')} ${n.id}`.toLowerCase().includes(q)),
  );
}

/** The tree node in words: `user` "About you", a project its path, a session its key. */
export function nodeLabel(n: Pick<NoteNode, 'node' | 'scope' | 'label'>): string {
  if (n.scope === 'user') return 'About you';
  if (n.scope === 'project') return n.label ?? n.node.replace(/^projects\//, '');
  return `session ${n.label ?? n.node.replace(/^sessions\//, '')}`;
}

export type DiffOp = { op: '=' | '+' | '-'; text: string };

/** A line diff (LCS) from `a` to `b`: unchanged, added and removed lines in order. */
export function lineDiff(a: string, b: string): DiffOp[] {
  const x = a.split('\n');
  const y = b.split('\n');
  const n = x.length;
  const m = y.length;
  if (n * m > 400_000) return [...x.map((t) => ({ op: '-' as const, text: t })), ...y.map((t) => ({ op: '+' as const, text: t }))];
  const L: number[][] = Array.from({ length: n + 1 }, () => new Array<number>(m + 1).fill(0));
  for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--) L[i][j] = x[i] === y[j] ? L[i + 1][j + 1] + 1 : Math.max(L[i + 1][j], L[i][j + 1]);
  const out: DiffOp[] = [];
  let i = 0;
  let j = 0;
  while (i < n && j < m) {
    if (x[i] === y[j]) {
      out.push({ op: '=', text: x[i] });
      i++;
      j++;
    } else if (L[i + 1][j] >= L[i][j + 1]) out.push({ op: '-', text: x[i++] });
    else out.push({ op: '+', text: y[j++] });
  }
  while (i < n) out.push({ op: '-', text: x[i++] });
  while (j < m) out.push({ op: '+', text: y[j++] });
  return out;
}

/** One journal row of the writes in words: "added gotcha (project) 0a1b..." or "refused: cap". */
export function writeLine(r: NoteWriteRow): string {
  const d = (r.detail ?? {}) as Record<string, unknown>;
  if (r.task === 'nightly') return `nightly: ${d.deleted ?? 0} archives deleted, ${d.duplicates ?? 0} duplicates, ${d.stale ?? 0} stale`;
  const why = typeof d.why === 'string' ? d.why : null;
  const op = String(d.op ?? r.decision ?? 'write');
  const who = r.task === 'api_write' ? "the model's script" : WRITER_WORDS[String(d.writer ?? 'qwen-inline')] ?? String(d.writer);
  if (why || (r.decision && !['add', 'update', 'archive', 'propose', 'ok'].includes(r.decision))) return `refused (${why ?? r.decision}): ${op} by ${who}`;
  const verb = op === 'add' ? (d.revised ? 'revised' : 'added') : op === 'update' ? 'updated' : op === 'archive' ? 'archived' : op === 'propose' ? 'proposed' : op;
  const what = [d.kind, d.scope ? `(${d.scope})` : null].filter(Boolean).join(' ');
  return `${verb} ${what} by ${who}${d.id ? ` · ${d.id}` : ''}`.replace(/\s+/g, ' ').trim();
}

export function notesSettings(doc: { memory?: { notes?: Partial<NotesSettings> } } | null | undefined): NotesSettings {
  const got = doc?.memory?.notes ?? {};
  return { ...NOTES_DEFAULTS, ...got, writer: { ...WRITER_DEFAULTS, ...(got.writer ?? {}) } };
}

export interface WriterRun {
  when: number;
  trigger: string;
  outcome: string;
  gate: string;
  seconds: number | null;
  tokens: number | null;
  ops: string;
  refused: string[];
}

/** A writer run's journal row (task `writer`) in parts for the table. */
export function writerRun(r: NoteWriteRow): WriterRun {
  const d = (r.detail ?? {}) as Record<string, unknown>;
  const ops = (Array.isArray(d.ops) ? d.ops : []) as Record<string, unknown>[];
  const by = (op: string) => ops.filter((o) => o.op === op && (op !== 'add' || !o.revised)).length;
  const revised = ops.filter((o) => o.revised).length;
  const parts = [by('add') ? `${by('add')} added` : '', revised ? `${revised} revised` : '', by('update') ? `${by('update')} updated` : '', by('archive') ? `${by('archive')} archived` : ''].filter(Boolean);
  const g = (d.gate ?? {}) as Record<string, unknown>;
  const stored = ops.length && ops.every((o) => o.stored === false) ? ' (shadow: not stored)' : '';
  return {
    when: r.ts_ms,
    trigger: String(d.trigger ?? '—'),
    outcome: typeof d.skipped === 'string' ? d.skipped : ops.length ? 'wrote' : d.none ? 'nothing worth keeping' : 'no block',
    gate: g.answer ? String(g.answer) : String(g.how ?? '—'),
    seconds: typeof d.ms === 'number' ? Math.round(d.ms / 100) / 10 : null,
    tokens: typeof d.tokens === 'number' ? d.tokens : null,
    ops: (parts.join(', ') || '—') + stored,
    refused: Array.isArray(d.refused) ? (d.refused as string[]) : [],
  };
}

/** Uses by day as one short line: "3 in 7 days (2 today)". */
export function usesLine(uses: Record<string, number>, today: string): string {
  const all = Object.values(uses).reduce((s, v) => s + v, 0);
  if (!all) return 'not injected yet';
  const t = uses[today] ?? 0;
  return `${all} injection${all === 1 ? '' : 's'}${t ? ` (${t} today)` : ''}`;
}

// SRV-703: where memory is on. The served model first (a model whose prompt has no place for the
// conductor's notes reads none on any profile), then each profile: on, in shadow, off (the model is
// told), or off with nothing added (`none`, or the switch not set). A profile is named with the
// clients it serves, the default one with "any other client".
export const SWITCH_WORDS: Record<string, string> = {
  on: 'on',
  shadow: 'in shadow: written, not read back',
  off: 'off, and the model is told',
  none: 'off',
};

export interface MemoryWhereLine {
  name: string;
  on: boolean;
  state: string;
  who: string;
}

export function memoryWhere(w: MemoryWhere | undefined | null): { model: string | null; lines: MemoryWhereLine[] } | null {
  if (!w) return null;
  const model = w.model.memory ? null : `Memory is off for the served model: ${w.model.words ?? w.model.why ?? 'its conductor fit'}. No profile reads or writes notes while it serves.`;
  const lines = w.profiles
    .filter((p) => p.mode !== 'raw')
    .map((p) => {
      const sw = p.memory in SWITCH_WORDS ? p.memory : 'none';
      const state = !w.model.memory && (sw === 'on' || sw === 'shadow') ? `${SWITCH_WORDS[sw]} in the profile, off for this model` : SWITCH_WORDS[sw] + (p.set ? '' : ' (not set)');
      const who = [...p.clients, ...(p.default ? ['any other client'] : [])].join(', ');
      return { name: p.name, on: w.model.memory && (sw === 'on' || sw === 'shadow'), state, who };
    })
    .sort((a, b) => Number(b.on) - Number(a.on) || a.name.localeCompare(b.name));
  return { model, lines };
}
