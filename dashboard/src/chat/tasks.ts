// VIS-E15: a main agent with workers. The words of the Task card (VIS-217), of the worker's part of
// the Context panel and of the Agents switch (VIS-219). A `task` call in the main thread is one
// card; its figures come from the task.* events, its transcript from the worker's part events (they
// carry `task_id` and never reach the main thread) or from GET .../tasks/{task_id}. Pure, so every
// state of the card is tested without a DOM.

import { exact } from '../lib/format';
import { sandboxPath, viewUrl } from './file-panes';
import type { ContextReport, Message, PaneRef, Task, TaskReport, ToolCallPart, WorkerContext } from './types';

export type TaskTone = 'live' | 'warn' | 'ok' | 'err' | 'muted';
export type TaskKey = 'running' | 'waiting' | 'done' | 'partial' | 'failed' | 'cancelled' | 'interrupted';

/** the worker's call that waits for permission, if any */
export function askingCall(t: Pick<Task, 'messages'>): { message: Message; call: ToolCallPart } | null {
  for (const m of t.messages ?? []) for (const p of m.parts) if (p.type === 'tool_call' && p.status === 'asking') return { message: m, call: p };
  return null;
}

/** the status chip: its key, its words (never a colour alone) and its tone. A running worker
 *  whose call asks reads "waiting for permission". */
export function taskStatus(t: Pick<Task, 'status' | 'phase' | 'messages'>): { key: TaskKey; word: string; tone: TaskTone } {
  const s = String(t.status);
  if (s === 'running') return t.phase === 'permission' || askingCall(t) ? { key: 'waiting', word: 'waiting for permission', tone: 'warn' } : { key: 'running', word: 'running', tone: 'live' };
  if (s.startsWith('partial')) return { key: 'partial', word: s, tone: 'warn' };
  if (s === 'done') return { key: 'done', word: 'done', tone: 'ok' };
  if (s === 'failed') return { key: 'failed', word: 'failed', tone: 'err' };
  if (s === 'interrupted') return { key: 'interrupted', word: 'interrupted', tone: 'warn' };
  return { key: 'cancelled', word: s || 'cancelled', tone: 'muted' };
}

/** "call 12 of 40" while it runs, "14 calls" after */
export function callsWord(t: Pick<Task, 'status' | 'steps' | 'max_steps'>): string {
  if (t.status === 'running') return `call ${exact(Math.max(1, t.steps))} of ${exact(t.max_steps)}`;
  return `${exact(t.steps)} call${t.steps === 1 ? '' : 's'}`;
}

/** the worker's tokens in and out; a running worker before its first usage frame says so instead
 *  of a made-up zero */
export function tokensWord(t: Pick<Task, 'status' | 'prompt_tokens' | 'completion_tokens'>): string {
  if (t.status === 'running' && !t.prompt_tokens && !t.completion_tokens) return 'no usage yet';
  return `${exact(t.prompt_tokens)} in · ${exact(t.completion_tokens)} out`;
}

/** seconds since the worker started, or its whole time once it ended */
export function taskSeconds(t: Pick<Task, 'status' | 'started' | 'ended' | 'stats'>, now: number): number {
  const start = Date.parse(t.started);
  if (t.status !== 'running' && t.stats?.ms != null) return Math.round(t.stats.ms / 1000);
  const end = t.ended ? Date.parse(t.ended) : t.status === 'running' ? now : NaN;
  return Number.isFinite(start) && Number.isFinite(end) ? Math.max(0, Math.round((end - start) / 1000)) : 0;
}

/** the folded card's line: the report's summary, else what ended it */
export function summaryLine(t: Pick<Task, 'status' | 'report' | 'error' | 'steps'>): string {
  const s = t.report?.summary?.split('\n')[0].trim();
  if (s) return s;
  if (t.error) return t.error.split('\n')[0];
  if (t.status === 'cancelled') return `cancelled by the user at call ${exact(t.steps)}`;
  if (t.status === 'interrupted') return 'The worker was interrupted. Continue resumes it, then the main agent.';
  return '';
}

/** a report that chatd built from the worker's steps says so */
export function sourceWord(t: Pick<Task, 'report_source'>): string | null {
  return t.report_source === 'chatd' ? 'report built by chatd from the worker\'s steps' : t.report_source === 'worker-second' ? 'the worker sent its report again after a field was missing' : null;
}

// ---- the report ---------------------------------------------------------------------------------

export interface ReportView {
  status: string;
  summary: string;
  changed: { path: string; change: string }[];
  checks: { text: string; call_id: string | null; shot: string | null }[];
  expected: string[];
  open: string[];
}

/** the report's fields as lines; an item that came as its text stays as it is */
export function reportView(r: TaskReport): ReportView {
  return {
    status: r.status,
    summary: r.summary,
    changed: (r.changed_files ?? []).map((f) => (typeof f === 'string' ? splitPath(f) : { path: f.path, change: f.change })),
    checks: (r.checks ?? []).map((c) => (typeof c === 'string' ? { text: c, call_id: null, shot: shotIn(c) } : { text: c.result ? `${c.what}: ${c.result}` : c.what, call_id: c.call_id ?? null, shot: c.shot ?? shotIn(c.result ?? '') })),
    expected: (r.expected ?? []).map((e) => (typeof e === 'string' ? e : `${e.item}: ${e.answer}`)),
    open: (r.open_problems ?? []).filter((x) => x && x.trim().toLowerCase() !== 'none'),
  };
}

function splitPath(line: string): { path: string; change: string } {
  const i = line.indexOf(': ');
  return i > 0 ? { path: line.slice(0, i), change: line.slice(i + 2) } : { path: line, change: '' };
}

// ---- chatd's check --------------------------------------------------------------------------------

export interface CheckView {
  /** the snapshot diff: each changed file and its line counts */
  changed: { path: string; delta: string; unreported: boolean }[];
  unreported: string[];
  /** the checks that really ran, each with its call */
  ran: { text: string; call_id: string | null; shot: string | null }[];
  /** what the report claims under Checks with no call behind it */
  unmatched: string[];
  /** any other line, as chatd wrote it */
  rest: string[];
}

const NONE = /^none\.?$/i;

/** chatd's check text ("- Changed in /workspace (snapshot diff): a (+41 -12), b (+3 -1)" and the
 *  lines after it) as its parts */
export function checkView(text: string | null | undefined): CheckView {
  const out: CheckView = { changed: [], unreported: [], ran: [], unmatched: [], rest: [] };
  if (!text) return out;
  const items = (v: string, sep: string) => (NONE.test(v.trim()) ? [] : v.split(sep).map((x) => x.trim()).filter(Boolean));
  for (const raw of text.split('\n')) {
    const line = raw.replace(/^\s*-\s*/, '').trim();
    if (!line || /^chatd check:?$/i.test(line)) continue;
    const m = /^([^:]+?)(?: \([^)]*\))?: (.*)$/.exec(line);
    const key = m?.[1].toLowerCase() ?? '';
    if (key === 'changed in /workspace') out.changed = items(m![2], ', ').map((f) => {
      const d = /^(.*?) (\([^)]*\))$/.exec(f);
      return { path: d ? d[1] : f, delta: d ? d[2].slice(1, -1) : '', unreported: false };
    });
    else if (key === 'changed but not in the report') out.unreported = items(m![2], ', ');
    else if (key === 'checks that ran') out.ran = items(m![2], '; ').map((x) => ({ text: x, call_id: /\bcall ([\w-]+)/.exec(x)?.[1] ?? null, shot: shotIn(x) }));
    else if (key === 'claimed in checks with no matching call') out.unmatched = items(m![2], '; ');
    else out.rest.push(line);
  }
  const un = new Set(out.unreported);
  out.changed = out.changed.map((f) => ({ ...f, unreported: un.has(f.path) }));
  return out;
}

/** a report check that chatd found no call for */
export function isUnmatched(check: { text: string }, v: CheckView): boolean {
  return v.unmatched.some((u) => check.text === u || check.text.startsWith(u));
}

// ---- screenshots ------------------------------------------------------------------------------------

export function shotIn(text: string): string | null {
  return /\bshot-\d+\b/.exec(text)?.[0] ?? null;
}

export interface ShotView {
  id: string;
  /** the thumbnail: the call's own image, else the file under .tandem/shots/ */
  src: string;
  pane: PaneRef;
}

/** the screenshots the report and the check name, each with its picture and its pane: the
 *  render_page call's own when the transcript holds it */
export function shotsOf(t: Pick<Task, 'report' | 'check' | 'messages'>, convId: string): ShotView[] {
  const ids = new Map<string, string | null>();
  const add = (shot: string | null, call: string | null) => shot && (!ids.has(shot) || !ids.get(shot)) && ids.set(shot, call);
  if (t.report) for (const c of reportView(t.report).checks) add(c.shot, c.call_id);
  for (const c of checkView(t.check).ran) add(c.shot, c.call_id);
  const calls = new Map<string, ToolCallPart>();
  for (const m of t.messages ?? []) for (const p of m.parts) if (p.type === 'tool_call') calls.set(p.id, p);
  return [...ids].map(([id, callId]) => {
    const call = callId ? calls.get(callId) : undefined;
    const img = call?.output?.images?.[0];
    const path = sandboxPath(`.tandem/shots/${id}.png`);
    return { id, src: img?.preview || img?.url || viewUrl(convId, path), pane: call?.pane?.kind === 'image' ? call.pane : { kind: 'image', ref: `file:${path}`, title: id } };
  });
}

// ---- the Context panel ------------------------------------------------------------------------------

const SPLIT: { key: keyof NonNullable<WorkerContext['split']>; label: string }[] = [
  { key: 'system', label: 'system prompt' },
  { key: 'tools', label: 'tool schemas' },
  { key: 'brief', label: 'brief' },
  { key: 'steps', label: 'its steps' },
  { key: 'files', label: 'files' },
];

/** the worker's split in a fixed order; the parts the report has */
export function workerParts(w: Pick<WorkerContext, 'split'>): { key: string; label: string; tokens: number }[] {
  if (!w.split) return [];
  return SPLIT.filter((s) => w.split![s.key] != null).map((s) => ({ key: s.key, label: s.label, tokens: w.split![s.key] ?? 0 }));
}

/** the worker's cache tier; a cloud model's worker has none on this box */
export function workerCacheWords(w: Pick<WorkerContext, 'cache'>, backend: ContextReport['model']['backend']): string {
  if (backend === 'cloud') return 'no cache tier on this box: the provider keeps it';
  if (!w.cache) return 'not reported yet';
  if (!w.cache.warm) return w.cache.cold ? `${exact(w.cache.cold)} cold to prefill` : 'nothing cached yet';
  return `${exact(w.cache.warm)} warm from the ${w.cache.source === 'none' ? 'cache' : w.cache.source}${w.cache.cold ? `, ${exact(w.cache.cold)} cold` : ''}`;
}

/** the line that stays after a worker ended, until the next task: "Last worker: Task 2, 51,204
 *  tokens, done". The tokens are its context as this device last saw it; after a reload the line
 *  goes without them. */
export function lastWorkerLine(task: Pick<Task, 'id' | 'n' | 'status' | 'phase' | 'messages'>, seen: WorkerContext | null | undefined): string {
  const used = seen && seen.task_id === task.id ? `, ${exact(seen.used)} tokens` : '';
  return `Last worker: Task ${task.n}${used}, ${taskStatus(task).word}`;
}

// ---- the Agents switch ------------------------------------------------------------------------------

export const AGENTS_HINT = 'A main agent plans and workers change the files';

/** before a switch in a chat with messages: the prompt head changes, so the next message reads
 *  the chat once more (the chat's context now, about what chatd re-reads) */
export function agentsAsk(tokens: number, running: boolean): string {
  return `The next message reads this chat once more (about ${exact(tokens)} tokens). ${running ? 'The run that is on keeps its mode; the switch applies at the next message.' : 'The switch applies at the next message.'}`;
}

/** after the switch: chatd's own figure from the settings answer (`reread_tokens`) */
export function agentsDone(on: boolean, reread: number | null | undefined): string {
  return `Agents ${on ? 'on' : 'off'} from the next message.${reread ? ` Switching re-reads this chat once: ${exact(reread)} tokens.` : ''}`;
}
