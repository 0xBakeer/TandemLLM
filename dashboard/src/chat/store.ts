// The conversation reducer: one event in, a new conversation out. Pure, so the live resume
// (a catch-up replay of events after Last-Event-ID) and the live stream go through the same
// code, and the order is unit-tested. Nothing here touches the DOM or the network.

import { compact, fixed } from '../lib/format';
import type { ChatEvent, Compaction, ContextReport, Conversation, Message, Part, Run, StampedEvent, Task, ToolCallPart, WorkerContext } from './types';

export const TPS_HISTORY = 120;

const TPS_SLOTS = TPS_HISTORY;

export function applyEvent(c: Conversation, ev: ChatEvent): Conversation {
  // a worker's events (VIS-E15) go to its task's transcript and never to the main thread
  const tid = workerTask(c, ev);
  if (tid) return applyWorkerEvent(c, tid, ev);
  switch (ev.type) {
    case 'run.started':
      return { ...c, run: { ...ev.run, tps_history: ev.run.tps_history ?? [] }, status: 'running', now: ev.run.now };
    case 'run.phase': {
      if (!c.run || c.run.id !== ev.run_id) return c;
      const run: Run = { ...c.run, phase: ev.phase, now: ev.now, tool: ev.tool ?? (ev.phase === 'tool' || ev.phase === 'permission' ? c.run.tool : null), prefill: ev.prefill ?? (ev.phase === 'prefill' ? c.run.prefill : null), step: ev.step ?? c.run.step, queue_position: ev.queue_position === undefined ? (ev.phase === 'queued' ? c.run.queue_position : null) : ev.queue_position };
      return { ...c, run, status: ev.phase === 'permission' ? 'waiting' : 'running', now: ev.now };
    }
    case 'run.metrics': {
      if (!c.run || c.run.id !== ev.run_id) return c;
      const hist = [...c.run.tps_history, ev.decode_tps].slice(-TPS_SLOTS);
      return { ...c, run: { ...c.run, decode_tps: ev.decode_tps, tokens_per_round: ev.tokens_per_round, acceptance: ev.acceptance, tokens: ev.tokens, ttft_ms: ev.ttft_ms === undefined ? c.run.ttft_ms : ev.ttft_ms, thinking: { used: ev.thinking_used, call: ev.thinking_call, budget: ev.thinking_budget === undefined ? c.run.thinking.budget : ev.thinking_budget }, prefill: ev.prefill ?? c.run.prefill, tps_history: hist } };
    }
    case 'run.done': {
      const messages = mapMessage(c.messages, ev.message_id, (m) => ({ ...m, status: ev.status, stats: ev.stats, parts: m.parts.map((p) => (p.type === 'thinking' ? { ...p, done: true } : p)) }));
      return { ...c, messages, run: null, status: ev.status === 'interrupted' ? 'error' : 'idle', now: null, turns: messages.filter((m) => m.role === 'assistant').length, head_id: ev.message_id };
    }
    case 'run.error': {
      const messages = mapMessage(c.messages, ev.message_id, (m) => ({ ...m, status: 'error', parts: [...m.parts, { type: 'error', message: ev.message, retryable: ev.retryable }] }));
      return { ...c, messages, run: null, status: 'error', now: null };
    }
    case 'message.created':
      return { ...c, messages: c.messages.some((m) => m.id === ev.message.id) ? c.messages : [...c.messages, ev.message], updated_at: ev.message.created_at, preview: ev.message.role === 'user' ? firstLine(ev.message) : c.preview, head_id: ev.message.id };
    case 'part.added':
    case 'part.delta':
    case 'part.done':
    case 'tool.output':
    case 'tool.done':
      return { ...c, messages: applyMessageEvent(c.messages, ev) };
    case 'tool.asking':
      return { ...c, status: 'waiting', messages: applyMessageEvent(c.messages, ev) };
    case 'tool.started':
      return { ...c, status: 'running', messages: applyMessageEvent(c.messages, ev) };
    case 'task.started': {
      const had = c.tasks?.find((t) => t.id === ev.task.id);
      const task: Task = { ...had, ...ev.task, messages: had?.messages ?? ev.task.messages ?? [] };
      return { ...c, tasks: had ? c.tasks!.map((t) => (t.id === task.id ? task : t)) : [...(c.tasks ?? []), task] };
    }
    case 'task.phase':
      return mapTask(c, ev.task_id, (t) => ({ ...t, phase: ev.phase, now: ev.now, steps: ev.step, max_steps: ev.max_steps, tool: ev.tool ?? null }));
    case 'task.metrics': {
      const { type: _t, ...m } = ev;
      return mapTask(c, ev.task_id, (t) => ({ ...t, prompt_tokens: ev.prompt_tokens, completion_tokens: ev.completion_tokens, decode_tps: ev.decode_tps, metrics: m }));
    }
    case 'task.done':
      return mapTask(c, ev.task_id, (t) => {
        const st = ev.stats ?? {};
        return {
          ...t,
          status: ev.status,
          report: ev.report,
          check: ev.check,
          stats: ev.stats ?? t.stats ?? null,
          steps: st.steps ?? t.steps,
          prompt_tokens: st.prompt_tokens ?? t.prompt_tokens,
          completion_tokens: st.completion_tokens ?? t.completion_tokens,
          cached_tokens: st.cached_tokens ?? t.cached_tokens,
          ended: t.ended ?? new Date().toISOString(),
          phase: null,
          now: null,
          tool: null,
          decode_tps: null,
          metrics: null,
          messages: t.messages?.map((m) => (m.status === 'streaming' ? { ...m, status: 'done', parts: m.parts.map((p) => (p.type === 'thinking' ? { ...p, done: true } : p)) } : m)),
        };
      });
    case 'queue.changed':
      return { ...c, queue: ev.queue };
    case 'context.changed':
      // the panel's "Last worker" line keeps the newest worker this device saw
      return { ...c, context: ev.context, last_worker: newestWorker(ev.context) ?? c.last_worker ?? null };
    case 'compaction':
      return { ...c, compactions: c.compactions.some((x) => x.id === ev.compaction.id) ? c.compactions : [...c.compactions, ev.compaction] };
    case 'conversation.updated':
      return { ...c, ...ev.patch };
    case 'sandbox.changed':
      return { ...c, sandbox: ev.sandbox };
    case 'usage.changed':
      return { ...c, usage: ev.usage };
    case 'todos.changed':
      return { ...c, todos: ev.todos };
    case 'settings.changed':
      return { ...c, settings: ev.settings };
    case 'branch.changed':
      // the view reloads the conversation on this event; the head moves here so a replay agrees
      return { ...c, head_id: ev.head_id };
    case 'ui.changed':
      return { ...c, ui: ev.ui };
    case 'pane.open':
    case 'artifact.created':
      // the view acts on these (opens the pane); nothing in the conversation changes
      return c;
  }
  return c;
}

/** The events of messages and their parts (the main thread's and a worker's transcript alike). */
function applyMessageEvent(ms: Message[], ev: ChatEvent): Message[] {
  switch (ev.type) {
    case 'part.added':
      return mapMessage(ms, ev.message_id, (m) => ({ ...m, parts: [...m.parts, ev.part] }));
    case 'part.delta':
      return mapMessage(ms, ev.message_id, (m) => {
        const parts = m.parts.slice();
        const p = parts[ev.index];
        if (!p) return m;
        if (p.type === 'text') parts[ev.index] = { ...p, text: p.text + ev.delta };
        else if (p.type === 'thinking') parts[ev.index] = { ...p, text: p.text + ev.delta };
        else if (p.type === 'tool_call') parts[ev.index] = { ...p, input: p.input + ev.delta };
        return { ...m, parts };
      });
    case 'part.done':
      return mapMessage(ms, ev.message_id, (m) => {
        const parts = m.parts.slice();
        const p = parts[ev.index];
        if (p?.type === 'thinking') parts[ev.index] = { ...p, done: true, tokens: ev.tokens ?? p.tokens };
        return { ...m, parts };
      });
    case 'tool.asking':
      return mapCall(ms, ev.message_id, ev.call_id, (t) => ({ ...t, status: 'asking', risk: (ev.risk as ToolCallPart['risk']) ?? t.risk, permission: { summary: ev.summary, suggested_rule: ev.suggested_rule } }));
    case 'tool.started':
      return mapCall(ms, ev.message_id, ev.call_id, (t) => ({ ...t, status: 'running', started_at: t.started_at ?? new Date().toISOString(), pane: ev.pane ?? t.pane ?? null, permission: null, approved: ev.approved ?? t.approved ?? null }));
    case 'tool.output':
      return mapCall(ms, ev.message_id, ev.call_id, (t) => ({ ...t, output: { content: (t.output?.content ?? '') + ev.delta, is_error: t.output?.is_error ?? false, bytes: (t.output?.bytes ?? 0) + ev.delta.length } }));
    case 'tool.done':
      return mapCall(ms, ev.message_id, ev.call_id, (t) => ({ ...t, status: ev.status, ended_at: new Date().toISOString(), permission: null, output: ev.output !== undefined ? ev.output : t.output ? { ...t.output, is_error: ev.is_error } : ev.status === 'denied' || ev.status === 'cancelled' ? null : { content: '', is_error: ev.is_error, bytes: 0 } }));
  }
  return ms;
}

/** the task a worker's event belongs to: its `task_id` (parts, messages), else the task whose
 *  transcript holds its message (the tool events, a worker run's end); null for the main thread */
function workerTask(c: Conversation, ev: ChatEvent): string | null {
  if ((ev.type === 'part.added' || ev.type === 'part.delta' || ev.type === 'part.done' || ev.type === 'message.created') && ev.task_id) return ev.task_id;
  if (!c.tasks?.length) return null;
  const mid = 'message_id' in ev && !ev.type.startsWith('task.') ? ev.message_id : null;
  if (!mid || c.messages.some((m) => m.id === mid)) return null;
  return c.tasks.find((t) => t.messages?.some((m) => m.id === mid))?.id ?? null;
}

/** A worker's event into its task's transcript. A task this device has no transcript of yet (a
 *  finished one, before it is opened) drops it: GET .../tasks/{task_id} has it. */
function applyWorkerEvent(c: Conversation, tid: string, ev: ChatEvent): Conversation {
  const task = c.tasks?.find((t) => t.id === tid);
  if (!task?.messages) return c;
  let ms = task.messages;
  switch (ev.type) {
    case 'message.created':
      if (!ms.some((m) => m.id === ev.message.id)) ms = [...ms, ev.message];
      break;
    case 'part.added':
      // a worker's answer can start with its first part: the card needs no message.created
      if (!ms.some((m) => m.id === ev.message_id)) ms = [...ms, { id: ev.message_id, role: 'assistant', created_at: new Date().toISOString(), parent_id: null, run_id: null, parts: [], status: 'streaming' }];
      ms = applyMessageEvent(ms, ev);
      break;
    case 'run.done':
      ms = mapMessage(ms, ev.message_id, (m) => ({ ...m, status: ev.status, stats: ev.stats, parts: m.parts.map((p) => (p.type === 'thinking' ? { ...p, done: true } : p)) }));
      break;
    case 'run.error':
      ms = mapMessage(ms, ev.message_id, (m) => ({ ...m, status: 'error', parts: [...m.parts, { type: 'error', message: ev.message, retryable: ev.retryable }] }));
      break;
    default:
      ms = applyMessageEvent(ms, ev);
  }
  const out = mapTask(c, tid, (t) => ({ ...t, messages: ms }));
  // a worker's prompt makes the chat wait for an answer like the main agent's does
  if (ev.type === 'tool.asking') return { ...out, status: 'waiting' };
  if (ev.type === 'tool.started' && c.run) return { ...out, status: 'running' };
  return out;
}

function mapTask(c: Conversation, id: string, f: (t: Task) => Task): Conversation {
  if (!c.tasks?.some((t) => t.id === id)) return c;
  return { ...c, tasks: c.tasks.map((t) => (t.id === id ? f(t) : t)) };
}

/** GET .../tasks over the tasks in hand (the stream may have run ahead of the list): an ended task
 *  beats a running copy, a running one keeps the stream's live figures, the transcript stays */
export function mergeTasks(have: Task[] | undefined, list: Task[]): Task[] {
  const byId = new Map((have ?? []).map((t) => [t.id, t]));
  const out = list.map((t) => {
    const h = byId.get(t.id);
    byId.delete(t.id);
    if (!h) return t;
    const base = t.status !== 'running' ? t : h.status !== 'running' ? h : { ...t, ...h };
    return { ...base, messages: h.messages ?? t.messages };
  });
  return [...out, ...byId.values()].sort((a, b) => a.n - b.n);
}

/** One task with its transcript (GET .../tasks/{task_id}). The answer's `seq`, when it carries
 *  one, is the last event it includes: the stream's events for this task after it go on top. A
 *  transcript the stream is building already stays. */
export function adoptTask(c: Conversation, full: Task & { seq?: string }, recent: StampedEvent[] = []): Conversation {
  const had = c.tasks?.find((t) => t.id === full.id);
  if (had?.messages?.length) return c;
  const { seq, ...task } = full;
  const merged: Task = { ...task, ...(had?.status === 'running' ? { status: had.status, steps: had.steps, phase: had.phase, now: had.now, tool: had.tool, decode_tps: had.decode_tps, metrics: had.metrics, prompt_tokens: had.prompt_tokens, completion_tokens: had.completion_tokens } : {}), messages: task.messages ?? [] };
  let out: Conversation = { ...c, tasks: had ? c.tasks!.map((t) => (t.id === full.id ? merged : t)) : [...(c.tasks ?? []), merged].sort((a, b) => a.n - b.n) };
  if (seq == null) return out;
  const floor = Number(seq) || 0;
  for (const ev of recent) if (ev.id && Number(ev.id) > floor && workerTask(out, ev.event) === full.id) out = applyEvent(out, ev.event);
  return out;
}

/** the running workers of a context report, by task: `workers`, or an older chatd's `worker` */
export function contextWorkers(ctx: Pick<ContextReport, 'workers' | 'worker'> | null | undefined): WorkerContext[] {
  return ctx?.workers ?? (ctx?.worker ? [ctx.worker] : []);
}

function newestWorker(ctx: ContextReport): WorkerContext | null {
  return contextWorkers(ctx).reduce<WorkerContext | null>((a, w) => (!a || w.n > a.n ? w : a), null);
}

/** the task a main-thread `task` call stands for */
export function taskForCall(c: Pick<Conversation, 'tasks'>, callId: string): Task | null {
  return c.tasks?.find((t) => t.call_id === callId) ?? null;
}

export function applyEvents(c: Conversation, evs: ChatEvent[]): Conversation {
  return evs.reduce(applyEvent, c);
}

/** A snapshot that replaces the conversation in hand (a reload after `branch.changed`, a branch
 *  switch): it holds every event up to its `seq`. The stream may already have delivered events
 *  after that seq to the old copy, so those are applied again on top; the ones up to it are in
 *  the snapshot and are not. Returns the conversation and the last seq it now includes, below
 *  which the view drops any event (it would apply twice). */
export function adopt(snap: Conversation, recent: StampedEvent[]): { conv: Conversation; floor: number } {
  let floor = Number(snap.seq) || 0;
  let conv = snap;
  for (const ev of recent) {
    const id = Number(ev.id);
    if (!ev.id || id <= floor || ev.event.type === 'branch.changed') continue;
    conv = applyEvent(conv, ev.event);
    floor = id;
  }
  return { conv: { ...conv, seq: String(floor) }, floor };
}

function mapMessage(ms: Message[], id: string, f: (m: Message) => Message): Message[] {
  const i = ms.findIndex((m) => m.id === id);
  if (i < 0) return ms;
  const out = ms.slice();
  out[i] = f(ms[i]);
  return out;
}

function mapCall(ms: Message[], mid: string, cid: string, f: (t: ToolCallPart) => ToolCallPart): Message[] {
  return mapMessage(ms, mid, (m) => ({ ...m, parts: m.parts.map((p): Part => (p.type === 'tool_call' && p.id === cid ? f(p) : p)) }));
}

/** The message's first line of text, for the rail's preview. */
export function firstLine(m: Message): string {
  const t = m.parts.find((p) => p.type === 'text') as { text: string } | undefined;
  return (t?.text ?? '').split('\n').find((l) => l.trim()) ?? '';
}

/** The visible text of a message (what Copy takes), parts joined by blank lines. */
export function messageText(m: Message): string {
  return m.parts
    .filter((p) => p.type === 'text')
    .map((p) => (p as { text: string }).text)
    .join('\n\n');
}

/** The call that waits for permission, if any, on the active branch. */
export function pendingPermission(c: Conversation): { message: Message; call: ToolCallPart } | null {
  for (let i = c.messages.length - 1; i >= 0; i--) {
    const m = c.messages[i];
    for (const p of m.parts) if (p.type === 'tool_call' && p.status === 'asking') return { message: m, call: p };
  }
  return null;
}

/** The last assistant message that was interrupted (chatd or the engine restarted): Continue resumes it. */
export function interruptedMessage(c: Conversation): Message | null {
  const m = c.messages[c.messages.length - 1];
  return m && m.role === 'assistant' && m.status === 'interrupted' ? m : null;
}

/** The Continue card of an interrupted answer (VIS-197): its words, from the reason chatd
 *  recorded (`stats.interrupted`), and whether its last part is the run's own notice (the limit
 *  of model calls, the sandbox failing), which the card then carries in its place. It said "The
 *  engine restarted" for every one, a run at its limit included. An answer from before the
 *  reason: its notice when it ends with one, else neutral words. */
/** The card under an interrupted answer, or none while a Continue run writes into that same message
 * (the card stayed under the new text until the run ended). */
export function interruptCard(m: Message, runMessageId?: string | null): { text: string; notice: boolean } | null {
  return runMessageId && runMessageId === m.id ? null : interruptWords(m);
}

export function interruptWords(m: Message): { text: string; notice: boolean } | null {
  if (m.status !== 'interrupted') return null;
  const last = m.parts[m.parts.length - 1];
  if (last && last.type === 'error' && !last.retryable) return { text: last.message, notice: true };
  const why = m.stats?.interrupted;
  const text =
    why === 'restart'
      ? 'The chat service restarted while this answer was being written. The text so far stays.'
      : why === 'engine'
        ? 'The engine stopped answering while this answer was being written. The text so far stays.'
        : 'This answer stopped before it was finished. The text so far stays.';
  return { text, notice: false };
}

/** The last assistant message with figures, for the Context panel when nothing runs. */
export function lastStats(c: Conversation) {
  for (let i = c.messages.length - 1; i >= 0; i--) {
    const m = c.messages[i];
    if (m.role === 'assistant' && m.stats) return m.stats;
  }
  return null;
}

/** the compactions by the message they sit above; one whose first kept message is not on this
 *  branch (or that names none) sits above the first message */
export function dividersByMessage(c: Pick<Conversation, 'compactions' | 'messages'>): Map<string, Compaction[]> {
  const out = new Map<string, Compaction[]>();
  const ids = new Set(c.messages.map((m) => m.id));
  for (const cp of c.compactions) {
    const at = cp.boundary_id && ids.has(cp.boundary_id) ? cp.boundary_id : c.messages[0]?.id;
    if (!at) continue;
    out.set(at, [...(out.get(at) ?? []), cp]);
  }
  return out;
}

export function dividerWords(cp: Compaction): string {
  const t = cp.duration_ms != null ? `, ${cp.duration_ms < 10_000 ? fixed(cp.duration_ms / 1000, 1) : Math.round(cp.duration_ms / 1000)} s` : '';
  return `Earlier messages summarized: ${compact(cp.before_tokens)} to ${compact(cp.after_tokens)} tokens${t}`;
}

/** the one-line head of a tool card: its main argument */
export function argsSummary(t: ToolCallPart): string {
  try {
    const o = JSON.parse(t.input) as Record<string, unknown>;
    const v = o.command ?? o.path ?? o.query ?? o.url ?? Object.values(o)[0];
    // a list (todo_write's items) or an object reads as its size, not as "[object Object]"
    if (Array.isArray(v)) return `${v.length} item${v.length === 1 ? '' : 's'}`;
    if (v && typeof v === 'object') return `${Object.keys(v).length} field${Object.keys(v).length === 1 ? '' : 's'}`;
    return v == null ? '' : String(v).split('\n')[0].slice(0, 90);
  } catch {
    return t.input.replace(/\s+/g, ' ').slice(0, 90);
  }
}
