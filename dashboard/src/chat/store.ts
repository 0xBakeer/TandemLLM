// The conversation reducer: one event in, a new conversation out. Pure, so the live resume
// (a catch-up replay of events after Last-Event-ID) and the live stream go through the same
// code, and the order is unit-tested. Nothing here touches the DOM or the network.

import type { ChatEvent, Conversation, Message, Part, Run, ToolCallPart } from './types';

export const TPS_HISTORY = 120;

const TPS_SLOTS = TPS_HISTORY;

export function applyEvent(c: Conversation, ev: ChatEvent): Conversation {
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
      return { ...c, run: { ...c.run, decode_tps: ev.decode_tps, tokens_per_round: ev.tokens_per_round, acceptance: ev.acceptance, tokens: ev.tokens, thinking: { used: ev.thinking_used, budget: ev.thinking_budget === undefined ? c.run.thinking.budget : ev.thinking_budget }, prefill: ev.prefill ?? c.run.prefill, tps_history: hist } };
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
      return { ...c, messages: mapMessage(c.messages, ev.message_id, (m) => ({ ...m, parts: [...m.parts, ev.part] })) };
    case 'part.delta':
      return {
        ...c,
        messages: mapMessage(c.messages, ev.message_id, (m) => {
          const parts = m.parts.slice();
          const p = parts[ev.index];
          if (!p) return m;
          if (p.type === 'text') parts[ev.index] = { ...p, text: p.text + ev.delta };
          else if (p.type === 'thinking') parts[ev.index] = { ...p, text: p.text + ev.delta };
          else if (p.type === 'tool_call') parts[ev.index] = { ...p, input: p.input + ev.delta };
          return { ...m, parts };
        }),
      };
    case 'part.done':
      return {
        ...c,
        messages: mapMessage(c.messages, ev.message_id, (m) => {
          const parts = m.parts.slice();
          const p = parts[ev.index];
          if (p?.type === 'thinking') parts[ev.index] = { ...p, done: true, tokens: ev.tokens ?? p.tokens };
          return { ...m, parts };
        }),
      };
    case 'tool.asking':
      return { ...c, status: 'waiting', messages: mapCall(c.messages, ev.message_id, ev.call_id, (t) => ({ ...t, status: 'asking', risk: (ev.risk as ToolCallPart['risk']) ?? t.risk, permission: { summary: ev.summary, suggested_rule: ev.suggested_rule } })) };
    case 'tool.started':
      return { ...c, status: 'running', messages: mapCall(c.messages, ev.message_id, ev.call_id, (t) => ({ ...t, status: 'running', started_at: t.started_at ?? new Date().toISOString(), pane: ev.pane ?? t.pane ?? null, permission: null })) };
    case 'tool.output':
      return { ...c, messages: mapCall(c.messages, ev.message_id, ev.call_id, (t) => ({ ...t, output: { content: (t.output?.content ?? '') + ev.delta, is_error: t.output?.is_error ?? false, bytes: (t.output?.bytes ?? 0) + ev.delta.length } })) };
    case 'tool.done':
      return { ...c, messages: mapCall(c.messages, ev.message_id, ev.call_id, (t) => ({ ...t, status: ev.status, ended_at: new Date().toISOString(), permission: null, output: ev.output !== undefined ? ev.output : t.output ? { ...t.output, is_error: ev.is_error } : ev.status === 'denied' || ev.status === 'cancelled' ? null : { content: '', is_error: ev.is_error, bytes: 0 } })) };
    case 'queue.changed':
      return { ...c, queue: ev.queue };
    case 'context.changed':
      return { ...c, context: ev.context };
    case 'compaction':
      return { ...c, compactions: c.compactions.some((x) => x.id === ev.compaction.id) ? c.compactions : [...c.compactions, ev.compaction] };
    case 'conversation.updated':
      return { ...c, ...ev.patch };
    case 'sandbox.changed':
      return { ...c, sandbox: ev.sandbox };
    case 'usage.changed':
      return { ...c, usage: ev.usage };
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

export function applyEvents(c: Conversation, evs: ChatEvent[]): Conversation {
  return evs.reduce(applyEvent, c);
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

/** The last assistant message with figures, for the Context panel when nothing runs. */
export function lastStats(c: Conversation) {
  for (let i = c.messages.length - 1; i >= 0; i--) {
    const m = c.messages[i];
    if (m.role === 'assistant' && m.stats) return m.stats;
  }
  return null;
}
