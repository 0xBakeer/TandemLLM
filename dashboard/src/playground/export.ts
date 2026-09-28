// The readout figures and the conversation exports (JSON, Markdown).

import { exact, fixed, ms, pct } from '../lib/format';
import type { Message, Setup, TurnStats } from './types';

export interface Figure {
  id: string;
  label: string;
  value: string;
  unit?: string;
  title?: string;
  /** shown on the per-turn line too */
  key?: boolean;
  na?: boolean;
}

const NA = 'not reported';

/** The figures of a turn, in readout order. Missing families say "not reported", never 0. */
export function readoutOf(s: TurnStats | undefined | null): Figure[] {
  const u = s?.usage;
  const t = s?.timings;
  const m = s?.metrics;
  const cached = u?.prompt_tokens_details?.cached_tokens ?? t?.cache_n;
  const reasoning = u?.completion_tokens_details?.reasoning_tokens ?? t?.reasoning_n;
  const ttft = t?.ttft_ms ?? m?.time_to_first_token_ms;
  const decode = t?.predicted_per_second ?? m?.tokens_per_second;
  const acc = t?.draft_n ? (t.draft_n_accepted ?? 0) / t.draft_n : m?.speculative_decoding?.draft_acceptance_rate;
  const num = (id: string, label: string, v: number | null | undefined, f: (x: number) => string, unit?: string, key = false, title?: string): Figure =>
    v == null || !Number.isFinite(v) ? { id, label, value: NA, key, na: true } : { id, label, value: f(v), unit, key, title };
  return [
    num('prompt', 'prompt', u?.prompt_tokens ?? t?.prompt_n, exact, 'tok'),
    num('cached', 'cached', cached, exact, 'tok', false, 'prompt tokens served from a cache'),
    num('completion', 'completion', u?.completion_tokens ?? t?.predicted_n, exact, 'tok'),
    num('reasoning', 'reasoning', reasoning, exact, 'tok'),
    num('ttft', 'time to first token', ttft, (x) => ms(x), undefined, true),
    num('queue', 'queue', t?.queue_ms ?? m?.queue_time_ms, (x) => ms(x)),
    num('prefill', 'prefill', t?.prompt_per_second, (x) => (x >= 100 ? exact(Math.round(x)) : fixed(x, 1)), 'tok/s'),
    num('decode', 'decode', decode, (x) => fixed(x, 1), 'tok/s', true),
    num('total', 'total', t?.total_ms, (x) => ms(x)),
    num('tpb', 'tokens per block', t?.tokens_per_block ?? m?.speculative_decoding?.mean_acceptance_length, (x) => fixed(x, 2), 'tok/blk', true),
    num('acceptance', 'draft acceptance', acc, (x) => pct(x, 0)),
    num('blocks', 'blocks', t?.blocks, exact),
    { id: 'finish', label: 'finish', value: s?.finish ?? '—', key: true },
    { id: 'cache', label: 'cache', value: t?.cache_source ?? NA, na: !t?.cache_source },
  ];
}

/** "410 ms · 38.2 tok/s · 4.30 tok/blk · stop" — the line under an assistant turn. */
export function readoutLine(s: TurnStats | undefined | null): string {
  const f = readoutOf(s).filter((x) => x.key);
  return f.map((x) => (x.na ? `${x.label} ${x.value}` : `${x.value}${x.unit ? ' ' + x.unit : ''}`)).join(' · ');
}

export interface ConversationExport {
  format: 'qse-playground-conversation';
  version: 1;
  exportedAt: string;
  model: string | null;
  setup: Setup;
  messages: Omit<Message, 'uid'>[];
  turns: { index: number; finish: string | null; usage?: TurnStats['usage']; timings?: TurnStats['timings']; metrics?: TurnStats['metrics']; error?: TurnStats['error'] }[];
}

export function toJsonExport(setup: Setup, messages: Message[], model: string | null, now = new Date()): string {
  const doc: ConversationExport = {
    format: 'qse-playground-conversation',
    version: 1,
    exportedAt: now.toISOString(),
    model,
    setup,
    messages: messages.map(({ uid: _u, stats: _s, ...m }) => m),
    turns: messages.map((m, index) => ({ m, index })).filter((x) => x.m.role === 'assistant' && x.m.stats).map(({ m, index }) => ({ index, finish: m.stats!.finish, usage: m.stats!.usage, timings: m.stats!.timings, metrics: m.stats!.metrics, error: m.stats!.error })),
  };
  return JSON.stringify(doc, null, 2);
}

export function toMarkdownExport(setup: Setup, messages: Message[], model: string | null, now = new Date()): string {
  const out: string[] = [`# Playground conversation`, '', `${model ?? 'the engine'} · ${now.toISOString()}`, ''];
  if (setup.system.trim()) out.push('## system', '', setup.system.trim(), '');
  for (const m of messages) {
    out.push(`## ${m.role}${m.name ? ` (${m.name})` : ''}`, '');
    if (m.reasoning?.trim()) out.push(...m.reasoning.trim().split('\n').map((l) => `> ${l}`), '');
    if (m.content.trim()) out.push(m.content.trim(), '');
    if (m.tool_calls?.length) for (const c of m.tool_calls) out.push(`\`\`\`json`, JSON.stringify({ tool_call: c.function.name, id: c.id, arguments: safeParse(c.function.arguments) }, null, 2), '```', '');
    if (m.role === 'assistant' && m.stats) out.push(`_${readoutLine(m.stats)}_`, '');
  }
  return out.join('\n');
}

function safeParse(s: string): unknown {
  try {
    return JSON.parse(s);
  } catch {
    return s;
  }
}

export function exportFileName(ext: 'json' | 'md', now = new Date()): string {
  const p = (n: number) => String(n).padStart(2, '0');
  return `playground-${now.getFullYear()}${p(now.getMonth() + 1)}${p(now.getDate())}-${p(now.getHours())}${p(now.getMinutes())}.${ext}`;
}
