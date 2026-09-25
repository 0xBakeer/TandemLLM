// The request builder (VIS-20): the body posted to /v1/chat/completions. Always present: model,
// messages, stream (and stream_options unless switched off). Everything else only when the
// person set it — a field left at "default" is not sent, so the server's own flags decide (and
// a restart with new flags is followed without a click here).

import { PARAMS } from './params';
import type { Message, ParamValues, Setup, ToolChoice, ToolDef } from './types';

export interface ChatRequestBody {
  model: string;
  messages: WireMessage[];
  stream: boolean;
  /** absent when include_usage is off: the server's own placement (the finish chunk by default) */
  stream_options?: { include_usage: true };
  tools?: ToolDef[];
  tool_choice?: ToolChoice;
  chat_template_kwargs?: { enable_thinking: boolean };
  [k: string]: unknown;
}

export interface WireMessage {
  role: string;
  content: string;
  tool_calls?: Message['tool_calls'];
  tool_call_id?: string;
  name?: string;
}

/** The messages as the engine needs them: no client ids, no reasoning, no stats. */
export function wireMessages(system: string, messages: Message[]): WireMessage[] {
  const out: WireMessage[] = [];
  if (system.trim()) out.push({ role: 'system', content: system });
  for (const m of messages) {
    const w: WireMessage = { role: m.role, content: m.content };
    if (m.role === 'assistant' && m.tool_calls?.length) w.tool_calls = m.tool_calls.map((c) => ({ id: c.id, type: 'function', function: { name: c.function.name, arguments: c.function.arguments } }));
    if (m.role === 'tool') {
      if (m.tool_call_id) w.tool_call_id = m.tool_call_id;
      if (m.name) w.name = m.name;
    }
    out.push(w);
  }
  return out;
}

export interface BuildOptions {
  model: string;
  tools?: ToolDef[] | null;
  stream?: boolean;
}

export function paramFields(params: ParamValues): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const p of PARAMS) {
    const v = params[p.key];
    if (v === undefined) continue;
    if (p.key === 'thinking' || p.key === 'include_usage') continue; // nested, handled below
    if (p.key === 'stop') {
      const list = (v as string[]).filter((s) => s.length > 0);
      if (list.length) out.stop = list;
      continue;
    }
    out[p.request] = v;
  }
  return out;
}

export function buildRequest(setup: Setup, messages: Message[], opts: BuildOptions): ChatRequestBody {
  const body: ChatRequestBody = {
    model: opts.model,
    messages: wireMessages(setup.system, messages),
    stream: opts.stream ?? true,
  };
  // include_usage on (the default here) asks for the separate `choices: []` usage chunk; off
  // sends no stream_options at all, so the server's own placement applies (the finish chunk
  // by default -- what Open WebUI's base models get).
  if (setup.params.include_usage !== false) body.stream_options = { include_usage: true };
  Object.assign(body, paramFields(setup.params));
  if (setup.params.thinking !== undefined) body.chat_template_kwargs = { enable_thinking: setup.params.thinking };
  if (opts.tools && opts.tools.length) {
    body.tools = opts.tools;
    if (setup.tool_choice !== 'auto') body.tool_choice = setup.tool_choice;
  }
  return body;
}

/** A curl line that reproduces the request against `origin`. */
export function asCurl(body: ChatRequestBody, origin: string): string {
  const json = JSON.stringify(body, null, 2).replace(/'/g, `'\\''`);
  return `curl -N ${origin}/v1/chat/completions \\\n  -H 'Content-Type: application/json' \\\n  -d '${json}'`;
}
