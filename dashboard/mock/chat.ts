// The mock chat engine: a scripted /v1/chat/completions that honours what the
// Playground sends — thinking on/off, the three reasoning formats, max_tokens (a `length`
// finish), stop strings, include_usage placement, tools + tool_choice (a user turn → a streamed
// call with argument deltas and finish_reason tool_calls; a tool turn → an answer that quotes the
// result), an abort (`abandoned` in the ledger) and the busy switch (503 + Retry-After). The
// answer is Markdown with a fenced code block so the renderer is exercised. Never in dist/.

import type { IncomingMessage, ServerResponse } from 'node:http';
import { randomBytes } from 'node:crypto';
import type { ChatMetrics, ChatTimings, ChatUsage } from '../src/api/types.ts';
import { makeRow, type LedgerRow } from './generate.ts';
import type { MockLive } from './live.ts';

export interface ChatContext {
  mode: () => string;
  draining: () => boolean;
  rng: () => number;
  /** the live engine state: running / generationTokens are moved while streaming */
  state: { running: number; generationTokens: number };
  /** the live registry: the request is visible while it streams */
  live?: MockLive;
  finishRow: (row: LedgerRow) => void;
  readBody: (req: IncomingMessage) => Promise<string>;
}

interface Msg {
  role: string;
  content?: string | null;
  name?: string;
  tool_calls?: { id: string; function: { name: string; arguments: string } }[];
}
interface ToolDef {
  type: string;
  function: { name: string; parameters?: { properties?: Record<string, { type?: string }> } };
}

interface Script {
  reasoning: string;
  answer: string;
  call: { name: string; args: string } | null;
}

const MODEL = 'qwen38-spark-engine';

/** What the mock says for a given transcript. */
export function scriptFor(body: Record<string, unknown>): Script {
  const messages = (body.messages as Msg[] | undefined) ?? [];
  const last = messages[messages.length - 1];
  const lastUser = [...messages].reverse().find((m) => m.role === 'user');
  const userText = String(lastUser?.content ?? '');
  const toolChoice = body.tool_choice as string | { function?: { name?: string } } | undefined;
  const tools = toolChoice === 'none' ? [] : ((body.tools as ToolDef[] | undefined) ?? []).filter((t) => t?.type === 'function' && t.function?.name);

  if (last?.role === 'tool') {
    const name = last.name ?? 'the tool';
    const content = String(last.content ?? '');
    let summary = 'that is the result, as returned.';
    try {
      const o = JSON.parse(content) as Record<string, unknown>;
      if (typeof o.temperature_c === 'number') summary = `it is ${o.temperature_c} °C${o.condition ? ` and ${o.condition}` : ''}${typeof o.wind_kmh === 'number' ? `, wind ${o.wind_kmh} km/h` : ''}.`;
      else if (Array.isArray(o.hits)) summary = `${o.hits.length} note${o.hits.length === 1 ? '' : 's'} match: ${(o.hits as { title?: string }[]).map((h) => h.title).filter(Boolean).join(', ')}.`;
      else if (typeof o.stdout === 'string') summary = `the program printed \`${o.stdout.trim()}\`.`;
      else if (o.ok === true) summary = 'the call succeeded.';
    } catch {
      /* not JSON */
    }
    return {
      reasoning: `The ${name} result is in. Summarise it for the user in one line and quote it.`,
      answer: `\`${name}\` answered:\n\n\`\`\`json\n${content}\n\`\`\`\n\nIn short: ${summary}`,
      call: null,
    };
  }

  if (tools.length) {
    const named = typeof toolChoice === 'object' && toolChoice?.function?.name ? tools.find((t) => t.function.name === toolChoice.function!.name) : undefined;
    const tool = named ?? tools[0];
    const props = tool.function.parameters?.properties ?? {};
    const args: Record<string, unknown> = {};
    for (const [k, p] of Object.entries(props)) {
      if (k === 'city') args.city = /\bin ([A-Z][\w-]+)/.exec(userText)?.[1] ?? 'Berlin';
      else if (k === 'query') args.query = userText.replace(/[?.!]+$/, '') || 'engine';
      else if (k === 'code') args.code = 'print(6 * 7)';
      else if (k === 'limit') args.limit = 5;
      else if (k === 'unit') continue;
      else if (p?.type === 'integer' || p?.type === 'number') args[k] = 1;
      else if (p?.type === 'boolean') args[k] = true;
      else if (p?.type === 'string') args[k] = 'example';
    }
    return {
      reasoning: `The user asks: "${userText.trim() || '…'}". The ${tool.function.name} tool covers exactly this, so call it rather than guess.`,
      answer: '',
      call: { name: tool.function.name, args: JSON.stringify(args) },
    };
  }

  if (/^\s*say hi\b/i.test(userText)) {
    return {
      reasoning: 'The user wants a short greeting. A friendly one-line answer with no extras is right here; nothing to look up, nothing to compute.',
      answer: 'Hi! The engine is up, the drafter is warm, and this reply came through the same queue as everyone else — ask away.',
      call: null,
    };
  }

  const topic = userText.trim().replace(/\s+/g, ' ').slice(0, 80) || 'nothing in particular';
  return {
    reasoning: `The user asks about "${topic}". Give the mechanism first, then one worked example in code, then the two numbers that decide it. Keep it under two hundred words.`,
    answer: [
      `## ${topic.replace(/[?.!]+$/, '')}`,
      '',
      `Here is the short version. The engine verifies a **block** of drafted tokens in one forward pass; the number it keeps per block is what the readout below calls *tokens per block*. More kept per block means fewer passes per answer, so the decode rate rises with it.`,
      '',
      '```python',
      'def tokens_per_block(accepted: list[int]) -> float:',
      '    """Mean accepted tokens per verify pass, +1 for the target token."""',
      '    return sum(accepted) / len(accepted) + 1',
      '',
      'print(tokens_per_block([3, 4, 2, 5]))  # 4.5',
      '```',
      '',
      '- **Acceptance** is the share of drafted tokens the target agrees with.',
      '- **Tokens per block** is the mean run length that survives, plus one.',
      '- The drafter loses first on code with unusual identifiers; prose stays high.',
      '',
      `That is the whole mechanism; the rest is tuning the block size against the acceptance curve you measure on your own text.`,
    ].join('\n'),
    call: null,
  };
}

export function createChatHandler(ctx: ChatContext) {
  const json = (res: ServerResponse, code: number, body: unknown, headers: Record<string, string> = {}) => {
    res.writeHead(code, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store', ...headers });
    res.end(JSON.stringify(body));
  };
  const delay = (ms: number) => new Promise((r) => setTimeout(r, ms));

  return async (req: IncomingMessage, res: ServerResponse): Promise<void> => {
    const raw = await ctx.readBody(req);
    let body: Record<string, unknown> = {};
    try {
      body = raw ? JSON.parse(raw) : {};
    } catch {
      return json(res, 400, { error: { message: 'bad json', type: 'invalid_request_error' } });
    }
    const mode = ctx.mode();
    if (mode === 'busy' || ctx.draining()) {
      return json(res, 503, { error: { message: ctx.draining() ? 'draining' : 'queue full (8 waiting)', type: 'server_busy', code: 503 } }, { 'Retry-After': '5' });
    }
    if (typeof body.temperature === 'number' && body.temperature > 2) {
      return json(res, 400, { error: { message: `temperature must be in [0, 2], got ${body.temperature}`, type: 'invalid_request_error', param: 'temperature' } });
    }
    if (body.tool_choice === 'required' && !(Array.isArray(body.tools) && body.tools.length)) {
      return json(res, 400, { error: { message: 'tool_choice "required" needs at least one tool', type: 'invalid_request_error', param: 'tool_choice' } });
    }
    const fmt = String(body.reasoning_format ?? 'tags');
    if (!['tags', 'reasoning_content', 'both'].includes(fmt)) {
      return json(res, 400, { error: { message: `reasoning_format must be one of ('tags', 'reasoning_content', 'both'), not '${fmt}'`, type: 'invalid_request_error', param: 'reasoning_format' } });
    }

    const stream = body.stream === true;
    const so = (body.stream_options as Record<string, unknown> | undefined) ?? undefined;
    const includeUsage = so?.include_usage;
    const kwargs = (body.chat_template_kwargs as Record<string, unknown> | undefined) ?? {};
    const thinking = kwargs.enable_thinking !== false;
    const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : 32768;
    const stops = Array.isArray(body.stop) ? (body.stop as string[]) : typeof body.stop === 'string' ? [body.stop] : [];
    const ua = String(req.headers['user-agent'] ?? '');
    const dash = ua.includes('qse-dashboard') || String(req.headers['x-requested-with'] ?? '') === 'qse-dashboard';
    const created = Math.floor(Date.now() / 1000);
    const id = 'chatcmpl-' + randomBytes(8).toString('hex');

    const script = scriptFor(body);
    const reasoningText = thinking ? script.reasoning : '';
    let answerText = script.answer;
    let finish: 'stop' | 'length' | 'tool_calls' = script.call ? 'tool_calls' : 'stop';
    for (const s of stops) {
      if (!s) continue;
      const i = answerText.indexOf(s);
      if (i >= 0) {
        answerText = answerText.slice(0, i);
        finish = 'stop';
      }
    }
    // "tokens" are words here; the budget cuts the answer and the finish says so
    const reasoningWords = reasoningText ? reasoningText.split(' ') : [];
    let answerWords = answerText ? answerText.split(' ') : [];
    const rTok = Math.min(maxTokens, reasoningWords.length);
    if (reasoningWords.length + answerWords.length > maxTokens) {
      answerWords = answerWords.slice(0, Math.max(0, maxTokens - rTok));
      if (!script.call) finish = 'length';
    }
    const argFrags = script.call ? chunkString(script.call.args, 6) : [];
    const completion = rTok + answerWords.length + argFrags.length + (script.call ? 2 : 0);
    const prompt = 38 + Math.round(JSON.stringify(body.messages ?? []).length / 4) + (Array.isArray(body.tools) ? body.tools.length * 60 : 0);
    const cached = prompt > 400 ? Math.floor(prompt * 0.6) : 0;
    const queueMs = 0.4;
    const promptMs = Math.round((prompt - cached) * 0.34 * 10) / 10 + 28;
    const tps = 60 + ctx.rng() * 12;
    const perTok = mode === 'slow' ? 150 : 1000 / tps;
    const predictedMs = Math.max(1, completion - 1) * (1000 / tps);
    const blocks = Math.max(1, Math.ceil(Math.max(1, completion - 1) / 4.3));
    const draft = blocks * 15;
    const accepted = Math.round(draft * 0.24);

    const row = makeRow(ctx.rng, Date.now(), false, {
      request_id: id,
      client_id: dash ? 'k:0b7e55c3d1f2' : 'anon',
      client_kind: dash ? 'dashboard' : 'curl',
      endpoint: 'chat',
      stream,
      status: 200,
      finish_reason: finish,
      prompt_tokens: prompt,
      cached_tokens: cached,
      completion_tokens: completion,
      reasoning_tokens: rTok,
      queue_ms: queueMs,
      prompt_ms: promptMs,
      ttft_ms: queueMs + promptMs,
      decode_ms: Math.round(predictedMs * 100) / 100,
      total_ms: Math.round((queueMs + promptMs + predictedMs) * 100) / 100,
      decode_tps: Math.round(tps * 100) / 100,
      prefill_tps: Math.round(((prompt - cached) / promptMs) * 1000 * 10) / 10,
      blocks,
      draft_tokens: draft,
      draft_accepted: accepted,
      tool_calls: script.call ? 1 : 0,
      thinking,
      cache_source: cached ? 'prefix' : 'none',
      max_tokens: maxTokens,
      error_type: null,
    });
    const usage: ChatUsage = {
      prompt_tokens: prompt,
      completion_tokens: completion,
      total_tokens: prompt + completion,
      prompt_tokens_details: { cached_tokens: cached },
      completion_tokens_details: { reasoning_tokens: rTok },
    };
    const timings: ChatTimings = {
      cache_n: cached,
      prompt_n: prompt,
      prompt_ms: promptMs,
      prompt_per_token_ms: Math.round((promptMs / Math.max(1, prompt - cached)) * 100) / 100,
      prompt_per_second: Math.round(((prompt - cached) / promptMs) * 1000 * 100) / 100,
      predicted_n: completion,
      predicted_ms: Math.round(predictedMs * 100) / 100,
      predicted_per_token_ms: Math.round((predictedMs / Math.max(1, completion)) * 100) / 100,
      predicted_per_second: Math.round(tps * 100) / 100,
      draft_n: draft,
      draft_n_accepted: accepted,
      ttft_ms: Math.round((queueMs + promptMs) * 100) / 100,
      queue_ms: queueMs,
      total_ms: row.total_ms ?? 0,
      blocks,
      tokens_per_block: Math.round((Math.max(1, completion - 1) / blocks) * 100) / 100,
      reasoning_n: rTok,
      cache_source: cached ? 'prefix' : 'none',
    };
    const metrics: ChatMetrics = {
      time_to_first_token_ms: timings.ttft_ms,
      generation_time_ms: timings.predicted_ms,
      queue_time_ms: queueMs,
      mean_itl_ms: timings.predicted_per_token_ms,
      tokens_per_second: Math.round((completion / (row.total_ms ?? 1)) * 1000 * 100) / 100,
      speculative_decoding: { mean_acceptance_length: timings.tokens_per_block, draft_acceptance_rate: Math.round((accepted / draft) * 10000) / 10000 },
    };
    const callId = 'call_' + randomBytes(12).toString('hex');
    const tagged = reasoningText ? `<think>\n${reasoningText}\n</think>\n\n` : '';
    const lr = ctx.live?.begin(row);
    if (lr) ctx.live!.lock(lr);

    if (!stream) {
      await delay(200);
      if (lr) {
        ctx.live!.first(lr, Date.now() - 100);
        ctx.live!.finish(lr, finish, 200, Date.now(), completion);
      }
      const content = (fmt === 'tags' || fmt === 'both' ? tagged : '') + answerWords.join(' ');
      const message: Record<string, unknown> = { role: 'assistant', content };
      if (fmt !== 'tags' && reasoningText) message.reasoning_content = reasoningText;
      if (script.call) message.tool_calls = [{ id: callId, type: 'function', function: { name: script.call.name, arguments: script.call.args } }];
      ctx.finishRow(row);
      return json(res, 200, { id, object: 'chat.completion', created, model: MODEL, choices: [{ index: 0, message, finish_reason: finish, logprobs: null }], usage, timings, metrics });
    }

    res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', Connection: 'keep-alive' });
    ctx.state.running = 1;
    const chunk = (delta: Record<string, unknown> | null, fin: string | null = null, extra: Record<string, unknown> = {}) =>
      res.write(`data: ${JSON.stringify({ id, object: 'chat.completion.chunk', created, model: MODEL, choices: delta === null && fin === null ? [] : [{ index: 0, delta: delta ?? {}, finish_reason: fin, logprobs: null }], ...extra })}\n\n`);
    let closed = false;
    req.on('close', () => (closed = true));
    await delay(queueMs + promptMs);
    if (lr) ctx.live!.first(lr);
    chunk({ role: 'assistant', content: '' });
    if (reasoningText) {
      if (fmt === 'tags' || fmt === 'both') chunk({ content: '<think>\n' });
      for (const w of reasoningWords) {
        if (closed) break;
        const piece = w + ' ';
        if (fmt === 'tags') chunk({ content: piece });
        else if (fmt === 'reasoning_content') chunk({ reasoning_content: piece });
        else chunk({ content: piece, reasoning_content: piece });
        ctx.state.generationTokens++;
        if (lr) ctx.live!.token(lr);
        await delay(perTok);
      }
      if (fmt === 'tags' || fmt === 'both') chunk({ content: '\n</think>\n\n' });
    }
    for (const w of answerWords) {
      if (closed) break;
      chunk({ content: w + ' ' });
      ctx.state.generationTokens++;
      if (lr) ctx.live!.token(lr);
      await delay(perTok);
    }
    if (script.call && !closed) {
      chunk({ tool_calls: [{ index: 0, id: callId, type: 'function', function: { name: script.call.name, arguments: '' } }] });
      for (const f of argFrags) {
        if (closed) break;
        chunk({ tool_calls: [{ index: 0, function: { arguments: f } }] });
        ctx.state.generationTokens++;
        if (lr) ctx.live!.token(lr);
        await delay(perTok * 2);
      }
    }
    ctx.state.running = 0;
    if (closed) {
      row.finish_reason = 'abandoned';
      if (lr) ctx.live!.finish(lr, 'abandoned', 200, Date.now(), lr.tokens);
      ctx.finishRow(row);
      return;
    }
    if (lr) ctx.live!.finish(lr, finish, 200, Date.now(), completion);
    if (includeUsage === true) {
      chunk({}, finish);
      chunk(null, null, { usage, timings, metrics });
    } else {
      chunk({}, finish, includeUsage === false ? {} : { usage, timings, metrics });
    }
    res.write('data: [DONE]\n\n');
    res.end();
    ctx.state.generationTokens -= completion; // applyRow adds the whole completion
    ctx.finishRow(row);
  };
}

function chunkString(s: string, n: number): string[] {
  const out: string[] = [];
  for (let i = 0; i < s.length; i += n) out.push(s.slice(i, i + n));
  return out;
}
