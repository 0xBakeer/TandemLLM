// The chat stream reducer (VIS-19): OpenAI chunks in, one growing turn out. It does not need to
// know the reasoning format: `<think>…</think>` in `content` (tags) is split off, a
// `reasoning_content` delta is taken as is, and when both arrive (format `both`) the tagged copy
// is dropped so nothing shows twice. Tool-call fragments accumulate by `index`; the usage block
// is taken from exactly one chunk (the finish chunk or the separate `choices: []` chunk).

import type { ChatMetrics, ChatTimings, ChatUsage } from '../api/types';
import type { ToolCall } from './types';

export interface ChatChunk {
  choices?: {
    index?: number;
    delta?: { role?: string; content?: string | null; reasoning_content?: string | null; tool_calls?: ToolCallDelta[] };
    finish_reason?: string | null;
  }[];
  usage?: ChatUsage;
  timings?: ChatTimings;
  metrics?: ChatMetrics;
  error?: { message: string; type?: string };
}

export interface ToolCallDelta {
  index: number;
  id?: string;
  type?: 'function';
  function?: { name?: string; arguments?: string };
}

export interface TurnState {
  content: string;
  reasoning: string;
  toolCalls: ToolCall[];
  finish: string | null;
  usage?: ChatUsage;
  timings?: ChatTimings;
  metrics?: ChatMetrics;
  error?: { message: string; type?: string } | null;
  chunks: number;
  /** true once answer text (outside a think block) has arrived */
  answerStarted: boolean;
  /** true while the tagged block is open */
  inThink: boolean;
}

const OPEN = '<think>';
const CLOSE = '</think>';

export class ChatStreamReducer {
  readonly state: TurnState = { content: '', reasoning: '', toolCalls: [], finish: null, error: null, chunks: 0, answerStarted: false, inThink: false };
  private buf = '';
  private tagged = '';
  private field = '';
  private usageTaken = false;
  private calls = new Map<number, ToolCall>();

  feed(raw: unknown): void {
    const c = raw as ChatChunk;
    this.state.chunks++;
    const ch = c.choices?.[0];
    const d = ch?.delta;
    if (d?.reasoning_content) {
      this.field += d.reasoning_content;
      this.state.reasoning = this.field;
    }
    if (d?.content) {
      this.buf += d.content;
      this.drain(false);
    }
    if (d?.tool_calls) for (const t of d.tool_calls) this.toolDelta(t);
    if (ch?.finish_reason) this.state.finish = ch.finish_reason;
    if (!this.usageTaken && (c.usage || c.timings || c.metrics)) {
      this.usageTaken = true;
      if (c.usage) this.state.usage = c.usage;
      if (c.timings) this.state.timings = c.timings;
      if (c.metrics) this.state.metrics = c.metrics;
    }
    if (c.error) this.state.error = c.error;
  }

  /** The stream ended (or was aborted): release what a possible partial tag was holding. */
  end(): TurnState {
    this.drain(true);
    return this.state;
  }

  private toolDelta(t: ToolCallDelta): void {
    const i = typeof t.index === 'number' ? t.index : this.calls.size;
    let call = this.calls.get(i);
    if (!call) {
      call = { id: t.id ?? '', type: 'function', function: { name: t.function?.name ?? '', arguments: '' } };
      this.calls.set(i, call);
    } else {
      if (t.id && !call.id) call.id = t.id;
      if (t.function?.name && !call.function.name) call.function.name = t.function.name;
    }
    if (t.function?.arguments) call.function.arguments += t.function.arguments;
    this.state.toolCalls = [...this.calls.entries()].sort((a, b) => a[0] - b[0]).map((e) => e[1]);
  }

  /** Route the content buffer: the tagged block to `tagged`, the rest to `content`. */
  private drain(final: boolean): void {
    for (;;) {
      if (this.state.inThink) {
        const j = this.buf.indexOf(CLOSE);
        if (j < 0) {
          const keep = final ? 0 : partialTail(this.buf, CLOSE);
          this.tagged += this.buf.slice(0, this.buf.length - keep);
          this.buf = this.buf.slice(this.buf.length - keep);
          break;
        }
        this.tagged += this.buf.slice(0, j);
        this.buf = this.buf.slice(j + CLOSE.length);
        this.state.inThink = false;
        // the model writes "\n\n" after the block; that is not part of the answer
        this.buf = this.buf.replace(/^\n{1,2}/, '');
      } else {
        const i = this.buf.indexOf(OPEN);
        if (i < 0) {
          const keep = final ? 0 : partialTail(this.buf, OPEN);
          this.emit(this.buf.slice(0, this.buf.length - keep));
          this.buf = this.buf.slice(this.buf.length - keep);
          break;
        }
        this.emit(this.buf.slice(0, i));
        this.buf = this.buf.slice(i + OPEN.length).replace(/^\n/, '');
        this.state.inThink = true;
      }
    }
    // The field copy wins when both exist (format `both`); the tagged copy is the fallback.
    this.state.reasoning = this.field.length ? this.field : this.tagged;
  }

  private emit(text: string): void {
    if (!text) return;
    this.state.content += text;
    if (this.state.content.trim()) this.state.answerStarted = true;
  }
}

/** How many trailing characters of `s` could still grow into `tag`. */
function partialTail(s: string, tag: string): number {
  const max = Math.min(s.length, tag.length - 1);
  for (let n = max; n > 0; n--) if (tag.startsWith(s.slice(s.length - n))) return n;
  return 0;
}
