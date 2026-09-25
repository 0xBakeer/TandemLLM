// VIS-19 / VIS-21 — the chat stream reducer: reasoning formats, split tags, tool-call deltas
// by index, usage taken once, the error field.
import { describe, expect, it } from 'vitest';
import { ChatStreamReducer } from '../src/playground/stream';

const delta = (d: Record<string, unknown>, finish: string | null = null, extra: Record<string, unknown> = {}) => ({ choices: [{ index: 0, delta: d, finish_reason: finish }], ...extra });

describe('reasoning formats', () => {
  it('tags: <think>…</think> in content goes to reasoning, the answer keeps no tag', () => {
    const r = new ChatStreamReducer();
    r.feed(delta({ role: 'assistant', content: '' }));
    r.feed(delta({ content: '<think>\nplan ' }));
    r.feed(delta({ content: 'it\n</think>\n\nHello ' }));
    r.feed(delta({ content: 'there' }));
    const s = r.end();
    expect(s.reasoning).toBe('plan it\n');
    expect(s.content).toBe('Hello there');
    expect(s.content).not.toContain('think');
    expect(s.answerStarted).toBe(true);
  });

  it('reasoning_content: the field is taken as is; content is pure', () => {
    const r = new ChatStreamReducer();
    r.feed(delta({ reasoning_content: 'a ' }));
    r.feed(delta({ reasoning_content: 'b' }));
    r.feed(delta({ content: 'answer' }));
    const s = r.end();
    expect(s.reasoning).toBe('a b');
    expect(s.content).toBe('answer');
  });

  it('both: the tagged copy is dropped so the reasoning is shown once', () => {
    const r = new ChatStreamReducer();
    r.feed(delta({ content: '<think>\nsame text\n</think>\n\n', reasoning_content: 'same text\n' }));
    r.feed(delta({ content: 'answer' }));
    const s = r.end();
    expect(s.reasoning).toBe('same text\n');
    expect(s.content).toBe('answer');
  });

  it('a tag split across chunks is still recognised, and a partial tail is released at the end', () => {
    const r = new ChatStreamReducer();
    r.feed(delta({ content: '<thi' }));
    expect(r.state.content).toBe('');
    r.feed(delta({ content: 'nk>reason</thi' }));
    r.feed(delta({ content: 'nk>\n\nans <' }));
    expect(r.state.content).toBe('ans ');
    const s = r.end();
    expect(s.reasoning).toBe('reason');
    expect(s.content).toBe('ans <');
  });

  it('answerStarted stays false while only the think block streams', () => {
    const r = new ChatStreamReducer();
    r.feed(delta({ content: '<think>\nstill thinking' }));
    expect(r.state.answerStarted).toBe(false);
    expect(r.state.inThink).toBe(true);
    r.feed(delta({ content: '\n</think>\n\nx' }));
    expect(r.state.answerStarted).toBe(true);
    expect(r.state.inThink).toBe(false);
  });
});

describe('usage, finish and errors', () => {
  it('usage on the finish chunk (default placement) is taken once', () => {
    const r = new ChatStreamReducer();
    r.feed(delta({ content: 'x' }));
    r.feed(delta({}, 'stop', { usage: { prompt_tokens: 1, completion_tokens: 2, total_tokens: 3 }, timings: { ttft_ms: 5 }, metrics: { tokens_per_second: 9 } }));
    const s = r.end();
    expect(s.finish).toBe('stop');
    expect(s.usage?.total_tokens).toBe(3);
    expect(s.timings?.ttft_ms).toBe(5);
    expect(s.metrics?.tokens_per_second).toBe(9);
    expect(s.chunks).toBe(2);
  });

  it('usage on a separate choices: [] chunk (include_usage) — a second block is ignored', () => {
    const r = new ChatStreamReducer();
    r.feed(delta({}, 'stop'));
    r.feed({ choices: [], usage: { prompt_tokens: 10, completion_tokens: 1, total_tokens: 11 } });
    r.feed({ choices: [], usage: { prompt_tokens: 99, completion_tokens: 99, total_tokens: 198 } });
    expect(r.end().usage?.prompt_tokens).toBe(10);
  });

  it('a failed finish chunk carries error alongside finish_reason', () => {
    const r = new ChatStreamReducer();
    r.feed(delta({ content: 'part' }));
    r.feed(delta({}, 'stop', { error: { message: 'CUDA out of memory', type: 'OutOfMemoryError' } }));
    const s = r.end();
    expect(s.error?.message).toBe('CUDA out of memory');
    expect(s.content).toBe('part');
  });
});

describe('tool calls', () => {
  it('accumulates argument deltas by index, first fragment carries id and name', () => {
    const r = new ChatStreamReducer();
    r.feed(delta({ tool_calls: [{ index: 0, id: 'call_a', type: 'function', function: { name: 'get_weather', arguments: '' } }] }));
    r.feed(delta({ tool_calls: [{ index: 0, function: { arguments: '{"city": "' } }] }));
    r.feed(delta({ tool_calls: [{ index: 0, function: { arguments: 'Berlin"}' } }] }));
    r.feed(delta({}, 'tool_calls'));
    const s = r.end();
    expect(s.toolCalls).toEqual([{ id: 'call_a', type: 'function', function: { name: 'get_weather', arguments: '{"city": "Berlin"}' } }]);
    expect(s.finish).toBe('tool_calls');
  });

  it('interleaved fragments for two indexes build two calls in index order', () => {
    const r = new ChatStreamReducer();
    r.feed(delta({ tool_calls: [{ index: 1, id: 'b', type: 'function', function: { name: 'second', arguments: '' } }] }));
    r.feed(delta({ tool_calls: [{ index: 0, id: 'a', type: 'function', function: { name: 'first', arguments: '{' } }] }));
    r.feed(delta({ tool_calls: [{ index: 1, function: { arguments: '{"q":1}' } }, { index: 0, function: { arguments: '}' } }] }));
    const s = r.end();
    expect(s.toolCalls.map((c) => c.function.name)).toEqual(['first', 'second']);
    expect(s.toolCalls[0].function.arguments).toBe('{}');
    expect(s.toolCalls[1].function.arguments).toBe('{"q":1}');
  });

  it('a sweep-only call arrives complete in one fragment', () => {
    const r = new ChatStreamReducer();
    r.feed(delta({ tool_calls: [{ index: 0, id: 'c', type: 'function', function: { name: 'run_python', arguments: '{"code": "print(1)"}' } }] }));
    expect(r.end().toolCalls[0].function.arguments).toBe('{"code": "print(1)"}');
  });

  it('arguments that never parse stay raw text', () => {
    const r = new ChatStreamReducer();
    r.feed(delta({ tool_calls: [{ index: 0, id: 'c', type: 'function', function: { name: 'f', arguments: '{"broken' } }] }));
    expect(r.end().toolCalls[0].function.arguments).toBe('{"broken');
  });
});
