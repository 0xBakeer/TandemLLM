// the request builder sends only what was changed.
import { describe, expect, it } from 'vitest';
import { asCurl, buildRequest, wireMessages } from '../src/playground/request';
import { changedKeys, defaultsFrom, validateParams, PARAMS } from '../src/playground/params';
import { emptySetup, type Message } from '../src/playground/types';

const msgs: Message[] = [{ uid: 'a', role: 'user', content: 'hi' }];

describe('buildRequest', () => {
  it('nothing changed → only model, messages, stream, stream_options', () => {
    const b = buildRequest(emptySetup(), msgs, { model: 'm' });
    expect(Object.keys(b).sort()).toEqual(['messages', 'model', 'stream', 'stream_options']);
    expect(b.stream).toBe(true);
    expect(b.stream_options).toEqual({ include_usage: true });
    expect(b.messages).toEqual([{ role: 'user', content: 'hi' }]);
  });

  it('every field maps to its request name; unset ones are absent', () => {
    const s = emptySetup();
    s.params = { temperature: 0.7, max_tokens: 128 };
    const b = buildRequest(s, msgs, { model: 'm' });
    expect(b.temperature).toBe(0.7);
    expect(b.max_tokens).toBe(128);
    for (const k of ['top_p', 'top_k', 'seed', 'presence_penalty', 'frequency_penalty', 'repetition_penalty', 'no_repeat_ngram_size', 'reasoning_effort', 'reasoning_format', 'draft_temperature', 'max_reasoning_tokens', 'chat_template_kwargs', 'tools', 'tool_choice']) expect(b).not.toHaveProperty(k);
    s.params = { top_p: 0.9, top_k: 40, seed: 7, presence_penalty: 0.5, frequency_penalty: -0.2, repetition_penalty: 1.1, no_repeat_ngram_size: 3, reasoning_effort: 'low', reasoning_format: 'both', draft_temperature: 0.3, max_reasoning_tokens: 512 };
    const c = buildRequest(s, msgs, { model: 'm' });
    expect(c).toMatchObject({ top_p: 0.9, top_k: 40, seed: 7, presence_penalty: 0.5, frequency_penalty: -0.2, repetition_penalty: 1.1, no_repeat_ngram_size: 3, reasoning_effort: 'low', reasoning_format: 'both', draft_temperature: 0.3, max_reasoning_tokens: 512 });
  });

  it('reset (undefined) removes a field again', () => {
    const s = emptySetup();
    s.params = { temperature: 0.7 };
    expect(buildRequest(s, msgs, { model: 'm' }).temperature).toBe(0.7);
    s.params = { temperature: undefined };
    expect(buildRequest(s, msgs, { model: 'm' })).not.toHaveProperty('temperature');
    expect(changedKeys(s.params)).toEqual([]);
  });

  it('thinking: untouched sends nothing, off sends false, on sends true', () => {
    const s = emptySetup();
    expect(buildRequest(s, msgs, { model: 'm' })).not.toHaveProperty('chat_template_kwargs');
    s.params = { thinking: false };
    expect(buildRequest(s, msgs, { model: 'm' }).chat_template_kwargs).toEqual({ enable_thinking: false });
    s.params = { thinking: true };
    expect(buildRequest(s, msgs, { model: 'm' }).chat_template_kwargs).toEqual({ enable_thinking: true });
  });

  it('stop is an array of the non-empty strings; an all-empty list is not sent', () => {
    const s = emptySetup();
    s.params = { stop: ['###', '', 'END'] };
    expect(buildRequest(s, msgs, { model: 'm' }).stop).toEqual(['###', 'END']);
    s.params = { stop: [''] };
    expect(buildRequest(s, msgs, { model: 'm' })).not.toHaveProperty('stop');
  });

  it('include_usage off sends no stream_options (the server\u2019s own placement)', () => {
    const s = emptySetup();
    s.params = { include_usage: false };
    const b = buildRequest(s, msgs, { model: 'm' });
    expect(b).not.toHaveProperty('stream_options');
    expect(Object.keys(b).sort()).toEqual(['messages', 'model', 'stream']);
  });

  it('tools and tool_choice: auto is implicit, the others are sent; no tools → neither', () => {
    const s = emptySetup();
    const tools = [{ type: 'function' as const, function: { name: 'f' } }];
    let b = buildRequest(s, msgs, { model: 'm', tools });
    expect(b.tools).toEqual(tools);
    expect(b).not.toHaveProperty('tool_choice');
    s.tool_choice = 'required';
    expect(buildRequest(s, msgs, { model: 'm', tools }).tool_choice).toBe('required');
    s.tool_choice = { type: 'function', function: { name: 'f' } };
    expect(buildRequest(s, msgs, { model: 'm', tools }).tool_choice).toEqual({ type: 'function', function: { name: 'f' } });
    b = buildRequest(s, msgs, { model: 'm', tools: [] });
    expect(b).not.toHaveProperty('tools');
    expect(b).not.toHaveProperty('tool_choice');
  });

  it('the system prompt is the first message only when non-empty', () => {
    const s = emptySetup();
    s.system = '  ';
    expect(buildRequest(s, msgs, { model: 'm' }).messages[0].role).toBe('user');
    s.system = 'Be terse.';
    expect(buildRequest(s, msgs, { model: 'm' }).messages[0]).toEqual({ role: 'system', content: 'Be terse.' });
  });
});

describe('wireMessages', () => {
  it('keeps tool_calls on the assistant turn and tool_call_id/name on the tool turn; drops uid, reasoning, stats', () => {
    const m: Message[] = [
      { uid: '1', role: 'user', content: 'weather?' },
      { uid: '2', role: 'assistant', content: '', reasoning: 'think', stats: { finish: 'tool_calls' }, tool_calls: [{ id: 'call_1', type: 'function', function: { name: 'get_weather', arguments: '{"city":"Berlin"}' } }] },
      { uid: '3', role: 'tool', tool_call_id: 'call_1', name: 'get_weather', content: '{"temperature_c":21}' },
    ];
    expect(wireMessages('', m)).toEqual([
      { role: 'user', content: 'weather?' },
      { role: 'assistant', content: '', tool_calls: [{ id: 'call_1', type: 'function', function: { name: 'get_weather', arguments: '{"city":"Berlin"}' } }] },
      { role: 'tool', content: '{"temperature_c":21}', tool_call_id: 'call_1', name: 'get_weather' },
    ]);
  });
});

describe('defaults and validation', () => {
  it('reads the server flags into defaults, 0 think_budget = unlimited', () => {
    const d = defaultsFrom({ engine: { max_len: 262144 }, flags: { args: { temperature: 0.0, top_p: 1.0, top_k: 0, rep_penalty: 1.0, presence_penalty: 0.0, frequency_penalty: 0.0, no_repeat_ngram: 0, default_max_tokens: 32768, reasoning_format: 'tags', reasoning_effort: 'medium', think_budget: 0 }, env: {} } } as never);
    expect(d).toMatchObject({ temperature: 0, top_p: 1, top_k: 0, repetition_penalty: 1, max_tokens: 32768, reasoning_format: 'tags', reasoning_effort: 'medium', max_len: 262144, thinking: true, include_usage: true });
    expect(d.max_reasoning_tokens).toBeNull();
    expect(defaultsFrom(null)).toEqual({});
  });

  it('validates ranges: temperature 3 is refused with the range, max_tokens is bounded by max_len, n-gram 1 is refused', () => {
    const e = validateParams({ temperature: 3, max_tokens: 300000, no_repeat_ngram_size: 1, top_k: 1.5, reasoning_effort: 'max' as never }, { max_len: 262144 });
    expect(e.temperature).toMatch(/at most 2/);
    expect(e.max_tokens).toMatch(/262,144/);
    expect(e.no_repeat_ngram_size).toMatch(/0 \(off\) or 2 to 10/);
    expect(e.top_k).toMatch(/whole number/);
    expect(e.reasoning_effort).toMatch(/low, medium, high, xhigh/);
    expect(validateParams({ temperature: 0.7, stop: ['a'] })).toEqual({});
    expect(validateParams({ stop: Array(9).fill('x') }).stop).toMatch(/up to 8/);
  });

  it('every param has a unique key and request name', () => {
    expect(new Set(PARAMS.map((p) => p.key)).size).toBe(PARAMS.length);
    expect(new Set(PARAMS.map((p) => p.request)).size).toBe(PARAMS.length);
  });
});

describe('asCurl', () => {
  it('quotes the body for a shell and targets the origin', () => {
    const c = asCurl(buildRequest(emptySetup(), [{ uid: 'x', role: 'user', content: "it's" }], { model: 'm' }), 'http://e:8000');
    expect(c.startsWith('curl -N http://e:8000/v1/chat/completions')).toBe(true);
    expect(c).toContain(`it'\\''s`);
  });
});
