// the readout mapping and the conversation exports.
import { describe, expect, it } from 'vitest';
import { exportFileName, readoutLine, readoutOf, toJsonExport, toMarkdownExport } from '../src/playground/export';
import { emptySetup, type Message, type TurnStats } from '../src/playground/types';

const stats: TurnStats = {
  finish: 'stop',
  usage: { prompt_tokens: 1204, completion_tokens: 96, total_tokens: 1300, prompt_tokens_details: { cached_tokens: 1024 }, completion_tokens_details: { reasoning_tokens: 12 } },
  timings: { cache_n: 1024, prompt_n: 1204, prompt_ms: 62, prompt_per_token_ms: 0.05, prompt_per_second: 2910.4, predicted_n: 96, predicted_ms: 2500, predicted_per_token_ms: 26, predicted_per_second: 38.24, draft_n: 300, draft_n_accepted: 183, ttft_ms: 410.2, queue_ms: 0.4, total_ms: 2911, blocks: 22, tokens_per_block: 4.3, reasoning_n: 12, cache_source: 'prefix' },
  metrics: { time_to_first_token_ms: 410.2, generation_time_ms: 2500, queue_time_ms: 0.4, mean_itl_ms: 26, tokens_per_second: 33, speculative_decoding: { mean_acceptance_length: 4.3, draft_acceptance_rate: 0.61 } },
};

describe('readout', () => {
  it('maps every figure with its unit', () => {
    const f = Object.fromEntries(readoutOf(stats).map((x) => [x.id, x]));
    expect(f.prompt).toMatchObject({ value: '1,204', unit: 'tok' });
    expect(f.cached).toMatchObject({ value: '1,024', unit: 'tok' });
    expect(f.completion).toMatchObject({ value: '96' });
    expect(f.reasoning).toMatchObject({ value: '12' });
    expect(f.ttft).toMatchObject({ value: '410 ms', key: true });
    expect(f.prefill).toMatchObject({ value: '2,910', unit: 'tok/s' });
    expect(f.decode).toMatchObject({ value: '38.2', unit: 'tok/s', key: true });
    expect(f.tpb).toMatchObject({ value: '4.30', unit: 'tok/blk', key: true });
    expect(f.acceptance).toMatchObject({ value: '61 %' });
    expect(f.finish).toMatchObject({ value: 'stop', key: true });
    expect(f.cache).toMatchObject({ value: 'prefix' });
    expect(readoutLine(stats)).toBe('410 ms · 38.2 tok/s · 4.30 tok/blk · stop');
  });
  it('a missing family says not reported, never 0', () => {
    const f = Object.fromEntries(readoutOf({ finish: 'stop', usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 } }).map((x) => [x.id, x]));
    expect(f.acceptance).toMatchObject({ value: 'not reported', na: true });
    expect(f.ttft).toMatchObject({ value: 'not reported', na: true });
    expect(f.prompt).toMatchObject({ value: '1' });
    expect(readoutOf(null).every((x) => x.na || x.id === 'finish')).toBe(true);
    // the timings-only shape (usage default off on the server) still fills the tokens
    const g = Object.fromEntries(readoutOf({ finish: 'stop', timings: stats.timings }).map((x) => [x.id, x]));
    expect(g.prompt.value).toBe('1,204');
    expect(g.reasoning.value).toBe('12');
  });
});

const messages: Message[] = [
  { uid: '1', role: 'user', content: 'weather?' },
  { uid: '2', role: 'assistant', content: '', reasoning: 'call it', tool_calls: [{ id: 'call_1', type: 'function', function: { name: 'get_weather', arguments: '{"city":"Berlin"}' } }], stats: { ...stats, finish: 'tool_calls' } },
  { uid: '3', role: 'tool', tool_call_id: 'call_1', name: 'get_weather', content: '{"temperature_c":21}' },
  { uid: '4', role: 'assistant', content: 'It is **21 °C**.', stats },
];

describe('exports', () => {
  it('JSON: the shape, no uids, per-turn stats by index', () => {
    const d = JSON.parse(toJsonExport({ ...emptySetup(), system: 'S' }, messages, 'qwen38-spark-engine', new Date('2026-09-25T12:00:00Z')));
    expect(d).toMatchObject({ format: 'qse-playground-conversation', version: 1, exportedAt: '2026-09-25T12:00:00.000Z', model: 'qwen38-spark-engine', setup: { system: 'S' } });
    expect(d.messages).toHaveLength(4);
    expect(d.messages[0]).not.toHaveProperty('uid');
    expect(d.messages[1]).not.toHaveProperty('stats');
    expect(d.messages[1].tool_calls[0].id).toBe('call_1');
    expect(d.turns.map((t: { index: number }) => t.index)).toEqual([1, 3]);
    expect(d.turns[0].finish).toBe('tool_calls');
    expect(d.turns[1].timings.ttft_ms).toBe(410.2);
  });
  it('Markdown: one heading per message, the readout line under each assistant turn, a tool call as fenced JSON', () => {
    const md = toMarkdownExport({ ...emptySetup(), system: 'S' }, messages, 'm', new Date('2026-09-25T12:00:00Z'));
    expect(md.match(/^## /gm)).toHaveLength(5); // system + 4 messages
    expect(md).toContain('## tool (get_weather)');
    expect(md).toContain('> call it');
    expect(md).toContain('```json\n{\n  "tool_call": "get_weather"');
    expect(md.match(/_410 ms · 38\.2 tok\/s · 4\.30 tok\/blk · (stop|tool_calls)_/g)).toHaveLength(2);
  });
  it('file names carry the local date and time', () => {
    expect(exportFileName('json', new Date(2026, 8, 25, 9, 5))).toBe('playground-20260925-0905.json');
    expect(exportFileName('md', new Date(2026, 8, 25, 9, 5))).toBe('playground-20260925-0905.md');
  });
});
