// A chat on a cloud model (0.11.3): the Context panel's words are true for a provider's model.
// Live (2026-10-17, DeepSeek V4 Flash) the panel said "not tracked for this model" for tok/s,
// "0, no cap" for a minute of thinking and "nothing cached yet" beside 46,208 cache hits.
import { describe, expect, it } from 'vitest';
import { budgetWords, cacheWords, draftFigure, kvWords, liveFigure, NO_DRAFTING, NOT_TRACKED, partWords, PROVIDER_DECIDES } from '../src/chat/context-words';
import { figsParts } from '../src/chat/turn-figures';
import { figsLine } from '../src/chat/thread';
import type { ContextReport, Run, TurnStats } from '../src/chat/types';

const run = (over: Partial<Run> = {}): Run => ({ phase: 'thinking', thinking: { used: 0, budget: null }, ...over }) as Run;
const cache = (c: Partial<ContextReport['cache']>): ContextReport['cache'] => ({ warm: 0, cold: 0, cold_eta_ms: null, source: 'none', slot: null, ...c });

describe('a cloud model in the Context panel', () => {
  it('says nothing drafts, instead of "not tracked"', () => {
    expect(liveFigure(null, String, run(), 'round', 'cloud', null)).toBe(NO_DRAFTING);
    expect(liveFigure(null, String, run(), 'accepted', 'cloud', null)).toBe(NO_DRAFTING);
    expect(draftFigure(null, String, true, 'cloud')).toBe(NO_DRAFTING);
    expect(draftFigure(null, String, true, 'engine')).toBe(NOT_TRACKED);
    // a local backend through the proxy may draft: its counts are not ours to say
    expect(liveFigure(null, String, run(), 'round', 'proxy', null)).toBe(NOT_TRACKED);
  });
  it("gives chatd's own tok/s, and why there is none yet", () => {
    expect(liveFigure(172.4, (v) => v.toFixed(1), run(), 'tps', 'cloud', null)).toBe('172.4');
    expect(liveFigure(null, String, run({ phase: 'prefill' }), 'tps', 'cloud', null)).toBe('waiting for the first token');
    expect(liveFigure(null, String, run({ phase: 'prefill' }), 'tps', 'engine', 'mtp')).toBe('prefilling');
    expect(liveFigure(null, String, run({ phase: 'tool' }), 'tps', 'cloud', null)).toBe('between model calls');
    expect(liveFigure(null, String, run({ phase: 'thinking' }), 'tps', 'cloud', null)).toMatch(/warming up/);
  });
  it('says the provider decides the thinking budget', () => {
    expect(budgetWords(null, 19_900, 19_900, 'cloud')).toBe(`19,900, ${PROVIDER_DECIDES}`);
    expect(budgetWords(0, 0, 0, 'cloud')).toBe('off');
    expect(budgetWords(null, 312, 312, 'engine')).toBe('312, no cap');
    expect(budgetWords(8192, 9000, 4000, 'engine')).toBe('4,000 of 8,192, this call');
  });
  it("says what the provider cached, not the engine's tiers", () => {
    expect(cacheWords(cache({ source: 'provider', warm: 46_208, cold: 212 }))).toEqual({ warm: '46,208 cached by the provider', cold: '212 uncached', none: null });
    expect(cacheWords(cache({ source: 'provider', warm: 0, cold: 7_542 })).none).toBe('nothing cached by the provider yet');
    expect(cacheWords(cache({ source: 'spill', warm: 900, cold: 40 }))).toEqual({ warm: '900 warm from the spill', cold: '40 cold to prefill', none: null });
    expect(cacheWords(cache({ source: 'none', warm: 0, cold: 596 }))).toEqual({ warm: null, cold: '596 cold to prefill', none: 'nothing cached yet' });
  });
  it("marks chatd's estimate of the split", () => {
    const split = { system: 380, tools: 5_400, images: 0, skills: 120, estimated: true };
    expect(partWords('system', 380, split, 'cloud')).toBe('380 est.');
    expect(partWords('tools', 5_400, split, 'cloud')).toBe('5,400 est.');
    expect(partWords('skills', 120, split, 'cloud')).toBe('120 est.');
    expect(partWords('messages', 48_000, split, 'cloud')).toBe('48,000 est.');
    expect(partWords('thinking', 3_407, split, 'cloud')).toBe('3,407');
    expect(kvWords({ kv_bytes: null, resident_bytes: null, sandbox_disk_bytes: null }, 'cloud')).toMatch(/provider keeps it/);
  });
});

describe("a cloud turn's figures line", () => {
  const s: TurnStats = { finish: 'tool_calls', ttft_ms: 1521, decode_tps: 302.5, prompt_tokens: 46_420, completion_tokens: 9_950, reasoning_tokens: 3_407, cached_tokens: 46_208, tokens_per_round: null, acceptance: null, total_ms: 34_459, cache_source: 'provider', cost_usd: 0.00614 };
  it('ends with the cost at the proxy prices', () => {
    const parts = figsParts(s);
    expect(parts.map((p) => p.k)).toEqual(['finish', 'tps', 'tok', 'ttft', 'cost']);
    expect(parts.at(-1)!.n).toBe('$0.0061');
    expect(figsLine(s)).toMatch(/· \$0\.0061$/);
  });
  it('has no cost on an engine turn', () => {
    expect(figsParts({ ...s, cost_usd: null }).some((p) => p.k === 'cost')).toBe(false);
  });
});
