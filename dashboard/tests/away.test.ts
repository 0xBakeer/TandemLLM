// OPS-65: the offline banner says what the status responder said.
import { describe, expect, it } from 'vitest';
import { awayText } from '../src/lib/away';

describe('awayText', () => {
  it('names the state, the time, the reason and the time left', () => {
    const t = awayText({ state: 'held', words: 'stopped for maintenance', since: 1760000000, reason: 'deploy 0.3.41-test', left_s: 780 }, () => '07:29');
    expect(t).toBe('TandemLLM is stopped for maintenance since 07:29 (deploy 0.3.41-test), back in about 13 min');
    expect(awayText({ state: 'starting', words: 'starting', since: null, reason: null, left_s: 40 })).toBe('TandemLLM is starting, back in about 40 s');
    expect(awayText(null)).toBe('The engine is not answering');
    expect(awayText({ state: 'held', words: 'x', since: null, reason: null, left_s: 0 })).toContain('back any moment');
  });
});
