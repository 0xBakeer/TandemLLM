// VIS-91 (QA VIS-92): Wren's turn in the browser. On the live box a plain question ended in eight
// "tried something that does not exist" lines, the 4-round stop and no answer, and the phone's
// sheet dropped and rose once per call. nextRound decides whether the page asks again and with
// the tools on or off, turnNotices gives a turn at most one notice, and toastFor says what the
// phone's sheet shows for a call.
import { describe, expect, it } from 'vitest';
import type { SetupChatMessage } from '../src/api/types';
import { MAX_ROUNDS, nextRound, toastFor, turnNotices } from '../src/setup/wren-turn';

const explainBad = { name: 'explain', ok: false, changed: false };
const explainOk = { name: 'explain', ok: true, changed: false };
const setOk = { name: 'set_field', ok: true, changed: true };
const setBad = { name: 'set_field', ok: false, changed: false };

describe('nextRound', () => {
  it('no call: the turn is over', () => {
    expect(nextRound({ round: 1, tools: true, text: 'Four slots.', calls: [], retried: false })).toBeNull();
  });
  it('a round with the tools off is the last one', () => {
    expect(nextRound({ round: 2, tools: false, text: '', calls: [setOk], retried: true })).toBeNull();
  });
  it('a refused call gets one repair with the tools on, then words with the tools off', () => {
    const r1 = nextRound({ round: 1, tools: true, text: '', calls: [explainBad, explainBad], retried: false });
    expect(r1).toEqual({ tools: true, retried: true });
    const r2 = nextRound({ round: 2, tools: true, text: '', calls: [explainBad, explainBad], retried: true });
    expect(r2).toEqual({ tools: false, retried: true });
  });
  it('the live sequence (two refused explains a round) asks at most three times and ends in words', () => {
    let next: ReturnType<typeof nextRound> = { tools: true, retried: false };
    let rounds = 0;
    while (next) {
      rounds++;
      const calls = next.tools ? [explainBad, explainBad] : [];
      next = nextRound({ round: rounds, tools: next.tools, text: next.tools ? '' : 'EXL3 2.5 is the fastest for code.', calls, retried: next.retried });
    }
    expect(rounds).toBe(3);
  });
  it('a highlight with words needs nothing more; a highlight alone asks for the words', () => {
    expect(nextRound({ round: 1, tools: true, text: 'It is here.', calls: [explainOk], retried: false })).toBeNull();
    expect(nextRound({ round: 1, tools: true, text: '', calls: [explainOk], retried: false })).toEqual({ tools: false, retried: false });
  });
  it('a change goes on, and the last of MAX_ROUNDS rounds has the tools off', () => {
    expect(nextRound({ round: 1, tools: true, text: 'Setting it.', calls: [setOk], retried: false })).toEqual({ tools: true, retried: false });
    expect(nextRound({ round: MAX_ROUNDS - 1, tools: true, text: '', calls: [setOk], retried: false })).toEqual({ tools: false, retried: false });
    let next: ReturnType<typeof nextRound> = { tools: true, retried: false };
    let rounds = 0;
    while (next) {
      rounds++;
      next = nextRound({ round: rounds, tools: next.tools, text: '', calls: next.tools ? [setOk] : [], retried: next.retried });
    }
    expect(rounds).toBe(MAX_ROUNDS);
  });
});

const call = (id: string, name: string) => ({ id, name, args: {} });
const tool = (id: string, ok: boolean, extra: Record<string, unknown> = {}): SetupChatMessage => ({ role: 'tool', tool_call_id: id, content: JSON.stringify(ok ? { ok, result: 'ok', ...extra } : { ok, error: 'no', ...extra }) });

describe('turnNotices', () => {
  it('eight refused calls in one turn: one notice at the end of the turn, none for a highlight', () => {
    const chat: SetupChatMessage[] = [{ role: 'user', content: 'Turn on flux' }];
    for (let r = 0; r < 4; r++) {
      chat.push({ role: 'assistant', content: '', tool_calls: [call(`a${r}`, 'toggle'), call(`b${r}`, 'toggle')] }, tool(`a${r}`, false), tool(`b${r}`, false));
    }
    chat.push({ role: 'assistant', content: 'There is no flux here.' });
    const n = turnNotices(chat);
    expect([...n.entries()]).toEqual([[chat.length - 1, 'unchanged']]);
  });
  it('the live chat: refused highlights leave no notice at all', () => {
    const chat: SetupChatMessage[] = [{ role: 'user', content: 'Which one is for coding tasks?' }];
    for (let r = 0; r < 4; r++) chat.push({ role: 'assistant', content: '', tool_calls: [call(`x${r}`, 'explain'), call(`y${r}`, 'explain')] }, tool(`x${r}`, false), tool(`y${r}`, false));
    expect(turnNotices(chat).size).toBe(0);
  });
  it('a repair that worked: no notice', () => {
    const chat: SetupChatMessage[] = [
      { role: 'user', content: 'Use two slots' },
      { role: 'assistant', content: '', tool_calls: [call('a', 'set_field')] },
      tool('a', false),
      { role: 'assistant', content: '', tool_calls: [call('b', 'set_field')] },
      tool('b', true, { summary: 'Wren set Parallel requests to 2' }),
      { role: 'assistant', content: 'Two slots now.' },
    ];
    expect(turnNotices(chat).size).toBe(0);
  });
  it('one notice per turn, a stale call says so, and each turn has its own', () => {
    const chat: SetupChatMessage[] = [
      { role: 'user', content: 'Turn vision off' },
      { role: 'assistant', content: '', tool_calls: [call('a', 'toggle')] },
      tool('a', false, { stale: true }),
      { role: 'assistant', content: 'You changed it first.' },
      { role: 'user', content: 'Turn on flux' },
      { role: 'assistant', content: '', tool_calls: [call('b', 'toggle')] },
      tool('b', false),
    ];
    expect([...turnNotices(chat).entries()]).toEqual([
      [3, 'stale'],
      [6, 'unchanged'],
    ]);
  });
});

describe('toastFor', () => {
  it('a change on a phone: the toast; a refused call, a highlight or a desktop: none', () => {
    expect(toastFor({ ok: true, summary: 'Wren turned off Vision' }, true)).toBe('Wren turned off Vision');
    expect(toastFor({ ok: false, summary: null }, true)).toBeNull();
    expect(toastFor({ ok: true, summary: null }, true)).toBeNull();
    expect(toastFor({ ok: true, summary: 'Wren turned off Vision' }, false)).toBeNull();
  });
});
