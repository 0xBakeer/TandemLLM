// VIS-219 / VIS-220: the Context panel and usage with workers, and the Agents switch. The running
// worker's context apart from the main agent's, the line that stays after it, the turn figures
// that name the workers' part, and the switch's question and answer.
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { applyEvent } from '../src/chat/store';
import { agentsAsk, agentsDone, AGENTS_HINT, lastWorkerLine, workerCacheWords, workerParts } from '../src/chat/tasks';
import { figsParts, workerFigures } from '../src/chat/turn-figures';
import { figsLine } from '../src/chat/thread';
import { MockChatBackend } from '../src/chat/mock';
import { DEFAULT_SETTINGS } from '../src/chat/mock-data';
import type { ContextReport, Conversation, TurnStats, WorkerContext } from '../src/chat/types';

const WORKER: WorkerContext = { task_id: 't2', n: 2, title: 'Ocean shader: fix the horizon banding', used: 31_000, window: 262_144, split: { system: 2_410, tools: 3_880, brief: 610, steps: 21_300, files: 2_800 }, step: 12, max_steps: 40, cache: { warm: 28_800, cold: 2_200, cold_eta_ms: 3_000, source: 'resident', slot: 'slot 0' } };
const ctx = (worker: WorkerContext | null) => ({ window: 262_144, used: 18_400, parts: [], compact_at: 209_715, cache: { warm: 0, cold: 0, cold_eta_ms: null, source: 'none', slot: null }, harness: null, memory: {}, model: { name: 'm', backend: 'engine' }, worker }) as unknown as ContextReport;

describe('a running worker gets its own section', () => {
  it('the worker\'s used tokens, its split in order and call 12 of 40, apart from the main context', () => {
    expect(workerParts(WORKER)).toEqual([
      { key: 'system', label: 'system prompt', tokens: 2_410 },
      { key: 'tools', label: 'tool schemas', tokens: 3_880 },
      { key: 'brief', label: 'brief', tokens: 610 },
      { key: 'steps', label: 'its steps', tokens: 21_300 },
      { key: 'files', label: 'files', tokens: 2_800 },
    ]);
    expect(workerParts({ split: null })).toEqual([]);
    expect(workerParts({ split: { system: 100, steps: 50 } }).map((p) => p.key)).toEqual(['system', 'steps']);
  });

  it('its cache tier; a cloud model\'s worker has none on this box, and the panel says so', () => {
    expect(workerCacheWords(WORKER, 'engine')).toBe('28,800 warm from the resident, 2,200 cold');
    expect(workerCacheWords(WORKER, 'cloud')).toBe('no cache tier on this box: the provider keeps it');
    expect(workerCacheWords({ cache: null }, 'engine')).toBe('not reported yet');
  });
});

describe('the last worker line stays after the task', () => {
  it('the stream keeps the newest worker this device saw when context.worker goes null', () => {
    let c = { id: 'c1', messages: [], context: ctx(null) } as unknown as Conversation;
    c = applyEvent(c, { type: 'context.changed', context: ctx({ ...WORKER, used: 51_204 }) });
    c = applyEvent(c, { type: 'context.changed', context: ctx(null) });
    expect(c.context.worker).toBeNull();
    expect(c.last_worker?.used).toBe(51_204);
  });

  it('"Last worker: Task 2, 51,204 tokens, done"; after a reload without the tokens', () => {
    const t = { id: 't2', n: 2, status: 'done', phase: null, messages: [] };
    expect(lastWorkerLine(t, { ...WORKER, used: 51_204 })).toBe('Last worker: Task 2, 51,204 tokens, done');
    expect(lastWorkerLine(t, null)).toBe('Last worker: Task 2, done');
    expect(lastWorkerLine({ ...t, status: 'partial (step limit)' }, { ...WORKER, task_id: 'other' })).toBe('Last worker: Task 2, partial (step limit)');
  });
});

describe('usage counts both', () => {
  const base: TurnStats = { finish: 'stop', ttft_ms: 310, decode_tps: 54.1, prompt_tokens: 127_614, completion_tokens: 9_342, reasoning_tokens: 0, cached_tokens: 98_400, tokens_per_round: null, acceptance: null, total_ms: 40_000 };

  it('main 31,204 · workers 2 (96,410) · total 127,614 prompt tokens', () => {
    const s: TurnStats = { ...base, workers: { count: 2, prompt_tokens: 96_410, completion_tokens: 7_904, cached_tokens: 81_000, ms: 620_000 } };
    expect(workerFigures(s)).toEqual([
      { k: 'main', n: '31,204', w: 'main ', pre: true },
      { k: 'workers', n: '2', w: 'workers ', pre: true, sub: '(96,410)' },
      { k: 'total', n: '127,614', w: 'total ', pre: true, sub: 'prompt tokens' },
    ]);
    // a row of its own under the figures line: the line keeps its readings
    expect(figsParts(s).map((f) => f.k)).toEqual(['tps', 'tok', 'ttft']);
    expect(figsLine(s)).toContain('main 31,204 · workers 2 (96,410) · total 127,614 prompt tokens');
  });

  it('Agents off looks as today: no worker readings, the same line', () => {
    expect(workerFigures(base)).toEqual([]);
    expect(workerFigures({ ...base, workers: { count: 0, prompt_tokens: 0, completion_tokens: 0, cached_tokens: 0, ms: 0 } })).toEqual([]);
    expect(figsParts(base).map((f) => f.k)).toEqual(['tps', 'tok', 'ttft']);
    expect(figsLine(base)).toBe('done · 310 ms to first token · 54.1 tok/s · 9,342 tok');
  });
});

describe('the Agents switch', () => {
  it('in a chat with messages it asks first, with the chat\'s tokens, and applies at the next message', () => {
    expect(agentsAsk(18_400, false)).toBe('The next message reads this chat once more (about 18,400 tokens). The switch applies at the next message.');
    expect(agentsAsk(18_400, true)).toMatch(/keeps its mode; the switch applies at the next message/);
    expect(AGENTS_HINT).toBe('A main agent plans and workers change the files');
  });

  it('after the switch the note carries chatd\'s re-read figure', () => {
    expect(agentsDone(true, 18_412)).toBe('Agents on from the next message. Switching re-reads this chat once: 18,412 tokens.');
    expect(agentsDone(false, null)).toBe('Agents off from the next message.');
  });

  it('new chats default to off; the settings page\'s default reaches new chats', async () => {
    expect(DEFAULT_SETTINGS.agents).toBe(false);
    vi.useFakeTimers();
    try {
      const mock = new MockChatBackend();
      expect((await mock.capabilities()).defaults.agents).toBe(false);
      await mock.updateDefaults({ agents: true });
      const c = await mock.createConversation();
      expect(c.settings.agents).toBe(true);
      mock.deleteConversation('c_sync');
      mock.deleteConversation('c_rename');
      mock.deleteConversation('c_waves');
    } finally {
      vi.useRealTimers();
    }
  });
});

describe('the mock answers the switch like SRV-583', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it('PATCH settings {agents} answers reread_tokens: what the next message reads again', async () => {
    const mock = new MockChatBackend();
    const s = await mock.updateSettings('c_explain', { agents: true });
    expect(s.agents).toBe(true);
    expect(s.reread_tokens).toBeGreaterThan(0);
    const other = await mock.updateSettings('c_explain', { thinking: false });
    expect(other.reread_tokens).toBeUndefined();
    const ocean = await mock.updateSettings('c_ocean', { agents: false });
    expect(ocean.reread_tokens).toBe((await mock.getConversation('c_ocean')).context.used);
    const fresh = await mock.createConversation();
    expect((await mock.updateSettings(fresh.id, { agents: true })).reread_tokens).toBe(0);
    mock.deleteConversation('c_sync');
    mock.deleteConversation('c_rename');
    mock.deleteConversation('c_waves');
  });

  it('a question in an Agents chat stays with the main agent: no task', async () => {
    const mock = new MockChatBackend();
    await mock.send('c_ocean', { client_msg_id: 'cm_q', text: 'How does tokens per round set the speed?' });
    await vi.advanceTimersByTimeAsync(30_000);
    const tasks = await mock.tasks('c_ocean');
    expect(tasks).toHaveLength(4);
    mock.deleteConversation('c_sync');
    mock.deleteConversation('c_rename');
    mock.deleteConversation('c_waves');
  });
});
