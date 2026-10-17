// SRV-579 / VIS-219 (live 2026-10-17 23:06): while a worker generated, the Context panel showed only
// the main agent ("10,774 of 262,144", Now "warming up, the first 50 tokens") and nothing of the
// worker. Each running worker now has its own section with its live figures, keyed by task_id (a
// list: SRV-589 runs two at once), and the Now dock and the Task card show its rate.
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { applyEvent, contextWorkers } from '../src/chat/store';
import { etaWords, runningWorkers, waitingWords, workerLine, workerPrefill, workerRate } from '../src/chat/tasks';
import { MockChatBackend } from '../src/chat/mock';
import { workerContext, workerMetrics } from '../src/chat/mock-data';
import { QseChatContext } from '../src/chat/context-panel';
import { QseChatThread } from '../src/chat/thread';
import type { ContextReport, Conversation, Run, Task, TaskMetrics, WorkerContext } from '../src/chat/types';

/** a lit template as its text: the static parts and the bound values, nested templates in place */
function text(v: unknown): string {
  if (Array.isArray(v)) return v.map(text).join('');
  if (v && typeof v === 'object' && 'strings' in v && 'values' in v) {
    const t = v as { strings: readonly string[]; values: unknown[] };
    return t.strings.map((s, i) => s + (i < t.values.length ? text(t.values[i]) : '')).join('');
  }
  return v == null || typeof v === 'symbol' || typeof v === 'function' || v === false ? '' : String(v);
}
const count = (s: string, needle: string) => s.split(needle).length - 1;

const task = (n: number, p: Partial<Task> = {}): Task => ({ id: `t${n}`, conversation_id: 'c1', run_id: 'r1', call_id: `call${n}`, n, title: `Part ${n}`, brief: '', expect: '', status: 'running', steps: 0, max_steps: 40, prompt_tokens: 0, completion_tokens: 0, cached_tokens: 0, started: '2026-10-17T21:00:00Z', ended: null, report: null, report_source: null, check: null, error: null, ...p });
const DECODE = workerMetrics('t1', 'writing', 31_400, 2_180);
const PREFILL = workerMetrics('t2', 'prefill', 12_800, 610, { done: 12_400, total: 26_402, tps: 1_950, eta_ms: 7_200, cached: 3_072 });
const run = (phase: Run['phase']): Run => ({ id: 'r1', conversation_id: 'c1', message_id: 'a1', started_at: '2026-10-17T21:00:00Z', started_by: null, phase, now: 'Task 1: Part 1, call 6 of 40: Writing', step: 2, max_steps: 50, queue_position: null, tool: null, prefill: null, decode_tps: null, tokens_per_round: null, acceptance: null, thinking: { used: 0, budget: null }, tokens: 0, tps_history: [] });
const ctx = (workers: WorkerContext[]) => ({ window: 262_144, used: 10_774, parts: [], compact_at: 209_715, cache: { warm: 0, cold: 0, cold_eta_ms: null, source: 'none', slot: null }, harness: null, memory: { kv_bytes: null, resident_bytes: null, sandbox_disk_bytes: null }, model: { name: 'm', profile: null, weights: '', drafter: 'LiLiCorr', max_len: 262_144, effort: 'auto', thinking: true, backend: 'engine' }, workers }) as unknown as ContextReport;
const two = () => [workerContext(task(1), 6, { phase: 'writing', now: 'Writing src/sky.js', metrics: DECODE }), workerContext(task(2), 3, { phase: 'prefill', now: 'Reading the prompt', metrics: PREFILL })];
const conv = (p: Partial<Conversation> = {}) => ({ id: 'c1', title: 'x', messages: [], settings: { agents: true, thinking: true }, run: run('task'), context: ctx(two()), tasks: [task(1, { steps: 6, phase: 'writing' }), task(2, { steps: 3, phase: 'prefill' })], compactions: [], usage: { turns: 0, prompt_tokens: 0, completion_tokens: 0, reasoning_tokens: 0, cached_tokens: 0, tool_calls: 0, wall_s: 0 }, sandbox: { state: 'stopped', disk_bytes: 0, disk_limit_bytes: 0, memory_bytes: null }, queue: [], ...p }) as unknown as Conversation;

describe('task.metrics stays with its task', () => {
  it('two tasks: each event updates its own task_id only; task.done clears the figures', () => {
    let c = conv();
    c = applyEvent(c, { type: 'task.metrics', ...DECODE });
    c = applyEvent(c, { type: 'task.metrics', ...PREFILL });
    expect(c.tasks![0].metrics).toEqual(DECODE);
    expect(c.tasks![0].decode_tps).toBe(37.2);
    expect(c.tasks![1].metrics?.prefill).toEqual({ done: 12_400, total: 26_402, tps: 1_950, eta_ms: 7_200, cached: 3_072 });
    expect(c.tasks![1].decode_tps).toBeNull();
    c = applyEvent(c, { type: 'task.metrics', ...DECODE, decode_tps: 39.4 });
    expect(c.tasks![0].metrics?.decode_tps).toBe(39.4);
    expect(c.tasks![1].metrics?.prefill?.done).toBe(12_400);
    c = applyEvent(c, { type: 'task.done', task_id: 't1', status: 'done', report: null, check: null });
    expect(c.tasks![0].metrics).toBeNull();
    expect(c.tasks![1].metrics).toEqual(PREFILL);
  });

  it('context.workers is a list; an older chatd\'s single worker reads as one; the newest is kept for the last-worker line', () => {
    expect(contextWorkers(ctx(two())).map((w) => w.task_id)).toEqual(['t1', 't2']);
    expect(contextWorkers({ worker: two()[0] }).map((w) => w.task_id)).toEqual(['t1']);
    expect(contextWorkers({ worker: null })).toEqual([]);
    let c = conv();
    c = applyEvent(c, { type: 'context.changed', context: ctx(two()) });
    c = applyEvent(c, { type: 'context.changed', context: ctx([]) });
    expect(c.last_worker?.task_id).toBe('t2');
  });
});

describe('the running workers', () => {
  it('by task number, the report\'s context with the newest task.metrics; an ended task leaves; a task the report does not list yet comes without context', () => {
    const c = applyEvent(conv(), { type: 'task.metrics', ...DECODE, decode_tps: 41 });
    const v = runningWorkers(c);
    expect(v.map((x) => [x.task_id, x.n, x.step, x.phase])).toEqual([['t1', 1, 6, 'writing'], ['t2', 2, 3, 'prefill']]);
    expect(v[0].m?.decode_tps).toBe(41);
    expect(v[1].ctx?.used).toBeGreaterThan(0);
    const ended = runningWorkers(conv({ tasks: [task(1, { status: 'done' }), task(2, { steps: 3 })] }));
    expect(ended.map((x) => x.task_id)).toEqual(['t2']);
    const fresh = runningWorkers(conv({ context: ctx([]), tasks: [task(3, { steps: 1, phase: 'thinking' })] }));
    expect(fresh).toHaveLength(1);
    expect(fresh[0].ctx).toBeNull();
  });

  it('their words: the rate, the dock line, the ETA and the main agent waiting', () => {
    const [d, p] = runningWorkers(conv());
    expect(workerRate(d)).toBe('37.2 tok/s');
    expect(workerRate(p)).toBe('prefill 12,400 / 26,402 · 1,950 tok/s');
    expect(workerLine(d)).toBe('Task 1 · call 6 of 40 · 37.2 tok/s');
    expect(workerLine(p)).toBe('Task 2 · call 3 of 40 · prefill 12,400 / 26,402 · 1,950 tok/s');
    expect(workerPrefill(d)).toBeNull();
    expect(workerRate({ phase: 'tool', m: { ...DECODE, decode_tps: null } })).toBeNull();
    // a prefill the row still names after the worker began to write is not shown
    expect(workerRate({ phase: 'writing', m: { ...PREFILL, phase: 'prefill', decode_tps: 36 } as TaskMetrics })).toBe('36.0 tok/s');
    expect(etaWords(7_200)).toBe('about 8 s left');
    expect(etaWords(null)).toBeNull();
    expect(waitingWords([d])).toBe('waiting for Task 1');
    expect(waitingWords([d, p])).toBe('waiting for Task 1, Task 2');
    expect(waitingWords([])).toBeNull();
  });
});

describe('the panel and the Now dock render one row per worker', () => {
  it('the Context panel: a section per running worker with its own Now; the main Now waits for both', () => {
    const el = new QseChatContext();
    el.conv = conv();
    const out = text(el.render());
    expect(count(out, 'data-sec="worker"')).toBe(2);
    expect(out).toContain('Worker: Task 1');
    expect(out).toContain('Worker: Task 2');
    expect(out).toContain('37.2<span class="ch-fig-unit">tok/s</span>');
    expect(out).toContain('1,950<span class="ch-fig-unit">tok/s prefill</span>');
    expect(out).toContain('12,400 / 26,402');
    expect(out).toContain('about 8 s left');
    expect(out).toContain('call 6 of 40');
    expect(out).toContain('call 3 of 40');
    expect(out).toContain('waiting for Task 1, Task 2');
    expect(out).not.toContain('warming up, the first 50 tokens');
    // one worker: one section; none: the last-worker line
    el.conv = conv({ context: ctx([two()[0]]), tasks: [task(1, { steps: 6, phase: 'writing' }), task(2, { status: 'done' })] });
    expect(count(text(el.render()), 'data-sec="worker"')).toBe(1);
    el.conv = conv({ run: null, context: ctx([]), tasks: [task(1, { status: 'done' })] });
    const idle = text(el.render());
    expect(count(idle, 'data-sec="worker"')).toBe(0);
    expect(idle).toContain('Last worker: Task 1');
  });

  it('a task.metrics moves only its own section', () => {
    const el = new QseChatContext();
    el.conv = applyEvent(conv(), { type: 'task.metrics', ...PREFILL, prefill: { ...PREFILL.prefill!, done: 20_000 } });
    const out = text(el.render());
    expect(out).toContain('20,000 / 26,402');
    expect(out).toContain('37.2<span class="ch-fig-unit">tok/s</span>');
  });

  it('the run status card at the top: "waiting for Task 1, Task 2" and each task\'s rate', () => {
    const el = new QseChatThread() as unknown as { conv: Conversation; renderNow(c: Conversation): unknown };
    const c = conv();
    el.conv = c;
    const out = text(el.renderNow(c));
    expect(out).toContain('waiting for Task 1, Task 2');
    expect(out).toContain('Task 1 · call 6 of 40 · 37.2 tok/s');
    expect(out).toContain('Task 2 · call 3 of 40 · prefill 12,400 / 26,402 · 1,950 tok/s');
    expect(count(out, 'class="ch-now-task"')).toBe(2);
    // a run that is not waiting on a worker keeps its own words
    const solo = conv({ run: { ...run('writing'), now: 'Writing' } });
    expect(text(el.renderNow(solo))).not.toContain('ch-now-task"');
  });
});

describe('the mock plays two workers at once', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it('c_parallel: two running tasks, both in context.workers with their figures; Stop ends both', async () => {
    const mock = new MockChatBackend();
    let c = await mock.getConversation('c_parallel');
    c = { ...c, tasks: await mock.tasks('c_parallel') };
    expect(c.run?.phase).toBe('task');
    const v = runningWorkers(c);
    expect(v.map((x) => x.n)).toEqual([1, 2]);
    expect(workerRate(v[0])).toBe('37.2 tok/s');
    expect(workerRate(v[1])).toBe('prefill 12,400 / 26,402 · 1,950 tok/s');
    const box = { conv: c };
    mock.subscribe('c_parallel', c.seq, { onEvent: (e) => (box.conv = applyEvent(box.conv, e.event)), onCurrent: () => undefined, onState: () => undefined });
    await vi.advanceTimersByTimeAsync(1_000);
    await mock.stop('c_parallel');
    await vi.advanceTimersByTimeAsync(1_000);
    expect(box.conv.run).toBeNull();
    expect(box.conv.tasks!.map((t) => t.status)).toEqual(['cancelled', 'cancelled']);
    expect((await mock.getConversation('c_parallel')).context.workers).toEqual([]);
    mock.deleteConversation('c_sync');
    mock.deleteConversation('c_rename');
    mock.deleteConversation('c_waves');
  });
});

describe('the Task card shows its rate', () => {
  it('from task.metrics, else from the context report after a reload', () => {
    const el = new QseChatThread() as unknown as { renderTask(c: Conversation, m: unknown, t: unknown, live: boolean): unknown };
    const call = (n: number) => ({ type: 'tool_call', id: `call${n}`, name: 'task', input: '{}', output: null, status: 'running', started_at: null, ended_at: null, permission: null, pane: null });
    const c = conv();
    expect(text(el.renderTask(c, { id: 'a1', parts: [] }, call(1), true))).toContain('>37.2 tok/s</span>');
    expect(text(el.renderTask(c, { id: 'a1', parts: [] }, call(2), true))).toContain('>prefill 12,400 / 26,402 · 1,950 tok/s</span>');
    const ended = conv({ tasks: [task(1, { status: 'done', steps: 9 }), task(2)] });
    expect(text(el.renderTask(ended, { id: 'a1', parts: [] }, call(1), false))).not.toContain('data-fig="rate"');
  });
});
