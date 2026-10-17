// VIS-217 / VIS-218: the Task card. A worker's task.* events and its parts (with `task_id`) build the
// card from events alone; they never reach the main thread. The card's states, the report and
// chatd's check, the screenshots, the permission prompt, and a device that attaches mid-task.
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { adoptTask, applyEvent, applyEvents, mergeTasks, taskForCall } from '../src/chat/store';
import { askingCall, callsWord, checkView, isUnmatched, reportView, shotsOf, sourceWord, summaryLine, taskSeconds, taskStatus, tokensWord } from '../src/chat/tasks';
import { taskAsking, taskTitle } from '../src/chat/thread';
import { actingPhase } from '../src/chat/verify';
import { MockChatBackend } from '../src/chat/mock';
import { OCEAN_TASKS } from '../src/chat/mock-data';
import type { ChatEvent, Conversation, Run, StampedEvent, Task, TaskReport, ToolCallPart } from '../src/chat/types';

const run = { id: 'r1', conversation_id: 'c1', message_id: 'a1', started_at: '2026-10-17T18:00:00Z', phase: 'task', now: 'Task 2: …', step: 3, max_steps: 100, tool: null, prefill: null, queue_position: null, tps_history: [], thinking: { used: 0, budget: null }, tokens: 0 } as unknown as Run;
const taskCall: ToolCallPart = { type: 'tool_call', id: 'call_t2', name: 'task', input: '{"title":"Ocean shader: fix the horizon banding","brief":"Goal: …","expect":"the FPS line"}', output: null, status: 'running', started_at: null, ended_at: null };

function conv(): Conversation {
  return {
    id: 'c1',
    status: 'running',
    run,
    messages: [
      { id: 'u1', role: 'user', created_at: '', parent_id: null, run_id: null, status: 'done', parts: [{ type: 'text', text: 'Fix the horizon.' }] },
      { id: 'a1', role: 'assistant', created_at: '', parent_id: 'u1', run_id: 'r1', status: 'streaming', parts: [{ type: 'text', text: 'A worker gets this.' }, taskCall] },
    ],
  } as unknown as Conversation;
}

const TASK: Task = { id: 't2', conversation_id: 'c1', run_id: 'r1', call_id: 'call_t2', n: 2, title: 'Ocean shader: fix the horizon banding', brief: 'Goal: no bands\nFiles: src/ocean.wgsl\nConstraints: none\nCheck: render_page index.html', expect: 'the FPS line', status: 'running', steps: 0, max_steps: 40, prompt_tokens: 0, completion_tokens: 0, cached_tokens: 0, started: '2026-10-17T18:42:00Z', ended: null, report: null, report_source: null, check: null, error: null };

const REPORT: TaskReport = {
  status: 'done',
  summary: 'The banding came from 8-bit normals; they are float now.',
  changed_files: [{ path: 'src/ocean.wgsl', change: 'normals in f32, smoothstep at the horizon' }],
  checks: [
    { what: 'render_page index.html', result: '0 console errors, shot-7', call_id: 'c-31', shot: 'shot-7' },
    { what: 'bash "npm test"', result: 'exit 0', call_id: null },
  ],
  expected: [{ item: 'the FPS line from the console', answer: '"fps 58.9"' }],
  open_problems: ['none'],
};
const CHECK = `- Changed in /workspace (snapshot diff): src/ocean.wgsl (+41 -12), src/main.js (+3 -1)
- Changed but not in the report: src/main.js
- Checks that ran: render_page index.html (call c-31: 0 console errors, shot-7)
- Claimed in Checks with no matching call: bash "npm test"`;

const started: ChatEvent = { type: 'task.started', run_id: 'r1', task: TASK };
const workerEvents: ChatEvent[] = [
  { type: 'part.added', message_id: 'w1', part: { type: 'thinking', text: '', tokens: null, done: false }, task_id: 't2' },
  { type: 'part.delta', message_id: 'w1', index: 0, delta: 'Look at the normals.', task_id: 't2' },
  { type: 'part.done', message_id: 'w1', index: 0, tokens: 5, task_id: 't2' },
  { type: 'part.added', message_id: 'w1', part: { type: 'tool_call', id: 'c-31', name: 'render_page', input: '{"path":"index.html"}', output: null, status: 'running', started_at: null, ended_at: null }, task_id: 't2' },
  { type: 'tool.started', message_id: 'w1', call_id: 'c-31', pane: { kind: 'image', ref: 'file:/workspace/.tandem/shots/c-31.png', title: 'shot-7' } },
  { type: 'tool.output', message_id: 'w1', call_id: 'c-31', delta: '0 console errors' },
  { type: 'tool.done', message_id: 'w1', call_id: 'c-31', status: 'done', is_error: false, output: { content: '0 console errors', is_error: false, bytes: 16, images: [{ id: 'shot-7', name: 'shot-7.png', kind: 'image', bytes: 9, mime: 'image/png', url: '/v1/chat-app/attachments/shot-7' }] } },
];
const phase12: ChatEvent = { type: 'task.phase', task_id: 't2', phase: 'tool', now: 'writing src/ocean.wgsl', step: 12, max_steps: 40, tool: { call_id: 'c-32', name: 'edit_file' } };
const metrics: ChatEvent = { type: 'task.metrics', task_id: 't2', prompt_tokens: 48_210, completion_tokens: 3_904, decode_tps: 54.2 };
const done: ChatEvent = { type: 'task.done', task_id: 't2', status: 'done', report: REPORT, check: CHECK, stats: { steps: 14, prompt_tokens: 48_210, completion_tokens: 3_904, cached_tokens: 41_000, ms: 372_000 } };

function task(c: Conversation): Task {
  return c.tasks!.find((t) => t.id === 't2')!;
}

describe('a running task shows its live state', () => {
  it('task.started, task.phase and task.metrics fill the card: title, running, the Now line, call 12 of 40, tokens, elapsed', () => {
    const c = applyEvents(conv(), [started, phase12, metrics]);
    const t = task(c);
    expect(taskForCall(c, 'call_t2')?.id).toBe('t2');
    expect(t.title).toBe('Ocean shader: fix the horizon banding');
    expect(taskStatus(t)).toEqual({ key: 'running', word: 'running', tone: 'live' });
    expect(t.now).toBe('writing src/ocean.wgsl');
    expect(callsWord(t)).toBe('call 12 of 40');
    expect(tokensWord(t)).toBe('48,210 in · 3,904 out');
    expect(taskSeconds(t, Date.parse('2026-10-17T18:48:12Z'))).toBe(372);
    expect(t.decode_tps).toBe(54.2);
  });

  it('a worker before its first usage says so, never a made-up zero', () => {
    expect(tokensWord(TASK)).toBe('no usage yet');
    expect(callsWord(TASK)).toBe('call 1 of 40');
  });

  it('the run phase `task` acts on the Now line', () => {
    expect(actingPhase('task')).toBe(true);
    const c = applyEvent(conv(), { type: 'run.phase', run_id: 'r1', phase: 'task', now: 'Task 2: Ocean shader: fix the horizon banding, call 12 of 40: writing src/ocean.wgsl' });
    expect(c.run?.phase).toBe('task');
    expect(c.now).toMatch(/call 12 of 40/);
  });

  it('a task call whose arguments still stream shows its title', () => {
    expect(taskTitle('{"title":"Ocean shader: fix the horizon banding","brief":"Goal: …","expect":"x"}')).toBe('Ocean shader: fix the horizon banding');
    expect(taskTitle('{"title":"Ocean sha')).toBe('Ocean sha');
    expect(taskTitle('{"brie')).toBe('giving a task to a worker');
  });
});

describe('the transcript', () => {
  it('worker parts go to the task, in order, and never to the main thread', () => {
    const before = conv();
    const c = applyEvents(before, [started, ...workerEvents]);
    expect(c.messages).toEqual(before.messages);
    const ms = task(c).messages!;
    expect(ms).toHaveLength(1);
    expect(ms[0].parts.map((p) => p.type)).toEqual(['thinking', 'tool_call']);
    expect(ms[0].parts[0]).toMatchObject({ text: 'Look at the normals.', done: true, tokens: 5 });
    const call = ms[0].parts[1] as ToolCallPart;
    expect(call.input).toBe('{"path":"index.html"}');
    expect(call.output?.content).toBe('0 console errors');
    expect(call.status).toBe('done');
  });

  it('a worker message.created and its run.done stay in the card: the main run goes on', () => {
    const c = applyEvents(conv(), [started, { type: 'message.created', message: { id: 'w0', role: 'user', created_at: '', parent_id: null, run_id: null, status: 'done', parts: [{ type: 'text', text: 'Task 2: …' }] }, task_id: 't2' }, ...workerEvents, { type: 'run.done', run_id: 'wr', message_id: 'w1', stats: { finish: 'stop' } as never, status: 'done' }]);
    expect(c.messages.map((m) => m.id)).toEqual(['u1', 'a1']);
    expect(c.run?.id).toBe('r1');
    expect(task(c).messages!.map((m) => m.id)).toEqual(['w0', 'w1']);
    expect(task(c).messages![1].status).toBe('done');
  });

  it('a task whose transcript this device has not loaded drops its parts until it is loaded', () => {
    const c = applyEvents({ ...conv(), tasks: [{ ...TASK, status: 'done' }] }, workerEvents.slice(0, 2));
    expect(task(c).messages).toBeUndefined();
    expect(c.messages).toEqual(conv().messages);
  });

  it('two tasks of a run keep their own transcripts', () => {
    const t3: Task = { ...TASK, id: 't3', n: 3, call_id: 'call_t3' };
    const c = applyEvents(conv(), [started, { type: 'task.started', run_id: 'r1', task: t3 }, workerEvents[0], { type: 'part.added', message_id: 'w3', part: { type: 'text', text: 'three' }, task_id: 't3' }]);
    expect(c.tasks!.map((t) => [t.n, t.messages!.length])).toEqual([[2, 1], [3, 1]]);
  });
});

describe('the report and chatd\'s check', () => {
  it('task.done brings the report and the check; src/main.js is changed but not in the report', () => {
    const c = applyEvents(conv(), [started, ...workerEvents, done]);
    const t = task(c);
    expect(taskStatus(t).key).toBe('done');
    expect(t.ended).toBeTruthy();
    expect(callsWord(t)).toBe('14 calls');
    expect(taskSeconds(t, Date.now())).toBe(372);
    const r = reportView(t.report!);
    expect(r.changed).toEqual([{ path: 'src/ocean.wgsl', change: 'normals in f32, smoothstep at the horizon' }]);
    expect(r.checks[0]).toEqual({ text: 'render_page index.html: 0 console errors, shot-7', call_id: 'c-31', shot: 'shot-7' });
    expect(r.expected).toEqual(['the FPS line from the console: "fps 58.9"']);
    expect(r.open).toEqual([]);
    const k = checkView(t.check);
    expect(k.changed).toEqual([
      { path: 'src/ocean.wgsl', delta: '+41 -12', unreported: false },
      { path: 'src/main.js', delta: '+3 -1', unreported: true },
    ]);
    expect(k.ran).toEqual([{ text: 'render_page index.html (call c-31: 0 console errors, shot-7)', call_id: 'c-31', shot: 'shot-7' }]);
    expect(k.unmatched).toEqual(['bash "npm test"']);
    expect(isUnmatched(r.checks[1], k)).toBe(true);
    expect(isUnmatched(r.checks[0], k)).toBe(false);
  });

  it('report items that come as lines are read as lines', () => {
    const r = reportView({ status: 'done', summary: 's', changed_files: ['src/a.js: one line'], checks: ['bash "npm test": exit 0'], expected: ['port: 5173'], open_problems: ['none'] });
    expect(r.changed).toEqual([{ path: 'src/a.js', change: 'one line' }]);
    expect(r.checks[0].text).toBe('bash "npm test": exit 0');
    expect(r.expected).toEqual(['port: 5173']);
  });

  it('a check of nothing has empty lists, and lines it does not know stay as they are', () => {
    const k = checkView('chatd check:\n- Changed in /workspace (snapshot diff): none\n- Checks that ran: none\n- the verify phase: passed');
    expect(k.changed).toEqual([]);
    expect(k.ran).toEqual([]);
    expect(k.rest).toEqual(['the verify phase: passed']);
    expect(checkView(null).changed).toEqual([]);
  });
});

describe('a screenshot id opens the image', () => {
  it('shot-7 takes the render_page call\'s own image and pane', () => {
    const c = applyEvents(conv(), [started, ...workerEvents, done]);
    const shots = shotsOf(task(c), 'c1');
    expect(shots).toHaveLength(1);
    expect(shots[0].id).toBe('shot-7');
    expect(shots[0].src).toBe('/v1/chat-app/attachments/shot-7');
    expect(shots[0].pane).toEqual({ kind: 'image', ref: 'file:/workspace/.tandem/shots/c-31.png', title: 'shot-7' });
  });

  it('without the transcript it opens the file under .tandem/shots/', () => {
    const shots = shotsOf({ report: REPORT, check: CHECK, messages: undefined }, 'c1');
    expect(shots[0].pane).toEqual({ kind: 'image', ref: 'file:/workspace/.tandem/shots/shot-7.png', title: 'shot-7' });
    expect(shots[0].src).toMatch(/\/conversations\/c1\/sandbox\/view\/\.tandem\/shots\/shot-7\.png$/);
  });
});

describe('a worker asks for permission', () => {
  const asking: ChatEvent[] = [
    started,
    { type: 'part.added', message_id: 'w1', part: { type: 'tool_call', id: 'c-12', name: 'bash', input: '{"command":"npm install lil-gui"}', output: null, status: 'running', started_at: null, ended_at: null }, task_id: 't2' },
    { type: 'task.phase', task_id: 't2', phase: 'permission', now: 'Waiting for permission: bash', step: 12, max_steps: 40, tool: { call_id: 'c-12', name: 'bash' } },
    { type: 'tool.asking', message_id: 'w1', call_id: 'c-12', summary: 'bash wants to run: npm install lil-gui', risk: 'exec' },
  ];

  it('the card says "Task 2 asks" with the command, and the chat waits', () => {
    const c = applyEvents(conv(), asking);
    const t = task(c);
    expect(taskStatus(t)).toEqual({ key: 'waiting', word: 'waiting for permission', tone: 'warn' });
    const a = askingCall(t)!;
    expect(a.call.id).toBe('c-12');
    expect(a.call.permission?.summary).toBe('bash wants to run: npm install lil-gui');
    expect(taskAsking(c)).toBe('Task 2 asks: bash wants to run: npm install lil-gui');
    expect(c.status).toBe('waiting');
    expect(c.messages).toEqual(conv().messages);
  });

  it('once answered the prompt closes and the chat runs again', () => {
    const c = applyEvents(conv(), [...asking, { type: 'tool.started', message_id: 'w1', call_id: 'c-12', approved: 'user' }]);
    expect(askingCall(task(c))).toBeNull();
    expect(c.status).toBe('running');
    expect(taskAsking({ tasks: [{ ...task(c), phase: 'tool' }] })).toBeNull();
  });
});

describe('a device that attaches mid-task', () => {
  it('the list, the transcript and the events after its seq give the same card as the stream did', () => {
    const stamped: StampedEvent[] = [started, ...workerEvents, phase12, metrics].map((event, i) => ({ id: String(i + 1), event }));
    const watcher = applyEvents(conv(), stamped.map((s) => s.event));
    // the phone: the snapshot (no tasks), then GET .../tasks, then the task at seq 5 and the tail
    const at = 5;
    const sofar = applyEvents(conv(), stamped.slice(0, at).map((s) => s.event));
    const phone0 = { ...conv(), tasks: mergeTasks(undefined, [{ ...task(watcher), messages: undefined }]) };
    const phone = adoptTask(phone0, { ...task(sofar), seq: String(at) }, stamped);
    // the tool events stamp their own clock: the two copies differ there only
    const unstamped = (t: Task) => JSON.parse(JSON.stringify(t.messages), (k, v) => (k === 'started_at' || k === 'ended_at' || k === 'created_at' ? null : v));
    expect(unstamped(task(phone))).toEqual(unstamped(task(watcher)));
    expect(callsWord(task(phone))).toBe(callsWord(task(watcher)));
    expect(tokensWord(task(phone))).toBe(tokensWord(task(watcher)));
  });

  it('the list does not undo what the stream has run ahead with', () => {
    const live = applyEvents(conv(), [started, ...workerEvents, done]);
    const merged = mergeTasks(live.tasks, [{ ...TASK }]);
    expect(merged[0].status).toBe('done');
    expect(merged[0].messages).toHaveLength(1);
    const ended = mergeTasks([{ ...TASK, messages: [] }], [{ ...TASK, status: 'failed' }]);
    expect(ended[0].status).toBe('failed');
  });

  it('a transcript the stream is building stays when the fetch comes late', () => {
    const live = applyEvents(conv(), [started, ...workerEvents]);
    expect(adoptTask(live, { ...TASK, messages: [] })).toBe(live);
  });
});

describe('the card\'s states', () => {
  it('a finished card is collapsed to its status and summary line', () => {
    const t = { ...TASK, status: 'done', report: REPORT };
    expect(summaryLine(t)).toBe('The banding came from 8-bit normals; they are float now.');
    expect(taskStatus(t).word).toBe('done');
  });

  it('partial at the step limit, failed, cancelled, interrupted, and a report built by chatd', () => {
    expect(taskStatus({ status: 'partial (step limit)' })).toEqual({ key: 'partial', word: 'partial (step limit)', tone: 'warn' });
    expect(taskStatus({ status: 'failed' }).tone).toBe('err');
    expect(taskStatus({ status: 'cancelled' })).toEqual({ key: 'cancelled', word: 'cancelled', tone: 'muted' });
    expect(taskStatus({ status: 'interrupted' }).key).toBe('interrupted');
    expect(summaryLine({ status: 'cancelled', report: null, error: null, steps: 9 })).toBe('cancelled by the user at call 9');
    expect(summaryLine({ status: 'interrupted', report: null, error: null, steps: 9 })).toMatch(/Continue resumes it/);
    expect(summaryLine({ status: 'failed', report: null, error: 'the sandbox stopped', steps: 2 })).toBe('the sandbox stopped');
    expect(sourceWord({ report_source: 'chatd' })).toMatch(/built by chatd/);
    expect(sourceWord({ report_source: 'worker' })).toBeNull();
  });
});

describe('the mock tier', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it('c_ocean serves a done, a partial, a failed and a cancelled task, without them in the snapshot', async () => {
    const mock = new MockChatBackend();
    const snap = await mock.getConversation('c_ocean');
    expect(snap.tasks).toBeUndefined();
    expect(snap.settings.agents).toBe(true);
    const list = await mock.tasks('c_ocean');
    expect(list.map((t) => taskStatus(t).key)).toEqual(['done', 'partial', 'failed', 'cancelled']);
    expect(list.every((t) => t.messages === undefined)).toBe(true);
    const calls = snap.messages.flatMap((m) => m.parts).filter((p): p is ToolCallPart => p.type === 'tool_call' && p.name === 'task');
    expect(calls.map((p) => p.id)).toEqual(OCEAN_TASKS.map((t) => t.call_id));
    const full = await mock.task('c_ocean', list[1].id);
    expect(full.report_source).toBe('chatd');
    expect(full.messages?.length).toBe(1);
    expect(shotsOf(full, 'c_ocean')[0].pane).toEqual({ kind: 'image', ref: 'image:shot-7', title: 'shot-7' });
    mock.deleteConversation('c_sync');
    mock.deleteConversation('c_rename');
    mock.deleteConversation('c_waves');
  });

  it('c_waves waits for permission at call 12 of 40; allowing it finishes the task and the turn counts the worker', async () => {
    const mock = new MockChatBackend();
    await vi.advanceTimersByTimeAsync(200);
    let c = await mock.getConversation('c_waves');
    c = { ...c, tasks: await mock.tasks('c_waves') };
    expect(c.status).toBe('waiting');
    expect(c.run?.phase).toBe('task');
    expect(c.run?.now).toBe('Task 1: Wave height slider with lil-gui, call 12 of 40: Waiting for permission: bash');
    expect(c.context.workers).toHaveLength(1);
    expect(c.context.workers![0]).toMatchObject({ n: 1, step: 12, max_steps: 40, phase: 'permission' });
    const t = await mock.task('c_waves', c.tasks![0].id);
    c = adoptTask(c, t);
    expect(taskStatus(c.tasks![0]).key).toBe('waiting');
    const ask = askingCall(c.tasks![0])!;
    expect(ask.call.name).toBe('bash');
    const box = { conv: c };
    mock.subscribe('c_waves', c.seq, { onEvent: (e) => (box.conv = applyEvent(box.conv, e.event)), onCurrent: () => undefined, onState: () => undefined });
    await vi.advanceTimersByTimeAsync(200);
    await mock.permission('c_waves', ask.call.id, 'once');
    await vi.advanceTimersByTimeAsync(30_000);
    const end = box.conv;
    expect(end.run).toBeNull();
    expect(end.tasks![0].status).toBe('done');
    expect(checkView(end.tasks![0].check).unreported).toEqual(['package-lock.json']);
    expect(end.context.workers).toEqual([]);
    expect(end.last_worker?.task_id).toBe(end.tasks![0].id);
    const stats = end.messages.at(-1)!.stats!;
    expect(stats.workers?.count).toBe(1);
    expect(stats.prompt_tokens).toBeGreaterThan(stats.workers!.prompt_tokens);
    expect(end.messages.flatMap((m) => m.parts).some((p) => p.type === 'tool_call' && p.name === 'bash')).toBe(false);
    mock.deleteConversation('c_sync');
    mock.deleteConversation('c_rename');
    mock.deleteConversation('c_waves');
  });

  it('Stop while a worker asks cancels the task and stops the turn', async () => {
    const mock = new MockChatBackend();
    await vi.advanceTimersByTimeAsync(200);
    await mock.stop('c_waves');
    await vi.advanceTimersByTimeAsync(2_000);
    const list = await mock.tasks('c_waves');
    expect(list[0].status).toBe('cancelled');
    const c = await mock.getConversation('c_waves');
    expect(c.run).toBeNull();
    expect(c.messages.at(-1)!.status).toBe('stopped');
    mock.deleteConversation('c_sync');
    mock.deleteConversation('c_rename');
  });
});
