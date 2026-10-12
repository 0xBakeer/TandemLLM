// the chat reducer: a run's events in order, a tool call's life, a catch-up replay equals the
// live path, and the attachment state machine's words.
import { describe, expect, it, onTestFinished, vi } from 'vitest';
import { adopt, applyEvent, applyEvents, firstLine, interruptedMessage, lastStats, messageText, pendingPermission } from '../src/chat/store';
import { attachLabel, initialAttachment, onEvent, onStreamState, onTick, LOST_AFTER_MS } from '../src/chat/connection';
import type { ChatEvent, Conversation, Run, StampedEvent, TurnStats } from '../src/chat/types';

function conv(): Conversation {
  return {
    id: 'c1',
    title: 't',
    created_at: '2026-10-11T10:00:00Z',
    updated_at: '2026-10-11T10:00:00Z',
    pinned: false,
    folder_id: null,
    status: 'idle',
    preview: '',
    model: 'TandemLLM',
    profile: 'coding',
    turns: 0,
    archived: false,
    version: 1,
    unread: false,
    head_id: 'u1',
    ui: { panes: [], active: null, split: 0.6 },
    seq: '0',
    messages: [{ id: 'u1', role: 'user', created_at: '2026-10-11T10:00:00Z', parent_id: null, run_id: null, parts: [{ type: 'text', text: 'fix the test\nplease' }], status: 'done' }],
    settings: { model: 'TandemLLM', profile: 'coding', effort: 'high', thinking: true, tools: ['bash'], packages: [], packages_blocked: [], packages_mode: 'auto', mcp: [], prompt_id: null, policies: {}, auto_accept_edits: false, max_steps: 50, sandbox_network: 'off' },
    run: null,
    context: { window: 262144, used: 100, parts: [], compact_at: 200000, cache: { warm: 0, cold: 100, cold_eta_ms: null, source: 'none', slot: null }, harness: null, memory: { kv_bytes: null, resident_bytes: null, sandbox_disk_bytes: null }, model: { name: 'TandemLLM', profile: 'coding', weights: 'EXL3', drafter: null, max_len: 262144, effort: 'high', thinking: true } },
    sandbox: { state: 'stopped', image: '', cwd: '/', cpus: 1, memory_bytes: 0, memory_limit_bytes: 1, disk_bytes: 0, disk_limit_bytes: 1, network: 'off', allowlist: [], uptime_s: 0, error: null },
    compactions: [],
    usage: { turns: 0, prompt_tokens: 0, completion_tokens: 0, reasoning_tokens: 0, cached_tokens: 0, tool_calls: 0, wall_s: 0 },
    queue: [],
  };
}

const run: Run = { id: 'r1', conversation_id: 'c1', message_id: 'a1', started_at: '2026-10-11T10:00:01Z', started_by: null, phase: 'queued', now: 'Queued', step: 0, max_steps: 50, queue_position: null, tool: null, prefill: null, decode_tps: null, tokens_per_round: null, acceptance: null, thinking: { used: 0, budget: 8192 }, tokens: 0, tps_history: [] };
const stats: TurnStats = { finish: 'stop', ttft_ms: 200, decode_tps: 55, prompt_tokens: 100, completion_tokens: 50, reasoning_tokens: 10, cached_tokens: 90, tokens_per_round: 4.2, acceptance: 0.7, total_ms: 900 };

const TURN: ChatEvent[] = [
  { type: 'run.started', run },
  { type: 'message.created', message: { id: 'a1', role: 'assistant', created_at: '2026-10-11T10:00:01Z', parent_id: 'u1', run_id: 'r1', parts: [], status: 'streaming' } },
  { type: 'run.phase', run_id: 'r1', phase: 'thinking', now: 'Thinking' },
  { type: 'part.added', message_id: 'a1', part: { type: 'thinking', text: '', tokens: null, done: false } },
  { type: 'part.delta', message_id: 'a1', index: 0, delta: 'read the ' },
  { type: 'part.delta', message_id: 'a1', index: 0, delta: 'test first' },
  { type: 'part.done', message_id: 'a1', index: 0, tokens: 6 },
  { type: 'part.added', message_id: 'a1', part: { type: 'tool_call', id: 'k1', name: 'bash', input: '', output: null, status: 'running', started_at: null, ended_at: null } },
  { type: 'part.delta', message_id: 'a1', index: 1, delta: '{"command":', field: 'input' },
  { type: 'part.delta', message_id: 'a1', index: 1, delta: '"pytest -q"}', field: 'input' },
  { type: 'tool.asking', message_id: 'a1', call_id: 'k1', summary: 'bash wants to run: pytest -q' },
  { type: 'run.phase', run_id: 'r1', phase: 'permission', now: 'Waiting for permission: bash', tool: { call_id: 'k1', name: 'bash' } },
  { type: 'tool.started', message_id: 'a1', call_id: 'k1', pane: { kind: 'terminal', ref: 'k1' } },
  { type: 'run.phase', run_id: 'r1', phase: 'tool', now: 'Running bash: pytest -q', tool: { call_id: 'k1', name: 'bash' } },
  { type: 'tool.output', message_id: 'a1', call_id: 'k1', delta: '2 passed' },
  { type: 'tool.done', message_id: 'a1', call_id: 'k1', status: 'done', is_error: false },
  { type: 'run.metrics', run_id: 'r1', decode_tps: 54.2, tokens_per_round: 4.1, acceptance: 0.7, tokens: 40, thinking_used: 6 },
  { type: 'part.added', message_id: 'a1', part: { type: 'text', text: '' } },
  { type: 'part.delta', message_id: 'a1', index: 2, delta: 'All green.' },
  { type: 'run.done', run_id: 'r1', message_id: 'a1', stats, status: 'done' },
];

describe('the chat reducer', () => {
  it('plays a whole turn: thinking, a tool call with permission, text, done', () => {
    const c = applyEvents(conv(), TURN);
    const a = c.messages[1];
    expect(a.status).toBe('done');
    expect(a.parts).toHaveLength(3);
    expect(a.parts[0]).toMatchObject({ type: 'thinking', text: 'read the test first', done: true, tokens: 6 });
    expect(a.parts[1]).toMatchObject({ type: 'tool_call', name: 'bash', input: '{"command":"pytest -q"}', status: 'done', output: { content: '2 passed', is_error: false } });
    expect((a.parts[1] as { pane?: unknown }).pane).toEqual({ kind: 'terminal', ref: 'k1' });
    expect(a.parts[2]).toMatchObject({ type: 'text', text: 'All green.' });
    expect(a.stats).toEqual(stats);
    expect(c.run).toBeNull();
    expect(c.status).toBe('idle');
    expect(c.turns).toBe(1);
  });

  it('the permission prompt is the waiting state, and it clears when the tool starts', () => {
    const asking = applyEvents(conv(), TURN.slice(0, 12));
    expect(asking.status).toBe('waiting');
    expect(asking.run?.phase).toBe('permission');
    const p = pendingPermission(asking);
    expect(p?.call.id).toBe('k1');
    expect(p?.call.permission?.summary).toBe('bash wants to run: pytest -q');
    const started = applyEvents(asking, TURN.slice(12, 14));
    expect(started.status).toBe('running');
    expect(pendingPermission(started)).toBeNull();
    expect(started.run?.now).toBe('Running bash: pytest -q');
  });

  it('metrics keep a 120-slot history and the phase keeps the tool during a tool phase', () => {
    let c = applyEvents(conv(), TURN.slice(0, 14));
    for (let i = 0; i < 130; i++) c = applyEvent(c, { type: 'run.metrics', run_id: 'r1', decode_tps: i, tokens_per_round: 4, acceptance: 0.7, tokens: i, thinking_used: 0 });
    expect(c.run?.tps_history).toHaveLength(120);
    expect(c.run?.tps_history[119]).toBe(129);
    expect(c.run?.tool?.name).toBe('bash');
    c = applyEvent(c, { type: 'run.phase', run_id: 'r1', phase: 'writing', now: 'Writing' });
    expect(c.run?.tool).toBeNull();
  });

  it('a denied tool keeps no output; a failed one marks the output as an error', () => {
    const base = applyEvents(conv(), TURN.slice(0, 11));
    const denied = applyEvent(base, { type: 'tool.done', message_id: 'a1', call_id: 'k1', status: 'denied', is_error: false });
    expect(denied.messages[1].parts[1]).toMatchObject({ status: 'denied', output: null, permission: null });
    const failed = applyEvents(base, [
      { type: 'tool.started', message_id: 'a1', call_id: 'k1' },
      { type: 'tool.output', message_id: 'a1', call_id: 'k1', delta: 'E assert' },
      { type: 'tool.done', message_id: 'a1', call_id: 'k1', status: 'failed', is_error: true },
    ]);
    expect(failed.messages[1].parts[1]).toMatchObject({ status: 'failed', output: { content: 'E assert', is_error: true } });
  });

  it('an error ends the run with an error part; a replay of the same events gives the same conversation', () => {
    const live = applyEvents(conv(), TURN.slice(0, 7));
    const err = applyEvent(live, { type: 'run.error', run_id: 'r1', message_id: 'a1', message: 'engine restarted', retryable: true });
    expect(err.status).toBe('error');
    expect(err.run).toBeNull();
    expect(err.messages[1].parts.at(-1)).toEqual({ type: 'error', message: 'engine restarted', retryable: true });
    const replay = TURN.slice(0, 7).reduce(applyEvent, conv());
    expect(replay).toEqual(live);
  });

  it('events for an unknown run or message leave the conversation alone', () => {
    const c = conv();
    expect(applyEvent(c, { type: 'run.phase', run_id: 'nope', phase: 'writing', now: 'x' })).toBe(c);
    expect(applyEvent(c, { type: 'part.delta', message_id: 'nope', index: 0, delta: 'x' })).toEqual(c);
    const twice = applyEvents(c, [TURN[1], TURN[1]]);
    expect(twice.messages).toHaveLength(2);
  });

  it('queue, context, compaction, settings and sandbox events replace their slices', () => {
    let c = conv();
    c = applyEvent(c, { type: 'queue.changed', queue: [{ id: 'q1', text: 'also check lint', created_at: 'x', client_msg_id: 'cm_1' }] });
    expect(c.queue).toHaveLength(1);
    c = applyEvent(c, { type: 'compaction', compaction: { id: 'cp1', at: 'x', before_tokens: 100, after_tokens: 40, summary: 's' } });
    c = applyEvent(c, { type: 'compaction', compaction: { id: 'cp1', at: 'x', before_tokens: 100, after_tokens: 40, summary: 's' } });
    expect(c.compactions).toHaveLength(1);
    c = applyEvent(c, { type: 'settings.changed', settings: { ...c.settings, effort: 'low' } });
    expect(c.settings.effort).toBe('low');
    c = applyEvent(c, { type: 'conversation.updated', patch: { title: 'renamed', pinned: true } });
    expect(c.title).toBe('renamed');
    expect(c.pinned).toBe(true);
  });

  it('v0.2 events: interrupted ends as an error state, cancelled tools, ui.changed, branch.changed moves the head', () => {
    let c = applyEvents(conv(), TURN.slice(0, 14));
    c = applyEvent(c, { type: 'tool.done', message_id: 'a1', call_id: 'k1', status: 'cancelled', is_error: false });
    expect(c.messages[1].parts[1]).toMatchObject({ status: 'cancelled', output: null });
    c = applyEvent(c, { type: 'run.done', run_id: 'r1', message_id: 'a1', stats, status: 'interrupted' });
    expect(c.status).toBe('error');
    expect(c.messages[1].status).toBe('interrupted');
    expect(interruptedMessage(c)?.id).toBe('a1');
    c = applyEvent(c, { type: 'ui.changed', ui: { panes: [{ id: 'p1', kind: 'terminal', ref: 'shell' }], active: 'p1', split: 0.5 } });
    expect(c.ui.active).toBe('p1');
    c = applyEvent(c, { type: 'branch.changed', head_id: 'zzz' });
    expect(c.head_id).toBe('zzz');
    // the output in tool.done wins over the streamed one
    const d = applyEvents(conv(), [...TURN.slice(0, 15), { type: 'tool.done', message_id: 'a1', call_id: 'k1', status: 'done', is_error: false, output: { content: 'final', is_error: false, bytes: 5, exit_code: 0 } }]);
    expect((d.messages[1].parts[1] as { output: { content: string } }).output.content).toBe('final');
  });

  it('a snapshot that arrives after some of the stream: the events past its seq go on top, the rest are not applied twice', () => {
    // a tool's started_at and ended_at come from the clock: one fixed instant for both reductions
    vi.useFakeTimers({ toFake: ['Date'] });
    vi.setSystemTime(new Date('2026-10-12T10:00:00Z'));
    onTestFinished(() => vi.useRealTimers());
    const stamped: StampedEvent[] = TURN.map((event, i) => ({ id: String(i + 1), event }));
    const full = applyEvents(conv(), TURN);
    // the server's snapshot at seq 6 (the thinking part streamed), while the device already had 1..10
    const snap: Conversation = { ...applyEvents(conv(), TURN.slice(0, 6)), seq: '6' };
    const { conv: got, floor } = adopt(snap, stamped.slice(0, 10));
    expect(floor).toBe(10);
    expect(got.seq).toBe('10');
    const rest = applyEvents(got, TURN.slice(10));
    expect(rest.messages).toEqual(full.messages);
    expect((rest.messages[1].parts[0] as { text: string }).text).toBe('read the test first');
    // a snapshot newer than everything the device has: nothing to add
    const ahead = adopt({ ...full, seq: String(TURN.length) }, stamped.slice(0, 5));
    expect(ahead.floor).toBe(TURN.length);
    expect(ahead.conv.messages).toEqual(full.messages);
  });

  it('text helpers: the first line, the copy text, the last figures', () => {
    const c = applyEvents(conv(), TURN);
    expect(firstLine(c.messages[0])).toBe('fix the test');
    expect(messageText(c.messages[1])).toBe('All green.');
    expect(lastStats(c)?.decode_tps).toBe(55);
    expect(lastStats(conv())).toBeNull();
  });
});

describe('the attachment to a running chat', () => {
  it('a fresh connection goes connecting, then live', () => {
    let a = initialAttachment(0);
    a = onStreamState(a, 'connecting', false, 0);
    expect(attachLabel(a).text).toBe('connecting');
    a = onStreamState(a, 'open', false, 10);
    expect(attachLabel(a)).toEqual({ text: 'live', tone: 'live' });
  });

  it('a resume replays first: catching up with the count, live at the current marker', () => {
    let a = onStreamState(initialAttachment(0), 'open', true, 0);
    expect(attachLabel(a).text).toBe('catching up');
    a = onEvent(a, false, 1);
    a = onEvent(a, false, 2);
    expect(attachLabel(a).text).toBe('catching up, 2 events');
    expect(a.current).toBe(false);
    a = onEvent(a, true, 3);
    expect(a.state).toBe('live');
    expect(a.current).toBe(true);
  });

  it('a dropped stream says reconnecting with the seconds, then lost after a minute', () => {
    let a = onStreamState(initialAttachment(0), 'open', false, 0);
    a = onStreamState(a, 'reconnecting', false, 1000);
    expect(attachLabel(a, 5000)).toEqual({ text: 'reconnecting since 4 s', tone: 'warn' });
    // a second drop while reconnecting keeps the first clock
    a = onStreamState(a, 'error', false, 20_000);
    expect(a.since).toBe(1000);
    a = onTick(a, 1000 + LOST_AFTER_MS);
    expect(attachLabel(a).text).toBe('stream lost, refresh');
    const back = onStreamState(a, 'open', true, 70_000);
    expect(back.state).toBe('catching-up');
  });

  it('a live event while reconnecting means the stream is back', () => {
    let a = onStreamState(initialAttachment(0), 'open', false, 0);
    a = onStreamState(a, 'reconnecting', false, 1);
    a = onEvent(a, true, 2);
    expect(a.state).toBe('live');
    expect(onStreamState(a, 'closed', false, 3).state).toBe('idle');
    expect(onStreamState(a, 'unauthorized', false, 3).state).toBe('lost');
  });
});

describe('the compaction divider', () => {
  const cp = { id: 'cp1', at: '2026-10-12T00:00:00Z', before_tokens: 214_000, after_tokens: 18_000, summary: 's', duration_ms: 1_200 };
  it('sits above the first message the model still reads, else above the first message', async () => {
    const { dividersByMessage, dividerWords } = await import('../src/chat/store');
    const c = conv();
    c.messages = [...c.messages, { ...c.messages[0], id: 'u2' }];
    expect([...dividersByMessage({ ...c, compactions: [{ ...cp, boundary_id: 'u2' }] }).keys()]).toEqual(['u2']);
    expect([...dividersByMessage({ ...c, compactions: [{ ...cp, boundary_id: 'gone' }] }).keys()]).toEqual(['u1']);
    expect(dividersByMessage({ ...c, messages: [], compactions: [cp] }).size).toBe(0);
    expect(dividerWords(cp)).toBe('Earlier messages summarized: 214K to 18K tokens, 1.2 s');
  });
});

describe('Open WebUI valves', () => {
  it('sends what was typed, in the valve types', async () => {
    const { valveValues } = await import('../src/chat/types');
    const kit = { valves: [
      { name: 'API_KEY', type: 'string', default: '', description: '', secret: true, value: '', set: false },
      { name: 'TIMEOUT', type: 'integer', default: 10, description: '', secret: false, value: 10, set: true },
      { name: 'DEBUG', type: 'boolean', default: false, description: '', secret: false, value: false, set: false },
    ] };
    expect(valveValues(kit, { API_KEY: 'k', TIMEOUT: '30', DEBUG: 'yes' })).toEqual({ API_KEY: 'k', TIMEOUT: 30, DEBUG: true });
    expect(valveValues(kit, { TIMEOUT: '' })).toEqual({ TIMEOUT: null });
    expect(valveValues(kit, {})).toEqual({});
  });
});

describe('plan mode', () => {
  it('reads the steps of a plan as the server does', async () => {
    const { planItems } = await import('../src/chat/types');
    expect(planItems('Plan:\n1. a\n2) **b**\n\n- not a step')).toEqual(['a', 'b']);
    expect(planItems('- one\n* two\n- [ ] three')).toEqual(['one', 'two', 'three']);
    expect(planItems('just do it')).toEqual(['just do it']);
    expect(planItems('')).toEqual([]);
  });
  it('takes a todos.changed into the conversation', () => {
    const c = applyEvent(conv(), { type: 'todos.changed', todos: [{ id: 't1', text: 'a', state: 'in_progress' }] });
    expect(c.todos).toEqual([{ id: 't1', text: 'a', state: 'in_progress' }]);
  });
});

describe('notifications', () => {
  it('say what happened, never what was written', async () => {
    const { notifyKind, NOTIFY_WORDS } = await import('../src/chat/notify');
    expect(notifyKind({ type: 'run.done', status: 'done' })).toBe('finished');
    expect(notifyKind({ type: 'run.done', status: 'error' })).toBe('error');
    expect(notifyKind({ type: 'run.done', status: 'stopped' })).toBe(null);
    expect(notifyKind({ type: 'tool.asking' })).toBe('permission');
    expect(notifyKind({ type: 'conversation.updated' })).toBe(null);
    expect(NOTIFY_WORDS).toEqual({ finished: 'finished', permission: 'waits for permission', error: 'error' });
  });
});

describe('MCP prompts', () => {
  it('fill their arguments from the text after the command', async () => {
    const { promptArgs } = await import('../src/chat/types');
    expect(promptArgs('app.py', ['file'])).toEqual({ file: 'app.py' });
    expect(promptArgs('lang=py the main loop', ['topic', 'lang'])).toEqual({ lang: 'py', topic: 'the main loop' });
    expect(promptArgs('x=1', ['file'])).toEqual({ file: 'x=1' });
    expect(promptArgs('', ['file'])).toEqual({});
  });
});
