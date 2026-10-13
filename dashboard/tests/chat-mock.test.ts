// the mock chat backend: the seeds, a scripted run end to end under fake timers, the permission
// prompt, stop, the queue, branches, the catch-up from Last-Event-ID, and the settings stores.
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { MockChatBackend } from '../src/chat/mock';
import { VersionConflict } from '../src/chat/backend';
import { applyEvent } from '../src/chat/store';
import type { Conversation, StampedEvent, ToolCallPart } from '../src/chat/types';

async function flush(ms: number): Promise<void> {
  await vi.advanceTimersByTimeAsync(ms);
}

/** a mock call that waits on a timer: start it, let the clock run, then read it */
async function settle<T>(p: Promise<T>, ms = 1_000): Promise<T> {
  await flush(ms);
  return p;
}

/** attach to a conversation and fold its events into a conversation */
function follow(mock: MockChatBackend, id: string, start: Conversation, lastId: string | null = null) {
  const box = { conv: start, replayed: 0, live: 0, current: false, states: [] as string[], ids: [] as string[] };
  const sub = mock.subscribe(id, lastId, {
    onEvent: (e: StampedEvent, replay) => {
      box.conv = applyEvent(box.conv, e.event);
      box.ids.push(e.id);
      if (replay) box.replayed++;
      else box.live++;
    },
    onCurrent: () => (box.current = true),
    onState: (s) => box.states.push(s),
  });
  return { box, sub };
}

describe('the mock chat backend', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it('seeds generic conversations, folders, and one run that is already on', async () => {
    const mock = new MockChatBackend();
    const { conversations, folders } = await mock.listConversations();
    expect(conversations.length).toBeGreaterThanOrEqual(7);
    expect(folders.map((f) => f.name)).toEqual(['Engine', 'Everyday']);
    const running = conversations.find((c) => c.id === mock.attachedId)!;
    expect(running.status).toBe('running');
    expect(running.now).toMatch(/Running bash|Calling bash/);
    const waiting = conversations.find((c) => c.id === 'c_rename')!;
    expect(waiting.status).toBe('waiting');
    const full = await mock.getConversation(mock.attachedId);
    await flush(100);
    expect(full.run?.started_by).toBe('another browser');
    const call = full.messages.at(-1)!.parts.find((p) => p.type === 'tool_call' && p.status === 'running') as ToolCallPart;
    expect(call.name).toBe('bash');
    expect(call.output?.content.split('\n').length).toBeGreaterThanOrEqual(3);
    mock.deleteConversation(mock.attachedId);
    mock.deleteConversation('c_rename');
  });

  it('a new chat in ask mode: send runs the script, asks for bash, continues on allow, and finishes with figures', async () => {
    const mock = new MockChatBackend();
    // ask mode: the default (auto) would run bash without a prompt (tests/chat-permission.test.ts)
    const c = await mock.createConversation({ permission_mode: 'ask' });
    const { box, sub } = follow(mock, c.id, c);
    await flush(200);
    await mock.send(c.id, { text: 'Fix the flaky test', client_msg_id: 'cm_1' });
    await flush(30_000);
    const asking = box.conv.messages.at(-1)!.parts.find((p) => p.type === 'tool_call' && p.status === 'asking') as ToolCallPart | undefined;
    expect(asking?.name).toBe('bash');
    expect(box.conv.status).toBe('waiting');
    expect(box.conv.title).toBe('Fix the flaky test');
    await mock.permission(c.id, asking!.id, 'always');
    await flush(100);
    expect(box.conv.settings.policies.bash).toBe('always');
    await flush(40_000);
    // edit_file asks too (its policy is ask); allow it once
    const edit = box.conv.messages.at(-1)!.parts.find((p) => p.type === 'tool_call' && p.status === 'asking') as ToolCallPart | undefined;
    expect(edit?.name).toBe('edit_file');
    await mock.permission(c.id, edit!.id, 'once');
    await flush(120_000);
    expect(box.conv.settings.policies.edit_file).toBeUndefined();
    // the second bash call ran without asking (the policy is always now)
    const calls = box.conv.messages.at(-1)!.parts.filter((p) => p.type === 'tool_call') as ToolCallPart[];
    expect(calls.filter((k) => k.name === 'bash')).toHaveLength(2);
    expect(calls.every((k) => k.status === 'done' || k.status === 'failed')).toBe(true);
    expect(box.conv.status).toBe('idle');
    expect(box.conv.messages.at(-1)!.stats?.finish).toBe('stop');
    expect(box.conv.usage.turns).toBe(1);
    expect(box.conv.context.used).toBeGreaterThan(0);
    sub.close();
    mock.deleteConversation(mock.attachedId);
    mock.deleteConversation('c_rename');
  });

  it('deny skips the tool and the model says what to do by hand; stop ends the turn as stopped', async () => {
    const mock = new MockChatBackend();
    const c = await mock.createConversation();
    const { box, sub } = follow(mock, c.id, c);
    await flush(200);
    await mock.send(c.id, { text: 'rename the module', client_msg_id: 'cm_2' });
    await flush(20_000);
    const asking = box.conv.messages.at(-1)!.parts.find((p) => p.type === 'tool_call' && p.status === 'asking') as ToolCallPart;
    expect(asking.name).toBe('edit_file');
    await mock.permission(c.id, asking.id, 'deny');
    await flush(2_000);
    expect((box.conv.messages.at(-1)!.parts.find((p) => p.type === 'tool_call' && p.name === 'edit_file') as ToolCallPart).status).toBe('denied');
    await mock.stop(c.id);
    await flush(500);
    expect(box.conv.messages.at(-1)!.status).toBe('stopped');
    expect(box.conv.run).toBeNull();
    sub.close();
    mock.deleteConversation(mock.attachedId);
    mock.deleteConversation('c_rename');
  });

  it('a message sent during a run is queued and goes in after the turn', async () => {
    const mock = new MockChatBackend();
    const c = await mock.createConversation();
    const { box, sub } = follow(mock, c.id, c);
    await flush(200);
    await mock.send(c.id, { text: 'hello', client_msg_id: 'cm_3' });
    await flush(300);
    // a retried send with the same client id makes no second message
    await mock.send(c.id, { text: 'hello', client_msg_id: 'cm_3' });
    await mock.send(c.id, { text: 'and plan my week', client_msg_id: 'cm_4' });
    await mock.send(c.id, { text: 'and plan my week', client_msg_id: 'cm_4' });
    expect(box.conv.queue).toHaveLength(1);
    await flush(60_000);
    expect(box.conv.queue).toHaveLength(0);
    expect(box.conv.messages.filter((m) => m.role === 'user').map((m) => (m.parts[0] as { text: string }).text)).toEqual(['hello', 'and plan my week']);
    sub.close();
    mock.deleteConversation(mock.attachedId);
    mock.deleteConversation('c_rename');
  });

  it('an edit makes a branch with a sibling pager, and switching shows the other version', async () => {
    const mock = new MockChatBackend();
    const c = await mock.getConversation('c_explain');
    const user = c.messages[0];
    await mock.edit('c_explain', user.id, 'Explain acceptance instead.', 'cm_5');
    await flush(40_000);
    const after = await mock.getConversation('c_explain');
    expect(after.messages[0].siblings).toEqual({ index: 1, count: 2 });
    expect((after.messages[0].parts[0] as { text: string }).text).toBe('Explain acceptance instead.');
    const back = await mock.switchBranch('c_explain', after.messages[0].id, 0);
    expect(back.messages[0].siblings).toEqual({ index: 0, count: 2 });
    expect((back.messages[0].parts[0] as { text: string }).text).toContain('tokens per round');
    mock.deleteConversation(mock.attachedId);
    mock.deleteConversation('c_rename');
  });

  it('a catch-up from Last-Event-ID replays only what was missed, then says current', async () => {
    const mock = new MockChatBackend();
    const c = await mock.createConversation();
    const first = follow(mock, c.id, c);
    await flush(200);
    await mock.send(c.id, { text: 'hi', client_msg_id: 'cm_6' });
    await flush(1_500);
    const seen = first.box.ids.at(-1)!;
    first.sub.close();
    await flush(3_000);
    const snapshot = await mock.getConversation(c.id);
    const second = follow(mock, c.id, first.box.conv, seen);
    await flush(2_000);
    expect(second.box.replayed).toBeGreaterThan(0);
    expect(second.box.current).toBe(true);
    expect(second.box.ids[0]).toBe(String(Number(seen) + 1));
    // the replay lands on the same text the server holds
    expect((second.box.conv.messages.at(-1)!.parts[0] as { text: string }).text.length).toBeGreaterThanOrEqual((snapshot.messages.at(-1)!.parts[0] as { text: string }).text.length);
    second.sub.close();
    mock.deleteConversation(mock.attachedId);
    mock.deleteConversation('c_rename');
  });

  it('settings stores: tools, packages, MCP servers, prompts, presets, compaction', async () => {
    const mock = new MockChatBackend();
    const t = await mock.setTool('bash', { policy: 'never' });
    expect(t.policy).toBe('never');
    expect((await mock.packages()).some((p) => p.enabled)).toBe(true);
    const bad = await settle(mock.testMcpServer({ name: 'x', transport: 'http', url: 'localhost:1' }));
    expect(bad.ok).toBe(false);
    const good = await settle(mock.testMcpServer({ name: 'files', transport: 'stdio', command: 'files-mcp', args: [] }));
    expect(good.ok).toBe(true);
    expect(good.tools).toHaveLength(2);
    const added = await mock.addMcpServer({ name: 'files', transport: 'stdio', command: 'files-mcp', args: [] });
    expect(added.status).toBe('starting');
    await flush(3_000);
    expect((await mock.mcpServers()).find((s) => s.id === added.id)?.status).toBe('connected');
    const p = await mock.savePrompt({ title: 'T', command: 't', body: 'b' });
    expect((await mock.prompts()).some((x) => x.id === p.id)).toBe(true);
    await mock.deletePrompt(p.id);
    expect((await mock.prompts()).some((x) => x.id === p.id)).toBe(false);
    const c = await mock.getConversation('c_exl3');
    expect(c.compactions).toHaveLength(1);
    await mock.compact('c_exl3');
    expect((await mock.getConversation('c_exl3')).compactions).toHaveLength(2);
    mock.deleteConversation(mock.attachedId);
    mock.deleteConversation('c_rename');
  });

  it('the sandbox: files, a read, an artifact, a shell that echoes', async () => {
    const mock = new MockChatBackend();
    const root = await mock.sandboxFiles('c_explain', '/work/sync');
    expect(root.map((f) => f.path.split('/').pop())).toEqual(['src', 'tests', 'pyproject.toml', 'README.md']);
    const f = await mock.readFile('c_explain', '/work/sync/src/sync/queue.py');
    expect(f.mime).toBe('text/x-python');
    expect((await mock.artifact('c_explain', 'preview:shop-page')).mime).toBe('text/html');
    let out = '';
    const term = mock.terminal('c_explain', 'shell', (d) => (out += d));
    term.write('pwd\r');
    await flush(500);
    expect(out).toContain('/work/sync\x1b[0m$ ');
    expect(out).toContain('pwd');
    expect(out.split('/work/sync').length).toBeGreaterThan(2);
    term.close();
    mock.deleteConversation(mock.attachedId);
    mock.deleteConversation('c_rename');
  });

  it('v0.2: version conflicts, the pane layout on the server, Continue, Send now, the seeds the contract asks for', async () => {
    const mock = new MockChatBackend();
    const c = await mock.getConversation('c_explain');
    expect(c.version).toBe(1);
    const s1 = await mock.updateConversation(c.id, { version: 1, title: 'renamed' });
    expect(s1.version).toBe(2);
    await expect(mock.updateConversation(c.id, { version: 1, title: 'stale' })).rejects.toBeInstanceOf(VersionConflict);
    const s2 = await mock.updateConversation(c.id, { version: 2, ui: { panes: [{ kind: 'terminal', ref: 'shell', title: 'Terminal' }], active: null, split: 0.6 } });
    expect(s2.version).toBe(3);
    const after = await mock.getConversation(c.id);
    expect(after.ui.panes[0].id).toBeTruthy();
    // the interrupted answer and Continue
    const sql = await mock.getConversation(mock.interruptedId);
    expect(sql.messages.at(-1)!.status).toBe('interrupted');
    const { box, sub } = follow(mock, sql.id, sql);
    await flush(200);
    await mock.continueRun(sql.id);
    await flush(30_000);
    expect(box.conv.messages.at(-1)!.status).toBe('done');
    expect(box.conv.messages).toHaveLength(sql.messages.length);
    expect((box.conv.messages.at(-1)!.parts[0] as { text: string }).text).toContain('CREATE INDEX CONCURRENTLY');
    sub.close();
    // the seeds: a cold cache, a sandbox stopped by the memory guard, a queued message under the running one
    expect((await mock.getConversation('c_exl3')).context.cache.cold).toBe(150_000);
    expect((await mock.getConversation('c_meal')).sandbox.stopped_reason).toBe('memory_guard');
    const sync = await mock.getConversation(mock.attachedId);
    expect(sync.queue).toHaveLength(1);
    expect(sync.ui.panes[0].kind).toBe('terminal');
    // Send now: the current answer stops and the queued message goes in
    const f2 = follow(mock, sync.id, sync);
    await flush(200);
    await mock.sendNow(sync.id, sync.queue[0].id);
    await flush(3_000);
    const users = f2.box.conv.messages.filter((m) => m.role === 'user');
    expect(users).toHaveLength(2);
    expect((users[1].parts[0] as { text: string }).text).toContain('ruff');
    f2.sub.close();
    mock.deleteConversation(mock.attachedId);
    mock.deleteConversation('c_rename');
  });
});
