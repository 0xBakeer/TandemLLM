// Permission modes (2026-10-13): the decision before a tool runs, the words, and the mock's runs
// under each mode (auto runs bash without asking and still asks for a write; yolo asks nothing).
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { MockChatBackend } from '../src/chat/mock';
import { approvedTitle, approvedWord, decideCall, modeDesc, modeOf, modeWord, PERMISSION_MODES } from '../src/chat/permission';
import { applyEvent } from '../src/chat/store';
import type { Conversation, StampedEvent, ToolCallPart } from '../src/chat/types';

describe('decideCall', () => {
  it('a never policy refuses in every mode; an always policy runs as a rule', () => {
    for (const m of PERMISSION_MODES) {
      expect(decideCall(m, 'never', 'exec')).toEqual({ run: 'deny', approved: null });
      expect(decideCall(m, 'always', 'write')).toEqual({ run: 'allow', approved: 'rule' });
    }
  });
  it('ask: every tool with an ask policy asks', () => {
    for (const r of ['exec', 'read', 'write', 'network', undefined] as const) expect(decideCall('ask', 'ask', r).run).toBe('ask');
  });
  it('auto: commands, reads and workspace writes run, the internet asks, an unknown risk asks', () => {
    expect(decideCall('auto', 'ask', 'exec')).toEqual({ run: 'allow', approved: 'auto' });
    expect(decideCall('auto', 'ask', 'read')).toEqual({ run: 'allow', approved: 'auto' });
    expect(decideCall('auto', 'ask', 'write')).toEqual({ run: 'allow', approved: 'auto' });
    expect(decideCall('auto', 'ask', 'network').run).toBe('ask');
    expect(decideCall('auto', 'ask', undefined).run).toBe('ask');
  });
  it('auto-accepted edits run as a rule in ask and auto', () => {
    expect(decideCall('ask', 'ask', 'write', true)).toEqual({ run: 'allow', approved: 'rule' });
    expect(decideCall('auto', 'ask', 'write', true)).toEqual({ run: 'allow', approved: 'rule' });
  });
  it('yolo: nothing asks', () => {
    for (const r of ['exec', 'read', 'write', 'network', undefined] as const) expect(decideCall('yolo', 'ask', r)).toEqual({ run: 'allow', approved: 'yolo' });
  });
});

describe('the words', () => {
  it('auto is the default; YOLO is spelt out loud', () => {
    expect(modeOf(null)).toBe('auto');
    expect(modeOf({ permission_mode: undefined })).toBe('auto');
    expect(modeOf({ permission_mode: 'yolo' })).toBe('yolo');
    expect(PERMISSION_MODES.map(modeWord)).toEqual(['ask', 'auto', 'YOLO']);
    for (const m of PERMISSION_MODES) expect(modeDesc(m).length).toBeGreaterThan(20);
  });
  it('the badge: nothing for the user, a word for the rest', () => {
    expect(approvedWord('user')).toBe('');
    expect(approvedWord('rule')).toBe('always');
    expect(approvedWord('auto')).toBe('auto');
    expect(approvedWord('yolo')).toBe('yolo');
    expect(approvedTitle('yolo')).toMatch(/YOLO/);
  });
});

async function flush(ms: number): Promise<void> {
  await vi.advanceTimersByTimeAsync(ms);
}
function follow(mock: MockChatBackend, id: string, start: Conversation) {
  const box = { conv: start };
  const sub = mock.subscribe(id, null, { onEvent: (e: StampedEvent) => (box.conv = applyEvent(box.conv, e.event)), onCurrent: () => {}, onState: () => {} });
  return { box, sub };
}
const calls = (c: Conversation) => c.messages.at(-1)!.parts.filter((p): p is ToolCallPart => p.type === 'tool_call');

describe('the mock under each mode', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it('auto (the default): bash and edit_file run without asking, with the badge', async () => {
    const mock = new MockChatBackend();
    const c = await mock.createConversation();
    expect(c.settings.permission_mode).toBe('auto');
    const { box, sub } = follow(mock, c.id, c);
    await flush(200);
    await mock.send(c.id, { text: 'Fix the flaky test', client_msg_id: 'cm_a' });
    await flush(70_000);
    expect(calls(box.conv).some((p) => p.status === 'asking' && p.name !== 'web_fetch')).toBe(false);
    const edit = calls(box.conv).find((p) => p.name === 'edit_file');
    if (edit) expect(edit.approved).toBe('auto');
    // the script's first bash call fails on purpose (the flaky test); it ran, and nobody was asked
    const bash = calls(box.conv).find((p) => p.name === 'bash');
    expect(['done', 'failed']).toContain(bash?.status);
    expect(bash?.approved).toBe('auto');
    sub.close();
  });

  it('yolo: nothing asks and the run finishes, every call marked yolo', async () => {
    const mock = new MockChatBackend();
    const c = await mock.createConversation({ permission_mode: 'yolo' });
    const { box, sub } = follow(mock, c.id, c);
    await flush(200);
    await mock.send(c.id, { text: 'Fix the flaky test', client_msg_id: 'cm_y' });
    await flush(200_000);
    expect(box.conv.status).toBe('idle');
    const cs = calls(box.conv);
    expect(cs.length).toBeGreaterThan(1);
    expect(cs.some((p) => p.status === 'asking' || p.status === 'denied')).toBe(false);
    expect(cs.every((p) => p.approved === 'yolo' || p.approved === 'rule')).toBe(true);
    expect(cs.some((p) => p.approved === 'yolo')).toBe(true);
    sub.close();
  });

  it('ask: bash asks as before', async () => {
    const mock = new MockChatBackend();
    const c = await mock.createConversation({ permission_mode: 'ask' });
    const { box, sub } = follow(mock, c.id, c);
    await flush(200);
    await mock.send(c.id, { text: 'Fix the flaky test', client_msg_id: 'cm_k' });
    await flush(30_000);
    expect(calls(box.conv).find((p) => p.status === 'asking')?.name).toBe('bash');
    sub.close();
  });

  it('the settings page changes the default for new chats', async () => {
    const mock = new MockChatBackend();
    expect((await mock.capabilities()).defaults.permission_mode).toBe('auto');
    await mock.updateDefaults({ permission_mode: 'yolo' });
    expect((await mock.capabilities()).defaults.permission_mode).toBe('yolo');
    expect((await mock.createConversation()).settings.permission_mode).toBe('yolo');
  });
});
