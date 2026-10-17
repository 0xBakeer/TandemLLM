// The Context panel's words for a figure it has no value for (VIS-137): a value, or why there is
// none, never a bare dash.
import { describe, expect, it } from 'vitest';
import { AFTER_TURN, NEXT_TURN, NOT_TRACKED, attachmentWords, kvWords, partWords, residentWords, sandboxCallWords, sandboxWords, skillTokens, turnFigure } from '../src/chat/context-words';
import type { Attachment, ContextReport, SandboxStatus } from '../src/chat/types';

describe('the Context panel says why a figure is missing', () => {
  const mem = (m: Partial<ContextReport['memory']>): ContextReport['memory'] => ({ kv_bytes: null, resident_bytes: null, sandbox_disk_bytes: null, ...m });
  it('names where the chat sits in the engine', () => {
    expect(kvWords(mem({ kv_tokens: 37614, kv_where: 'slot' }), 'engine')).toBe("37,614 tokens, in the engine's slot");
    expect(kvWords(mem({ kv_tokens: 900, kv_where: 'park' }), 'engine')).toBe('900 tokens, parked in RAM');
    expect(kvWords(mem({ kv_where: 'nvme_spill' }), 'engine')).toMatch(/NVMe/);
    expect(kvWords(mem({ kv_where: 'other' }), 'engine')).toMatch(/another chat holds the slot/);
    expect(kvWords(mem({ kv_where: 'none' }), 'engine')).toMatch(/none held/);
    expect(kvWords(mem({}), 'engine')).toMatch(/does not report/);
    expect(kvWords(mem({ kv_tokens: 5, kv_where: 'slot' }), 'proxy')).toBe(NOT_TRACKED);
  });
  it("gives the resident prefix's size or whose it is", () => {
    expect(residentWords(mem({ resident_owner: 'this', resident_bytes: 1474560 }), 'engine')).toBe('1.5 MB');
    expect(residentWords(mem({ resident_owner: 'other' }), 'engine')).toBe('held by another chat');
    expect(residentWords(mem({ resident_owner: 'none' }), 'engine')).toBe('none held');
    expect(residentWords(mem({}), 'proxy')).toBe(NOT_TRACKED);
    expect(residentWords(mem({}), 'cloud')).toBe('none on this box');
  });
  it("gives the sandbox's memory, or says the runtime did not", () => {
    const sb = (s: Partial<SandboxStatus>) => ({ state: 'running', memory_bytes: null, ...s }) as SandboxStatus;
    expect(sandboxWords(sb({ memory_bytes: 31_449_088 }))).toBe('running, 31 MB');
    expect(sandboxWords(sb({}))).toBe('running, memory not reported by the runtime');
    expect(sandboxWords(sb({ state: 'waiting_memory' }))).toBe('waiting memory');
  });
  it('says on the call why the sandbox could not run it (SRV-481, SRV-485)', () => {
    const why = 'MemAvailable 3.2 GiB: a 1 GiB sandbox would leave the engine under its 4 GiB floor';
    expect(sandboxCallWords({ reason: 'no_memory', why, waited_s: 60, limit_gib: 1, floor_gib: 4, mem_available_gib: 3.2, engine_released_gib: 0.5 })).toBe(`The sandbox did not start: ${why} (waited 60 s, the engine gave back 0.5 GiB). The command did not run.`);
    expect(sandboxCallWords({ reason: 'busy', why: '4 sandbox(es) of other chats are running commands, the most at once', waited_s: 60, limit_gib: 1 })).toBe('The sandbox did not start: 4 sandbox(es) of other chats are running commands, the most at once (waited 60 s). The command did not run.');
    expect(sandboxCallWords({ reason: 'oom', limit_gib: 1 })).toBe('Out of memory (1 GiB sandbox limit): the kernel killed a process of this call.');
  });
  it('gives an attachment its size and tokens, and says when it goes as a file (SRV-487, VIS-169)', () => {
    const at = (x: Partial<Attachment>) => ({ id: 'a1', name: 'TandemLLM-and-StairCut.html', kind: 'text', bytes: 9_455_990, mime: 'text/html', ...x }) as Attachment;
    expect(attachmentWords(at({ tokens: 6123, inline: true }))).toEqual({ size: '9 MB · 6.1K tok', warn: null });
    expect(attachmentWords(at({ tokens: 2_363_997, inline: false, path: '/workspace/uploads/TandemLLM-and-StairCut.html' })).warn).toBe('TandemLLM-and-StairCut.html is too long to include (about 2,363,997 tokens): the model gets one line with its name, size and path (/workspace/uploads/TandemLLM-and-StairCut.html).');
    expect(attachmentWords(at({ tokens: 40_000, inline: false })).warn).toMatch(/size and path in the chat's workspace\.$/);
    expect(attachmentWords(at({ kind: 'image', mime: 'image/png', bytes: 2048, tokens: null })).size).toBe('2 KB');
  });
  it("counts a skill's tokens, or says when they are counted", () => {
    expect(skillTokens(1234)).toBe('1,234 tokens');
    expect(skillTokens(null)).toBe(NEXT_TURN);
  });
  it('says a figure comes after the first turn, or is not tracked', () => {
    expect(turnFigure(4.21, (v) => v.toFixed(2), true)).toBe('4.21');
    expect(turnFigure(null, String, false)).toBe(AFTER_TURN);
    expect(turnFigure(null, String, true)).toBe(NOT_TRACKED);
  });
  it('says when the system prompt and the tools are counted', () => {
    const split = { system: 115, tools: 1187, history: 30000, injected: 7698, images: 0 };
    expect(partWords('system', 115, split, 'engine')).toBe('115');
    expect(partWords('system', 0, null, 'engine')).toBe('next turn');
    expect(partWords('tools', 0, undefined, 'proxy')).toBe('not tracked');
    expect(partWords('files', 0, null, 'engine')).toBe('0');
  });
});
