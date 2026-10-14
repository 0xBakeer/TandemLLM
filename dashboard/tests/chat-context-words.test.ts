// The Context panel's words for a figure it has no value for (VIS-137): a value, or why there is
// none, never a bare dash.
import { describe, expect, it } from 'vitest';
import { AFTER_TURN, NEXT_TURN, NOT_TRACKED, kvWords, residentWords, sandboxWords, skillTokens, turnFigure } from '../src/chat/context-words';
import type { ContextReport, SandboxStatus } from '../src/chat/types';

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
    expect(residentWords(mem({}), 'cloud')).toBe(NOT_TRACKED);
  });
  it("gives the sandbox's memory, or says the runtime did not", () => {
    const sb = (s: Partial<SandboxStatus>) => ({ state: 'running', memory_bytes: null, ...s }) as SandboxStatus;
    expect(sandboxWords(sb({ memory_bytes: 31_449_088 }))).toBe('running, 31 MB');
    expect(sandboxWords(sb({}))).toBe('running, memory not reported by the runtime');
    expect(sandboxWords(sb({ state: 'waiting_memory' }))).toBe('waiting memory');
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
});
