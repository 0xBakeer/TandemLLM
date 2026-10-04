import { describe, expect, it } from 'vitest';
import type { LiveRequest } from '../src/api/types';
import { voiceLine } from '../src/lib/live';

const row = (id: string, phase: LiveRequest['phase'], line: LiveRequest['line'], ended: number | null = null) =>
  ({ request_id: id, phase, line, ended_ms_ago: ended }) as unknown as LiveRequest;

describe('voiceLine (narrator v3)', () => {
  it('is undefined for a server that sends no lines, so no slot is reserved', () => {
    expect(voiceLine([{ request_id: 'a', phase: 'decode' } as unknown as LiveRequest])).toBeUndefined();
  });
  it('takes the line of the request the Now line is about', () => {
    const rows = [row('a', 'decode', { kind: 'thinking', text: 'thinking 25 s: the loop', at_ms: 1 }), row('b', 'decode', { kind: 'tool', text: 'writing a.js', at_ms: 2 })];
    expect(voiceLine(rows, 'b')?.text).toBe('writing a.js');
  });
  it('else the first request in flight that has one, before a finished one', () => {
    const rows = [row('d', 'done', { kind: 'thought', text: 'thought 1 min: x', at_ms: 1 }, 100), row('a', 'decode', null), row('p', 'prefill', { kind: 'prefill', text: 'reading 9 tokens', at_ms: 1 })];
    expect(voiceLine(rows, 'a')?.text).toBe('reading 9 tokens');
  });
  it('is null (a reserved blank line) when no request has said anything yet', () => {
    expect(voiceLine([row('a', 'decode', null)], 'a')).toBeNull();
  });
});
