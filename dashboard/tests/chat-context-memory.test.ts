// SRV-471 (QA SRV-472): the Context panel's memory part. A Chat tab turn says how many notes the
// index named and what they cost, the notes sent in full with the message, or why memory is off;
// it never shows a bare dash.
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { HARNESS_PENDING, NOT_TRACKED, memoryWords } from '../src/chat/context-words';
import type { ContextReport, HarnessMemory } from '../src/chat/types';

const h = (m: Partial<HarnessMemory> | null): ContextReport['harness'] => ({ class: 'chat', effort: 'medium', packages: [], memory: m === null ? null : { mode: 'on', why: null, index_notes: 0, index_tokens: 0, bodies: 0, body_tokens: 0, card_tokens: 190, ...m } });

describe("the Context panel's memory notes", () => {
  it('names the index and the notes sent with the message', () => {
    expect(memoryWords(h({ index_notes: 10, index_tokens: 407 }), 'engine')).toBe('10 notes in the index, 407 tokens');
    expect(memoryWords(h({ index_notes: 1, index_tokens: 40, bodies: 2, body_tokens: 180 }), 'engine')).toBe('1 note in the index, 40 tokens; 2 in full for this message, 180 tokens');
    expect(memoryWords(h({}), 'engine')).toBe('on, no notes yet');
  });
  it('says why it is off: a profile without the switch has it off', () => {
    expect(memoryWords(h({ mode: 'off', why: 'profile' }), 'engine')).toBe('off in this profile');
    expect(memoryWords(h({ mode: 'off', why: 'model_fit' }), 'engine')).toBe('off for this model');
    expect(memoryWords(h({ mode: 'off', why: 'header' }), 'engine')).toBe('off for this request');
    expect(memoryWords(h({ mode: 'off', why: 'unset' }), 'engine')).toBe('off in this profile');
    expect(memoryWords(h(null), 'engine')).toBe('not reported by this engine');
    expect(memoryWords(h({ mode: 'shadow' }), 'engine')).toMatch(/^shadow/);
  });
  it('waits for the first turn, and is not tracked on another backend', () => {
    expect(memoryWords(null, 'engine')).toBe(HARNESS_PENDING);
    expect(memoryWords(h({ index_notes: 3 }), 'proxy')).toBe(NOT_TRACKED);
  });
  it('is a row of the Skills section', () => {
    const src = readFileSync(new URL('../src/chat/context-panel.ts', import.meta.url), 'utf8');
    const skills = src.slice(src.indexOf('private renderSkills('), src.indexOf('private renderMemory('));
    expect(skills).toContain('<div data-harness="memory"><dt>memory notes</dt><dd>${memoryWords(c.context.harness, c.context.model.backend)}</dd></div>');
  });
});
