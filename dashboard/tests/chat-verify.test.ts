// The verify phase in the thread (SRV-532, QA SRV-533): the harness's check card, the Now line's
// phase, and the answer's "Checked / Not checked" block from chatd's `verify` part.
import { describe, expect, it } from 'vitest';
import { applyEvent } from '../src/chat/store';
import type { Conversation, Run, ToolCallPart, VerifyPart } from '../src/chat/types';
import { actingPhase, isHarnessCall, verifyBlock, verifyTone } from '../src/chat/verify';

const run = { id: 'r1', conversation_id: 'c1', message_id: 'a1', phase: 'writing', now: 'Writing', step: 2, tool: null, prefill: null, queue_position: null } as unknown as Run;
function conv(): Conversation {
  return {
    id: 'c1', status: 'running', run,
    messages: [{ id: 'a1', role: 'assistant', created_at: '', parent_id: 'u1', run_id: 'r1', status: 'streaming', parts: [{ type: 'text', text: 'Done: index.html.' }] }],
  } as unknown as Conversation;
}

const PART: VerifyPart = {
  type: 'verify',
  state: 'passed',
  text: 'Verified by the harness.\nChecked: /workspace/index.html in headless Chromium at 1280 and 390 px (harness check 1): no console errors, the screenshot: brightness 62 %, 40 % of it one colour (#88ccff); the screenshot, by the model.\nNot checked: input and interactions; sound and motion over time.',
  checked: ['/workspace/index.html in headless Chromium at 1280 and 390 px (harness check 1): no console errors, the screenshot: brightness 62 %, 40 % of it one colour (#88ccff)', 'the screenshot, by the model'],
  not_checked: ['input and interactions', 'sound and motion over time'],
};

describe('the Checked / Not checked block', () => {
  it('takes its lists from the part and its lead from the first line', () => {
    const b = verifyBlock(PART);
    expect(b.lead).toBe('Verified by the harness.');
    expect(b.checked).toEqual(PART.checked);
    expect(b.unchecked).toEqual(['input and interactions', 'sound and motion over time']);
  });

  it('a part with the text alone reads its lines', () => {
    const b = verifyBlock({ text: 'Not verified.\nNot checked: /workspace/index.html in a browser: this box has no sandbox to run a browser in; input and interactions; sound and motion over time.' });
    expect(b.lead).toBe('Not verified.');
    expect(b.checked).toEqual([]);
    expect(b.unchecked).toEqual(['/workspace/index.html in a browser: this box has no sandbox to run a browser in', 'input and interactions', 'sound and motion over time']);
  });

  it('its tone follows the state', () => {
    expect(verifyTone('passed')).toBe('ok');
    expect(verifyTone('problems')).toBe('warn');
    expect(verifyTone('stale')).toBe('warn');
    expect(verifyTone('not_run')).toBe('err');
  });
});

describe('the check in the run', () => {
  it('the harness call is told apart from the model\'s own render_page', () => {
    expect(isHarnessCall({ harness: 'verify' })).toBe(true);
    expect(isHarnessCall({} as Pick<ToolCallPart, 'harness'>)).toBe(false);
  });

  it('the verifying phase keeps its tool and acts on the Now line', () => {
    const c = applyEvent(conv(), { type: 'run.phase', run_id: 'r1', phase: 'verifying', now: 'Verifying the app: loading /workspace/index.html in a browser (check 1 of 2)', tool: { call_id: 'verify_1', name: 'render_page' } });
    expect(c.run?.phase).toBe('verifying');
    expect(c.run?.tool?.call_id).toBe('verify_1');
    expect(c.status).toBe('running');
    expect(actingPhase('verifying')).toBe(true);
    expect(actingPhase('writing')).toBe(false);
  });

  it('the block arrives as a part of the answer', () => {
    const c = applyEvent(conv(), { type: 'part.added', message_id: 'a1', part: PART });
    expect(c.messages[0].parts.map((p) => p.type)).toEqual(['text', 'verify']);
  });
});
