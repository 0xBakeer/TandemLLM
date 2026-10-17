// VIS-181 (QA VIS-182): the Context panel while an agent turn runs. Live (2026-10-16, Nemotron 3.5
// Super VL, a tool call a step) Now said "not tracked for this model" for the speed and the
// acceptance, the slot row said "none held" under a KV in the slot.
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { NOT_TRACKED, liveFigure, slotWords } from '../src/chat/context-words';
import { applyEvent } from '../src/chat/store';
import type { Conversation, Run } from '../src/chat/types';

const run = (phase: Run['phase']): Run => ({ id: 'r1', conversation_id: 'c1', message_id: 'a1', started_at: '2026-10-16T15:44:00Z', started_by: null, phase, now: '', step: 3, max_steps: 50, queue_position: null, tool: null, prefill: null, decode_tps: null, tokens_per_round: null, acceptance: null, thinking: { used: 0, budget: null }, tokens: 0, tps_history: [] });
const one = (v: number) => v.toFixed(1);
const mem = (kv_where: string | null) => ({ kv_bytes: null, resident_bytes: null, sandbox_disk_bytes: null, kv_where }) as Conversation['context']['memory'];

describe('the Now figures of a live run', () => {
  it('shows the value when there is one', () => {
    expect(liveFigure(36.4, one, run('thinking'), 'tps', 'engine', 'LiLiCorr')).toBe('36.4');
    expect(liveFigure(0.54, (v) => `${Math.round(v * 100)} %`, run('tool'), 'accepted', 'engine', 'LiLiCorr')).toBe('54 %');
  });
  it('says why a figure is missing at this moment, never "not tracked" on our engine', () => {
    expect(liveFigure(null, one, run('tool'), 'tps', 'engine', 'LiLiCorr')).toBe('between model calls');
    expect(liveFigure(null, one, run('thinking'), 'tps', 'engine', 'LiLiCorr')).toBe('warming up, the first 50 tokens');
    expect(liveFigure(null, one, run('prefill'), 'tps', 'engine', 'LiLiCorr')).toBe('prefilling');
    expect(liveFigure(null, one, run('queued'), 'round', 'engine', 'LiLiCorr')).toBe('queued');
    expect(liveFigure(null, one, run('thinking'), 'accepted', 'engine', 'LiLiCorr')).toBe('measured when a model call ends');
    expect(liveFigure(null, one, run('thinking'), 'accepted', 'engine', null)).toBe('no drafter');
    for (const ph of ['tool', 'thinking', 'writing'] as const) expect(liveFigure(null, one, run(ph), 'tps', 'engine', 'x')).not.toBe(NOT_TRACKED);
  });
  it("is not tracked on another backend, except chatd's own count of its stream (0.11.3)", () => {
    expect(liveFigure(null, one, run('thinking'), 'round', 'proxy', null)).toBe(NOT_TRACKED);
    expect(liveFigure(null, one, run('thinking'), 'tps', 'proxy', null)).toBe('warming up, the first 50 tokens');
  });
  it("keeps the call's thinking beside the run's from run.metrics", () => {
    const c = { run: run('thinking') } as Conversation;
    const next = applyEvent(c, { type: 'run.metrics', run_id: 'r1', decode_tps: 36, tokens_per_round: 2.8, acceptance: 0.55, tokens: 1043, thinking_used: 897, thinking_call: 100 });
    expect(next.run?.thinking).toEqual({ used: 897, call: 100, budget: null });
  });
  it('the panel reads them through liveFigure while a run is live, and caps the budget meter per call', () => {
    const src = readFileSync(new URL('../src/chat/context-panel.ts', import.meta.url), 'utf8');
    const now = src.slice(src.indexOf('private renderNow('), src.indexOf('private renderModel('));
    expect(now).toContain("r ? liveFigure(v, fmt, r, kind, be, drafter) : kind === 'tps' ? turnFigure(v, fmt, !!last) : draftFigure(v, fmt, !!last, be)");
    expect(now).toContain("fig(r ? r.acceptance : last?.acceptance");
    expect(now).toContain('(capped / budget)');
  });
});

describe('the slot row', () => {
  it("agrees with the KV row when no request runs", () => {
    expect(slotWords(null, mem('slot'))).toBe('held by this chat');
    expect(slotWords(null, mem('other'))).toBe('held by another chat');
    expect(slotWords(null, mem('park'))).toBe('held by another chat');
    expect(slotWords(null, mem('none'))).toBe('none held');
    expect(slotWords('side', mem('slot'))).toBe('side');
  });
});
