// The Live panel's words for contract 1.1 (VIS-24, tests of VIS-25): the Now line, the activity
// cell, the timeline, the stop tone and the stream's health, over the messages recorded on the box
// (tests/fixtures/live-1.1-box.json) and over one hand-made message per state and stop reason.
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import type { ActivityState, Live, LiveActivity, LiveEngine, LiveRequest, LiveStop, StopReason } from '../src/api/types';
import { activityCell, clientFlag, continuesText, etaWord, isMoving, nowLine, ordinal, pathSegments, pickRunning, roundWords, secs, stateWord, stopTone, streamHealth, timelineList, timelineSegments, tokensSplit } from '../src/lib/activity';

const recentRows = (r: Live['recent'], n = 20) => (r ?? []).slice(0, n);
import { emptyRing, push } from '../src/lib/live';

const FIX = JSON.parse(readFileSync(join(__dirname, '..', '..', 'tests', 'fixtures', 'live-1.1-box.json'), 'utf8')) as { messages: Record<string, Live> };
const msg = (k: string): Live => {
  const m = FIX.messages[k];
  if (!m) throw new Error(`no fixture message ${k}`);
  return m;
};

/** A minimal 1.1 request with an activity in `state`. */
function req(state: ActivityState, over: Partial<LiveActivity> = {}, row: Partial<LiveRequest> = {}): LiveRequest {
  const activity: LiveActivity = {
    state,
    label: 'x',
    since_ms: 1200,
    constrained: null,
    queue: null,
    prefill: null,
    decode: null,
    tool: null,
    reasoning: null,
    client: { connected: true, silent_ms: 10, gone_ms: null },
    continues: null,
    step: null,
    stop: null,
    ...over,
  };
  return {
    request_id: 'chatcmpl-000000000000000000000000',
    phase: state === 'done' ? 'done' : state === 'queued' ? 'queued' : state === 'prefilling' ? 'prefill' : 'decode',
    finish_reason: null,
    status: 200,
    model: 'm',
    client: { id: 'k:1', kind: 'opencode' },
    endpoint: 'chat',
    stream: true,
    thinking: true,
    temperature: 0,
    prompt_tokens: 100,
    cached_tokens: 0,
    forwarded_tokens: 100,
    tokens: 12,
    blocks: 3,
    tokens_per_block: 4,
    elapsed_ms: 5000,
    queue_ms: 1,
    prompt_ms: 100,
    ttft_ms: 101,
    decode_ms: 4899,
    prefill_tps: 1000,
    decode_tps: 40,
    decode_tps_now: 44.1,
    cache_source: 'none',
    max_tokens: 1000,
    ended_ms_ago: null,
    activity,
    timeline: [{ t_ms: 0, state: 'queued' }],
    ...row,
  };
}
const engine = (state: LiveEngine['state'], over: Partial<LiveEngine> = {}): LiveEngine => ({ state, label: state === 'busy' ? 'Busy' : 'Idle', since_ms: 4210, model: 'm', version: 'v', draining: false, kv: null, memory: null, store: null, waiting_for_client: null, ...over });
const stop = (reason: StopReason, over: Partial<LiveStop> = {}): LiveStop => ({ reason, detail: null, state: 'writing', tokens_sent: 1, silent_ms: 0, client_gone_ms: null, sentence: `${reason} sentence`, ...over });

describe('the Now line over the recorded messages', () => {
  it('prefilling with progress: the label as sent, the rate, the bar', () => {
    const m = msg('opencode_prefilling');
    const n = nowLine(m.engine, m.requests)!;
    expect(n.headline).toBe('Prefilling 12,288 of 29,200 (42 %)');
    expect(n.glyph).toBe('◐');
    expect(n.tone).toBe('prefill');
    expect(n.progress).toBeCloseTo(0.42, 1);
    expect(n.client).toBe('opencode');
    expect(n.moving).toBe(true);
    expect(n.numbers.some((x) => /tok\/s$/.test(x))).toBe(true);
    expect(n.warning).toBeNull();
  });
  it('thinking: the rate and the thinking tokens', () => {
    const m = msg('opencode_thinking');
    const n = nowLine(m.engine, m.requests)!;
    expect(n.headline).toBe('Thinking');
    expect(n.glyph).toBe('◍');
    expect(n.tone).toBe('think');
    expect(n.numbers.join(' ')).toMatch(/tok\/s .* thinking tokens/);
  });
  it('writing', () => {
    const n = nowLine(msg('writing').engine, msg('writing').requests)!;
    expect(n.headline).toBe('Writing');
    expect(n.glyph).toBe('●');
    expect(n.tone).toBe('write');
  });
  it('calling a tool: the name in the label, the argument bytes beside it', () => {
    const m = msg('opencode_tool_call');
    const n = nowLine(m.engine, m.requests)!;
    expect(n.headline).toBe('Calling tool write');
    expect(n.glyph).toBe('⚒');
    expect(n.tone).toBe('tool');
    expect(n.numbers[0]).toMatch(/of arguments$/);
  });
  it('waiting for the client: its own line even with only finished rows', () => {
    const m = msg('waiting_for_client');
    const n = nowLine(m.engine, m.requests)!;
    expect(n.headline).toBe('Waiting for client: running tool write');
    expect(n.glyph).toBe('◌');
    expect(n.tone).toBe('wait');
    expect(n.sinceMs).toBe(95.4);
    expect(n.client).toBe('opencode');
    expect(n.requestId).toBe('chatcmpl-96d6fc52dee74c52abab732a');
  });
  it('the next turn continues the previous request', () => {
    const r = msg('continues').requests[0];
    expect(continuesText(r.activity!.continues)).toBe('continues chatcmpl-96d6fc52d after 2.0 s (client ran write)');
  });
  it('a client that left mid-prefill: the warning on the line and the flag on the row', () => {
    const m = msg('client_gone');
    const n = nowLine(m.engine, m.requests)!;
    expect(n.headline).toBe('Prefilling 2,048 of 60,014 (3 %)');
    expect(n.warning).toBe('client disconnected');
    expect(clientFlag(m.requests[0].activity)).toBe('client disconnected');
    expect(clientFlag({ ...m.requests[0].activity!, client: { connected: false, silent_ms: 12_000, gone_ms: 12_000 } })).toBe('client disconnected 12 s ago');
  });
  it('idle with recent stops', () => {
    const m = msg('idle_with_recent');
    const n = nowLine(m.engine, m.requests)!;
    expect(n.headline).toBe('Idle');
    expect(n.tone).toBe('idle');
    expect(n.moving).toBe(false);
    expect(recentRows(m.recent).map((x) => x.stop.reason)).toContain('abandoned');
    expect(recentRows(m.recent).length).toBeLessThanOrEqual(20);
  });
  it('every recorded message has a Now line, and the stop sentences read as sent', () => {
    for (const [k, m] of Object.entries(FIX.messages)) {
      const n = nowLine(m.engine, m.requests);
      expect(n, k).not.toBeNull();
      expect(n!.headline.length, k).toBeGreaterThan(0);
      for (const x of m.recent ?? []) expect(x.stop.sentence.length, `${k} ${x.request_id}`).toBeGreaterThan(0);
    }
  });
});

describe('the Now line, one per engine state', () => {
  it('a 1.0 server has no Now line', () => {
    expect(nowLine(undefined, [])).toBeNull();
    expect(nowLine(null, [req('writing')])).toBeNull();
  });
  it('starting and draining', () => {
    expect(nowLine(engine('starting'), [])).toMatchObject({ headline: 'Starting', tone: 'warn', glyph: '◌' });
    expect(nowLine(engine('draining', { draining: true }), [])).toMatchObject({ headline: 'Draining', tone: 'warn', glyph: '◑' });
  });
  it('busy with the activity switched off: the engine label alone, the phase glyph', () => {
    const r = { ...req('writing'), activity: null, timeline: null };
    const n = nowLine(engine('busy', { label: 'Busy' }), [r])!;
    expect(n.headline).toBe('Busy');
    expect(n.glyph).toBe('●');
    expect(n.numbers).toEqual([]);
    expect(n.moving).toBe(true);
  });
  it('queued: the wait and the timeout', () => {
    const n = nowLine(engine('busy'), [req('queued', { label: 'Queued, about 2nd in line', queue: { place: 2, wait_ms: 5300, timeout_s: 120, place_is_estimate: true } })])!;
    expect(n.headline).toBe('Queued, about 2nd in line');
    expect(n.numbers).toEqual(['waiting 5.3 s', 'gives up after 120 s']);
    expect(n.tone).toBe('queued');
  });
  it('prefilling: cached tokens and the ETA; a single-call prefill gets no bar', () => {
    const p = { done: 36864, total: 48210, cached: 12288, pct: 76.5, tps_now: 2310, tps_avg: 1822.6, eta_ms: 4900, progress: 'chunked' as const, at_ms: 100 };
    const n = nowLine(engine('busy'), [req('prefilling', { label: 'Prefilling 36,864 of 48,210 (76 %)', prefill: p })])!;
    expect(n.numbers).toEqual(['2310 tok/s', '12,288 cached', 'about 5 s left']);
    expect(n.progress).toBeCloseTo(0.765, 3);
    const s = nowLine(engine('busy'), [req('prefilling', { label: 'Prefilling 48,210 tokens', prefill: { ...p, done: null, pct: null, progress: 'single_call' } })])!;
    expect(s.progress).toBeNull();
  });
  it('replaying, closing the reasoning, finishing', () => {
    expect(nowLine(engine('busy'), [req('replaying', { label: 'Replaying a cached answer' }, { tokens: 812 })])!.numbers).toEqual(['812 tokens replayed, no model run']);
    const c = nowLine(engine('busy'), [req('closing_reasoning', { label: 'Closing the reasoning (budget)', reasoning: { closed_by: 'budget', tokens: 4000 }, decode: dec({ thinking_tokens: 4000 }) })])!;
    expect(c.numbers).toEqual(['closed by the budget', '4,000 thinking tokens']);
    expect(c.tone).toBe('think');
    const f = nowLine(engine('busy'), [req('finishing', { label: 'Finishing: saving the session state', step: 'saving_state', decode: dec({ tokens: 1422 }) })])!;
    expect(f.headline).toBe('Finishing: saving the session state');
    expect(f.glyph).toBe('…');
    expect(f.numbers).toEqual(['1,422 tokens']);
  });
  it('writing with a JSON schema on; a tool call with calls done', () => {
    const w = nowLine(engine('busy'), [req('writing', { label: 'Writing', constrained: 'response_format', decode: dec({ content_tokens: 380, tps_now: 44.1 }) })])!;
    expect(w.numbers).toEqual(['44.1 tok/s', '380 content tokens', 'JSON schema']);
    const t = nowLine(engine('busy'), [req('tool_call', { label: 'Calling tool write_file', tool: { index: 1, name: 'write_file', arg_bytes: 18234, calls_done: 1 }, decode: dec({ tps_now: 41 }) })])!;
    expect(t.numbers).toEqual(['18.2 KB of arguments', '1 call done', '41.0 tok/s']);
  });
  it('picks the request furthest along, never a finished one', () => {
    const rows = [req('done', { stop: stop('stop') }), req('prefilling'), req('writing', {}, { request_id: 'chatcmpl-w' }), req('queued')];
    expect(pickRunning(rows)!.request_id).toBe('chatcmpl-w');
    expect(pickRunning([req('done', { stop: stop('stop') })])).toBeNull();
  });
});

function dec(over: Partial<NonNullable<LiveActivity['decode']>> = {}): NonNullable<LiveActivity['decode']> {
  return { tokens: 100, thinking_tokens: 0, content_tokens: 0, tool_tokens: 0, tps_now: null, tps_avg: null, rounds: 20, tokens_per_round: 4.1, tokens_per_round_now: 4.4, ms_per_round: 91.8, ms_per_round_now: 90.9, accept_mean: 3, ...over };
}

describe('the activity cell, one per state', () => {
  it('has a glyph and a word for every state', () => {
    const states: ActivityState[] = ['queued', 'prefilling', 'replaying', 'thinking', 'closing_reasoning', 'writing', 'tool_call', 'finishing', 'done'];
    const seen = new Set<string>();
    for (const s of states) {
      const c = activityCell(req(s, s === 'done' ? { stop: stop('stop') } : {}).activity);
      expect(c.glyph.length, s).toBeGreaterThan(0);
      expect(c.word.length, s).toBeGreaterThan(0);
      seen.add(c.glyph + c.word);
    }
    expect(seen.size).toBe(states.length);
  });
  it('the details: place and wait, per cent, tokens, the tool name and bytes, the finishing step', () => {
    expect(activityCell(req('queued', { queue: { place: 2, wait_ms: 5300, timeout_s: 120, place_is_estimate: true } }).activity)).toMatchObject({ glyph: '○', word: 'queued', detail: 'about 2nd · 5.3 s' });
    expect(activityCell(req('prefilling', { prefill: { done: 36864, total: 48210, cached: 0, pct: 76.4, tps_now: null, tps_avg: null, eta_ms: null, progress: 'chunked', at_ms: null } }).activity)).toMatchObject({ glyph: '◐', word: 'prefilling', detail: '76 %', moving: true });
    expect(activityCell(req('prefilling', { prefill: { done: null, total: 48210, cached: 0, pct: null, tps_now: null, tps_avg: null, eta_ms: null, progress: 'single_call', at_ms: null } }).activity).detail).toBe('48,210 tokens');
    expect(activityCell(req('thinking', { decode: dec({ thinking_tokens: 611 }) }).activity)).toMatchObject({ glyph: '◍', word: 'thinking', detail: '611 tok' });
    expect(activityCell(req('writing', { decode: dec({ content_tokens: 380 }) }).activity)).toMatchObject({ glyph: '●', word: 'writing', detail: '380 tok', tone: 'write' });
    expect(activityCell(req('tool_call', { tool: { index: 0, name: 'write_file', arg_bytes: 18234, calls_done: 0 } }).activity)).toMatchObject({ glyph: '⚒', word: 'write_file', detail: '18.2 KB', tone: 'tool' });
    expect(activityCell(req('finishing', { step: 'saving_state' }).activity)).toMatchObject({ glyph: '…', word: 'finishing', detail: 'saving state', moving: false });
    expect(activityCell(req('replaying').activity, { phase: 'decode', finish_reason: null, status: 200, tokens: 812 })).toMatchObject({ glyph: '↻', word: 'replaying', detail: '812 tok' });
  });
  it('done: the stop tone decides the glyph', () => {
    expect(activityCell(req('done', { stop: stop('stop') }).activity)).toMatchObject({ glyph: '✓', word: 'stop', tone: 'ok' });
    expect(activityCell(req('done', { stop: stop('abandoned') }).activity)).toMatchObject({ glyph: '✗', word: 'abandoned', tone: 'bad' });
    expect(activityCell(req('done', { stop: stop('tool_calls') }).activity)).toMatchObject({ glyph: '✓', word: 'tool call' });
  });
  it('the 1.0 fallback: VIS-23 phase words when the activity is missing', () => {
    expect(activityCell(null, { phase: 'decode', finish_reason: null, status: 200, tokens: 1 })).toMatchObject({ glyph: '●', word: 'decoding', tone: 'write', moving: true });
    expect(activityCell(undefined, { phase: 'prefill', finish_reason: null, status: 200, tokens: 0 })).toMatchObject({ glyph: '◐', word: 'prefilling', tone: 'prefill' });
    expect(activityCell(null, { phase: 'done', finish_reason: 'error', status: 200, tokens: 5 })).toMatchObject({ glyph: '✕', word: 'error', tone: 'bad' });
    expect(activityCell(null, { phase: 'done', finish_reason: 'stop', status: 200, tokens: 5 })).toMatchObject({ glyph: '✓', word: 'stop', tone: 'ok' });
  });
});

describe('stops', () => {
  it('every reason has a glyph and a word; abandoned, error, timeout, cancelled are red, refused and rejected amber', () => {
    const reasons: StopReason[] = ['stop', 'length', 'tool_calls', 'timeout', 'abandoned', 'error', 'refused', 'rejected', 'cancelled'];
    for (const r of reasons) {
      const t = stopTone(stop(r));
      expect(t.word.length, r).toBeGreaterThan(0);
      if (['abandoned', 'error', 'timeout', 'cancelled'].includes(r)) expect(t, r).toMatchObject({ glyph: '✗', tone: 'bad' });
      else if (['refused', 'rejected'].includes(r)) expect(t, r).toMatchObject({ glyph: '!', tone: 'warn' });
      else expect(t, r).toMatchObject({ glyph: '✓', tone: 'ok' });
    }
    expect(stopTone(null)).toMatchObject({ glyph: '✓', word: 'done', tone: 'ok' });
  });
  it('the recorded stops: the sentence says why', () => {
    const rec = recentRows(msg('idle_with_recent').recent);
    const ab = rec.find((x) => x.stop.reason === 'abandoned')!;
    expect(ab.stop.sentence).toBe('abandoned by the client after 3.9 s of silent prefill, 0 tokens sent');
    expect(stopTone(ab.stop).tone).toBe('bad');
    const rj = rec.find((x) => x.stop.reason === 'rejected')!;
    expect(rj.stop.sentence).toBe('rejected: the prompt is longer than the context window');
    expect(stopTone(rj.stop).tone).toBe('warn');
  });
});

describe('the timeline', () => {
  it('one segment per state, width by time, the last one open until now; done has no width', () => {
    const tl = [
      { t_ms: 0, state: 'queued' as const },
      { t_ms: 2, state: 'prefilling' as const, detail: '12288 cached' },
      { t_ms: 20_000, state: 'thinking' as const },
      { t_ms: 34_000, state: 'writing' as const },
      { t_ms: 48_000, state: 'tool_call' as const, detail: 'write_file' },
    ];
    const segs = timelineSegments(tl, 52_000);
    expect(segs.map((s) => s.state)).toEqual(['queued', 'prefilling', 'thinking', 'writing', 'tool_call']);
    expect(segs.reduce((a, s) => a + s.share, 0)).toBeCloseTo(1, 6);
    expect(segs[1].share).toBeCloseTo((20_000 - 2) / 52_000, 6);
    expect(segs[4].endMs).toBe(52_000);
    expect(segs[0].showLabel).toBe(false); // 2 ms of 52 s
    expect(segs[1].showLabel).toBe(true); // 61 px of 160 fits "prefilling"
    expect(segs[3].showLabel).toBe(false); // 43 px does not fit "writing"
    expect(timelineSegments(tl, 52_000, 400)[3].showLabel).toBe(true);
    expect(segs[1].detail).toBe('12288 cached');
    expect(segs[4].word).toBe('tool call');
    const done = timelineSegments([...tl, { t_ms: 50_000, state: 'done' }], null);
    expect(done.map((s) => s.state)).not.toContain('done');
    expect(done[done.length - 1].endMs).toBe(50_000);
  });
  it('empty or a single instant: no crash, equal shares', () => {
    expect(timelineSegments(null, 10)).toEqual([]);
    expect(timelineSegments([], 10)).toEqual([]);
    const one = timelineSegments([{ t_ms: 0, state: 'queued' }], 0);
    expect(one).toHaveLength(1);
    expect(one[0].share).toBe(1);
  });
  it('the recorded opencode turn', () => {
    const r = msg('opencode_tool_call').requests[0];
    const segs = timelineSegments(r.timeline, r.elapsed_ms);
    expect(segs.map((s) => s.state)).toEqual(['queued', 'prefilling', 'thinking', 'tool_call', 'writing', 'tool_call']);
    expect(segs[1].share).toBeGreaterThan(0.9); // 39 s of prefill in a 41.5 s request
    expect(segs[1].showLabel).toBe(true);
  });
  it('the list: the last six transitions as words', () => {
    const r = msg('opencode_tool_call').requests[0];
    const list = timelineList(r.timeline, 6);
    expect(list).toHaveLength(6);
    expect(list[0]).toEqual({ at: '+0.0 s', word: 'queued', detail: undefined });
    expect(list[2]).toEqual({ at: '+40 s', word: 'thinking', detail: undefined });
    expect(list[5]).toMatchObject({ word: 'tool call', detail: 'write' });
  });
  it('a recent row\'s path: equal segments, done dropped', () => {
    const p = pathSegments(['queued', 'prefilling', 'thinking', 'tool_call', 'done']);
    expect(p).toHaveLength(4);
    expect(p[0].share).toBe(0.25);
    expect(p.every((s) => !s.showLabel)).toBe(true);
  });
});

describe('the words', () => {
  it('state words, ordinals, seconds, ETA, token split', () => {
    expect(stateWord('closing_reasoning')).toBe('closing reasoning');
    expect(stateWord('tool_call')).toBe('tool call');
    expect([1, 2, 3, 4, 11, 12, 13, 21, 22, 101, 111].map(ordinal)).toEqual(['1st', '2nd', '3rd', '4th', '11th', '12th', '13th', '21st', '22nd', '101st', '111th']);
    expect(secs(800)).toBe('0.8 s');
    expect(secs(34_200)).toBe('34 s');
    expect(secs(125_000)).toBe('2 m 05 s');
    expect(secs(null)).toBe('');
    expect(etaWord(4900)).toBe('about 5 s left');
    expect(etaWord(70_000)).toBe('about 1 m 10 s left');
    expect(etaWord(200)).toBeNull();
    expect(etaWord(null)).toBeNull();
    expect(tokensSplit(dec({ thinking_tokens: 611, content_tokens: 380 }))).toBe('611 thinking · 380 content');
    expect(tokensSplit(dec({ thinking_tokens: 611, content_tokens: 380, tool_tokens: 40 }))).toBe('611 thinking · 380 content · 40 tool');
    expect(tokensSplit(dec())).toBeNull();
    expect(tokensSplit(null)).toBeNull();
    expect(isMoving('thinking')).toBe(true);
    expect(isMoving('finishing')).toBe(false);
    expect(isMoving(null)).toBe(false);
  });
  it('continues: the gap, the tools, and (inferred)', () => {
    expect(continuesText({ request_id: 'chatcmpl-77aa000000000000000000', gap_ms: 8410, tool_names: ['bash'], inferred: true })).toBe('continues chatcmpl-77aa00000 after 8.4 s (client ran bash) (inferred)');
    expect(continuesText({ request_id: 'chatcmpl-77aa', gap_ms: null, tool_names: [], inferred: false })).toBe('continues chatcmpl-77aa');
    expect(continuesText(null)).toBeNull();
  });
  it('round words under the decode figure', () => {
    expect(roundWords([req('writing', { decode: dec() })])).toBe('4.4 tokens per round · 90.9 ms a round (last second)');
    expect(roundWords([req('writing', { decode: dec({ tokens_per_round_now: null, ms_per_round_now: null }) })])).toBe('4.1 tokens per round · 91.8 ms a round');
    expect(roundWords([req('prefilling')])).toBeNull();
    expect(roundWords([{ ...req('writing'), activity: null }])).toBeNull();
    expect(roundWords(msg('opencode_tool_call').requests)).toBe('6.4 tokens per round · 143.7 ms a round (last second)');
  });
});

describe('the health of the stream itself', () => {
  it('streaming with its cadence', () => {
    expect(streamHealth('open', 1000, 900, 1100, 0.25)).toMatchObject({ word: 'streaming', cadence: '4/s', tone: 'ok', stale: false });
    expect(streamHealth('open', 1000, 900, 1100, 1)).toMatchObject({ word: 'streaming', cadence: '1 s', tone: 'ok' });
  });
  it('a silent open stream says so and dims the page: warn after 2 × interval + 1.5 s, red past 6 s', () => {
    expect(streamHealth('open', 0, 0, 1900, 0.25)).toMatchObject({ word: 'streaming', stale: false });
    expect(streamHealth('open', 0, 0, 2500, 0.25)).toMatchObject({ word: 'no update for 2.5 s', tone: 'warn', stale: true });
    expect(streamHealth('open', 0, 0, 7000, 0.25)).toMatchObject({ word: 'no update for 7.0 s', tone: 'bad', stale: true });
    expect(streamHealth('open', 0, 0, 3000, 1)).toMatchObject({ word: 'streaming', stale: false }); // 1 Hz: 3.5 s allowed
  });
  it('reconnecting since N s, paused, signed out, connecting', () => {
    expect(streamHealth('reconnecting', 0, 1000, 5000, 0.25)).toMatchObject({ word: 'reconnecting since 4.0 s', tone: 'warn', stale: true });
    expect(streamHealth('error', 0, 1000, 9000, 0.25)).toMatchObject({ word: 'reconnecting since 8.0 s', tone: 'bad' });
    expect(streamHealth('reconnecting', 0, 4900, 5000, 0.25).word).toBe('reconnecting');
    expect(streamHealth('closed', null, null, 5000, 1)).toMatchObject({ word: 'paused', tone: 'muted', stale: true });
    expect(streamHealth('unauthorized', null, null, 5000, 1)).toMatchObject({ word: 'signed out', tone: 'bad' });
    expect(streamHealth('connecting', null, null, 5000, 1)).toMatchObject({ word: 'connecting', stale: false });
    expect(streamHealth('connecting', 100, null, 5000, 1).stale).toBe(true);
  });
});

describe('four events a second, one sparkline point a second', () => {
  it('push ignores the three events that carry the same sample.t', () => {
    const ring = emptyRing();
    const s = (t: number) => ({ t, decode_tps: 40, prefill_tps: null, running: 1, waiting: 0, tokens: 0 });
    for (let i = 0; i < 12; i++) push(ring, s(100 + Math.floor(i / 4)), 200);
    expect(ring.samples.map((x) => x.t)).toEqual([100, 101, 102]);
  });
  it('the recorded fixture: sample.t is the same across the 4 Hz events of one second', () => {
    // the sampler keeps one history sample a second (server/live.py): the ring must not grow per event
    const a = msg('opencode_tool_call').sample!;
    const ring = emptyRing();
    push(ring, a, a.t + 1);
    push(ring, { ...a }, a.t + 1.25);
    push(ring, { ...a }, a.t + 1.5);
    expect(ring.samples).toHaveLength(1);
  });
});
