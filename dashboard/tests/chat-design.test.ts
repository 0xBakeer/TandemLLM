// VIS-185 (QA VIS-186): the Chat tab's design pass. The figures line as readings, and the CSS
// rules that keep the thread from scrolling sideways (the e2e in chat-overflow.spec.ts measures
// the same on real layouts).
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { figsParts } from '../src/chat/turn-figures';
import type { TurnStats } from '../src/chat/types';

const stats = (s: Partial<TurnStats> = {}): TurnStats => ({
  finish: 'stop', ttft_ms: 49_910, decode_tps: 30.31, prompt_tokens: 40_000, completion_tokens: 305, reasoning_tokens: 144,
  tokens_per_round: 3.04, acceptance: 0.52, ...s,
}) as TurnStats;
const text = (s: TurnStats) => figsParts(s).map((f) => (f.pre ? f.w + f.n : f.n + f.w) + (f.sub ? ' ' + f.sub : '')).join(' | ');

describe('the figures line', () => {
  it('reads speed, tokens, first token, rounds and acceptance, in that order', () => {
    expect(figsParts(stats()).map((f) => f.k)).toEqual(['tps', 'tok', 'ttft', 'round', 'acc']);
    expect(text(stats())).toBe('30.3 tok/s | 305 tok (144 thinking) | first token 49.91 s | 3.0 per round | 52 % accepted');
  });

  it('leaves out a plain finish and names any other first', () => {
    expect(figsParts(stats()).some((f) => f.k === 'finish')).toBe(false);
    expect(figsParts(stats({ finish: 'length' }))[0]).toEqual({ k: 'finish', n: '', w: 'length' });
  });

  it('drops the readings a backend does not report', () => {
    const s = stats({ decode_tps: null, ttft_ms: null, tokens_per_round: null, acceptance: null, reasoning_tokens: 0 });
    expect(text(s)).toBe('305 tok');
  });

  it('keeps the number apart from its words (two inks)', () => {
    const f = figsParts(stats()).find((x) => x.k === 'ttft')!;
    expect(f).toMatchObject({ n: '49.91 s', w: 'first token ', pre: true });
  });
});

describe('nothing in the thread is wider than its column', () => {
  const css = readFileSync(new URL('../src/styles/chat.css', import.meta.url), 'utf8');
  const rule = (sel: string) => {
    const i = css.indexOf(`\n${sel} {`);
    expect(i, sel).toBeGreaterThan(-1);
    return css.slice(i, css.indexOf('}', i));
  };

  it('the thread scrolls up and down only', () => {
    expect(rule('.ch-scroll')).toMatch(/overflow-x: hidden/);
    expect(rule('.ch-col')).toMatch(/overflow-wrap: break-word/);
  });

  it('inline code, links, user text and tool output wrap anywhere', () => {
    expect(rule('.ch-md :not(pre) > code')).toMatch(/overflow-wrap: anywhere/);
    expect(rule('.ch-md a')).toMatch(/overflow-wrap: anywhere/);
    expect(rule('.ch-text')).toMatch(/overflow-wrap: anywhere/);
    expect(rule('.ch-call-pre')).toMatch(/overflow-wrap: anywhere/);
  });

  it('code blocks and tables scroll in their own box; a tool row cuts its argument inside the card', () => {
    expect(rule('.ch-md .md-code pre')).toMatch(/overflow: auto/);
    expect(rule('.ch-md .md-table')).toMatch(/overflow-x: auto/);
    expect(rule('.ch-call-args')).toMatch(/text-overflow: ellipsis/);
    expect(rule('.ch-call-name')).toMatch(/text-overflow: ellipsis/);
  });

  it('the figures line is one row that ends in an ellipsis', () => {
    const r = rule('.ch-figs');
    expect(r).toMatch(/white-space: nowrap/);
    expect(r).toMatch(/text-overflow: ellipsis/);
  });

  it('the composer pills never scroll sideways on a phone: the model pill gives way, the narrowest phones wrap', () => {
    const phone = css.slice(css.indexOf('@media (max-width: 768px)'));
    const pills = phone.slice(phone.indexOf('  .ch-pills {'), phone.indexOf('}', phone.indexOf('  .ch-pills {')));
    expect(pills).not.toMatch(/overflow-x/);
    expect(phone).toMatch(/\.ch-pillwrap:first-child \{\s*flex: 0 1 auto;/);
    expect(css).toMatch(/@media \(max-width: 360px\) \{\s*\.ch-pills \{\s*flex-wrap: wrap;/);
  });

  it('the panes tabs scroll in their own strip, apart from plus, dock and close', () => {
    expect(rule('.ch-pane-tabs')).toMatch(/overflow-x: auto/);
    expect(rule('.ch-pane-strip')).not.toMatch(/overflow-x/);
  });
});
