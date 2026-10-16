// VIS-199 (QA VIS-200): the two window steppers of the memory step. − and + walk the ladder from 8,192
// up to the model's own maximum, clipped by the profile and the first chat; any number can be typed
// (131,072, 128k, 1m); out of range or off the card's step stays as typed with the model's real
// limit, never clamped.
import { describe, expect, it } from 'vitest';
import bolt from '../mock/setup/nemotron-3.5-bolt-74b-a9b.json';
import superVl from '../mock/setup/nemotron-3.5-super-vl-120b-a12b.json';
import flashNext from '../mock/setup/qwen3.8-flash-next.json';
import qwen from '../mock/setup/qwen3.8-27b.json';
import type { SetupModel } from '../src/api/types';
import { estimate } from '../src/setup/estimate';
import { chatCost } from '../src/setup/slots';
import { defaults } from '../src/setup/state';
import { S } from '../src/setup/strings';
import { checkWindow, parseTokens, rungs, shortTokens, stepWindow, windowRange } from '../src/setup/window';

const VL = superVl as unknown as SetupModel;
const Q = qwen as unknown as SetupModel;
const FN = flashNext as unknown as SetupModel;
const BOLT = bolt as unknown as SetupModel;
// Super VL's card holds 1,048,576 tokens (its own maximum, as the super-vl-1m card reports it)
const VL1M = VL;

describe('parseTokens: what can be typed', () => {
  it('plain numbers, with separators, and the k and m forms', () => {
    expect(parseTokens('131072')).toBe(131_072);
    expect(parseTokens('131,072')).toBe(131_072);
    expect(parseTokens(' 1 048 576 ')).toBe(1_048_576);
    expect(parseTokens("262'144")).toBe(262_144);
    expect(parseTokens('128k')).toBe(131_072);
    expect(parseTokens('256K')).toBe(262_144);
    expect(parseTokens('1m')).toBe(1_048_576);
    expect(parseTokens('0.5M')).toBe(524_288);
  });
  it('anything else is not a number of tokens', () => {
    for (const bad of ['', 'abc', '12,34', '1e6', '-8192', '8k8', '1,0000']) expect(parseTokens(bad)).toBeNull();
  });
});

describe('the range and the ladder', () => {
  it('the main window: the card and the profile; the model\'s own words above it', () => {
    const r = windowRange(VL1M, defaults(VL1M), 'window');
    expect(r).toMatchObject({ min: 8192, max: 1_048_576, step: 4096 });
    expect(r.maxWhy).toBe('This model holds at most 1,048,576 tokens.');
    expect(rungs(r)).toEqual([8192, 16_384, 32_768, 65_536, 131_072, 262_144, 524_288, 1_048_576]);
    expect(windowRange(Q, defaults(Q), 'window').maxWhy).toBe(S.windowAtMost(262_144));
  });
  it('a profile with a shorter largest window caps it, and says so by its name (Flash-Next, SRV-337)', () => {
    const sel = { ...defaults(FN), profile: 'flashnext-4.05' };
    const r = windowRange(FN, sel, 'window');
    const prof = FN.profiles.find((p) => p.id === 'flashnext-4.05')!;
    expect(r.max).toBe(208_896);
    expect(r.maxWhy).toBe(S.windowAboveProfile(prof.label, 208_896));
    expect(r.maxWhy).toMatch(/holds at most 208,896 tokens on this machine\.$/);
    expect(stepWindow(131_072, 1, r)).toBe(208_896);
    expect(checkWindow('256k', r)).toEqual({ ok: false, why: r.maxWhy });
  });
  it('the extra chats never hold more than the first chat (the engine cuts a side slot to slot 0)', () => {
    const r = windowRange(VL1M, { ...defaults(VL1M), window: 131_072 }, 'side_window');
    expect(r.max).toBe(131_072);
    expect(r.maxWhy).toBe("The extra chats hold at most the first chat's 131,072 tokens.");
    expect(checkWindow('262144', r)).toEqual({ ok: false, why: r.maxWhy });
  });
  it('+ and − walk the rungs; from a typed value between two, the next rung that way', () => {
    const r = windowRange(VL1M, defaults(VL1M), 'window');
    expect(stepWindow(262_144, 1, r)).toBe(524_288);
    expect(stepWindow(524_288, 1, r)).toBe(1_048_576);
    expect(stepWindow(1_048_576, 1, r)).toBe(1_048_576);
    expect(stepWindow(8192, -1, r)).toBe(8192);
    expect(stepWindow(8192, 1, r)).toBe(16_384);
    expect(stepWindow(100_000 - (100_000 % 4096), 1, r)).toBe(131_072);
    expect(stepWindow(98_304, -1, r)).toBe(65_536);
  });
  it('nothing caps it at 262,144 but a card that says so: Bolt stops there, Super VL goes on to 1M', () => {
    expect(VL.parallelism.window.max).toBe(1_048_576);
    expect(VL.parallelism.side_window.max).toBe(1_048_576);
    const r = windowRange(BOLT, defaults(BOLT), 'window');
    expect(rungs(r).at(-1)).toBe(262_144);
    expect(stepWindow(262_144, 1, r)).toBe(262_144);
  });
  it('a top that is not a rung is a rung of its own', () => {
    const r = { min: 8192, max: 200_704, step: 4096, maxWhy: '' };
    expect(rungs(r)).toEqual([8192, 16_384, 32_768, 65_536, 131_072, 200_704]);
    expect(stepWindow(131_072, 1, r)).toBe(200_704);
  });
});

describe('checkWindow: never clamped, always a reason', () => {
  const r = windowRange(BOLT, defaults(BOLT), 'window');
  it('takes a value on the step inside the range', () => {
    expect(checkWindow('100k', r)).toEqual({ ok: true, n: 102_400 });
    expect(checkWindow('258,048', r)).toEqual({ ok: true, n: 258_048 });
  });
  it('above the model: its real limit', () => {
    expect(checkWindow('1m', r)).toEqual({ ok: false, why: 'This model holds at most 262,144 tokens.' });
    expect(checkWindow('1m', windowRange(VL, defaults(VL), 'window'))).toEqual({ ok: true, n: 1_048_576 });
    expect(checkWindow('1,048,577', windowRange(VL, defaults(VL), 'window'))).toEqual({ ok: false, why: 'This model holds at most 1,048,576 tokens.' });
  });
  it('below, off the step, not a number', () => {
    expect(checkWindow('4096', r)).toEqual({ ok: false, why: 'At least 8,192 tokens.' });
    expect(checkWindow('100000', r)).toEqual({ ok: false, why: 'Use a multiple of 4,096 tokens, like 98,304 or 102,400.' });
    expect(checkWindow('', r)).toEqual({ ok: false, why: S.windowNotANumber });
    expect(checkWindow('lots', r)).toEqual({ ok: false, why: 'Type a number of tokens, like 32k or 131,072.' });
  });
  it('the 27B\'s step is 8,192', () => {
    expect(checkWindow('100000', windowRange(Q, defaults(Q), 'window'))).toEqual({ ok: false, why: 'Use a multiple of 8,192 tokens, like 98,304 or 106,496.' });
  });
});

describe('the memory follows the window', () => {
  it('1M on Super VL fits the estimate; the extra chats\' cost line follows their window', () => {
    const sel = { ...defaults(VL1M), window: 1_048_576, side_window: 524_288 };
    const e = estimate(VL1M, sel);
    expect(e.parts.find((p) => p.id === 'kv')!.gb).toBeGreaterThan(estimate(VL1M, defaults(VL1M)).parts.find((p) => p.id === 'kv')!.gb);
    expect(chatCost(VL1M, sel)).toMatchObject({ window: 524_288, full: 5.1 });
  });
  it('the short form', () => {
    expect([8192, 131_072, 262_144, 258_048, 1_048_576].map(shortTokens)).toEqual(['8k', '128k', '256k', '252k', '1M']);
  });
});
