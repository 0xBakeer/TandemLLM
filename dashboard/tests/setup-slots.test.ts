// VIS-199 (QA VIS-200): the parallel-chats stepper of the memory step. A typed count is never
// clamped, the card's measured range is a note and not a cap, the setup's 64 is the cap (SRV-707), memory
// decides below it with the bar's own estimate, and one extra chat costs what the bar adds.
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import superVl from '../mock/setup/nemotron-3.5-super-vl-120b-a12b.json';
import flashNext from '../mock/setup/qwen3.8-flash-next.json';
import qwen from '../mock/setup/qwen3.8-27b.json';
import type { SetupModel } from '../src/api/types';
import { stepIssues } from '../src/setup/checks';
import { estimate } from '../src/setup/estimate';
import { deriveSteps } from '../src/setup/fields';
import { chatCost, ENGINE_MAX_SLOTS, fitMax, parseSlots, slotsBlocker, slotsRange, slotsVerdict, stepSlots } from '../src/setup/slots';
import { defaults } from '../src/setup/state';
import { S } from '../src/setup/strings';

const VL = superVl as unknown as SetupModel;
const Q = qwen as unknown as SetupModel;
const FN = flashNext as unknown as SetupModel;

describe('the engine limit', () => {
  it('is the server\'s ENGINE_MAX_SLOTS (server/setup/registry.py), 64', () => {
    const py = readFileSync(new URL('../../server/setup/registry.py', import.meta.url), 'utf8');
    expect(Number(/^ENGINE_MAX_SLOTS = (\d+)$/m.exec(py)?.[1])).toBe(ENGINE_MAX_SLOTS);
    expect(ENGINE_MAX_SLOTS).toBe(64);
  });
  it('is the range, whatever the card measured; Wren\'s schema takes it too', () => {
    expect(slotsRange(VL)).toEqual({ min: 1, max: 64, measured: 3 });
    expect(slotsRange(Q)).toEqual({ min: 1, max: 64, measured: 8 });   // SRV-711: 8 lanes measured
    expect(deriveSteps(VL).find((s) => s.id === 'parallelism')!.fields[0]).toMatchObject({ field: 'slots', min: 1, max: 64 });
  });
});

describe('parseSlots: what is typed', () => {
  const r = slotsRange(VL);
  it('takes a whole number in range, spaces around it too', () => {
    expect(parseSlots('5', r)).toEqual({ ok: true, n: 5 });
    expect(parseSlots(' 16 ', r)).toEqual({ ok: true, n: 16 });
    expect(parseSlots('04', r)).toEqual({ ok: true, n: 4 });
  });
  it('refuses the rest with the reason and never clamps', () => {
    expect(parseSlots('', r)).toEqual({ ok: false, why: S.slotsEmpty });
    expect(parseSlots('2.5', r)).toEqual({ ok: false, why: S.slotsNotWhole });
    expect(parseSlots('-1', r)).toEqual({ ok: false, why: S.slotsNotWhole });
    expect(parseSlots('abc', r)).toEqual({ ok: false, why: S.slotsNotWhole });
    expect(parseSlots('0', r)).toEqual({ ok: false, why: 'At least 1 chat.' });
    expect(parseSlots('17', r)).toEqual({ ok: true, n: 17 });
    expect(parseSlots('65', r)).toEqual({ ok: false, why: S.slotsEngineMax(64) });
    expect(S.slotsEngineMax(64)).toMatch(/at most 64 chats at once/);
  });
});

describe('stepSlots: minus, plus and the arrow keys', () => {
  const r = slotsRange(VL);
  it('steps by one and stops at 1 and at 64', () => {
    expect(stepSlots(3, 1, r)).toBe(4);
    expect(stepSlots(3, -1, r)).toBe(2);
    expect(stepSlots(1, -1, r)).toBe(1);
    expect(stepSlots(16, 1, r)).toBe(17);
    expect(stepSlots(64, 1, r)).toBe(64);
  });
});

describe('chatCost: what one more chat adds', () => {
  it('Super VL, the pool off: its whole 262,144-token window and the slot\'s fixed part, 3.1 GiB, from the start', () => {
    // 2.0 GiB of KV and side_slot_const_gib 1.1 (measured at load: 3.10 GiB of MemAvailable a side slot, ENG-363)
    const c = chatCost(VL, defaults(VL));
    expect(c).toEqual({ now: 3.1, full: 3.1, window: 262_144 });
    expect(S.slotsCost(c.now, c.full, c.window)).toBe('Each extra chat takes about 3.1 GiB at a 262,144-token window.');
  });
  it('the pool on: its typical fill to start, its window at most', () => {
    const c = chatCost(VL, { ...defaults(VL), kv_pool: true });
    expect(c.now).toBeCloseTo(1.35, 5);
    expect(S.slotsCost(c.now, c.full, c.window)).toBe('Each extra chat takes about 1.4 GiB to start, up to 3.1 GiB at a 262,144-token window.');
  });
  it('follows the one window every chat has (SRV-707)', () => {
    expect(chatCost(VL, { ...defaults(VL), window: 65_536 }).window).toBe(65_536);
    expect(chatCost(VL, { ...defaults(VL), window: 32_768 }).window).toBe(32_768);
  });
  it('is exactly what the bar adds for one more chat (27B with its fixed part, Super VL, pool on and off)', () => {
    for (const [m, pool] of [[Q, true], [Q, false], [VL, true], [VL, false]] as const) {
      const sel = { ...defaults(m), kv_pool: pool, window: 65_536, slots: 2 };
      const delta = estimate(m, { ...sel, slots: 3 }).total_gb - estimate(m, sel).total_gb;
      expect(Math.abs(delta - chatCost(m, sel).now)).toBeLessThan(0.02);
    }
  });
});

describe('fitMax and the verdict: memory decides', () => {
  it('fitMax is the last count that fits; one more does not (or it is the setup\'s 64)', () => {
    for (const m of [Q, VL]) {
      for (const pool of [true, false]) {
        const sel = { ...defaults(m), kv_pool: pool };
        const n = fitMax(m, sel);
        expect(n).toBeGreaterThanOrEqual(1);
        expect(estimate(m, { ...sel, slots: n }).fits).toBe(true);
        if (n < 64) expect(estimate(m, { ...sel, slots: n + 1 }).fits).toBe(false);
      }
    }
    expect(fitMax(VL, defaults(VL))).toBe(8);
    expect(defaults(VL).slots).toBe(3);     // SPD-245: the default is the card's measured max
  });
  it('above the card\'s measured range: allowed, with the untested note', () => {
    const v = slotsVerdict(VL, { ...defaults(VL), slots: 5 });
    expect(v).toMatchObject({ n: 5, untested: true, over: false });
    expect(S.slotsUntested(3)).toBe('Measured up to 3 on this model. More is untested.');
    expect(slotsVerdict(VL, { ...defaults(VL), slots: 3 }).untested).toBe(false);
    expect(slotsVerdict(VL, defaults(VL)).untested).toBe(false);   // 3 by default, the measured max (SPD-245)
    expect(slotsVerdict(VL, { ...defaults(VL), slots: 8 }).untested).toBe(true);
  });
  it('a count that does not fit: over, by how much, and the most that fit; Next is held with that reason', () => {
    const sel = { ...defaults(Q), kv_pool: false, slots: 4 };
    const v = slotsVerdict(Q, sel);
    expect(v).toMatchObject({ n: 4, over: true, fitMax: 2 });
    expect(v.overGb).toBeGreaterThan(0);
    const why = slotsBlocker(v, null);
    expect(why).toBe(S.slotsOver(4, v.overGb, 2));
    expect(why).toMatch(/^4 chats do not fit: \d+\.\d GiB over\. At most 2 fit with these settings\.$/);
    expect(stepIssues(Q, sel, null, estimate(Q, sel)).parallelism).toEqual([why]);
  });
  it('a setup over the budget even with one chat: the count is not to blame, Next is not held for it', () => {
    const sel = { ...defaults(FN), profile: 'flashnext-6.05' };
    const v = slotsVerdict(FN, sel);
    expect(v.fitMax).toBe(0);
    expect(v.over).toBe(false);
    expect(slotsBlocker(v, null)).toBeNull();
  });
  it('a typed value the engine does not take holds Next with its own reason', () => {
    const v = slotsVerdict(VL, defaults(VL));
    expect(slotsBlocker(v, parseSlots('65', slotsRange(VL)))).toBe(S.slotsEngineMax(64));
    expect(slotsBlocker(v, parseSlots('40', slotsRange(VL)))).toBeNull();
    expect(slotsBlocker(v, parseSlots('', slotsRange(VL)))).toBe(S.slotsEmpty);
    expect(slotsBlocker(v, parseSlots('6', slotsRange(VL)))).toBeNull();
    expect(S.nextRefused(S.slotsEmpty)).toBe('Next stays on this step. Type how many chats, 1 or more.');
  });
  it('the estimate takes 64 and refuses 65 with the setup\'s reason, as the server does', () => {
    expect(estimate(VL, { ...defaults(VL), slots: 17, window: 32_768 }).parts.find((p) => p.id === 'side_slots')).toBeTruthy();
    expect(() => estimate(VL, { ...defaults(VL), slots: 65 })).toThrow(/above 64, the most lanes the setup takes/);
  });
  it('past 16 chats the verdict says a round checks in more passes (SRV-707)', () => {
    expect(slotsVerdict(VL, { ...defaults(VL), slots: 16 }).passes).toBe(1);
    expect(slotsVerdict(VL, { ...defaults(VL), slots: 17, window: 32_768 }).passes).toBe(2);
    expect(S.slotsPasses(2)).toBe('Past 16 chats a step checks the drafts in 2 passes instead of one, so each chat runs slower.');
  });
});

describe('the guarantee: how many chats can all hold the full window (SRV-707)', () => {
  it('is the estimate\'s lanes_full, every chat with the pool off when it fits', () => {
    const off = { ...defaults(VL) };
    expect(off.kv_pool).toBe(false);
    const e = estimate(VL, off);
    expect(slotsVerdict(VL, off).full).toBe(e.lanes_full);
    expect(e.lanes_full).toBe(3);          // its 3 lanes by default (SPD-245)
    expect(S.lanesFull(8, 8, 262_144, 2, false)).toBe('All 8 chats can be 262,144 tokens long at the same time (2.0 GiB each).');
  });
  it('with the pool on the 27B: fewer than the chats, and what happens past it', () => {
    const sel = defaults(Q);
    const e = estimate(Q, sel);
    expect(sel.slots).toBe(8);
    expect(e.lanes_full).toBe(Math.min(8, Math.floor(e.room_gb / e.lane_full_gb)));
    expect(e.lanes_full).toBeLessThan(8);
    expect(S.lanesFull(1, 8, 262_144, 26.06, true)).toMatch(/^1 of 8 chats can be 262,144 tokens long .* a new chat waits for memory before it starts, idle chats give their memory back first, and a chat that runs out of memory mid-answer pauses and continues when memory is free\.$/);
    expect(S.lanesFull(0, 8, 262_144, 26.06, true)).toMatch(/^Not one chat can reach 262,144 tokens/);
  });
});
