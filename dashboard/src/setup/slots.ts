// The parallel-chats stepper of the memory step (VIS-199): what a typed count means, what one more
// chat costs, and whether the count fits. No card cap: a card's slots.max is how far the model was
// measured (above it the step says untested), the engine's limit is ENGINE_MAX_SLOTS, and memory
// decides everything below that with the estimate the bar draws (estimate.ts, the server's copy),
// so the stepper and the bar can never disagree.

import type { SetupModel, SetupSelection } from '../api/types';
import { estimateLoose, GIB, KV_TYPICAL_TOKENS, overBy } from './estimate';
import { ENGINE_MAX_SLOTS } from './fields';
import { S } from './strings';

export { ENGINE_MAX_SLOTS };

export interface SlotsRange {
  min: number;
  max: number;
  /** the card's measured upper end */
  measured: number;
}

export function slotsRange(m: SetupModel): SlotsRange {
  const p = m.parallelism.slots;
  return { min: p.min, max: ENGINE_MAX_SLOTS, measured: p.max };
}

/** A typed count: the number, or why it is not one the engine takes. Never clamped: a value out of
 *  range stays what was typed, with the reason. */
export type SlotsParse = { ok: true; n: number } | { ok: false; why: string };

export function parseSlots(raw: string, r: SlotsRange): SlotsParse {
  const t = raw.trim();
  if (t === '') return { ok: false, why: S.slotsEmpty };
  if (!/^\d+$/.test(t)) return { ok: false, why: S.slotsNotWhole };
  const n = Number(t);
  if (n < r.min) return { ok: false, why: S.slotsBelow(r.min) };
  if (n > r.max) return { ok: false, why: S.slotsEngineMax(r.max) };
  return { ok: true, n };
}

/** − and + (and the arrow keys): one step, inside the range; the buttons are disabled at the ends. */
export function stepSlots(n: number, delta: number, r: SlotsRange): number {
  return Math.min(r.max, Math.max(r.min, Math.round(n) + delta));
}

export interface ChatCost {
  /** what one more chat adds to the bar now: its fixed part and its KV as committed (the pool on:
   *  up to the typical fill; off: its whole window) */
  now: number;
  /** the same chat at its full window */
  full: number;
  /** the extra chats' window, tokens */
  window: number;
}

/** One extra chat, from the estimate's own terms (side_slot_const_gib + side window x KV a token). */
export function chatCost(m: SetupModel, sel: SetupSelection): ChatCost {
  const p = m.parallelism;
  const side = Math.min(sel.side_window ?? sel.window, sel.window);
  const fill = p.typical_fill_tokens ?? KV_TYPICAL_TOKENS;
  const full = (side * p.kv_bytes_per_token) / GIB;
  const typical = (Math.min(side, fill) * p.kv_bytes_per_token) / GIB;
  const c = p.side_slot_const_gib;
  return { now: c + (sel.kv_pool ? typical : full), full: c + full, window: side };
}

/** The most chats that fit with the rest of the selection as it is; 0 when not even one does. */
export function fitMax(m: SetupModel, sel: SetupSelection): number {
  const r = slotsRange(m);
  let best = 0;
  for (let n = r.min; n <= r.max; n++) {
    const e = estimateLoose(m, { ...sel, slots: n });
    // the cost only grows with the count: the first that does not fit ends the search
    if (!e?.fits) break;
    best = n;
  }
  return best;
}

export interface SlotsVerdict {
  n: number;
  range: SlotsRange;
  /** above the card's measured range */
  untested: boolean;
  /** the count is why it does not fit (fewer would): Next and apply refuse it */
  over: boolean;
  /** GiB over the budget at this count */
  overGb: number;
  fitMax: number;
  cost: ChatCost;
}

export function slotsVerdict(m: SetupModel, sel: SetupSelection): SlotsVerdict {
  const range = slotsRange(m);
  const n = sel.slots;
  const most = fitMax(m, sel);
  const e = estimateLoose(m, sel);
  const overGb = e ? overBy(e) : 0;
  return { n, range, untested: n > range.measured, over: most >= 1 && n > most, overGb, fitMax: most, cost: chatCost(m, sel) };
}

/** What stops Next on the memory step because of the count: a typed value the engine does not take,
 *  or a count that does not fit. null: nothing. */
export function slotsBlocker(v: SlotsVerdict, draft: SlotsParse | null): string | null {
  if (draft && !draft.ok) return draft.why;
  if (v.over) return S.slotsOver(v.n, v.overGb, v.fitMax);
  return null;
}
