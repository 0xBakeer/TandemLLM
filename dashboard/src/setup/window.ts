// The window stepper of the memory step (VIS-199; SRV-707: one window for every lane). − and + walk a
// ladder of doublings from 8,192 up to the model's own maximum (the card's window.max, 1,048,576 on
// Super VL), clipped by the profile; the field takes any number of tokens ("131072", "131,072", "128k",
// "1m").
// A typed value outside the range or off the card's step stays as typed with the reason, never
// clamped, and the selection keeps its last good window (the same rules as the chats stepper).

import type { SetupModel, SetupSelection } from '../api/types';
import { windowMax } from './state';
import { S } from './strings';

export type WindowKey = 'window';

export const LADDER_FIRST = 8_192;

export interface WindowRange {
  min: number;
  max: number;
  step: number;
  /** what the field says above `max`: the model's or the profile's limit */
  maxWhy: string;
}

/** The range the window field takes now: the card's, up to the profile's largest window. */
export function windowRange(m: SetupModel, sel: SetupSelection, _key: WindowKey = 'window'): WindowRange {
  const card = m.parallelism.window;
  const max = windowMax(m, sel.profile);
  const prof = m.profiles.find((p) => p.id === sel.profile);
  return { min: card.min, max, step: card.step, maxWhy: max < card.max && prof ? S.windowAboveProfile(prof.label, max) : S.windowAtMost(max) };
}

/** A number of tokens as typed: digits with , _ ' or spaces between them, or a k (1,024) or m
 *  (1,048,576) form such as 128k, 1m or 1.5m. null when it is not one. */
export function parseTokens(raw: string): number | null {
  const t = raw.trim().toLowerCase().replace(/\s+/g, '');
  if (!t) return null;
  const short = /^(\d+(?:\.\d+)?)([km])$/.exec(t);
  if (short) {
    const n = Number(short[1]) * (short[2] === 'k' ? 1024 : 1_048_576);
    return Number.isInteger(n) ? n : Math.round(n);
  }
  if (!/^\d{1,3}(?:[,_']\d{3})+$|^\d+$/.test(t)) return null;
  return Number(t.replace(/[,_']/g, ''));
}

export type WindowParse = { ok: true; n: number } | { ok: false; why: string };

export function checkWindow(raw: string, r: WindowRange): WindowParse {
  const n = parseTokens(raw);
  if (n == null) return { ok: false, why: S.windowNotANumber };
  if (n > r.max) return { ok: false, why: r.maxWhy };
  if (n < r.min) return { ok: false, why: S.windowAtLeast(r.min) };
  if (r.step && n % r.step) {
    const lo = Math.max(r.min, Math.floor(n / r.step) * r.step);
    const hi = Math.min(r.max, lo + r.step);
    return { ok: false, why: S.windowOffStep(r.step, lo, hi) };
  }
  return { ok: true, n };
}

/** The rungs inside the range: the doublings from 8,192 up to the model's own maximum, which is a
 *  rung of its own when it is not a doubling. */
export function rungs(r: WindowRange): number[] {
  const out = new Set<number>([r.min, r.max]);
  for (let n = LADDER_FIRST; n <= r.max; n *= 2) if (n >= r.min) out.add(n);
  return [...out].sort((a, b) => a - b);
}

/** One rung up (dir 1) or down (-1) from `n`; from a value between two rungs, the next one that way. */
export function stepWindow(n: number, dir: number, r: WindowRange): number {
  const rs = rungs(r);
  if (dir > 0) return rs.find((x) => x > n) ?? r.max;
  return [...rs].reverse().find((x) => x < n) ?? r.min;
}

/** 262,144 -> "256k", 1,048,576 -> "1M": the short form beside the field. */
export function shortTokens(n: number): string {
  if (n >= 1_048_576 && n % 1_048_576 === 0) return `${n / 1_048_576}M`;
  const k = n / 1024;
  return Number.isInteger(k) ? `${k}k` : `${k.toFixed(1)}k`;
}
