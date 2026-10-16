// The figures line under an answer, as readings (VIS-185). Pure, so it is tested without a DOM.

import { exact, fixed, ms, pct } from '../lib/format';
import type { TurnStats } from './types';

export interface TurnFigure {
  /** the reading's key: the line hides some on a phone by it */
  k: 'finish' | 'tps' | 'tok' | 'ttft' | 'round' | 'acc';
  /** the number, set a step brighter than its words */
  n: string;
  w: string;
  /** a detail a phone drops */
  sub?: string;
  /** the words read before the number */
  pre?: boolean;
}

/** the figures line as readings: the number apart from its words, so the line can set them in two
 *  inks (`pre`: the word reads before it), and a key per reading, so a phone can keep the ones
 *  that fit (VIS-185). A plain finish ("done") is left out: the line under an answer says that. */
export function figsParts(s: TurnStats): TurnFigure[] {
  const out: TurnFigure[] = [];
  if (s.finish !== 'stop') out.push({ k: 'finish', n: '', w: s.finish });
  if (s.decode_tps != null) out.push({ k: 'tps', n: fixed(s.decode_tps, 1), w: ' tok/s' });
  out.push({ k: 'tok', n: exact(s.completion_tokens), w: ' tok', sub: s.reasoning_tokens ? `(${exact(s.reasoning_tokens)} thinking)` : undefined });
  if (s.ttft_ms != null) out.push({ k: 'ttft', n: ms(s.ttft_ms), w: 'first token ', pre: true });
  if (s.tokens_per_round != null) out.push({ k: 'round', n: fixed(s.tokens_per_round, 1), w: ' per round' });
  if (s.acceptance != null) out.push({ k: 'acc', n: pct(s.acceptance, 0), w: ' accepted' });
  return out;
}

