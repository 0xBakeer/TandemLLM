// The figures line under an answer, as readings (VIS-185). Pure, so it is tested without a DOM.

import { exact, fixed, money, ms, pct } from '../lib/format';
import type { TurnStats } from './types';

export interface TurnFigure {
  /** the reading's key: the line hides some on a phone by it */
  k: 'finish' | 'tps' | 'tok' | 'ttft' | 'round' | 'acc' | 'cost' | 'main' | 'workers' | 'total';
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
  // a cloud model's turn at the proxy's prices
  if (s.cost_usd != null) out.push({ k: 'cost', n: money(Math.round(s.cost_usd * 1e6)), w: '' });
  return out;
}

/** VIS-E15: a turn that gave tasks reads its prompt tokens apart and together, "main 31,204 ·
 *  workers 2 (96,410) · total 127,614 prompt tokens" (the turn's totals count both), on a row of
 *  its own under the figures line, so none of it is cut. A turn without workers has none. */
export function workerFigures(s: Pick<TurnStats, 'prompt_tokens' | 'workers'>): TurnFigure[] {
  const w = s.workers;
  if (!w?.count) return [];
  return [
    { k: 'main', n: exact(Math.max(0, s.prompt_tokens - w.prompt_tokens)), w: 'main ', pre: true },
    { k: 'workers', n: exact(w.count), w: 'workers ', pre: true, sub: `(${exact(w.prompt_tokens)})` },
    { k: 'total', n: exact(s.prompt_tokens), w: 'total ', pre: true, sub: 'prompt tokens' },
  ];
}

