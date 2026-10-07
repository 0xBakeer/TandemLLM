// The side lane (server/preempt.py): a short request -- a client's title or summary
// call, the router's own question to Qwen -- served between two blocks (or two prefill chunks) of
// the generation that holds the engine, which gets the engine back exactly as it was. The words a
// row shows for it, so such a request reads as served beside a turn and not as a refusal.

import type { SideLane } from '../api/types';

export function laneWord(l: SideLane): string {
  return l === 'pending' ? 'side lane, waiting' : 'side lane';
}

export function laneTitle(l: SideLane): string {
  switch (l) {
    case 'decode':
      return 'served between two decode blocks of the running turn, which went on unchanged';
    case 'prefill':
      return 'served between two prefill chunks of the running turn, which went on unchanged';
    case 'self':
      return 'served in the side lane with no turn running';
    default:
      return 'waiting for the running turn’s next block';
  }
}
