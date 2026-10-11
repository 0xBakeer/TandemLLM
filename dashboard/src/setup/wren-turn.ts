// Wren's turn in the browser, as data (VIS-91): whether the page asks again after a round of tool
// calls, and with the tools on or off; which notice a turn shows; what the phone's sheet shows for
// a call. No DOM here, so the unit tests reach it.
//
// On 0.7.1 a plain question on the Model step ended in eight "tried something that does not exist"
// lines, the 4-round stop and no answer: the 2B pointed at the step's name after every answer, the
// page refused it, and every refused call meant one more round and one more line.

import type { SetupChatMessage } from '../api/types';

export const MAX_ROUNDS = 4;

/** calls that change nothing on the page: a refusal of one needs no repair and no notice */
const LOOK_ONLY = new Set(['explain', 'read_device']);

export interface RoundCall {
  name: string;
  /** the page ran it (or it was a highlight it could make) */
  ok: boolean;
  changed?: boolean;
}

export interface Round {
  /** the round that just ended, from 1 */
  round: number;
  /** it was asked with the tools on */
  tools: boolean;
  /** its words */
  text: string;
  calls: RoundCall[];
  /** a repair was asked for earlier in this turn */
  retried: boolean;
}

/** null: the turn is over. Else the next round, with the tools on or off. A refused change gets one
 *  repair with the tools on; after that, after a highlight with no words, and in the last of
 *  MAX_ROUNDS rounds, the tools are off and the answer is words. */
export function nextRound(r: Round): { tools: boolean; retried: boolean } | null {
  if (!r.tools || !r.calls.length) return null;
  const last = r.round + 1 >= MAX_ROUNDS;
  if (r.calls.some((c) => !c.ok && !LOOK_ONLY.has(c.name))) return r.retried ? { tools: false, retried: true } : { tools: !last, retried: true };
  if (r.calls.every((c) => LOOK_ONLY.has(c.name) && c.name !== 'read_device')) return r.text.trim() ? null : { tools: false, retried: r.retried };
  return { tools: !last, retried: r.retried };
}

/** At most one notice a turn, after its last message: a change the page refused and no later
 *  change in the same turn made. A refused highlight says nothing. */
export function turnNotices(chat: SetupChatMessage[]): Map<number, 'stale' | 'unchanged'> {
  const out = new Map<number, 'stale' | 'unchanged'>();
  let start = 0;
  const turn = (end: number) => {
    const names = new Map<string, string>();
    let refused: 'stale' | 'unchanged' | null = null;
    for (let i = start; i < end; i++) {
      const m = chat[i];
      if (m.role === 'assistant') for (const c of m.tool_calls ?? []) names.set(c.id, c.name);
      if (m.role !== 'tool') continue;
      if (LOOK_ONLY.has(names.get(m.tool_call_id ?? '') ?? '')) continue;
      let r: { ok?: boolean; stale?: boolean } = {};
      try {
        r = JSON.parse(m.content);
      } catch {
        /* an old or foreign message */
      }
      if (r.ok === false) refused = r.stale || refused === 'stale' ? 'stale' : 'unchanged';
      else if (r.ok === true) refused = null;
    }
    if (refused && end > start) out.set(end - 1, refused);
  };
  chat.forEach((m, i) => {
    if (m.role !== 'user' || i === 0) return;
    turn(i);
    start = i;
  });
  turn(chat.length);
  return out;
}

/** The phone's sheet stays open while Wren works; a change shows here as a toast with Show me.
 *  A refused call, a highlight and a desktop (the field is beside the panel) show none. */
export function toastFor(out: { ok: boolean; summary: string | null }, sheet: boolean): string | null {
  return sheet && out.ok && out.summary ? out.summary : null;
}
