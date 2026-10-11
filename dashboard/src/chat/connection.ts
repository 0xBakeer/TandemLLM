// The attachment to a running chat, as a state machine with words: connecting, live,
// catching up (a replay from Last-Event-ID), reconnecting (with the seconds), lost (after a
// minute). Pure: the view feeds it stream states and event counts, and reads the label.

import type { StreamState } from '../lib/sse';

export type AttachState = 'idle' | 'connecting' | 'live' | 'catching-up' | 'reconnecting' | 'lost';

export interface Attachment {
  state: AttachState;
  /** when the stream was last open or an event arrived, ms */
  since: number;
  /** events replayed so far in the catch-up */
  replayed: number;
  /** the stream said it is current (the catch-up is over) */
  current: boolean;
}

export const LOST_AFTER_MS = 60_000;

export function initialAttachment(now = Date.now()): Attachment {
  return { state: 'idle', since: now, replayed: 0, current: false };
}

/** The transport changed state. A fresh connection with a Last-Event-ID replays first, so `open`
 *  lands in catching-up until the stream says `current` (or the first live event arrives). */
export function onStreamState(a: Attachment, s: StreamState, resuming: boolean, now = Date.now()): Attachment {
  switch (s) {
    case 'connecting':
      return { ...a, state: 'connecting', since: now, replayed: 0, current: false };
    case 'open':
      return { ...a, state: resuming ? 'catching-up' : 'live', since: now, replayed: 0, current: !resuming };
    case 'reconnecting':
    case 'error':
      return a.state === 'reconnecting' || a.state === 'lost' ? a : { ...a, state: 'reconnecting', since: now, current: false };
    case 'closed':
      return { ...a, state: 'idle', since: now };
    case 'unauthorized':
      return { ...a, state: 'lost', since: now };
  }
}

/** An event arrived. `current` marks the stream's "you are up to date" event. */
export function onEvent(a: Attachment, current: boolean, now = Date.now()): Attachment {
  if (a.state === 'catching-up') return current ? { ...a, state: 'live', since: now, current: true } : { ...a, replayed: a.replayed + 1, since: now };
  return { ...a, state: a.state === 'reconnecting' || a.state === 'connecting' ? 'live' : a.state, since: now, current: true };
}

/** Time passed: a reconnect that lasts a minute is a lost stream. */
export function onTick(a: Attachment, now = Date.now()): Attachment {
  if (a.state === 'reconnecting' && now - a.since >= LOST_AFTER_MS) return { ...a, state: 'lost' };
  return a;
}

/** The words at the Now line's right end. */
export function attachLabel(a: Attachment, now = Date.now()): { text: string; tone: 'muted' | 'live' | 'warn' | 'err' } {
  switch (a.state) {
    case 'idle':
      return { text: '', tone: 'muted' };
    case 'connecting':
      return { text: 'connecting', tone: 'muted' };
    case 'live':
      return { text: 'live', tone: 'live' };
    case 'catching-up':
      return { text: a.replayed ? `catching up, ${a.replayed} events` : 'catching up', tone: 'muted' };
    case 'reconnecting':
      return { text: `reconnecting since ${Math.max(0, Math.round((now - a.since) / 1000))} s`, tone: 'warn' };
    case 'lost':
      return { text: 'stream lost, refresh', tone: 'err' };
  }
}
