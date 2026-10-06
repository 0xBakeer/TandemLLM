// OPS-65: the shell's offline banner in words, from what the reverse proxy's status responder
// (ops/engine_status.py) said about an engine that is away; the old line when it said nothing.
import type { EngineAway } from '../api/client';

export function awayText(a: EngineAway | null, now: (s: number) => string = (s) => new Date(s * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })): string {
  if (!a) return 'The engine is not answering';
  const since = a.since ? ` since ${now(a.since)}` : '';
  const left = a.left_s == null ? '' : a.left_s >= 90 ? `, back in about ${Math.round(a.left_s / 60)} min` : a.left_s > 0 ? `, back in about ${Math.max(1, Math.round(a.left_s))} s` : ', back any moment';
  return `TandemLLM is ${a.words}${since}${a.reason ? ` (${a.reason})` : ''}${left}`;
}
