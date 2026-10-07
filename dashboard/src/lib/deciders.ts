// The Deciders screen's words and the test box's question (VIS-46): pure functions over the
// /v1/conductor/deciders/status contract, so the screen and its tests read the same thing.

import type { DeciderDecision, DeciderModel, DeciderState, JevAnswer, SystemOneQuestion } from '../api/conductor';
import { gibOf, REASON_WORDS } from './residency';

export const STATE_WORDS: Record<DeciderState, string> = {
  not_loaded: 'not loaded',
  loading: 'loading the weights',
  warming: 'warming up',
  loaded: 'loaded',
  unloading: 'unloading',
  failed: 'failed',
};

/** The role contract a model signs (SRV-133), in one line: "decider · idle · preemptible · 100 ms
 *  then PASS · writes nothing · shares text · claims 4.4 GiB". */
export function roleWords(m: Pick<DeciderModel, 'role'>): string | null {
  const r = m.role;
  if (!r) return null;
  return [r.kind, r.window.replace('_', ' '), r.preemptible ? 'preemptible' : 'not preemptible', `${r.latency_budget_ms} ms then ${r.fallback}`, `writes ${r.writes.replace(/_/g, ' ')}`, `shares ${r.sharing.join(', ') || 'nothing'}`, r.memory_claim_gib == null ? 'no claim' : `claims ${gibOf(r.memory_claim_gib)}`].join(' · ');
}

/** The badge class of a model's state. */
export function stateBadge(s: DeciderState): string {
  if (s === 'loaded') return 'badge-ok';
  if (s === 'failed') return 'badge-warn';
  if (s === 'loading' || s === 'warming' || s === 'unloading') return 'badge-shadow';
  return '';
}

/** FX10: a memory refusal is the planner keeping room for requests, not a fault: the words say so
 * first, then the numbers. */
export const KEPT_FREE = 'not loaded: kept free for requests';

/** "not loaded: kept free for requests (MemAvailable 29.0 GiB, less 8.3 GiB the engine may still
 * take for a long request and the 6.5 GiB claim, leaves 14.2 GiB; the box keeps 19.0 GiB free)";
 * any other refusal in the planner's words ("refused: a hold owns the board"). */
export function refusalWords(d: DeciderDecision | null | undefined): string | null {
  if (!d || d.allowed) return null;
  const why = d.reason ? REASON_WORDS[d.reason] ?? d.reason : 'refused';
  if (d.reason !== 'refused_memory') return why;
  if (d.mem_available_gib == null || d.after_gib == null) return `${KEPT_FREE} (MemAvailable not read)`;
  // OPS-47: what the engine may still take up to its declared peak comes off first. A single
  // request is admitted by the engine's own peak, not by MemAvailable, so this room is the only
  // thing that keeps the floor at a long prefill (docs/parallel-requests.md, beside the small models)
  const less = d.engine_growth_gib
    ? `less ${gibOf(d.engine_growth_gib)} the engine may still take for a long request and the ${gibOf(d.claim_gib)} claim`
    : `less the ${gibOf(d.claim_gib)} claim`;
  const keep = d.margin_gib ? `${gibOf(d.need_gib)} free (a ${gibOf(d.floor_gib)} floor and a ${gibOf(d.margin_gib)} margin)` : `${gibOf(d.need_gib)} free`;
  return `${KEPT_FREE} (MemAvailable ${gibOf(d.mem_available_gib)}, ${less}, leaves ${gibOf(d.after_gib)}; the box keeps ${keep})`;
}

/** What a load now would get: "fits, 22.8 GiB left after the claim" or the refusal. */
export function plannerWords(d: DeciderDecision | null | undefined): string {
  if (!d) return '—';
  return d.allowed ? `fits, ${gibOf(d.after_gib)} left after the claim` : refusalWords(d) ?? 'refused';
}

/** The "a load now" tile's value: fits, kept free (memory) or refused (a hold, a load running). */
export function plannerValue(d: DeciderDecision | null | undefined): string {
  if (!d) return '—';
  return d.allowed ? 'fits' : d.reason === 'refused_memory' ? 'kept free' : 'refused';
}

/** Who a journaled engine load was for: "the shadow reader (qwen3.5-0.8b)" for role `shadow`,
 * else the model the source names. */
export function loadWho(d: DeciderDecision | null | undefined): string {
  if (!d) return '';
  const name = (d.source ?? '').replace(/^engine:/, '');
  if (d.role === 'shadow') return name ? `the shadow reader (${name})` : 'the shadow reader';
  return name;
}

/** "laya-english 12 · laya-multilingual 3", most used first; "no call yet". */
export function picksWords(picks: Record<string, number>): string {
  const e = Object.entries(picks).sort((a, b) => b[1] - a[1]);
  return e.length ? e.map(([m, n]) => `${m} ${n}`).join(' · ') : 'no call yet';
}

/** "512 / 192 / 20": context, question and option limits. */
export function limitsWords(m: Pick<DeciderModel, 'limits'>): string {
  const l = m.limits;
  return [l.context_tokens, l.question_tokens, l.max_options].map((x) => (x == null ? '—' : String(x))).join(' / ');
}

/** The test box's one choice question: options separated by commas or new lines, 2 to 20 of them,
 * each an id without spaces. A string is the reason the question cannot be asked. */
export function choiceQuestion(instructions: string, options: string): SystemOneQuestion | string {
  const ins = instructions.trim();
  if (!ins) return 'Write the question.';
  const ids = options.split(/[,\n]/).map((x) => x.trim()).filter(Boolean);
  if (ids.length < 2) return 'Give at least two options.';
  if (ids.length > 20) return 'At most 20 options (the model card’s limit).';
  if (new Set(ids).size !== ids.length) return 'Two options have the same name.';
  return { type: 'choice', instructions: ins, criteria: Object.fromEntries(ids.map((o) => [o, null])) };
}

/** An answer's probabilities, largest first: [option, p]. A noul is [false, true]. */
export function probsOf(a: JevAnswer): [string, number][] {
  if (a.type === 'noul') return [['true', a.noul ?? 0], ['false', 1 - (a.noul ?? 0)]];
  return Object.entries(a.probabilities ?? {}).sort((x, y) => y[1] - x[1]);
}
