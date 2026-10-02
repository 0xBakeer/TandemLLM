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

/** "refused: it would break the floor (MemAvailable 28.0 GiB, claim 8.2 GiB, 19.8 GiB left, 21.0 needed)". */
export function refusalWords(d: DeciderDecision | null | undefined): string | null {
  if (!d || d.allowed) return null;
  const why = d.reason ? REASON_WORDS[d.reason] ?? d.reason : 'refused';
  if (d.reason !== 'refused_memory') return why;
  const nums = [`MemAvailable ${gibOf(d.mem_available_gib)}`];
  // OPS-47: what the engine may still take up to its declared peak comes off first
  if (d.engine_growth_gib) nums.push(`engine growth ${gibOf(d.engine_growth_gib)}`);
  nums.push(`claim ${gibOf(d.claim_gib)}`);
  if (d.after_gib != null) nums.push(`${gibOf(d.after_gib)} left`, `${gibOf(d.need_gib)} needed`);
  return `${why} (${nums.join(', ')})`;
}

/** What a load now would get: "fits, 22.8 GiB left after the claim" or the refusal. */
export function plannerWords(d: DeciderDecision | null | undefined): string {
  if (!d) return '—';
  return d.allowed ? `fits, ${gibOf(d.after_gib)} left after the claim` : refusalWords(d) ?? 'refused';
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
