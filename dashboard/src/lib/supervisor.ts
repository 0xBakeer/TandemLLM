// The session monitor where Khaled already looks (VIS-44): the Live banner's line, a request row's
// badge and its drawer, and the header's mode indicator. Pure functions over the live message, a
// ledger row and the journal's events, no DOM. A field an older server does not send gives null,
// and null renders nothing. The Conductor view reuses the action words and `eventWhat`.

import type { LiveConductor, LiveDecision, MonitorMode, RequestConductor } from '../api/types';
import { exact } from './format';

/** What an action did, in the past tense: "closed the reasoning". */
export const ACTION_WORDS: Record<string, string> = {
  CLOSE_REASONING: 'closed the reasoning',
  CONTINUE_HIDDEN: 'continued an empty turn',
  REGENERATE_CALL: 'had the tool calls written again',
  REPAIR_CALL: 'repaired a tool call',
  GIVE_UP: 'gave up the turn',
  PROMPT_TAIL_NOTE: 'added a hidden note',
  SET_THINK_BUDGET: 'set the thinking budget',
  NOTIFY: 'pushed a notification',
  PASS: 'passed',
};

/** The parts of a journal event (`/v1/conductor/events`) these words read. */
export interface EventLike {
  id?: string;
  ts_ms?: number;
  task: string | null;
  tier?: number | null;
  mode: 'shadow' | 'act' | null;
  decision: string | null;
  trigger?: string | null;
  score?: number | null;
  cap?: string | null;
  hidden_tokens?: number | null;
  tokens_at_cut?: number | null;
  outcome?: { kind?: string | null; finish?: string | null } | null;
}

/** A monitor decision executed something (mode act, not a pass). */
export function isAct(e: Pick<EventLike, 'mode' | 'decision'>): boolean {
  return e.mode === 'act' && e.decision != null && e.decision !== 'PASS';
}

/** A shadow decision that would have done something. */
export function isShadow(e: Pick<EventLike, 'mode' | 'decision'>): boolean {
  return e.mode === 'shadow' && e.decision != null && e.decision !== 'PASS';
}

/** A monitor decision in words for the feed: "C4 had the tool calls written again (unreadable...)";
 * a shadow one says what it would have done. */
export function eventWhat(e: Pick<EventLike, 'task' | 'decision' | 'mode' | 'trigger' | 'cap'>): string {
  const d = e.decision ?? '';
  const words = ACTION_WORDS[d] ?? d.toLowerCase();
  const verb = e.mode === 'shadow' && d !== 'PASS' ? `would have ${words}` : words;
  return `${e.task ?? '—'} ${verb}${e.trigger ? ` (${e.trigger})` : ''}${e.cap ? ` · cap ${e.cap}` : ''}`;
}

/** A score or probability: two decimals, a whole number as it is. */
export function scoreText(v: number | null | undefined): string | null {
  if (v == null || !Number.isFinite(v)) return null;
  return Number.isInteger(v) ? String(v) : v.toFixed(2);
}

// ---- the Live banner ----------------------------------------------------------------------------

export interface SupervisorLine {
  text: string; // "Supervisor: closed the reasoning at 2,880 tokens (D1, score 0.93)"
  title: string; // the same with the rule's reason and the age
  shadow: boolean;
}

/**
 * The banner's line for the running request: its last action, or, with none, its newest shadow
 * decision ("would have ..."). Null for a request the monitor does not watch, for a monitor that
 * has decided nothing yet, and for a 1.1 server.
 */
export function supervisorLine(c: LiveConductor | null | undefined): SupervisorLine | null {
  if (!c) return null;
  const last = c.last;
  const d: LiveDecision | null = c.last_action ?? (last && isShadow({ mode: last.mode as EventLike['mode'], decision: last.decision }) ? last : null);
  if (!d) return null;
  const shadow = d.mode === 'shadow';
  const words = ACTION_WORDS[d.decision] ?? d.decision.toLowerCase();
  const at = d.at_tokens != null ? ` at ${exact(d.at_tokens)} tokens` : '';
  const score = scoreText(d.score);
  const why = [d.task, score != null ? `score ${score}` : d.reason].filter(Boolean).join(', ');
  const text = `Supervisor${shadow ? ' (shadow)' : ''}: ${shadow ? 'would have ' : ''}${words}${at}${why ? ` (${why})` : ''}`;
  const extra = [d.reason && score != null ? d.reason : null, d.tier != null ? `tier ${d.tier}` : null, d.ago_ms != null ? `${agoWords(d.ago_ms)} ago` : null].filter(Boolean);
  return { text, title: extra.length ? `${text} · ${extra.join(' · ')}` : text, shadow };
}

function agoWords(ms: number): string {
  const s = ms / 1000;
  if (s < 10) return `${s.toFixed(1)} s`;
  if (s < 90) return `${Math.round(s)} s`;
  return `${Math.round(s / 60)} min`;
}

// ---- a request's badge ----------------------------------------------------------------------------

export interface Badge {
  n: number;
  kind: 'act' | 'shadow'; // filled for actions, outlined for shadow decisions
  label: string; // "2 supervisor actions" / "1 shadow decision"
}

/** The row's badge: the actions as a filled count; with none, the shadow decisions outlined;
 *  null when the monitor neither acted nor would have (and on an older server). */
export function badge(c: Pick<LiveConductor | RequestConductor, 'acts'> & { shadow?: number } | null | undefined): Badge | null {
  if (!c) return null;
  if (c.acts > 0) return { n: c.acts, kind: 'act', label: `${c.acts} supervisor action${c.acts === 1 ? '' : 's'}` };
  const s = c.shadow ?? 0;
  if (s > 0) return { n: s, kind: 'shadow', label: `${s} shadow decision${s === 1 ? '' : 's'}` };
  return null;
}

// ---- the drawer -------------------------------------------------------------------------------------

export interface EventLine {
  id: string;
  time: string; // "14:02:11"
  task: string;
  tier: string; // "tier 0"
  mode: 'act' | 'shadow';
  score: string; // "0.93", "—"
  action: string; // "closed the reasoning", "would have had the tool calls written again"
  hidden: string; // "22 tok", "—"
  outcome: string; // "tool call · tool_calls"
}

/** One drawer line per decision that did, or would have done, something; passes are left out. */
export function eventLines(events: EventLike[], timeFmt: (ms: number) => string): EventLine[] {
  return events
    .filter((e) => isAct(e) || isShadow(e))
    .sort((a, b) => (a.ts_ms ?? 0) - (b.ts_ms ?? 0))
    .map((e, i) => {
      const words = ACTION_WORDS[e.decision!] ?? e.decision!.toLowerCase();
      const o = e.outcome;
      return {
        id: e.id ?? String(i),
        time: e.ts_ms != null ? timeFmt(e.ts_ms) : '—',
        task: e.task ?? '—',
        tier: e.tier != null ? `tier ${e.tier}` : '—',
        mode: isAct(e) ? 'act' : 'shadow',
        score: scoreText(e.score) ?? '—',
        action: `${isAct(e) ? '' : 'would have '}${words}${e.tokens_at_cut != null ? ` at ${exact(e.tokens_at_cut)} tokens` : ''}`,
        hidden: e.hidden_tokens ? `${exact(e.hidden_tokens)} tok` : '—',
        outcome: o ? [o.kind ? o.kind.replace(/_/g, ' ') : null, o.finish ?? null].filter(Boolean).join(' · ') || '—' : 'pending',
      };
    });
}

// ---- the header ------------------------------------------------------------------------------------

/** "monitor shadow" for the header's indicator; null on a server that sends no mode. */
export function modeWord(m: { mode: MonitorMode } | null | undefined): string | null {
  return m ? `monitor ${m.mode}` : null;
}
