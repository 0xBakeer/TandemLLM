// Simple mode of the Setup Guide: the model cards STAY — people know the model they want. What simple mode shortens is the way through
// them: three pages instead of eight — 1 the cards, 2 the quant cards with the noise lines
// stripped, 3 everything else folded into title bars nobody has to open — and it answers the
// quiet questions itself: the model's full context window (not the 32K a cautious default
// picks), up to four chats, never past the card's slots.max, the decision models and the
// response cache on. Every page can
// open the full options; the switch at the top returns to the eight steps at any time.
//
// The mode is a browser memory (localStorage), never a server state: a shared box has several
// beginners and one owner, and the owner's advanced progress must not be hidden from them.

import { html, nothing, type TemplateResult } from 'lit';
import type { SetupModel, SetupSelection } from '../api/types';
import { windowMax, type SetupAction } from './state';
import { estimateLoose } from './estimate';
import { SIMPLE } from './strings';

export const SIMPLE_KEY = 'tandem.setup.simple';

/** the stored mode; an absent key means simple — a newcomer starts on the short way. */
export function simpleOn(): boolean {
  try {
    const v = localStorage.getItem(SIMPLE_KEY);
    return v === null ? true : v === '1';
  } catch {
    return true; // private mode: simple, per session
  }
}

export function setSimple(on: boolean): void {
  try {
    localStorage.setItem(SIMPLE_KEY, on ? '1' : '0');
  } catch {
    // no storage: the switch lives in this session only
  }
}

/** the in-model simple flow: the quant cards, then the one folded page (the tiles are page 1
 *  and they are the advanced ones — people read model names, not modes). */

/** the wizard answers a newcomer would not ask, as ordinary actions — so the reducer, the
 *  memory bar and the plan see exactly what a click would have produced. The window is the
 *  model's own longest one the profile takes (the model was made for it); the parallel chats
 *  go to four where the model takes them; the decision models and the response cache come on
 *  when the model offers them. */
export function simpleDefaultsActions(m: SetupModel, sel: SetupSelection): SetupAction[] {
  const a: SetupAction[] = [];
  const w = windowMax(m, sel.profile);
  if (sel.window !== w) a.push({ type: 'parallel', key: 'window', value: w });
  // up to four chats at the official window — never past what the card measured (slots.max),
  // and the pool goes on only when it buys chats the plain reservation cannot carry: every
  // window committed at load costs a full window a chat, so the pool earns its rows or stays
  // as the box left it.
  const cap = Math.max(1, Math.min(4, m.parallelism.slots?.max ?? 4));
  const fit = (pool: boolean, n: number) =>
    (estimateLoose(m, { ...sel, window: w, kv_pool: pool, slots: n })?.fits ?? false);
  let plain = cap; while (plain > 1 && !fit(false, plain)) plain--;
  let pooled = cap; while (pooled > 1 && !fit(true, pooled)) pooled--;
  const slots = pooled > plain ? pooled : plain;
  const pool = pooled > plain ? true : sel.kv_pool;
  if (sel.kv_pool !== pool) a.push({ type: 'parallel', key: 'kv_pool', value: pool });
  if (sel.slots !== slots) a.push({ type: 'parallel', key: 'slots', value: slots });
  for (const id of ['laya', 'response_cache']) {
    // these two are components (memory-bearing toggles, not methods) where a model has them
    if ((m.components ?? []).some((c) => c.id === id) && sel.components?.[id] !== true) a.push({ type: 'component', id, on: true });
  }
  return a;
}

/** the mode switch, top of the page: two names, the current one is the pressed button. */
export function simpleSwitch(on: boolean, onToggle: (v: boolean) => void): TemplateResult {
  return html`<div class="setup-mode-switch" role="group" aria-label=${SIMPLE.switchLabel}>
    <button class="btn btn-sm ${on ? 'btn-primary' : 'btn-ghost'}" aria-pressed=${on} @click=${() => onToggle(true)}>${SIMPLE.simple}</button>
    <button class="btn btn-sm ${on ? '' : 'btn-primary'} ${on ? 'btn-ghost' : ''}" aria-pressed=${!on} @click=${() => onToggle(false)}>${SIMPLE.advanced}</button>
    <span class="muted setup-mode-note">${on ? SIMPLE.modeNoteSimple : SIMPLE.modeNoteAdvanced}</span>
  </div>`;
}

/** one fold of the third page: a step's whole body behind its own title bar. */
/** Every folded step starts shut — EXCEPT the Review fold: on the third simple page the
 *  review IS the page, and a shut fold there hides the very reasons Finish cannot run. */
export function simpleFold(label: string, step: string, body: TemplateResult): TemplateResult {
  return html`<details class="setup-fold" data-fold=${step} ?open=${step === 'review'}>
    <summary class="setup-fold-head">${label}</summary>
    <div class="setup-fold-body">${body}</div>
  </details>`;
}
