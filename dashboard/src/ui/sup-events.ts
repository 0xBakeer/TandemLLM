// <qse-sup-events>: the drawer of one request's supervisor decisions (VIS-44), for the Live view's
// rows and the Dev tab's finished requests. It reads the conductor's journal for that request
// (`/v1/conductor/events?request=`) and lists each decision that did, or would have done,
// something: time, task, tier, mode, score or probability, action, hidden tokens, outcome. It
// reads again when the row's count moves. A server without the journal says so in one line.
// `supBadge` is the row's badge: the actions filled, shadow decisions outlined.

import { html, nothing, type TemplateResult } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import { api, describeError } from '../api/client';
import { timeShort } from '../lib/format';
import { eventLines, type Badge, type EventLike, type EventLine } from '../lib/supervisor';
import { LightElement } from './base';

@customElement('qse-sup-events')
export class QseSupEvents extends LightElement {
  @property({ type: String }) requestId = '';
  /** the row's acts plus shadow decisions: a change reads the journal again */
  @property({ type: Number }) count = 0;
  @state() private lines: EventLine[] | null = null;
  @state() private error: string | null = null;
  private loadedFor = '';

  protected willUpdate(): void {
    const key = `${this.requestId}:${this.count}`;
    if (this.requestId && key !== this.loadedFor) {
      this.loadedFor = key;
      void this.load();
    }
  }

  private async load(): Promise<void> {
    const rid = this.requestId;
    try {
      const r = await api.conductorEvents(rid);
      if (rid !== this.requestId) return;
      this.lines = eventLines(r.events as EventLike[], (ms) => timeShort(new Date(ms).toISOString()));
      this.error = null;
    } catch (e) {
      const d = describeError(e);
      this.error = d.status === 404 ? 'the conductor journal is not on this server' : `${d.endpoint}: ${d.message}`;
    }
  }

  render() {
    if (this.error) return html`<p class="muted small sup-events-none">${this.error}</p>`;
    if (!this.lines) return html`<p class="muted small" aria-busy="true">Reading the journal…</p>`;
    if (!this.lines.length) return html`<p class="muted small sup-events-none">No supervisor decision is in the journal for this request yet.</p>`;
    return html`<ol class="sup-events" aria-label="supervisor decisions">
      ${this.lines.map(
        (l) => html`<li class="sup-event is-${l.mode}" data-sup-event=${l.mode}>
          <span class="num sup-ev-time">${l.time}</span>
          <span class="sup-ev-task"><b>${l.task}</b> <span class="muted">${l.tier}</span></span>
          <span class="badge ${l.mode === 'act' ? 'badge-on' : 'badge-shadow'} sup-ev-mode">${l.mode === 'act' ? 'acted' : 'shadow'}</span>
          <span class="sup-ev-action">${l.action}</span>
          <span class="num muted sup-ev-score" title="score or probability">${l.score === '—' ? '—' : `score ${l.score}`}</span>
          <span class="num muted sup-ev-hidden" title="hidden tokens">${l.hidden}</span>
          <span class="muted sup-ev-outcome" title="outcome">${l.outcome}</span>
        </li>`,
      )}
    </ol>`;
  }
}

/** The row's badge as a button that opens the drawer: "2" filled, "1" outlined for shadow. */
export function supBadge(b: Badge | null, open: boolean, toggle: () => void): TemplateResult | typeof nothing {
  if (!b) return nothing;
  return html`<button type="button" class="sup-badge is-${b.kind}" data-sup-badge=${b.kind} aria-expanded=${open} aria-label=${`${b.label}, ${open ? 'hide' : 'show'} them`} title=${b.label} @click=${(e: Event) => (e.stopPropagation(), toggle())}><span aria-hidden="true">◈</span>${b.n}</button>`;
}
