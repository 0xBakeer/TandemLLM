// <qse-restart-card>: the one screen of a restart (VIS-209). While a Setup Guide job restarts the
// engine, the page's own server is away: the card asks /v1/setup/status every 2 s (the engine, else the
// status responder through the proxy or on its port; lib/restart.ts) and shows the job's step, the
// load's phases, the time it has taken and about how long is left. With no answer at all it counts the
// time against a typical start; far past it, it says so and offers a refresh. It shows nothing once
// the engine answers (`engine.state` up): the job view goes on from there.

import { html, nothing } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import { apiBase } from '../api/client';
import type { SetupJob } from '../api/types';
import { cardModel, fetchStatus, type SetupStatus, statusUrls } from '../lib/restart';
import { LightElement } from '../ui/base';
import { icon } from '../ui/bits';

@customElement('qse-restart-card')
export class QseRestartCard extends LightElement {
  @property({ attribute: false }) job: SetupJob | null = null;
  /** when the restart began, ms (the switch step's start; else when the card appeared) */
  @property({ type: Number }) since = 0;
  @state() private st: SetupStatus | null = null;
  @state() private asked = false;
  @state() private now = Date.now();
  private timer: ReturnType<typeof setInterval> | null = null;
  private polling = false;
  private ticks = 0;
  private born = Date.now();

  connectedCallback(): void {
    super.connectedCallback();
    this.born = Date.now();
    void this.poll();
    this.timer = setInterval(() => {
      this.now = Date.now();
      if (++this.ticks % 2 === 0) void this.poll();
    }, 1000);
  }
  disconnectedCallback(): void {
    if (this.timer) clearInterval(this.timer);
    super.disconnectedCallback();
  }

  private async poll(): Promise<void> {
    if (this.polling) return;
    this.polling = true;
    try {
      const st = await fetchStatus(statusUrls(apiBase()));
      this.st = st;
      this.asked = true;
      if (st?.engine?.state === 'up') this.dispatchEvent(new CustomEvent('engine-up', { bubbles: true }));
    } finally {
      this.polling = false;
    }
  }

  render() {
    if (this.st?.engine?.state === 'up') return nothing;
    const m = cardModel(this.job, this.asked ? this.st : null, this.now, this.since || this.born);
    return html`<section class="rcard" role="status" aria-live="polite" data-restart-card data-blind=${m.blind ? 'yes' : 'no'}>
      <header class="rcard-head">
        <span class="inst-spin" aria-hidden="true"></span>
        <div>
          <h2 class="rcard-title">${m.title}</h2>
          ${m.step ? html`<p class="rcard-step muted" data-restart-step>${m.step}</p>` : nothing}
        </div>
      </header>
      <div class="rcard-bar" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow=${Math.round(m.progress * 100)}><span style="width:${Math.round(m.progress * 100)}%"></span></div>
      <p class="rcard-time num"><span data-restart-elapsed>${m.elapsed}</span> · <span data-restart-left>${m.left}</span></p>
      ${m.phases.length
        ? html`<ol class="rcard-phases">
            ${m.phases.map(
              (p) => html`<li class="rcard-phase ${p.done ? 'is-done' : ''} ${p.current ? 'is-current' : ''}" data-phase=${p.id}>
                <span class="rcard-phase-mark" aria-hidden="true">${p.done ? icon('check') : p.current ? html`<span class="inst-spin"></span>` : html`<span class="inst-dot"></span>`}</span>
                <span>${p.label}${p.current && p.id === 'weights' && m.detail ? html`<span class="muted small rcard-detail"> · ${m.detail}</span>` : nothing}</span>
              </li>`,
            )}
          </ol>`
        : html`<p class="muted small" data-restart-waiting>${m.blind ? 'The engine is loading. This page cannot see its progress from here, so it counts the time; it keeps checking every 2 s.' : 'The engine is loading.'}</p>`}
      <p class="muted small">The dashboard comes back by itself when the engine answers. You can leave this page; the install goes on.</p>
      ${m.late
        ? html`<div class="inst-away is-lost" role="alert" data-restart-late>${icon('warn')}
            <div><p>This takes much longer than usual. The engine may have failed to start: refresh the page to see the install's state.</p>
            <div class="inst-actions"><button class="btn btn-sm" @click=${() => location.reload()}>Refresh the page</button></div></div>
          </div>`
        : nothing}
    </section>`;
  }
}
