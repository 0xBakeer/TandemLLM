// VIS-68 (the owner, 2026-10-07): the speed guard's result as one neutral line in the build
// information, never a warning: "passed for 0.4.0 · 7 Oct 2026", "not run for this build". It reads
// GET /v1/conductor/evals once (the release tree's shipped report included, server/conductor/
// routes.py); the guard's numbers are the line's title. Without the conductor the route answers
// 404 and the line says the check is not run for this build.
import { html } from 'lit';
import { customElement, state } from 'lit/decorators.js';
import { conductor } from '../api/conductor';
import { speedCheck, type SpeedCheck } from '../lib/conductor';
import { LightElement } from './base';

@customElement('qse-speed-check')
export class SpeedCheckLine extends LightElement {
  @state() private check: SpeedCheck | null = null;

  connectedCallback(): void {
    super.connectedCallback();
    if (!this.check) void this.load();
  }

  private async load(): Promise<void> {
    try {
      const ev = await conductor.evals();
      this.check = speedCheck(ev.reports, ev.engine);
    } catch {
      this.check = { state: 'not_run', text: 'not run for this build', title: 'no speed guard report on this server' };
    }
  }

  render() {
    if (!this.check) return html`<span class="muted">…</span>`;
    return html`<span class="speed-check" data-speed-check=${this.check.state} title=${this.check.title}>${this.check.text}</span>`;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'qse-speed-check': SpeedCheckLine;
  }
}
