// <qse-timing-bar>: one request's time, to scale — queue | prefill | decode — with the numbers.
// Shared by the Dev requests table and the Performance scatter's selection.

import { html, nothing } from 'lit';
import { customElement, property } from 'lit/decorators.js';
import { ms } from '../lib/format';
import { LightElement } from '../ui/base';

@customElement('qse-timing-bar')
export class QseTimingBar extends LightElement {
  @property({ type: Number }) queue: number | null = null;
  @property({ type: Number }) prefill: number | null = null;
  @property({ type: Number }) decode: number | null = null;

  render() {
    const q = this.queue ?? 0;
    const p = this.prefill ?? 0;
    const d = this.decode ?? 0;
    const total = q + p + d;
    if (total <= 0) return html`<div class="timing timing-empty">no timing recorded</div>`;
    const pct = (v: number) => `${((v / total) * 100).toFixed(2)}%`;
    const seg = (cls: string, v: number, name: string) =>
      v > 0
        ? html`<div class="timing-seg ${cls}" style="width:${pct(v)}" title="${name} ${ms(v)}">
            ${v / total > 0.14 ? html`<span class="timing-seg-label">${name} <b class="num">${ms(v)}</b></span>` : nothing}
          </div>`
        : nothing;
    return html`<div class="timing" role="img" aria-label=${`queue ${ms(q)}, prefill ${ms(p)}, decode ${ms(d)}, total ${ms(total)}`}>
      <div class="timing-track">${seg('seg-queue', q, 'queue')}${seg('seg-prefill', p, 'prefill')}${seg('seg-decode', d, 'decode')}</div>
      <div class="timing-legend">
        <span><i class="swatch seg-queue"></i>queue <b class="num">${ms(q)}</b></span>
        <span><i class="swatch seg-prefill"></i>prefill <b class="num">${ms(p)}</b></span>
        <span><i class="swatch seg-decode"></i>decode <b class="num">${ms(d)}</b></span>
        <span class="timing-total">total <b class="num">${ms(total)}</b></span>
      </div>
    </div>`;
  }
}
