// <qse-memory-bar>: the sticky memory bar of every setup step (D5, VIS-78). One track scaled to the
// box: the selection's parts stacked by category, the conversations' typical KV solid and the worst
// case as an outline, a line where the engine stops (device - floor) with the kept-free room
// hatched beyond it, an orange tick at what the box uses right now (live MemAvailable), and the
// overflow in red with a sentence when it does not fit. Hover or focus a part for its figure. On a
// phone the bar is one compact line; the legend opens below it. Every line has a fixed height, so
// a recompute never moves the page.

import { html, nothing } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import type { SetupEstimate, SetupModel } from '../api/types';
import { LightElement } from '../ui/base';
import { categoryOf, overBy, type PartCategory } from './estimate';
import { S } from './strings';

const CATS: PartCategory[] = ['model', 'helpers', 'caches', 'engine', 'system', 'chats'];

@customElement('qse-memory-bar')
export class QseMemoryBar extends LightElement {
  @property({ attribute: false }) model: SetupModel | null = null;
  @property({ attribute: false }) estimate: SetupEstimate | null = null;
  /** live MemAvailable, GiB */
  @property({ attribute: false }) available: number | null = null;
  /** the server's estimate did not answer: the page's own copy is shown */
  @property({ type: Boolean }) local = false;
  @property({ attribute: false }) overText: string | null = null;
  @state() private open = false;
  @state() private focusPart: string | null = null;

  render() {
    const e = this.estimate;
    const m = this.model;
    if (!e || !m) return html`<section class="membar is-empty" aria-label=${S.memTitle}><div class="membar-line"><span class="membar-title">${S.memTitle}</span></div><div class="membar-track"></div></section>`;
    const budget = e.budget_gb ?? e.device_gb - e.floor_gb;
    const over = overBy(e);
    const kv = e.parts.find((p) => p.kind === 'kv');
    const scale = Math.max(e.device_gb, e.total_gb);
    const pct = (x: number) => `${((x / scale) * 100).toFixed(3)}%`;
    let x = 0;
    const segs = e.parts.map((p) => {
      const s = { p, left: x, cat: categoryOf(m, p.id) };
      x += p.gb;
      return s;
    });
    const used = this.available != null ? Math.max(0, e.device_gb - this.available) : null;
    const sums = new Map<PartCategory, number>();
    for (const s of segs) sums.set(s.cat, (sums.get(s.cat) ?? 0) + s.p.gb);
    const focus = segs.find((s) => s.p.id === this.focusPart);
    const detail = focus ? `${focus.p.label}: ${focus.p.gb.toFixed(2)} GiB${focus.p.worst_gb != null && focus.p.worst_gb !== focus.p.gb ? ` (${S.memoryUpTo(focus.p.worst_gb)})` : ''}` : null;
    const summary = `${S.memTitle}: ${S.memFits(e.total_gb, budget)}. ${over > 0 ? this.overText ?? '' : S.memLeft(budget - e.total_gb)}`;

    return html`<section class="membar ${over > 0 ? 'is-over' : ''} ${this.open ? 'is-open' : ''}" aria-label=${S.memTitle} data-fits=${e.fits} data-total=${e.total_gb.toFixed(2)}>
      <div class="membar-line">
        <span class="membar-title">${S.memTitle}</span>
        <span class="membar-fig num" data-membar-total><b>${e.total_gb.toFixed(1)}</b><span class="membar-of"> / ${budget.toFixed(1)} GiB</span></span>
        <span class="membar-say ${over > 0 ? 'is-over' : ''}" data-membar-say role=${over > 0 ? 'alert' : nothing}>${over > 0 ? this.overText ?? S.overSentence(over, null) : S.memLeft(budget - e.total_gb)}</span>
        <span class="membar-now num">${used != null ? html`<i class="membar-now-dot" aria-hidden="true"></i>${S.memNow(used)}` : nothing}</span>
        <span class="membar-device num">${S.memDevice(e.device_gb)}</span>
        <button class="btn btn-ghost btn-sm membar-toggle" aria-expanded=${this.open} aria-controls="membar-more" aria-label=${this.open ? S.memCollapse : S.memExpand} @click=${() => (this.open = !this.open)}>
          <svg class="icon" viewBox="0 0 20 20" aria-hidden="true"><path d="m5 8 5 5 5-5" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" /></svg>
        </button>
      </div>
      <div class="membar-track" role="img" aria-label=${summary}>
        <span class="membar-free" style="left:${pct(budget)};width:${pct(Math.max(0, e.device_gb - budget))}" title=${`${S.memFloor}: ${e.floor_gb} GiB`}><span class="membar-floor-label">${S.memFloor} ${e.floor_gb} GiB</span></span>
        ${segs.map(
          (s) => html`<span
            class="membar-seg cat-${s.cat} ${s.p.kind === 'kv' ? 'is-kv' : ''} ${this.focusPart === s.p.id ? 'is-focus' : ''}"
            style="left:${pct(s.left)};width:${pct(s.p.gb)}"
            data-part=${s.p.id}
            tabindex="0"
            aria-label=${`${s.p.label} ${s.p.gb.toFixed(1)} GiB`}
            @mouseenter=${() => (this.focusPart = s.p.id)}
            @mouseleave=${() => (this.focusPart = null)}
            @focus=${() => (this.focusPart = s.p.id)}
            @blur=${() => (this.focusPart = null)}
          ></span>`,
        )}
        ${kv && kv.worst_gb != null && kv.worst_gb > kv.gb ? html`<span class="membar-worst" style="left:${pct(e.total_gb)};width:${pct(Math.min(kv.worst_gb - kv.gb, scale - e.total_gb))}" title=${`${S.memWorst}: ${kv.worst_gb.toFixed(1)} GiB`}></span>` : nothing}
        ${over > 0 ? html`<span class="membar-over" style="left:${pct(budget)};width:${pct(e.total_gb - budget)}"></span>` : nothing}
        <span class="membar-floor" style="left:${pct(budget)}"></span>
        ${used != null ? html`<span class="membar-nowtick" style="left:${pct(Math.min(used, scale))}" title=${S.memNow(used)}></span>` : nothing}
      </div>
      <div class="membar-more" id="membar-more">
        <ul class="membar-legend">
          ${CATS.filter((c) => sums.has(c)).map((c) => html`<li class="cat-${c}"><i class="membar-sw" aria-hidden="true"></i>${S.memCat[c]} <span class="num">${(sums.get(c) ?? 0).toFixed(1)}</span></li>`)}
          <li class="cat-free"><i class="membar-sw" aria-hidden="true"></i>${S.memFloor} <span class="num">${e.floor_gb}</span></li>
        </ul>
        <p class="membar-detail" aria-live="polite">${detail ?? (this.local ? S.memLocal : e.notes[0] ?? ' ')}</p>
      </div>
    </section>`;
  }
}
