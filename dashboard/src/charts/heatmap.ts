// <qse-heatmap>: the year of tokens — 53 week columns × 7 rows, Monday first, month labels,
// a five-step cobalt ramp by quantiles of the non-zero days, today outlined in signal orange.
// Every in-range cell is reachable by keyboard (roving tabindex, arrow keys); the tooltip is
// the same text for hover, tap and focus. Scrolls horizontally on phones with today in view.

import { html, nothing, svg } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import type { HeatCell, HeatGrid } from '../lib/heatmap';
import { compact, dateLong, exact, tps } from '../lib/format';
import { ChartElement } from './chart-base';

const DAY_LABELS = ['Mon', '', 'Wed', '', 'Fri', '', 'Sun'];

@customElement('qse-heatmap')
export class QseHeatmap extends ChartElement {
  @property({ attribute: false }) grid: HeatGrid | null = null;
  @property({ type: String }) metricLabel = 'tokens';
  @property({ type: Number }) cell = 12;
  @property({ type: Number }) gap = 3;
  @state() private active: string | null = null; // date of the hovered/focused cell
  @state() private focusDate: string | null = null;
  private pinned = false;

  updated(changed: Map<string, unknown>): void {
    if (changed.has('grid') && this.grid) {
      if (!this.focusDate || !this.grid.cells.some((c) => c.date === this.focusDate && c.inRange)) this.focusDate = this.grid.to;
      // Keep the current week in view on narrow screens.
      const scroller = this.querySelector<HTMLElement>('.heat-scroll');
      if (scroller) requestAnimationFrame(() => (scroller.scrollLeft = scroller.scrollWidth));
    }
  }

  private cellAt(date: string): HeatCell | undefined {
    return this.grid?.cells.find((c) => c.date === date);
  }

  private moveFocus(dCol: number, dRow: number) {
    if (!this.grid || !this.focusDate) return;
    const cur = this.cellAt(this.focusDate);
    if (!cur) return;
    const target = this.grid.cells.find((c) => c.col === cur.col + dCol && c.row === cur.row + dRow);
    if (target && target.inRange) {
      this.focusDate = target.date;
      this.active = target.date;
      this.updateComplete.then(() => this.querySelector<SVGElement>(`[data-date="${target.date}"]`)?.focus());
    }
  }

  private onKey = (e: KeyboardEvent) => {
    const map: Record<string, [number, number]> = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
    if (map[e.key]) {
      e.preventDefault();
      this.moveFocus(...map[e.key]);
    } else if (e.key === 'Escape') {
      this.active = null;
    }
  };

  render() {
    const g = this.grid;
    if (!g) return html`<div class="heat-empty"></div>`;
    // Fill the available width: cell size from the container, 10..17 px, gap 3 px.
    const left = 30;
    const gp = this.gap;
    const avail = (this.width || 800) - left;
    const c = Math.max(10, Math.min(17, Math.floor(avail / g.cols) - gp));
    const step = c + gp;
    const top = 18;
    const W = left + g.cols * step;
    const H = top + 7 * step;
    const act = this.active ? this.cellAt(this.active) : null;

    return html`
      <div class="heat">
        <div class="heat-scroll">
          <svg class="heat-svg" width=${W} height=${H} viewBox="0 0 ${W} ${H}" role="grid" aria-label="tokens per day for the last year" @keydown=${this.onKey}>
            ${g.months.map((m) => svg`<text class="axis" x=${left + m.col * step} y="11">${m.label}</text>`)}
            ${DAY_LABELS.map((l, i) => (l ? svg`<text class="axis" x="0" y=${top + i * step + c - 2}>${l}</text>` : nothing))}
            ${g.cells.map((cell) => {
              if (!cell.inRange) return nothing;
              const x = left + cell.col * step;
              const y = top + cell.row * step;
              const isFocus = cell.date === this.focusDate;
              return svg`<rect
                class="heat-cell level-${cell.level} ${cell.isToday ? 'is-today' : ''} ${this.active === cell.date ? 'is-active' : ''}"
                data-date=${cell.date}
                data-level=${cell.level}
                x=${x} y=${y} width=${c} height=${c} rx="2.5"
                tabindex=${isFocus ? 0 : -1}
                role="gridcell"
                aria-label=${`${dateLong(cell.date)}: ${exact(cell.value)} ${this.metricLabel}`}
                @mouseenter=${() => !this.pinned && (this.active = cell.date)}
                @mouseleave=${() => !this.pinned && (this.active = null)}
                @focus=${() => {
                  this.focusDate = cell.date;
                  this.active = cell.date;
                }}
                @blur=${() => !this.pinned && (this.active = null)}
                @click=${() => {
                  this.pinned = this.active === cell.date ? !this.pinned : true;
                  this.active = cell.date;
                  this.focusDate = cell.date;
                }}
              />`;
            })}
          </svg>
        </div>
        <div class="heat-foot">
          <div class="heat-legend" aria-label="less to more">
            <span>less</span>
            ${[1, 2, 3, 4, 5].map((l) => html`<span class="heat-swatch level-${l}"></span>`)}
            <span>more</span>
          </div>
          <div class="heat-tip" aria-live="polite">
            ${act
              ? html`<span class="heat-tip-date">${dateLong(act.date)}${act.isToday ? html` <em>today</em>` : nothing}</span>
                  ${act.bucket && act.bucket.requests > 0
                    ? html`<span class="heat-tip-vals">
                        <span><b class="num">${compact(act.bucket.total_tokens)}</b> tokens</span>
                        <span>in <b class="num">${compact(act.bucket.prompt_tokens - act.bucket.cached_tokens)}</b></span>
                        <span>cached <b class="num">${compact(act.bucket.cached_tokens)}</b></span>
                        <span>out <b class="num">${compact(act.bucket.completion_tokens)}</b></span>
                        <span>reasoning <b class="num">${compact(act.bucket.reasoning_tokens)}</b></span>
                        <span><b class="num">${exact(act.bucket.requests)}</b> requests</span>
                        <span>p50 <b class="num">${tps(act.bucket.decode_tps_p50)}</b></span>
                      </span>`
                    : html`<span class="heat-tip-vals"><span>no requests</span></span>`}`
              : html`<span class="heat-tip-hint">Hover, tap or use the arrow keys to read a day.</span>`}
          </div>
        </div>
      </div>
    `;
  }
}
