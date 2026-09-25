// <qse-bars>: stacked columns (≤ 4 series), 2 px surface gaps between segments and columns,
// rounded caps, hairline grid, per-column hover tooltip, keyboard navigation, legend.

import { html, nothing, svg } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import { ChartElement, colorVar, type SeriesColor } from './chart-base';
import { labelIndices, linear, nearestIndex, niceMax, niceTicks } from './svg';

export interface BarSeries {
  name: string;
  color: SeriesColor;
  values: number[];
}

@customElement('qse-bars')
export class QseBars extends ChartElement {
  @property({ attribute: false }) series: BarSeries[] = [];
  @property({ attribute: false }) labels: string[] = [];
  @property({ attribute: false }) format: (v: number) => string = (v) => String(Math.round(v));
  @property({ type: String }) unit = '';
  @property({ type: String, attribute: 'aria-label' }) ariaLabelText = '';
  @state() private hover: number | null = null;

  private get n(): number {
    return Math.max(0, ...this.series.map((s) => s.values.length));
  }

  render() {
    const W = this.width || 600;
    const H = this.height;
    const pad = { l: 44, r: 12, t: 12, b: 26 };
    const n = this.n;
    const totals = Array.from({ length: n }, (_, i) => this.series.reduce((a, s) => a + (s.values[i] ?? 0), 0));
    const top = niceMax(Math.max(0, ...totals));
    const y = linear([0, top], [H - pad.b, pad.t]);
    const ticks = niceTicks(0, top, 4);
    const x0 = pad.l;
    const x1 = W - pad.r;
    const slot = n ? (x1 - x0) / n : 0;
    const gap = slot >= 6 ? 2 : slot >= 3 ? 1 : 0.4;
    const bw = Math.min(24, Math.max(0.8, slot - gap));
    const xl = labelIndices(n, Math.max(2, Math.min(8, Math.floor((x1 - x0) / 90))));
    const hover = this.hover != null && this.hover < n ? this.hover : null;
    const hasData = totals.some((t) => t > 0);

    const setHover = (clientX: number) => {
      const rect = this.getBoundingClientRect();
      this.hover = nearestIndex(clientX - rect.left - slot / 2, x0, x1 - slot, n);
    };

    return html`
      <div
        class="chart chart-bars"
        tabindex="0"
        role="img"
        aria-label=${this.ariaLabelText || `${this.series.map((s) => s.name).join(', ')} per period`}
        @mousemove=${(e: MouseEvent) => setHover(e.clientX)}
        @mouseleave=${() => (this.hover = null)}
        @touchstart=${(e: TouchEvent) => e.touches[0] && setHover(e.touches[0].clientX)}
        @touchmove=${(e: TouchEvent) => e.touches[0] && setHover(e.touches[0].clientX)}
        @keydown=${(e: KeyboardEvent) => {
          if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
            e.preventDefault();
            const cur = this.hover ?? n - 1;
            this.hover = Math.max(0, Math.min(n - 1, cur + (e.key === 'ArrowLeft' ? -1 : 1)));
          } else if (e.key === 'Escape') this.hover = null;
        }}
      >
        <svg width=${W} height=${H} viewBox="0 0 ${W} ${H}" shape-rendering="crispEdges">
          ${ticks.map(
            (t) => svg`<line class="grid" x1=${x0} x2=${x1} y1=${y(t)} y2=${y(t)} />
              <text class="axis" x=${x0 - 6} y=${y(t) + 3.5} text-anchor="end">${this.format(t)}</text>`,
          )}
          ${xl.map((i) => svg`<text class="axis" x=${x0 + i * slot + bw / 2} y=${H - 8} text-anchor=${i === 0 ? 'start' : i === n - 1 ? 'end' : 'middle'}>${this.labels[i] ?? ''}</text>`)}
          ${!hasData ? svg`<text class="axis chart-nodata" x=${(x0 + x1) / 2} y=${H / 2} text-anchor="middle">no data in range</text>` : nothing}
          ${Array.from({ length: n }, (_, i) => {
            let acc = 0;
            // Thin bars snap to whole pixels so a 1 px column keeps its true hue.
            const cx = bw < 2 ? Math.round(x0 + i * slot) : x0 + i * slot + (slot - bw) / 2;
            const w = bw < 2 ? 1 : bw;
            return svg`<g class=${hover === i ? 'is-hover' : ''}>${this.series.map((s, si) => {
              const v = s.values[i] ?? 0;
              if (v <= 0) return nothing;
              const yTop = y(acc + v);
              const yBot = y(acc);
              acc += v;
              const isTop = si === this.series.length - 1 || this.series.slice(si + 1).every((t) => (t.values[i] ?? 0) <= 0);
              const h = Math.max(0.5, yBot - yTop - (si > 0 ? gap : 0));
              return svg`<rect x=${cx} y=${yTop} width=${w} height=${h} fill=${colorVar(s.color)} rx=${isTop && bw >= 4 ? 2 : 0} />`;
            })}</g>`;
          })}
          ${hover != null ? svg`<rect class="hover-band" x=${x0 + hover * slot} y=${pad.t} width=${slot} height=${H - pad.b - pad.t} />` : nothing}
        </svg>
        ${hover != null
          ? html`<div class="tip" style="left:${Math.min(Math.max(x0 + hover * slot + bw / 2, 120), W - 120)}px" role="status">
              <div class="tip-title">${this.labels[hover] ?? ''}</div>
              ${[...this.series].reverse().map((s) => html`<div class="tip-row"><span class="swatch" style="background:${colorVar(s.color)}"></span><span>${s.name}</span><span class="num">${this.format(s.values[hover] ?? 0)}${this.unit ? ' ' + this.unit : ''}</span></div>`)}
              ${this.series.length > 1 ? html`<div class="tip-row tip-row-total"><span></span><span>total</span><span class="num">${this.format(totals[hover])}${this.unit ? ' ' + this.unit : ''}</span></div>` : nothing}
            </div>`
          : nothing}
        ${this.series.length >= 2
          ? html`<div class="legend">${this.series.map((s) => html`<span class="legend-item"><span class="swatch" style="background:${colorVar(s.color)}"></span>${s.name}</span>`)}</div>`
          : nothing}
      </div>
    `;
  }
}
