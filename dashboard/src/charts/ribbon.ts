// <qse-ribbon>: time series drawn as glowing ribbons (the house motif: a blurred wide stroke
// under a crisp core), an optional p50–p90 band, hairline grid, crosshair + tooltip, keyboard
// navigation with the arrow keys, a legend for two or more series. One y axis, always.

import { html, nothing, svg } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import { ChartElement, colorVar, nextId, type SeriesColor } from './chart-base';
import { bandPath, labelIndices, linear, log10, logTicks, monotonePath, nearestIndex, niceMax, niceTicks, type Pt, type Scale } from './svg';

export interface RibbonSeries {
  name: string;
  color: SeriesColor;
  values: (number | null)[];
  /** upper curve of a band drawn from `values` up to `band` (e.g. p90 over p50) */
  band?: (number | null)[];
  bandName?: string;
  dashed?: boolean;
}

@customElement('qse-ribbon')
export class QseRibbon extends ChartElement {
  @property({ attribute: false }) series: RibbonSeries[] = [];
  @property({ attribute: false }) labels: string[] = []; // x label per index
  @property({ attribute: false }) format: (v: number) => string = (v) => String(Math.round(v));
  @property({ type: Boolean }) log = false;
  @property({ type: String }) unit = '';
  @property({ type: Number }) xLabelCount = 6;
  @property({ type: String, attribute: 'aria-label' }) ariaLabelText = '';
  @state() private hover: number | null = null;
  private glowId = nextId('rb');

  private get n(): number {
    return Math.max(0, ...this.series.map((s) => s.values.length));
  }

  private setHoverFromEvent(e: MouseEvent | TouchEvent, x0: number, x1: number) {
    const rect = this.getBoundingClientRect();
    const cx = 'touches' in e ? e.touches[0]?.clientX : e.clientX;
    if (cx == null) return;
    this.hover = nearestIndex(cx - rect.left, x0, x1, this.n);
  }

  private onKey = (e: KeyboardEvent) => {
    if (this.n === 0) return;
    if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
      e.preventDefault();
      const cur = this.hover ?? this.n - 1;
      this.hover = Math.max(0, Math.min(this.n - 1, cur + (e.key === 'ArrowLeft' ? -1 : 1)));
    } else if (e.key === 'Home') this.hover = 0;
    else if (e.key === 'End') this.hover = this.n - 1;
    else if (e.key === 'Escape') this.hover = null;
  };

  render() {
    const W = this.width || 600;
    const H = this.height;
    const pad = { l: 44, r: 12, t: 12, b: 26 };
    const n = this.n;
    const x0 = pad.l;
    const x1 = W - pad.r;
    const all: number[] = [];
    for (const s of this.series) {
      for (const v of s.values) if (v != null && Number.isFinite(v)) all.push(v);
      if (s.band) for (const v of s.band) if (v != null && Number.isFinite(v)) all.push(v);
    }
    const hasData = n > 0 && all.length > 0;
    const yMaxRaw = hasData ? Math.max(...all) : 1;
    const yMinRaw = hasData ? Math.min(...all) : 0;
    let y: Scale;
    let ticks: number[];
    if (this.log) {
      const lo = Math.max(1e-3, yMinRaw / 1.5);
      const hi = yMaxRaw * 1.3;
      y = log10([lo, hi], [H - pad.b, pad.t]);
      ticks = logTicks(lo, hi);
    } else {
      const top = niceMax(yMaxRaw);
      y = linear([0, top], [H - pad.b, pad.t]);
      ticks = niceTicks(0, top, 4);
    }
    const x = linear([0, Math.max(1, n - 1)], [x0, x1]);
    const pts = (vals: (number | null)[]): (Pt | null)[] => vals.map((v, i) => (v == null || !Number.isFinite(v) ? null : { x: x(i), y: y(v) }));
    const xl = labelIndices(n, Math.min(this.xLabelCount, Math.max(2, Math.floor((x1 - x0) / 90))));
    const hover = this.hover != null && this.hover < n ? this.hover : null;
    const legend = this.series.length >= 2 || this.series.some((s) => s.band);

    return html`
      <div
        class="chart chart-ribbon"
        tabindex="0"
        role="img"
        aria-label=${this.ariaLabelText || `${this.series.map((s) => s.name).join(', ')} over time`}
        @mousemove=${(e: MouseEvent) => this.setHoverFromEvent(e, x0, x1)}
        @mouseleave=${() => (this.hover = null)}
        @touchstart=${(e: TouchEvent) => this.setHoverFromEvent(e, x0, x1)}
        @touchmove=${(e: TouchEvent) => this.setHoverFromEvent(e, x0, x1)}
        @keydown=${this.onKey}
      >
        <svg width=${W} height=${H} viewBox="0 0 ${W} ${H}">
          <defs>
            <filter id="${this.glowId}-glow" x="-10%" y="-40%" width="120%" height="180%">
              <feGaussianBlur stdDeviation="5" />
            </filter>
          </defs>
          ${ticks.map(
            (t) => svg`<line class="grid" x1=${x0} x2=${x1} y1=${y(t)} y2=${y(t)} />
              <text class="axis" x=${x0 - 6} y=${y(t) + 3.5} text-anchor="end">${this.format(t)}</text>`,
          )}
          ${xl.map((i) => svg`<text class="axis" x=${x(i)} y=${H - 8} text-anchor=${i === 0 ? 'start' : i === n - 1 ? 'end' : 'middle'}>${this.labels[i] ?? ''}</text>`)}
          ${!hasData ? svg`<text class="axis chart-nodata" x=${(x0 + x1) / 2} y=${H / 2} text-anchor="middle">no data in range</text>` : nothing}
          ${this.series.map((s) => {
            const p = pts(s.values);
            const d = monotonePath(p);
            const col = colorVar(s.color);
            return svg`
              ${s.band ? svg`<path class="band" d=${bandPath(pts(s.band), p)} fill=${col} />` : nothing}
              <path d=${d} fill="none" stroke=${col} stroke-width="9" stroke-linecap="round" stroke-linejoin="round" opacity="0.28" filter="url(#${this.glowId}-glow)" />
              <path d=${d} fill="none" stroke=${col} stroke-width="2" stroke-linecap="round" stroke-linejoin="round" stroke-dasharray=${s.dashed ? '4 4' : nothing} />
              ${s.band ? svg`<path d=${monotonePath(pts(s.band))} fill="none" stroke=${col} stroke-width="1.2" opacity="0.7" stroke-dasharray="3 3" />` : nothing}
            `;
          })}
          ${hover != null
            ? svg`
              <line class="crosshair" x1=${x(hover)} x2=${x(hover)} y1=${pad.t} y2=${H - pad.b} />
              ${this.series.map((s) => {
                const v = s.values[hover];
                if (v == null) return nothing;
                return svg`<circle cx=${x(hover)} cy=${y(v)} r="4.5" fill=${colorVar(s.color)} class="ring" />`;
              })}`
            : nothing}
        </svg>
        ${hover != null
          ? html`<div class="tip" style="left:${Math.min(Math.max(x(hover), 120), W - 120)}px" role="status">
              <div class="tip-title">${this.labels[hover] ?? ''}</div>
              ${this.series.map(
                (s) => html`<div class="tip-row"><span class="swatch" style="background:${colorVar(s.color)}"></span><span>${s.name}</span><span class="num">${s.values[hover] == null ? '—' : this.format(s.values[hover] as number) + (this.unit ? ' ' + this.unit : '')}</span></div>
                  ${s.band ? html`<div class="tip-row tip-row-sub"><span class="swatch swatch-dash" style="border-color:${colorVar(s.color)}"></span><span>${s.bandName ?? 'p90'}</span><span class="num">${s.band[hover] == null ? '—' : this.format(s.band[hover] as number) + (this.unit ? ' ' + this.unit : '')}</span></div>` : nothing}`,
              )}
            </div>`
          : nothing}
        ${legend
          ? html`<div class="legend">
              ${this.series.map((s) => html`<span class="legend-item"><span class="swatch" style="background:${colorVar(s.color)}"></span>${s.name}</span>${s.band ? html`<span class="legend-item"><span class="swatch swatch-dash" style="border-color:${colorVar(s.color)}"></span>${s.bandName ?? 'p90'}</span>` : nothing}`)}
            </div>`
          : nothing}
      </div>
    `;
  }
}
