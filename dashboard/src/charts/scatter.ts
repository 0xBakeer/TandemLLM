// <qse-scatter>: one dot per request — decode tok/s against prompt tokens (log x). Colour =
// cache source (validated 3-slot palette + muted for replays), shape = thinking (circle) or not
// (diamond); 2 px surface ring on every dot; hover tooltip; click selects a request.

import { html, nothing, svg } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import { ChartElement, colorVar, type SeriesColor } from './chart-base';
import { linear, log10, logTicks, niceMax, niceTicks } from './svg';

export interface ScatterPoint {
  id: number;
  x: number; // prompt tokens
  y: number; // decode tok/s
  group: string; // cache source
  diamond: boolean; // thinking off
  label: string; // tooltip title
  detail: string; // tooltip body
}

export const SCATTER_COLORS: Record<string, SeriesColor> = { none: 'cobalt', prefix: 'orange', session: 'aqua', response: 'muted' };

@customElement('qse-scatter')
export class QseScatter extends ChartElement {
  @property({ attribute: false }) points: ScatterPoint[] = [];
  @property({ attribute: false }) groups: { key: string; name: string }[] = [];
  @property({ type: String, attribute: 'aria-label' }) ariaLabelText = '';
  @property({ type: Boolean }) logY = false;
  @state() private hover: number | null = null;

  render() {
    const W = this.width || 600;
    const H = this.height;
    const pad = { l: 44, r: 12, t: 12, b: 28 };
    const xs = this.points.map((p) => p.x);
    const ys = this.points.map((p) => p.y);
    const xMin = Math.max(1, Math.min(...xs, 100));
    const xMax = Math.max(...xs, 1000) * 1.2;
    const x = log10([xMin, xMax], [pad.l, W - pad.r]);
    const yTop = niceMax(Math.max(...ys, 10));
    const yMin = Math.max(1, Math.min(...ys, 10));
    const y = this.logY ? log10([yMin / 1.5, yTop * 1.5], [H - pad.b, pad.t]) : linear([0, yTop], [H - pad.b, pad.t]);
    const xt = logTicks(xMin, xMax);
    const yt = this.logY ? logTicks(yMin / 1.5, yTop * 1.5) : niceTicks(0, yTop, 4);
    const h = this.hover != null ? this.points[this.hover] : null;

    const fmtX = (v: number) => (v >= 1000 ? `${v / 1000}k` : String(v));
    const shape = (p: ScatterPoint, r: number) =>
      p.diamond
        ? svg`<path d="M${x(p.x)},${y(p.y) - r} l${r},${r} l-${r},${r} l-${r},-${r} z" />`
        : svg`<circle cx=${x(p.x)} cy=${y(p.y)} r=${r} />`;

    return html`
      <div class="chart chart-scatter" role="img" aria-label=${this.ariaLabelText || 'decode speed against prompt length, one dot per request'} @mouseleave=${() => (this.hover = null)}>
        <svg width=${W} height=${H} viewBox="0 0 ${W} ${H}">
          ${yt.map((t) => svg`<line class="grid" x1=${pad.l} x2=${W - pad.r} y1=${y(t)} y2=${y(t)} /><text class="axis" x=${pad.l - 6} y=${y(t) + 3.5} text-anchor="end">${fmtX(t)}</text>`)}
          ${xt.map((t) => svg`<text class="axis" x=${x(t)} y=${H - 8} text-anchor="middle">${fmtX(t)}</text>`)}
          ${W >= 520 ? svg`<text class="axis axis-title" x=${W - pad.r} y=${H - 8} text-anchor="end">prompt tokens</text>` : nothing}
          <text class="axis axis-title" x=${pad.l + 4} y=${pad.t + 2} text-anchor="start">tok/s</text>
          ${this.points.length === 0 ? svg`<text class="axis chart-nodata" x=${W / 2} y=${H / 2} text-anchor="middle">no requests</text>` : nothing}
          ${this.points.map(
            (p, i) => svg`<g
                class="dot ${this.hover === i ? 'is-hover' : ''}"
                fill=${colorVar(SCATTER_COLORS[p.group] ?? 'muted')}
                @mouseenter=${() => (this.hover = i)}
                @click=${() => this.dispatchEvent(new CustomEvent('select', { detail: p.id, bubbles: true }))}
                tabindex="0"
                role="button"
                aria-label=${`${p.label}: ${p.detail}`}
                @focus=${() => (this.hover = i)}
                @keydown=${(e: KeyboardEvent) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    this.dispatchEvent(new CustomEvent('select', { detail: p.id, bubbles: true }));
                  }
                }}
              >${shape(p, this.hover === i ? 6 : 4.5)}</g>`,
          )}
        </svg>
        ${h
          ? html`<div class="tip" style="left:${Math.min(Math.max(x(h.x), 120), W - 120)}px;top:${Math.max(0, y(h.y) - 70)}px" role="status">
              <div class="tip-title">${h.label}</div>
              <div class="tip-body">${h.detail}</div>
            </div>`
          : nothing}
        <div class="legend">
          ${this.groups.map((g) => html`<span class="legend-item"><span class="swatch swatch-round" style="background:${colorVar(SCATTER_COLORS[g.key] ?? 'muted')}"></span>${g.name}</span>`)}
          <span class="legend-item"><span class="swatch swatch-round swatch-outline"></span>thinking</span>
          <span class="legend-item"><span class="swatch swatch-diamond swatch-outline"></span>no thinking</span>
        </div>
      </div>
    `;
  }
}
