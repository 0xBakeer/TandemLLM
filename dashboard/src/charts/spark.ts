// <qse-spark>: a sparkline for a stat figure (VIS-23). No axes, no grid: the number above it is
// the axis. Two forms: a `ribbon` (the house glow under a crisp line) for a stream such as decode
// tok/s, and `dots` for an event series such as prefills, where a line between two points would
// claim a rate that never existed. Null values are gaps. The newest point is signal orange (the
// house rule: orange is now). Hover / touch / arrow keys show the value and how long ago.

import { html, nothing, svg } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import { ChartElement, colorVar, nextId, type SeriesColor } from './chart-base';
import { linear, monotonePath, nearestIndex, type Pt } from './svg';

@customElement('qse-spark')
export class QseSpark extends ChartElement {
  @property({ attribute: false }) values: (number | null)[] = [];
  @property({ type: String }) mode: 'ribbon' | 'dots' = 'ribbon';
  @property({ type: String }) color: SeriesColor = 'cobalt';
  @property({ attribute: false }) format: (v: number) => string = (v) => String(Math.round(v));
  @property({ type: String }) unit = '';
  /** seconds per slot, for the "N s ago" in the tip */
  @property({ type: Number }) step = 1;
  @property({ type: String, attribute: 'aria-label' }) ariaLabelText = '';
  @state() private hover: number | null = null;
  private glowId = nextId('sp');

  private setHover(e: MouseEvent | TouchEvent) {
    const rect = this.getBoundingClientRect();
    const cx = 'touches' in e ? e.touches[0]?.clientX : e.clientX;
    if (cx == null) return;
    this.hover = nearestIndex(cx - rect.left, 0, rect.width, this.values.length);
  }

  private onKey = (e: KeyboardEvent) => {
    const n = this.values.length;
    if (!n) return;
    if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
      e.preventDefault();
      const cur = this.hover ?? n - 1;
      this.hover = Math.max(0, Math.min(n - 1, cur + (e.key === 'ArrowLeft' ? -1 : 1)));
    } else if (e.key === 'Home') this.hover = 0;
    else if (e.key === 'End') this.hover = n - 1;
    else if (e.key === 'Escape') this.hover = null;
  };

  render() {
    const W = this.width || 300;
    const H = this.height || 44;
    const n = this.values.length;
    const vals = this.values.filter((v): v is number => v != null && Number.isFinite(v));
    const max = vals.length ? Math.max(...vals) : 1;
    const y = linear([0, max > 0 ? max * 1.1 : 1], [H - 3, 4]);
    const x = linear([0, Math.max(1, n - 1)], [3, W - 3]);
    const pts: (Pt | null)[] = this.values.map((v, i) => (v == null || !Number.isFinite(v) ? null : { x: x(i), y: y(v) }));
    const col = colorVar(this.color);
    let last = -1;
    for (let i = n - 1; i >= 0; i--) {
      if (pts[i]) {
        last = i;
        break;
      }
    }
    const hover = this.hover != null && this.hover < n ? this.hover : null;
    const hv = hover != null ? this.values[hover] : null;
    const ago = hover != null ? (n - 1 - hover) * this.step : 0;
    return html`<div
      class="spark spark-${this.mode}"
      tabindex="0"
      role="img"
      aria-label=${this.ariaLabelText || `last ${Math.round(n * this.step)} seconds`}
      @mousemove=${this.setHover}
      @mouseleave=${() => (this.hover = null)}
      @touchstart=${this.setHover}
      @touchmove=${this.setHover}
      @keydown=${this.onKey}
    >
      <svg width=${W} height=${H} viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
        <defs>
          <filter id="${this.glowId}-glow" x="-5%" y="-60%" width="110%" height="220%"><feGaussianBlur stdDeviation="3.5" /></filter>
        </defs>
        ${!vals.length ? svg`<line class="spark-base" x1="3" x2=${W - 3} y1=${H - 3} y2=${H - 3} />` : nothing}
        ${this.mode === 'ribbon' && vals.length
          ? svg`
            <path d=${monotonePath(pts)} fill="none" stroke=${col} stroke-width="7" stroke-linecap="round" stroke-linejoin="round" opacity="0.26" filter="url(#${this.glowId}-glow)" />
            <path d=${monotonePath(pts)} fill="none" stroke=${col} stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" />`
          : nothing}
        ${this.mode === 'dots' ? pts.map((p) => (p ? svg`<circle cx=${p.x} cy=${p.y} r="2.6" fill=${col} opacity="0.85" />` : nothing)) : nothing}
        ${last >= 0 ? svg`<circle class="spark-now" cx=${pts[last]!.x} cy=${pts[last]!.y} r=${this.mode === 'dots' ? 3.4 : 3} />` : nothing}
        ${hover != null && pts[hover] ? svg`<line class="crosshair" x1=${pts[hover]!.x} x2=${pts[hover]!.x} y1="2" y2=${H - 2} /><circle class="ring" cx=${pts[hover]!.x} cy=${pts[hover]!.y} r="4" fill=${col} />` : nothing}
      </svg>
      ${hover != null
        ? html`<div class="tip tip-spark" style="left:${Math.min(Math.max(x(hover), 70), W - 70)}px" role="status">
            <span class="num">${hv == null ? 'no sample' : this.format(hv) + (this.unit ? ' ' + this.unit : '')}</span><span class="muted">${ago === 0 ? 'now' : `${ago} s ago`}</span>
          </div>`
        : nothing}
    </div>`;
  }
}
