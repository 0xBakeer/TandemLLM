// <qse-stack>: decode tok/s by backend, stacked, over a fixed time window that scrolls left (the
// parallel view). One y axis (tok/s). The top outline glows (the house motif); each layer is a
// translucent fill under its own crisp edge. A sample is one second of the server's series;
// a gap longer than 3 s (nothing in flight) drops to zero instead of bridging.

import { html, nothing, svg } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import { ChartElement, colorVar, nextId, type SeriesColor } from './chart-base';
import { linear, niceMax, niceTicks } from './svg';
import { stack, type Sample } from '../lib/parallel';

@customElement('qse-stack')
export class QseStack extends ChartElement {
  @property({ attribute: false }) series: Sample[] = [];
  @property({ attribute: false }) order: string[] = [];
  @property({ attribute: false }) color: (id: string) => SeriesColor = () => 'cobalt';
  @property({ attribute: false }) label: (id: string) => string = (id) => id;
  /** the window in seconds, ending at the newest sample (or `now`) */
  @property({ type: Number }) span = 300;
  @property({ type: Number }) now = 0;
  @state() private hover: number | null = null;
  private glowId = nextId('st');

  private onMove(e: MouseEvent | TouchEvent, xs: number[]) {
    const rect = this.getBoundingClientRect();
    const cx = 'touches' in e ? e.touches[0]?.clientX : e.clientX;
    if (cx == null || !xs.length) return;
    const px = cx - rect.left;
    let best = 0;
    for (let i = 1; i < xs.length; i++) if (Math.abs(xs[i] - px) < Math.abs(xs[best] - px)) best = i;
    this.hover = best;
  }

  render() {
    const W = this.width || 600;
    const H = this.height;
    const pad = { l: 46, r: 10, t: 10, b: 24 };
    const x0 = pad.l;
    const x1 = W - pad.r;
    const end = this.now || (this.series.length ? this.series[this.series.length - 1].t : Date.now() / 1000);
    const start = end - this.span;
    const ser = this.series.filter((s) => s.t >= start - 2);
    // a gap of more than 3 s gets two zero samples, so the area drops instead of bridging
    const pts: Sample[] = [];
    for (let i = 0; i < ser.length; i++) {
      const s = ser[i];
      const prev = pts[pts.length - 1];
      if (prev && s.t - prev.t > 3) {
        pts.push({ t: prev.t + 1, by: {}, in_flight: 0 }, { t: s.t - 1, by: {}, in_flight: 0 });
      }
      pts.push(s);
    }
    const layers = stack(pts, this.order);
    const tops = pts.map((_, i) => (layers.length ? layers[layers.length - 1].bands[i][1] : 0));
    const top = niceMax(Math.max(10, ...tops));
    const y = linear([0, top], [H - pad.b, pad.t]);
    const x = linear([start, end], [x0, x1]);
    const ticks = niceTicks(0, top, 4);
    const xs = pts.map((s) => x(s.t));
    const area = (bands: [number, number][]) => {
      if (!bands.length) return '';
      let d = `M${xs[0].toFixed(1)},${y(bands[0][1]).toFixed(1)}`;
      for (let i = 1; i < bands.length; i++) d += `L${xs[i].toFixed(1)},${y(bands[i][1]).toFixed(1)}`;
      for (let i = bands.length - 1; i >= 0; i--) d += `L${xs[i].toFixed(1)},${y(bands[i][0]).toFixed(1)}`;
      return d + 'Z';
    };
    const edge = (vals: number[]) => vals.map((v, i) => `${i ? 'L' : 'M'}${xs[i].toFixed(1)},${y(v).toFixed(1)}`).join('');
    const hover = this.hover != null && this.hover < pts.length ? this.hover : null;
    const step = this.span > 120 ? 60 : 30;
    const xlab = Array.from({ length: Math.floor(this.span / step) + 1 }, (_, i) => i * step);
    return html`<div
      class="chart chart-stack"
      tabindex="0"
      role="img"
      aria-label="decode tokens per second by backend, stacked, the last ${this.span} seconds"
      @mousemove=${(e: MouseEvent) => this.onMove(e, xs)}
      @mouseleave=${() => (this.hover = null)}
      @touchstart=${(e: TouchEvent) => this.onMove(e, xs)}
      @touchmove=${(e: TouchEvent) => this.onMove(e, xs)}
    >
      <svg width=${W} height=${H} viewBox="0 0 ${W} ${H}">
        <defs>
          <filter id="${this.glowId}-glow" x="-5%" y="-40%" width="110%" height="180%"><feGaussianBlur stdDeviation="4" /></filter>
        </defs>
        ${ticks.map((t) => svg`<line class="grid" x1=${x0} x2=${x1} y1=${y(t)} y2=${y(t)} /><text class="axis" x=${x0 - 6} y=${y(t) + 3.5} text-anchor="end">${t >= 1000 ? `${(t / 1000).toFixed(t % 1000 ? 1 : 0)}k` : t}</text>`)}
        ${xlab.map((s) => svg`<text class="axis" x=${x(end - s)} y=${H - 7} text-anchor=${s === 0 ? 'end' : s === this.span ? 'start' : 'middle'}>${s === 0 ? 'now' : s % 60 ? `−${s} s` : `−${s / 60} min`}</text>`)}
        ${!pts.length ? svg`<text class="axis chart-nodata" x=${(x0 + x1) / 2} y=${H / 2} text-anchor="middle">nothing decoded in this window</text>` : nothing}
        ${layers.map((L) => {
          const col = colorVar(this.color(L.id));
          return svg`<path class="stack-area" data-layer=${L.id} d=${area(L.bands)} fill=${col} /><path d=${edge(L.bands.map((b) => b[1]))} fill="none" stroke=${col} stroke-width="1.4" stroke-linejoin="round" />`;
        })}
        ${pts.length ? svg`<path d=${edge(tops)} fill="none" stroke="var(--ink)" stroke-width="7" opacity="0.10" filter="url(#${this.glowId}-glow)" /><path d=${edge(tops)} fill="none" stroke="var(--ink)" stroke-width="1.1" opacity="0.55" stroke-linejoin="round" />` : nothing}
        ${hover != null ? svg`<line class="crosshair" x1=${xs[hover]} x2=${xs[hover]} y1=${pad.t} y2=${H - pad.b} />` : nothing}
      </svg>
      ${hover != null
        ? html`<div class="tip" style="left:${Math.min(Math.max(xs[hover], 110), W - 110)}px" role="status">
            <div class="tip-title">${Math.max(0, Math.round(end - pts[hover].t))} s ago · ${Math.round(tops[hover])} tok/s</div>
            ${[...this.order].reverse().map((id) => {
              const v = pts[hover].by[id];
              return v ? html`<div class="tip-row"><span class="swatch" style="background:${colorVar(this.color(id))}"></span><span>${this.label(id)}</span><span class="num">${Math.round(v)} tok/s</span></div>` : nothing;
            })}
          </div>`
        : nothing}
    </div>`;
  }
}
