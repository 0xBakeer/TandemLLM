// WIP (wip/upstream-import, F5) — <qse-hw-rail>: the always-on hardware strip, ported from
// TandemDash-Fork. Fixed drawer on the right edge, visible on EVERY view — hardware belongs in
// the shell, not buried one view deep. Collapsed by default on narrow screens, user-remembered
// on wide ones; when open on a desktop it shifts the content instead of covering it.
//
// Data is the EXISTING `/v1/dashboard/system` route (hosts[0].cpu / memory + the top-level gpu
// block) — nothing new server-side. 0.9.1 samples on demand, so the 2-min trend history is a
// CLIENT-side ring: one point per 2 s poll, 60 points, kept while the rail is open.
//
// Mobile house rule (the user's phone dies on GPU-heavy pages): ≤760px the rail never opens by
// default, opens as a plain overlay when asked, and no backdrop-filter anywhere.
//
// Disable: revert this commit (one shell tag + one import + this file + one CSS file).

import { html } from 'lit';
import { customElement, state } from 'lit/decorators.js';
import { request } from '../api/client';
import type { RibbonSeries } from '../charts/ribbon';
import { bytes, fixed } from '../lib/format';
import { icon } from '../ui/bits';
import { LightElement } from '../ui/base';

// <qse-ribbon> is registered by main.ts, like every chart the views use.

const LS_OPEN = 'qse.hw-rail';
const LS_RING = 'qse.hw-rail.ring';
const RING = 60; // 60 x 2 s = the fork's 2-minute window
const STEP_S = 2;
const PEEK_S = 10; // while the rail is closed, keep sampling slower — a peek opens on history
const STALE_S = 120; // older than the whole window: nothing worth restoring

// Module-level: survives the component re-creating (a reload, a shell remount). People PEEK at
// this rail — hide it, come back ten minutes later, open it — and a rail that redraws itself
// empty every time is a rail that lies about the machine. Persisted, so a reload keeps history.
let ring: RingPoint[] = [];
let lastSample = 0; // epoch s of the newest ring point (0 = empty)
let lastSaved = 0;

function restoreRing(): void {
  try {
    const raw = localStorage.getItem(LS_RING);
    if (!raw) return;
    const v = JSON.parse(raw) as { t: number; pts: RingPoint[] };
    const age = Date.now() / 1000 - v.t;
    // every point is validated, not just the shape: a pre-B1 save whose elements
    // are null or of the old form would pass a length check and then brick every
    // render from localStorage — a shape the map in renderBody cannot survive.
    // !(age >= 0) rejects a legacy save without t: NaN sails past the stale guard.
    if (!Array.isArray(v.pts)) return;
    const pts = v.pts.filter((x) => x && typeof x === 'object');
    if (!pts.length || !(age >= 0) || age > STALE_S + 60) return;
    // null-fill the gap so the ribbon shows the hole rather than pretending continuity;
    // the holes get only the space the real points leave over — fill-then-slice used to
    // let a long away-time evict every sample the restore was meant to keep
    const missing = Math.max(0, Math.min(RING - pts.length, Math.floor(age / STEP_S)));
    const hole: RingPoint = { cpu_temp_c: null, gpu_temp_c: null, gpu_power_w: null, gpu_clk_mhz: null, ram_used_gib: null };
    ring = [...pts, ...Array.from({ length: missing }, () => ({ ...hole }))].slice(-RING);
    lastSample = v.t;
  } catch { /* private mode, quota, old shape — start empty */ }
}
function saveRing(): void {
  const now = Date.now() / 1000;
  if (now - lastSaved < 10) return;
  lastSaved = now;
  try { localStorage.setItem(LS_RING, JSON.stringify({ t: lastSample, pts: ring })); } catch { /* fine */ }
}
restoreRing();

interface SystemHost {
  cpu?: { count?: number; utilization?: number | null; load?: (number | null)[]; temperature_c?: number | null } | null;
  memory?: { total_bytes?: number | null; used_bytes?: number | null; available_bytes?: number | null;
             swap_total_bytes?: number | null; swap_used_bytes?: number | null } | null;
  fs?: { path?: string; total_bytes: number; free_bytes: number }[] | null;
  net?: { name: string; rx_bps: number | null; tx_bps: number | null }[] | null;
}
interface SystemGpu {
  name?: string | null; temperature_c?: number | null; power_w?: number | null;
  sm_clock_mhz?: number | null; utilization?: number | null;
}
interface SystemBlock { hosts?: SystemHost[]; gpu?: SystemGpu | null }
interface RingPoint {
  /** epoch seconds of the sample (B1): real timestamps, because the closed rail samples every
   *  PEEK_S=10 s, so `(n-1-i)*STEP_S` ages lie after minutes closed. Legacy restored points
   *  without `t` fall back to the relative age. */
  t?: number;
  cpu_temp_c: number | null; gpu_temp_c: number | null; gpu_power_w: number | null;
  gpu_clk_mhz: number | null; ram_used_gib: number | null;
}
/** a wire rate in B/s, spoken short: 9070 -> '9.1K' — the LAN tile carries both directions. */
function rate(bps: number | null | undefined): string {
  if (bps == null) return '–';
  if (bps >= 1e9) return `${(bps / 1e9).toFixed(1)}G`;
  if (bps >= 1e6) return `${(bps / 1e6).toFixed(1)}M`;
  if (bps >= 1e3) return `${(bps / 1e3).toFixed(1)}K`;
  return `${Math.round(bps)}`;
}

/** the freshest non-null sample of a ring — the 'now' figure shown next to a legend swatch */
function lastOf(vals: (number | null)[]): number | null {
  for (let i = vals.length - 1; i >= 0; i--) if (vals[i] != null) return vals[i];
  return null;
}

/** B1: a point's age in seconds — real when the point carries a timestamp, relative (the old
 *  2 s-per-point assumption) only for legacy points without one. Never negative. */
export function pointAgeS(p: RingPoint | undefined, i: number, n: number, nowS: number): number {
  if (p?.t != null) return Math.max(0, Math.round(nowS - p.t));
  return Math.max(0, (n - 1 - i) * STEP_S);
}
export function ageLabel(age: number): string {
  return age === 0 ? 'now' : age >= 60 ? `${Math.round(age / 60)}m` : `${age}s`;
}

@customElement('qse-hw-rail')
export class QseHwRail extends LightElement {
  @state() private sys: SystemBlock | null = null;
  @state() private ring: RingPoint[] = ring;
  @state() private open = (() => {
    try {
      const v = localStorage.getItem(LS_OPEN);
      // a desktop 'open' must not reopen the rail as a phone overlay on first load:
      // the stored state only wins where the rail is a sidebar, not an overlay
      if (v != null && window.matchMedia('(min-width: 761px)').matches) return v === '1';
    } catch { /* private mode */ }
    return window.matchMedia('(min-width: 1201px)').matches; // desktop default: open
  })();

  private timer: ReturnType<typeof setInterval> | null = null;

  connectedCallback(): void {
    super.connectedCallback();
    void this.poll();
    this.syncTimer();
    document.addEventListener('visibilitychange', this.onVis);
  }
  disconnectedCallback(): void {
    document.removeEventListener('visibilitychange', this.onVis);
    this.stopTimer();
    document.body.classList.remove('hw-rail-open');
    super.disconnectedCallback();
  }
  private onVis = (): void => {
    this.syncTimer();
    if (document.visibilityState === 'visible') void this.poll();
  };
  private syncTimer(): void {
    this.stopTimer();
    // visible tab = always sampling (open fast, closed slow) — the peek is the point
    if (document.visibilityState === 'visible') {
      this.timer = setInterval(() => void this.poll(), (this.open ? STEP_S : PEEK_S) * 1000);
    }
  }
  private stopTimer(): void {
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
  }
  // B2: connectedCallback, on-visibility and toggle can all fire a poll into the same
  // in-flight request — dedup like poll.ts does, so two ring points cannot land sub-2 s apart.
  private inflight: Promise<void> | null = null;
  private poll(): Promise<void> {
    if (this.inflight) return this.inflight;
    this.inflight = this.runPoll().catch(() => undefined).finally(() => (this.inflight = null));
    return this.inflight;
  }
  private async runPoll(): Promise<void> {
    try {
      const s = await request<SystemBlock>('/v1/dashboard/system');
      const host = s.hosts?.[0];
      const gib = (v: number | null | undefined): number | null => (v == null ? null : v / 2 ** 30);
      ring = [...ring, {
        t: Math.floor(Date.now() / 1000),
        cpu_temp_c: host?.cpu?.temperature_c ?? null,
        gpu_temp_c: s.gpu?.temperature_c ?? null,
        gpu_power_w: s.gpu?.power_w ?? null,
        gpu_clk_mhz: s.gpu?.sm_clock_mhz ?? null,
        ram_used_gib: host?.memory?.total_bytes != null
          ? gib(host.memory.total_bytes - (host.memory.available_bytes ?? 0))
          : null,
      }].slice(-RING);
      lastSample = Date.now() / 1000;
      saveRing();
      // a remount owns the state; the ring append above still stands — history
      // is the whole point of the persistence, a dead instance may keep feeding it
      if (this.isConnected) this.sys = s;
      if (this.isConnected) this.ring = ring;
    } catch { /* keep the last snapshot */ }
  }
  private toggle(): void {
    this.open = !this.open;
    try { localStorage.setItem(LS_OPEN, this.open ? '1' : '0'); } catch { /* fine */ }
    this.syncTimer();
    if (this.open) void this.poll();
  }

  updated(): void {
    document.body.classList.toggle('hw-rail-open', this.open);
  }

  render() {
    return html`
      <button type="button" class="hw-tab ${this.open ? 'is-open' : ''}"
        title=${this.open ? 'Hide hardware rail' : 'Show hardware rail'}
        aria-expanded=${this.open ? 'true' : 'false'} aria-label="Hardware rail"
        @click=${() => this.toggle()}>
        ${icon('system')}
      </button>
      <aside class="hw-rail ${this.open ? '' : 'is-hidden'}" aria-label="Hardware">
        ${this.renderBody()}
      </aside>`;
  }

  private tile(label: string, value: string, warn = false, cls = '') {
    return html`<div class="hw-tile ${warn ? 'is-warn' : ''} ${cls}">
      <div class="hw-tile-label">${label}</div>
      <div class="hw-tile-value num">${value}</div>
    </div>`;
  }
  /** a 2-min trend at 2 s/point — a real ribbon: y scale with values, x ticks (now … 2m).
   *  B1: ages come from the points' real timestamps, not position, so minutes of closed-rail
   *  peek sampling label honestly. */
  private chart(label: string, series: RibbonSeries[], format: (v: number) => string, unit: string, autoY = false) {
    const n = series[0]?.values.length ?? 0;
    const nowS = Date.now() / 1000;
    const labels = this.ring.slice(-n).map((p, i) => ageLabel(pointAgeS(p, i, n, nowS)));
    return html`<div class="hw-chart">
      <div class="hw-chart-label">${label}</div>
      <qse-ribbon .series=${series} .labels=${labels} .format=${format}
        unit=${unit} .height=${84} .xLabelCount=${3} .autoY=${autoY}
        aria-label=${label}></qse-ribbon>
    </div>`;
  }
  private line(name: string, color: RibbonSeries['color'], values: (number | null)[], legendValue?: number | null): RibbonSeries {
    return { name, color, values, legendValue };
  }

  private head() {
    return html`<div class="hw-rail-head">
      <span>Hardware${this.sys?.hosts?.[0]?.cpu?.count ? ` · ${this.sys.hosts[0]!.cpu!.count} cores` : ''}</span>
      <button type="button" class="btn btn-ghost btn-icon hw-close" aria-label="Hide hardware rail"
        title="Hide hardware rail" @click=${() => this.toggle()}>${icon('x')}</button>
    </div>`;
  }

  private renderBody() {
    const host = this.sys?.hosts?.[0];
    const gpu = this.sys?.gpu;
    const cpu = host?.cpu;
    const mem = host?.memory;
    if (!cpu && !gpu) return html`${this.head()}<div class="hw-rail-empty">sampling…</div>`;
    const ring = this.ring;
    // the busiest non-virtual interface is 'the' LAN for the tile; '/' is 'the' disk
    const rootFs = (host?.fs ?? []).find((f) => f.path === '/') ?? (host?.fs ?? [])[0] ?? null;
    const nets = (host?.net ?? []).filter((n) => n.rx_bps != null || n.tx_bps != null);
    const lan = nets.length
      ? nets.reduce((a, b) => (Math.max(b.rx_bps ?? 0, b.tx_bps ?? 0) > Math.max(a.rx_bps ?? 0, a.tx_bps ?? 0) ? b : a))
      : null;
    const r = (v: number | null | undefined): string => (v == null ? '–' : String(Math.round(v)));
    return html`
      ${this.head()}
      <div class="hw-tiles">
        ${this.tile('GPU Temp', gpu?.temperature_c != null ? `${r(gpu.temperature_c)} °C` : '–', (gpu?.temperature_c ?? 0) > 85)}
        ${this.tile('GPU Power', gpu?.power_w != null ? `${fixed(gpu.power_w, 1)} W` : '–')}
        ${this.tile('RAM Used', mem?.used_bytes != null ? bytes(mem.used_bytes, 1) : '–')}
        ${this.tile('CPU', cpu?.utilization != null ? `${r(cpu.utilization * 100)} %` : '–', (cpu?.utilization ?? 0) > 0.9)}
        ${this.tile('Disk free', rootFs?.free_bytes != null ? bytes(rootFs.free_bytes, 0) : '–',
      rootFs ? rootFs.free_bytes < 0.1 * (rootFs.total_bytes || 1) : false)}
        ${lan ? this.tile('LAN', `↓${rate(lan.rx_bps)} ↑${rate(lan.tx_bps)}`, false, 'hw-tile-duo') : this.tile('LAN', '–', false, 'hw-tile-duo')}
      </div>
      <div class="hw-rail-sub num">
        ${cpu?.load?.[0] != null ? `load ${fixed(cpu.load[0], 2)}` : ''}
        ${cpu?.temperature_c != null ? ` · CPU ${r(cpu.temperature_c)}°C` : ''}
        ${cpu?.utilization != null ? ` · CPU ${r(cpu.utilization * 100)} %` : ''}
        ${gpu?.utilization != null ? ` · GPU ${r(gpu.utilization * 100)} %` : ''}
      </div>
      <div class="hw-sparks">
        ${this.chart('Temp °C', [
          this.line('GPU', 'red', ring.map((p) => p.gpu_temp_c), lastOf(ring.map((p) => p.gpu_temp_c))),
          this.line('CPU', 'cobalt', ring.map((p) => p.cpu_temp_c), lastOf(ring.map((p) => p.cpu_temp_c)))], (v) => `${Math.round(v)}`, ' °C')}
        ${this.chart('GPU power W', [this.line('GPU W', 'violet', ring.map((p) => p.gpu_power_w))], (v) => `${Math.round(v)}`, ' W')}
        ${this.chart('GPU clock MHz', [this.line('MHz', 'aqua', ring.map((p) => p.gpu_clk_mhz))], (v) => `${Math.round(v)}`, ' MHz', true)}
        ${this.chart('Unified Memory GiB', [this.line('used', 'cobalt', ring.map((p) => p.ram_used_gib))], (v) => `${Math.round(v)}`, ' GiB')}
      </div>
      <div class="hw-rail-foot num">${this.ring.length}/${RING} · ${this.open ? STEP_S : PEEK_S} s sampler · \
${(this.ring[this.ring.length - 1]?.t ?? 0) - (this.ring[0]?.t ?? 0)} s window</div>`;
  }
}
