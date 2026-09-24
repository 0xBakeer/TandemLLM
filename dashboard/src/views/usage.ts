// Usage — a year of tokens at a glance (VIS-14). Hero: tokens this year over the heatmap.
// Stat tiles today / 7 d / 30 d / year with the input-cached-output-reasoning split, trends,
// speed over time, top days, model/client filters kept in the URL hash.

import { html, nothing, type TemplateResult } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import { api, describeError } from '../api/client';
import type { Summary, Totals, Usage, UsageBucket } from '../api/types';
import { compact, dateLong, exact, ms, tps } from '../lib/format';
import { buildHeatGrid, bucketDayKey, streaks, yearWindow, type HeatGrid, type HeatMetric } from '../lib/heatmap';
import { setParams } from '../lib/router';
import { addDays, browserTz, dayKey, diffDays, todayKey } from '../lib/time';
import { LightElement, type Loadable } from '../ui/base';
import { icon, panel, skeleton, statTile, errorState, emptyState } from '../ui/bits';

type Range = '30d' | '90d' | '365d' | 'custom';

const SPLIT: { key: 'uncached' | 'cached' | 'completion' | 'reasoning'; name: string; color: string }[] = [
  { key: 'uncached', name: 'input', color: 'cobalt' },
  { key: 'cached', name: 'cached input', color: 'aqua' },
  { key: 'completion', name: 'output', color: 'orange' },
  { key: 'reasoning', name: 'reasoning', color: 'violet' },
];

function split(t: { prompt_tokens: number; cached_tokens: number; completion_tokens: number; reasoning_tokens: number }) {
  return {
    uncached: Math.max(0, t.prompt_tokens - t.cached_tokens),
    cached: t.cached_tokens,
    completion: Math.max(0, t.completion_tokens - t.reasoning_tokens),
    reasoning: t.reasoning_tokens,
  };
}

function sumBuckets(b: UsageBucket[]): Pick<Totals, 'requests' | 'errors' | 'prompt_tokens' | 'cached_tokens' | 'completion_tokens' | 'reasoning_tokens' | 'total_tokens'> {
  const out = { requests: 0, errors: 0, prompt_tokens: 0, cached_tokens: 0, completion_tokens: 0, reasoning_tokens: 0, total_tokens: 0 };
  for (const x of b) {
    out.requests += x.requests;
    out.errors += x.errors;
    out.prompt_tokens += x.prompt_tokens;
    out.cached_tokens += x.cached_tokens;
    out.completion_tokens += x.completion_tokens;
    out.reasoning_tokens += x.reasoning_tokens;
    out.total_tokens += x.total_tokens;
  }
  return out;
}

@customElement('qse-usage')
export class QseUsage extends LightElement {
  @property({ attribute: false }) summary: Loadable<Summary> = { state: 'loading' };
  @property({ attribute: false }) params: URLSearchParams = new URLSearchParams();
  @state() private usage: Loadable<Usage> = { state: 'loading' };
  @state() private custom: Loadable<Usage> | null = null;
  @state() private metric: HeatMetric = 'total_tokens';
  private tz = browserTz();
  private lastKey = '';
  private timer: ReturnType<typeof setInterval> | null = null;

  connectedCallback(): void {
    super.connectedCallback();
    this.timer = setInterval(() => document.visibilityState === 'visible' && this.load(true), 60_000);
  }
  disconnectedCallback(): void {
    if (this.timer) clearInterval(this.timer);
    super.disconnectedCallback();
  }

  updated(changed: Map<string, unknown>): void {
    if (changed.has('params')) {
      const key = `${this.model}|${this.client}|${this.range}|${this.from}|${this.to}`;
      if (key !== this.lastKey) {
        this.lastKey = key;
        void this.load();
      }
    }
  }

  private get model() {
    return this.params.get('model');
  }
  private get client() {
    return this.params.get('client');
  }
  private get range(): Range {
    const r = this.params.get('range');
    return r === '30d' || r === '365d' || r === 'custom' ? r : '90d';
  }
  private get from() {
    return this.params.get('from');
  }
  private get to() {
    return this.params.get('to');
  }

  private async load(quiet = false): Promise<void> {
    if (!quiet) this.usage = { state: 'loading' };
    const today = todayKey(this.tz);
    const { from, to } = yearWindow(today);
    try {
      const u = await api.usage({ from, to, bucket: 'day', tz: this.tz, model: this.model, client: this.client });
      this.usage = { state: 'ready', data: u };
    } catch (e) {
      const d = describeError(e);
      this.usage = { state: 'error', endpoint: d.endpoint, message: d.message };
    }
    if (this.range === 'custom' && this.from && this.to) {
      this.custom = { state: 'loading' };
      try {
        const u = await api.usage({ from: this.from, to: this.to, bucket: 'day', tz: this.tz, model: this.model, client: this.client });
        this.custom = { state: 'ready', data: u };
      } catch (e) {
        const d = describeError(e);
        this.custom = { state: 'error', endpoint: d.endpoint, message: d.message };
      }
    } else this.custom = null;
  }

  private setRange(r: Range) {
    if (r === 'custom') {
      const today = todayKey(this.tz);
      setParams({ range: 'custom', from: this.from ?? addDays(today, -44), to: this.to ?? today });
    } else setParams({ range: r === '90d' ? null : r, from: null, to: null });
  }

  /** The buckets the trend charts show. */
  private chartBuckets(u: Usage): { buckets: UsageBucket[]; note: string } {
    if (this.range === 'custom') {
      if (this.custom?.state === 'ready') return { buckets: this.custom.data.buckets, note: `${dateLong(this.custom.data.from)} – ${dateLong(this.custom.data.to)}` };
      return { buckets: [], note: 'custom range' };
    }
    const n = this.range === '30d' ? 30 : this.range === '90d' ? 90 : 365;
    return { buckets: u.buckets.slice(-n), note: `last ${n} days` };
  }

  render() {
    const filtered = !!(this.model || this.client);
    return html`
      <div class="view view-usage">
        ${this.renderHero(filtered)}
        ${this.renderFilters()}
        ${this.renderHeatmap()}
        ${this.renderCharts()}
        ${this.renderTopDays()}
      </div>
    `;
  }

  // ---- hero + stat tiles ------------------------------------------------------------------
  private renderHero(filtered: boolean): TemplateResult {
    const s = this.summary;
    const u = this.usage;
    if (s.state === 'loading' || (filtered && u.state === 'loading')) return html`<section class="hero">${skeleton(2, 'skeleton-hero')}</section>`;
    if (s.state === 'error') return html`<section class="hero">${errorState(s.endpoint, s.message)}</section>`;
    if (s.state === 'empty') return html`<section class="hero">${emptyState('No ledger yet')}</section>`;
    const sum = s.data;
    const today = todayKey(this.tz);

    // Windows: from the summary, or from the filtered usage buckets when a filter is on.
    let windows: { key: string; label: string; t: ReturnType<typeof sumBuckets> }[];
    let year: ReturnType<typeof sumBuckets>;
    let streak = { current: sum.streak.current_days, longest: sum.streak.longest_days, activeDays: sum.windows['365d'].active_days };
    if (filtered && u.state === 'ready') {
      const b = u.data.buckets;
      const win = (n: number) => sumBuckets(b.slice(-n));
      windows = [
        { key: 'today', label: 'Today', t: win(1) },
        { key: '7d', label: '7 days', t: win(7) },
        { key: '30d', label: '30 days', t: win(30) },
        { key: '365d', label: 'Year', t: win(365) },
      ];
      year = win(365);
      streak = streaks(b, today);
    } else {
      windows = [
        { key: 'today', label: 'Today', t: sum.windows.today },
        { key: '7d', label: '7 days', t: sum.windows['7d'] },
        { key: '30d', label: '30 days', t: sum.windows['30d'] },
        { key: '365d', label: 'Year', t: sum.windows['365d'] },
      ];
      year = sum.windows['365d'];
    }
    const ledgerAge = sum.ledger.since ? diffDays(dayKey(Date.parse(sum.ledger.since), this.tz), today) : null;

    return html`
      <section class="hero">
        <div class="hero-main">
          <div class="hero-figure">
            <span class="hero-number num" title=${exact(year.total_tokens) + ' tokens'}>${compact(year.total_tokens, 2)}</span>
            <span class="hero-caption">tokens in the last year${filtered ? html` <em>filtered</em>` : nothing}</span>
          </div>
          <dl class="hero-facts">
            <div><dt>requests</dt><dd class="num">${exact(year.requests)}</dd></div>
            <div><dt>active days</dt><dd class="num">${streak.activeDays}</dd></div>
            <div><dt>current streak</dt><dd class="num">${streak.current}<span class="unit">d</span></dd></div>
            <div><dt>longest streak</dt><dd class="num">${streak.longest}<span class="unit">d</span></dd></div>
          </dl>
        </div>
        <div class="stats" role="list">
          ${windows.map((w) => this.renderWindow(w.key, w.label, w.t))}
        </div>
        ${ledgerAge != null && ledgerAge < 7
          ? html`<p class="note note-info">History starts on ${dateLong(sum.ledger.since as string)} — the ledger is ${ledgerAge === 0 ? 'new today' : `${ledgerAge} day${ledgerAge === 1 ? '' : 's'} old`}; there is no backfill.</p>`
          : nothing}
      </section>
    `;
  }

  private renderWindow(key: string, label: string, t: ReturnType<typeof sumBuckets> & Partial<Totals>): TemplateResult {
    const sp = split(t);
    const total = Math.max(1, sp.uncached + sp.cached + sp.completion + sp.reasoning);
    return html`<div class="stat stat-window" role="listitem" data-window=${key}>
      <div class="stat-label">${label}</div>
      <div class="stat-value"><span class="num" title=${exact(t.total_tokens) + ' tokens'}>${compact(t.total_tokens)}</span><span class="stat-unit">tokens</span></div>
      <div class="stat-sub"><span class="num">${exact(t.requests)}</span> requests${(t.errors ?? 0) > 0 || (t.refused ?? 0) > 0 ? html` <span class="stat-err"><span class="num">${(t.errors ?? 0) + (t.refused ?? 0)}</span> failed</span>` : nothing}</div>
      <div class="split" role="img" aria-label=${SPLIT.map((s) => `${s.name} ${exact(sp[s.key])}`).join(', ')}>
        ${SPLIT.map((s) => (sp[s.key] > 0 ? html`<span class="split-seg" style="width:${(sp[s.key] / total) * 100}%;background:var(--series-${s.color})" title="${s.name} ${exact(sp[s.key])}"></span>` : nothing))}
      </div>
      <div class="split-legend">
        ${SPLIT.map((s) => html`<span><i class="swatch" style="background:var(--series-${s.color})"></i>${s.name} <b class="num">${compact(sp[s.key])}</b></span>`)}
      </div>
    </div>`;
  }

  // ---- filters --------------------------------------------------------------------------
  private renderFilters(): TemplateResult {
    const dims = this.usage.state === 'ready' ? this.usage.data.dimensions : { models: [], clients: [] };
    const chips: TemplateResult[] = [];
    if (this.model) chips.push(html`<button class="chip" @click=${() => setParams({ model: null })}>model ${this.model} ${icon('x')}</button>`);
    if (this.client) {
      const c = dims.clients.find((x) => x.id === this.client);
      chips.push(html`<button class="chip" @click=${() => setParams({ client: null })}>client ${c?.label ?? this.client} ${icon('x')}</button>`);
    }
    return html`<div class="toolbar" role="group" aria-label="filters and range">
      <label class="field">
        <span>Metric</span>
        <select @change=${(e: Event) => (this.metric = (e.target as HTMLSelectElement).value as HeatMetric)} .value=${this.metric}>
          <option value="total_tokens">total tokens</option>
          <option value="completion_tokens">output tokens</option>
          <option value="requests">requests</option>
        </select>
      </label>
      <label class="field">
        <span>Model</span>
        <select @change=${(e: Event) => setParams({ model: (e.target as HTMLSelectElement).value || null })} .value=${this.model ?? ''}>
          <option value="">all</option>
          ${dims.models.map((m) => html`<option value=${m} ?selected=${m === this.model}>${m}</option>`)}
        </select>
      </label>
      <label class="field">
        <span>Client</span>
        <select @change=${(e: Event) => setParams({ client: (e.target as HTMLSelectElement).value || null })} .value=${this.client ?? ''}>
          <option value="">all</option>
          ${dims.clients.map((c) => html`<option value=${c.id} ?selected=${c.id === this.client}>${c.label ?? c.id} (${c.kind})</option>`)}
        </select>
      </label>
      <div class="seg" role="radiogroup" aria-label="range">
        ${(['30d', '90d', '365d', 'custom'] as Range[]).map(
          (r) => html`<button role="radio" aria-checked=${this.range === r} class="seg-btn ${this.range === r ? 'is-on' : ''}" @click=${() => this.setRange(r)}>${r === 'custom' ? 'custom' : r.replace('d', ' d')}</button>`,
        )}
      </div>
      ${this.range === 'custom'
        ? html`<label class="field"><span>From</span><input type="date" .value=${this.from ?? ''} max=${todayKey(this.tz)} @change=${(e: Event) => setParams({ from: (e.target as HTMLInputElement).value })} /></label>
            <label class="field"><span>To</span><input type="date" .value=${this.to ?? ''} max=${todayKey(this.tz)} @change=${(e: Event) => setParams({ to: (e.target as HTMLInputElement).value })} /></label>`
        : nothing}
      ${chips.length ? html`<div class="chips">${chips}</div>` : nothing}
    </div>`;
  }

  // ---- heatmap --------------------------------------------------------------------------
  private renderHeatmap(): TemplateResult {
    const body = (() => {
      const u = this.usage;
      if (u.state === 'loading') return skeleton(4, 'skeleton-heat');
      if (u.state === 'error') return errorState(u.endpoint, u.message, () => this.load());
      if (u.state === 'empty') return emptyState('No usage recorded yet');
      const today = todayKey(this.tz);
      const grid: HeatGrid = buildHeatGrid(u.data.buckets, { from: u.data.from, to: u.data.to, metric: this.metric, today });
      const label = this.metric === 'requests' ? 'requests' : this.metric === 'completion_tokens' ? 'output tokens' : 'tokens';
      const active = u.data.buckets.filter((b) => b.requests > 0).length;
      return html`<qse-heatmap .grid=${grid} .metricLabel=${label} .cell=${12} .gap=${3}></qse-heatmap>
        <p class="panel-foot">${active} active day${active === 1 ? '' : 's'} of ${u.data.buckets.length}; colour steps are quantiles of the non-zero days, so a quiet month still shows its shape. Days are local (${u.data.tz}).</p>`;
    })();
    const metricName = this.metric === 'requests' ? 'Requests per day' : this.metric === 'completion_tokens' ? 'Output tokens per day' : 'Tokens per day';
    return panel(metricName, body, { sub: 'the last 365 days, Monday first, today outlined', cls: 'panel-heat', id: 'heatmap' });
  }

  // ---- charts ---------------------------------------------------------------------------
  private renderCharts(): TemplateResult {
    const u = this.usage;
    if (u.state !== 'ready') return html`<div class="grid-2">${panel('Tokens per day', u.state === 'loading' ? skeleton(3) : nothing)}${panel('Requests per day', u.state === 'loading' ? skeleton(3) : nothing)}</div>`;
    const c = this.custom;
    if (this.range === 'custom' && c && c.state !== 'ready') {
      return html`<div class="grid-2">${panel('Tokens per day', c.state === 'loading' ? skeleton(3) : c.state === 'error' ? errorState(c.endpoint, c.message) : nothing)}</div>`;
    }
    const { buckets, note } = this.chartBuckets(u.data);
    const labels = buckets.map((b) => {
      const d = bucketDayKey(b);
      return buckets.length > 120 ? d.slice(0, 7) : d.slice(5);
    });
    const spl = buckets.map((b) => split(b));
    const fmtTok = (v: number) => compact(v);
    return html`
      <div class="grid-2">
        ${panel(
          'Tokens per day',
          html`<qse-bars
            .series=${SPLIT.map((s) => ({ name: s.name, color: s.color as never, values: spl.map((x) => x[s.key]) }))}
            .labels=${labels}
            .format=${fmtTok}
            unit="tok"
            .height=${220}
            aria-label="tokens per day, split by input, cached input, output and reasoning"
          ></qse-bars>`,
          { sub: note, id: 'chart-tokens' },
        )}
        ${panel(
          'Requests per day',
          html`<qse-ribbon .series=${[{ name: 'requests', color: 'cobalt', values: buckets.map((b) => b.requests) }]} .labels=${labels} .format=${(v: number) => exact(v)} .height=${220} aria-label="requests per day"></qse-ribbon>`,
          { sub: note, id: 'chart-requests' },
        )}
        ${panel(
          'Decode speed',
          html`<qse-ribbon
            .series=${[{ name: 'p50', color: 'cobalt', values: buckets.map((b) => b.decode_tps_p50), band: buckets.map((b) => b.decode_tps_p90), bandName: 'p90' }]}
            .labels=${labels}
            .format=${(v: number) => String(Math.round(v))}
            unit="tok/s"
            .height=${200}
            aria-label="decode tokens per second, p50 and p90 per day"
          ></qse-ribbon>`,
          { sub: 'tok/s per day, p50 and p90; replays and errors excluded', id: 'chart-decode' },
        )}
        ${panel(
          'Time to first token',
          html`<qse-ribbon
            .series=${[{ name: 'p50', color: 'orange', values: buckets.map((b) => b.ttft_ms_p50), band: buckets.map((b) => b.ttft_ms_p90), bandName: 'p90' }]}
            .labels=${labels}
            .format=${(v: number) => ms(v)}
            log
            .height=${200}
            aria-label="time to first token, p50 and p90 per day, log scale"
          ></qse-ribbon>`,
          { sub: 'per day, p50 and p90, log scale', id: 'chart-ttft' },
        )}
      </div>
    `;
  }

  // ---- top days -------------------------------------------------------------------------
  private renderTopDays(): TemplateResult {
    const u = this.usage;
    const body = (() => {
      if (u.state === 'loading') return skeleton(5);
      if (u.state !== 'ready') return nothing;
      const dims = u.data.dimensions.clients;
      if (u.data.top_days.length === 0) return emptyState('No days with usage in the last year');
      return html`<table class="table table-top">
        <thead><tr><th>Day</th><th class="num-col">Tokens</th><th class="num-col">Requests</th><th>Top client</th></tr></thead>
        <tbody>
          ${u.data.top_days.map((d) => {
            const c = dims.find((x) => x.id === d.top_client);
            return html`<tr>
              <td>${dateLong(d.date)}</td>
              <td class="num-col num" title=${exact(d.total_tokens)}>${compact(d.total_tokens)}</td>
              <td class="num-col num">${exact(d.requests)}</td>
              <td class="cell-wrap">${c ? html`${c.label ?? c.id}${c.label !== c.kind ? html` <span class="muted">${c.kind}</span>` : nothing}` : (d.top_client ?? '—')}</td>
            </tr>`;
          })}
        </tbody>
      </table>`;
    })();
    return panel('Top days', body, { sub: 'the ten heaviest days of the last year', id: 'top-days' });
  }
}

export { tps as _tps };
