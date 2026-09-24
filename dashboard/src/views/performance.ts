// Performance — how fast is the engine now, and how has that changed (VIS-15). The live strip
// is computed in the browser from successive /metrics scrapes; the scatter and the history come
// from the ledger API. One y axis per chart; the reading aids come from server/METRICS.md.

import { html, nothing, type TemplateResult } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import { api, describeError } from '../api/client';
import type { RequestRow, Usage } from '../api/types';
import { compact, exact, fixed, ms, pct, tps, timeShort, dateTimeShort } from '../lib/format';
import { liveNumbers, parsePrometheus, histogram, histogramDelta, histogramQuantile, type LiveNumbers, type Scrape } from '../lib/prom';
import { poll, type Poller } from '../lib/poll';
import { setParams } from '../lib/router';
import { addDays, browserTz, todayKey } from '../lib/time';
import { LightElement, type Loadable } from '../ui/base';
import { icon, panel, skeleton, statTile, errorState, emptyState } from '../ui/bits';
import type { ScatterPoint } from '../charts/scatter';

declare const __GRAFANA_URL__: string;

type HRange = '24h' | '7d' | '30d' | '365d';

@customElement('qse-performance')
export class QsePerformance extends LightElement {
  @property({ attribute: false }) params: URLSearchParams = new URLSearchParams();
  @state() private scrapes: Scrape[] = []; // last 5 minutes
  @state() private live: LiveNumbers | null = null;
  @state() private metricsError: { endpoint: string; message: string } | null = null;
  @state() private recent: Loadable<RequestRow[]> = { state: 'loading' };
  @state() private history: Loadable<Usage> = { state: 'loading' };
  @state() private showReplays = false;
  @state() private selected: RequestRow | null = null;
  private tz = browserTz();
  private livePoll: Poller | null = null;
  private recentPoll: Poller | null = null;
  private histPoll: Poller | null = null;
  private lastKey = '';

  private get range(): HRange {
    const r = this.params.get('range');
    return r === '24h' || r === '30d' || r === '365d' ? r : '7d';
  }
  private get model() {
    return this.params.get('model');
  }
  private get client() {
    return this.params.get('client');
  }

  connectedCallback(): void {
    super.connectedCallback();
    this.livePoll = poll(() => this.scrape(), 5000);
    this.recentPoll = poll(() => this.loadRecent(), 30_000);
    this.histPoll = poll(() => this.loadHistory(), 60_000);
    this.livePoll.start();
    this.recentPoll.start();
    this.histPoll.start();
  }
  disconnectedCallback(): void {
    this.livePoll?.stop();
    this.recentPoll?.stop();
    this.histPoll?.stop();
    super.disconnectedCallback();
  }
  updated(changed: Map<string, unknown>): void {
    if (changed.has('params')) {
      const key = `${this.range}|${this.model}|${this.client}`;
      if (key !== this.lastKey) {
        this.lastKey = key;
        this.history = { state: 'loading' };
        void this.histPoll?.refresh();
        void this.recentPoll?.refresh();
      }
    }
  }

  private async scrape(): Promise<void> {
    try {
      const text = await api.metrics();
      const s = parsePrometheus(text);
      const keep = this.scrapes.filter((x) => s.at - x.at <= 5 * 60_000);
      keep.push(s);
      this.scrapes = keep;
      const prev = keep.length >= 2 ? keep[keep.length - 2] : null;
      const live = liveNumbers(prev, s);
      // TTFT p50 over the last five minutes: delta against the oldest scrape in the window.
      const oldest = keep[0];
      if (oldest && oldest !== s) {
        const d = histogramDelta(histogram(oldest, 'qse_time_to_first_token_seconds'), histogram(s, 'qse_time_to_first_token_seconds'));
        if (d && d.count > 0) live.ttftP50 = histogramQuantile(d, 0.5);
      }
      this.live = live;
      this.metricsError = null;
    } catch (e) {
      const d = describeError(e);
      this.metricsError = { endpoint: d.endpoint || '/metrics', message: d.message };
    }
  }

  private async loadRecent(): Promise<void> {
    try {
      const r = await api.requests({ limit: 500, model: this.model, client: this.client });
      this.recent = { state: 'ready', data: r.requests };
    } catch (e) {
      const d = describeError(e);
      this.recent = { state: 'error', endpoint: d.endpoint, message: d.message };
    }
  }

  private async loadHistory(): Promise<void> {
    const today = todayKey(this.tz);
    const days = this.range === '24h' ? 2 : this.range === '7d' ? 7 : this.range === '30d' ? 30 : 365;
    const bucket = days <= 31 ? 'hour' : 'day';
    try {
      const u = await api.usage({ from: addDays(today, -(days - 1)), to: today, bucket, tz: this.tz, model: this.model, client: this.client });
      if (this.range === '24h') u.buckets = u.buckets.slice(-24);
      this.history = { state: 'ready', data: u };
    } catch (e) {
      const d = describeError(e);
      this.history = { state: 'error', endpoint: d.endpoint, message: d.message };
    }
  }

  private lastRequest(): RequestRow | null {
    if (this.recent.state !== 'ready') return null;
    return this.recent.data.find((r) => r.decode_tps != null && r.cache_source !== 'response') ?? null;
  }

  render() {
    return html`<div class="view view-performance">
      ${this.renderLive()}
      <div class="grid-2 grid-2-wide">
        ${this.renderScatter()}
        ${this.renderSelected()}
      </div>
      ${this.renderHistory()}
    </div>`;
  }

  // ---- live strip --------------------------------------------------------------------------
  private renderLive(): TemplateResult {
    const l = this.live;
    const last = this.lastRequest();
    const err = this.metricsError;
    const tiles: TemplateResult[] = [];
    if (err && !l) {
      return html`<section class="live">${errorState(err.endpoint, err.message, () => this.livePoll?.refresh())}</section>`;
    }
    if (!l) return html`<section class="live">${skeleton(2, 'skeleton-live')}</section>`;
    const generating = l.generating && l.decodeTps != null && l.decodeTps > 0;
    tiles.push(
      statTile({
        label: generating ? 'decode now' : 'decode, last request',
        value: generating ? fixed(l.decodeTps, 1) : last ? fixed(last.decode_tps, 1) : '—',
        unit: 'tok/s',
        sub: generating ? html`<span class="pulse"></span>generating` : last ? `${timeShort(last.ts)} · ${last.completion_tokens} tokens` : 'no request yet',
        na: !generating && !last,
      }),
    );
    tiles.push(statTile({ label: 'time to first token, 5 min p50', value: l.ttftP50 == null ? 'none' : ms(l.ttftP50 * 1000), sub: 'from /metrics histogram deltas', na: l.ttftP50 == null }));
    tiles.push(
      l.specReported
        ? statTile({ label: 'tokens per block', value: fixed(l.tokensPerBlock, 2), sub: 'the number the engine is tuned on', na: l.tokensPerBlock == null })
        : statTile({ label: 'tokens per block', value: 'not reported', sub: 'no speculation families in /metrics', na: true }),
    );
    tiles.push(
      l.specReported
        ? statTile({ label: 'draft acceptance', value: l.acceptance == null ? '—' : (l.acceptance * 100).toFixed(1), unit: l.acceptance == null ? '' : '%', sub: 'accepted / drafted', na: l.acceptance == null })
        : statTile({ label: 'draft acceptance', value: 'not reported', sub: 'no speculation families in /metrics', na: true }),
    );
    tiles.push(statTile({ label: 'queue', value: `${l.running ?? 0} running`, sub: `${l.waiting ?? 0} waiting`, warn: (l.waiting ?? 0) > 6 }));
    return html`<section class="live" aria-label="live speed">
      <div class="live-head">
        <h2 class="panel-title">Live</h2>
        <span class="panel-sub">from <code>/metrics</code> every 5 s${err ? html` · <span class="warn-ink">${err.message}</span>` : nothing}</span>
        ${__GRAFANA_URL__ ? html`<a class="btn btn-ghost btn-sm" href=${__GRAFANA_URL__} target="_blank" rel="noopener">Open in Grafana ${icon('external')}</a>` : nothing}
      </div>
      <div class="stats stats-5">${tiles}</div>
    </section>`;
  }

  // ---- scatter -----------------------------------------------------------------------------
  private renderScatter(): TemplateResult {
    const body = (() => {
      const r = this.recent;
      if (r.state === 'loading') return skeleton(4);
      if (r.state === 'error') return errorState(r.endpoint, r.message, () => this.recentPoll?.refresh());
      if (r.state === 'empty') return emptyState('No requests yet');
      const rows = r.data.filter((x) => x.decode_tps != null && x.prompt_tokens != null && (this.showReplays || x.cache_source !== 'response'));
      const replays = r.data.filter((x) => x.cache_source === 'response').length;
      const pts: ScatterPoint[] = rows.map((x) => ({
        id: x.id,
        x: Math.max(1, x.prompt_tokens as number),
        y: x.decode_tps as number,
        group: x.cache_source ?? 'none',
        diamond: !x.thinking,
        label: `${dateTimeShort(x.ts)} · ${x.client.label ?? x.client.kind}`,
        detail: `${tps(x.decode_tps)} · ${exact(x.prompt_tokens)} prompt (${exact(x.cached_tokens)} cached) · ${exact(x.completion_tokens)} out · TTFT ${ms(x.ttft_ms)} · ${x.cache_source}`,
      }));
      return html`<qse-scatter
          .points=${pts}
          .groups=${[
            { key: 'none', name: 'no cache' },
            { key: 'prefix', name: 'prefix cache' },
            { key: 'session', name: 'session cache' },
            ...(this.showReplays ? [{ key: 'response', name: 'response replay' }] : []),
          ]}
          .height=${280}
          .logY=${this.showReplays}
          @select=${(e: CustomEvent<number>) => (this.selected = r.data.find((x) => x.id === e.detail) ?? null)}
        ></qse-scatter>
        <p class="panel-foot">${pts.length} of the last ${r.data.length} requests${replays ? html`; ${replays} response-cache replay${replays === 1 ? '' : 's'} ${this.showReplays ? 'shown, y axis log' : 'hidden'} — a replay's speed is a replay, not a decode` : nothing}. Click a dot for its timing.</p>`;
    })();
    return panel('Recent requests', body, {
      sub: 'decode tok/s against prompt length, from the ledger',
      id: 'scatter',
      tools: html`<label class="switch"><input type="checkbox" .checked=${this.showReplays} @change=${(e: Event) => (this.showReplays = (e.target as HTMLInputElement).checked)} /><span>show cache replays</span></label>`,
    });
  }

  private renderSelected(): TemplateResult {
    const s = this.selected;
    const body = s
      ? html`<div class="req-detail">
          <div class="req-detail-head"><code>${s.request_id}</code><span class="muted">${dateTimeShort(s.ts)} · ${s.client.label ?? s.client.id} · ${s.endpoint}${s.stream ? ', stream' : ''}</span></div>
          <qse-timing-bar .queue=${s.queue_ms} .prefill=${s.prompt_ms} .decode=${s.decode_ms}></qse-timing-bar>
          <dl class="kv kv-grid">
            <div><dt>prompt</dt><dd class="num">${exact(s.prompt_tokens)}</dd></div>
            <div><dt>cached</dt><dd class="num">${exact(s.cached_tokens)}</dd></div>
            <div><dt>completion</dt><dd class="num">${exact(s.completion_tokens)}</dd></div>
            <div><dt>reasoning</dt><dd class="num">${exact(s.reasoning_tokens)}</dd></div>
            <div><dt>decode</dt><dd class="num">${tps(s.decode_tps)}</dd></div>
            <div><dt>prefill</dt><dd class="num">${tps(s.prefill_tps)}</dd></div>
            <div><dt>tokens/block</dt><dd class="num">${fixed(s.tokens_per_block, 2)}</dd></div>
            <div><dt>acceptance</dt><dd class="num">${s.draft_tokens ? pct((s.draft_accepted ?? 0) / s.draft_tokens) : '—'}</dd></div>
            <div><dt>drafted</dt><dd class="num">${exact(s.draft_tokens)}</dd></div>
            <div><dt>accepted</dt><dd class="num">${exact(s.draft_accepted)}</dd></div>
            <div><dt>cache</dt><dd>${s.cache_source ?? '—'}</dd></div>
            <div><dt>finish</dt><dd>${s.finish_reason ?? '—'}${s.error_type ? html` <span class="err-ink">${s.error_type}</span>` : nothing}</dd></div>
          </dl>
        </div>`
      : emptyState('Pick a request', 'Click a dot in the scatter to see where its time went: queue, prefill, decode.');
    return panel('Timing breakdown', body, { id: 'timing', sub: 'the selected request' });
  }

  // ---- history -----------------------------------------------------------------------------
  private renderHistory(): TemplateResult {
    const h = this.history;
    const rangeBtns = html`<div class="seg" role="radiogroup" aria-label="history range">
      ${(['24h', '7d', '30d', '365d'] as HRange[]).map((r) => html`<button role="radio" aria-checked=${this.range === r} class="seg-btn ${this.range === r ? 'is-on' : ''}" @click=${() => setParams({ range: r === '7d' ? null : r })}>${r.replace(/(\d+)([hd])/, '$1 $2')}</button>`)}
    </div>`;
    const dims = h.state === 'ready' ? h.data.dimensions : { models: [], clients: [] };
    const toolbar = html`<div class="toolbar">
      ${rangeBtns}
      <label class="field"><span>Model</span>
        <select @change=${(e: Event) => setParams({ model: (e.target as HTMLSelectElement).value || null })}>
          <option value="">all</option>${dims.models.map((m) => html`<option value=${m} ?selected=${m === this.model}>${m}</option>`)}
        </select></label>
      <label class="field"><span>Client</span>
        <select @change=${(e: Event) => setParams({ client: (e.target as HTMLSelectElement).value || null })}>
          <option value="">all</option>${dims.clients.map((c) => html`<option value=${c.id} ?selected=${c.id === this.client}>${c.label ?? c.id} (${c.kind})</option>`)}
        </select></label>
    </div>`;
    if (h.state === 'loading') return html`<section class="history"><div class="history-head"><h2 class="panel-title">History</h2>${toolbar}</div>${skeleton(6)}</section>`;
    if (h.state === 'error') return html`<section class="history"><div class="history-head"><h2 class="panel-title">History</h2>${toolbar}</div>${errorState(h.endpoint, h.message, () => this.histPoll?.refresh())}</section>`;
    if (h.state === 'empty') return html`<section class="history">${emptyState('No history')}</section>`;
    const u = h.data;
    const b = u.buckets;
    const labels = b.map((x) => (u.bucket === 'hour' ? (b.length > 48 ? x.start.slice(5, 13).replace('T', ' ') : x.start.slice(11, 16)) : x.start.slice(5, 10)));
    const src = `${u.bucket === 'hour' ? 'hourly' : 'daily'} buckets from the ledger (400 days kept; Prometheus keeps 15)`;
    return html`<section class="history" id="history">
      <div class="history-head">
        <div class="panel-titles"><h2 class="panel-title">History</h2><p class="panel-sub">${src}</p></div>
        ${toolbar}
      </div>
      <div class="grid-2">
        ${panel('Decode speed', html`<qse-ribbon .series=${[{ name: 'p50', color: 'cobalt', values: b.map((x) => x.decode_tps_p50), band: b.map((x) => x.decode_tps_p90), bandName: 'p90' }]} .labels=${labels} .format=${(v: number) => String(Math.round(v))} unit="tok/s" .height=${200}></qse-ribbon>`, { sub: 'tok/s, p50 and p90' })}
        ${panel('Time to first token', html`<qse-ribbon .series=${[{ name: 'p50', color: 'orange', values: b.map((x) => x.ttft_ms_p50), band: b.map((x) => x.ttft_ms_p90), bandName: 'p90' }]} .labels=${labels} .format=${(v: number) => ms(v)} log .height=${200}></qse-ribbon>`, { sub: 'p50 and p90, log scale' })}
        ${panel('Prefill speed', html`<qse-ribbon .series=${[{ name: 'p50', color: 'cobalt', values: b.map((x) => x.prefill_tps_p50) }]} .labels=${labels} .format=${(v: number) => compact(v)} unit="tok/s" .height=${180}></qse-ribbon>`, { sub: 'uncached prompt tokens per second, p50' })}
        ${panel('Tokens per block', html`<qse-ribbon .series=${[{ name: 'mean', color: 'violet', values: b.map((x) => x.tokens_per_block_mean) }]} .labels=${labels} .format=${(v: number) => v.toFixed(1)} .height=${180}></qse-ribbon>`, { sub: 'committed tokens per verified block, mean' })}
        ${panel('Draft acceptance', html`<qse-ribbon .series=${[{ name: 'acceptance', color: 'aqua', values: b.map((x) => (x.draft_acceptance == null ? null : x.draft_acceptance * 100)) }]} .labels=${labels} .format=${(v: number) => v.toFixed(0)} unit="%" .height=${180}></qse-ribbon>`, { sub: 'accepted / drafted tokens' })}
        ${panel(
          'Reading the numbers',
          html`<ul class="reading">
            <li><b>Tokens per block</b> is the number the engine is tuned on: how many tokens one verify pass commits. The row's mean speed moves with it.</li>
            <li><b>Acceptance falls first</b> when the drafter loses the workload — watch it before decode tok/s drops.</li>
            <li>A per-request tok/s is not the row number: the atlas row is 256-in / 256-out at temperature 0; chats are longer, cached and mixed.</li>
            <li>The block split (narrow / wide / deep arms) and the verify-time histogram are in Grafana.</li>
          </ul>`,
          { cls: 'panel-reading' },
        )}
      </div>
    </section>`;
  }
}
