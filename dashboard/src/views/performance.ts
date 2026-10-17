// Performance — how fast is the engine now, and how has that changed. The Live panel at
// the top (views/live-panel.ts) streams /v1/dashboard/live once a second; the three
// 5-minute figures under it come from successive /metrics scrapes; the scatter and the history
// come from the ledger API. One y axis per chart; the reading aids come from server/METRICS.md.

import { html, nothing, type TemplateResult } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import { api, describeError } from '../api/client';
import type { RequestRow, Usage } from '../api/types';
import { compact, exact, fixed, ms, NOT_REPORTED, pct, tps, dateTimeShort } from '../lib/format';
import { backendEngines, backendFigures, liveNumbers, parsePrometheus, histogram, histogramDelta, histogramQuantile, type LiveNumbers, type Scrape } from '../lib/prom';
import { renderEngines } from './engines';
import { poll, type Poller } from '../lib/poll';
import { cacheSummary, cacheTier, isReplay, reuse, TIER_LABEL, TIERS } from '../lib/cache-source';
import { laneTitle, laneWord } from '../lib/lane';
import { setParams } from '../lib/router';
import { addDays, browserTz, todayKey } from '../lib/time';
import { LightElement, type Loadable } from '../ui/base';
import { panel, skeleton, statTile, errorState, emptyState } from '../ui/bits';
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
  private recentSeq = 0;
  private histSeq = 0;
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
      const key = `${this.range}|${this.model}|${this.client}|${this.params.get('backend')}`;
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
      // a request ended since the last scrape: the "last request" tile reads the recent list, which
      // otherwise reloads every 30 s and would show the one before it until then
      if ((live.finished ?? 0) > 0) void this.recentPoll?.refresh();
    } catch (e) {
      const d = describeError(e);
      this.metricsError = { endpoint: d.endpoint || '/metrics', message: d.message };
    }
  }

  private async loadRecent(): Promise<void> {
    // a generation token: two quick filter changes must not let the older response land
    const seq = ++this.recentSeq;
    try {
      const r = await api.requests({ limit: 500, model: this.model, client: this.client, backend: this.params.get('backend') });
      if (seq !== this.recentSeq) return;
      this.recent = { state: 'ready', data: r.requests };
    } catch (e) {
      if (seq !== this.recentSeq) return;
      const d = describeError(e);
      this.recent = { state: 'error', endpoint: d.endpoint, message: d.message };
    }
  }

  private async loadHistory(): Promise<void> {
    const seq = ++this.histSeq;
    const today = todayKey(this.tz);
    const days = this.range === '24h' ? 2 : this.range === '7d' ? 7 : this.range === '30d' ? 30 : 365;
    const bucket = days <= 31 ? 'hour' : 'day';
    try {
      const u = await api.usage({ from: addDays(today, -(days - 1)), to: today, bucket, tz: this.tz, model: this.model, client: this.client });
      if (seq !== this.histSeq) return;
      if (this.range === '24h') u.buckets = u.buckets.slice(-24);
      this.history = { state: 'ready', data: u };
    } catch (e) {
      if (seq !== this.histSeq) return;
      const d = describeError(e);
      this.history = { state: 'error', endpoint: d.endpoint, message: d.message };
    }
  }

  render() {
    return html`<div class="view view-performance">
      ${this.renderLive()}
      ${this.renderEngines()}
      <div class="grid-2 grid-2-wide">
        ${this.renderScatter()}
        ${this.renderSelected()}
      </div>
      ${this.renderCacheSources()}
      ${this.renderHistory()}
    </div>`;
  }

  // ---- live: the panel and the 5-minute figures from /metrics --------------
  private renderLive(): TemplateResult {
    const l = this.live;
    const err = this.metricsError;
    const tiles: TemplateResult[] = [];
    if (l) {
      tiles.push(statTile({ label: 'time to first token, 5 min p50', value: l.ttftP50 == null ? 'none' : ms(l.ttftP50 * 1000), sub: 'queue wait included', na: l.ttftP50 == null }));
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
    }
    return html`<qse-live-panel .grafanaUrl=${__GRAFANA_URL__} .debug=${this.params.get('debug') === 'live'}></qse-live-panel>
      <section class="live-metrics" aria-label="last five minutes">
        ${err && !l ? errorState(err.endpoint, err.message, () => this.livePoll?.refresh()) : !l ? skeleton(1, 'skeleton-live') : html`<div class="stats stats-3">${tiles}</div>`}
        <p class="live-metrics-sub">from <code>/metrics</code> every 5 s${err && l ? html` · <span class="warn-ink">${err.message}</span>` : nothing}</p>
      </section>`;
  }

  // ---- P2, P3: the vLLM and SGLang backends, from the same /metrics scrapes -----------------
  private renderEngines(): TemplateResult | typeof nothing {
    const keep = this.scrapes;
    const last = keep.length ? keep[keep.length - 1] : null;
    const first = keep.length >= 2 ? keep[0] : null;
    const cards = backendEngines(last).map((b) => ({ id: b.id, label: b.id, kind: b.kind, version: b.version, figures: backendFigures(first, last, b.id, b.kind) }));
    return renderEngines(cards, 'the qse_backend_* families of /metrics, over the last 5 minutes of scrapes');
  }

  // ---- scatter -----------------------------------------------------------------------------
  private renderScatter(): TemplateResult {
    const body = (() => {
      const r = this.recent;
      if (r.state === 'loading') return skeleton(4);
      if (r.state === 'error') return errorState(r.endpoint, r.message, () => this.recentPoll?.refresh());
      if (r.state === 'empty') return emptyState('No requests yet');
      // a replay (the response cache, a summary written ahead) decoded nothing, and a side-lane
      // request is a short plain decode beside a turn: neither is a point on the decode chart
      const replays = r.data.filter((x) => isReplay(x.cache_source));
      const laned = r.data.filter((x) => x.lane && !isReplay(x.cache_source) && x.decode_tps != null).length;
      const rows = r.data.filter((x) => x.decode_tps != null && x.prompt_tokens != null && !isReplay(x.cache_source) && !x.lane);
      const pts: ScatterPoint[] = rows.map((x) => ({
        id: x.id,
        x: Math.max(1, x.prompt_tokens as number),
        y: x.decode_tps as number,
        group: cacheTier(x.cache_source),
        diamond: !x.thinking,
        measured: x.measured_at ?? 'engine',
        label: `${dateTimeShort(x.ts)} · ${x.client.label ?? x.client.kind}`,
        detail: `${tps(x.decode_tps)} · ${exact(x.prompt_tokens)} prompt (${exact(x.cached_tokens)} cached) · ${exact(x.completion_tokens)} out · TTFT ${ms(x.ttft_ms)} · ${TIER_LABEL[cacheTier(x.cache_source)]}`,
      }));
      const notes = [
        replays.length ? `${replays.length} replay${replays.length === 1 ? '' : 's'} (answered from memory, no decode) ${this.showReplays ? 'listed below' : 'hidden'}` : '',
        laned ? `${laned} side-lane request${laned === 1 ? '' : 's'} not plotted (a short plain decode beside a turn)` : '',
      ].filter(Boolean);
      return html`<qse-scatter
          .points=${pts}
          .groups=${TIERS.filter((t) => t !== 'response').map((t) => ({ key: t, name: TIER_LABEL[t] }))}
          .height=${280}
          @select=${(e: CustomEvent<number>) => (this.selected = r.data.find((x) => x.id === e.detail) ?? null)}
        ></qse-scatter>
        <p class="panel-foot">${pts.length} of the last ${r.data.length} requests${notes.length ? `; ${notes.join('; ')}` : ''}. Click a dot for its timing.</p>
        ${this.showReplays && replays.length
          ? html`<ul class="scatter-replays" aria-label="replays: TTFT only, no decode">
              ${replays.map(
                (x) => html`<li>
                  <button class="link-btn" type="button" @click=${() => (this.selected = x)}>${dateTimeShort(x.ts)}</button>
                  <span>${x.client.label ?? x.client.kind}</span>
                  <span class="num">${exact(x.prompt_tokens)} prompt</span>
                  <span class="num">${exact(x.completion_tokens)} out</span>
                  <span class="num">TTFT ${ms(x.ttft_ms)}</span>
                  <span class="cache-tag tier-response">${x.cache_source === 'prepared' ? 'summary written ahead' : 'response cache'}</span>
                </li>`,
              )}
            </ul>`
          : nothing}`;
    })();
    return panel('Recent requests', body, {
      sub: 'decode tok/s against prompt length, from the ledger',
      id: 'scatter',
      tools: html`<label class="switch"><input type="checkbox" .checked=${this.showReplays} @change=${(e: Event) => (this.showReplays = (e.target as HTMLInputElement).checked)} /><span>list cache replays</span></label>`,
    });
  }

  private renderSelected(): TemplateResult {
    const s = this.selected;
    const body = s
      ? html`<div class="req-detail">
          <div class="req-detail-head"><code>${s.request_id}</code><span class="muted">${dateTimeShort(s.ts)} · ${s.client.label ?? s.client.id} · ${s.endpoint}${s.stream ? ', stream' : ''}</span></div>
          ${s.measured_at === 'proxy'
            ? html`<p class="muted small" data-measured="proxy">Measured at the proxy (${s.backend?.label ?? s.backend?.id}): TTFT ${ms(s.ttft_ms)} holds the network and the backend's queue; the queue, prefill and draft figures below are ${NOT_REPORTED}.</p>`
            : html`<qse-timing-bar .queue=${s.queue_ms} .prefill=${s.prompt_ms} .decode=${s.decode_ms}></qse-timing-bar>`}
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
            <div><dt>cache</dt><dd>${s.prompt_tokens == null ? '—' : html`<span class="cache-tag tier-${cacheTier(s.cache_source)}">${TIER_LABEL[cacheTier(s.cache_source)]}</span>`}</dd></div>
            <div><dt>reused</dt><dd class="num">${exact(reuse(s.prompt_tokens, s.cached_tokens).reused)}</dd></div>
            <div><dt>forwarded</dt><dd class="num">${exact(reuse(s.prompt_tokens, s.cached_tokens).forwarded)}</dd></div>
            <div><dt>finish</dt><dd>${s.finish_reason ?? '—'}${s.error_type ? html` <span class="err-ink">${s.error_type}</span>` : nothing}</dd></div>
            ${s.lane ? html`<div><dt>served</dt><dd><span class="lane-tag" data-lane=${s.lane} title=${laneTitle(s.lane)}>${laneWord(s.lane)}</span></dd></div>` : nothing}
          </dl>
        </div>`
      : emptyState('Pick a request', 'Click a dot in the scatter to see where its time went: queue, prefill, decode.');
    return panel('Timing breakdown', body, { id: 'timing', sub: 'the selected request' });
  }

  // ---- where the prompts came from, the last hour ------------------------------------------
  private renderCacheSources(): TemplateResult {
    const r = this.recent;
    const body = (() => {
      if (r.state === 'loading') return skeleton(2);
      if (r.state === 'error') return errorState(r.endpoint, r.message, () => this.recentPoll?.refresh());
      if (r.state === 'empty') return emptyState('No requests yet');
      const sum = cacheSummary(r.data, Date.now());
      if (!sum.requests) return emptyState('No request in the last hour');
      const total = sum.reused + sum.forwarded;
      return html`<div class="cache-sum-bar" role="img" aria-label="prompt tokens by source">
          ${sum.rows.map((x) => html`<span class="tier-${x.tier}" style="width:${total ? ((x.reused + x.forwarded) / total) * 100 : 0}%" title="${x.label}: ${exact(x.reused + x.forwarded)} prompt tokens"></span>`)}
        </div>
        <table class="cache-sum">
          <thead><tr><th>source</th><th class="num">requests</th><th class="num">reused</th><th class="num">forwarded</th><th class="num opt">TTFT p50</th></tr></thead>
          <tbody>
            ${sum.rows.map((x) => html`<tr data-tier=${x.tier}>
              <td><span class="cache-tag tier-${x.tier}">${x.label}</span></td>
              <td class="num">${exact(x.requests)}</td>
              <td class="num" title=${exact(x.reused)}>${compact(x.reused)}</td>
              <td class="num" title=${exact(x.forwarded)}>${compact(x.forwarded)}</td>
              <td class="num opt">${x.ttftP50 == null ? '—' : ms(x.ttftP50)}</td>
            </tr>`)}
          </tbody>
        </table>
        <p class="panel-foot">${exact(sum.requests)} requests${r.data.length >= 500 ? ' (of the newest 500)' : ''}; ${sum.reusedShare == null ? '—' : pct(sum.reusedShare)} of their prompt tokens came out of a cache. Reused: resumed without a forward; forwarded: prefilled.</p>`;
    })();
    return panel('Cache sources, last hour', body, { id: 'cache-sources', sub: 'where each prompt came from: RAM (rows in place, the end anchor, the state store), the NVMe, or cold' });
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
