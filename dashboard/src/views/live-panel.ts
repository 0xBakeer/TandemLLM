// <qse-live-panel> (VIS-23): what the engine is doing right now, once a second, from
// /v1/dashboard/live over a server-sent event stream. Two figures with five-minute sparklines
// (decode tok/s of everything decoding, prefill tok/s), a line of counts, and one row per request
// in flight -- a table on a desktop, cards on a phone. A finished request stays 30 s with its
// final numbers. The stream closes while the tab is hidden and reopens when it is visible again.

import { html, nothing, type TemplateResult } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import { api, describeError } from '../api/client';
import type { Live, LiveRequest, LiveSample } from '../api/types';
import { compact, exact, fixed, ms } from '../lib/format';
import { agoShort, elapsed, emptyRing, orderRows, peak, phaseWords, push, seedRing, slots, sumDecodeNow, type Ring } from '../lib/live';
import { ResumableStream, type StreamState } from '../lib/sse';
import { LightElement } from '../ui/base';
import { emptyState, errorState, icon } from '../ui/bits';

const STATE_WORD: Record<StreamState, string> = { connecting: 'connecting', open: 'streaming', reconnecting: 'reconnecting', closed: 'paused', unauthorized: 'signed out', error: 'reconnecting' };

@customElement('qse-live-panel')
export class QseLivePanel extends LightElement {
  @property({ type: String }) grafanaUrl = '';
  @state() private live: Live | null = null;
  @state() private ring: Ring = emptyRing();
  @state() private streamState: StreamState = 'closed';
  @state() private error: { endpoint: string; message: string } | null = null;
  @state() private tickAt = 0; // bumps every event so the sparkline slots move
  private stream: ResumableStream | null = null;
  private clockTimer: ReturnType<typeof setInterval> | null = null;

  connectedCallback(): void {
    super.connectedCallback();
    document.addEventListener('visibilitychange', this.onVisibility);
    void this.firstPaint();
    if (document.visibilityState === 'visible') this.open();
    // the slots are anchored to the wall clock: redraw once a second even when the stream is quiet
    this.clockTimer = setInterval(() => (this.tickAt = Date.now()), 1000);
  }
  disconnectedCallback(): void {
    document.removeEventListener('visibilitychange', this.onVisibility);
    this.close();
    if (this.clockTimer) clearInterval(this.clockTimer);
    super.disconnectedCallback();
  }

  private onVisibility = () => {
    if (document.visibilityState === 'visible') this.open();
    else this.close();
  };

  /** One plain fetch so the first paint does not wait for the stream's first event. */
  private async firstPaint(): Promise<void> {
    try {
      const l = await api.live();
      if (!this.live) this.apply(l);
    } catch (e) {
      const d = describeError(e);
      if (d.status !== 401) this.error = { endpoint: d.endpoint || '/v1/dashboard/live', message: d.message };
    }
  }

  private open(): void {
    if (this.stream) return;
    this.stream = new ResumableStream({
      url: () => api.liveUrl(),
      onEvent: (ev) => {
        if (ev.event !== 'live') return;
        try {
          this.apply(JSON.parse(ev.data) as Live);
        } catch {
          /* a malformed event is skipped */
        }
      },
      onState: (st, detail) => {
        this.streamState = st;
        if (st === 'open') this.error = null;
        if (st === 'error') this.error = { endpoint: '/v1/dashboard/live', message: detail ?? 'stream closed' };
        if (st === 'unauthorized') window.dispatchEvent(new CustomEvent('qse-unauthorized'));
      },
    });
    this.stream.start();
  }
  private close(): void {
    this.stream?.close();
    this.stream = null;
    this.streamState = 'closed';
  }

  private apply(l: Live): void {
    const now = Date.now() / 1000;
    if (l.history) this.ring = seedRing(l.history, now);
    else this.ring = push(this.ring, l.sample, now);
    this.live = l;
    this.tickAt = Date.now();
  }

  render() {
    const l = this.live;
    const st = this.streamState;
    const head = html`<div class="live-head">
      <h2 class="panel-title">Live</h2>
      <span class="live-state" data-state=${st}><span class="dot dot-${st}"></span>${STATE_WORD[st]}${st === 'open' ? html` · <span class="num">1 s</span>` : nothing}</span>
      ${this.error && st !== 'open' ? html`<span class="panel-sub warn-ink">${this.error.endpoint}: ${this.error.message}</span>` : nothing}
      ${this.grafanaUrl ? html`<a class="btn btn-ghost btn-sm" href=${this.grafanaUrl} target="_blank" rel="noopener">Open in Grafana ${icon('external')}</a>` : nothing}
    </div>`;
    if (!l) {
      return html`<section class="live" aria-label="live">
        ${head}
        ${this.error ? errorState(this.error.endpoint, this.error.message, () => void this.firstPaint()) : html`<div class="skeleton skeleton-live" aria-busy="true"><div class="skeleton-line" style="width:60%"></div><div class="skeleton-line" style="width:80%"></div></div>`}
      </section>`;
    }
    const now = this.tickAt / 1000;
    const s = slots(this.ring, now);
    const rows = orderRows(l.requests);
    const decodeNow = l.now.decode_tps ?? sumDecodeNow(rows);
    const c = l.counts;
    return html`<section class="live" aria-label="live">
      ${head}
      <div class="live-now">
        ${this.figure({
          id: 'decode',
          label: 'decode, all requests',
          value: decodeNow,
          unit: 'tok/s',
          sub: c.decoding ? `${c.decoding} decoding · last 2 s` : 'nothing decoding',
          values: s.decode,
          mode: 'ribbon',
          peak: peak(s.decode),
          format: (v: number) => String(Math.round(v)),
        })}
        ${this.figure({
          id: 'prefill',
          label: 'prefill',
          value: l.now.prefill_tps,
          unit: 'tok/s',
          sub: l.now.prefilling ? this.prefillingWords(rows) : l.now.last_prefill_ms_ago == null ? 'no prefill yet' : `last prefill ${agoShort(l.now.last_prefill_ms_ago)}`,
          values: s.prefill,
          mode: 'dots',
          peak: peak(s.prefill),
          format: (v: number) => compact(v),
          busy: l.now.prefilling,
        })}
      </div>
      <dl class="live-counts" aria-label="request counts">
        ${this.count('in flight', c.in_flight, c.in_flight > 0)}
        ${this.count('queued', c.queued, c.queued > 0)}
        ${this.count('prefilling', c.prefilling, c.prefilling > 0)}
        ${this.count('decoding', c.decoding, c.decoding > 0)}
        ${this.count('done, last minute', c.completed_1m)}
        ${this.count('served since start', c.served)}
        ${this.count('errors', c.errors, false, c.errors > 0)}
        ${this.count('refused', c.refused, false, c.refused > 0)}
      </dl>
      ${rows.length ? this.rows(rows) : emptyState('No request in flight', 'The list fills the moment one arrives; a finished request stays 30 s with its final numbers.')}
    </section>`;
  }

  private prefillingWords(rows: LiveRequest[]): string {
    const p = rows.find((r) => r.phase === 'prefill');
    return p && p.prompt_tokens != null ? `prefilling ${exact(p.prompt_tokens)} tokens · ${elapsed(p.elapsed_ms)}` : 'prefilling';
  }

  private figure(f: { id: string; label: string; value: number | null; unit: string; sub: string; values: (number | null)[]; mode: 'ribbon' | 'dots'; peak: number | null; format: (v: number) => string; busy?: boolean }): TemplateResult {
    const na = f.value == null;
    return html`<div class="live-fig ${na ? 'is-na' : ''} ${f.busy ? 'is-busy' : ''}" id="live-${f.id}">
      <div class="live-fig-label">${f.label}</div>
      <div class="live-fig-value"><span class="live-fig-num">${na ? '—' : f.format === compact ? compact(f.value) : fixed(f.value, 1)}</span><span class="live-fig-unit">${f.unit}</span></div>
      <div class="live-fig-sub">${f.busy ? html`<span class="pulse"></span>` : nothing}${f.sub}</div>
      <qse-spark .values=${f.values} mode=${f.mode} .height=${44} .format=${f.format} unit=${f.unit} aria-label="${f.label}, last 5 minutes"></qse-spark>
      <div class="live-fig-foot"><span>5 min</span><span class="num">${f.peak == null ? 'no samples yet' : `peak ${f.format(f.peak)} ${f.unit}`}</span></div>
    </div>`;
  }

  private count(label: string, n: number, hot = false, warn = false): TemplateResult {
    return html`<div class="live-count ${hot ? 'is-hot' : ''} ${warn ? 'is-warn' : ''}"><dd class="num">${n}</dd><dt>${label}</dt></div>`;
  }

  private rows(rows: LiveRequest[]): TemplateResult {
    return html`<div class="live-list" role="region" aria-label="each request">
      <table class="live-table">
        <thead>
          <tr>
            <th>phase</th><th>request</th><th>client</th><th class="num-col">prompt</th><th class="num-col">tokens</th><th class="num-col">TTFT</th><th class="num-col">prefill</th><th class="num-col">decode now / avg</th><th class="num-col">tok/blk</th><th class="num-col">elapsed</th>
          </tr>
        </thead>
        <tbody>${rows.map((r) => this.row(r))}</tbody>
      </table>
      <div class="live-cards">${rows.map((r) => this.card(r))}</div>
    </div>`;
  }

  private phase(r: LiveRequest): TemplateResult {
    const p = phaseWords(r);
    return html`<span class="phase phase-${p.cls}"><span class="phase-glyph" aria-hidden="true">${p.glyph}</span>${p.word}</span>`;
  }

  private meta(r: LiveRequest): TemplateResult {
    return html`<span class="live-meta">${r.model}${r.temperature != null ? html` · t ${r.temperature}` : nothing}${r.thinking ? ' · thinking' : ''}${r.cache_source && r.cache_source !== 'none' ? ` · ${r.cache_source}` : ''}${r.endpoint === 'completions' ? ' · completions' : ''}${r.stream ? '' : ' · json'}</span>`;
  }

  private prompt(r: LiveRequest): TemplateResult {
    if (r.prompt_tokens == null) return html`—`;
    return html`${exact(r.prompt_tokens)}${r.cached_tokens ? html` <span class="muted">(${exact(r.cached_tokens)} cached)</span>` : nothing}`;
  }

  private decodeCell(r: LiveRequest): TemplateResult {
    if (r.phase === 'decode') return html`<b class="live-now-num">${fixed(r.decode_tps_now, 1)}</b> / ${fixed(r.decode_tps, 1)}`;
    if (r.phase === 'done') return html`${fixed(r.decode_tps, 1)}`;
    return html`—`;
  }

  private row(r: LiveRequest): TemplateResult {
    return html`<tr class="live-row is-${r.phase}" data-id=${r.request_id}>
      <td>${this.phase(r)}</td>
      <td><code class="live-id" title=${r.request_id}>${r.request_id.slice(0, 18)}</code><br />${this.meta(r)}</td>
      <td>${r.client.kind}${r.phase === 'done' ? html`<br /><span class="muted">${agoShort(r.ended_ms_ago)}</span>` : nothing}</td>
      <td class="num-col">${this.prompt(r)}</td>
      <td class="num-col"><b>${exact(r.tokens)}</b></td>
      <td class="num-col">${ms(r.ttft_ms)}</td>
      <td class="num-col">${r.prefill_tps == null ? '—' : compact(r.prefill_tps)}</td>
      <td class="num-col">${this.decodeCell(r)}</td>
      <td class="num-col">${fixed(r.tokens_per_block, 2)}</td>
      <td class="num-col">${elapsed(r.elapsed_ms)}</td>
    </tr>`;
  }

  private card(r: LiveRequest): TemplateResult {
    return html`<article class="live-card is-${r.phase}" data-id=${r.request_id}>
      <header class="live-card-head">
        ${this.phase(r)}
        <span class="muted">${r.client.kind}${r.phase === 'done' ? ` · ${agoShort(r.ended_ms_ago)}` : ''}</span>
      </header>
      <div class="live-card-id"><code class="live-id" title=${r.request_id}>${r.request_id.slice(0, 18)}</code> ${this.meta(r)}</div>
      <dl class="live-card-grid">
        <div><dt>tokens</dt><dd class="num"><b>${exact(r.tokens)}</b></dd></div>
        <div><dt>decode now / avg</dt><dd class="num">${this.decodeCell(r)}</dd></div>
        <div><dt>prefill</dt><dd class="num">${r.prefill_tps == null ? '—' : compact(r.prefill_tps) + ' tok/s'}</dd></div>
        <div><dt>prompt</dt><dd class="num">${this.prompt(r)}</dd></div>
        <div><dt>TTFT</dt><dd class="num">${ms(r.ttft_ms)}</dd></div>
        <div><dt>tok/blk · elapsed</dt><dd class="num">${fixed(r.tokens_per_block, 2)} · ${elapsed(r.elapsed_ms)}</dd></div>
      </dl>
    </article>`;
  }
}

export type { LiveSample };
