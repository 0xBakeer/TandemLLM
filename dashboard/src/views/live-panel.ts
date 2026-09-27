// <qse-live-panel> (VIS-23, VIS-24): what the engine is doing right now, from /v1/dashboard/live
// over a server-sent event stream. Contract 1.1 adds the "Now" line on top (one glyph and one
// sentence for the engine, with its live number and the time in that state, at up to 4 events a
// second), an activity cell and a timeline per request, and the last 20 stops with a sentence
// each. The two figures with five-minute sparklines, the counts and the request list are VIS-23's;
// a 1.0 server (no `engine`, no `activity`) gets exactly that panel. The head shows the health of
// the stream itself, so a silent page is never mistaken for an idle engine. The stream closes while
// the tab is hidden and reopens when it is visible again.

import { html, nothing, type TemplateResult } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import { api, describeError } from '../api/client';
import type { Live, LiveRecent, LiveRequest, LiveSample } from '../api/types';
import { activityCell, clientFlag, constrainedWord, continuesText, nowLine, pathSegments, roundWords, secs, stopTone, streamHealth, timelineList, timelineSegments, tokensSplit, type Segment } from '../lib/activity';
import { compact, exact, fixed, ms, timeShort } from '../lib/format';
import { agoShort, elapsed, emptyRing, orderRows, peak, push, seedRing, slots, sumDecodeNow, type Ring } from '../lib/live';
import { navigate } from '../lib/router';
import { ResumableStream, type StreamState } from '../lib/sse';
import { LightElement } from '../ui/base';
import { emptyState, errorState, icon } from '../ui/bits';

@customElement('qse-live-panel')
export class QseLivePanel extends LightElement {
  @property({ type: String }) grafanaUrl = '';
  /** `#/performance?debug=live`: log every event's arrival and show the last 20 gaps. */
  @property({ type: Boolean }) debug = false;
  @state() private live: Live | null = null;
  @state() private ring: Ring = emptyRing();
  @state() private streamState: StreamState = 'closed';
  @state() private error: { endpoint: string; message: string } | null = null;
  @state() private tickAt = 0; // bumps every event and once a second, so the ages and the slots move
  @state() private gaps: number[] = [];
  private lastEventAt: number | null = null;
  private stateSince: number | null = null;
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
      if (!this.live) this.apply(l, false);
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
          this.apply(JSON.parse(ev.data) as Live, true);
        } catch {
          /* a malformed event is skipped */
        }
      },
      onState: (st, detail) => {
        if (st !== this.streamState) this.stateSince = Date.now();
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
    this.stateSince = Date.now();
  }

  /** Merge one message. The sparklines keep one sample a second: `push` ignores a repeated `sample.t`. */
  private apply(l: Live, fromStream: boolean): void {
    const at = Date.now();
    const now = at / 1000;
    if (l.history) this.ring = seedRing(l.history, now);
    else this.ring = push(this.ring, l.sample, now);
    this.live = l;
    if (fromStream) {
      if (this.debug) {
        const gap = this.lastEventAt == null ? null : at - this.lastEventAt;
        console.log(`[live] seq ${l.seq ?? '-'} at ${new Date(at).toISOString()}${gap == null ? '' : ` gap ${gap} ms`}`);
        if (gap != null) this.gaps = [...this.gaps.slice(-19), gap];
      }
      this.lastEventAt = at;
    }
    this.tickAt = at;
  }

  render() {
    const l = this.live;
    const st = this.streamState;
    const health = streamHealth(st, this.lastEventAt, this.stateSince, this.tickAt || Date.now(), l?.interval_s ?? 1);
    const head = html`<div class="live-head">
      <h2 class="panel-title">Live</h2>
      <span class="live-state tone-${health.tone}" data-state=${st} aria-live="polite"
        ><span class="dot dot-${st} ${health.stale && st === 'open' ? 'dot-stale' : ''}"></span>${health.word}${health.cadence && !health.stale ? html` · <span class="num">${health.cadence}</span>` : nothing}${health.stale && health.ageMs != null && st !== 'open' ? html` · <span class="num">last update ${secs(health.ageMs)} ago</span>` : nothing}</span
      >
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
    const nl = nowLine(l.engine, rows);
    const has11 = !!l.engine; // contract 1.1: the Now line, activity cells, timelines and the Last 20
    const rounds = has11 ? roundWords(rows) : null;
    return html`<section class="live ${health.stale ? 'is-stale' : ''} ${has11 ? 'is-v11' : ''}" aria-label="live">
      ${head}
      ${nl ? this.nowLine(nl) : nothing}
      <div class="live-now">
        ${this.figure({
          id: 'decode',
          label: 'decode, all requests',
          value: decodeNow,
          unit: 'tok/s',
          sub: c.decoding ? `${c.decoding} decoding · last 2 s` : 'nothing decoding',
          foot: rounds,
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
          foot: null,
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
      ${rows.length ? this.rows(rows) : has11 ? emptyState('Idle. Nothing in flight.', 'The list fills the moment a request arrives; a finished request stays 30 s with its final numbers.') : emptyState('No request in flight', 'The list fills the moment one arrives; a finished request stays 30 s with its final numbers.')}
      ${has11 && l.recent !== null ? this.recent((l.recent ?? []).slice(0, 20)) : nothing}
      ${this.debug ? this.debugOverlay() : nothing}
    </section>`;
  }

  // ---- the Now line -------------------------------------------------------------------------------

  private nowLine(n: ReturnType<typeof nowLine> & object): TemplateResult {
    return html`<div class="now tone-${n.tone} ${n.moving ? 'is-moving' : ''}" id="live-now" role="status" aria-live="polite" aria-atomic="true">
      <span class="now-glyph" aria-hidden="true">${n.glyph}</span>
      <div class="now-main">
        <div class="now-line">
          <span class="now-headline">${n.headline}</span>
          ${n.numbers.length ? html`<span class="now-numbers num">${n.numbers.join(' · ')}</span>` : nothing}
        </div>
        ${n.warning ? html`<div class="now-warning"><span class="flag-glyph" aria-hidden="true">✗</span>${n.warning}</div>` : nothing}
        ${n.progress != null ? html`<div class="now-bar" role="progressbar" aria-label="prefill" aria-valuemin="0" aria-valuemax="100" aria-valuenow=${Math.round(n.progress * 100)}><div class="now-bar-fill" style="width:${(n.progress * 100).toFixed(1)}%"></div></div>` : nothing}
      </div>
      <div class="now-side">
        ${n.sinceMs != null ? html`<span class="now-since num" title="time in this state">${secs(n.sinceMs)}</span>` : nothing}
        ${n.client ? html`<span class="now-client">${n.client}</span>` : nothing}
      </div>
    </div>`;
  }

  private prefillingWords(rows: LiveRequest[]): string {
    const p = rows.find((r) => r.phase === 'prefill');
    if (!p) return 'prefilling';
    const pf = p.activity?.prefill;
    if (pf && pf.done != null && pf.pct != null && pf.progress !== 'single_call') return `prefilling ${exact(pf.done)} of ${exact(pf.total)} · ${Math.round(pf.pct)} %`;
    return p.prompt_tokens != null ? `prefilling ${exact(p.prompt_tokens)} tokens · ${elapsed(p.elapsed_ms)}` : 'prefilling';
  }

  private figure(f: { id: string; label: string; value: number | null; unit: string; sub: string; foot: string | null; values: (number | null)[]; mode: 'ribbon' | 'dots'; peak: number | null; format: (v: number) => string; busy?: boolean }): TemplateResult {
    const na = f.value == null;
    return html`<div class="live-fig ${na ? 'is-na' : ''} ${f.busy ? 'is-busy' : ''}" id="live-${f.id}">
      <div class="live-fig-label">${f.label}</div>
      <div class="live-fig-value"><span class="live-fig-num">${na ? '—' : f.format === compact ? compact(f.value) : fixed(f.value, 1)}</span><span class="live-fig-unit">${f.unit}</span></div>
      <div class="live-fig-sub">${f.busy ? html`<span class="pulse"></span>` : nothing}${f.sub}</div>
      ${f.foot ? html`<div class="live-fig-rounds num">${f.foot}</div>` : nothing}
      <qse-spark .values=${f.values} mode=${f.mode} .height=${44} .format=${f.format} unit=${f.unit} aria-label="${f.label}, last 5 minutes"></qse-spark>
      <div class="live-fig-foot"><span>5 min</span><span class="num">${f.peak == null ? 'no samples yet' : `peak ${f.format(f.peak)} ${f.unit}`}</span></div>
    </div>`;
  }

  private count(label: string, n: number, hot = false, warn = false): TemplateResult {
    return html`<div class="live-count ${hot ? 'is-hot' : ''} ${warn ? 'is-warn' : ''}"><dd class="num">${n}</dd><dt>${label}</dt></div>`;
  }

  // ---- the requests -------------------------------------------------------------------------------

  private rows(rows: LiveRequest[]): TemplateResult {
    const v11 = rows.some((r) => r.activity);
    return html`<div class="live-list" role="region" aria-label="each request">
      <table class="live-table">
        <thead>
          <tr>
            <th>${v11 ? 'activity' : 'phase'}</th><th>request</th><th>client</th><th class="num-col">prompt</th><th class="num-col">tokens</th><th class="num-col">TTFT</th><th class="num-col">prefill</th><th class="num-col">decode now / avg</th><th class="num-col">tok/blk</th><th class="num-col">elapsed</th>${v11 ? html`<th class="tl-col">timeline</th>` : nothing}
          </tr>
        </thead>
        <tbody>${rows.map((r) => this.row(r, v11))}</tbody>
      </table>
      <div class="live-cards">${rows.map((r) => this.card(r))}</div>
    </div>`;
  }

  private cell(r: LiveRequest): TemplateResult {
    const c = activityCell(r.activity, r);
    return html`<span class="phase act tone-${c.tone} ${c.moving ? 'is-moving' : ''}"><span class="phase-glyph" aria-hidden="true">${c.glyph}</span>${c.word}${c.detail ? html` <span class="act-detail num">${c.detail}</span>` : nothing}</span>`;
  }

  private flags(r: LiveRequest): TemplateResult | typeof nothing {
    const a = r.activity;
    if (!a) return nothing;
    const flag = r.phase !== 'done' ? clientFlag(a) : null;
    const cont = continuesText(a.continues);
    const tag = a.constrained ? constrainedWord(a.constrained) : null;
    if (!flag && !cont && !tag) return nothing;
    return html`<div class="live-flags">
      ${flag ? html`<span class="flag flag-bad"><span class="flag-glyph" aria-hidden="true">✗</span>${flag}</span>` : nothing}
      ${cont ? html`<span class="flag flag-continues"><span class="flag-glyph" aria-hidden="true">↩</span>${cont}</span>` : nothing}
      ${tag ? html`<span class="tag">${tag}</span>` : nothing}
    </div>`;
  }

  private meta(r: LiveRequest): TemplateResult {
    return html`<span class="live-meta">${r.model}${r.temperature != null ? html` · t ${r.temperature}` : nothing}${r.thinking ? ' · thinking' : ''}${r.cache_source && r.cache_source !== 'none' ? ` · ${r.cache_source}` : ''}${r.endpoint === 'completions' ? ' · completions' : ''}${r.stream ? '' : ' · json'}</span>`;
  }

  private prompt(r: LiveRequest): TemplateResult {
    if (r.prompt_tokens == null) return html`—`;
    return html`${exact(r.prompt_tokens)}${r.cached_tokens ? html` <span class="muted">(${exact(r.cached_tokens)} cached)</span>` : nothing}`;
  }

  private tokens(r: LiveRequest, split = true): TemplateResult {
    const s = split ? tokensSplit(r.activity?.decode) : null;
    return html`<b>${exact(r.tokens)}</b>${s ? html`<br /><span class="muted live-split">${s}</span>` : nothing}`;
  }

  private decodeCell(r: LiveRequest): TemplateResult {
    if (r.phase === 'decode') return html`<b class="live-now-num">${fixed(r.decode_tps_now, 1)}</b> / ${fixed(r.decode_tps, 1)}`;
    if (r.phase === 'done') return html`${fixed(r.decode_tps, 1)}`;
    return html`—`;
  }

  private strip(segs: Segment[], bad: boolean, label: string): TemplateResult | typeof nothing {
    if (!segs.length) return nothing;
    return html`<div class="tl ${bad ? 'is-bad' : ''}" aria-hidden="true" title=${label}>
      ${segs.map((s) => html`<span class="tl-seg tone-${s.tone}" style="flex-grow:${Math.max(1, Math.round(s.share * 1000))}">${s.showLabel ? html`<span class="tl-word">${s.word}</span>` : nothing}</span>`)}
    </div>`;
  }

  private timeline(r: LiveRequest, list = false): TemplateResult | typeof nothing {
    if (!r.timeline?.length) return nothing;
    const bad = stopTone(r.activity?.stop).tone === 'bad';
    const items = timelineList(r.timeline, list ? 6 : 16);
    const ol = html`<ol class="${list ? 'tl-list' : 'sr-only'}" aria-label="state transitions">
      ${items.map((i) => html`<li><span class="num">${i.at}</span> ${i.word}${i.detail ? html` <span class="muted">${i.detail}</span>` : nothing}</li>`)}
    </ol>`;
    if (list) return ol;
    return html`${this.strip(timelineSegments(r.timeline, r.elapsed_ms), bad, items.map((i) => `${i.at} ${i.word}`).join(', '))}${ol}`;
  }

  private row(r: LiveRequest, v11: boolean): TemplateResult {
    return html`<tr class="live-row is-${r.phase}" data-id=${r.request_id}>
      <td>${this.cell(r)}</td>
      <td><code class="live-id" title=${r.request_id}>${r.request_id.slice(0, 18)}</code><br />${this.meta(r)}${this.flags(r)}</td>
      <td>${r.client.kind}${r.phase === 'done' ? html`<br /><span class="muted">${agoShort(r.ended_ms_ago)}</span>` : nothing}</td>
      <td class="num-col">${this.prompt(r)}</td>
      <td class="num-col">${this.tokens(r)}</td>
      <td class="num-col">${ms(r.ttft_ms)}</td>
      <td class="num-col">${r.prefill_tps == null ? '—' : compact(r.prefill_tps)}</td>
      <td class="num-col">${this.decodeCell(r)}</td>
      <td class="num-col">${fixed(r.tokens_per_block, 2)}</td>
      <td class="num-col">${elapsed(r.elapsed_ms)}</td>
      ${v11 ? html`<td class="tl-col">${this.timeline(r)}</td>` : nothing}
    </tr>`;
  }

  private card(r: LiveRequest): TemplateResult {
    return html`<article class="live-card is-${r.phase}" data-id=${r.request_id}>
      <header class="live-card-head">
        ${this.cell(r)}
        <span class="muted">${r.client.kind}${r.phase === 'done' ? ` · ${agoShort(r.ended_ms_ago)}` : ''}</span>
      </header>
      <div class="live-card-id"><code class="live-id" title=${r.request_id}>${r.request_id.slice(0, 18)}</code> ${this.meta(r)}</div>
      ${this.flags(r)}
      <dl class="live-card-grid">
        <div><dt>tokens</dt><dd class="num">${this.tokens(r, false)}</dd></div>
        <div><dt>decode now / avg</dt><dd class="num">${this.decodeCell(r)}</dd></div>
        <div><dt>prefill</dt><dd class="num">${r.prefill_tps == null ? '—' : compact(r.prefill_tps) + ' tok/s'}</dd></div>
        <div><dt>prompt</dt><dd class="num">${this.prompt(r)}</dd></div>
        <div><dt>TTFT</dt><dd class="num">${ms(r.ttft_ms)}</dd></div>
        <div><dt>tok/blk · elapsed</dt><dd class="num">${fixed(r.tokens_per_block, 2)} · ${elapsed(r.elapsed_ms)}</dd></div>
      </dl>
      ${tokensSplit(r.activity?.decode) ? html`<div class="muted live-split live-card-split num">${tokensSplit(r.activity?.decode)}</div>` : nothing}
      ${this.timeline(r, true)}
    </article>`;
  }

  // ---- the last 20 requests -----------------------------------------------------------------

  /** One table; on a phone the CSS stacks each row into a block with the sentence first. */
  private recent(rows: LiveRecent[]): TemplateResult {
    return html`<section class="recent" aria-label="last 20 requests">
      <h3 class="recent-title">Last 20 requests <span class="muted">how each one ended</span></h3>
      ${rows.length
        ? html`<div class="recent-list">
            <table class="recent-table">
              <thead>
                <tr>
                  <th>ended</th><th>client</th><th class="tl-col">path</th><th class="num-col">tokens</th><th class="num-col">TTFT</th><th class="num-col">decode</th><th>tools</th><th>stop</th>
                </tr>
              </thead>
              <tbody>
                ${rows.map((x) => {
                  const t = stopTone(x.stop);
                  const open = () => navigate('dev', { request: x.request_id });
                  return html`<tr class="recent-row tone-${t.tone}" data-id=${x.request_id} @click=${open}>
                    <td class="num rc-when">${timeShort(x.ended_at)}</td>
                    <td class="rc-client">${x.client_kind}</td>
                    <td class="tl-col rc-path">${this.strip(pathSegments(x.path), t.tone === 'bad', x.path.join(' → '))}</td>
                    <td class="num-col rc-tokens">${exact(x.tokens)}<span class="rc-unit"> tok</span></td>
                    <td class="num-col rc-ttft"><span class="rc-unit">TTFT </span>${ms(x.ttft_ms)}</td>
                    <td class="num-col rc-decode">${x.decode_tps == null ? '—' : fixed(x.decode_tps, 1)}<span class="rc-unit"> tok/s</span></td>
                    <td class="rc-tools">${x.tool_names.length ? html`<code class="tools">${x.tool_names.join(', ')}</code>` : html`<span class="muted rc-unit">—</span>`}</td>
                    <td class="rc-stop">
                      <button class="recent-open" type="button" title="Open in the Dev tab" @click=${(e: Event) => (e.stopPropagation(), open())}>
                        <span class="stop tone-${t.tone}"><span class="phase-glyph" aria-hidden="true">${t.glyph}</span>${t.word}</span> <span class="recent-sentence">${x.stop.sentence}</span>
                      </button>
                    </td>
                  </tr>`;
                })}
              </tbody>
            </table>
          </div>`
        : html`<p class="recent-empty muted">No request has finished since the engine started.</p>`}
    </section>`;
  }

  // ---- ?debug=live ---------------------------------------------------------------------------------

  private debugOverlay(): TemplateResult {
    const g = this.gaps;
    const max = g.length ? Math.max(...g) : 0;
    return html`<div class="live-debug" aria-label="event gaps">
      <div>live events · last ${g.length} gaps · max ${max} ms</div>
      <div class="live-debug-gaps">${g.map((x) => html`<span class="${x > 2000 ? 'is-bad' : x > 600 ? 'is-warn' : ''}">${x}</span>`)}</div>
    </div>`;
  }
}

export type { LiveSample };
