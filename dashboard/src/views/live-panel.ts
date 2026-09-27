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
    // Every slot is always there with a fixed height (VIS-30): the headline and its numbers are one
    // line each, cut with an ellipsis (the whole text is the tooltip); the warning line and the
    // progress bar sit in a reserved strip, so a state change never moves the page under the card.
    const numbers = n.numbers.join(' · ');
    return html`<div class="now tone-${n.tone} ${n.moving ? 'is-moving' : ''}" id="live-now" role="status" aria-live="polite" aria-atomic="true">
      <span class="now-glyph" aria-hidden="true">${n.glyph}</span>
      <div class="now-main">
        <div class="now-line" title=${numbers ? `${n.headline} · ${numbers}` : n.headline}>
          <span class="now-headline">${n.headline}</span>
          <span class="now-numbers num">${numbers || '\u00a0'}</span>
        </div>
        <div class="now-sub">
          <div class="now-warning" title=${n.warning ?? ''}>${n.warning ? html`<span class="flag-glyph" aria-hidden="true">✗</span><span class="clip">${n.warning}</span>` : nothing}</div>
          ${n.progress != null ? html`<div class="now-bar" role="progressbar" aria-label="prefill" aria-valuemin="0" aria-valuemax="100" aria-valuenow=${Math.round(n.progress * 100)}><div class="now-bar-fill" style="width:${(n.progress * 100).toFixed(1)}%"></div></div>` : nothing}
        </div>
      </div>
      <div class="now-side">
        <span class="now-since num" title="time in this state">${n.sinceMs != null ? secs(n.sinceMs) : '\u00a0'}</span>
        <span class="now-client">${n.client ?? '\u00a0'}</span>
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
      <div class="live-fig-sub" title=${f.sub}>${f.busy ? html`<span class="pulse"></span>` : nothing}<span class="clip">${f.sub}</span></div>
      <div class="live-fig-rounds num" title=${f.foot ?? ''}>${f.foot ?? nothing}</div>
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
      <table class="live-table ${v11 ? 'is-v11' : ''}">
        <thead>
          <tr>
            <th class="c-act">${v11 ? 'activity' : 'phase'}</th><th class="c-req">request</th><th class="c-client">client</th><th class="num-col c-prompt">prompt</th><th class="num-col c-tokens">tokens</th><th class="num-col c-ttft">TTFT</th><th class="num-col c-prefill">prefill</th><th class="num-col c-decode" title="decode now / avg">decode now / avg</th><th class="num-col c-tpb">tok/blk</th><th class="num-col c-elapsed">elapsed</th>${v11 ? html`<th class="tl-col c-tl">timeline</th>` : nothing}
          </tr>
        </thead>
        <tbody>${rows.map((r) => this.row(r, v11))}</tbody>
      </table>
      <div class="live-cards">${rows.map((r) => this.card(r, v11))}</div>
    </div>`;
  }

  private cell(r: LiveRequest): TemplateResult {
    const c = activityCell(r.activity, r);
    return html`<span class="phase act tone-${c.tone} ${c.moving ? 'is-moving' : ''}" title=${c.detail ? `${c.word} ${c.detail}` : c.word}><span class="phase-glyph" aria-hidden="true">${c.glyph}</span><span class="clip">${c.word}${c.detail ? html` <span class="act-detail num">${c.detail}</span>` : nothing}</span></span>`;
  }

  /** One line, cut with an ellipsis (the whole text is the tooltip). With `reserve` (a 1.1 list) the
   *  line is there even when empty, so a row or a card keeps its height when a flag comes or goes. */
  private flags(r: LiveRequest, reserve = false): TemplateResult | typeof nothing {
    const a = r.activity;
    const flag = a && r.phase !== 'done' ? clientFlag(a) : null;
    const cont = a ? continuesText(a.continues) : null;
    const tag = a?.constrained ? constrainedWord(a.constrained) : null;
    if (!flag && !cont && !tag) return reserve ? html`<div class="live-flags is-empty" aria-hidden="true"></div>` : nothing;
    return html`<div class="live-flags" title=${[flag, cont, tag].filter(Boolean).join(' · ')}>
      ${flag ? html`<span class="flag flag-bad"><span class="flag-glyph" aria-hidden="true">✗</span><span class="clip">${flag}</span></span>` : nothing}
      ${tag ? html`<span class="tag">${tag}</span>` : nothing}
      ${cont ? html`<span class="flag flag-continues"><span class="flag-glyph" aria-hidden="true">↩</span><span class="clip2">${cont}</span></span>` : nothing}
    </div>`;
  }

  private meta(r: LiveRequest): TemplateResult {
    return html`<span class="live-meta clip">${r.model}${r.temperature != null ? html` · t ${r.temperature}` : nothing}${r.thinking ? ' · thinking' : ''}${r.cache_source && r.cache_source !== 'none' ? ` · ${r.cache_source}` : ''}${r.endpoint === 'completions' ? ' · completions' : ''}${r.stream ? '' : ' · json'}</span>`;
  }

  /** The card's one-line form: the prompt, then the cached share in brackets. */
  private prompt(r: LiveRequest): TemplateResult {
    if (r.prompt_tokens == null) return html`—`;
    return html`${exact(r.prompt_tokens)}${r.cached_tokens ? html` <span class="muted">(${exact(r.cached_tokens)} cached)</span>` : nothing}`;
  }

  /** The table's form: the prompt on the first line, the cached share under it (a reserved line). */
  private promptCell(r: LiveRequest): TemplateResult {
    const cached = r.cached_tokens ? `${exact(r.cached_tokens)} cached` : null;
    return html`<span class="clip">${r.prompt_tokens == null ? '—' : exact(r.prompt_tokens)}</span><span class="clip muted live-sub" title=${cached ?? ''}>${cached ?? '\u00a0'}</span>`;
  }

  private tokens(r: LiveRequest, split = true): TemplateResult {
    if (!split) return html`<b>${exact(r.tokens)}</b>`;
    const s = tokensSplit(r.activity?.decode);
    return html`<b class="clip">${exact(r.tokens)}</b><span class="clip2 muted live-split" title=${s ?? ''}>${s ?? '\u00a0'}</span>`;
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
      ${items.map((i) => html`<li title=${list ? `${i.at} ${i.word}${i.detail ? ` ${i.detail}` : ''}` : ''}><span class="num">${i.at}</span> ${i.word}${i.detail ? html` <span class="muted">${i.detail}</span>` : nothing}</li>`)}
    </ol>`;
    if (list) return ol;
    return html`${this.strip(timelineSegments(r.timeline, r.elapsed_ms), bad, items.map((i) => `${i.at} ${i.word}`).join(', '))}${ol}`;
  }

  private row(r: LiveRequest, v11: boolean): TemplateResult {
    // every cell is a fixed stack of one-line slots (VIS-30): a line that comes and goes (the age, the
    // cached share, the token split, the flags) is always there, blank when it has nothing to say
    return html`<tr class="live-row is-${r.phase}" data-id=${r.request_id}>
      <td class="c-act">${this.cell(r)}</td>
      <td class="c-req"><span class="clip"><code class="live-id" title=${r.request_id}>${r.request_id.slice(0, 18)}</code></span>${this.meta(r)}${this.flags(r, v11)}</td>
      <td class="c-client"><span class="clip" title=${r.client.kind}>${r.client.kind}</span><span class="clip muted live-sub">${r.phase === 'done' ? agoShort(r.ended_ms_ago) : '\u00a0'}</span></td>
      <td class="num-col c-prompt">${this.promptCell(r)}</td>
      <td class="num-col c-tokens">${this.tokens(r)}</td>
      <td class="num-col c-ttft"><span class="clip">${ms(r.ttft_ms)}</span></td>
      <td class="num-col c-prefill"><span class="clip">${r.prefill_tps == null ? '—' : compact(r.prefill_tps)}</span></td>
      <td class="num-col c-decode"><span class="clip">${this.decodeCell(r)}</span></td>
      <td class="num-col c-tpb"><span class="clip">${fixed(r.tokens_per_block, 2)}</span></td>
      <td class="num-col c-elapsed"><span class="clip">${elapsed(r.elapsed_ms)}</span></td>
      ${v11 ? html`<td class="tl-col c-tl">${this.timeline(r)}</td>` : nothing}
    </tr>`;
  }

  private card(r: LiveRequest, v11 = false): TemplateResult {
    return html`<article class="live-card is-${r.phase}" data-id=${r.request_id}>
      <header class="live-card-head">
        ${this.cell(r)}
        <span class="muted live-card-client">${r.client.kind}${r.phase === 'done' ? ` · ${agoShort(r.ended_ms_ago)}` : ''}</span>
      </header>
      <div class="live-card-id"><code class="live-id" title=${r.request_id}>${r.request_id.slice(0, 18)}</code> ${this.meta(r)}</div>
      ${this.flags(r, v11)}
      <dl class="live-card-grid">
        <div><dt>tokens</dt><dd class="num">${this.tokens(r, false)}</dd></div>
        <div><dt>decode now / avg</dt><dd class="num">${this.decodeCell(r)}</dd></div>
        <div><dt>prefill</dt><dd class="num">${r.prefill_tps == null ? '—' : compact(r.prefill_tps) + ' tok/s'}</dd></div>
        <div><dt>prompt</dt><dd class="num">${this.prompt(r)}</dd></div>
        <div><dt>TTFT</dt><dd class="num">${ms(r.ttft_ms)}</dd></div>
        <div><dt>tok/blk · elapsed</dt><dd class="num">${fixed(r.tokens_per_block, 2)} · ${elapsed(r.elapsed_ms)}</dd></div>
      </dl>
      ${v11 || tokensSplit(r.activity?.decode) ? html`<div class="muted live-split live-card-split num">${tokensSplit(r.activity?.decode) ?? '\u00a0'}</div>` : nothing}
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
                  <th class="rc-when">ended</th><th class="rc-client">client</th><th class="tl-col rc-path">path</th><th class="num-col rc-tokens">tokens</th><th class="num-col rc-ttft">TTFT</th><th class="num-col rc-decode">decode</th><th class="rc-tools">tools</th><th class="rc-stop">stop</th>
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
                    <td class="rc-tools" title=${x.tool_names.join(', ')}>${x.tool_names.length ? html`<code class="tools">${x.tool_names.join(', ')}</code>` : html`<span class="muted rc-unit">—</span>`}</td>
                    <td class="rc-stop">
                      <button class="recent-open" type="button" title="${x.stop.sentence} (open in the Dev tab)" @click=${(e: Event) => (e.stopPropagation(), open())}>
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
