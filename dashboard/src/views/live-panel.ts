// <qse-live-panel>: what the engine is doing right now, from /v1/dashboard/live
// over a server-sent event stream. Contract 1.1 adds the "Now" line on top (one glyph and one
// sentence for the engine, with its live number and the time in that state, at up to 4 events a
// second), an activity cell and a timeline per request, and the last 20 stops with a sentence
// each. The two figures with five-minute sparklines, the counts and the request list are the;
// a 1.0 server (no `engine`, no `activity`) gets exactly that panel. The head shows the health of
// the stream itself, so a silent page is never mistaken for an idle engine. The stream closes while
// the tab is hidden and reopens when it is visible again.
//
// VIS-40: on a server that sends the live rates (`windows`), both figures and the Now line show the
// speed NOW with each request's warm-up left out -- decode from token 50, prefill after its first
// chunk -- and "—" when nothing runs; the prefill figure never shows a finished request's rate.
// The charts plot those live samples. Finished rows keep their end-to-end numbers (timings), with
// the windowed ones beside them. An older server gets the figures it always had.
//
// VIS-44: on a server whose engine block names the session monitor's mode, the Now line keeps a
// line for the running request's last supervisor action ("Supervisor: closed the reasoning at
// 2,880 tokens (D1, score 0.93)", or what a shadow decision would have done), and every row
// carries a badge -- its actions filled, its shadow decisions outlined -- that opens a drawer with
// each decision from the conductor's journal. Without the mode or the fields nothing of it renders.

import { html, nothing, type TemplateResult } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import { api, describeError } from '../api/client';
import type { Live, LiveRecent, LiveRequest, LiveSample } from '../api/types';
import { activityCell, clientFlag, constrainedWord, continuesText, nowLine, pathSegments, roundWords, secs, stopTone, streamHealth, timelineList, timelineSegments, tokensSplit, type Segment } from '../lib/activity';
import { compact, exact, fixed, ms, timeShort, tps } from '../lib/format';
import { agoShort, decodeFigure, elapsed, emptyRing, hasLiveRates, mostlyCachedWords, orderRows, peak, prefillFigure, prefillingWords, push, rateText, seedRing, slots, sumDecodeNow, warmingWords, type Ring } from '../lib/live';
import { cacheTier, reuse, TIER_LABEL } from '../lib/cache-source';
import { navigate } from '../lib/router';
import { ResumableStream, type StreamState } from '../lib/sse';
import { badge, supervisorLine, type SupervisorLine } from '../lib/supervisor';
import { LightElement } from '../ui/base';
import { emptyState, errorState, icon } from '../ui/bits';
import { supBadge } from '../ui/sup-events';

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
  /** the request whose supervisor drawer is open (VIS-44) */
  @state() private drawer: string | null = null;
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
    const lr = hasLiveRates(l);
    const df = lr ? decodeFigure(l, rows) : null;
    const pf = lr ? prefillFigure(l, rows) : null;
    const decodeNow = df ? df.value : (l.now.decode_tps ?? sumDecodeNow(rows));
    const c = l.counts;
    const nl = nowLine(l.engine, rows, l.windows);
    const has11 = !!l.engine; // contract 1.1: the Now line, activity cells, timelines and the Last 20
    const rounds = has11 ? roundWords(rows) : null;
    return html`<section class="live ${health.stale ? 'is-stale' : ''} ${has11 ? 'is-v11' : ''}" aria-label="live">
      ${head}
      ${nl ? this.nowLine(nl, l.engine?.conductor ? supervisorLine(rows.find((r) => r.request_id === nl.requestId)?.activity?.conductor) : undefined) : nothing}
      <div class="live-now">
        ${this.figure({
          id: 'decode',
          label: 'decode, all requests',
          value: decodeNow,
          unit: 'tok/s',
          sub: df ? df.sub : c.decoding ? `${c.decoding} decoding · last 2 s` : 'nothing decoding',
          caption: df?.caption,
          captionTitle: df?.captionTitle,
          foot: rounds,
          values: lr ? s.decodeLive : s.decode,
          mode: 'ribbon',
          peak: peak(lr ? s.decodeLive : s.decode),
          format: (v: number) => String(Math.round(v)),
        })}
        ${pf
          ? this.figure({
              id: 'prefill',
              label: 'prefill',
              value: pf.value,
              text: rateText,
              unit: 'tok/s',
              sub: pf.sub,
              caption: pf.caption,
              captionTitle: pf.captionTitle,
              foot: null,
              values: s.prefillLive,
              mode: 'ribbon',
              peak: peak(s.prefillLive),
              format: (v: number) => rateText(v),
              busy: l.now.prefilling,
            })
          : this.figure({
              id: 'prefill',
              label: 'prefill',
              value: l.now.prefill_tps,
              unit: 'tok/s',
              sub: l.now.prefilling ? prefillingWords(rows) : l.now.last_prefill_ms_ago == null ? 'no prefill yet' : `last prefill ${agoShort(l.now.last_prefill_ms_ago)}`,
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

  /** `sup`: undefined on a server without the session monitor (no slot); null reserves the line. */
  private nowLine(n: ReturnType<typeof nowLine> & object, sup?: SupervisorLine | null): TemplateResult {
    // Every slot is always there with a fixed height: the headline and its numbers are one
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
        ${sup !== undefined ? html`<div class="now-sup ${sup?.shadow ? 'is-shadow' : ''}" data-sup-line title=${sup?.title ?? ''}>${sup ? html`<span class="now-sup-glyph" aria-hidden="true">◈</span><span class="clip">${sup.text}</span>` : '\u00a0'}</div>` : nothing}
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

  private figure(f: { id: string; label: string; value: number | null; text?: (v: number) => string; unit: string; sub: string; caption?: string | null; captionTitle?: string | null; foot: string | null; values: (number | null)[]; mode: 'ribbon' | 'dots'; peak: number | null; format: (v: number) => string; busy?: boolean }): TemplateResult {
    const na = f.value == null;
    return html`<div class="live-fig ${na ? 'is-na' : ''} ${f.busy ? 'is-busy' : ''}" id="live-${f.id}">
      <div class="live-fig-label">${f.label}</div>
      <div class="live-fig-value"><span class="live-fig-num">${na ? '—' : f.text ? f.text(f.value!) : f.format === compact ? compact(f.value) : fixed(f.value, 1)}</span><span class="live-fig-unit">${f.unit}</span>${f.caption ? html`<span class="live-fig-cap" title=${f.captionTitle ?? ''}>${f.caption}</span>` : nothing}</div>
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
    const b = badge(a?.conductor);
    const mon = b ? `${b.label}${a?.conductor?.gave_up ? ', gave up' : ''}` : null;
    if (!flag && !cont && !tag && !mon) return reserve ? html`<div class="live-flags is-empty" aria-hidden="true"></div>` : nothing;
    return html`<div class="live-flags" title=${[flag, cont, tag, mon, a?.conductor?.last_action ? `last: ${a.conductor.last_action.task} ${a.conductor.last_action.decision} (${a.conductor.last_action.reason ?? ''})` : null].filter(Boolean).join(' · ')}>
      ${flag ? html`<span class="flag flag-bad"><span class="flag-glyph" aria-hidden="true">✗</span><span class="clip">${flag}</span></span>` : nothing}
      ${supBadge(b, this.drawer === r.request_id, () => (this.drawer = this.drawer === r.request_id ? null : r.request_id))}
      ${a?.conductor?.gave_up ? html`<span class="tag" data-monitor-gave-up>gave up</span>` : nothing}
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
    if (this.live?.windows && r.decode_warmup !== undefined) {
      // VIS-40: in flight, the rates from token 50 on; done, the end-to-end number (timings)
      if (r.phase === 'decode') return html`<b class="live-now-num">${fixed(r.decode_live_now_tps, 1)}</b> / ${fixed(r.decode_live_avg_tps, 1)}`;
      if (r.phase === 'done') return html`${fixed(r.decode_tps, 1)}`;
      return html`—`;
    }
    if (r.phase === 'decode') return html`<b class="live-now-num">${fixed(r.decode_tps_now, 1)}</b> / ${fixed(r.decode_tps, 1)}`;
    if (r.phase === 'done') return html`${fixed(r.decode_tps, 1)}`;
    return html`—`;
  }

  /** The prefill figure of a row: the running prefill's now-rate after its warm-up chunk, the
   *  finished one's end-to-end rate (timings), as it always was. */
  private prefillCell(r: LiveRequest): string {
    const p = r.activity?.prefill;
    if (this.live?.windows && r.phase === 'prefill' && p && p.tps_live !== undefined) return rateText(p.tps_live);
    return r.prefill_tps == null ? '—' : compact(r.prefill_tps);
  }

  /** VIS-40: the line under a row's rate and its tooltip, which holds both numbers side by side:
   *  the windowed one (from token 50, after the first chunk) and the end-to-end one (timings). */
  private rateNotes(r: LiveRequest): { decode: [string, string]; prefill: [string, string] } | null {
    const w = this.live?.windows;
    if (!w || r.decode_warmup === undefined) return null;
    const blank: [string, string] = ['\u00a0', ''];
    const from = `from token ${w.decode_warmup_tokens}`;
    const after = w.prefill_warmup_chunks === 0 ? 'from the start' : w.prefill_warmup_chunks === 1 ? 'after the first chunk' : `after ${w.prefill_warmup_chunks} chunks`;
    const fromChunk = `from chunk ${w.prefill_warmup_chunks + 1}`;
    let decode = blank;
    if (r.phase === 'decode' && r.decode_warmup) decode = r.decode_warmup.reached ? [from, `now over the last ${w.now_window_s} s / average, ${from}`] : [warmingWords(r, w), `the live rate starts at decoded token ${w.decode_warmup_tokens}`];
    else if (r.phase === 'done' && r.decode_tps != null) decode = [r.decode_live_avg_tps != null ? `${from}: ${fixed(r.decode_live_avg_tps, 1)}` : '\u00a0', `decode ${from}: ${tps(r.decode_live_avg_tps)} · decode end to end (timings): ${tps(r.decode_tps)}`];
    const p = r.activity?.prefill;
    const mostly = mostlyCachedWords(r);
    let prefill = blank;
    if (r.phase === 'prefill' && p) {
      if (p.progress === 'single_call') prefill = ['one call', 'a single-call prefill reports no progress, so it has no live rate'];
      else if (p.tps_live == null) prefill = ['warming up', `the live rate starts ${after}`];
      else prefill = [fromChunk, `now over the last ${w.now_window_s} s of chunks, ${after}`];
    } else if (r.phase !== 'queued' && r.prefill_tps != null) {
      prefill = [p?.tps_live_avg != null ? `${fromChunk}: ${compact(p.tps_live_avg)}` : '\u00a0', `prefill ${after}: ${tps(p?.tps_live_avg)} · prefill end to end (timings): ${tps(r.prefill_tps)}`];
    }
    if (mostly) prefill = ['mostly cached', prefill[1] ? `${mostly} · ${prefill[1]}` : mostly];
    return { decode, prefill };
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
    // every cell is a fixed stack of one-line slots: a line that comes and goes (the age, the
    // cached share, the token split, the flags) is always there, blank when it has nothing to say
    const notes = this.rateNotes(r);
    return html`<tr class="live-row is-${r.phase}" data-id=${r.request_id}>
      <td class="c-act">${this.cell(r)}</td>
      <td class="c-req"><span class="clip"><code class="live-id" title=${r.request_id}>${r.request_id.slice(0, 18)}</code></span>${this.meta(r)}${this.flags(r, v11)}</td>
      <td class="c-client"><span class="clip" title=${r.client.kind}>${r.client.kind}</span><span class="clip muted live-sub">${r.phase === 'done' ? agoShort(r.ended_ms_ago) : '\u00a0'}</span></td>
      <td class="num-col c-prompt">${this.promptCell(r)}</td>
      <td class="num-col c-tokens">${this.tokens(r)}</td>
      <td class="num-col c-ttft"><span class="clip">${ms(r.ttft_ms)}</span></td>
      <td class="num-col c-prefill"><span class="clip">${this.prefillCell(r)}</span>${notes ? html`<span class="clip muted live-sub live-rate-note" title=${notes.prefill[1]}>${notes.prefill[0]}</span>` : nothing}</td>
      <td class="num-col c-decode"><span class="clip">${this.decodeCell(r)}</span>${notes ? html`<span class="clip muted live-sub live-rate-note" title=${notes.decode[1]}>${notes.decode[0]}</span>` : nothing}</td>
      <td class="num-col c-tpb"><span class="clip">${fixed(r.tokens_per_block, 2)}</span></td>
      <td class="num-col c-elapsed"><span class="clip">${elapsed(r.elapsed_ms)}</span></td>
      ${v11 ? html`<td class="tl-col c-tl">${this.timeline(r)}</td>` : nothing}
    </tr>
    ${this.drawer === r.request_id ? html`<tr class="live-drawer" data-drawer=${r.request_id}><td colspan=${v11 ? 11 : 10}>${this.drawerBody(r)}</td></tr>` : nothing}`;
  }

  /** The supervisor's decisions on this request (VIS-44), read from the journal. */
  private drawerBody(r: LiveRequest): TemplateResult {
    const c = r.activity?.conductor;
    return html`<qse-sup-events .requestId=${r.request_id} .count=${(c?.acts ?? 0) + (c?.shadow ?? 0)}></qse-sup-events>`;
  }

  private card(r: LiveRequest, v11 = false): TemplateResult {
    const notes = this.rateNotes(r);
    return html`<article class="live-card is-${r.phase}" data-id=${r.request_id}>
      <header class="live-card-head">
        ${this.cell(r)}
        <span class="muted live-card-client">${r.client.kind}${r.phase === 'done' ? ` · ${agoShort(r.ended_ms_ago)}` : ''}</span>
      </header>
      <div class="live-card-id"><code class="live-id" title=${r.request_id}>${r.request_id.slice(0, 18)}</code> ${this.meta(r)}</div>
      ${this.flags(r, v11)}
      <dl class="live-card-grid">
        <div><dt>tokens</dt><dd class="num">${this.tokens(r, false)}</dd></div>
        <div><dt>decode now / avg</dt><dd class="num" title=${notes?.decode[1] || notes?.decode[0].trim() || ''}>${this.decodeCell(r)}</dd></div>
        <div><dt>prefill</dt><dd class="num" title=${notes?.prefill[1] || notes?.prefill[0].trim() || ''}>${this.prefillCell(r) === '—' ? '—' : this.prefillCell(r) + ' tok/s'}${notes && notes.prefill[0] === 'mostly cached' ? html` <span class="muted live-rate-note">· mostly cached</span>` : nothing}</dd></div>
        <div><dt>prompt</dt><dd class="num">${this.prompt(r)}</dd></div>
        <div><dt>TTFT</dt><dd class="num">${ms(r.ttft_ms)}</dd></div>
        <div><dt>tok/blk · elapsed</dt><dd class="num">${fixed(r.tokens_per_block, 2)} · ${elapsed(r.elapsed_ms)}</dd></div>
      </dl>
      ${v11 || tokensSplit(r.activity?.decode) ? html`<div class="muted live-split live-card-split num">${tokensSplit(r.activity?.decode) ?? '\u00a0'}</div>` : nothing}
      ${this.timeline(r, true)}
      ${this.drawer === r.request_id ? html`<div class="live-card-drawer" data-drawer=${r.request_id}>${this.drawerBody(r)}</div>` : nothing}
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
                  <th class="rc-when">ended</th><th class="rc-client">client</th><th class="tl-col rc-path">path</th><th class="num-col rc-tokens">tokens</th><th class="rc-cache">cache</th><th class="num-col rc-ttft">TTFT</th><th class="num-col rc-decode">decode</th><th class="rc-tools">tools</th><th class="rc-stop">stop</th>
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
                    <td class="rc-cache">${this.cacheCell(x)}</td>
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

  /** Where the prompt came from (RAM resident, the end anchor, the NVMe, cold) and what it reused
   * and forwarded; empty for a row from a server before 0.3.21-test, which sends neither. */
  private cacheCell(x: LiveRecent): TemplateResult {
    if (x.prompt_tokens == null || x.cache_source === undefined) return html`<span class="muted rc-unit">—</span>`;
    const tier = cacheTier(x.cache_source);
    const u = reuse(x.prompt_tokens, x.cached_tokens, x.forwarded_tokens);
    const title = `${TIER_LABEL[tier]}: ${exact(u.reused)} of ${exact(x.prompt_tokens)} prompt tokens reused, ${exact(u.forwarded)} forwarded`;
    return html`<span class="cache-tag tier-${tier}" title=${title}>${TIER_LABEL[tier]}</span><span class="cache-split muted num" title=${title}>${compact(u.reused)} reused · ${compact(u.forwarded)} new</span>`;
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
