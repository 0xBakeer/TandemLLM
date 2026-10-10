// Performance → Parallel: every request in flight on every backend at once (our engine, vLLM,
// SGLang, cloud APIs, the fake backend), one lane each, and the sums: the aggregate decode rate
// stacked by backend over five minutes, requests in flight per backend, TTFT p50/p90, the cache
// hit, the cost per minute. One poll a second of `/v1/dashboard/parallel` (the server builds the
// document once for every viewer; a poll carries only the series' new samples).
//
// The numbers are the server's (server/parallel.py), never estimated here: a figure a backend does
// not give reads "not reported", a relayed request's prompt and cache read "at the end" until the
// usage arrives. Rows keep their place (lib/parallel.ts Slots): a request holds its lane from its
// arrival until its "done" fades, a new one takes the first free lane.

import { html, nothing, type TemplateResult } from 'lit';
import { customElement, state } from 'lit/decorators.js';
import { repeat } from 'lit/directives/repeat.js';
import { api, describeError } from '../api/client';
import { colorVar } from '../charts/chart-base';
import '../charts/stack';
import { compact, exact, money, ms, pct } from '../lib/format';
import { barScale, backendOrder, Colors, inFlightWords, jointWords, kindLabel, mergeSeries, Slots, slotWord, sourceWord, stallLine, stateLine, stateTone, tokenBar, windowWord, type BackendRow, type EngineSlots, type Lane, type Parallel, type Sample, type Slot } from '../lib/parallel';
import { poll, type Poller } from '../lib/poll';
import { LightElement } from '../ui/base';
import { errorState, skeleton } from '../ui/bits';

const NR = 'not reported';
const TONES = ['wait', 'prefill', 'think', 'tool', 'write'] as const;
const TONE_WORD: Record<(typeof TONES)[number], string> = { wait: 'waiting', prefill: 'prefilling', think: 'thinking', tool: 'tool call', write: 'writing' };

function rate(v: number | null | undefined): string {
  if (v == null) return '—';
  return v >= 100 ? Math.round(v).toLocaleString('en-US') : v.toFixed(1);
}

@customElement('qse-parallel')
export class QseParallel extends LightElement {
  @state() private doc: Parallel | null = null;
  @state() private series: Sample[] = [];
  @state() private error: { endpoint: string; message: string } | null = null;
  @state() private bytes = 0;
  private slots = new Slots();
  private colors = new Colors();
  private scale = 0;
  private span = 300;
  private poller: Poller | null = null;

  connectedCallback(): void {
    super.connectedCallback();
    this.poller = poll(() => this.load(), 1000);
    this.poller.start();
  }
  disconnectedCallback(): void {
    this.poller?.stop();
    super.disconnectedCallback();
  }

  private async load(): Promise<void> {
    try {
      const since = this.series.length ? this.series[this.series.length - 1].t : null;
      const d = await api.parallel(since);
      this.bytes = JSON.stringify(d).length;
      this.series = mergeSeries(this.series, d.series ?? []);
      this.colors.seed(backendOrder(this.series, d.backends));
      this.slots.update(d.lanes, d.recent, Date.now());
      this.doc = d;
      this.error = null;
    } catch (e) {
      const x = describeError(e);
      this.error = { endpoint: x.endpoint || '/v1/dashboard/parallel', message: x.message };
    }
  }

  render() {
    const d = this.doc;
    if (!d) {
      return html`<div class="par">${this.error ? errorState(this.error.endpoint, this.error.message, () => this.poller?.refresh()) : skeleton(6, 'skeleton-live')}</div>`;
    }
    const order = backendOrder(this.series, d.backends);
    // the chart's window: five minutes once there is that much, else what there is (at least one)
    const first = this.series.length ? this.series[0].t : d.at;
    this.span = Math.min(300, Math.max(60, Math.ceil((d.at - first + 5) / 30) * 30));
    return html`<div class="par" data-in-flight=${d.totals.in_flight}>
      ${this.renderHero(d, order)}
      ${this.renderKpis(d)}
      ${this.renderBackends(d)}
      ${this.renderLanes(d)}
      ${this.error ? html`<p class="par-foot warn-ink" role="status">${this.error.message} (the last numbers stay until the next poll answers)</p>` : nothing}
    </div>`;
  }

  // ---- the aggregate: the big number, the state mix, the stacked chart -------------------------
  private renderHero(d: Parallel, order: string[]): TemplateResult {
    const t = d.totals;
    const live = d.lanes;
    const counts = Object.fromEntries(TONES.map((k) => [k, live.filter((l) => stateTone(l) === k).length])) as Record<(typeof TONES)[number], number>;
    const byId = new Map(d.backends.map((b) => [b.id, b]));
    const stall = stallLine(t, live, d.windows.now_s);
    return html`<section class="panel par-hero" aria-label="aggregate decode">
      <div class="par-hero-num">
        <div class="par-kicker">Aggregate decode, now</div>
        <div class="par-big ${t.decode_tps == null ? 'is-na' : ''}" data-total-tps>
          <span class="num">${t.decode_tps == null ? (t.decoding ? NR : '0') : rate(t.decode_tps)}</span><span class="par-big-unit">tok/s</span>
        </div>
        <div class="par-hero-sub">
          <span><b class="num">${t.in_flight}</b> in flight</span>
          <span><b class="num">${t.decoding}</b> decoding</span>
          <span><b class="num">${t.backends}</b> backend${t.backends === 1 ? '' : 's'}</span>
          <span class="par-nr ${t.decode_unreported ? '' : 'is-hidden'}" title="no live count for these (server/parallel.py): the usage at the end has the exact one">${t.decode_unreported} not counted live</span>
          <span class=${t.prefill_tps != null ? '' : 'is-hidden'}>prefill <b class="num">${rate(t.prefill_tps)}</b> tok/s</span>
        </div>
        <div class="par-stall warn-ink" data-stall role="status" title="no decoding lane wrote a token over the last ${d.windows.now_s} s: the rates count tokens, so this 0 is a real stall">${stall ?? '\u00a0'}</div>
        <div class="par-mix" role="img" aria-label="requests in flight by state">
          <div class="par-mix-bar">
            ${TONES.map((k) => html`<span class="par-mix-seg tone-${k}" style="flex-grow:${counts[k]};flex-basis:0;${counts[k] ? '' : 'padding:0;'}" title="${counts[k]} ${TONE_WORD[k]}"></span>`)}
            ${live.length ? nothing : html`<span class="par-mix-seg tone-idle" style="flex-grow:1"></span>`}
          </div>
          <div class="par-mix-legend">
            ${TONES.map((k) => html`<span class="par-mix-item ${counts[k] ? '' : 'is-zero'}" data-tone=${k}><span class="par-dot tone-${k}"></span>${TONE_WORD[k]} <b class="num">${counts[k]}</b></span>`)}
          </div>
        </div>
      </div>
      <div class="par-hero-chart">
        <qse-stack .series=${this.series} .order=${order} .color=${(id: string) => this.colors.of(id)} .label=${(id: string) => id} .now=${d.at} .span=${this.span} height="196"></qse-stack>
        <div class="par-legend">
          ${order.map((id) => {
            const b = byId.get(id);
            return html`<span class="par-legend-item" data-backend=${id}><span class="swatch" style="background:${colorVar(this.colors.of(id))}"></span>${id}<span class="num">${b?.decode_tps != null ? `${rate(b.decode_tps)} tok/s` : b?.decoding ? NR : b?.in_flight ? 'waiting' : 'idle'}</span></span>`;
          })}
        </div>
      </div>
    </section>`;
  }

  private renderKpis(d: Parallel): TemplateResult {
    const t = d.totals;
    const min = Math.round(d.windows.stats_s / 60);
    const tile = (label: string, value: TemplateResult | string, sub: TemplateResult | string, key: string, na = false) =>
      html`<div class="stat ${na ? 'is-na' : ''}" data-kpi=${key}><div class="stat-label">${label}</div><div class="stat-value">${value}</div><div class="stat-sub">${sub}</div></div>`;
    return html`<div class="stats par-kpis">
      ${tile('In flight', html`<span class="num">${t.in_flight}</span>`, inFlightWords(d.lanes, t.decoding), 'in-flight')}
      ${tile(
        'TTFT p50 · p90',
        t.ttft_p50 == null ? 'none yet' : html`<span class="num">${ms(t.ttft_p50)}</span><span class="stat-unit">· ${ms(t.ttft_p90)}</span>`,
        `${t.ttft_n} requests, last ${min} min`,
        'ttft',
        t.ttft_p50 == null,
      )}
      ${tile(
        'Cache hit',
        t.cache_hit == null ? NR : html`<span class="num">${pct(t.cache_hit, 1)}</span>`,
        t.cache_hit == null ? `no cached-token counts in the last ${min} min` : html`${compact(t.cached_tokens_window)} of ${compact(t.prompt_tokens_window)} prompt tokens · ${t.cache_reported} requests`,
        'cache',
        t.cache_hit == null,
      )}
      ${tile(
        'Cost per minute',
        t.cost_per_min == null ? 'none' : html`<span class="num">${money(t.cost_per_min)}</span>`,
        html`${money(t.cost_window)} in ${min} min${t.cost_in_flight != null ? html` · <span title="completion tokens so far at the output price; the prompt's part comes with the usage">${money(t.cost_in_flight)} running</span>` : nothing}`,
        'cost',
        t.cost_per_min == null,
      )}
      ${tile('Tokens out', html`<span class="num">${compact(t.tokens_window)}</span>`, `${t.finished} requests finished, last ${min} min`, 'tokens')}
      ${tile('Errors', html`<span class="num">${t.errors}</span>`, `${t.refused} refused, last ${min} min`, 'errors')}
    </div>`;
  }

  // ---- per backend ---------------------------------------------------------------------------------
  private renderBackends(d: Parallel): TemplateResult | typeof nothing {
    if (!d.backends.length) return nothing;
    const byBackend = new Map<string, Lane[]>();
    for (const l of d.lanes) byBackend.set(l.backend, [...(byBackend.get(l.backend) ?? []), l]);
    return html`<div class="par-backends">
      ${repeat(
        d.backends,
        (b) => b.id,
        (b) => this.renderBackend(b, byBackend.get(b.id) ?? [], d),
      )}
    </div>`;
  }

  private renderBackend(b: BackendRow, lanes: Lane[], d: Parallel): TemplateResult {
    const col = colorVar(this.colors.of(b.id));
    const fr = b.frames;
    const frames = b.engine_view
      ? "the engine's own count, read from its board"
      : !fr
        ? b.kind === 'tandemllm' && b.id === 'engine'
          ? 'tokens counted by the engine'
          : 'frames not checked yet'
        : fr.verified
          ? fr.tools_verified === false
            ? `text frames = tokens (checked); tool-call frames carry several`
            : `frames = tokens (${fr.passed}/${fr.checked} answers checked)`
          : fr.passed === fr.checked
            ? `checking frames against the usage: ${fr.passed} of 2 answers so far`
            : `frames ≠ tokens (${fr.checked - fr.passed} of ${fr.checked} answers): no live count`;
    const min = Math.round(d.windows.stats_s / 60);
    return html`<article class="par-bk" data-backend=${b.id} style="--bk:${col}">
      <header class="par-bk-head">
        <span class="swatch" style="background:${col}"></span>
        <span class="par-bk-id">${b.id}</span>
        <span class="par-kind">${kindLabel(b.kind)}</span>
      </header>
      <div class="par-bk-models" title=${b.models.join(', ')}>${b.models.join(', ') || html`<span class="ink-3">idle</span>`}</div>
      <div class="par-bk-rate ${b.decode_tps == null && b.decoding ? 'is-na' : ''}">
        <span class="num">${b.decode_tps != null ? rate(b.decode_tps) : b.decoding ? NR : '0'}</span><span class="par-bk-unit">tok/s</span>
      </div>
      <div class="par-bk-dots" aria-label="${b.in_flight} in flight">
        <span class="par-bk-dotrow">${lanes.map((l) => html`<span class="par-dot tone-${stateTone(l)}" title="${stateLine(l)}"></span>`)}</span>
        <span class="par-bk-count">${b.in_flight} in flight${b.waiting ? ` · ${b.waiting} waiting` : ''}</span>
      </div>
      <dl class="par-bk-figs">
        <div><dt>TTFT p50 · p90</dt><dd class="num">${b.ttft_p50 == null ? '—' : `${ms(b.ttft_p50)} · ${ms(b.ttft_p90)}`}</dd></div>
        <div><dt>cache hit</dt><dd class="num ${b.cache_hit == null ? 'par-nr' : ''}">${b.cache_hit == null ? (b.finished ? NR : '—') : pct(b.cache_hit, 1)}</dd></div>
        <div><dt>cost / min</dt><dd class="num">${b.cost_per_min == null ? (b.cost_window == null ? 'no price' : '$0') : money(b.cost_per_min)}</dd></div>
        <div><dt>done, ${min} min</dt><dd class="num">${b.finished}${b.errors ? html` <span class="err-ink">· ${b.errors} err</span>` : nothing}</dd></div>
      </dl>
      <div class="par-bk-frames" title="how live tokens are counted for this backend">${frames}</div>
      ${b.id === 'engine' && d.engine_slots ? this.renderSlots(d.engine_slots) : nothing}
    </article>`;
  }

  /** The engine's sequence slots (`--parallel N`): one chip a slot, its window and what it does,
   *  and how many verify passes were shared. Shown only when the engine has side slots. */
  private renderSlots(es: EngineSlots): TemplateResult {
    return html`<div class="par-slots" data-slots=${es.n} aria-label="${es.n} sequence slots, ${es.running} busy">
      <div class="par-slots-row">
        ${es.slots.map(
          (s) => html`<span class="par-slot ${s.busy ? 'is-busy' : ''} ${s.phase === 'decode' ? 'is-decode' : ''}" data-slot=${s.index} title="slot ${s.index}: window ${exact(s.window)} tokens">
            <span class="par-slot-i num">${s.index}</span><span class="par-slot-w">${windowWord(s.window)}</span><span class="par-slot-s">${slotWord(s)}</span>
          </span>`,
        )}
      </div>
      <div class="par-slots-foot" title="verify passes the decoding requests shared (one read of the weights for all of them), by how many blocks were in the pass">
        shared passes: <span class="num">${jointWords(es.joint_hist)}</span>${es.admission && es.admission.waits ? html` · <span title="requests that waited for memory (admission by bytes)">${es.admission.waits} waited for memory</span>` : nothing}
      </div>
    </div>`;
  }

  // ---- the lanes -------------------------------------------------------------------------------------
  private renderLanes(d: Parallel): TemplateResult {
    const slots = this.slots.slots;
    const lanes = slots.map((s) => s.lane).filter((l): l is Lane => l != null);
    this.scale = barScale(lanes, this.scale);
    return html`<section class="panel par-lanes" aria-label="requests in flight">
      <header class="panel-head">
        <div class="panel-titles">
          <h2 class="panel-title">Lanes <span class="par-count num">${d.totals.in_flight}</span></h2>
          <p class="panel-sub">one row per request, kept in place; a finished one stays 3 s. Bars share one scale (${exact(this.scale)} tokens); light: thinking</p>
        </div>
      </header>
      <div class="par-table" role="table">
        <div class="par-row par-row-head" role="row">
          <span role="columnheader">backend · model</span><span role="columnheader">state</span><span role="columnheader">tokens</span>
          <span role="columnheader" class="r">now</span><span role="columnheader" class="r">avg</span><span role="columnheader" class="r">TTFT</span>
          <span role="columnheader" class="r">prefill</span><span role="columnheader" class="r">cache</span><span role="columnheader" class="r">cost</span><span role="columnheader" class="r">elapsed</span>
        </div>
        ${slots.length ? nothing : html`<div class="par-empty">Nothing in flight. Requests appear here the moment they arrive, on every backend.</div>`}
        ${repeat(
          slots.map((s, i) => [s, i] as [Slot, number]),
          ([, i]) => i,
          ([s]) => (s.lane ? this.renderLane(s.lane) : html`<div class="par-row par-row-free" role="row" aria-hidden="true"><span class="par-free">free</span></div>`),
        )}
      </div>
      <p class="par-foot">
        Server: ${d.build_us ? html`built in ${(d.build_us.last / 1000).toFixed(2)} ms (max ${(d.build_us.max / 1000).toFixed(1)} ms)` : nothing} · ${(this.bytes / 1024).toFixed(1)} KB a poll · one poll a second.
        Live tokens are the engine's own count, or a backend's stream frames once they were checked against its usage; the rest is "${NR}" until the usage arrives.
      </p>
    </section>`;
  }

  private renderLane(l: Lane): TemplateResult {
    const tone = stateTone(l);
    const col = colorVar(this.colors.of(l.backend));
    const bar = tokenBar(l, this.scale);
    const done = l.state === 'done';
    const who = l.client.session ?? l.client.user ?? l.client.kind;
    const live = !done;
    const nowCell = l.decode_now != null ? rate(l.decode_now) : live && (tone === 'think' || tone === 'write' || tone === 'tool') ? NR : '—';
    const avgCell = l.decode_avg != null ? rate(l.decode_avg) : l.tokens == null && (live || l.tokens_source == null) && l.frames ? NR : '—';
    let ttft: TemplateResult | string = l.ttft_ms != null ? ms(l.ttft_ms) : '—';
    if (l.ttft_ms == null && (l.state === 'waiting' || l.state === 'waiting_memory' || l.state === 'queued' || l.state === 'prefilling') && l.waiting_ms != null) ttft = html`<span class="par-wait">${ms(l.waiting_ms)}…</span>`;
    let prefill: TemplateResult | string = '—';
    const ptps = l.prefill?.tps ?? l.prefill?.tps_avg;
    if (ptps != null) prefill = html`<span title=${l.prefill?.tps != null ? 'tokens a second, now' : 'tokens a second over the whole prefill'}>${rate(ptps)}</span>`;
    else if (!l.prefill) prefill = html`<span class="par-nr" title="the proxy sees the time to the first token, not the backend's prefill">${NR}</span>`;
    let cache: TemplateResult | string;
    if (l.cache_hit != null) cache = html`<span title="${exact(l.cached_tokens)} of ${exact(l.prompt_tokens)} prompt tokens cached">${pct(l.cache_hit, 0)}</span>`;
    else if (live && l.backend !== 'engine') cache = html`<span class="par-nr" title="cached tokens come with the usage at the end">at the end</span>`;
    else if (done && l.prompt_tokens != null && l.cached_tokens == null) cache = html`<span class="par-nr">${NR}</span>`;
    else cache = '—';
    let cost: TemplateResult | string = '—';
    if (l.cost_micros != null && !(l.cost_partial && l.cost_micros === 0)) cost = l.cost_partial ? html`<span title="completion tokens so far at the output price; the prompt's part comes with the usage">${money(l.cost_micros)}+</span>` : money(l.cost_micros);
    const tokens = l.tokens != null ? exact(l.tokens) : l.frames ? html`<span class="par-nr" title=${sourceWord(l)}>${exact(l.frames)} frames</span>` : '0';
    return html`<div class="par-row tone-${tone} ${done ? 'is-done' : ''}" role="row" data-lane=${l.id} data-backend=${l.backend} data-state=${l.state} style="--bk:${col}">
      <span class="par-who" role="cell">
        <span class="par-who-top"><span class="par-bk-tag">${l.backend}</span><span class="par-model">${l.model ?? ''}</span></span>
        <span class="par-who-sub">${l.slot != null ? html`<span class="par-slot-tag" title="the engine's sequence slot">slot ${l.slot}</span> · ` : nothing}${l.via === 'engine' ? html`<span class="par-via" title="the engine's own live count, state and prefill (it files this request under the proxy's id)">engine view</span> · ` : nothing}${l.client.kind}${who && who !== l.client.kind ? html` · <span class="par-sess">${who}</span>` : nothing}</span>
      </span>
      <span class="par-state" role="cell"><span class="par-dot tone-${tone} ${live && tone !== 'wait' ? 'is-live' : ''}"></span><span class="par-state-word">${l.label ?? stateLine(l)}</span></span>
      <span class="par-tokens" role="cell" title=${sourceWord(l)}>
        <span class="par-bar">${bar ? html`<span class="par-bar-think" style="width:${(bar.think * 100).toFixed(2)}%"></span><span class="par-bar-out" style="width:${(bar.out * 100).toFixed(2)}%"></span>` : html`<span class="par-bar-unknown"></span>`}</span>
        <span class="par-tok-num num">${tokens}${l.thinking_tokens ? html`<span class="par-think-n"> · ${exact(l.thinking_tokens)} think</span>` : nothing}${l.tool_chars ? html`<span class="par-think-n"> · ${compact(l.tool_chars)} chars of call</span>` : nothing}</span>
      </span>
      <span class="par-fig r num ${nowCell === NR ? 'par-nr' : ''}" role="cell" data-cell="now"><span class="par-fig-k">now</span>${nowCell}</span>
      <span class="par-fig r num ${avgCell === NR ? 'par-nr' : ''}" role="cell" data-cell="avg"><span class="par-fig-k">avg</span>${avgCell}</span>
      <span class="par-fig r num" role="cell" data-cell="ttft"><span class="par-fig-k">TTFT</span>${ttft}</span>
      <span class="par-fig r num" role="cell" data-cell="prefill"><span class="par-fig-k">prefill</span>${prefill}</span>
      <span class="par-fig r num" role="cell" data-cell="cache"><span class="par-fig-k">cache</span>${cache}</span>
      <span class="par-fig r num" role="cell" data-cell="cost"><span class="par-fig-k">cost</span>${cost}</span>
      <span class="par-fig r num" role="cell" data-cell="elapsed"><span class="par-fig-k">elapsed</span>${ms(l.elapsed_ms)}</span>
    </div>`;
  }
}
