// <qse-chat-context>: the Context panel. The context-window ribbon first: one bar for the window,
// split by part, glowing where the cache is warm, dim where it is cold, an orange tip where the
// prefill is now. Then Now (tok/s with a sparkline, tokens per round, acceptance, the thinking
// budget, the prefill), the model and pack, the memory this chat holds, the compaction events,
// and the usage. A text twin of the ribbon serves screen readers.

import { html, nothing, svg, type TemplateResult } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import { bytes, exact, fixed, money, ms, pct, timeShort } from '../lib/format';
import { LightElement } from '../ui/base';
import { icon } from '../ui/bits';
import '../charts/spark';
import { lastStats } from './store';
import type { Attachment as Attach } from './connection';
import { effortWord, type Conversation, type ContextPartKind } from './types';
import { elapsedWords } from './thread';
import { budgetWords, cacheWords, draftFigure, HARNESS_PENDING, kvWords, liveFigure, memoryWords, partWords, residentWords, sandboxWords, skillTokens, slotWords, turnFigure } from './context-words';

const PART_COLOR: Record<ContextPartKind, string> = {
  system: 'var(--series-muted)',
  tools: 'var(--series-aqua)',
  skills: 'var(--series-violet)',
  files: 'var(--series-yellow)',
  images: 'var(--series-red)',
  messages: 'var(--cobalt)',
  thinking: 'var(--think)',
  reserved: 'transparent',
};

@customElement('qse-chat-context')
export class QseChatContext extends LightElement {
  @property({ attribute: false }) conv: Conversation | null = null;
  @property({ attribute: false }) attach: Attach | null = null;
  @property({ type: Number }) now = Date.now();
  @state() private openSummary: string | null = null;
  @state() private hover: ContextPartKind | null = null;

  private fire(name: string, detail: unknown = {}): void {
    this.dispatchEvent(new CustomEvent(name, { detail, bubbles: true, composed: true }));
  }

  render() {
    const c = this.conv;
    if (!c) return html`<div class="ch-ctx-empty muted small">Open a chat to see its context.</div>`;
    return html`<div class="ch-ctx-scroll">
      ${this.renderWindow(c)}
      ${this.renderNow(c)}
      ${this.renderModel(c)}
      ${this.renderSkills(c)}
      ${this.renderMemory(c)}
      ${this.renderCompaction(c)}
      ${this.renderUsage(c)}
    </div>`;
  }

  // ---- the ribbon -----------------------------------------------------------------------------

  private renderWindow(c: Conversation): TemplateResult {
    const ctx = c.context;
    const W = 1000;
    const H = 26;
    const total = Math.max(1, ctx.window);
    const parts = ctx.parts.filter((p) => p.tokens > 0);
    let x = 0;
    const segs: TemplateResult[] = [];
    const warmEnd = (ctx.cache.warm / total) * W;
    const pf = c.run?.prefill;
    for (const p of parts) {
      const w = (p.tokens / total) * W;
      if (p.kind === 'reserved') {
        segs.push(svg`<rect class="ch-rib-reserved" x=${x} y="4" width=${Math.max(0, w)} height=${H - 8} rx="2" />`);
      } else {
        const warmW = Math.max(0, Math.min(w, warmEnd - x));
        const coldW = Math.max(0, w - warmW);
        if (warmW > 0) segs.push(svg`<rect class="ch-rib-seg is-warm ${this.hover === p.kind ? 'is-hover' : ''}" x=${x} y="0" width=${warmW} height=${H} fill=${PART_COLOR[p.kind]} />`);
        if (coldW > 0) segs.push(svg`<rect class="ch-rib-seg is-cold ${this.hover === p.kind ? 'is-hover' : ''}" x=${x + warmW} y="0" width=${coldW} height=${H} fill=${PART_COLOR[p.kind]} />`);
      }
      x += w;
    }
    const usedPct = ctx.used / total;
    const tip = pf ? ((ctx.cache.warm + pf.done) / total) * W : null;
    const twin = `${exact(ctx.used)} of ${exact(ctx.window)} tokens used; ${exact(ctx.cache.warm)} warm, ${exact(ctx.cache.cold)} cold`;
    const cw = cacheWords(ctx.cache);
    return html`<section class="ch-ctx-sec" data-sec="window">
      <h3 class="ch-ctx-h">Context window</h3>
      <div class="ch-rib-wrap">
        <svg class="ch-rib" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img" aria-label=${twin}>
          <defs><filter id="ch-rib-glow" x="-2%" y="-60%" width="104%" height="220%"><feGaussianBlur stdDeviation="3" /></filter></defs>
          <rect class="ch-rib-track" x="0" y="0" width=${W} height=${H} rx="3" />
          ${warmEnd > 0 ? svg`<rect class="ch-rib-glow" x="0" y="0" width=${warmEnd} height=${H} filter="url(#ch-rib-glow)" />` : nothing}
          ${segs}
          ${tip != null ? svg`<rect class="ch-rib-tip" x=${Math.max(0, tip - 3)} y="-3" width="6" height=${H + 6} rx="1.5" />` : nothing}
          ${usedPct > 0 ? svg`<line class="ch-rib-used" x1=${(ctx.used / total) * W} x2=${(ctx.used / total) * W} y1="-2" y2=${H + 2} />` : nothing}
        </svg>
      </div>
      <div class="ch-rib-line num"><b>${exact(ctx.used)}</b> of ${exact(ctx.window)}, ${pct(usedPct, 0)}</div>
      <p class="ch-rib-cache" data-cache=${ctx.cache.source}>${cw.warm ? html`<span class="ch-warm">${cw.warm}</span>` : cw.none}${cw.cold ? html`, <span class="ch-cold">${cw.cold}</span>` : ''}${pf ? html`, prefilling ${exact(pf.done)} of ${exact(pf.total)}` : ''}</p>
      <ul class="ch-legend" @mouseleave=${() => (this.hover = null)}>
        ${ctx.parts.map((p) => html`<li class="${p.tokens ? '' : 'is-zero'}" @mouseenter=${() => (this.hover = p.kind)}><span class="swatch ${p.kind === 'reserved' ? 'swatch-outline' : ''}" style=${p.kind === 'reserved' ? '' : `background:${PART_COLOR[p.kind]}`}></span><span class="ch-legend-label">${p.label}</span><span class="num ch-legend-n">${partWords(p.kind, p.tokens, ctx.split, c.context.model.backend)}</span></li>`)}
      </ul>
      <div class="ch-ctx-row"><button type="button" class="btn btn-sm" ?disabled=${!!c.run || ctx.used < 4_000} title="fold the older turns into a summary" @click=${() => this.fire('ch-compact')}>Compact now</button></div>
    </section>`;
  }

  // ---- now ------------------------------------------------------------------------------------

  private renderNow(c: Conversation): TemplateResult {
    const r = c.run;
    const last = lastStats(c);
    const tps = r ? r.decode_tps : last?.decode_tps ?? null;
    const hist = r?.tps_history ?? [];
    // the contract's null is "no cap" (auto effort, or a model without levels), never a made-up figure
    const budget = !c.settings.thinking ? 0 : r ? r.thinking.budget : null;
    const used = r?.thinking.used ?? last?.reasoning_tokens ?? 0;
    // the budget caps each model call: its meter is the call's thinking
    const capped = r ? r.thinking.call ?? r.thinking.used : used;
    const be = c.context.model.backend;
    const drafter = c.context.model.drafter;
    const fig = (v: number | null | undefined, fmt: (v: number) => string, kind: 'tps' | 'round' | 'accepted') => (r ? liveFigure(v, fmt, r, kind, be, drafter) : kind === 'tps' ? turnFigure(v, fmt, !!last) : draftFigure(v, fmt, !!last, be));
    // this call's first token while a run is on (a cloud model's: the network and the provider's
    // queue in it), else the last turn's
    const ttft = r ? r.ttft_ms : last?.ttft_ms;
    const elapsed = r ? Math.max(0, Math.round((this.now - Date.parse(r.started_at)) / 1000)) : null;
    return html`<section class="ch-ctx-sec ${r ? 'is-live' : 'is-idle'}" data-sec="now">
      <h3 class="ch-ctx-h">${r ? 'Now' : 'Last turn'}</h3>
      <div class="ch-fig">
        <div class="ch-fig-value num">${tps != null ? html`${fixed(tps, 1)}<span class="ch-fig-unit">tok/s</span>` : html`<span class="ch-fig-why muted small">${fig(null, String, 'tps')}</span>`}</div>
        ${hist.length > 1 ? html`<qse-spark .values=${hist} mode="ribbon" .height=${36} .format=${(v: number) => v.toFixed(1)} unit=" tok/s" aria-label="decode tok/s, last minute"></qse-spark>` : html`<div class="ch-fig-flat"></div>`}
      </div>
      <dl class="kv ch-kv">
        <div><dt>phase</dt><dd>${r ? r.phase : last ? last.finish : 'idle'}</dd></div>
        ${r && elapsed != null ? html`<div><dt>running for</dt><dd class="num">${elapsedWords(elapsed)}</dd></div>` : nothing}
        ${r || last ? html`<div data-fig="ttft"><dt>first token</dt><dd class="num">${ttft != null ? ms(ttft) : r ? 'waiting for it' : turnFigure(null, String, true)}</dd></div>` : nothing}
        <div><dt>tokens per round</dt><dd class="num">${fig(r ? r.tokens_per_round : last?.tokens_per_round, (v) => fixed(v, 2), 'round')}</dd></div>
        <div><dt>accepted</dt><dd class="num">${fig(r ? r.acceptance : last?.acceptance, (v) => pct(v, 0), 'accepted')}</dd></div>
        ${r?.prefill ? html`<div><dt>prefill</dt><dd class="num">${exact(r.prefill.done)} / ${exact(r.prefill.total)}</dd></div>` : nothing}
        ${r?.tool ? html`<div><dt>tool</dt><dd><code>${r.tool.name}</code></dd></div>` : nothing}
      </dl>
      <div class="ch-meter-block">
        <div class="meter-head"><span>thinking budget</span><span class="num">${budgetWords(budget, used, capped, be)}</span></div>
        <div class="meter ch-meter"><span class="meter-fill ${budget && be !== 'cloud' && capped / budget > 0.85 ? 'is-warn' : ''}" style="width:${budget && be !== 'cloud' ? Math.min(100, (capped / budget) * 100).toFixed(1) : 0}%"></span></div>
      </div>
    </section>`;
  }

  private renderModel(c: Conversation): TemplateResult {
    const m = c.context.model;
    return html`<section class="ch-ctx-sec" data-sec="model">
      <h3 class="ch-ctx-h">Model and pack</h3>
      <dl class="kv ch-kv">
        <div><dt>model</dt><dd>${m.name}</dd></div>
        <div><dt>profile</dt><dd>${m.profile ?? 'none'}</dd></div>
        <div><dt>weights</dt><dd>${m.weights || 'not reported by this backend'}</dd></div>
        <div><dt>drafter</dt><dd>${m.drafter ?? 'none'}</dd></div>
        <div><dt>window</dt><dd class="num">${m.max_len ? exact(m.max_len) : 'not reported by this backend'}</dd></div>
        <div><dt>effort</dt><dd>${m.thinking ? effortWord(m.effort) : 'thinking off'}</dd></div>
      </dl>
    </section>`;
  }

  /** SRV-361: the chat's pins with their tokens, and what the harness put in last */
  private renderSkills(c: Conversation): TemplateResult {
    const sk = c.context.skills;
    const picked = c.context.harness?.packages ?? [];
    if (!sk) return html``;
    return html`<section class="ch-ctx-sec" data-sec="skills">
      <h3 class="ch-ctx-h">Skills</h3>
      <dl class="kv ch-kv">
        <div><dt>mode</dt><dd>${sk.mode === 'auto' ? 'the harness picks' : sk.mode === 'manual' ? 'pins only' : 'the model reads them'}</dd></div>
        ${sk.pinned.map((p) => html`<div data-pin=${p.name}><dt>pinned</dt><dd>${p.name} ${p.missing ? html`<span class="warn-ink">no longer in the library</span>` : html`<span class="num muted">${skillTokens(p.tokens ?? picked.find((x) => x.name === p.name)?.tokens)}</span>`}</dd></div>`)}
        ${picked.filter((p) => !sk.pinned.some((x) => x.name === p.name)).map((p) => html`<div data-harness=${p.name}><dt>harness</dt><dd>${p.name} <span class="num muted">${skillTokens(p.tokens)}</span></dd></div>`)}
        ${sk.mode === 'auto' && !c.context.harness && (c.context.model.backend ?? 'engine') === 'engine' ? html`<div data-harness="none"><dt>harness</dt><dd class="muted">${HARNESS_PENDING}</dd></div>` : nothing}
        ${!sk.pinned.length && !picked.length ? html`<div><dt>pinned</dt><dd class="muted">none</dd></div>` : nothing}
        <div data-harness="memory"><dt>memory notes</dt><dd>${memoryWords(c.context.harness, c.context.model.backend)}</dd></div>
      </dl>
    </section>`;
  }

  private renderMemory(c: Conversation): TemplateResult {
    const mem = c.context.memory;
    const sb = c.sandbox;
    return html`<section class="ch-ctx-sec" data-sec="memory">
      <h3 class="ch-ctx-h">Memory on the box</h3>
      <dl class="kv ch-kv">
        <div data-mem="kv"><dt>KV for this chat</dt><dd>${kvWords(mem, c.context.model.backend)}</dd></div>
        <div data-mem="resident"><dt>resident prefix</dt><dd class=${mem.resident_owner === 'this' ? 'num' : ''}>${residentWords(mem, c.context.model.backend)}</dd></div>
        <div data-mem="slot"><dt>slot</dt><dd>${slotWords(c.context.cache.slot, mem)}</dd></div>
        <div data-mem="sandbox"><dt>sandbox</dt><dd class=${sb.memory_bytes != null ? 'num' : ''}>${sandboxWords(sb)}</dd></div>
        <div><dt>sandbox disk</dt><dd class="num">${bytes(sb.disk_bytes, 0)} of ${bytes(sb.disk_limit_bytes, 0)}</dd></div>
      </dl>
      <div class="ch-ctx-row"><button type="button" class="linkish" @click=${() => this.fire('ch-open-settings', { section: 'sandbox' })}>Sandbox settings</button></div>
    </section>`;
  }

  private renderCompaction(c: Conversation): TemplateResult {
    return html`<section class="ch-ctx-sec" data-sec="compaction">
      <h3 class="ch-ctx-h">Compaction</h3>
      ${c.compactions.length
        ? html`<ul class="ch-cp-list">
            ${c.compactions.map((cp) => html`<li><div class="ch-cp-row"><span class="num">${timeShort(cp.at)}</span><span class="num">${exact(cp.before_tokens)} → ${exact(cp.after_tokens)}</span><button type="button" class="linkish" aria-expanded=${this.openSummary === cp.id} @click=${() => (this.openSummary = this.openSummary === cp.id ? null : cp.id)}>${this.openSummary === cp.id ? 'hide summary' : 'show summary'}</button></div>${this.openSummary === cp.id ? html`<p class="ch-cp-summary">${cp.summary}</p>` : nothing}</li>`)}
          </ul>`
        : html`<p class="muted small">No compaction yet. The engine folds the older turns into a summary when the window fills, or when you press Compact now.</p>`}
    </section>`;
  }

  private renderUsage(c: Conversation): TemplateResult {
    const u = c.usage;
    return html`<section class="ch-ctx-sec" data-sec="usage">
      <h3 class="ch-ctx-h">Usage, this chat</h3>
      <dl class="kv ch-kv">
        <div><dt>turns</dt><dd class="num">${exact(u.turns)}</dd></div>
        <div><dt>tokens in</dt><dd class="num">${exact(u.prompt_tokens)}</dd></div>
        <div><dt>tokens out</dt><dd class="num">${exact(u.completion_tokens)}</dd></div>
        <div><dt>thinking</dt><dd class="num">${exact(u.reasoning_tokens)}</dd></div>
        <div><dt>cached</dt><dd class="num">${exact(u.cached_tokens)}</dd></div>
        <div><dt>tool calls</dt><dd class="num">${exact(u.tool_calls)}</dd></div>
        <div><dt>time</dt><dd class="num">${elapsedWords(u.wall_s)}</dd></div>
        ${u.cost_usd != null ? html`<div data-usage="cost"><dt>cost</dt><dd class="num" title="at the prices set per model under Backends">${money(Math.round(u.cost_usd * 1e6))}</dd></div>` : nothing}
      </dl>
      <div class="ch-ctx-row"><a class="linkish" href="#/usage">${icon('usage')} All usage</a></div>
    </section>`;
  }
}
