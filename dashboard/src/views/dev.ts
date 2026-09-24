// Dev — the LM Studio-style developer view (VIS-16): a live log console, the recent requests
// with a to-scale timing bar, the server's effective configuration, and a test request box
// that goes through the public chat endpoint like every other client.

import { html, nothing, type TemplateResult } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import { repeat } from 'lit/directives/repeat.js';
import { api, describeError } from '../api/client';
import type { ChatMetrics, ChatTimings, ChatUsage, LogLevel, LogLine, RequestRow, SystemInfo } from '../api/types';
import { exact, fixed, ms, pct, tps, timeShort, dateTimeShort } from '../lib/format';
import { poll, type Poller } from '../lib/poll';
import { readJsonStream, ResumableStream, type StreamState } from '../lib/sse';
import { LightElement, type Loadable } from '../ui/base';
import { copyButton, icon, panel, skeleton, errorState, emptyState } from '../ui/bits';

const ROW_H = 22;
const MAX_LINES = 10_000;
const LEVELS: LogLevel[] = ['debug', 'info', 'warning', 'error'];

type ConsoleLine = LogLine | { seq: number; gap: number };

@customElement('qse-dev')
export class QseDev extends LightElement {
  @property({ attribute: false }) params: URLSearchParams = new URLSearchParams();

  // console
  @state() private lines: ConsoleLine[] = [];
  @state() private level: LogLevel = 'info';
  @state() private grep = '';
  @state() private follow = true;
  @state() private pending = 0;
  @state() private wrap = false;
  @state() private streamState: StreamState = 'closed';
  @state() private scrollY = 0;
  @state() private viewportH = 420;
  private stream: ResumableStream | null = null;
  private maxSeq = 0;
  private grepTimer: ReturnType<typeof setTimeout> | null = null;

  // requests
  @state() private requests: Loadable<RequestRow[]> = { state: 'loading' };
  @state() private expanded: number | null = null;
  @state() private nextBefore: number | null = null;
  @state() private loadingMore = false;
  private reqPoll: Poller | null = null;

  // config
  @state() private system: Loadable<SystemInfo> = { state: 'loading' };
  @state() private cfgSearch = '';

  // test box
  @state() private prompt = 'Say hi in one line.';
  @state() private maxTokens = 256;
  @state() private thinking = false;
  @state() private testState: 'idle' | 'running' | 'done' | 'error' = 'idle';
  @state() private reasoning = '';
  @state() private answer = '';
  @state() private testResult: { usage?: ChatUsage; timings?: ChatTimings; metrics?: ChatMetrics } | null = null;
  @state() private testError: string | null = null;
  private testCtrl: AbortController | null = null;

  connectedCallback(): void {
    super.connectedCallback();
    this.openStream();
    this.reqPoll = poll(() => this.loadRequests(), 10_000);
    this.reqPoll.start();
    void this.loadSystem();
  }
  disconnectedCallback(): void {
    this.stream?.close();
    this.reqPoll?.stop();
    this.testCtrl?.abort();
    super.disconnectedCallback();
  }

  // ---- console ------------------------------------------------------------------------------
  private openStream(resume = false): void {
    this.stream?.close();
    if (!resume) {
      this.lines = [];
      this.maxSeq = 0;
      this.pending = 0;
    }
    const s = new ResumableStream({
      url: (lastId) => api.logsUrl({ level: this.level, grep: this.grep || null, since: lastId, backlog: 500 }),
      onEvent: (ev) => {
        if (ev.event === 'log') {
          try {
            const line = JSON.parse(ev.data) as LogLine;
            if (line.seq <= this.maxSeq) return; // a duplicate after a reconnect
            this.maxSeq = line.seq;
            this.push(line);
          } catch {
            /* skip a malformed line */
          }
        } else if (ev.event === 'gap') {
          try {
            const g = JSON.parse(ev.data) as { dropped: number };
            this.push({ seq: this.maxSeq + 0.5, gap: g.dropped });
          } catch {
            /* ignore */
          }
        }
      },
      onState: (st) => {
        this.streamState = st;
        if (st === 'unauthorized') window.dispatchEvent(new CustomEvent('qse-unauthorized'));
      },
    });
    if (resume && this.stream?.lastId) s.lastId = this.stream.lastId;
    this.stream = s;
    s.start();
  }

  private push(line: ConsoleLine): void {
    const next = this.lines.length >= MAX_LINES ? this.lines.slice(this.lines.length - MAX_LINES + 1) : this.lines.slice();
    next.push(line);
    this.lines = next;
    if (this.follow) this.updateComplete.then(() => this.scrollToEnd());
    else this.pending++;
  }

  private scrollToEnd(): void {
    const el = this.querySelector<HTMLElement>('.console-scroll');
    if (el) el.scrollTop = el.scrollHeight;
  }

  private onConsoleScroll = (e: Event) => {
    const el = e.currentTarget as HTMLElement;
    this.scrollY = el.scrollTop;
    this.viewportH = el.clientHeight;
    const atEnd = el.scrollHeight - el.scrollTop - el.clientHeight < ROW_H * 2;
    if (!atEnd && this.follow) this.follow = false;
    if (atEnd && !this.follow && this.pending === 0) this.follow = true;
  };

  private resume(): void {
    this.follow = true;
    this.pending = 0;
    this.updateComplete.then(() => this.scrollToEnd());
  }

  private setLevel(l: LogLevel): void {
    this.level = l;
    this.openStream();
  }
  private setGrep(v: string): void {
    this.grep = v;
    if (this.grepTimer) clearTimeout(this.grepTimer);
    this.grepTimer = setTimeout(() => this.openStream(), 350);
  }

  private highlight(msg: string): TemplateResult | string {
    if (!this.grep) return msg;
    const q = this.grep.toLowerCase();
    const parts: (string | TemplateResult)[] = [];
    let i = 0;
    const lower = msg.toLowerCase();
    for (;;) {
      const j = lower.indexOf(q, i);
      if (j < 0) break;
      parts.push(msg.slice(i, j), html`<mark>${msg.slice(j, j + q.length)}</mark>`);
      i = j + q.length;
    }
    parts.push(msg.slice(i));
    return html`${parts}`;
  }

  private renderConsole(): TemplateResult {
    const total = this.lines.length;
    const virtual = !this.wrap;
    let first = 0;
    let last = total;
    if (virtual) {
      first = Math.max(0, Math.floor(this.scrollY / ROW_H) - 10);
      last = Math.min(total, Math.ceil((this.scrollY + this.viewportH) / ROW_H) + 10);
    } else if (total > 600) first = total - 600;
    const slice = this.lines.slice(first, last);
    const stateLabel: Record<StreamState, string> = { connecting: 'connecting', open: 'live', reconnecting: 'reconnecting', closed: 'closed', unauthorized: 'signed out', error: 'reconnecting' };

    const tools = html`
      <div class="seg" role="radiogroup" aria-label="level">
        ${LEVELS.map((l) => html`<button role="radio" aria-checked=${this.level === l} class="seg-btn ${this.level === l ? 'is-on' : ''}" @click=${() => this.setLevel(l)}>${l}</button>`)}
      </div>
      <label class="field field-search"><span class="sr-only">Search</span>${icon('search')}<input type="search" placeholder="search (server-side)" .value=${this.grep} @input=${(e: Event) => this.setGrep((e.target as HTMLInputElement).value)} /></label>
      <button class="btn btn-ghost btn-sm ${this.follow ? 'is-on' : ''}" title=${this.follow ? 'Pause' : 'Follow'} aria-pressed=${this.follow} @click=${() => (this.follow ? (this.follow = false) : this.resume())}>${this.follow ? icon('pause') : icon('play')} ${this.follow ? 'following' : 'paused'}</button>
      <button class="btn btn-ghost btn-sm ${this.wrap ? 'is-on' : ''}" title="Wrap long lines" aria-pressed=${this.wrap} @click=${() => (this.wrap = !this.wrap)}>${icon('wrap')}</button>
      <button class="btn btn-ghost btn-sm" title="Clear view" @click=${() => ((this.lines = []), (this.pending = 0))}>${icon('trash')}</button>
    `;
    const body = html`
      <div class="console ${this.wrap ? 'is-wrap' : ''}" data-state=${this.streamState}>
        <div class="console-status"><span class="dot dot-${this.streamState}"></span>${stateLabel[this.streamState]} · <span class="num">${total}</span> lines${this.streamState === 'reconnecting' || this.streamState === 'error' ? html` · resuming after <code>${this.stream?.lastId ?? '—'}</code>` : nothing}</div>
        <div class="console-scroll" @scroll=${this.onConsoleScroll} tabindex="0" role="log" aria-live="off">
          <div class="console-spacer" style="height:${virtual ? total * ROW_H : 'auto'}px">
            <div class="console-rows" style="transform:translateY(${virtual ? first * ROW_H : 0}px)">
              ${repeat(
                slice,
                (l) => l.seq,
                (l) =>
                  'gap' in l
                    ? html`<div class="log-line log-gap" style="height:${ROW_H}px">gap: ${l.gap} lines dropped</div>`
                    : html`<div class="log-line level-${l.level}" data-seq=${l.seq} style=${virtual ? `height:${ROW_H}px` : ''}>
                        <span class="log-ts">${timeShort(l.ts)}</span>
                        <span class="log-tag tag-${l.source}">${l.source}</span>
                        <span class="log-msg">${l.request_id && l.source === 'req'
                          ? html`<a href="#/dev" class="log-req" @click=${(e: Event) => {
                              e.preventDefault();
                              this.jumpToRequest(l.request_id as string);
                            }}>${this.highlight(l.msg)}</a>`
                          : this.highlight(l.msg)}</span>
                        <span class="log-copy">${copyButton(`${l.ts} ${l.level} ${l.msg}`, 'Copy line')}</span>
                      </div>`,
              )}
            </div>
          </div>
        </div>
        ${!this.follow && this.pending > 0 ? html`<button class="chip chip-float" @click=${() => this.resume()}>${icon('down')} ${this.pending} new line${this.pending === 1 ? '' : 's'}</button>` : nothing}
        ${total === 0 && this.streamState === 'open' ? html`<div class="console-empty">No lines at this level yet.</div>` : nothing}
      </div>
    `;
    return panel('Log', body, { sub: 'the engine\'s log, live; no prompt or answer text is ever logged', tools, cls: 'panel-console', id: 'console' });
  }

  private async jumpToRequest(rid: string): Promise<void> {
    const find = () => (this.requests.state === 'ready' ? this.requests.data.find((r) => r.request_id === rid) : undefined);
    let row = find();
    if (!row) {
      // A request that just finished is not in the table until the next poll: fetch now.
      await this.reqPoll?.refresh();
      row = find();
    }
    if (row) {
      this.expanded = row.id;
      await this.updateComplete;
      this.querySelector(`[data-req="${row.id}"]`)?.scrollIntoView({ block: 'center', behavior: 'smooth' });
    }
  }

  // ---- requests -----------------------------------------------------------------------------
  private async loadRequests(): Promise<void> {
    try {
      const r = await api.requests({ limit: 25 });
      if (this.requests.state === 'ready') {
        // Keep the older pages already loaded: merge newest-first by id.
        const seen = new Set(r.requests.map((x) => x.id));
        const older = this.requests.data.filter((x) => !seen.has(x.id) && x.id < (r.requests[r.requests.length - 1]?.id ?? Infinity));
        this.requests = { state: 'ready', data: [...r.requests, ...older] };
      } else {
        this.requests = { state: 'ready', data: r.requests };
        this.nextBefore = r.next_before;
      }
      if (this.nextBefore == null && r.next_before != null && this.requests.data.length <= 25) this.nextBefore = r.next_before;
    } catch (e) {
      const d = describeError(e);
      this.requests = { state: 'error', endpoint: d.endpoint, message: d.message };
    }
  }

  private async loadMore(): Promise<void> {
    if (this.requests.state !== 'ready' || this.nextBefore == null) return;
    this.loadingMore = true;
    try {
      const r = await api.requests({ limit: 25, before: this.nextBefore });
      this.requests = { state: 'ready', data: [...this.requests.data, ...r.requests] };
      this.nextBefore = r.next_before;
    } catch {
      /* the next poll shows the error state */
    } finally {
      this.loadingMore = false;
    }
  }

  private renderRequests(): TemplateResult {
    const r = this.requests;
    const body = (() => {
      if (r.state === 'loading') return skeleton(6);
      if (r.state === 'error') return errorState(r.endpoint, r.message, () => this.reqPoll?.refresh());
      if (r.state === 'empty' || r.data.length === 0) return emptyState('No requests in the ledger yet', 'The first chat through the engine shows up here within ten seconds.');
      return html`<div class="table-scroll">
          <table class="table table-req">
            <thead>
              <tr><th>Time</th><th>Client</th><th>Endpoint</th><th>Finish</th><th class="num-col">Prompt</th><th class="num-col">Cached</th><th class="num-col">Out</th><th class="num-col">Reasoning</th><th class="num-col">TTFT</th><th class="num-col">Decode</th><th class="num-col">tok/blk</th></tr>
            </thead>
            <tbody>
              ${repeat(
                r.data,
                (x) => x.id,
                (x) => html`<tr class="req-row ${this.expanded === x.id ? 'is-open' : ''} ${x.status >= 400 ? 'is-failed' : ''}" data-req=${x.id} tabindex="0" aria-expanded=${this.expanded === x.id}
                    @click=${() => (this.expanded = this.expanded === x.id ? null : x.id)}
                    @keydown=${(e: KeyboardEvent) => (e.key === 'Enter' || e.key === ' ') && (e.preventDefault(), (this.expanded = this.expanded === x.id ? null : x.id))}>
                    <td class="num">${timeShort(x.ts)}</td>
                    <td>${x.client.label ?? x.client.id}${x.client.label !== x.client.kind ? html`<span class="muted"> ${x.client.kind}</span>` : nothing}</td>
                    <td>${x.endpoint}${x.stream ? '' : html`<span class="muted"> sync</span>`}${x.thinking ? html` <span class="tag tag-think">think</span>` : nothing}${x.tool_calls ? html` <span class="tag">tools ${x.tool_calls}</span>` : nothing}</td>
                    <td><span class="finish finish-${x.finish_reason ?? 'none'}">${x.finish_reason ?? x.status}</span></td>
                    <td class="num-col num">${exact(x.prompt_tokens)}</td>
                    <td class="num-col num">${exact(x.cached_tokens)}</td>
                    <td class="num-col num">${exact(x.completion_tokens)}</td>
                    <td class="num-col num">${exact(x.reasoning_tokens)}</td>
                    <td class="num-col num">${ms(x.ttft_ms)}</td>
                    <td class="num-col num" title=${x.cache_source === 'response' ? `replayed at ${tps(x.decode_tps)}` : nothing}>${x.cache_source === 'response' ? 'replay' : fixed(x.decode_tps, 1)}</td>
                    <td class="num-col num">${fixed(x.tokens_per_block, 2)}</td>
                  </tr>
                  ${this.expanded === x.id
                    ? html`<tr class="req-expand"><td colspan="11">
                        <div class="req-detail">
                          <div class="req-detail-head"><code>${x.request_id}</code>${copyButton(x.request_id, 'Copy request id')}<span class="muted">${dateTimeShort(x.ts)} · status ${x.status} · max_tokens not recorded here · text is not recorded</span></div>
                          <qse-timing-bar .queue=${x.queue_ms} .prefill=${x.prompt_ms} .decode=${x.decode_ms}></qse-timing-bar>
                          <dl class="kv kv-grid">
                            <div><dt>drafted</dt><dd class="num">${exact(x.draft_tokens)}</dd></div>
                            <div><dt>accepted</dt><dd class="num">${exact(x.draft_accepted)}</dd></div>
                            <div><dt>acceptance</dt><dd class="num">${x.draft_tokens ? pct((x.draft_accepted ?? 0) / x.draft_tokens) : '—'}</dd></div>
                            <div><dt>blocks</dt><dd class="num">${exact(x.blocks)}</dd></div>
                            <div><dt>prefill</dt><dd class="num">${tps(x.prefill_tps)}</dd></div>
                            <div><dt>cache</dt><dd>${x.cache_source ?? '—'}</dd></div>
                            <div><dt>tool calls</dt><dd class="num">${x.tool_calls}</dd></div>
                            <div><dt>thinking</dt><dd>${x.thinking ? 'on' : 'off'}</dd></div>
                            <div><dt>error</dt><dd>${x.error_type ?? '—'}</dd></div>
                          </dl>
                          <p class="muted small">The ledger keeps no prompt or answer text — only counts and timings.</p>
                        </div>
                      </td></tr>`
                    : nothing}`,
              )}
            </tbody>
          </table>
        </div>
        ${this.nextBefore != null ? html`<button class="btn btn-ghost" ?disabled=${this.loadingMore} @click=${() => this.loadMore()}>${this.loadingMore ? 'Loading…' : 'Load older'}</button>` : nothing}`;
    })();
    return panel('Recent requests', body, { sub: 'newest first, refreshed every 10 s; click a row for its timing', id: 'requests' });
  }

  // ---- config -------------------------------------------------------------------------------
  private async loadSystem(): Promise<void> {
    try {
      this.system = { state: 'ready', data: await api.system() };
    } catch (e) {
      const d = describeError(e);
      this.system = { state: 'error', endpoint: d.endpoint, message: d.message };
    }
  }

  private renderConfig(): TemplateResult {
    const s = this.system;
    const body = (() => {
      if (s.state === 'loading') return skeleton(6);
      if (s.state === 'error') return errorState(s.endpoint, s.message, () => this.loadSystem());
      if (s.state === 'empty') return nothing;
      const sys = s.data;
      const q = this.cfgSearch.trim().toLowerCase();
      const groups: { name: string; rows: [string, string][] }[] = [
        { name: 'serving', rows: [] },
        { name: 'cache', rows: [] },
        { name: 'drafter', rows: [] },
        { name: 'kernels', rows: [] },
        { name: 'environment', rows: [] },
      ];
      for (const [k, v] of Object.entries(sys.flags.args)) {
        const g = /cache|prefix|session|response/.test(k) ? 1 : /draft|tree|len_|budget|deep/.test(k) ? 2 : 0;
        groups[g].rows.push([k, typeof v === 'string' ? v : JSON.stringify(v)]);
      }
      for (const [k, v] of Object.entries(sys.flags.env)) {
        const g = /^QWEN38_(TREE|DEEP|DRAFT|VERIFY|DF2|LEN)/.test(k) ? 2 : k.startsWith('QWEN38_') ? 3 : 4;
        groups[g].rows.push([k, v]);
      }
      const filtered = groups.map((g) => ({ ...g, rows: g.rows.filter(([k, v]) => !q || k.toLowerCase().includes(q) || v.toLowerCase().includes(q)) })).filter((g) => g.rows.length);
      const card = {
        model: sys.engine.model,
        max_len: sys.engine.max_len,
        drafter: sys.engine.drafter,
        tree: sys.engine.tree,
        reasoning_format: sys.engine.reasoning_format,
        reasoning_effort: sys.engine.reasoning_effort,
        version: sys.engine.version,
        git_sha: sys.engine.git_sha,
        code_sha256: sys.engine.code_sha256,
      };
      return html`
        <div class="model-card">
          <dl class="kv kv-grid">
            <div><dt>model</dt><dd>${card.model}</dd></div>
            <div><dt>max_len</dt><dd class="num">${exact(card.max_len)}</dd></div>
            <div><dt>drafter</dt><dd>${card.drafter}${card.tree ? ', tree' : ''}</dd></div>
            <div><dt>reasoning</dt><dd>${card.reasoning_format}, ${card.reasoning_effort}</dd></div>
            <div><dt>version</dt><dd class="num">${card.version} <span class="muted">${card.git_sha}</span></dd></div>
            <div><dt>code hash</dt><dd class="num">${card.code_sha256.slice(0, 12)}</dd></div>
          </dl>
          ${copyButton(JSON.stringify(card, null, 2), 'Copy as JSON')}
        </div>
        <label class="field field-search"><span class="sr-only">Search flags</span>${icon('search')}<input type="search" placeholder="search flags, e.g. DEEP" .value=${this.cfgSearch} @input=${(e: Event) => (this.cfgSearch = (e.target as HTMLInputElement).value)} /></label>
        ${filtered.length === 0 ? emptyState('No flag matches', `Nothing contains "${this.cfgSearch}".`) : nothing}
        <div class="cfg-scroll">
        ${filtered.map(
          (g) => html`<div class="cfg-group">
            <h3 class="cfg-group-title">${g.name}</h3>
            <table class="table table-cfg"><tbody>${g.rows.map(([k, v]) => html`<tr><td class="cfg-key">${k}</td><td class="cfg-val ${v === '<redacted>' ? 'is-secret' : ''}">${v}</td></tr>`)}</tbody></table>
          </div>`,
        )}
        </div>
      `;
    })();
    return panel('Server configuration', body, { sub: 'effective flags and environment; secrets are redacted by the engine', id: 'config' });
  }

  // ---- test request -------------------------------------------------------------------------
  private async send(): Promise<void> {
    this.testCtrl?.abort();
    const ctrl = new AbortController();
    this.testCtrl = ctrl;
    this.testState = 'running';
    this.reasoning = '';
    this.answer = '';
    this.testResult = null;
    this.testError = null;
    const body = {
      model: this.system.state === 'ready' ? this.system.data.engine.model : 'qwen38-spark-engine',
      messages: [{ role: 'user', content: this.prompt }],
      max_tokens: this.maxTokens,
      stream: true,
      stream_options: { include_usage: true },
      chat_template_kwargs: { enable_thinking: this.thinking },
    };
    try {
      const res = await fetch(api.chatUrl(), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'qse-dashboard' },
        body: JSON.stringify(body),
        signal: ctrl.signal,
        credentials: 'same-origin',
      });
      if (res.status === 503 || res.status === 429) {
        const retry = res.headers.get('Retry-After');
        let msg = `HTTP ${res.status}`;
        try {
          msg = (await res.json()).error?.message ?? msg;
        } catch {
          /* no body */
        }
        this.testError = `${res.status}: ${msg}${retry ? ` — retry after ${retry} s` : ''}`;
        this.testState = 'error';
        return;
      }
      if (!res.ok || !res.body) {
        this.testError = `HTTP ${res.status}`;
        this.testState = 'error';
        return;
      }
      let inThink = false;
      let buf = '';
      const flush = () => {
        // Route <think>…</think> to the reasoning pane; anything else to the answer.
        for (;;) {
          if (inThink) {
            const j = buf.indexOf('</think>');
            if (j < 0) {
              this.reasoning += buf;
              buf = '';
              return;
            }
            this.reasoning += buf.slice(0, j);
            buf = buf.slice(j + 8);
            inThink = false;
          } else {
            const i = buf.indexOf('<think>');
            if (i < 0) {
              // keep a possible partial tag in the buffer
              const keep = buf.lastIndexOf('<');
              if (keep >= 0 && buf.length - keep < 7 && '<think>'.startsWith(buf.slice(keep))) {
                this.answer += buf.slice(0, keep);
                buf = buf.slice(keep);
              } else {
                this.answer += buf;
                buf = '';
              }
              return;
            }
            this.answer += buf.slice(0, i);
            buf = buf.slice(i + 7);
            inThink = true;
          }
        }
      };
      for await (const chunk of readJsonStream(res.body)) {
        const c = chunk as { choices?: { delta?: { content?: string; reasoning_content?: string } }[]; usage?: ChatUsage; timings?: ChatTimings; metrics?: ChatMetrics };
        const d = c.choices?.[0]?.delta;
        if (d?.reasoning_content) this.reasoning += d.reasoning_content;
        if (d?.content) {
          buf += d.content;
          flush();
        }
        if (c.usage || c.timings || c.metrics) this.testResult = { usage: c.usage, timings: c.timings, metrics: c.metrics };
      }
      this.answer += buf;
      this.testState = 'done';
    } catch (e) {
      if ((e as Error).name === 'AbortError') {
        this.testState = 'done';
        return;
      }
      this.testError = (e as Error).message;
      this.testState = 'error';
    }
  }

  private renderTestBox(): TemplateResult {
    const t = this.testResult?.timings;
    const body = html`
      <form class="testbox" @submit=${(e: Event) => (e.preventDefault(), this.send())}>
        <label class="field field-block"><span>Prompt</span><textarea rows="3" .value=${this.prompt} @input=${(e: Event) => (this.prompt = (e.target as HTMLTextAreaElement).value)}></textarea></label>
        <div class="testbox-row">
          <label class="field"><span>max_tokens</span><input type="number" min="1" max="32768" .value=${String(this.maxTokens)} @input=${(e: Event) => (this.maxTokens = Number((e.target as HTMLInputElement).value) || 256)} /></label>
          <label class="switch"><input type="checkbox" .checked=${this.thinking} @change=${(e: Event) => (this.thinking = (e.target as HTMLInputElement).checked)} /><span>thinking</span></label>
          <span class="grow"></span>
          ${this.testState === 'running'
            ? html`<button type="button" class="btn" @click=${() => this.testCtrl?.abort()}>${icon('stop')} Stop</button>`
            : html`<button type="submit" class="btn btn-primary">${icon('send')} Send</button>`}
        </div>
      </form>
      ${this.testError ? html`<div class="state state-error" role="alert"><div class="state-title">${icon('warn')} Request refused</div><div class="state-body">${this.testError}</div></div>` : nothing}
      ${this.reasoning ? html`<details class="test-reasoning" open><summary>reasoning</summary><pre>${this.reasoning}</pre></details>` : nothing}
      ${this.answer || this.testState === 'running' ? html`<pre class="test-answer ${this.testState === 'running' ? 'is-streaming' : ''}">${this.answer}</pre>` : nothing}
      ${this.testResult
        ? html`<div class="test-result">
            <div class="stats stats-4">
              <div class="stat"><div class="stat-label">time to first token</div><div class="stat-value"><span class="num">${t ? ms(t.ttft_ms) : '—'}</span></div></div>
              <div class="stat"><div class="stat-label">decode</div><div class="stat-value"><span class="num">${t ? fixed(t.predicted_per_second, 1) : '—'}</span><span class="stat-unit">tok/s</span></div></div>
              <div class="stat"><div class="stat-label">tokens per block</div><div class="stat-value"><span class="num">${t ? fixed(t.tokens_per_block, 2) : '—'}</span></div></div>
              <div class="stat"><div class="stat-label">completion</div><div class="stat-value"><span class="num">${this.testResult.usage ? exact(this.testResult.usage.completion_tokens) : '—'}</span><span class="stat-unit">tokens</span></div></div>
            </div>
            <div class="test-json">
              ${(['usage', 'timings', 'metrics'] as const).map((k) => (this.testResult?.[k] ? html`<div><h4>${k}</h4><pre>${JSON.stringify(this.testResult[k], null, 2)}</pre></div>` : nothing))}
            </div>
          </div>`
        : nothing}
      <p class="muted small">Goes through the public <code>POST /v1/chat/completions</code> with <code>stream: true</code> and <code>include_usage</code>, tagged as the dashboard client; nothing typed here is stored by the dashboard.</p>
    `;
    return panel('Test request', body, { sub: 'the same queue as everyone else; usage and timings shown exactly as sent', id: 'testbox' });
  }

  render() {
    return html`<div class="view view-dev">
      ${this.renderConsole()}
      ${this.renderRequests()}
      <div class="grid-2 grid-2-config">
        ${this.renderConfig()}
        ${this.renderTestBox()}
      </div>
    </div>`;
  }
}
