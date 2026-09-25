// Playground — the chat bench (VIS-19..22): a multi-turn streaming conversation against the
// public chat endpoint with the reasoning shown, editable roles and a system prompt, a
// parameters column that sends only what was changed, the function-calling flow with mock
// results, a per-turn readout of the engine's figures, presets in the browser, and exports.

import { html, nothing, type TemplateResult } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import { repeat } from 'lit/directives/repeat.js';
import { unsafeHTML } from 'lit/directives/unsafe-html.js';
import { api, describeError } from '../api/client';
import type { SystemInfo } from '../api/types';
import { exact } from '../lib/format';
import { readJsonStream } from '../lib/sse';
import { LightElement } from '../ui/base';
import { copyButton, icon } from '../ui/bits';
import { exportFileName, readoutLine, readoutOf, toJsonExport, toMarkdownExport } from '../playground/export';
import { renderMarkdown } from '../playground/markdown';
import { defaultsFrom, validateParams, type Defaults } from '../playground/params';
import { clearDraft, deletePreset, exportPresets, importPresets, loadDraft, loadPresets, presetFrom, renamePreset, saveDraft, setupEquals, setupFrom, upsertPreset } from '../playground/presets';
import { asCurl, buildRequest, type ChatRequestBody } from '../playground/request';
import { ChatStreamReducer } from '../playground/stream';
import { echoResult, prettyArgs, templateFor, toolResultMessages, validateTools } from '../playground/tools';
import { emptySetup, ROLES, uid, type Message, type Role, type Setup, type TurnStats } from '../playground/types';
import type { PresetAction, QsePgSetup } from '../playground/setup-panel';
import '../playground/setup-panel';

const FALLBACK_MODEL = 'qwen38-spark-engine';

interface LastResponse {
  message: { role: 'assistant'; content: string; reasoning_content?: string; tool_calls?: Message['tool_calls'] };
  finish_reason: string | null;
  usage?: TurnStats['usage'];
  timings?: TurnStats['timings'];
  metrics?: TurnStats['metrics'];
  error?: TurnStats['error'];
  chunks: number;
  bytes: number;
}

@customElement('qse-playground')
export class QsePlayground extends LightElement {
  @property({ attribute: false }) params: URLSearchParams = new URLSearchParams();

  @state() private setup: Setup = emptySetup();
  @state() private messages: Message[] = [];
  @state() private presets = loadPresets();
  @state() private presetName: string | null = null;
  @state() private storageOk = true;
  @state() private defaults: Defaults = {};
  @state() private defaultsReady = false;
  @state() private model: string | null = null;
  @state() private streaming = false;
  @state() private editing: { uid: string; text: string; role: Role } | null = null;
  @state() private composer = '';
  // a touch keyboard's Enter is a newline; a keyboard's Enter sends
  @state() private enterSends = !(typeof matchMedia === 'function' && matchMedia('(pointer: coarse)').matches);
  @state() private drawer: 'none' | 'raw' | 'json' = 'none';
  @state() private setupOpen = false;
  @state() private exportOpen = false;
  @state() private readoutMore = false;
  @state() private notice: { kind: 'ok' | 'warn'; text: string } | null = null;
  @state() private lastRequest: ChatRequestBody | null = null;
  @state() private lastResponse: LastResponse | null = null;
  @state() private toolResults: Record<string, string> = {};
  private ctrl: AbortController | null = null;
  private draftTimer: ReturnType<typeof setTimeout> | null = null;
  private noticeTimer: ReturnType<typeof setTimeout> | null = null;
  private stickToEnd = true;

  connectedCallback(): void {
    super.connectedCallback();
    const d = loadDraft();
    if (d) {
      this.setup = d.setup;
      this.messages = d.messages.map((m) => ({ ...m, uid: m.uid || uid() }));
      this.presetName = d.presetName;
    }
    void this.loadSystem();
  }
  disconnectedCallback(): void {
    this.ctrl?.abort();
    if (this.draftTimer) clearTimeout(this.draftTimer);
    super.disconnectedCallback();
  }

  private async loadSystem(): Promise<void> {
    try {
      const s: SystemInfo = await api.system();
      this.defaults = defaultsFrom(s);
      this.defaultsReady = true;
      this.model = s.engine.model;
    } catch (e) {
      const d = describeError(e);
      if (d.status !== 401) this.say('warn', `Server defaults are not available (${d.endpoint || 'the engine'}: ${d.message}); the fields still work, the engine's own defaults apply.`);
    }
  }

  // ---- state helpers ----------------------------------------------------------------------
  private say(kind: 'ok' | 'warn', text: string): void {
    this.notice = { kind, text };
    if (this.noticeTimer) clearTimeout(this.noticeTimer);
    this.noticeTimer = setTimeout(() => (this.notice = null), kind === 'ok' ? 4000 : 9000);
  }

  private persist(): void {
    if (this.draftTimer) clearTimeout(this.draftTimer);
    this.draftTimer = setTimeout(() => {
      // the draft never keeps a turn that is still streaming
      const ok = saveDraft({ version: 1, setup: this.setup, messages: this.messages.filter((m) => !m.stats || m.stats.finish !== null || m.role !== 'assistant'), presetName: this.presetName });
      if (!ok && this.storageOk) this.storageOk = false;
    }, 300);
  }

  private setSetup(s: Setup): void {
    this.setup = s;
    this.persist();
  }
  private setMessages(m: Message[]): void {
    this.messages = m;
    this.persist();
  }

  private get toolsValidation() {
    return validateTools(this.setup.toolsText);
  }
  private get paramErrors() {
    return validateParams(this.setup.params, this.defaults);
  }
  private get blocked(): string | null {
    const pe = Object.values(this.paramErrors);
    if (pe.length) return pe[0];
    const te = this.toolsValidation.errors;
    if (te.length) return `tools: ${te[0]}`;
    return null;
  }
  private get presetDirty(): boolean {
    if (!this.presetName) return false;
    const p = this.presets.find((x) => x.name === this.presetName);
    return !!p && !setupEquals(setupFrom(p), this.setup);
  }

  private currentRequest(extra: Message[] = []): ChatRequestBody {
    return buildRequest(this.setup, [...this.messages, ...extra], { model: this.model ?? FALLBACK_MODEL, tools: this.toolsValidation.tools });
  }

  private scrollTranscript(): void {
    if (!this.stickToEnd) return;
    this.updateComplete.then(() => {
      const el = this.querySelector<HTMLElement>('.pg-transcript');
      if (el) el.scrollTop = el.scrollHeight;
    });
  }

  // ---- the conversation -------------------------------------------------------------------
  private send(): void {
    if (this.streaming) return;
    const text = this.composer.trim();
    if (text) {
      this.setMessages([...this.messages, { uid: uid(), role: 'user', content: text }]);
      this.composer = '';
    } else if (!this.messages.length) return;
    void this.run();
  }

  private regenerate(): void {
    if (this.streaming) return;
    let msgs = this.messages.slice();
    while (msgs.length && msgs[msgs.length - 1].role === 'assistant') msgs.pop();
    if (!msgs.length) return;
    this.setMessages(msgs);
    void this.run();
  }

  private stop(): void {
    this.ctrl?.abort();
  }

  private newChat(): void {
    this.ctrl?.abort();
    this.setMessages([]);
    this.lastRequest = null;
    this.lastResponse = null;
    this.toolResults = {};
    this.editing = null;
    clearDraft();
    this.persist();
  }

  private async run(): Promise<void> {
    const blocked = this.blocked;
    if (blocked) return this.say('warn', blocked);
    this.ctrl?.abort();
    const ctrl = new AbortController();
    this.ctrl = ctrl;
    const body = this.currentRequest();
    this.lastRequest = body;
    this.lastResponse = null;
    const turn: Message = { uid: uid(), role: 'assistant', content: '', reasoning: '', stats: { finish: null } };
    this.messages = [...this.messages, turn];
    this.streaming = true;
    this.stickToEnd = true;
    this.scrollTranscript();
    const started = performance.now();
    const reducer = new ChatStreamReducer();
    let bytes = 0;
    const apply = (final = false) => {
      const st = final ? reducer.end() : reducer.state;
      const idx = this.messages.findIndex((m) => m.uid === turn.uid);
      if (idx < 0) return;
      const next: Message = {
        ...turn,
        content: st.content,
        reasoning: st.reasoning,
        tool_calls: st.toolCalls.length ? st.toolCalls : undefined,
        stats: { ...(this.messages[idx].stats ?? { finish: null }), finish: final ? st.finish ?? (st.error ? 'error' : 'stop') : null, usage: st.usage, timings: st.timings, metrics: st.metrics, error: st.error ?? null, elapsedMs: performance.now() - started, chunks: st.chunks, bytes },
      };
      const msgs = this.messages.slice();
      msgs[idx] = next;
      this.messages = msgs;
      this.scrollTranscript();
      return next;
    };
    try {
      const res = await fetch(api.chatUrl(), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'qse-dashboard' },
        body: JSON.stringify(body),
        signal: ctrl.signal,
        credentials: 'same-origin',
      });
      if (!res.ok || !res.body) {
        let msg = `HTTP ${res.status}`;
        let type: string | undefined;
        try {
          const j = await res.json();
          msg = j?.error?.message ?? msg;
          type = j?.error?.type;
        } catch {
          /* no body */
        }
        const retry = res.headers.get('Retry-After');
        const text = `${res.status}: ${msg}${retry ? ` — retry after ${retry} s` : ''}`;
        this.finishTurn(turn.uid, { finish: 'error', error: { message: text, type }, elapsedMs: performance.now() - started });
        this.lastResponse = { message: { role: 'assistant', content: '' }, finish_reason: null, error: { message: text, type }, chunks: 0, bytes: 0 };
        return;
      }
      const counting = new TransformStream<Uint8Array, Uint8Array>({
        transform(chunk, c) {
          bytes += chunk.byteLength;
          c.enqueue(chunk);
        },
      });
      for await (const chunk of readJsonStream(res.body.pipeThrough(counting))) {
        reducer.feed(chunk);
        apply();
      }
      const final = apply(true);
      const st = reducer.state;
      this.lastResponse = { message: { role: 'assistant', content: st.content, ...(st.reasoning ? { reasoning_content: st.reasoning } : {}), ...(st.toolCalls.length ? { tool_calls: st.toolCalls } : {}) }, finish_reason: st.finish, usage: st.usage, timings: st.timings, metrics: st.metrics, error: st.error ?? undefined, chunks: st.chunks, bytes };
      if (final?.tool_calls?.length) {
        // prefill the result boxes with the templates' examples
        const r = { ...this.toolResults };
        for (const c of final.tool_calls) if (r[c.id] === undefined) r[c.id] = templateFor(c.function.name)?.example ?? '';
        this.toolResults = r;
      }
    } catch (e) {
      if ((e as Error).name === 'AbortError') {
        const st = reducer.end();
        const idx = this.messages.findIndex((m) => m.uid === turn.uid);
        if (idx >= 0) {
          const msgs = this.messages.slice();
          msgs[idx] = { ...msgs[idx], content: st.content, reasoning: st.reasoning, tool_calls: st.toolCalls.length ? st.toolCalls : undefined, stats: { finish: 'aborted', usage: st.usage, timings: st.timings, metrics: st.metrics, elapsedMs: performance.now() - started, chunks: st.chunks, bytes } };
          this.messages = msgs;
        }
        this.lastResponse = { message: { role: 'assistant', content: st.content }, finish_reason: 'aborted', chunks: st.chunks, bytes };
      } else {
        this.finishTurn(turn.uid, { finish: 'error', error: { message: (e as Error).message }, elapsedMs: performance.now() - started });
      }
    } finally {
      if (this.ctrl === ctrl) {
        this.streaming = false;
        this.ctrl = null;
      }
      this.persist();
    }
  }

  private finishTurn(id: string, stats: TurnStats): void {
    const idx = this.messages.findIndex((m) => m.uid === id);
    if (idx < 0) return;
    const msgs = this.messages.slice();
    msgs[idx] = { ...msgs[idx], stats: { ...msgs[idx].stats, ...stats } };
    this.messages = msgs;
  }

  // ---- edits ------------------------------------------------------------------------------
  private startEdit(m: Message): void {
    this.editing = { uid: m.uid, text: m.content, role: m.role };
  }
  private saveEdit(resend: boolean): void {
    const e = this.editing;
    if (!e) return;
    const idx = this.messages.findIndex((m) => m.uid === e.uid);
    if (idx < 0) {
      this.editing = null;
      return;
    }
    const msgs = this.messages.slice();
    const prev = msgs[idx];
    msgs[idx] = { ...prev, content: e.text, role: e.role, ...(e.role !== 'assistant' ? { tool_calls: undefined, reasoning: undefined, stats: undefined } : {}), ...(e.role !== 'tool' ? { tool_call_id: undefined, name: undefined } : {}) };
    this.editing = null;
    if (resend) {
      this.setMessages(msgs.slice(0, idx + 1));
      void this.run();
    } else this.setMessages(msgs);
  }
  private deleteMessage(id: string): void {
    this.setMessages(this.messages.filter((m) => m.uid !== id));
    if (this.editing?.uid === id) this.editing = null;
  }
  private addMessage(role: Role): void {
    const m: Message = { uid: uid(), role, content: '' };
    this.setMessages([...this.messages, m]);
    this.editing = { uid: m.uid, text: '', role };
    this.stickToEnd = true;
    this.scrollTranscript();
    this.updateComplete.then(() => this.querySelector<HTMLTextAreaElement>(`[data-uid="${m.uid}"] textarea`)?.focus());
  }

  private sendToolResults(m: Message): void {
    if (!m.tool_calls?.length || this.streaming) return;
    const missing = m.tool_calls.filter((c) => !(this.toolResults[c.id] ?? '').trim());
    if (missing.length) this.say('warn', `${missing.length} call${missing.length === 1 ? ' has' : 's have'} no result; sent as an empty string.`);
    const idx = this.messages.findIndex((x) => x.uid === m.uid);
    const results = toolResultMessages(m.tool_calls, this.toolResults);
    this.setMessages([...this.messages.slice(0, idx + 1), ...results]);
    void this.run();
  }

  // ---- presets ----------------------------------------------------------------------------
  private onPreset(a: PresetAction): void {
    switch (a.action) {
      case 'save': {
        const r = upsertPreset(this.presets, presetFrom(a.name, this.setup));
        this.presets = r.list;
        this.presetName = a.name;
        if (!r.persisted) this.storageOk = false;
        this.say(r.persisted ? 'ok' : 'warn', r.persisted ? `Saved preset "${a.name}".` : `"${a.name}" is kept for this page only — storage is blocked.`);
        this.persist();
        break;
      }
      case 'load': {
        const p = this.presets.find((x) => x.name === a.name);
        if (!p) return;
        this.setSetup(setupFrom(p));
        this.presetName = p.name;
        this.say('ok', `Loaded "${p.name}".`);
        break;
      }
      case 'delete': {
        this.presets = deletePreset(this.presets, a.name).list;
        if (this.presetName === a.name) this.presetName = null;
        this.persist();
        break;
      }
      case 'rename': {
        this.presets = renamePreset(this.presets, a.name, a.to).list;
        if (this.presetName === a.name) this.presetName = a.to;
        this.persist();
        break;
      }
      case 'export':
        this.download(exportPresets(this.presets), `playground-presets-${new Date().toISOString().slice(0, 10)}.json`, 'application/json');
        break;
      case 'import': {
        const r = importPresets(this.presets, a.text);
        this.presets = r.list;
        if (!r.added) this.say('warn', `Import refused: ${r.errors[0] ?? 'no presets in the file'}`);
        else this.say(r.errors.length ? 'warn' : 'ok', `Imported ${r.added} preset${r.added === 1 ? '' : 's'}${r.errors.length ? `; skipped: ${r.errors.join('; ')}` : '.'}`);
        break;
      }
    }
  }

  private download(text: string, name: string, type: string): void {
    const a = document.createElement('a');
    const url = URL.createObjectURL(new Blob([text], { type }));
    a.href = url;
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  private exportConversation(kind: 'json' | 'md'): void {
    this.exportOpen = false;
    if (kind === 'json') this.download(toJsonExport(this.setup, this.messages, this.model), exportFileName('json'), 'application/json');
    else this.download(toMarkdownExport(this.setup, this.messages, this.model), exportFileName('md'), 'text/markdown');
  }

  // ---- rendering --------------------------------------------------------------------------
  private renderMessage(m: Message, i: number): TemplateResult {
    const isLast = i === this.messages.length - 1;
    const live = this.streaming && isLast && m.role === 'assistant' && m.stats?.finish === null;
    const editing = this.editing?.uid === m.uid;
    const answered = m.tool_calls?.length ? this.messages.slice(i + 1).some((x) => x.role === 'tool') : false;
    return html`<article class="pg-msg role-${m.role} ${live ? 'is-live' : ''} ${editing ? 'is-editing' : ''}" data-uid=${m.uid} data-role=${m.role}>
      <header class="pg-msg-head">
        <span class="pg-role">${m.role}${m.role === 'tool' && m.name ? html` <code>${m.name}</code>` : nothing}</span>
        ${!editing && !live
          ? html`<span class="pg-msg-tools">
              ${copyButton(m.content, 'Copy message')}
              <button type="button" class="btn btn-ghost btn-sm" aria-label="Edit message" title="Edit" @click=${() => this.startEdit(m)}>${icon('edit')}</button>
              <button type="button" class="btn btn-ghost btn-sm" aria-label="Delete message" title="Delete" @click=${() => this.deleteMessage(m.uid)}>${icon('trash')}</button>
            </span>`
          : nothing}
      </header>
      <div class="pg-msg-body">${editing ? this.renderEditor(m) : this.renderBody(m, live, answered)}</div>
      ${m.role === 'assistant' && m.stats && !editing ? html`<footer class="pg-msg-foot num" data-testid="readout-line">${m.stats.finish === null ? html`<span class="pulse"></span> streaming` : readoutLine(m.stats)}${m.stats.finish === 'aborted' ? html` · <span class="warn-ink">aborted</span>` : nothing}</footer>` : nothing}
    </article>`;
  }

  private renderEditor(m: Message): TemplateResult {
    const e = this.editing!;
    return html`<div class="pg-edit">
      <textarea rows="4" .value=${e.text} @input=${(ev: Event) => (this.editing = { ...e, text: (ev.target as HTMLTextAreaElement).value })} @keydown=${(ev: KeyboardEvent) => {
        if (ev.key === 'Escape') this.editing = null;
        if (ev.key === 'Enter' && (ev.metaKey || ev.ctrlKey)) this.saveEdit(e.role === 'user');
      }}></textarea>
      <div class="pg-edit-row">
        <label class="field"><span>role</span><select .value=${e.role} @change=${(ev: Event) => (this.editing = { ...e, role: (ev.target as HTMLSelectElement).value as Role })}>${ROLES.map((r) => html`<option value=${r} ?selected=${r === e.role}>${r}</option>`)}</select></label>
        <span class="grow"></span>
        <button type="button" class="btn btn-ghost btn-sm" @click=${() => (this.editing = null)}>Cancel</button>
        <button type="button" class="btn btn-sm" @click=${() => this.saveEdit(false)}>Save</button>
        ${e.role === 'user' ? html`<button type="button" class="btn btn-primary btn-sm" ?disabled=${this.streaming} @click=${() => this.saveEdit(true)}>${icon('send')} Save & resend</button>` : nothing}
        ${m.role !== e.role ? html`<span class="muted small">role changes to ${e.role}</span>` : nothing}
      </div>
    </div>`;
  }

  private renderBody(m: Message, live: boolean, answered: boolean): TemplateResult {
    if (m.role === 'assistant') {
      const thinkingOnly = live && !m.content.trim() && !m.tool_calls?.length;
      return html`
        ${m.reasoning
          ? html`<details class="pg-think" ?open=${thinkingOnly}>
              <summary>${live && thinkingOnly ? html`<span class="pulse"></span> thinking` : 'thinking'}${m.stats?.usage?.completion_tokens_details?.reasoning_tokens != null ? html` <span class="num muted">${exact(m.stats.usage.completion_tokens_details.reasoning_tokens)} tokens</span>` : nothing}</summary>
              <div class="pg-think-body">${m.reasoning}</div>
            </details>`
          : nothing}
        ${m.content || (live && !m.reasoning && !m.tool_calls?.length) ? html`<div class="pg-md ${live ? 'is-streaming' : ''}">${unsafeHTML(renderMarkdown(m.content))}</div>` : nothing}
        ${m.tool_calls?.length ? this.renderToolCalls(m, answered, live) : nothing}
        ${m.stats?.error ? html`<div class="pg-msg-error" role="alert">${icon('warn')} <span>${m.stats.error.message}</span></div>` : nothing}
      `;
    }
    if (m.role === 'tool') return html`<pre class="pg-tool-result num">${m.content || html`<span class="muted">(empty result)</span>`}</pre>`;
    return html`<div class="pg-text">${m.content || html`<span class="muted">(empty)</span>`}</div>`;
  }

  private renderToolCalls(m: Message, answered: boolean, live: boolean): TemplateResult {
    return html`<div class="pg-calls">
      ${m.tool_calls!.map((c) => {
        const args = prettyArgs(c.function.arguments);
        const tpl = templateFor(c.function.name);
        return html`<div class="pg-call" data-call=${c.id}>
          <div class="pg-call-head"><span class="pg-call-name"><code>${c.function.name || '…'}</code>${c.id ? html`<span class="muted small num">${c.id}</span>` : nothing}</span>${copyButton(c.function.arguments, 'Copy arguments')}</div>
          <pre class="pg-call-args ${args.parsed ? '' : 'is-raw'} ${live ? 'is-streaming' : ''}">${args.text || (live ? '' : '{}')}</pre>
          ${!answered && !live
            ? html`<div class="pg-call-result">
                <label class="field field-block"><span>result for <code>${c.function.name}</code></span>
                  <textarea rows="2" placeholder="what the tool returned (any text; JSON is usual)" .value=${this.toolResults[c.id] ?? ''} @input=${(e: Event) => (this.toolResults = { ...this.toolResults, [c.id]: (e.target as HTMLTextAreaElement).value })}></textarea>
                </label>
                <div class="pg-call-picks">
                  ${tpl ? html`<button type="button" class="btn btn-sm" @click=${() => (this.toolResults = { ...this.toolResults, [c.id]: tpl.example })}>template result</button>` : nothing}
                  <button type="button" class="btn btn-sm" @click=${() => (this.toolResults = { ...this.toolResults, [c.id]: echoResult(c) })}>echo the arguments</button>
                  <button type="button" class="btn btn-sm" @click=${() => (this.toolResults = { ...this.toolResults, [c.id]: '{"ok": true}' })}><code>{"ok": true}</code></button>
                </div>
              </div>`
            : nothing}
        </div>`;
      })}
      ${!answered && !live ? html`<button type="button" class="btn btn-primary pg-send-results" ?disabled=${this.streaming} @click=${() => this.sendToolResults(m)}>${icon('send')} Send result${m.tool_calls!.length > 1 ? 's' : ''} and continue</button>` : nothing}
    </div>`;
  }

  private renderComposer(): TemplateResult {
    const blocked = this.blocked;
    return html`<form class="pg-composer" @submit=${(e: Event) => (e.preventDefault(), this.send())}>
      <label class="sr-only" for="pg-composer">Message</label>
      <textarea id="pg-composer" rows="3" placeholder=${this.messages.length ? 'Reply as user…' : 'Ask the engine something…'} .value=${this.composer} @input=${(e: Event) => (this.composer = (e.target as HTMLTextAreaElement).value)}
        @keydown=${(e: KeyboardEvent) => {
          if (e.key !== 'Enter') return;
          if (e.metaKey || e.ctrlKey || (this.enterSends && !e.shiftKey)) {
            e.preventDefault();
            this.send();
          }
        }}></textarea>
      <div class="pg-composer-row">
        <label class="switch" title="Enter sends; Shift+Enter is a newline. Off: Enter is a newline, ⌘/Ctrl+Enter sends."><input type="checkbox" .checked=${this.enterSends} @change=${(e: Event) => (this.enterSends = (e.target as HTMLInputElement).checked)} /><span>Enter sends</span></label>
        <div class="pg-add">
          <label class="field"><span>+ add</span>
            <select aria-label="Add a message with a role" .value=${''} @change=${(e: Event) => {
              const r = (e.target as HTMLSelectElement).value as Role | '';
              (e.target as HTMLSelectElement).value = '';
              if (r) this.addMessage(r);
            }}>
              <option value="">message…</option>
              ${ROLES.map((r) => html`<option value=${r}>${r}</option>`)}
            </select>
          </label>
        </div>
        <span class="grow"></span>
        ${blocked ? html`<span class="pg-blocked warn-ink small" role="status">${blocked}</span>` : nothing}
        ${this.streaming
          ? html`<button type="button" class="btn pg-stop" @click=${() => this.stop()}>${icon('stop')} Stop</button>`
          : html`<button type="submit" class="btn btn-primary pg-send" ?disabled=${!!blocked || (!this.composer.trim() && !this.messages.length)}>${icon('send')} Send</button>`}
      </div>
    </form>`;
  }

  private renderReadout(): TemplateResult {
    const last = [...this.messages].reverse().find((m) => m.role === 'assistant' && m.stats && m.stats.finish !== null);
    const figs = readoutOf(last?.stats);
    return html`<section class="pg-readout ${this.readoutMore ? 'is-more' : ''}" aria-label="last response readout" data-testid="readout">
      <div class="pg-readout-strip">
        ${figs.map((f) => html`<div class="pg-fig ${f.key ? 'is-key' : ''} ${f.na ? 'is-na' : ''}" data-fig=${f.id} title=${f.title ?? nothing}><span class="pg-fig-label">${f.label}</span><span class="pg-fig-value num">${f.value}${f.unit ? html`<span class="pg-fig-unit">${f.unit}</span>` : nothing}</span></div>`)}
      </div>
      <div class="pg-readout-tools">
        <button type="button" class="btn btn-ghost btn-sm pg-readout-more" aria-expanded=${this.readoutMore} @click=${() => (this.readoutMore = !this.readoutMore)}>${this.readoutMore ? 'less' : 'more'}</button>
        <button type="button" class="btn btn-ghost btn-sm ${this.drawer === 'json' ? 'is-on' : ''}" aria-pressed=${this.drawer === 'json'} ?disabled=${!this.lastRequest} @click=${() => (this.drawer = this.drawer === 'json' ? 'none' : 'json')}>JSON</button>
        <button type="button" class="btn btn-ghost btn-sm ${this.drawer === 'raw' ? 'is-on' : ''}" aria-pressed=${this.drawer === 'raw'} @click=${() => (this.drawer = this.drawer === 'raw' ? 'none' : 'raw')}>Raw request</button>
      </div>
    </section>`;
  }

  private renderDrawer(): TemplateResult | typeof nothing {
    if (this.drawer === 'raw') {
      // the composer's text is the next user message: the drawer shows exactly what Send posts
      const pending = this.composer.trim();
      const body = this.currentRequest(pending ? [{ uid: 'pending', role: 'user', content: pending }] : []);
      const json = JSON.stringify(body, null, 2);
      return html`<section class="pg-drawer" data-drawer="raw">
        <header class="pg-drawer-head"><h3>Raw request</h3><span class="muted small">what the next Send posts to <code>POST /v1/chat/completions</code></span><span class="grow"></span>${copyButton(json, 'Copy JSON')}<button type="button" class="btn btn-ghost btn-sm" @click=${async () => {
          try {
            await navigator.clipboard.writeText(asCurl(body, location.origin));
            this.say('ok', 'curl copied.');
          } catch {
            this.say('warn', 'The clipboard is blocked.');
          }
        }}>copy as curl</button><button type="button" class="btn btn-ghost btn-sm" aria-label="Close" @click=${() => (this.drawer = 'none')}>${icon('x')}</button></header>
        <pre class="pg-json" data-testid="raw-request">${json}</pre>
      </section>`;
    }
    if (this.drawer === 'json') {
      const req = this.lastRequest ? JSON.stringify(this.lastRequest, null, 2) : '';
      const res = this.lastResponse ? JSON.stringify({ message: this.lastResponse.message, finish_reason: this.lastResponse.finish_reason, ...(this.lastResponse.error ? { error: this.lastResponse.error } : {}), usage: this.lastResponse.usage, timings: this.lastResponse.timings, metrics: this.lastResponse.metrics }, null, 2) : '';
      return html`<section class="pg-drawer pg-drawer-2" data-drawer="json">
        <div class="pg-drawer-pane">
          <header class="pg-drawer-head"><h3>Last request</h3><span class="grow"></span>${copyButton(req, 'Copy request')}</header>
          <pre class="pg-json" data-testid="last-request">${req || html`<span class="muted">nothing sent yet</span>`}</pre>
        </div>
        <div class="pg-drawer-pane">
          <header class="pg-drawer-head"><h3>Last response</h3>${this.lastResponse ? html`<span class="muted small num">${this.lastResponse.chunks} chunks · ${exact(this.lastResponse.bytes)} B</span>` : nothing}<span class="grow"></span>${copyButton(res, 'Copy response')}<button type="button" class="btn btn-ghost btn-sm" aria-label="Close" @click=${() => (this.drawer = 'none')}>${icon('x')}</button></header>
          <pre class="pg-json" data-testid="last-response">${res || html`<span class="muted">${this.streaming ? 'streaming…' : 'no response yet'}</span>`}</pre>
        </div>
      </section>`;
    }
    return nothing;
  }

  private renderToolbar(): TemplateResult {
    const canRegen = !this.streaming && this.messages.some((m) => m.role === 'user' || m.role === 'tool');
    return html`<div class="pg-toolbar">
      <button type="button" class="btn btn-sm" @click=${() => this.newChat()}>${icon('plus')} New chat</button>
      <button type="button" class="btn btn-sm" ?disabled=${!canRegen} @click=${() => this.regenerate()}>${icon('refresh')} Regenerate</button>
      <div class="pg-menu">
        <button type="button" class="btn btn-sm" aria-haspopup="menu" aria-expanded=${this.exportOpen} ?disabled=${!this.messages.length} @click=${() => (this.exportOpen = !this.exportOpen)}>${icon('down')} Export</button>
        ${this.exportOpen ? html`<div class="pg-menu-pop" role="menu"><button type="button" role="menuitem" class="btn btn-ghost btn-sm" @click=${() => this.exportConversation('json')}>as JSON</button><button type="button" role="menuitem" class="btn btn-ghost btn-sm" @click=${() => this.exportConversation('md')}>as Markdown</button></div>` : nothing}
      </div>
      <span class="grow"></span>
      ${this.notice ? html`<span class="pg-notice pg-notice-${this.notice.kind}" role="status">${this.notice.text}</span>` : nothing}
      <label class="field pg-preset-pick"><span>preset</span>
        <select aria-label="Load a preset" .value=${this.presetName ?? ''} @change=${(e: Event) => {
          const n = (e.target as HTMLSelectElement).value;
          if (n) this.onPreset({ action: 'load', name: n });
        }}>
          <option value="">${this.presets.length ? 'choose…' : 'none saved'}</option>
          ${this.presets.map((p) => html`<option value=${p.name} ?selected=${p.name === this.presetName}>${p.name}${p.name === this.presetName && this.presetDirty ? ' •' : ''}</option>`)}
        </select>
      </label>
      <button type="button" class="btn btn-sm pg-save-preset" @click=${() => {
        this.setupOpen = true;
        const panel = this.querySelector<QsePgSetup>('qse-pg-setup');
        panel?.show('presets');
        this.updateComplete.then(() => panel?.querySelector<HTMLInputElement>('#pg-preset-name')?.focus());
      }}>Save preset</button>
      <button type="button" class="btn btn-sm pg-setup-toggle" aria-expanded=${this.setupOpen} aria-controls="pg-setup" @click=${() => (this.setupOpen = !this.setupOpen)}>${icon('sliders')} Setup${this.presetDirty || Object.keys(this.setup.params).length || this.setup.system ? html`<span class="pg-dot" aria-hidden="true"></span>` : nothing}</button>
    </div>`;
  }

  render() {
    return html`<div class="view view-playground pg ${this.setupOpen ? 'is-setup-open' : ''}" @click=${(e: Event) => {
      if (this.exportOpen && !(e.target as HTMLElement).closest('.pg-menu')) this.exportOpen = false;
    }}>
      ${this.renderToolbar()}
      <div class="pg-bench">
        <section class="pg-main" aria-label="conversation">
          <div class="pg-transcript" role="log" aria-live="polite" @scroll=${(e: Event) => {
            const el = e.currentTarget as HTMLElement;
            this.stickToEnd = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
          }}>
            ${this.messages.length
              ? repeat(
                  this.messages,
                  (m) => m.uid,
                  (m, i) => this.renderMessage(m, i),
                )
              : html`<div class="pg-empty">
                  <p class="pg-empty-title">A bench for the engine.</p>
                  <p class="muted">Type a message and send it, or set a system prompt, parameters and tools on the right first. Every turn shows the engine's own figures; nothing typed here leaves this browser except the request itself.</p>
                </div>`}
          </div>
          ${this.renderComposer()}
          ${this.renderReadout()}
          ${this.renderDrawer()}
        </section>
        <div class="pg-scrim" @click=${() => (this.setupOpen = false)}></div>
        <aside class="pg-setup" id="pg-setup" aria-label="setup">
          <header class="pg-setup-head"><h2>Setup</h2><span class="grow"></span><button type="button" class="btn btn-ghost btn-icon pg-setup-close" aria-label="Close setup" @click=${() => (this.setupOpen = false)}>${icon('x')}</button></header>
          <qse-pg-setup
            .setup=${this.setup}
            .defaults=${this.defaults}
            .defaultsReady=${this.defaultsReady}
            .presets=${this.presets}
            .presetName=${this.presetName}
            .presetDirty=${this.presetDirty}
            .tools=${this.toolsValidation}
            .paramErrors=${this.paramErrors}
            .storageOk=${this.storageOk}
            @pg-change=${(e: CustomEvent<Setup>) => this.setSetup(e.detail)}
            @pg-preset=${(e: CustomEvent<PresetAction>) => this.onPreset(e.detail)}
          ></qse-pg-setup>
        </aside>
      </div>
    </div>`;
  }
}
