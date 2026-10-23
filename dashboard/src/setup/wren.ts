// <qse-wren>: the Setup Guide's assistant panel (D1, D6, D7, VIS-80). Docked right on a desktop; at
// 1100 px and below a floating figure opens it as a bottom sheet (focus kept inside, Esc closes,
// the page behind does not scroll). The sheet stays open while Wren works: a change shows inside
// it as a toast with Show me, which closes the sheet at the field (VIS-91; it dropped to a peek
// and rose again once per call, refused ones too).
//
// Asleep: a Wake button loads the model (POST /v1/setup/assistant/load, status polled every 500 ms
// for the progress ring); a refusal for memory says so with the numbers, the wizard works without
// it. Ready: the chat streams from POST /v1/setup/assistant/chat; every tool call is executed by
// the page through `host.execute` (validated there, the same actions as a click), the result goes
// back as a tool message and the turn continues as wren-turn.ts nextRound says: a refused change
// gets one repair, and the last round asks for words with the tools off, so a turn ends in words.
// A turn shows at most one notice. Stop aborts and keeps the text. Finished: after Finish Wren is
// asleep while the setup runs.

import { html, nothing, type TemplateResult } from 'lit';
import { customElement, property, query, state } from 'lit/decorators.js';
import { repeat } from 'lit/directives/repeat.js';
import { unsafeHTML } from 'lit/directives/unsafe-html.js';
import { ApiError, describeError } from '../api/client';
import type { SetupAssistantStatus, SetupChatEvent, SetupChatMessage, SetupPageState, SetupSource, SetupStepId, SetupToolCall } from '../api/types';
import { renderMarkdown } from '../playground/markdown';
import { LightElement } from '../ui/base';
import { icon } from '../ui/bits';
import { setupApi, streamChat } from './api';
import './figure';
import type { QseWrenFigure } from './figure';
import { nextFigureState, type FigureEvent, type FigureState } from './figure-state';
import { S } from './strings';
import { nextRound, toastFor, turnNotices, type RoundCall } from './wren-turn';

export interface ToolOutcome {
  ok: boolean;
  /** what goes back to the model */
  result: string;
  /** the chip's sentence ("Wren turned off Vision"); null: nothing changed on the page */
  summary: string | null;
  /** the change can be undone (this session only) */
  undo: (() => void) | null;
  /** scroll to the changed field and highlight it (the phone's Show me) */
  reveal?: (() => void) | null;
  stale?: boolean;
}

/** What the panel needs from the wizard. */
export interface WrenHost {
  pageState(): SetupPageState;
  /** validate and execute one call (the hop and the highlight happen there) */
  execute(call: SetupToolCall, turnStartedAt: number): Promise<ToolOutcome>;
  /** the chat history changed (the wizard saves it in tandem.setup.v1) */
  saveChat(chat: SetupChatMessage[]): void;
  /** MemAvailable now, GiB, for the "cannot load" sentence */
  memAvailable(): number | null;
}

type Phase = 'checking' | 'asleep' | 'loading' | 'ready' | 'refused' | 'failed' | 'finished';

interface Chip {
  summary: string | null;
  ok: boolean;
  stale?: boolean;
  undo: (() => void) | null;
  undone: boolean;
}

let callSeq = 0;

/** the server's reasons start in lower case ("not enough memory beside the engine: ...") */
function sentence(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

@customElement('qse-wren')
export class QseWren extends LightElement {
  @property({ attribute: false }) host: WrenHost | null = null;
  @property({ attribute: false }) step: SetupStepId = 'models';
  @property({ attribute: false }) chat: SetupChatMessage[] = [];
  /** Finish was pressed: asleep while the setup runs */
  @property({ type: Boolean }) finished = false;

  @state() private phase: Phase = 'checking';
  @state() private status: SetupAssistantStatus | null = null;
  @state() private figure: FigureState = 'sleeping';
  @state() private progress: number | null = null;
  @state() private streaming = false;
  @state() private partial = '';
  @state() private composer = '';
  @state() private error: string | null = null;
  @state() private notice: string | null = null;
  @state() private sources: SetupSource[] = [];
  @state() private open = false;
  /** the phone's last change: a toast inside the sheet, and its chip in the log pulses */
  @state() private toast: { id: string; summary: string; reveal: (() => void) | null } | null = null;
  @state() private chips = new Map<string, Chip>();
  @query('qse-wren-figure.wren-main') private fig!: QseWrenFigure | null;
  @query('qse-wren-figure.wren-fab-fig') private fabFig!: QseWrenFigure | null;
  @query('.wren-log') private log!: HTMLElement | null;
  @query('.wren-input') private input!: HTMLTextAreaElement | null;

  private ctrl: AbortController | null = null;
  private loadPoll: ReturnType<typeof setTimeout> | null = null;
  private ping: ReturnType<typeof setInterval> | null = null;
  private loadStartedAt = 0;
  private lastUserText = '';
  private opener: HTMLElement | null = null;
  private toastTimer: ReturnType<typeof setTimeout> | null = null;

  connectedCallback(): void {
    super.connectedCallback();
    void this.checkStatus();
    this.ping = setInterval(() => {
      if (this.phase === 'ready' && document.visibilityState === 'visible') void this.checkStatus(true);
    }, 30_000);
    document.addEventListener('keydown', this.onKey);
  }
  disconnectedCallback(): void {
    this.ctrl?.abort();
    if (this.loadPoll) clearTimeout(this.loadPoll);
    if (this.ping) clearInterval(this.ping);
    if (this.toastTimer) clearTimeout(this.toastTimer);
    document.removeEventListener('keydown', this.onKey);
    this.lockScroll(false);
    super.disconnectedCallback();
  }

  protected updated(changed: Map<string, unknown>): void {
    if (changed.has('finished') && this.finished && this.phase !== 'finished') void this.sleepForInstall();
  }

  private emit(ev: FigureEvent): void {
    this.figure = nextFigureState(this.figure, ev);
  }

  // ---- load / unload ------------------------------------------------------------------------

  private async checkStatus(quiet = false): Promise<void> {
    if (this.finished) {
      this.phase = 'finished';
      this.figure = 'sleeping';
      return;
    }
    try {
      const s = await setupApi.assistant.status();
      this.applyStatus(s);
    } catch (e) {
      if (quiet) return;
      this.phase = 'failed';
      this.error = describeError(e).status === 0 ? S.wrenOffline : describeError(e).message;
      this.emit('error');
    }
  }

  private applyStatus(s: SetupAssistantStatus): void {
    this.status = s;
    switch (s.state) {
      case 'loaded':
        if (this.phase !== 'ready') {
          this.phase = 'ready';
          this.progress = null;
          this.figure = 'loading';
          this.emit('loaded');
        }
        break;
      case 'loading':
        this.phase = 'loading';
        this.progress = this.timeProgress(s.loading_s ?? null);
        this.emit('load-progress');
        this.scheduleLoadPoll();
        break;
      case 'refused':
        this.phase = 'refused';
        this.emit('error');
        break;
      case 'failed':
        this.phase = 'failed';
        this.error = s.reason ?? S.wrenFailed;
        this.emit('error');
        break;
      default:
        this.phase = s.model == null ? 'refused' : 'asleep';
        this.figure = 'sleeping';
    }
  }

  /** without a server figure: a curve over the seconds since the load started (the server's
   *  loading_s, else this page's clock) that approaches 95 % over about 20 s */
  private timeProgress(loadingS: number | null): number {
    const t = loadingS ?? (Date.now() - this.loadStartedAt) / 1000;
    return Math.min(0.95, 1 - Math.exp(-t / 8));
  }

  private scheduleLoadPoll(): void {
    if (this.loadPoll) clearTimeout(this.loadPoll);
    this.loadPoll = setTimeout(() => void this.checkStatus(true), 500);
  }

  private async wake(): Promise<void> {
    this.error = null;
    this.loadStartedAt = Date.now();
    this.phase = 'loading';
    this.progress = 0;
    this.emit('load-start');
    try {
      this.applyStatus(await setupApi.assistant.load());
    } catch (e) {
      const d = describeError(e);
      const body = e instanceof ApiError ? (e.body as { assistant?: SetupAssistantStatus } | undefined) : undefined;
      if (body?.assistant) this.status = body.assistant;
      if (d.status === 409) {
        this.phase = 'refused';
        if (body?.assistant?.reason) this.status = { ...(this.status as SetupAssistantStatus), reason: body.assistant.reason };
      } else {
        this.phase = 'failed';
        this.error = d.status === 0 ? S.wrenOffline : d.message;
      }
      this.emit('error');
    }
  }

  private async sleep(): Promise<void> {
    this.stop();
    try {
      this.applyStatus(await setupApi.assistant.unload());
    } catch {
      /* the status ping finds out */
    }
    this.phase = 'asleep';
    this.emit('unload');
  }

  /** Finish: stop any answer, unload, and stay asleep while the setup runs. */
  async sleepForInstall(): Promise<void> {
    this.ctrl?.abort();
    this.phase = 'finished';
    this.emit('unload');
    this.close();
    try {
      await setupApi.assistant.unload();
    } catch {
      /* the apply job unloads it too (its unload_assistant step) */
    }
  }

  // ---- chat ---------------------------------------------------------------------------------

  private stop(): void {
    if (!this.ctrl) return;
    this.ctrl.abort();
  }

  private async send(text: string): Promise<void> {
    const t = text.trim();
    if (!t || this.streaming || !this.host) return;
    this.lastUserText = t;
    this.composer = '';
    this.error = null;
    this.notice = null;
    this.toast = null;
    const history: SetupChatMessage[] = [...this.chat, { role: 'user', content: t }];
    this.setChat(history);
    await this.turn(history);
  }

  private async retry(): Promise<void> {
    if (!this.lastUserText) return;
    this.error = null;
    const last = this.chat[this.chat.length - 1];
    const history = last?.role === 'user' ? this.chat : [...this.chat, { role: 'user' as const, content: this.lastUserText }];
    this.setChat(history);
    await this.turn(history);
  }

  private setChat(c: SetupChatMessage[]): void {
    this.chat = c;
    this.host?.saveChat(c);
    void this.updateComplete.then(() => this.scrollLog());
  }

  private async turn(start: SetupChatMessage[]): Promise<void> {
    const host = this.host!;
    const turnStartedAt = Date.now();
    let history = start;
    this.streaming = true;
    this.emit('send');
    this.ctrl = new AbortController();
    const signal = this.ctrl.signal;
    let tools = true;
    let retried = false;
    let unchanged = false;
    try {
      for (let round = 1; ; round++) {
        let text = '';
        const calls: SetupToolCall[] = [];
        const toolMsgs: SetupChatMessage[] = [];
        const outcomes: RoundCall[] = [];
        let first = true;
        let failure: string | null = null;
        this.partial = '';
        await streamChat(
          { messages: history, page: host.pageState(), ...(tools ? {} : { tools: false }) },
          (ev: SetupChatEvent) => {
            if (ev.type === 'state') this.emit(ev.state === 'thinking' ? 'state-thinking' : ev.state === 'loading' ? 'load-progress' : 'first-token');
            else if (ev.type === 'text') {
              text += ev.delta;
              this.partial = text;
              if (first) {
                first = false;
                this.emit('first-token');
              } else this.emit('delta');
              this.fig?.blip();
              this.fabFig?.blip();
              this.scrollLog();
            } else if (ev.type === 'tool_call') {
              calls.push({ id: ev.id || `call_${++callSeq}`, name: ev.name, args: ev.args });
            } else if (ev.type === 'done') {
              this.sources = ev.sources ?? [];
              if (ev.unchanged) unchanged = true;
            } else if (ev.type === 'error') failure = ev.message;
          },
          signal,
        );
        if (failure && /another tab/i.test(failure)) {
          // one chat at a time on the box: say so, no error pose
          this.notice = S.wrenBusy;
          break;
        }
        if (failure) throw new ApiError('/v1/setup/assistant/chat', 500, failure, 'stream');
        // the tool calls run in order; a valid one hops and highlights, a refused one only goes
        // back to the model, and none of them moves the phone's sheet
        for (const call of calls) {
          // Stop means stop: buffered tool calls must not run after it.
          if (signal.aborted) break;
          const out = await host.execute(call, turnStartedAt);
          if (out.ok) {
            this.emit('tool-call');
            void this.fig?.hop();
            this.emit('acted');
          }
          toolMsgs.push({ role: 'tool', tool_call_id: call.id, content: JSON.stringify(out.ok ? { ok: true, result: out.result, ...(out.summary ? { summary: out.summary } : {}) } : { ok: false, error: out.result, ...(out.stale ? { stale: true } : {}) }) });
          this.chips = new Map(this.chips).set(call.id, { summary: out.summary, ok: out.ok, stale: out.stale, undo: out.undo, undone: false });
          outcomes.push({ name: call.name, ok: out.ok, changed: !!out.summary });
          const toast = toastFor(out, this.open && this.sheetMode());
          if (toast) this.showToast(call.id, toast, out.reveal ?? null);
        }
        const assistant: SetupChatMessage = { role: 'assistant', content: text, ...(calls.length ? { tool_calls: calls } : {}) };
        history = [...history, assistant, ...toolMsgs];
        this.partial = '';
        this.setChat(history);
        const next = nextRound({ round, tools, text, calls: outcomes, retried });
        if (!next) break;
        ({ tools, retried } = next);
      }
      // the server refused a change twice and nothing changed: one notice, unless the page's own
      // refusals already gave this turn its notice
      if (unchanged && !turnNotices(history).has(history.length - 1)) this.notice = S.wrenNotChanged;
      this.emit('done');
    } catch (e) {
      if ((e as Error).name === 'AbortError') {
        // Stop: keep what came, marked
        if (this.partial) this.setChat([...history, { role: 'assistant', content: `${this.partial} ${S.wrenStopped}` }]);
        this.partial = '';
        this.emit('stop');
      } else {
        const d = describeError(e);
        if (d.status === 409) this.notice = S.wrenBusy;
        else if (d.status === 0) this.error = S.wrenOffline;
        else if (d.status === 503 || /unload|asleep|not loaded/i.test(d.message)) {
          this.error = S.wrenWentToSleep;
          void this.checkStatus(true);
        } else this.error = d.message || S.wrenError;
        if (this.partial) this.setChat([...history, { role: 'assistant', content: this.partial }]);
        this.partial = '';
        this.emit(d.status === 409 ? 'done' : 'error');
      }
    } finally {
      this.streaming = false;
      this.ctrl = null;
    }
  }

  private undoChip(id: string): void {
    const c = this.chips.get(id);
    if (!c?.undo || c.undone) return;
    c.undo();
    this.chips = new Map(this.chips).set(id, { ...c, undone: true });
  }

  private scrollLog(): void {
    const l = this.log;
    if (l) l.scrollTop = l.scrollHeight;
  }

  // ---- the sheet (1100 px and below) --------------------------------------------------------

  private sheetMode(): boolean {
    return typeof matchMedia === 'function' && matchMedia('(max-width: 1100px)').matches;
  }

  private toggleOpen(): void {
    if (this.open) this.close();
    else {
      this.opener = document.activeElement as HTMLElement | null;
      this.open = true;
      this.lockScroll(true);
      void this.updateComplete.then(() => (this.querySelector<HTMLElement>('.wren-panel .wren-input, .wren-panel .btn-primary, .wren-close') ?? null)?.focus());
    }
  }

  private close(): void {
    if (!this.open) return;
    this.open = false;
    this.lockScroll(false);
    this.opener?.focus?.();
  }

  private lockScroll(on: boolean): void {
    document.documentElement.classList.toggle('wren-sheet-open', on && this.sheetMode());
  }

  /** a change on a phone: the toast stays 12 s, or until Show me or the next question */
  private showToast(id: string, summary: string, reveal: (() => void) | null): void {
    this.toast = { id, summary, reveal };
    if (this.toastTimer) clearTimeout(this.toastTimer);
    this.toastTimer = setTimeout(() => (this.toast = null), 12_000);
  }

  /** Show me: the sheet closes and the page scrolls to the field, which pulses */
  private async showMe(): Promise<void> {
    const reveal = this.toast?.reveal;
    this.toast = null;
    this.close();
    await this.updateComplete;
    requestAnimationFrame(() => reveal?.());
  }

  private onKey = (e: KeyboardEvent): void => {
    if (!this.open || !this.sheetMode()) return;
    if (e.key === 'Escape') {
      e.preventDefault();
      this.close();
      return;
    }
    if (e.key !== 'Tab') return;
    const panel = this.querySelector<HTMLElement>('.wren-panel');
    if (!panel) return;
    const items = [...panel.querySelectorAll<HTMLElement>('button:not([disabled]), textarea:not([disabled]), a[href], input:not([disabled])')].filter((x) => x.offsetParent !== null);
    if (!items.length) return;
    const first = items[0];
    const last = items[items.length - 1];
    if (e.shiftKey && document.activeElement === first) {
      e.preventDefault();
      last.focus();
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault();
      first.focus();
    } else if (!panel.contains(document.activeElement)) {
      e.preventDefault();
      first.focus();
    }
  };

  // ---- render -------------------------------------------------------------------------------

  private statusWord(): string {
    if (this.phase === 'finished') return S.wrenFinished;
    if (this.phase === 'asleep') return S.wrenAsleep;
    if (this.phase === 'refused') return S.wrenCannotLoad;
    if (this.phase === 'failed') return S.wrenFailed;
    if (this.phase === 'loading') return S.wrenWaking(this.progress);
    return S.wrenStatus[this.figure] ?? '';
  }

  render() {
    const st = this.figure;
    return html`
      <button class="wren-fab ${this.open ? 'is-open' : ''}" aria-label=${this.open ? S.wrenClose : S.wrenOpen} aria-expanded=${this.open} aria-controls="wren-panel" @click=${() => this.toggleOpen()} data-figure=${st}>
        <qse-wren-figure class="wren-fab-fig" .state=${st} .size=${50} .progress=${this.progress}></qse-wren-figure>
      </button>
      <div class="wren-scrim ${this.open ? 'is-open' : ''}" @click=${() => this.close()}></div>
      <aside id="wren-panel" class="wren-panel ${this.open ? 'is-open' : ''}" aria-label=${S.wrenName} data-phase=${this.phase} data-figure=${st} role="complementary">
        <div class="wren-handle" aria-hidden="true"></div>
        <header class="wren-head">
          <qse-wren-figure class="wren-main" .state=${st} .size=${76} .progress=${this.progress}></qse-wren-figure>
          <div class="wren-titles">
            <h2 class="wren-name">${S.wrenName}</h2>
            <p class="wren-status" role="status" aria-live="polite" data-wren-status>${this.statusWord()}</p>
            ${this.status?.model && this.phase === 'ready' ? html`<p class="wren-model">${this.status.model}${this.status.fallback ? html` <span title=${S.wrenFallback}>(${S.wrenFallback})</span>` : nothing}</p>` : nothing}
          </div>
          <div class="wren-head-tools">
            ${this.phase === 'ready' && !this.streaming ? html`<button class="btn btn-ghost btn-icon" title=${S.wrenSleep} aria-label=${S.wrenSleep} @click=${() => void this.sleep()}>${icon('moon')}</button>` : nothing}
            <button class="btn btn-ghost btn-icon wren-close" aria-label=${S.wrenClose} title=${S.wrenClose} @click=${() => this.close()}>${icon('x')}</button>
          </div>
        </header>
        ${this.renderBody()}
      </aside>
    `;
  }

  private hero(): TemplateResult {
    return html`<qse-wren-figure class="wren-hero" .state=${this.figure} .size=${148} .progress=${this.progress}></qse-wren-figure>`;
  }

  private renderBody(): TemplateResult {
    switch (this.phase) {
      case 'checking':
        return html`<div class="wren-body wren-center">${this.hero()}</div>`;
      case 'finished':
        return html`<div class="wren-body wren-center" data-wren-finished>${this.hero()}<p>${S.wrenFinished}</p></div>${this.renderComposer(true)}`;
      case 'asleep':
        return html`<div class="wren-body wren-center">
            ${this.hero()}
            <p class="wren-lead">${S.wrenAsleep}</p>
            <p class="muted small">${S.wrenAsleepBody}</p>
            <button class="btn btn-primary wren-wake" @click=${() => void this.wake()}>${S.wrenWake}</button>
          </div>`;
      case 'loading':
        return html`<div class="wren-body wren-center">
            ${this.hero()}
            <p class="wren-lead">${S.wrenWaking(this.progress)}</p>
            <div class="wren-loadbar" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow=${this.progress != null ? Math.round(this.progress * 100) : nothing} aria-label=${S.wrenWaking(null)}><span style="width:${Math.round((this.progress ?? 0) * 100)}%"></span></div>
          </div>`;
      case 'refused': {
        const s = this.status;
        const why = s?.model == null && s != null ? S.wrenNoModel : S.wrenCannotLoadMemory(s?.claim_gib ?? null, this.host?.memAvailable() ?? null);
        return html`<div class="wren-body wren-center" data-wren-refused>
            ${this.hero()}
            <p class="wren-lead">${S.wrenCannotLoad}</p>
            <p class="muted small">${s?.reason ? sentence(s.reason) : why}</p>
            ${s?.reason ? html`<p class="muted small">${why}</p>` : nothing}
            ${s?.model != null ? html`<button class="btn wren-wake" @click=${() => void this.wake()}>${icon('refresh')} ${S.wrenRetry}</button>` : nothing}
          </div>`;
      }
      case 'failed':
        return html`<div class="wren-body wren-center" data-wren-failed>
            ${this.hero()}
            <p class="wren-lead">${S.wrenFailed}</p>
            <p class="muted small">${this.error ?? ''}</p>
            <button class="btn wren-wake" @click=${() => void this.wake()}>${icon('refresh')} ${S.wrenRetry}</button>
          </div>`;
      case 'ready':
        return this.renderChat();
    }
  }

  private renderChat(): TemplateResult {
    const msgs = this.chat;
    const notes = turnNotices(msgs);
    return html`
      <div class="wren-body wren-log" aria-live="off">
        ${msgs.length === 0 ? html`<p class="wren-msg is-assistant wren-hello">${S.wrenHello}</p>` : nothing}
        ${repeat(
          msgs.map((m, i) => ({ m, i })),
          ({ i }) => i,
          ({ m, i }) => html`${this.renderMessage(m, i === msgs.length - 1)}${notes.has(i) ? html`<p class="wren-note" data-tool-rejected>${notes.get(i) === 'stale' ? S.tool.stale : S.wrenNotChanged}</p>` : nothing}`,
        )}
        ${this.partial ? html`<div class="wren-msg is-assistant is-streaming">${unsafeHTML(renderMarkdown(this.partial))}</div>` : nothing}
        ${this.streaming && !this.partial ? html`<div class="wren-msg is-assistant wren-typing" aria-hidden="true"><span></span><span></span><span></span></div>` : nothing}
        ${this.notice ? html`<p class="wren-note">${this.notice}</p>` : nothing}
        ${this.error
          ? html`<div class="wren-error" role="alert" data-wren-error>
              <span>${this.error}</span>
              <button class="btn btn-sm" @click=${() => void this.retry()}>${icon('refresh')} ${S.wrenRetry}</button>
            </div>`
          : nothing}
      </div>
      ${this.toast
        ? html`<div class="wren-toast" role="status" data-wren-toast>
            <span class="wren-chip-dot" aria-hidden="true"></span>
            <span class="wren-toast-text">${this.toast.summary}</span>
            ${this.toast.reveal ? html`<button class="btn btn-sm" @click=${() => void this.showMe()}>${S.wrenShowMe}</button>` : nothing}
          </div>`
        : nothing}
      ${this.renderChips()} ${this.renderComposer(false)}
    `;
  }

  private renderMessage(m: SetupChatMessage, last: boolean): TemplateResult | typeof nothing {
    if (m.role === 'user') return html`<div class="wren-msg is-user">${m.content}</div>`;
    if (m.role === 'assistant') {
      if (!m.content.trim()) return nothing;
      return html`<div class="wren-msg is-assistant">${unsafeHTML(renderMarkdown(m.content))}
        ${last && this.sources.length ? html`<p class="wren-sources">${S.wrenSources} ${this.sources.map((s, i) => html`${i ? ', ' : ''}<span title=${s.path}>${s.title || s.path}</span>`)}</p>` : nothing}
      </div>`;
    }
    // a tool result: a chip with Undo; a refused call shows nothing here (turnNotices gives the
    // turn at most one notice)
    let parsed: { ok?: boolean; summary?: string; error?: string; stale?: boolean } = {};
    try {
      parsed = JSON.parse(m.content);
    } catch {
      /* an old or foreign message */
    }
    const chip = m.tool_call_id ? this.chips.get(m.tool_call_id) : undefined;
    if (parsed.ok === false || chip?.ok === false) return nothing;
    const summary = chip?.summary ?? parsed.summary;
    if (!summary) return nothing;
    return html`<div class="wren-chip ${chip?.undone ? 'is-undone' : ''} ${this.toast?.id === m.tool_call_id ? 'is-fresh' : ''}" data-tool-chip>
      <span class="wren-chip-dot" aria-hidden="true"></span>
      <span>${summary}</span>
      ${chip?.undo ? html`<button class="btn btn-ghost btn-sm" ?disabled=${chip.undone} @click=${() => this.undoChip(m.tool_call_id!)}>${chip.undone ? S.wrenUndone : S.wrenUndo}</button>` : nothing}
    </div>`;
  }

  private renderChips(): TemplateResult | typeof nothing {
    if (this.streaming) return nothing;
    const chips = S.wrenChips[this.step] ?? [];
    return html`<div class="wren-suggest">${chips.map((c) => html`<button class="wren-suggestion" @click=${() => void this.send(c)}>${c}</button>`)}</div>`;
  }

  private renderComposer(disabled: boolean): TemplateResult {
    return html`<form
      class="wren-composer"
      @submit=${(e: Event) => {
        e.preventDefault();
        void this.send(this.composer);
      }}
    >
      <label class="sr-only" for="wren-input">${S.wrenPlaceholder}</label>
      <textarea
        id="wren-input"
        class="wren-input"
        rows="1"
        .value=${this.composer}
        ?disabled=${disabled}
        placeholder=${disabled ? S.wrenFinished : S.wrenPlaceholder}
        @focus=${() => this.emit('focus')}
        @blur=${() => !this.composer && this.emit('blur')}
        @input=${(e: Event) => {
          this.composer = (e.target as HTMLTextAreaElement).value;
          this.emit('input');
        }}
        @keydown=${(e: KeyboardEvent) => {
          if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
            e.preventDefault();
            void this.send(this.composer);
          }
        }}
      ></textarea>
      ${this.streaming
        ? html`<button type="button" class="btn wren-stop" @click=${() => this.stop()} aria-label=${S.wrenStop}>${icon('stop')}<span>${S.wrenStop}</span></button>`
        : html`<button type="submit" class="btn btn-primary wren-send" ?disabled=${disabled || !this.composer.trim()} aria-label=${S.wrenSend}>${icon('send')}</button>`}
    </form>`;
  }
}
