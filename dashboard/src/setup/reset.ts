// <qse-reset>: the reset levels (D9, VIS-82), at the bottom of the System view and on the setup
// page. Five cards from mild to severe; "Review reset" opens a dialog (a full-screen sheet on a
// phone) with exactly what goes and its size, from GET /v1/setup/reset, before anything happens.
// Levels 2 to 5 need their word typed exactly; level 1 is this browser's own (tandem.setup.v1) and
// needs a plain confirm. Every server level is off while requests run. Level 2 hands over to the
// install progress view (the job restarts the engine).

import { html, nothing, type TemplateResult } from 'lit';
import { customElement, property, query, state } from 'lit/decorators.js';
import { describeError } from '../api/client';
import type { SetupDevice, SetupResetLevel, SetupResetLevels, SetupResetResult } from '../api/types';
import { LightElement } from '../ui/base';
import { icon } from '../ui/bits';
import { setupApi } from './api';
import './install';
import '../styles/setup.css';
import { bytesText, confirmState, confirmValue, type ResetLevelNo } from './reset-logic';
import { clearSaved, safeStorage } from './state';
import { R } from './strings';

/** Sizes above this are trees the server may have stopped counting in: "at least". */
const HUGE = 500 * 2 ** 30;

@customElement('qse-reset')
export class QseReset extends LightElement {
  @state() private levels: SetupResetLevels | null = null;
  @state() private loadError: string | null = null;
  @state() private running: number | null = null;
  @state() private openLevel: ResetLevelNo | null = null;
  @state() private typed = '';
  @state() private sending = false;
  @state() private result: { kind: 'ok' | 'partial' | 'error' | 'local'; text: string; res?: SetupResetResult } | null = null;
  @state() private errorText: string | null = null;
  /** the page around it is the setup guide (it starts over after level 1) */
  @property({ type: Boolean }) inSetup = false;
  @query('dialog') private dialog!: HTMLDialogElement | null;
  private poll: ReturnType<typeof setInterval> | null = null;
  private inflight: Promise<void> | null = null;

  connectedCallback(): void {
    super.connectedCallback();
    void this.load();
    this.poll = setInterval(() => {
      if (document.visibilityState === 'visible' && this.levels?.busy) void this.load();
    }, 4000);
  }
  disconnectedCallback(): void {
    if (this.poll) clearInterval(this.poll);
    super.disconnectedCallback();
  }

  private async load(): Promise<void> {
    try {
      const [lv, dev] = await Promise.all([setupApi.resetLevels(), setupApi.device().catch(() => null as SetupDevice | null)]);
      this.levels = lv;
      this.running = dev ? dev.engine.running + dev.engine.waiting : null;
      this.loadError = null;
    } catch (e) {
      const d = describeError(e);
      this.loadError = d.status === 401 ? R.unauthorized : d.status === 0 ? R.offline : `${R.loadError} ${d.message}`;
    }
  }

  private level(n: ResetLevelNo): SetupResetLevel | null {
    return this.levels?.levels.find((l) => l.level === n) ?? null;
  }

  private review(n: ResetLevelNo): void {
    this.openLevel = n;
    this.typed = '';
    this.result = null;
    this.errorText = null;
    void this.updateComplete.then(() => {
      const d = this.dialog;
      if (d && !d.open) d.showModal();
      d?.querySelector<HTMLInputElement>('.reset-word')?.focus();
    });
    if (n > 1) void this.load();
  }

  private closeDialog(): void {
    this.dialog?.close();
    this.openLevel = null;
  }

  /** one request however often it is clicked */
  private confirm(): Promise<void> {
    return (this.inflight ??= this.doConfirm().finally(() => (this.inflight = null)));
  }

  private async doConfirm(): Promise<void> {
    const n = this.openLevel;
    if (n == null) return;
    if (n === 1) {
      clearSaved(safeStorage());
      window.dispatchEvent(new CustomEvent('qse-setup-reset'));
      this.result = { kind: 'local', text: R.doneLocal };
      return;
    }
    const lv = this.level(n);
    if (!lv) return;
    const cs = confirmState(lv, this.typed, { busy: !!this.levels?.busy, sending: this.sending });
    if (!cs.enabled) return;
    this.sending = true;
    this.errorText = null;
    try {
      const res = await setupApi.reset({ level: lv.level, confirm: confirmValue(lv, this.typed) });
      const freed = res.removed.reduce((a, r) => a + r.bytes, 0);
      if (res.ok === false || (res.failed && res.failed.length)) this.result = { kind: 'partial', text: R.partial(res.failed?.length ?? 0), res };
      else this.result = { kind: 'ok', text: res.job ? res.message : R.done(bytesText(freed)), res };
      window.dispatchEvent(new CustomEvent('qse-system-refresh'));
      void this.load();
    } catch (e) {
      const d = describeError(e);
      if (d.status === 409) {
        this.errorText = R.busy(this.running);
        void this.load();
      } else if (d.status === 401 || d.status === 403) this.errorText = R.unauthorized;
      else if (d.status === 0) this.errorText = R.offline;
      else this.errorText = `${R.error} ${d.message}`;
    } finally {
      this.sending = false;
    }
  }

  render() {
    const busy = !!this.levels?.busy;
    return html`<div class="reset" data-busy=${busy}>
      <p class="reset-intro muted">${R.intro}</p>
      ${this.loadError ? html`<div class="reset-load-error" role="alert">${icon('warn')} <span>${this.loadError}</span> <button class="btn btn-sm" @click=${() => void this.load()}>${icon('refresh')} ${R.retry}</button></div>` : nothing}
      ${busy ? html`<p class="reset-busy" id="reset-busy" role="status" data-reset-busy>${icon('info')} ${R.busy(this.running)}</p>` : nothing}
      <ol class="reset-levels">${([1, 2, 3, 4, 5] as ResetLevelNo[]).map((n) => this.card(n, busy))}</ol>
      ${this.renderDialog()}
    </div>`;
  }

  private card(n: ResetLevelNo, busy: boolean): TemplateResult {
    const lv = n === 1 ? null : this.level(n);
    const server = n > 1;
    const disabled = server && (busy || !lv);
    const danger = n === 5;
    return html`<li class="reset-card ${danger ? 'is-danger' : ''}" data-level=${n}>
      <span class="reset-n num" aria-hidden="true">${n}</span>
      <div class="reset-text">
        <h3 class="reset-title">${R.levelTitle[n]} ${danger ? html`<span class="reset-danger-tag">${icon('warn')} ${R.danger}</span>` : nothing}</h3>
        <p class="muted small">${lv?.what ?? R.levelWhat[n]}</p>
        <p class="reset-meta small">
          ${server ? html`<span class="num">${lv ? (lv.bytes >= HUGE ? R.atLeast(bytesText(lv.bytes)) : bytesText(lv.bytes)) : '–'}</span> <span class="muted">${lv?.restart ? R.restart : R.noRestart}</span>` : html`<span class="muted">${R.browserOnly}</span>`}
        </p>
      </div>
      <button class="btn ${danger ? 'btn-danger' : ''} reset-review" ?disabled=${disabled} @click=${() => this.review(n)} aria-describedby=${busy && server ? 'reset-busy' : nothing}>${R.review}</button>
    </li>`;
  }

  private renderDialog(): TemplateResult {
    const n = this.openLevel;
    const lv = n && n > 1 ? this.level(n) : null;
    const busy = !!this.levels?.busy;
    const cs = n === 1 ? { enabled: !this.result, reason: null } : confirmState(lv, this.typed, { busy, sending: this.sending });
    const items = lv?.items ?? [];
    const present = items.filter((i) => i.exists);
    const word = lv?.confirm_word ?? null;
    const done = this.result != null;
    return html`<dialog class="reset-dialog ${n === 5 ? 'is-danger' : ''}" aria-labelledby="reset-dialog-title" @close=${() => (this.openLevel = null)} @cancel=${() => (this.openLevel = null)}>
      ${n == null
        ? nothing
        : html`<form method="dialog" class="reset-dialog-body" @submit=${(e: Event) => {
            e.preventDefault();
            if (!done) void this.confirm();
            else this.closeDialog();
          }}>
            <header class="reset-dialog-head">
              <h2 id="reset-dialog-title">${R.levelTitle[n]}</h2>
              <button type="button" class="btn btn-ghost btn-icon" aria-label=${R.close} @click=${() => this.closeDialog()}>${icon('x')}</button>
            </header>
            <p class="muted">${lv?.what ?? R.levelWhat[n]}</p>
            ${n > 1
              ? html`<h3 class="reset-dialog-sub">${R.dialogItems}</h3>
                  ${present.length
                    ? html`<ul class="reset-items">
                          ${items.map((i) => html`<li class=${i.exists ? '' : 'is-missing'}><span class="reset-item-label">${i.label}<code>${i.path}</code></span><span class="num">${i.exists ? bytesText(i.bytes) : R.missing}</span></li>`)}
                        </ul>
                        <p class="reset-total"><span>${R.dialogTotal(lv!.bytes >= HUGE ? R.atLeast(bytesText(lv!.bytes)) : bytesText(lv!.bytes))}</span><span class="muted">${lv!.restart ? R.restart : R.noRestart}</span></p>`
                    : html`<p class="reset-nothing" data-reset-empty>${R.dialogNothing}</p>`}`
              : nothing}
            ${busy && n > 1 && !done ? html`<p class="reset-busy" role="status">${icon('info')} ${R.busy(this.running)}</p>` : nothing}
            ${word && present.length && !done
              ? html`<label class="reset-confirm">
                  <span>${R.typeWord('')}<b class="reset-word-show">${word}</b></span>
                  <input class="reset-word" type="text" .value=${this.typed} aria-label=${R.typeWord(word)} autocomplete="off" autocapitalize="off" autocorrect="off" spellcheck="false" inputmode="text" @input=${(e: Event) => (this.typed = (e.target as HTMLInputElement).value)} />
                </label>`
              : nothing}
            ${this.errorText ? html`<p class="reset-error" role="alert">${icon('warn')} ${this.errorText}</p>` : nothing}
            ${this.result ? this.renderResult() : nothing}
            <footer class="reset-dialog-foot">
              ${done
                ? html`<button type="button" class="btn btn-primary" @click=${() => this.closeDialog()}>${R.close}</button>`
                : html`<button type="button" class="btn" @click=${() => this.closeDialog()}>${R.cancel}</button>
                    <button type="submit" class="btn ${n >= 3 ? 'btn-danger' : 'btn-primary'} reset-go" ?disabled=${!cs.enabled} data-reason=${cs.reason ?? nothing}>${this.sending ? R.working : n === 1 ? R.confirmLocal : R.confirm(n)}</button>`}
            </footer>
          </form>`}
    </dialog>`;
  }

  private renderResult(): TemplateResult {
    const r = this.result!;
    const res = r.res;
    return html`<div class="reset-result is-${r.kind}" role="status" data-reset-result=${r.kind}>
      <p>${r.kind === 'ok' || r.kind === 'local' ? icon('check') : icon('warn')} ${r.text}</p>
      ${res?.failed?.length ? html`<ul class="reset-failed">${res.failed.map((f) => html`<li><code>${f.path}</code> <span class="muted">${f.message}</span></li>`)}</ul>` : nothing}
      ${res?.restart_needed && !res.job ? html`<p class="muted small">${R.restartNeeded}</p>` : nothing}
      ${res?.job ? html`<qse-install-progress .jobId=${res.job.id} .initial=${res.job} compact></qse-install-progress>` : nothing}
      ${r.kind === 'local' && !this.inSetup ? html`<p class="small"><a href="#/setup">${R.openSetup}</a></p>` : nothing}
    </div>`;
  }
}
