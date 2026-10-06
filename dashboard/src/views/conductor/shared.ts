// Shared parts of the Conductor view: the editor controller every config screen uses (a draft of
// one document, the server's own validation as you type, save as a new version with a note, the
// version list, the diff against live, restore), the two dialogs (the admin token for writes, a
// confirmation), the decision drawer, the session monitor's event row and drawer (SRV-115: one
// journal for the router and the monitor), and a few small templates.

import { html, nothing, type LitElement, type TemplateResult } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import { ApiError, describeError } from '../../api/client';
import { adminToken, conductor, setAdminToken, type Change, type DecisionRow, type DocKind, type DocResponse, type MonitorEvent, type VersionRow } from '../../api/conductor';
import { classColor, clone, eventWhat, isAct, readablePath, short, SOURCE_WORDS, stageName } from '../../lib/conductor';
import { dateTimeShort, ms, timeShort } from '../../lib/format';
import { LightElement } from '../../ui/base';
import { copyButton, icon } from '../../ui/bits';

// ---------------------------------------------------------------- small templates
export function clsChip(id: string | null, ids: string[], label?: string): TemplateResult {
  const color = classColor(id, ids);
  return html`<span class="cls" style="--cls: var(--series-${color})" data-class=${id ?? 'none'}>${label ?? id ?? '—'}</span>`;
}

export function confView(c: number | null | undefined, accept = 0.6): TemplateResult {
  if (c == null) return html`<span class="conf muted">—</span>`;
  return html`<span class="conf" title=${`confidence ${c.toFixed(4)}`}><span class="conf-bar ${c < accept ? 'is-low' : ''}"><span style="width:${Math.round(c * 100)}%"></span></span>${c.toFixed(2)}</span>`;
}

export function modeSeg<T extends string>(values: readonly T[], current: T, pick: (v: T) => void, opts: { disabled?: boolean; label?: string; cls?: string; small?: boolean } = {}): TemplateResult {
  return html`<div class="seg ${opts.small ? 'seg-sm' : ''} ${opts.cls ?? ''}" role="radiogroup" aria-label=${opts.label ?? 'mode'}>
    ${values.map(
      (v) => html`<button type="button" class="seg-btn mode-${v} ${v === current ? 'is-on' : ''}" role="radio" aria-checked=${v === current ? 'true' : 'false'} ?disabled=${opts.disabled} @click=${() => v !== current && pick(v)}>${v}</button>`,
    )}
  </div>`;
}

export function sourceWords(s: string | null): string {
  return s ? SOURCE_WORDS[s] ?? s : '—';
}

// ---------------------------------------------------------------- dialogs
@customElement('qse-cond-dialog')
export class QseCondDialog extends LightElement {
  @property({ attribute: false }) kind: 'token' | 'confirm' = 'confirm';
  @property({ attribute: false }) heading = '';
  @property({ attribute: false }) lines: (string | TemplateResult)[] = [];
  @property({ attribute: false }) confirmLabel = 'Confirm';
  @property({ attribute: false }) done: (ok: boolean, token?: string, remember?: boolean) => void = () => undefined;
  @state() private error: string | null = null;

  /** VIS-52 (UT1, 2026-10-06): Escape and a route change close it like Cancel; the element it
   *  was opened from gets the focus back */
  private readonly onKey = (e: KeyboardEvent) => {
    if (e.key === 'Escape') {
      e.preventDefault();
      this.done(false);
    }
  };
  private readonly onRoute = () => this.done(false);

  connectedCallback(): void {
    super.connectedCallback();
    document.addEventListener('keydown', this.onKey);
    window.addEventListener('hashchange', this.onRoute);
    window.addEventListener('popstate', this.onRoute);
  }

  disconnectedCallback(): void {
    document.removeEventListener('keydown', this.onKey);
    window.removeEventListener('hashchange', this.onRoute);
    window.removeEventListener('popstate', this.onRoute);
    super.disconnectedCallback();
  }

  private submit(e: Event) {
    e.preventDefault();
    if (this.kind === 'token') {
      const v = (this.querySelector<HTMLInputElement>('#cond-token')?.value ?? '').trim();
      if (!v) {
        this.error = 'Enter the admin token.';
        return;
      }
      this.done(true, v, this.querySelector<HTMLInputElement>('#cond-remember')?.checked ?? false);
    } else this.done(true);
  }

  render() {
    return html`<div class="modal-back" @click=${() => this.done(false)}></div>
      <form class="modal" role="dialog" aria-modal="true" aria-label=${this.heading} @submit=${(e: Event) => this.submit(e)}>
        <h2>${this.heading}</h2>
        ${this.lines.map((l) => html`<p class="muted small">${l}</p>`)}
        ${this.kind === 'token'
          ? html`<label class="field field-block"><span>Admin token</span><input id="cond-token" type="password" autocomplete="current-password" required autofocus /></label>
              <label class="switch"><input id="cond-remember" type="checkbox" /> remember on this device</label>`
          : nothing}
        ${this.error ? html`<p class="err-ink small" role="alert">${this.error}</p>` : nothing}
        <div class="modal-actions">
          <button type="button" class="btn" @click=${() => this.done(false)}>Cancel</button>
          <button type="submit" class="btn btn-primary" data-confirm>${this.confirmLabel}</button>
        </div>
      </form>`;
  }
}

function openDialog(setup: (d: QseCondDialog) => void): Promise<{ ok: boolean; token?: string; remember?: boolean }> {
  return new Promise((resolve) => {
    const d = document.createElement('qse-cond-dialog') as QseCondDialog;
    const from = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    setup(d);
    let closed = false;
    d.done = (ok, token, remember) => {
      if (closed) return;
      closed = true;
      d.remove();
      if (from && from.isConnected) from.focus();
      resolve({ ok, token, remember });
    };
    document.body.appendChild(d);
  });
}

export function confirmDialog(heading: string, lines: (string | TemplateResult)[], confirmLabel: string): Promise<boolean> {
  return openDialog((d) => {
    d.kind = 'confirm';
    d.heading = heading;
    d.lines = lines;
    d.confirmLabel = confirmLabel;
  }).then((r) => r.ok);
}

const TOKEN_WHERE = 'ops/make-secrets.sh made it; it is in secrets.env on the box';
/** VIS-52: the one token dialog on screen; a second caller waits for its answer */
let asking: Promise<boolean> | null = null;

/** Ask for the admin token; true when one was entered. `read` names the view a read (a 401 on
 *  a GET) was for: the dialog then says the view needs the token to show it, not "Unlock
 *  editing" (VIS-52, UT1 2026-10-06). One dialog at a time: concurrent callers share it. */
export function askToken(reason?: string, read?: string): Promise<boolean> {
  if (asking) return asking;
  asking = openDialog((d) => {
    d.kind = 'token';
    if (read) {
      d.heading = 'Admin token needed';
      d.lines = [`${read} needs the admin token to show its data (${TOKEN_WHERE}).`, 'It stays in this browser tab, or on this device if you tick the box. Cancel keeps the view closed; nothing changes.'];
      d.confirmLabel = 'Show';
    } else {
      d.heading = 'Unlock editing';
      d.lines = [reason ?? `Changing the conductor needs the admin token (${TOKEN_WHERE}).`, 'It stays in this browser tab, or on this device if you tick the box, and is sent only with a change.'];
      d.confirmLabel = 'Unlock';
    }
  }).then((r) => {
    asking = null;
    if (r.ok && r.token) setAdminToken(r.token, !!r.remember);
    return r.ok && !!r.token;
  });
  return asking;
}

/** The request needed the admin token (a 401 the dialog was cancelled for). */
export function isTokenError(e: unknown): boolean {
  return e instanceof ApiError && e.type === 'token_needed';
}

/** Run a call; on a 401 ask for the token once and try again. `read`: the view a read is for
 *  (the dialog's wording, VIS-52). */
export async function withToken<T>(fn: () => Promise<T>, read?: string): Promise<T> {
  try {
    return await fn();
  } catch (e) {
    if (isTokenError(e)) {
      if (adminToken()) setAdminToken(null);
      if (await askToken(undefined, read)) return await fn();
    }
    throw e;
  }
}

/** VIS-52: what a view that needs the token shows after Cancel: why, and a way to sign in. */
export function tokenState(what: string, retry: () => void): TemplateResult {
  return html`<div class="state" data-state="token" role="status">
    <div class="state-title">${icon('info')} Sign in with the admin token to see ${what}</div>
    <p class="state-body muted small">Reading this view needs the admin token (${TOKEN_WHERE}).</p>
    <button type="button" class="btn" data-sign-in @click=${async () => { if (await askToken(undefined, `This view`)) retry(); }}>Enter admin token</button>
  </div>`;
}

// ---------------------------------------------------------------- the editor controller
export class DocEditor<T> {
  base: DocResponse<T> | null = null;
  draft: T | null = null;
  loading = true;
  loadError: string | null = null;
  errors: string[] = [];
  changes: Change[] = [];
  note = '';
  saving = false;
  saved: string | null = null;
  saveError: string | null = null;
  versions: VersionRow[] | null = null;
  showVersions = false;
  showDiff = false;
  /** Called after a save or a restore went through (the view refreshes its header). */
  onSaved: (() => void) | null = null;
  private checkTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(
    readonly host: LitElement,
    public kind: DocKind,
    public name: string | null = null,
  ) {}

  get dirty(): boolean {
    return this.changes.length > 0;
  }
  get locked(): string | null {
    return (this.base?.locked as string | null | undefined) ?? null;
  }

  async load(): Promise<void> {
    this.loading = true;
    this.host.requestUpdate();
    try {
      this.base = await conductor.doc<T>(this.kind, this.name);
      this.draft = clone(this.base.doc);
      this.errors = [];
      this.changes = [];
      this.loadError = null;
    } catch (e) {
      const d = describeError(e);
      this.loadError = `${d.endpoint}: ${d.message}`;
    } finally {
      this.loading = false;
      this.host.requestUpdate();
    }
  }

  /** Replace the draft; the server checks it (schema and cross rules) a moment later. */
  set(next: T): void {
    this.draft = next;
    this.saved = null;
    this.host.requestUpdate();
    if (this.checkTimer) clearTimeout(this.checkTimer);
    this.checkTimer = setTimeout(() => void this.check(), 250);
  }

  async check(): Promise<void> {
    if (!this.draft) return;
    try {
      const r = await conductor.check<T>(this.kind, this.name, this.draft);
      this.errors = r.errors;
      this.changes = r.changes;
    } catch (e) {
      this.errors = [describeError(e).message];
    }
    this.host.requestUpdate();
  }

  reset(): void {
    if (!this.base) return;
    this.draft = clone(this.base.doc);
    this.errors = [];
    this.changes = [];
    this.host.requestUpdate();
  }

  async save(): Promise<boolean> {
    if (!this.draft || !this.base) return false;
    if (!this.note.trim()) {
      this.saveError = 'Write a one-line change note first.';
      this.host.requestUpdate();
      return false;
    }
    this.saving = true;
    this.saveError = null;
    this.host.requestUpdate();
    try {
      const r = await withToken(() => conductor.put<T>(this.kind, this.name, this.draft as T, this.note.trim(), this.base!.version));
      this.saved = `version ${r.version}, reloaded ${r.loaded_at ? timeShort(r.loaded_at) : 'now'}`;
      this.note = '';
      await this.load();
      if (this.showVersions) await this.loadVersions();
      this.onSaved?.();
      return true;
    } catch (e) {
      const err = e as ApiError & { errors?: string[] };
      this.saveError = err.status === 409 ? `${err.message}. Reload to see it.` : err.message;
      if (err.errors?.length) this.errors = err.errors;
      return false;
    } finally {
      this.saving = false;
      this.host.requestUpdate();
    }
  }

  async loadVersions(): Promise<void> {
    try {
      this.versions = (await conductor.versions(this.kind, this.name)).versions;
    } catch (e) {
      this.saveError = describeError(e).message;
    }
    this.host.requestUpdate();
  }

  async restore(v: number): Promise<void> {
    const ok = await confirmDialog(`Restore version ${v}?`, [`A new version equal to version ${v} becomes current, with the note "restore of v${v}". The server loads it for the next request.`], 'Restore');
    if (!ok) return;
    try {
      const r = await withToken(() => conductor.restore(this.kind, this.name, v));
      this.saved = `version ${r.version} (restore of v${v}), reloaded ${r.loaded_at ? timeShort(r.loaded_at) : 'now'}`;
      await this.load();
      await this.loadVersions();
      this.onSaved?.();
    } catch (e) {
      this.saveError = describeError(e).message;
      this.host.requestUpdate();
    }
  }
}

export function errorsList(ed: DocEditor<unknown>): TemplateResult | typeof nothing {
  if (!ed.errors.length) return nothing;
  return html`<ul class="errors" role="alert" data-errors>${ed.errors.slice(0, 12).map((e) => html`<li>${e}</li>`)}</ul>`;
}

export function diffList(changes: Change[]): TemplateResult {
  if (!changes.length) return html`<p class="muted small">No changes.</p>`;
  return html`<div class="diff" data-diff>
    ${changes.map((c) => html`<div class="diff-row"><span>${readablePath(c.path)}</span><span class="diff-before">${short(c.before, 160)}</span><span class="diff-after">${short(c.after, 160)}</span></div>`)}
  </div>`;
}

/** The versions of a document, newest first, each but the live one with its restore. */
export function versionsPanel(ed: DocEditor<unknown>, what: string): TemplateResult {
  return html`<section class="panel" data-versions><header class="panel-head"><h2 class="panel-title">Versions of ${what}</h2></header>
    ${ed.versions == null
      ? html`<p class="muted small">Loading…</p>`
      : html`<div class="table-scroll"><table class="table">
          <thead><tr><th>version</th><th>note</th><th>author</th><th>when</th><th></th></tr></thead>
          <tbody>${ed.versions.map(
            (v) => html`<tr data-version=${v.version}>
              <td class="num">v${v.version}${v.current ? html` <span class="badge badge-on">live</span>` : nothing}</td>
              <td class="cell-wrap">${v.note ?? '—'}</td>
              <td class="muted">${v.author ?? '—'}</td>
              <td class="num muted">${v.created_at ? dateTimeShort(v.created_at) : 'built in'}</td>
              <td>${v.current ? nothing : html`<button class="btn btn-sm" @click=${() => ed.restore(v.version)}>Restore</button>`}</td>
            </tr>`,
          )}</tbody></table></div>`}
  </section>`;
}

/** The save bar under an editor: the note, save, reset, diff against live, versions, restore. */
export function saveBar(ed: DocEditor<unknown>, what: string): TemplateResult {
  const meta = ed.base?.meta;
  if (ed.locked) {
    return html`<div class="savebar" data-savebar><span class="badge badge-lock">${icon('warn')} locked</span><span class="status">${ed.locked}. Every control is read-only.</span></div>`;
  }
  return html`
    ${ed.showDiff ? html`<section class="panel" data-diff-panel><header class="panel-head"><h2 class="panel-title">Changes against live (v${meta?.version ?? 0})</h2></header>${diffList(ed.changes)}</section>` : nothing}
    ${ed.showVersions ? versionsPanel(ed, what) : nothing}
    <div class="savebar" data-savebar>
      <span class="status num" title=${meta?.note ?? ''}>live v${meta?.version ?? 0}${meta?.loaded_at ? html` · reloaded ${timeShort(meta.loaded_at)}` : nothing}</span>
      <input class="note-input" placeholder="one-line change note (required)" .value=${ed.note} @input=${(e: Event) => (ed.note = (e.target as HTMLInputElement).value)} aria-label="change note" maxlength="200" />
      <button class="btn btn-primary" ?disabled=${!ed.dirty || ed.saving || ed.errors.length > 0} @click=${() => ed.save()} data-save>${ed.saving ? 'Saving…' : 'Save as new version'}</button>
      <button class="btn btn-ghost" ?disabled=${!ed.dirty} @click=${() => ed.reset()}>Reset</button>
      <button class="btn btn-ghost ${ed.showDiff ? 'is-on' : ''}" @click=${() => {
        ed.showDiff = !ed.showDiff;
        void ed.check();
      }}>Diff (${ed.changes.length})</button>
      <button class="btn btn-ghost ${ed.showVersions ? 'is-on' : ''}" @click=${() => {
        ed.showVersions = !ed.showVersions;
        if (ed.showVersions) void ed.loadVersions();
        else ed.host.requestUpdate();
      }} data-versions-btn>Versions</button>
      ${ed.saved ? html`<span class="status" role="status" data-saved>${icon('check')} ${ed.saved}</span>` : nothing}
      ${ed.saveError ? html`<span class="err-ink small" role="alert">${ed.saveError}</span>` : nothing}
    </div>`;
}

// ---------------------------------------------------------------- the decision drawer
@customElement('qse-cond-decision')
export class QseCondDecision extends LightElement {
  @property({ attribute: false }) row: DecisionRow | null = null;
  @property({ attribute: false }) ids: string[] = [];
  @property({ attribute: false }) close: () => void = () => undefined;
  @state() private labelling = false;
  @state() private msg: string | null = null;

  private async label(l: 'right' | 'wrong' | 'clear') {
    if (!this.row) return;
    this.labelling = true;
    try {
      await withToken(() => conductor.label(this.row!.id, l));
      this.row = { ...this.row, label: l === 'clear' ? null : l };
      this.msg = l === 'clear' ? 'label cleared' : `labelled ${l}`;
    } catch (e) {
      this.msg = describeError(e).message;
    } finally {
      this.labelling = false;
    }
  }

  render() {
    const r = this.row;
    if (!r) return nothing;
    const accepted = r.stages.find((s) => !s.error && !s.below);
    return html`<div class="drawer-back" @click=${() => this.close()}></div>
      <aside class="drawer" role="dialog" aria-label="decision" data-drawer>
        <header class="drawer-head">
          <div>
            <div class="panel-title">${r.kind === 'side' ? clsChip(null, this.ids, 'side request') : clsChip(r.class_id, this.ids)} ${r.mode === 'shadow' ? html`<span class="badge badge-shadow">shadow</span>` : html`<span class="badge badge-on">on</span>`}</div>
            <div class="muted small num">${dateTimeShort(new Date(r.ts_ms).toISOString())} · ${r.profile} v${r.profile_version} · router v${r.router_version}</div>
          </div>
          <button class="btn btn-ghost btn-icon" aria-label="Close" @click=${() => this.close()}>${icon('x')}</button>
        </header>
        <div class="drawer-body">
          <section class="drawer-section">
            <h3>Decision</h3>
            <dl class="kv">
              <div><dt>class</dt><dd>${r.class_id ?? '—'} ${confView(r.confidence)}</dd></div>
              <div><dt>decided by</dt><dd>${sourceWords(r.source)}${r.kept ? html` <span class="badge badge-warn">${r.kept.replace(/_/g, ' ')}</span>` : nothing}</dd></div>
              <div><dt>classifier time</dt><dd class="num">${ms(r.classifier_ms)}</dd></div>
              <div><dt>effort</dt><dd>${r.effort ?? 'unchanged'}${r.budget != null ? html` <span class="muted num">budget ${r.budget}</span>` : nothing} <span class="muted">(${r.budget_source ?? 'server'})</span></dd></div>
              <div><dt>thinking</dt><dd>${r.thinking ?? 'on'}</dd></div>
              <div><dt>modes</dt><dd class="num">${Object.entries(r.decisions ?? {}).map(([k, v]) => `${k} ${v}`).join(' · ')}</dd></div>
            </dl>
          </section>
          ${r.stages.length
            ? html`<section class="drawer-section"><h3>Classifier stages</h3><div class="stage-list">
                ${r.stages.map(
                  (s) => html`<div class="stage ${s === accepted ? 'is-accepted' : ''} ${s.error ? 'is-error' : ''}">
                    <strong>${stageName(s.stage)}</strong>
                    <div>${s.error ? html`<span class="err-ink">${s.error}</span>` : html`${s.label} ${confView(s.confidence, s.accept ?? 0.6)}${s.below ? html` <span class="muted small">below ${s.accept}</span>` : nothing}`}
                      ${s.stage === 'qwen_context' && s.prompt_tokens != null ? html`<div class="muted small num" data-context-prefill>${s.reason === 'long' ? 'long prompt' : 'follow-up'} · prompt ${s.prompt_tokens.toLocaleString('en-US')} tokens · prefilled ${(s.prefilled ?? 0).toLocaleString('en-US')} · reused ${(s.reused ?? 0).toLocaleString('en-US')}</div>` : nothing}
                      ${s.probs && Object.keys(s.probs).length > 1
                        ? html`<div class="probs">${Object.entries(s.probs)
                            .sort((a, b) => b[1] - a[1])
                            .slice(0, 6)
                            .map(([k, v]) => html`<div class="prob"><span>${k}</span><span class="prob-bar"><span style="width:${Math.round(v * 100)}%"></span></span><span>${v.toFixed(2)}</span></div>`)}</div>`
                        : nothing}
                    </div>
                    <span class="num muted">${ms(s.ms)}${s.cached ? ' (cached)' : ''}</span>
                  </div>`,
                )}</div></section>`
            : nothing}
          ${(r.injected ?? []).map(
            (n) => html`<section class="drawer-section" data-injected>
              <h3>Injected at token ${n.pos.toLocaleString('en-US')} · ${n.tokens.toLocaleString('en-US')} tokens ${copyButton(n.text, 'Copy the injected text')}</h3>
              <p class="muted small">Because ${n.reason}. Parts: ${n.parts.map((p) => `${p.kind} ${p.name}`).join(', ')}. The client never saw this text.</p>
              <pre class="note-text">${n.text}</pre>
            </section>`,
          )}
          ${r.replayed?.count ? html`<p class="muted small">Replayed ${r.replayed.count} earlier note(s), ${r.replayed.tokens.toLocaleString('en-US')} tokens, at the same positions, so the resident prefix kept matching.</p>` : nothing}
          ${r.events?.length
            ? html`<section class="drawer-section" data-monitor-events><h3>The session monitor in this request</h3>
                ${r.events.map((e) => eventBlock(e))}</section>`
            : nothing}
          ${r.shadow ? html`<section class="drawer-section"><h3>Shadow (not applied)</h3><pre class="note-text">${JSON.stringify(r.shadow, null, 2)}</pre></section>` : nothing}
          ${r.digest ? html`<section class="drawer-section"><h3>What the classifier read</h3><pre class="note-text">${r.digest}</pre></section>` : nothing}
          <section class="drawer-section">
            <h3>Outcome</h3>
            ${r.outcome
              ? html`<dl class="kv">
                  <div><dt>finish</dt><dd>${r.outcome.finish_reason ?? 'running'}</dd></div>
                  <div><dt>tool calls</dt><dd class="num">${r.outcome.tool_calls ?? 0} ${(r.outcome.tool_names ?? []).join(', ')}</dd></div>
                  <div><dt>reasoning tokens</dt><dd class="num">${r.outcome.reasoning_tokens ?? '—'}</dd></div>
                  <div><dt>time</dt><dd class="num">${ms(r.outcome.total_ms)}</dd></div>
                  ${r.next_checker?.length ? html`<div><dt>next checker</dt><dd>${r.next_checker.map((c) => html`<span class="badge ${c.verdict === 'PASS' ? 'badge-ok' : 'badge-warn'}">${c.verdict}</span> <span class="muted small">${c.checker} ${c.reasons.slice(0, 2).join('; ')}</span>`)}</dd></div>` : nothing}
                </dl>`
              : html`<p class="muted small">The turn is still running.</p>`}
          </section>
          <section class="drawer-section">
            <h3>Was this right?</h3>
            <div class="toolbar" style="margin:0">
              <button class="btn ${r.label === 'right' ? 'btn-primary' : ''}" ?disabled=${this.labelling} @click=${() => this.label('right')} data-label="right">${icon('check')} this was right</button>
              <button class="btn ${r.label === 'wrong' ? 'btn-primary' : ''}" ?disabled=${this.labelling} @click=${() => this.label('wrong')} data-label="wrong">${icon('x')} this was wrong</button>
              ${r.label ? html`<button class="btn btn-ghost" @click=${() => this.label('clear')}>clear</button>` : nothing}
              ${this.msg ? html`<span class="muted small" role="status">${this.msg}</span>` : nothing}
            </div>
            <p class="muted small">Labels feed the precision on the Compare screen and the promotion rule.</p>
          </section>
          <p class="muted small num">decision ${r.id} · request ${r.request_id ?? '—'} · session ${short(r.session_key, 40)}</p>
        </div>
      </aside>`;
  }
}

export function feedRow(r: DecisionRow, ids: string[], open: (r: DecisionRow) => void, what: string): TemplateResult {
  return html`<button class="feed-row" data-decision=${r.id} @click=${() => open(r)}>
    <span class="feed-time">${timeShort(new Date(r.ts_ms).toISOString())}</span>
    <span class="feed-cls">${r.kind === 'side' ? clsChip(null, ids, 'side request') : clsChip(r.class_id, ids)}</span>
    <span class="feed-conf">${confView(r.confidence)} <span class="muted small">${sourceWords(r.source)}</span></span>
    <span class="feed-what">${what}</span>
    <span class="feed-act">${r.acted ? html`<span class="badge badge-on">acted</span>` : nothing}${r.mode === 'shadow' || r.shadow ? html` <span class="badge badge-shadow">shadow</span>` : nothing}${!r.acted && !(r.mode === 'shadow' || r.shadow) ? html`<span class="badge">pass</span>` : nothing}</span>
  </button>`;
}

// ---------------------------------------------------------------- the session monitor's events
function modeBadge(e: Pick<MonitorEvent, 'mode' | 'decision'>): TemplateResult {
  if (isAct(e)) return html`<span class="badge badge-on">acted</span>`;
  if (e.mode === 'shadow' && e.decision !== 'PASS') return html`<span class="badge badge-shadow">shadow</span>`;
  return html`<span class="badge">pass</span>`;
}

/** One monitor decision with its hidden text, for a drawer: what, why, where, what came of it. */
export function eventBlock(e: MonitorEvent): TemplateResult {
  const hidden = (e.hidden_tokens ?? 0) > 0;
  return html`<div class="event-block" data-event=${e.id}>
    <div class="event-head">${modeBadge(e)} <strong>${eventWhat(e)}</strong></div>
    <dl class="kv">
      <div><dt>safe point</dt><dd>${e.sp ?? '—'} · rule ${e.rule ?? '—'} · policy v${e.policy_version ?? '—'}</dd></div>
      ${e.tokens_at_cut != null ? html`<div><dt>reasoning tokens</dt><dd class="num">${e.tokens_at_cut.toLocaleString('en-US')}</dd></div>` : nothing}
      ${e.outcome ? html`<div><dt>outcome</dt><dd>${e.outcome.kind ?? '—'}${e.outcome.finish ? html` <span class="muted">(${e.outcome.finish})</span>` : nothing}</dd></div>` : nothing}
      ${e.next_checker?.length ? html`<div><dt>next checker</dt><dd>${e.next_checker.map((c) => html`<span class="badge ${c.verdict === 'PASS' ? 'badge-ok' : 'badge-warn'}">${c.verdict}</span> <span class="muted small">${c.checker} ${c.reasons.slice(0, 2).join('; ')}</span>`)}</dd></div>` : nothing}
    </dl>
    ${e.text
      ? html`<h4 class="small">${hidden ? html`Hidden text at token ${(e.position ?? 0).toLocaleString('en-US')} · ${(e.hidden_tokens ?? 0).toLocaleString('en-US')} tokens` : 'Visible text'} ${copyButton(e.text, 'Copy the text')}</h4><pre class="note-text">${e.text}</pre>`
      : nothing}
    ${e.detail && Object.keys(e.detail).length ? html`<details><summary class="muted small">detail</summary><pre class="note-text">${JSON.stringify(e.detail, null, 2)}</pre></details>` : nothing}
  </div>`;
}

export function eventRow(e: MonitorEvent, open: (e: MonitorEvent) => void): TemplateResult {
  return html`<button class="feed-row" data-event-row=${e.id} @click=${() => open(e)}>
    <span class="feed-time">${timeShort(new Date(e.ts_ms).toISOString())}</span>
    <span class="feed-cls"><span class="badge mono">${e.task ?? '—'}</span></span>
    <span class="feed-conf muted small">${e.sp ?? ''}</span>
    <span class="feed-what">${eventWhat(e)}${(e.hidden_tokens ?? 0) > 0 ? html` <span class="muted small num">${e.hidden_tokens} hidden tokens</span>` : nothing}</span>
    <span class="feed-act">${modeBadge(e)}</span>
  </button>`;
}

@customElement('qse-cond-event')
export class QseCondEvent extends LightElement {
  @property({ attribute: false }) ev: MonitorEvent | null = null;
  @property({ attribute: false }) close: () => void = () => undefined;
  @state() private labelling = false;
  @state() private msg: string | null = null;

  private async label(l: 'right' | 'wrong' | 'clear') {
    if (!this.ev) return;
    this.labelling = true;
    try {
      await withToken(() => conductor.label(this.ev!.id, l));
      this.ev = { ...this.ev, label: l === 'clear' ? null : l };
      this.msg = l === 'clear' ? 'label cleared' : `labelled ${l}`;
    } catch (e) {
      this.msg = describeError(e).message;
    } finally {
      this.labelling = false;
    }
  }

  render() {
    const e = this.ev;
    if (!e) return nothing;
    return html`<div class="drawer-back" @click=${() => this.close()}></div>
      <aside class="drawer" role="dialog" aria-label="monitor decision" data-event-drawer>
        <header class="drawer-head">
          <div>
            <div class="panel-title"><span class="badge mono">${e.task}</span> ${modeBadge(e)}</div>
            <div class="muted small num">${dateTimeShort(new Date(e.ts_ms).toISOString())} · ${e.profile ?? 'no profile'} · ${e.client_kind ?? '—'}</div>
          </div>
          <button class="btn btn-ghost btn-icon" aria-label="Close" @click=${() => this.close()}>${icon('x')}</button>
        </header>
        <div class="drawer-body">
          <section class="drawer-section">${eventBlock(e)}</section>
          <section class="drawer-section">
            <h3>Was this right?</h3>
            <div class="toolbar" style="margin:0">
              <button class="btn ${e.label === 'right' ? 'btn-primary' : ''}" ?disabled=${this.labelling} @click=${() => this.label('right')} data-label="right">${icon('check')} this was right</button>
              <button class="btn ${e.label === 'wrong' ? 'btn-primary' : ''}" ?disabled=${this.labelling} @click=${() => this.label('wrong')} data-label="wrong">${icon('x')} this was wrong</button>
              ${e.label ? html`<button class="btn btn-ghost" @click=${() => this.label('clear')}>clear</button>` : nothing}
              ${this.msg ? html`<span class="muted small" role="status">${this.msg}</span>` : nothing}
            </div>
            <p class="muted small">Labels on shadow decisions feed the task's precision and its promotion to act (Compare).</p>
          </section>
          <p class="muted small num">event ${e.id} · request ${e.request_id ?? '—'} · session ${e.session_key ? html`<a href=${`#/conductor?screen=sessions&session=${encodeURIComponent(e.session_key)}`}>${short(e.session_key, 40)}</a>` : '—'}</p>
        </div>
      </aside>`;
  }
}
