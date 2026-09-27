// <qse-app>: the shell — session gate, top bar with the status pill, the rail / bottom tab bar,
// hash routing, theme toggle, sign-out. The shell owns the summary poll (60 s) and hands it to
// the Usage view so the pill and the cards come from one request.

import { html, nothing, type TemplateResult } from 'lit';
import { customElement, state } from 'lit/decorators.js';
import { api, describeError, onApi } from './api/client';
import type { Summary, LiveStatus } from './api/types';
import { duration, shortHash } from './lib/format';
import { poll, type Poller } from './lib/poll';
import { onRoute, parseHash, type Route, type View } from './lib/router';
import { currentTheme, toggleTheme, watchSystemTheme, type Theme } from './lib/theme';
import { browserTz } from './lib/time';
import { LightElement, type Loadable } from './ui/base';
import { icon } from './ui/bits';

declare const __DASH_VERSION__: string;

const TABS: { view: View; label: string }[] = [
  { view: 'usage', label: 'Usage' },
  { view: 'performance', label: 'Performance' },
  { view: 'dev', label: 'Dev' },
  { view: 'playground', label: 'Playground' },
  { view: 'system', label: 'System' },
];

@customElement('qse-app')
export class QseApp extends LightElement {
  @state() private auth: 'checking' | 'in' | 'out' = 'checking';
  @state() private route: Route = parseHash();
  @state() private theme: Theme = currentTheme();
  @state() private summary: Loadable<Summary> = { state: 'loading' };
  @state() private offline = false;
  @state() private loginError: string | null = null;
  @state() private loginBusy = false;
  @state() private version: string | null = null;
  @state() private codeHash: string | null = null;
  private summaryPoll: Poller | null = null;
  private unsub: (() => void)[] = [];
  private tz = browserTz();

  connectedCallback(): void {
    super.connectedCallback();
    watchSystemTheme();
    this.unsub.push(onRoute((r) => (this.route = r)));
    this.unsub.push(
      onApi((ev) => {
        if (ev === 'unauthorized') this.signedOut();
        else if (ev === 'offline') this.offline = true;
        else if (ev === 'online') this.offline = false;
      }),
    );
    const onTheme = (e: Event) => (this.theme = (e as CustomEvent<Theme>).detail);
    const onUnauth = () => this.signedOut();
    window.addEventListener('qse-theme', onTheme);
    window.addEventListener('qse-unauthorized', onUnauth);
    this.unsub.push(() => window.removeEventListener('qse-theme', onTheme));
    this.unsub.push(() => window.removeEventListener('qse-unauthorized', onUnauth));
    void this.checkSession();
  }
  disconnectedCallback(): void {
    for (const u of this.unsub) u();
    this.summaryPoll?.stop();
    super.disconnectedCallback();
  }

  private async checkSession(): Promise<void> {
    try {
      await api.session.get();
      this.signedIn();
    } catch (e) {
      const d = describeError(e);
      if (d.status === 401) this.auth = 'out';
      else {
        // The engine is unreachable: show the shell offline rather than a login screen.
        this.auth = 'out';
        this.offline = true;
        this.loginError = d.status === 0 ? `engine unreachable at ${d.endpoint}` : `${d.endpoint}: ${d.message}`;
      }
    }
  }

  private signedIn(): void {
    this.auth = 'in';
    this.loginError = null;
    this.summaryPoll?.stop();
    this.summaryPoll = poll(() => this.loadSummary(), 60_000);
    this.summaryPoll.start();
    void this.loadVersion();
  }

  private signedOut(): void {
    if (this.auth === 'out') return;
    this.auth = 'out';
    this.summaryPoll?.stop();
    this.summary = { state: 'loading' };
  }

  private async loadSummary(): Promise<void> {
    try {
      const s = await api.summary(this.tz);
      this.summary = { state: 'ready', data: s };
    } catch (e) {
      const d = describeError(e);
      if (d.status !== 401) this.summary = { state: 'error', endpoint: d.endpoint, message: d.message };
    }
  }

  private async loadVersion(): Promise<void> {
    try {
      const s = await api.system();
      this.version = s.engine.version;
      this.codeHash = s.engine.code_sha256;
    } catch {
      /* the pill shows what it can */
    }
  }

  private async login(e: Event): Promise<void> {
    e.preventDefault();
    const input = this.querySelector<HTMLInputElement>('#token');
    const token = input?.value ?? '';
    if (!token) return;
    this.loginBusy = true;
    this.loginError = null;
    try {
      await api.session.login(token);
      if (input) input.value = '';
      this.signedIn();
    } catch (err) {
      const d = describeError(err);
      this.loginError = d.status === 401 ? 'The token was refused.' : d.status === 429 ? 'Too many attempts — wait a minute.' : d.status === 0 ? `engine unreachable at ${d.endpoint}` : `${d.endpoint}: ${d.message}`;
    } finally {
      this.loginBusy = false;
    }
  }

  private async logout(): Promise<void> {
    try {
      await api.session.logout();
    } catch {
      /* already out */
    }
    this.signedOut();
  }

  private status(): LiveStatus {
    if (this.offline) return 'offline';
    if (this.summary.state === 'ready') return this.summary.data.live.status;
    if (this.summary.state === 'error') return 'offline';
    return 'ok';
  }

  render() {
    if (this.auth === 'checking') return html`<div class="boot" aria-busy="true"><span class="boot-mark"></span></div>`;
    if (this.auth === 'out') return this.renderLogin();
    const st = this.status();
    const live = this.summary.state === 'ready' ? this.summary.data.live : null;
    return html`
      <div class="shell">
        <header class="topbar">
          <div class="topbar-left">
            <span class="brand"><span class="brand-mark" aria-hidden="true"></span><span class="brand-name">qwen38-spark-engine</span></span>
            <span class="pill pill-${st}" data-status=${st} role="status">${st}</span>
            <span class="topbar-meta num">
              ${this.version ? html`<span title="engine version">${this.version}</span>` : nothing}
              ${this.codeHash ? html`<span title="code hash">${shortHash(this.codeHash)}</span>` : nothing}
              ${live ? html`<span title="uptime">up ${duration(live.uptime_s)}</span>` : nothing}
            </span>
          </div>
          <div class="topbar-right">
            <button class="btn btn-ghost btn-icon" aria-label=${this.theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'} title="Theme" @click=${() => (this.theme = toggleTheme())}>${icon(this.theme === 'dark' ? 'sun' : 'moon')}</button>
            <button class="btn btn-ghost btn-icon" aria-label="Sign out" title="Sign out" @click=${() => this.logout()}>${icon('out')}</button>
          </div>
        </header>
        <nav class="rail" aria-label="views">
          ${TABS.map((t) => html`<a class="rail-tab ${this.route.view === t.view ? 'is-on' : ''}" href="#/${t.view}" aria-current=${this.route.view === t.view ? 'page' : nothing}>${icon(t.view)}<span>${t.label}</span></a>`)}
        </nav>
        <main class="content" id="main">
          ${this.offline ? html`<div class="banner banner-offline" role="alert">${icon('warn')} The engine is not answering — panels keep their last data and say which endpoint failed.</div>` : nothing}
          ${this.renderView()}
        </main>
      </div>
    `;
  }

  private renderView(): TemplateResult {
    switch (this.route.view) {
      case 'usage':
        return html`<qse-usage .summary=${this.summary} .params=${this.route.params}></qse-usage>`;
      case 'performance':
        return html`<qse-performance .params=${this.route.params}></qse-performance>`;
      case 'dev':
        return html`<qse-dev .params=${this.route.params}></qse-dev>`;
      case 'playground':
        return html`<qse-playground .params=${this.route.params}></qse-playground>`;
      case 'system':
        return html`<qse-system></qse-system>`;
    }
  }

  private renderLogin(): TemplateResult {
    return html`
      <div class="login">
        <form class="login-card" @submit=${(e: Event) => this.login(e)}>
          <span class="brand"><span class="brand-mark" aria-hidden="true"></span><span class="brand-name">qwen38-spark-engine</span></span>
          <h1 class="login-title">Dashboard</h1>
          <p class="login-sub">Enter the admin token to open the dashboard. The session lives in an HttpOnly cookie for 400 days; the token itself is never stored.</p>
          <label class="field field-block"><span>Admin token</span><input id="token" type="password" autocomplete="current-password" required autofocus /></label>
          ${this.loginError ? html`<p class="login-error" role="alert">${this.loginError}</p>` : nothing}
          <button class="btn btn-primary" type="submit" ?disabled=${this.loginBusy}>${this.loginBusy ? 'Checking…' : 'Open the dashboard'}</button>
          <div class="login-foot">
            <button type="button" class="btn btn-ghost btn-sm" @click=${() => (this.theme = toggleTheme())}>${icon(this.theme === 'dark' ? 'sun' : 'moon')} ${this.theme === 'dark' ? 'light' : 'dark'} theme</button>
            <span class="muted small num">dashboard ${__DASH_VERSION__}</span>
          </div>
        </form>
      </div>
    `;
  }
}
