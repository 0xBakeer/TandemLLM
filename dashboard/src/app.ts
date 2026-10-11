// <qse-app>: the shell — session gate, top bar with the status pill, the rail / bottom tab bar,
// hash routing, theme toggle. Beside the pill, the session monitor's mode (off, shadow,
// on) when the server has one (VIS-44). The dashboard asks for no password anywhere (AU1,
// 2026-10-06): the shell opens straight into its views, with no sign-in screen and no sign-out
// button. With no password set on the server every route is open; once one is set, a 401 shows a
// banner (text only, no prompt); the setup guide alone has a sign-in, for its own routes. The shell owns the summary poll (60 s) and hands it to
// the Usage view so the pill and the cards come from one request. In setup mode (OPS-93: /health
// says `mode: "setup"`, every /v1/dashboard/* route answers 503 setup_mode) the shell polls nothing
// but /health, shows the setup guide only, and starts its polls once the box serves a model.

import { html, nothing, type TemplateResult } from 'lit';
import { customElement, state } from 'lit/decorators.js';
import { api, apiBase, describeError, engineAway, onApi } from './api/client';
import { awayText } from './lib/away';
import { config } from './api/config';
import { perfEngineNote } from './lib/perf-note';
import type { Summary, LiveStatus } from './api/types';
import { duration, shortHash } from './lib/format';
import { modeWord } from './lib/supervisor';
import { poll, type Poller } from './lib/poll';
import { navigate, onRoute, parseHash, type Route, type View } from './lib/router';
import { currentTheme, toggleTheme, watchSystemTheme, type Theme } from './lib/theme';
import { browserTz } from './lib/time';
import { LightElement, type Loadable } from './ui/base';
import { icon } from './ui/bits';
import { S } from './setup/strings';


/** `short`: the label in the phone's bottom bar, where seven tabs share 390 px */
const TABS: { view: View; label: string; short?: string }[] = [
  { view: 'chat', label: 'Chat' },
  { view: 'usage', label: 'Usage' },
  { view: 'performance', label: 'Performance', short: 'Perf' },
  { view: 'dev', label: 'Dev' },
  { view: 'playground', label: 'Playground', short: 'Play' },
  { view: 'conductor', label: 'Conductor', short: 'Cond' },
  { view: 'backends', label: 'Backends', short: 'Backend' },
  { view: 'system', label: 'System' },
  { view: 'setup', label: 'Setup' },
];

/** The Playground view is its own chunk; the element upgrades in place once the chunk has loaded. */
let playgroundChunk: Promise<unknown> | null = null;
const loadPlayground = () => (playgroundChunk ??= import('./views/playground'));
/** The Chat tab (VIS-99): the conversation, its panes and the Context panel, its own chunk; the
 *  mock backend and the remote client load inside it on demand. */
let chatChunk: Promise<unknown> | null = null;
const loadChat = () => (chatChunk ??= import('./chat/view'));
/** The Conductor view, its own chunk too: the editors are not needed on the pages watched daily. */
let conductorChunk: Promise<unknown> | null = null;
const loadConductor = () => (conductorChunk ??= import('./views/conductor'));
/** The Backends view (the proxy's registry), its own chunk: set up once, opened rarely. */
let backendsChunk: Promise<unknown> | null = null;
const loadBackends = () => (backendsChunk ??= import('./views/backends'));
/** The setup guide (VIS-78): the wizard, Wren and the reset levels, its own chunk. */
let setupChunk: Promise<unknown> | null = null;
const loadSetup = () => (setupChunk ??= import('./setup/view'));
/** Performance → Parallel (every request on every backend at once), its own chunk. */
let parallelChunk: Promise<unknown> | null = null;
const loadParallel = () => (parallelChunk ??= import('./views/parallel'));

/** The two pages of Performance: the engine's own (the Live panel, the scatter, the history) and
 *  every request in flight on every backend at once. */
function perfTabs(on: 'engine' | 'parallel', parallel: number | null = null): TemplateResult {
  const go = (tab: string | null) => navigate('performance', tab ? { tab } : {});
  return html`<div class="perf-tabs" role="tablist" aria-label="Performance pages">
    <div class="seg">
      <button class="seg-btn ${on === 'engine' ? 'is-on' : ''}" role="tab" aria-selected=${on === 'engine'} @click=${() => go(null)}>Engine</button>
      <button class="seg-btn ${on === 'parallel' ? 'is-on' : ''}" role="tab" aria-selected=${on === 'parallel'} data-tab="parallel" @click=${() => go('parallel')}>Parallel</button>
    </div>
    <span class="perf-tabs-note">${on === 'parallel' ? 'every request in flight, on every backend' : perfEngineNote(parallel)}</span>
  </div>`;
}

@customElement('qse-app')
export class QseApp extends LightElement {
  /** a 401 came back: the server has an admin password set (never while none is) */
  @state() private refused = false;
  @state() private route: Route = parseHash();
  @state() private theme: Theme = currentTheme();
  @state() private summary: Loadable<Summary> = { state: 'loading' };
  @state() private offline = false;
  @state() private version: string | null = null;
  @state() private codeHash: string | null = null;
  /** the engine's parallel sequences (/v1/config engine.parallel.n), read once for the subtitle */
  @state() private engineParallel: number | null = null;
  /** OPS-93: the box runs in setup mode (only the dashboard and Wren); every other view is off */
  @state() private setupMode = false;
  /** /health has answered once (or failed): until then no view mounts, so a box in setup mode
   *  gets no request a view would make */
  @state() private booted = false;
  private parallelAsked = false;
  private summaryPoll: Poller | null = null;
  private setupPoll: Poller | null = null;
  private unsub: (() => void)[] = [];
  private tz = browserTz();

  connectedCallback(): void {
    super.connectedCallback();
    watchSystemTheme();
    this.unsub.push(
      onRoute((r) => {
        this.route = r;
        if (this.setupMode && r.view !== 'setup') navigate('setup');
      }),
    );
    this.unsub.push(
      onApi((ev) => {
        if (ev === 'unauthorized') this.refused = true;
        else if (ev === 'offline') this.offline = true;
        else if (ev === 'online') this.offline = false;
      }),
    );
    const onTheme = (e: Event) => (this.theme = (e as CustomEvent<Theme>).detail);
    const onUnauth = () => (this.refused = true);
    const onSignedIn = () => {
      this.refused = false;
      if (!this.setupMode) {
        void this.loadSummary();
        void this.loadVersion();
      }
    };
    // a setup job ends: the box may have left setup mode (the job restarted it serving)
    const onJobFinal = () => void this.checkSetupMode();
    window.addEventListener('qse-theme', onTheme);
    window.addEventListener('qse-unauthorized', onUnauth);
    window.addEventListener('qse-signed-in', onSignedIn);
    window.addEventListener('job-final', onJobFinal);
    this.unsub.push(() => window.removeEventListener('qse-theme', onTheme));
    this.unsub.push(() => window.removeEventListener('qse-unauthorized', onUnauth));
    this.unsub.push(() => window.removeEventListener('qse-signed-in', onSignedIn));
    this.unsub.push(() => window.removeEventListener('job-final', onJobFinal));
    void this.checkSetupMode().then(() => {
      this.booted = true;
      if (!this.setupMode) this.start();
    });
  }
  disconnectedCallback(): void {
    for (const u of this.unsub) u();
    this.summaryPoll?.stop();
    this.setupPoll?.stop();
    super.disconnectedCallback();
  }

  /** once per page load: how many requests the engine runs at once (SRV-221). The engine's own
   *  /v1/dashboard/parallel answers on every server (proxy on or off); its `engine_slots` is
   *  absent without side slots, so an answer without it means one. A server without that route
   *  falls back to /v1/config's engine block (proxy on); with neither, no number is claimed. */
  private askParallel(): void {
    if (this.parallelAsked) return;
    this.parallelAsked = true;
    api
      .parallel()
      .then((p) => (this.engineParallel = p.engine_slots?.n ?? 1))
      .catch(() =>
        config
          .overview()
          .then((o) => (this.engineParallel = o.engine?.embedded && o.engine.parallel ? o.engine.parallel.n : null))
          .catch(() => {
            /* no number: the subtitle says none */
          }),
      );
  }

  private start(): void {
    this.summaryPoll?.stop();
    this.summaryPoll = poll(() => this.loadSummary(), 60_000);
    this.summaryPoll.start();
    void this.loadVersion();
  }

  /** /health says `mode: "setup"` on a box started by `ops/start.sh --setup`: open the guide, and
   *  ask again every 10 s until the box serves a model (an install ends with a restart) */
  private async checkSetupMode(): Promise<void> {
    let h: { mode?: string } | null;
    try {
      h = await fetch(apiBase() + '/health', { cache: 'no-store', credentials: 'same-origin' }).then((r) => (r.ok ? r.json() : null));
    } catch {
      // unreachable (an install restarts the engine): setup mode stays as it was
      return;
    }
    const was = this.setupMode;
    this.setupMode = h?.mode === 'setup';
    if (this.setupMode) {
      if (this.route.view !== 'setup') navigate('setup');
      if (!this.setupPoll) {
        this.setupPoll = poll(() => this.checkSetupMode(), 10_000);
        this.setupPoll.start();
      }
    } else if (was) {
      this.setupPoll?.stop();
      this.setupPoll = null;
      this.start();
    }
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

  private status(): LiveStatus | 'setup' {
    if (this.setupMode) return 'setup';
    if (this.offline) return 'offline';
    if (this.summary.state === 'ready') return this.summary.data.live.status;
    if (this.summary.state === 'error') return 'offline';
    return 'ok';
  }

  render() {
    const st = this.status();
    const live = this.summary.state === 'ready' ? this.summary.data.live : null;
    return html`
      <div class="shell" data-view=${this.route.view}>
        <header class="topbar">
          <div class="topbar-left">
            <span class="brand"><span class="brand-mark" aria-hidden="true"></span><span class="brand-name">TandemLLM</span></span>
            <span class="pill pill-${st}" data-status=${st} role="status" title=${st === 'setup' ? S.setupModeBanner : nothing}>${st === 'setup' ? S.setupModePill : st}</span>
            ${live?.conductor ? html`<a class="pill pill-monitor is-${live.conductor.mode}" href="#/conductor?screen=monitor" data-monitor-mode=${live.conductor.mode} title="the session monitor's mode on this server (--conductor)">${modeWord(live.conductor)}</a>` : nothing}
            <span class="topbar-meta num">
              ${this.version ? html`<span title="engine version">${this.version}</span>` : nothing}
              ${this.codeHash ? html`<span title="code hash">${shortHash(this.codeHash)}</span>` : nothing}
              ${live ? html`<span title="uptime">up ${duration(live.uptime_s)}</span>` : nothing}
            </span>
          </div>
          <div class="topbar-right">
            <button class="btn btn-ghost btn-icon" aria-label=${this.theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'} title="Theme" @click=${() => (this.theme = toggleTheme())}>${icon(this.theme === 'dark' ? 'sun' : 'moon')}</button>
          </div>
        </header>
        <nav class="rail" aria-label="views">
          ${TABS.map((t) =>
            this.setupMode && t.view !== 'setup'
              ? html`<span class="rail-tab is-off" aria-disabled="true" title=${S.setupModeTab}>${icon(t.view)}<span class="rail-label">${t.label}</span>${t.short ? html`<span class="rail-short" aria-hidden="true">${t.short}</span>` : nothing}</span>`
              : html`<a class="rail-tab ${this.route.view === t.view ? 'is-on' : ''}" href="#/${t.view}" aria-current=${this.route.view === t.view ? 'page' : nothing}>${icon(t.view)}<span class="rail-label">${t.label}</span>${t.short ? html`<span class="rail-short" aria-hidden="true">${t.short}</span>` : nothing}</a>`,
          )}
        </nav>
        <main class="content" id="main">
          ${this.offline ? html`<div class="banner banner-offline" role="alert" data-engine-away=${engineAway()?.state ?? 'unreachable'}>${icon('warn')} ${awayText(engineAway())} — panels keep their last data and say which endpoint failed.</div>` : nothing}
          ${this.refused && this.route.view !== 'setup' ? html`<div class="banner banner-offline" role="status" data-refused>${icon('warn')} The server answered 401: an admin password is set on it, and the dashboard has no sign-in yet. Unset QSE_ADMIN_TOKEN to open it again (docs/server.md).</div>` : nothing}
          ${this.setupMode ? html`<div class="banner setup-mode-banner" role="status" data-setup-mode>${icon('info')} ${S.setupModeBanner}</div>` : nothing}
          ${this.renderView()}
        </main>
      </div>
    `;
  }

  private renderView(): TemplateResult {
    if (!this.booted || (this.setupMode && this.route.view !== 'setup')) return html``;
    switch (this.route.view) {
      case 'chat':
        void loadChat();
        return html`<qse-chat .params=${this.route.params} .path=${this.route.path}></qse-chat>`;
      case 'usage':
        return html`<qse-usage .summary=${this.summary} .params=${this.route.params}></qse-usage>`;
      case 'performance':
        if (this.route.params.get('tab') === 'parallel') {
          void loadParallel();
          return html`${perfTabs('parallel')}<qse-parallel></qse-parallel>`;
        }
        this.askParallel();
        return html`${perfTabs('engine', this.engineParallel)}<qse-performance .params=${this.route.params}></qse-performance>`;
      case 'dev':
        return html`<qse-dev .params=${this.route.params}></qse-dev>`;
      case 'playground':
        void loadPlayground();
        return html`<qse-playground .params=${this.route.params}></qse-playground>`;
      case 'conductor':
        void loadConductor();
        return html`<qse-conductor .params=${this.route.params}></qse-conductor>`;
      case 'backends':
        void loadBackends();
        return html`<qse-backends .params=${this.route.params}></qse-backends>`;
      case 'system':
        return html`<qse-system></qse-system>`;
      case 'setup':
        void loadSetup();
        return html`<qse-setup .path=${this.route.path}></qse-setup>`;
    }
  }
}
