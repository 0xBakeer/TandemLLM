// Conductor — how TandemLLM decides, and every knob of it (VIS-42, SRV-129). A header with the
// active profile's three-way switch (raw, shadow, on; "on" asks first) and the orange lines that
// matter everywhere (no speed guard for this build, a config file that failed to load), then seven
// screens: Overview (tiles, the 24 h ribbons, the live decision feed, the monitor's last actions),
// Routing (task classes and their policy, the classifier stages, effort levels, the note wrapper,
// a test box), Monitor (the session monitor's journal and its rules: the tool checks, the
// reasoning arms, the empty-turn nudges, caps and texts), Profiles (tiles and the full editor),
// Packages (the skills the router may inject), Sessions (timelines and the export, router and
// monitor in one) and Compare (precision and promotion, the eval tables, the classifier report).
// Loaded on demand, like the Playground.

import { html, nothing, type TemplateResult } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import { describeError } from '../api/client';
import { conductor, type ConfigInfo, type Meta, type ProfileMode, type ProfileTile, type RouterDoc } from '../api/conductor';
import { speedGuard, type GuardState } from '../lib/conductor';
import { timeShort } from '../lib/format';
import { poll, type Poller } from '../lib/poll';
import { setParams } from '../lib/router';
import { LightElement } from '../ui/base';
import { icon } from '../ui/bits';
import { confirmDialog, modeSeg, withToken } from './conductor/shared';
import './conductor/overview';
import './conductor/routing';
import './conductor/monitor';
import './conductor/profiles';
import './conductor/packages';
import './conductor/sessions';
import './conductor/compare';

export type Screen = 'overview' | 'routing' | 'monitor' | 'profiles' | 'packages' | 'sessions' | 'compare';
const SCREENS: { id: Screen; label: string }[] = [
  { id: 'overview', label: 'Overview' },
  { id: 'routing', label: 'Routing' },
  { id: 'monitor', label: 'Monitor' },
  { id: 'profiles', label: 'Profiles' },
  { id: 'packages', label: 'Packages' },
  { id: 'sessions', label: 'Sessions' },
  { id: 'compare', label: 'Compare' },
];

@customElement('qse-conductor')
export class QseConductor extends LightElement {
  @property({ attribute: false }) params: URLSearchParams = new URLSearchParams();
  @state() private tiles: ProfileTile[] = [];
  @state() private defaultProfile = 'chat';
  @state() private meta: Meta | null = null;
  @state() private config: ConfigInfo | null = null;
  @state() private ids: string[] = [];
  @state() private guard: GuardState | null = null;
  @state() private error: string | null = null;
  @state() private switching = false;
  @state() private reloaded: string | null = null;
  @state() private switchError: string | null = null;
  private poller: Poller | null = null;

  connectedCallback(): void {
    super.connectedCallback();
    this.poller = poll(() => this.load(), 15_000);
    this.poller.start();
  }
  disconnectedCallback(): void {
    this.poller?.stop();
    super.disconnectedCallback();
  }

  private get screen(): Screen {
    const s = this.params.get('screen') as Screen | null;
    return s && SCREENS.some((x) => x.id === s) ? s : 'overview';
  }
  private get active(): string {
    return this.params.get('profile') ?? this.defaultProfile;
  }

  private async load(): Promise<void> {
    try {
      const [p, meta, config, router, evals] = await Promise.all([
        conductor.profiles(),
        this.meta ? Promise.resolve(this.meta) : conductor.meta(),
        conductor.config(),
        conductor.doc<RouterDoc>('router'),
        conductor.evals(),
      ]);
      this.tiles = p.profiles;
      this.defaultProfile = p.default_profile;
      this.meta = meta;
      this.config = config;
      this.ids = router.doc.classes.map((c) => c.id);
      this.guard = speedGuard(evals.reports, evals.engine);
      this.error = null;
    } catch (e) {
      const d = describeError(e);
      this.error = d.status === 404 ? 'The conductor is off on this server (start it with --conductor-config).' : `${d.endpoint}: ${d.message}`;
    }
  }

  private async setMode(mode: ProfileMode): Promise<void> {
    const tile = this.tiles.find((t) => t.name === this.active);
    if (!tile || tile.locked) return;
    if (mode === 'on') {
      const ok = await confirmDialog(`Turn the ${tile.name} profile on?`, [
        'Every decision type set to on in this profile will change what the model sees on its next request: effort, hidden skills and packages, rules.',
        'Shadow decisions stay in shadow. Requests sent with X-Tandem: raw (benchmarks) are not affected.',
      ], 'Turn on');
      if (!ok) return;
    }
    this.switching = true;
    this.switchError = null;
    try {
      const r = await withToken(() => conductor.setMode(tile.name, mode, `mode to ${mode}`, tile.version));
      this.reloaded = `v${r.version} · reloaded ${r.loaded_at ? timeShort(r.loaded_at) : 'now'}`;
      await this.load();
    } catch (e) {
      this.switchError = describeError(e).message;
    } finally {
      this.switching = false;
    }
  }

  render() {
    const tile = this.tiles.find((t) => t.name === this.active);
    const loadErrors = this.config ? Object.entries(this.config.store.documents).filter(([, d]) => d.error) : [];
    return html`<div class="view view-conductor">
      <div class="cond-head">
        <div class="cond-head-row">
          <div class="cond-title">${icon('conductor')} Conductor <span class="muted small num" title=${`the one model; ${this.config?.server.raw_model ?? 'the raw engine name'} is a hidden alias, and X-Tandem: raw serves the raw path`}>TandemLLM, the one model · raw by X-Tandem: raw</span></div>
          <div class="mode-switch" data-mode-switch>
            <label class="field"><span>profile</span>
              <select aria-label="active profile" @change=${(e: Event) => setParams({ profile: (e.target as HTMLSelectElement).value })}>
                ${this.tiles.map((t) => html`<option value=${t.name} ?selected=${t.name === this.active}>${t.name}${t.locked ? ' (locked)' : ''}</option>`)}
              </select>
            </label>
            ${tile ? modeSeg(['raw', 'shadow', 'on'] as const, tile.mode, (m) => void this.setMode(m), { disabled: !!tile.locked || this.switching, label: `mode of ${tile.name}` }) : nothing}
            ${this.reloaded ? html`<span class="reload" role="status" data-reloaded>${this.reloaded}</span>` : nothing}
          </div>
        </div>
        ${this.switchError ? html`<div class="guard-line" role="alert">${icon('warn')} ${this.switchError}</div>` : nothing}
        ${this.error ? html`<div class="guard-line" role="alert">${icon('warn')} ${this.error}</div>` : nothing}
        ${this.guard && this.guard.tone !== 'ok'
          ? html`<div class="guard-line ${this.guard.tone === 'note' ? 'guard-note' : ''}" data-guard=${this.guard.status}>${icon(this.guard.tone === 'note' ? 'info' : 'warn')}<span>${this.guard.line}${this.guard.detail ? html`. <span class="small">${this.guard.detail}</span>` : nothing}</span></div>`
          : nothing}
        ${loadErrors.map(([k, d]) => html`<div class="guard-line" role="alert">${icon('warn')} ${k}: ${d.error?.message} (kept version ${d.version})</div>`)}
        ${tile?.locked ? html`<div class="note" data-locked>${icon('warn')} ${tile.name} is locked: ${tile.locked}.</div>` : nothing}
        <nav class="cond-tabs" aria-label="conductor screens">
          ${SCREENS.map((s) => html`<a class="cond-tab ${s.id === this.screen ? 'is-on' : ''}" href=${`#/conductor?screen=${s.id}${this.params.get('profile') ? `&profile=${encodeURIComponent(this.active)}` : ''}`} aria-current=${s.id === this.screen ? 'page' : nothing}>${s.label}</a>`)}
        </nav>
      </div>
      ${this.renderScreen()}
    </div>`;
  }

  private renderScreen(): TemplateResult {
    const common = { ids: this.ids, meta: this.meta, tiles: this.tiles };
    switch (this.screen) {
      case 'overview':
        return html`<qse-cond-overview .ids=${common.ids} .tiles=${common.tiles} .active=${this.active}></qse-cond-overview>`;
      case 'routing':
        return html`<qse-cond-routing .ids=${common.ids} .meta=${common.meta} .active=${this.active} .onSaved=${() => this.load()}></qse-cond-routing>`;
      case 'monitor':
        return html`<qse-cond-monitor .meta=${common.meta} .onSaved=${() => this.load()}></qse-cond-monitor>`;
      case 'profiles':
        return html`<qse-cond-profiles .ids=${common.ids} .meta=${common.meta} .tiles=${common.tiles} .selected=${this.params.get('edit')} .onSaved=${() => this.load()}></qse-cond-profiles>`;
      case 'packages':
        return html`<qse-cond-packages .ids=${common.ids}></qse-cond-packages>`;
      case 'sessions':
        return html`<qse-cond-sessions .ids=${common.ids} .selected=${this.params.get('session')}></qse-cond-sessions>`;
      case 'compare':
        return html`<qse-cond-compare .ids=${common.ids} .tiles=${common.tiles} .onSaved=${() => this.load()}></qse-cond-compare>`;
    }
  }
}
