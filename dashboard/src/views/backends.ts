// Backends — the proxy's registry (proxy/config_api.py): which engines and cloud APIs this
// dashboard sits in front of, their keys, models, prices, routing rules and users. Screens:
// list (default), a backend's page (?backend=<id>), routing (?screen=routing), users
// (?screen=users), and Setup (?screen=setup, and the first screen while no backend exists).
// Keys are write-only: a saved key shows its last 4 characters and a hash prefix, never more.
// A field the environment sets (TANDEM_*) is locked and names its variable.

import { html, nothing, type TemplateResult } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import { describeError } from '../api/client';
import { config, type BackendConfig, type BackendKind, type ConfigOverview, type FeatureValue, type ModelRow, type Price, type ProxyUser, type Rule, type SeedPrices, type TestResult } from '../api/config';
import { ago, compact, exact, money, ms } from '../lib/format';
import { setParams } from '../lib/router';
import { LightElement, type Loadable } from '../ui/base';
import { copyButton, emptyState, errorState, icon, panel, skeleton } from '../ui/bits';
import { engineCard, ENGINE_KINDS } from './engines';

type Screen = 'list' | 'routing' | 'users' | 'setup';

const KIND_LABEL: Record<BackendKind, string> = {
  tandemllm: 'TandemLLM engine',
  vllm: 'vLLM',
  sglang: 'SGLang',
  generic: 'OpenAI-compatible engine',
  openai: 'OpenAI',
  deepseek: 'DeepSeek',
};

const FEATURE_NOTE: Record<string, string> = {
  mode: 'the harness for this model: off relays bytes unchanged',
  class: 'task class: one short extra request per new user message on a foreign backend',
  effort: 'effort through the backend’s own switch; a budget is a word, not a hard cap',
  packages: 'packages as a note at the end of the user message; costs input tokens every turn',
  side: 'side requests (titles, summaries) capped and with thinking off; saves tokens',
  loops: 'reasoning-loop detection on the streamed words; closing needs continue_final_message',
  e1: 'empty-turn guard: one more full request when it fires',
  tools: 'tool-call checks: hold and repair free, regenerate is one more request',
  compaction: 'compaction keeps the TandemLLM section; the backend reads the client’s history cold',
  states: 'state lines in the stream (waiting, thinking, calling a tool)',
};


/** a vLLM or SGLang card's one line: the engine's version, its page, KV use and requests */
export function engineLine(b: BackendConfig): TemplateResult {
  const e = b.engine;
  if (!e) return html`<p class="bk-card-engine muted">engine: not read yet</p>`;
  const f = e.figures;
  const parts = [
    e.version ? `${KIND_LABEL[b.kind]} ${e.version}` : null,
    e.metrics === 'ok' ? null : e.metrics === 'off' ? 'no /metrics' : e.metrics === 'error' ? 'metrics unreadable' : 'metrics not read yet',
    f?.kv_usage != null ? `KV ${Math.round(f.kv_usage * 100)} %` : null,
    f?.running != null ? `${f.running} running` : null,
    f?.prefix_hit_rate != null ? `prefix hits ${Math.round(f.prefix_hit_rate * 100)} %` : null,
  ].filter(Boolean);
  return html`<p class="bk-card-engine" data-engine-line=${e.metrics}>${parts.join(' · ') || 'engine read'}</p>`;
}

/** a field's lock line: "set by TANDEM_BACKEND_X_URL" */
export function lockOf(b: BackendConfig | null, field: string): string | null {
  return b?.locks?.[field] ?? null;
}

/** the comma list a text field holds */
export function listOf(text: string): string[] {
  return text
    .split(/[,\n]/)
    .map((x) => x.trim())
    .filter(Boolean);
}

interface Draft {
  id: string;
  kind: BackendKind;
  label: string;
  url: string;
  key: string;
  keyAction: 'keep' | 'set' | 'remove';
  enabled: boolean;
  models_mode: 'discover' | 'list';
  models: string;
  fallback: string;
  max_inflight: number;
  inject_usage: boolean;
  /** P1.5: where the System view reads this backend's machine */
  host_mode: HostMode;
  host_url: string;
  host_dcgm: string;
}

type HostMode = 'auto' | 'same' | 'collector' | 'exporter' | 'off';
const HOST_MODES: [HostMode, string][] = [
  ['auto', 'auto: this machine when the URL is'],
  ['same', 'this machine'],
  ['collector', 'remote collector'],
  ['exporter', 'node_exporter + DCGM'],
  ['off', 'none'],
];

function draftFor(b: BackendConfig | null, kind: BackendKind, ov: ConfigOverview | null): Draft {
  if (b)
    return {
      id: b.id, kind: b.kind, label: b.label === b.id ? '' : b.label, url: b.url, key: '', keyAction: 'keep', enabled: b.enabled,
      models_mode: b.models_mode, models: b.models.join(', '), fallback: b.fallback.join(', '), max_inflight: b.max_inflight, inject_usage: b.inject_usage,
      host_mode: (b.host?.mode ?? 'auto') as HostMode, host_url: b.host?.url ?? '', host_dcgm: b.host?.dcgm_url ?? '',
    };
  const k = ov?.kinds?.[kind];
  return {
    id: kind === 'generic' ? 'engine1' : kind === 'vllm' ? 'vllm1' : kind === 'sglang' ? 'sglang1' : kind === 'tandemllm' ? 'engine2' : kind, kind, label: '', url: k?.url ?? '', key: '', keyAction: 'set', enabled: true,
    models_mode: 'discover', models: '', fallback: '', max_inflight: 64, inject_usage: k?.inject_usage ?? true,
    host_mode: 'auto', host_url: '', host_dcgm: '',
  };
}

@customElement('qse-backends')
export class QseBackends extends LightElement {
  @property({ attribute: false }) params: URLSearchParams = new URLSearchParams();
  @state() private ov: Loadable<ConfigOverview> = { state: 'loading' };
  @state() private off: string | null = null;
  @state() private draft: Draft | null = null;
  @state() private editing: string | null = null; // backend id, or '' for a new one
  @state() private test: (TestResult & { busy?: boolean }) | null = null;
  @state() private busy = false;
  @state() private formError: { message: string; field?: string; variable?: string } | null = null;
  @state() private newKey: { label: string; key: string } | null = null;
  @state() private rules: Rule[] | null = null;
  @state() private seed: SeedPrices | null = null;
  @state() private modelDrafts: Record<string, Partial<ModelRow> & { price?: Price | null }> = {};
  @state() private flash: string | null = null;
  private timer: ReturnType<typeof setInterval> | null = null;

  connectedCallback(): void {
    super.connectedCallback();
    void this.load();
    this.timer = setInterval(() => document.visibilityState === 'visible' && !this.editing && void this.load(true), 15_000);
  }
  disconnectedCallback(): void {
    if (this.timer) clearInterval(this.timer);
    super.disconnectedCallback();
  }

  private get screen(): Screen {
    const s = this.params.get('screen');
    if (s === 'routing' || s === 'users' || s === 'setup') return s;
    if (this.ov.state === 'ready' && this.ov.data.setup_needed && !this.params.get('backend')) return 'setup';
    return 'list';
  }
  private get selected(): string | null {
    return this.params.get('backend');
  }

  private async load(quiet = false): Promise<void> {
    if (!quiet) this.ov = { state: 'loading' };
    try {
      const d = await config.overview();
      this.ov = { state: 'ready', data: d };
      this.off = null;
      if (!this.rules || !quiet) this.rules = d.rules.map((r) => ({ ...r }));
    } catch (e) {
      const d = describeError(e);
      if (d.status === 404 && /--backends off/.test(d.message)) {
        this.off = d.message;
        this.ov = { state: 'empty' };
      } else this.ov = { state: 'error', endpoint: d.endpoint, message: d.message };
    }
  }

  private say(msg: string): void {
    this.flash = msg;
    setTimeout(() => {
      if (this.flash === msg) this.flash = null;
    }, 4000);
  }

  private fail(e: unknown): void {
    const d = describeError(e) as { message: string; status: number };
    const err = (e as { body?: { error?: { field?: string; variable?: string } } })?.body?.error;
    this.formError = { message: d.message, field: err?.field, variable: err?.variable };
  }

  // ---- render ---------------------------------------------------------------------------
  render() {
    return html`<div class="view view-backends">
      ${this.renderNav()}
      ${this.flash ? html`<p class="note note-info" role="status">${icon('check')} ${this.flash}</p>` : nothing}
      ${this.renderBody()}
    </div>`;
  }

  private renderNav(): TemplateResult {
    const tabs: { s: Screen; label: string }[] = [
      { s: 'list', label: 'Backends' },
      { s: 'routing', label: 'Routing' },
      { s: 'users', label: 'Users' },
      { s: 'setup', label: 'Setup' },
    ];
    const cur = this.selected ? 'list' : this.screen;
    return html`<div class="toolbar bk-nav" role="group" aria-label="backends screens">
      <div class="seg" role="tablist">
        ${tabs.map((t) => html`<button role="tab" aria-selected=${cur === t.s} class="seg-btn ${cur === t.s ? 'is-on' : ''}" @click=${() => setParams({ screen: t.s === 'list' ? null : t.s, backend: null })}>${t.label}</button>`)}
      </div>
      ${this.ov.state === 'ready' ? html`<span class="muted small bk-keysrc">${this.keySource(this.ov.data)}</span>` : nothing}
    </div>`;
  }

  private keySource(d: ConfigOverview): string {
    const where = d.key_source === 'env' ? 'TANDEM_SECRETS_KEY' : d.key_source === 'keychain' ? 'the OS keychain' : d.key_source === 'file' ? 'the key file beside the config' : 'made on the first saved key';
    return `keys at rest: AES-256-GCM, master key from ${where}`;
  }

  private renderBody(): TemplateResult {
    if (this.off)
      return panel('The proxy is off on this server', html`<div class="state"><p class="state-body">This engine runs without <code>--backends</code>, so it serves its own model only. To put other engines and cloud APIs behind this dashboard, set <code>QSE_BACKENDS=~/.tandemllm/config.sqlite3</code> in ops/serve.env and restart, or run the standalone proxy: <code>python -m proxy</code>. docs/proxy.md has both.</p></div>`, { id: 'proxy-off' });
    const o = this.ov;
    if (o.state === 'loading') return panel('Backends', skeleton(4));
    if (o.state === 'error') return panel('Backends', errorState(o.endpoint, o.message, () => this.load()));
    if (o.state !== 'ready') return nothing as never;
    const d = o.data;
    if (this.editing !== null && this.draft) return this.renderForm(d);
    if (this.selected) {
      const b = d.backends.find((x) => x.id === this.selected);
      return b ? this.renderDetail(d, b) : panel('Backend', emptyState(`No backend '${this.selected}'`));
    }
    switch (this.screen) {
      case 'setup':
        return this.renderSetup(d);
      case 'routing':
        return this.renderRouting(d);
      case 'users':
        return this.renderUsers(d);
      default:
        return this.renderList(d);
    }
  }

  // ---- setup ----------------------------------------------------------------------------
  private renderSetup(d: ConfigOverview): TemplateResult {
    const card = (kind: BackendKind, title: string, body: string, cta: string) => html`<button class="bk-setup-card" data-kind=${kind} @click=${() => this.startAdd(kind, d)}>
      <span class="bk-setup-title">${title}</span>
      <span class="bk-setup-body">${body}</span>
      <span class="btn btn-primary btn-sm">${cta}</span>
    </button>`;
    return panel(
      d.setup_needed ? 'Set up the first backend' : 'Add a backend',
      html`<p class="panel-sub bk-lead">${d.engine?.embedded ? html`This server runs <b>${d.engine.model}</b> itself; every backend you add sits beside it, routed by model id.` : html`Pick what this dashboard sits in front of. One backend tested and saved, and its models show up in <code>/v1/models</code> under their real names.`}</p>
        <div class="bk-setup">
          ${card('tandemllm', 'Our engine', 'A TandemLLM server at a URL (another box, or this one behind the standalone proxy).', 'Add a TandemLLM URL')}
          ${card('vllm', 'vLLM', 'A vLLM server: its requests relayed, and its own /metrics read for KV, prefix-cache hits, the queue and speculation.', 'Add vLLM')}
          ${card('sglang', 'SGLang', 'An SGLang server started with --enable-metrics: relayed, with its KV, cache hit rate and speculation read.', 'Add SGLang')}
          ${card('generic', 'Another engine', 'llama.cpp, LM Studio, Ollama or any OpenAI-compatible URL: relayed and measured at the proxy.', 'Add an engine')}
          ${card('deepseek', 'A cloud provider', 'DeepSeek or OpenAI with your own key. The key is stored encrypted and never shown again.', 'Add DeepSeek')}
          ${card('openai', 'OpenAI', 'Chat completions, embeddings and the Responses API, relayed as they are.', 'Add OpenAI')}
        </div>
        <p class="panel-foot">Headless instead: the same objects as TANDEM_* variables in an env file (<code>TANDEM_BACKENDS=deepseek</code>, <code>TANDEM_BACKEND_DEEPSEEK_KIND=deepseek</code>, <code>…_KEY</code>). A field a variable sets is locked here and names the variable.</p>`,
      { id: 'setup' },
    );
  }

  private startAdd(kind: BackendKind, d: ConfigOverview): void {
    const base = draftFor(null, kind, d);
    let id = base.id;
    for (let i = 2; d.backends.some((b) => b.id === id); i++) id = `${base.id}${i}`;
    this.draft = { ...base, id };
    this.editing = '';
    this.test = null;
    this.formError = null;
  }

  // ---- the list -------------------------------------------------------------------------
  private renderList(d: ConfigOverview): TemplateResult {
    const tools = html`<button class="btn btn-primary btn-sm" @click=${() => setParams({ screen: 'setup', backend: null })}>${icon('plus')} Add backend</button>`;
    if (!d.backends.length) return panel('Backends', emptyState('No backend yet', html`<button class="btn btn-primary" @click=${() => setParams({ screen: 'setup' })}>Open Setup</button>`), { tools });
    return panel(
      'Backends',
      html`<div class="bk-cards" role="list">${d.backends.map((b) => this.renderCard(b))}</div>
        ${d.engine?.embedded ? html`<p class="panel-foot">${d.engine.model} is this engine's own model: it is not a backend here and its requests never pass the relay.</p>` : nothing}`,
      { sub: `${d.backends.length} configured · ${d.config_path}`, tools, id: 'backends' },
    );
  }

  private renderCard(b: BackendConfig): TemplateResult {
    const st = b.enabled ? b.health.state : 'disabled';
    const models = (b.model_rows ?? []).filter((m) => m.present && m.enabled).length;
    return html`<article class="bk-card is-${st}" role="listitem" data-backend=${b.id}>
      <header class="bk-card-head">
        <a class="bk-card-title" href=${`#/backends?backend=${encodeURIComponent(b.id)}`}>${b.label}</a>
        <span class="pill bk-state is-${st}" data-state=${st}>${st}</span>
      </header>
      <div class="bk-card-kind"><span class="badge">${KIND_LABEL[b.kind] ?? b.kind}</span>${b.cloud ? html`<span class="badge">cloud</span>` : nothing}${Object.keys(b.locks).length ? html`<span class="badge badge-lock" title="fields set by environment variables">${icon('info')} env</span>` : nothing}</div>
      <div class="bk-card-url mono clip" title=${b.url}>${b.url}</div>
      <dl class="bk-card-facts">
        <div><dt>models</dt><dd class="num">${models}</dd></div>
        <div><dt>in flight</dt><dd class="num">${b.in_flight}<span class="muted">/${b.max_inflight}</span></dd></div>
        <div><dt>latency</dt><dd class="num">${b.health.latency_ms != null ? ms(b.health.latency_ms) : '—'}</dd></div>
        <div><dt>cost today</dt><dd class="num">${money((b as BackendConfig & { cost_today_micros?: number | null }).cost_today_micros)}</dd></div>
      </dl>
      ${b.health.last_error && st !== 'up' ? html`<p class="bk-err" role="alert">${icon('warn')} ${b.health.last_error}</p>` : nothing}
      ${ENGINE_KINDS.has(b.kind) ? engineLine(b) : nothing}
      <footer class="bk-card-foot">
        <span class="muted small">key ${b.key.key_set ? html`…${b.key.last4 ?? '????'}` : 'none'} · harness ${b.harness.mode}</span>
        <span class="grow"></span>
        <button class="btn btn-ghost btn-sm" @click=${() => this.retest(b)}>Test</button>
        <button class="btn btn-ghost btn-sm" @click=${() => this.startEdit(b)}>${icon('edit')} Edit</button>
      </footer>
    </article>`;
  }

  private async retest(b: BackendConfig): Promise<void> {
    try {
      const r = await config.retest(b.id);
      this.say(r.ok ? `${b.id}: up, ${ms(r.latency_ms)}, ${r.models.length} models` : `${b.id}: ${r.error ?? 'down'}`);
    } catch (e) {
      this.say(`${b.id}: ${describeError(e).message}`);
    }
    void this.load(true);
  }

  private startEdit(b: BackendConfig): void {
    this.draft = draftFor(b, b.kind, null);
    this.editing = b.id;
    this.test = null;
    this.formError = null;
  }

  // ---- add / edit -----------------------------------------------------------------------
  private renderForm(d: ConfigOverview): TemplateResult {
    const dr = this.draft as Draft;
    const isNew = this.editing === '';
    const b = isNew ? null : d.backends.find((x) => x.id === this.editing) ?? null;
    const set = (patch: Partial<Draft>) => (this.draft = { ...dr, ...patch });
    const lock = (f: string) => lockOf(b, f);
    const field = (name: string, label: string, input: TemplateResult, hint?: string | TemplateResult) => {
      const l = lock(name);
      const bad = this.formError?.field === name || (this.formError?.variable && this.formError.variable === l);
      return html`<label class="field field-block bk-field ${l ? 'is-locked' : ''} ${bad ? 'is-bad' : ''}" data-field=${name}>
        <span>${label}${l ? html` <span class="badge badge-lock" title="the environment wins over the GUI">${icon('info')} set by ${l}</span>` : nothing}</span>
        ${input}
        ${hint ? html`<span class="hint">${hint}</span>` : nothing}
      </label>`;
    };
    const kinds: BackendKind[] = ['tandemllm', 'vllm', 'sglang', 'generic', 'openai', 'deepseek'];
    const cloud = d.kinds[dr.kind]?.cloud;
    const keyHint = b?.key.key_set
      ? html`saved: ends in <b class="mono">${b.key.last4 ?? '????'}</b>, sha256 <span class="mono">${b.key.sha256_6}</span>${b.key.source === 'env' ? ' (from the environment)' : ''}${b.key.readable === false ? html` <b class="err-ink">unreadable: the master key changed, enter it again</b>` : nothing}. Leave empty to keep it.`
      : cloud
        ? 'Stored encrypted, never shown again.'
        : 'Only if the server needs a bearer token.';
    return panel(
      isNew ? `Add: ${KIND_LABEL[dr.kind]}` : `Edit ${b?.label ?? dr.id}`,
      html`<form class="bk-form" @submit=${(e: Event) => this.save(e, isNew)} novalidate>
        <div class="bk-form-grid">
          ${field('kind', 'Kind', html`<select ?disabled=${!!lock('kind')} .value=${dr.kind} @change=${(e: Event) => {
            const kind = (e.target as HTMLSelectElement).value as BackendKind;
            const k = d.kinds[kind];
            set({ kind, url: isNew && k?.url ? k.url : dr.url, inject_usage: k?.inject_usage ?? true });
          }}>${kinds.map((k) => html`<option value=${k} ?selected=${k === dr.kind}>${KIND_LABEL[k]}</option>`)}</select>`)}
          ${field('id', 'Id', html`<input class="mono" .value=${dr.id} ?disabled=${!isNew} pattern="[a-z0-9][a-z0-9_-]{0,31}" @input=${(e: Event) => set({ id: (e.target as HTMLInputElement).value.trim() })} />`, isNew ? 'lower case, digits, - and _; it prefixes a model when two backends serve one id' : undefined)}
          ${field('label', 'Label', html`<input .value=${dr.label} placeholder=${dr.id} @input=${(e: Event) => set({ label: (e.target as HTMLInputElement).value })} />`)}
          ${field('url', 'Base URL', html`<input class="mono" type="url" .value=${dr.url} ?disabled=${!!lock('url')} placeholder="http://host:8000/v1" @input=${(e: Event) => set({ url: (e.target as HTMLInputElement).value.trim() })} />`, 'the OpenAI SDK’s base_url; a bare host gets /v1')}
          ${field(
            'key',
            'API key',
            html`<div class="bk-key">
              <input type="password" autocomplete="new-password" .value=${dr.key} ?disabled=${!!lock('key') || dr.keyAction === 'remove'} placeholder=${b?.key.key_set ? `•••• ${b.key.last4 ?? ''}` : 'sk-…'} @input=${(e: Event) => set({ key: (e.target as HTMLInputElement).value, keyAction: (e.target as HTMLInputElement).value ? 'set' : isNew ? 'set' : 'keep' })} />
              ${b?.key.key_set && !lock('key') ? html`<button type="button" class="btn btn-ghost btn-sm" @click=${() => set({ keyAction: dr.keyAction === 'remove' ? 'keep' : 'remove', key: '' })}>${dr.keyAction === 'remove' ? 'Keep the key' : 'Remove key'}</button>` : nothing}
            </div>`,
            keyHint,
          )}
          ${field('max_inflight', 'Max in flight', html`<input type="number" min="1" max="4096" .value=${String(dr.max_inflight)} ?disabled=${!!lock('max_inflight')} @input=${(e: Event) => set({ max_inflight: Number((e.target as HTMLInputElement).value) })} />`, 'past it a request gets 503 and Retry-After (or its fallback)')}
          ${field('models_mode', 'Models', html`<select .value=${dr.models_mode} ?disabled=${!!lock('models_mode')} @change=${(e: Event) => set({ models_mode: (e.target as HTMLSelectElement).value as Draft['models_mode'] })}><option value="discover" ?selected=${dr.models_mode === 'discover'}>discover (GET /models)</option><option value="list" ?selected=${dr.models_mode === 'list'}>this list only</option></select>`)}
          ${field('models', dr.models_mode === 'list' ? 'Model ids' : 'Extra model ids', html`<input class="mono" .value=${dr.models} ?disabled=${!!lock('models')} placeholder="deepseek-flash, deepseek-v4-pro" @input=${(e: Event) => set({ models: (e.target as HTMLInputElement).value })} />`, 'comma separated')}
          ${field('fallback', 'Fallback backends', html`<input class="mono" .value=${dr.fallback} ?disabled=${!!lock('fallback')} placeholder=${d.backends.filter((x) => x.id !== dr.id).map((x) => x.id).join(', ') || 'none yet'} @input=${(e: Event) => set({ fallback: (e.target as HTMLInputElement).value })} />`, 'tried in order on a connect error, 429, 502 or 503 before the first byte')}
          ${field('host.mode', 'Host', html`<select .value=${dr.host_mode} ?disabled=${!!lock('host.mode')} @change=${(e: Event) => set({ host_mode: (e.target as HTMLSelectElement).value as HostMode })}>${HOST_MODES.map(([m, w]) => html`<option value=${m} ?selected=${m === dr.host_mode}>${w}</option>`)}</select>`, 'where the System view reads the machine behind this backend')}
          ${dr.host_mode === 'collector' || dr.host_mode === 'exporter'
            ? field('host.url', dr.host_mode === 'collector' ? 'Collector URL' : 'node_exporter URL', html`<input class="mono" type="url" .value=${dr.host_url} ?disabled=${!!lock('host.url')} placeholder=${dr.host_mode === 'collector' ? 'http://gpu-box:9108' : 'http://gpu-box:9100/metrics'} @input=${(e: Event) => set({ host_url: (e.target as HTMLInputElement).value.trim() })} />`, dr.host_mode === 'collector' ? html`<code>python -m proxy.hostcollect --host 0.0.0.0</code> on that host; its token in TANDEM_COLLECTOR_TOKEN on both` : 'machine figures only, no per-process memory')
            : nothing}
          ${dr.host_mode === 'exporter'
            ? field('host.dcgm_url', 'DCGM exporter URL', html`<input class="mono" type="url" .value=${dr.host_dcgm} ?disabled=${!!lock('host.dcgm_url')} placeholder="http://gpu-box:9400/metrics" @input=${(e: Event) => set({ host_dcgm: (e.target as HTMLInputElement).value.trim() })} />`, 'GPU use, framebuffer, power, temperature, clocks')
            : nothing}
          <div class="bk-switches">
            <label class="switch ${lock('enabled') ? 'is-locked' : ''}"><input type="checkbox" .checked=${dr.enabled} ?disabled=${!!lock('enabled')} @change=${(e: Event) => set({ enabled: (e.target as HTMLInputElement).checked })} /> enabled</label>
            <label class="switch" title="ask a stream for its usage frame and drop it before the client"><input type="checkbox" .checked=${dr.inject_usage} ?disabled=${!!lock('inject_usage')} @change=${(e: Event) => set({ inject_usage: (e.target as HTMLInputElement).checked })} /> ask streams for usage</label>
          </div>
        </div>
        ${this.formError ? html`<p class="login-error" role="alert">${this.formError.message}</p>` : nothing}
        ${this.renderTest()}
        <div class="bk-form-foot">
          <button type="button" class="btn btn-ghost" ?disabled=${this.test?.busy} @click=${() => this.runTest(d, b)}>${icon('refresh')} Test connection</button>
          <span class="grow"></span>
          ${!isNew && b ? html`<button type="button" class="btn btn-ghost bk-delete" @click=${() => this.deleteBackend(b)}>${icon('trash')} Delete</button>` : nothing}
          <button type="button" class="btn btn-ghost" @click=${() => this.cancel()}>Cancel</button>
          <button type="submit" class="btn btn-primary" ?disabled=${this.busy}>${this.busy ? 'Saving…' : isNew ? 'Save backend' : 'Save'}</button>
        </div>
      </form>`,
      { id: 'backend-form' },
    );
  }

  private renderTest(): TemplateResult | typeof nothing {
    const t = this.test;
    if (!t) return nothing;
    if (t.busy) return html`<p class="note note-info bk-test" role="status">Testing…</p>`;
    return html`<div class="bk-test ${t.ok ? 'is-ok' : 'is-bad'}" role="status" data-test=${t.ok ? 'ok' : 'fail'}>
      ${t.ok ? html`${icon('check')} reachable in <b class="num">${ms(t.latency_ms)}</b>, ${t.models.length} model${t.models.length === 1 ? '' : 's'}` : html`${icon('warn')} ${t.error ?? 'not reachable'}`}
      ${t.ok && t.models.length ? html`<div class="bk-test-models mono">${t.models.slice(0, 24).join(' · ')}${t.models.length > 24 ? ` · and ${t.models.length - 24} more` : ''}</div>` : nothing}
    </div>`;
  }

  private async runTest(d: ConfigOverview, b: BackendConfig | null): Promise<void> {
    const dr = this.draft as Draft;
    this.test = { ok: false, latency_ms: null, error: null, models: [], busy: true };
    try {
      const body: { kind: BackendKind; url?: string; key?: string; id?: string } = { kind: dr.kind, url: dr.url || d.kinds[dr.kind]?.url || undefined };
      if (dr.key) body.key = dr.key;
      else if (b?.key.key_set && dr.keyAction !== 'remove') body.id = b.id;
      this.test = await config.test(body);
    } catch (e) {
      this.test = { ok: false, latency_ms: null, error: describeError(e).message, models: [] };
    }
  }

  private async save(e: Event, isNew: boolean): Promise<void> {
    e.preventDefault();
    const dr = this.draft as Draft;
    const d = this.ov.state === 'ready' ? this.ov.data : null;
    const b = isNew ? null : d?.backends.find((x) => x.id === this.editing) ?? null;
    this.busy = true;
    this.formError = null;
    const doc: Record<string, unknown> = {
      kind: dr.kind, label: dr.label || null, url: dr.url, enabled: dr.enabled, models_mode: dr.models_mode,
      models: listOf(dr.models), fallback: listOf(dr.fallback), max_inflight: dr.max_inflight, inject_usage: dr.inject_usage,
      host: { mode: dr.host_mode, url: dr.host_url || null, dcgm_url: dr.host_dcgm || null },
    };
    // a locked field is sent unchanged or not at all
    if (b) for (const f of Object.keys(b.locks)) delete doc[f];
    try {
      if (isNew) {
        if (dr.key) doc.key = dr.key;
        const nb = await config.create({ ...(doc as object), id: dr.id, kind: dr.kind } as never);
        this.say(`${nb.id} saved${nb.test ? `: ${nb.test.state}${nb.test.latency_ms != null ? `, ${ms(nb.test.latency_ms)}` : ''}` : ''}`);
        this.editing = null;
        setParams({ screen: null, backend: nb.id });
      } else if (b) {
        if (dr.keyAction === 'remove') doc.key = null;
        else if (dr.key) doc.key = dr.key;
        await config.update(b.id, doc);
        this.say(`${b.id} saved`);
        this.editing = null;
      }
      await this.load(true);
    } catch (err) {
      this.fail(err);
    } finally {
      this.busy = false;
    }
  }

  private async deleteBackend(b: BackendConfig): Promise<void> {
    if (!confirm(`Delete backend ${b.id}? Its key, models, prices and rules go with it. The ledger's rows stay.`)) return;
    try {
      await config.remove(b.id);
      this.say(`${b.id} deleted`);
      this.editing = null;
      setParams({ backend: null });
      await this.load(true);
    } catch (err) {
      this.fail(err);
    }
  }

  private cancel(): void {
    this.editing = null;
    this.draft = null;
    this.formError = null;
    this.test = null;
  }

  // ---- a backend's page -----------------------------------------------------------------
  private renderDetail(d: ConfigOverview, b: BackendConfig): TemplateResult {
    const rows = b.model_rows ?? [];
    const tools = html`<button class="btn btn-ghost btn-sm" @click=${() => this.retest(b)}>${icon('refresh')} Test</button><button class="btn btn-ghost btn-sm" @click=${() => this.startEdit(b)}>${icon('edit')} Edit</button><button class="btn btn-ghost btn-sm" @click=${() => setParams({ backend: null })}>${icon('x')} Close</button>`;
    return html`${panel(
      html`${b.label} <span class="pill bk-state is-${b.enabled ? b.health.state : 'disabled'}">${b.enabled ? b.health.state : 'disabled'}</span>`,
      html`<dl class="kv-grid bk-detail-kv">
          <div><dt>kind</dt><dd>${KIND_LABEL[b.kind]}</dd></div>
          <div><dt>url</dt><dd class="mono clip">${b.url}</dd></div>
          <div><dt>key</dt><dd>${b.key.key_set ? html`<span class="mono">…${b.key.last4 ?? '????'}</span> <span class="muted">sha256 ${b.key.sha256_6} · ${b.key.source}</span>` : 'none'}</dd></div>
          <div><dt>health</dt><dd>${b.health.state}${b.health.since ? html` <span class="muted">since ${ago(new Date(b.health.since * 1000).toISOString())}</span>` : nothing}${b.health.latency_ms != null ? html` · ${ms(b.health.latency_ms)}` : nothing}</dd></div>
          <div><dt>in flight</dt><dd class="num">${b.in_flight} of ${b.max_inflight}</dd></div>
          <div><dt>timeouts</dt><dd class="num">connect ${b.timeouts.connect} s · first byte ${b.timeouts.first_byte} s · idle ${b.timeouts.idle} s</dd></div>
          <div><dt>routes</dt><dd class="mono small">${b.routes.join(' ')}</dd></div>
          <div><dt>host</dt><dd>${HOST_MODES.find(([m]) => m === (b.host?.mode ?? 'auto'))?.[1] ?? b.host?.mode}${b.host?.url ? html` <span class="mono small">${b.host.url}</span>` : nothing} · <a href="#/system">System</a></dd></div>
          <div><dt>fallback</dt><dd>${b.fallback.length ? b.fallback.join(' → ') : 'none'}</dd></div>
          <div><dt>updated</dt><dd>${ago(new Date(b.updated_at).toISOString())} by ${b.updated_by ?? '—'}</dd></div>
        </dl>
        ${b.health.last_error ? html`<p class="bk-err" role="alert">${icon('warn')} ${b.health.last_error}</p>` : nothing}`,
      { tools, id: 'backend-detail' },
    )}
    ${ENGINE_KINDS.has(b.kind) ? panel('Engine', engineCard({ id: b.id, label: b.label, kind: b.kind, engine: b.engine ?? null, figures: b.engine?.figures ?? null }), { id: 'backend-engine', sub: 'its own /metrics and discovery route, read every 15 s (layer L1)' }) : nothing}
    ${this.renderModels(d, b, rows)}`;
  }

  private async loadSeed(): Promise<void> {
    if (this.seed) return;
    try {
      this.seed = await config.seedPrices();
    } catch {
      this.seed = { as_of: '', note: '', providers: {} };
    }
  }

  private seedPrice(b: BackendConfig, model: string): Price | null {
    const p = this.seed?.providers?.[b.kind]?.[model];
    return p ?? null;
  }

  private renderModels(d: ConfigOverview, b: BackendConfig, rows: ModelRow[]): TemplateResult {
    void this.loadSeed();
    if (!rows.length) return panel('Models', emptyState('No models yet', b.models_mode === 'discover' ? 'The first health check lists them; Test runs one now.' : 'Add model ids in Edit.'), { id: 'models' });
    const groups = d.feature_groups;
    return panel(
      'Models',
      html`<div class="table-scroll"><table class="table bk-models">
          <thead><tr><th>model</th><th>public id</th><th class="num-col">context</th><th class="num-col" title="USD per million tokens">input</th><th class="num-col">cached</th><th class="num-col">output</th><th>on</th><th></th></tr></thead>
          <tbody>${rows.map((m) => this.renderModelRow(b, m, groups))}</tbody>
        </table></div>
        <p class="panel-foot">Prices are USD per million tokens, applied to a row when it is written, so a later change does not rewrite history. ${this.seed?.as_of ? html`The fill button uses the public list prices of ${this.seed.as_of} shipped with this release; nothing is fetched and nothing is applied without a click.` : nothing} Feature switches are stored now and act from the harness release (P4); this release relays every byte unchanged.</p>`,
      { sub: `${rows.filter((m) => m.present).length} listed by the backend`, id: 'models' },
    );
  }

  private renderModelRow(b: BackendConfig, m: ModelRow, groups: string[]): TemplateResult {
    const key = `${b.id}/${m.model}`;
    const dr = this.modelDrafts[key];
    const price = (dr?.price !== undefined ? dr.price : m.price) ?? null;
    const lockP = m.locks?.price;
    const seed = this.seedPrice(b, m.model);
    const setPrice = (k: keyof Price, v: string) => {
      const cur: Price = { input: null, cached_input: null, cache_write: null, output: null, ...(price ?? {}) };
      (cur as unknown as Record<string, unknown>)[k] = v === '' ? null : Number(v);
      this.modelDrafts = { ...this.modelDrafts, [key]: { ...dr, price: cur } };
    };
    const num = (k: keyof Price) => html`<input class="bk-price" type="number" min="0" step="0.001" inputmode="decimal" aria-label=${`${m.model} ${k} price`} ?disabled=${!!lockP} .value=${price?.[k] != null ? String(price[k]) : ''} @input=${(e: Event) => setPrice(k, (e.target as HTMLInputElement).value)} />`;
    const changed = !!dr;
    const open = this.params.get('model') === m.model;
    return html`<tr class=${m.present ? '' : 'is-absent'} data-model=${m.model}>
        <td class="mono cell-wrap">${m.model}${m.present ? nothing : html` <span class="muted small">not listed now</span>`}</td>
        <td><input class="bk-pub mono" .value=${dr?.public_id ?? m.public_id ?? ''} placeholder="(same)" aria-label=${`${m.model} public id`} @input=${(e: Event) => (this.modelDrafts = { ...this.modelDrafts, [key]: { ...dr, public_id: (e.target as HTMLInputElement).value || null } })} /></td>
        <td class="num-col num">${m.context_length ? compact(m.context_length) : '—'}</td>
        <td class="num-col">${num('input')}</td>
        <td class="num-col">${num('cached_input')}</td>
        <td class="num-col">${num('output')}</td>
        <td><label class="switch"><input type="checkbox" aria-label=${`${m.model} enabled`} .checked=${dr?.enabled ?? m.enabled} @change=${(e: Event) => (this.modelDrafts = { ...this.modelDrafts, [key]: { ...dr, enabled: (e.target as HTMLInputElement).checked } })} /></label></td>
        <td class="bk-row-tools">
          ${lockP ? html`<span class="badge badge-lock" title="price set by ${lockP}">${lockP}</span>` : seed && !price ? html`<button class="btn btn-ghost btn-sm" title="the list price of ${this.seed?.as_of}" @click=${() => (this.modelDrafts = { ...this.modelDrafts, [key]: { ...dr, price: { ...seed } } })}>fill price</button>` : nothing}
          <button class="btn btn-ghost btn-sm" aria-expanded=${open} @click=${() => setParams({ model: open ? null : m.model })}>${icon('sliders')}</button>
          ${changed ? html`<button class="btn btn-primary btn-sm" @click=${() => this.saveModel(b, m)}>Save</button>` : nothing}
        </td>
      </tr>
      ${open
        ? html`<tr class="bk-features-row"><td colspan="8"><div class="bk-features">${groups.map((g) => {
            const v = (dr?.features?.[g] ?? m.features[g] ?? 'inherit') as FeatureValue;
            return html`<label class="field field-block bk-feature"><span>${g}</span><select .value=${v} @change=${(e: Event) => (this.modelDrafts = { ...this.modelDrafts, [key]: { ...dr, features: { ...(dr?.features ?? {}), [g]: (e.target as HTMLSelectElement).value as FeatureValue } } })}>${(['inherit', 'off', 'shadow', 'on'] as FeatureValue[]).map((x) => html`<option value=${x} ?selected=${x === v}>${x}</option>`)}</select><span class="hint">${FEATURE_NOTE[g] ?? ''}</span></label>`;
          })}</div></td></tr>`
        : nothing}`;
  }

  private async saveModel(b: BackendConfig, m: ModelRow): Promise<void> {
    const key = `${b.id}/${m.model}`;
    const dr = this.modelDrafts[key] ?? {};
    const patch: Record<string, unknown> = {};
    if (dr.public_id !== undefined) patch.public_id = dr.public_id;
    if (dr.enabled !== undefined) patch.enabled = dr.enabled;
    if (dr.price !== undefined) patch.price = dr.price ? { input: dr.price.input, cached_input: dr.price.cached_input, cache_write: dr.price.cache_write, output: dr.price.output, currency: 'USD', ...(dr.price.offpeak ? { offpeak: dr.price.offpeak } : {}) } : null;
    if (dr.features) patch.features = dr.features;
    try {
      await config.model(b.id, m.model, patch);
      const next = { ...this.modelDrafts };
      delete next[key];
      this.modelDrafts = next;
      this.say(`${m.model} saved`);
      await this.load(true);
    } catch (err) {
      this.say(`${m.model}: ${describeError(err).message}`);
    }
  }

  // ---- routing --------------------------------------------------------------------------
  private renderRouting(d: ConfigOverview): TemplateResult {
    const rules = this.rules ?? [];
    const ids = d.backends.map((b) => b.id);
    const setRule = (i: number, patch: Partial<Rule>) => (this.rules = rules.map((r, j) => (j === i ? { ...r, ...patch } : r)));
    const move = (i: number, by: number) => {
      const j = i + by;
      if (j < 0 || j >= rules.length) return;
      const next = rules.slice();
      [next[i], next[j]] = [next[j], next[i]];
      this.rules = next;
    };
    const dirty = JSON.stringify(rules.map((r) => [r.id, r.pattern, r.backend, r.enabled])) !== JSON.stringify(d.rules.map((r) => [r.id, r.pattern, r.backend, r.enabled]));
    return panel(
      'Routing rules',
      html`<p class="panel-sub bk-lead">A request goes to: an explicit <code>&lt;backend&gt;/</code> prefix, else the first rule whose pattern matches the model id, else the one backend that serves it, else a 404 <code>model_not_found</code>. Patterns are shell globs (<code>gpt-*</code>).</p>
        ${rules.length
          ? html`<ol class="bk-rules">${rules.map((r, i) => {
              const locked = r.source === 'env';
              return html`<li class="bk-rule ${locked ? 'is-locked' : ''}" data-rule=${i}>
                <span class="bk-rule-n num">${i + 1}</span>
                <input class="mono" aria-label="pattern" .value=${r.pattern} ?disabled=${locked} @input=${(e: Event) => setRule(i, { pattern: (e.target as HTMLInputElement).value })} />
                <span class="muted">→</span>
                <select aria-label="backend" ?disabled=${locked} @change=${(e: Event) => setRule(i, { backend: (e.target as HTMLSelectElement).value })}>${ids.map((id) => html`<option value=${id} ?selected=${id === r.backend}>${id}</option>`)}</select>
                <label class="switch"><input type="checkbox" aria-label="enabled" .checked=${r.enabled} ?disabled=${locked} @change=${(e: Event) => setRule(i, { enabled: (e.target as HTMLInputElement).checked })} /></label>
                ${locked ? html`<span class="badge badge-lock">set by ${r.variable}</span>` : nothing}
                <span class="grow"></span>
                <button class="btn btn-ghost btn-icon btn-sm" aria-label="move up" @click=${() => move(i, -1)}>↑</button>
                <button class="btn btn-ghost btn-icon btn-sm" aria-label="move down" @click=${() => move(i, 1)}>↓</button>
                ${locked ? nothing : html`<button class="btn btn-ghost btn-icon btn-sm" aria-label="remove rule" @click=${() => (this.rules = rules.filter((_, j) => j !== i))}>${icon('trash')}</button>`}
              </li>`;
            })}</ol>`
          : emptyState('No rules', 'Each model then goes to the one backend that serves it.')}
        <div class="bk-form-foot">
          <button class="btn btn-ghost" ?disabled=${!ids.length} @click=${() => (this.rules = [...rules, { pattern: '', backend: ids[0], enabled: true }])}>${icon('plus')} Add rule</button>
          <span class="grow"></span>
          ${dirty ? html`<button class="btn btn-ghost" @click=${() => (this.rules = d.rules.map((r) => ({ ...r })))}>Reset</button><button class="btn btn-primary" @click=${() => this.saveRules()}>Save rules</button>` : nothing}
        </div>`,
      { id: 'routing' },
    );
  }

  private async saveRules(): Promise<void> {
    try {
      const got = await config.rules((this.rules ?? []).map((r) => ({ id: r.id, pattern: r.pattern, backend: r.backend, enabled: r.enabled })));
      this.rules = got.rules.map((r) => ({ ...r }));
      this.say('rules saved');
      await this.load(true);
    } catch (err) {
      this.say(describeError(err).message);
    }
  }

  // ---- users ----------------------------------------------------------------------------
  private renderUsers(d: ConfigOverview): TemplateResult {
    return panel(
      'Users and keys',
      html`<p class="panel-sub bk-lead">${d.users_active ? html`<b>Keys are required:</b> a request to the OpenAI routes from another machine needs one of these keys${d.engine?.embedded ? html`, also for ${d.engine.model}` : nothing}. Requests from this machine itself pass.` : html`No key yet: the OpenAI routes are open on the network this server listens on. The first key you add makes them require one${d.engine?.embedded ? html`, also for ${d.engine.model}: give every client its key first` : nothing}.`} The ledger keeps the key's label per request.</p>
        ${this.newKey
          ? html`<div class="bk-newkey" role="alert">
              <p><b>${this.newKey.label}</b>'s key, shown this once. It is stored as its sha256 only.</p>
              <div class="bk-newkey-row"><code class="mono">${this.newKey.key}</code>${copyButton(this.newKey.key)}</div>
              <button class="btn btn-ghost btn-sm" @click=${() => (this.newKey = null)}>I saved it</button>
            </div>`
          : nothing}
        ${d.users.length
          ? html`<div class="table-scroll"><table class="table bk-users"><thead><tr><th>label</th><th>key</th><th>created</th><th>last seen</th><th>on</th><th></th></tr></thead><tbody>
              ${d.users.map((u) => this.renderUser(u))}
            </tbody></table></div>`
          : nothing}
        <form class="bk-adduser" @submit=${(e: Event) => this.addUser(e)}>
          <label class="field"><span>New user</span><input id="bk-user-label" required pattern="[A-Za-z0-9][A-Za-z0-9 ._@-]{0,63}" placeholder="laptop" /></label>
          <button class="btn btn-primary btn-sm" type="submit">${icon('plus')} Make a key</button>
        </form>`,
      { id: 'users' },
    );
  }

  private renderUser(u: ProxyUser): TemplateResult {
    const env = u.source === 'env';
    return html`<tr data-user=${u.label}>
      <td>${u.label}</td>
      <td class="mono">${env ? html`<span class="badge badge-lock">TANDEM_API_KEYS</span>` : `…${u.key_hint ?? ''}`}</td>
      <td>${u.created_at ? ago(new Date(u.created_at).toISOString()) : '—'}</td>
      <td>${u.last_seen_at ? ago(new Date(u.last_seen_at).toISOString()) : 'never'}</td>
      <td><label class="switch"><input type="checkbox" aria-label=${`${u.label} enabled`} .checked=${u.enabled} ?disabled=${env} @change=${(e: Event) => this.toggleUser(u, (e.target as HTMLInputElement).checked)} /></label></td>
      <td>${env || u.id == null ? nothing : html`<button class="btn btn-ghost btn-icon btn-sm" aria-label=${`delete ${u.label}`} @click=${() => this.removeUser(u)}>${icon('trash')}</button>`}</td>
    </tr>`;
  }

  private async addUser(e: Event): Promise<void> {
    e.preventDefault();
    const input = this.querySelector<HTMLInputElement>('#bk-user-label');
    const label = input?.value.trim() ?? '';
    if (!label) return;
    try {
      const r = await config.addUser(label);
      this.newKey = { label: r.user.label, key: r.key };
      if (input) input.value = '';
      await this.load(true);
    } catch (err) {
      this.say(describeError(err).message);
    }
  }

  private async toggleUser(u: ProxyUser, enabled: boolean): Promise<void> {
    if (u.id == null) return;
    try {
      await config.setUser(u.id, enabled);
      await this.load(true);
    } catch (err) {
      this.say(describeError(err).message);
    }
  }

  private async removeUser(u: ProxyUser): Promise<void> {
    if (u.id == null || !confirm(`Delete ${u.label}'s key? Clients that use it get a 401 from now on.`)) return;
    try {
      await config.removeUser(u.id);
      await this.load(true);
    } catch (err) {
      this.say(describeError(err).message);
    }
  }
}

export { exact as _exact };
