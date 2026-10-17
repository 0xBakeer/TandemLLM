// <qse-chat-settings>: the settings page in the thread's column, with a sub-navigation on the
// left (tabs on a phone). Tools (source, on or off, the policy: ask, always, never), skills and
// packages (upload a SKILL.md zip, remove your own), MCP servers (status, tools, add stdio or HTTP
// with a Test button, restart, remove), Open WebUI tools (import a Tools file, its valves), prompts
// and presets (Open WebUI style), web search (the provider, a SearXNG instance, Test; VIS-157), and
// this chat's sandbox (state, limits, network, reset).

import { html, nothing, type TemplateResult } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import { bytes, duration } from '../lib/format';
import { LightElement } from '../ui/base';
import { icon } from '../ui/bits';
import type { ChatBackend } from './backend';
import type { Capabilities, Conversation, McpServer, McpServerDraft, PackageInfo, Preset, Prompt, SafeSearch, TimeRange, ToolInfo, ToolPolicy, UserToolkit, WebSearchProvider, WebSearchTest, WebSearchView } from './types';
import { draftErrors, draftOf, draftPatch, hostOf, LANGUAGES, MAX_RESULTS, overrideOf, overridePatch, PROVIDERS, providerWord, SAFE_SEARCH, TIME_RANGES, timeWord, type WebSearchDraft } from './websearch';
import { EFFORTS, effortWord, valveValues } from './types';
import { modeDesc, modeOf, modeWord, PERMISSION_MODES } from './permission';
import { NOTIFY_WORDS, notifyPrefs, notifyState, setNotifyPrefs, type NotifyKind } from './notify';
import { AGENTS_HINT, agentsAsk } from './tasks';

/** a stored secret's field (an Open WebUI valve, the SearXNG auth value): masked, never prefilled */
const SECRET_INPUT = 'password';

export type SettingsSection = 'tools' | 'packages' | 'mcp' | 'search' | 'prompts' | 'sandbox' | 'device';
export const SECTIONS: { id: SettingsSection; label: string }[] = [
  { id: 'tools', label: 'Tools' },
  { id: 'packages', label: 'Skills and packages' },
  { id: 'mcp', label: 'MCP servers' },
  { id: 'search', label: 'Web search' },
  { id: 'prompts', label: 'Prompts and presets' },
  { id: 'sandbox', label: 'Sandbox' },
  { id: 'device', label: 'This device' },
];

@customElement('qse-chat-settings')
export class QseChatSettings extends LightElement {
  @property({ attribute: false }) backend: ChatBackend | null = null;
  @property({ attribute: false }) caps: Capabilities | null = null;
  @property({ attribute: false }) conv: Conversation | null = null;
  @property({ type: String }) section: SettingsSection = 'tools';
  @state() private tools: ToolInfo[] = [];
  @state() private pkgs: PackageInfo[] = [];
  @state() private kits: UserToolkit[] = [];
  /** typed valve values per toolkit, sent on Save */
  @state() private valveDraft: Record<string, Record<string, string>> = {};
  @state() private busy: string | null = null;
  @state() private mcp: McpServer[] = [];
  @state() private prompts: Prompt[] = [];
  @state() private presets: Preset[] = [];
  @state() private filter = '';
  @state() private openServer: string | null = null;
  @state() private adding = false;
  @state() private draft: McpServerDraft = { name: '', transport: 'stdio', command: '', args: [], env: {}, url: '', headers: {} };
  @state() private test: { state: 'idle' | 'running' | 'done'; ok?: boolean; message?: string; tools?: { name: string; description: string }[] } = { state: 'idle' };
  @state() private editPrompt: Partial<Prompt> | null = null;
  @state() private editPreset: Partial<Preset> | null = null;
  @state() private confirmReset = false;
  /** VIS-219: this chat's Agents switch waits for its confirmation */
  @state() private confirmAgents = false;
  @state() private notice: string | null = null;
  /** web search: the stored settings (null: an older server without the routes), the form */
  @state() private ws: WebSearchView | null = null;
  @state() private wsDraft: WebSearchDraft | null = null;
  @state() private wsTest: { state: 'idle' | 'running' | 'done'; r?: WebSearchTest; error?: string } = { state: 'idle' };
  @state() private wsSaving = false;

  updated(changed: Map<string, unknown>): void {
    if (changed.has('backend') && this.backend) void this.load();
  }

  private async load(): Promise<void> {
    const b = this.backend!;
    const [tools, pkgs, mcp, prompts, presets, kits, ws] = await Promise.all([b.tools(), b.packages(), b.mcpServers(), b.prompts(), b.presets(), b.userToolkits().catch(() => []), b.webSearch().catch(() => null)]);
    this.ws = ws;
    this.wsDraft = ws ? draftOf(ws.settings) : null;
    this.kits = kits;
    this.tools = tools;
    this.pkgs = pkgs;
    this.mcp = mcp;
    this.prompts = prompts;
    this.presets = presets;
  }

  private fire(name: string, detail: unknown = {}): void {
    this.dispatchEvent(new CustomEvent(name, { detail, bubbles: true, composed: true }));
  }

  private say(text: string): void {
    this.notice = text;
    setTimeout(() => (this.notice = null), 3000);
  }

  render() {
    return html`<div class="ch-settings">
      <header class="ch-settings-head">
        <button type="button" class="btn btn-ghost btn-sm" @click=${() => this.fire('ch-settings-back')}>${icon('left')} Back to the chat</button>
        <h2>Settings</h2>
        ${this.notice ? html`<span class="ch-notice" role="status">${this.notice}</span>` : nothing}
      </header>
      <div class="ch-settings-body">
        <nav class="ch-settings-nav" role="tablist" aria-label="settings sections">
          ${SECTIONS.map((s) => html`<a class="ch-settings-tab ${this.section === s.id ? 'is-on' : ''}" role="tab" aria-selected=${this.section === s.id} href=${`#/chat/settings/${s.id}`} @click=${(e: Event) => (e.preventDefault(), this.fire('ch-settings-nav', { section: s.id }))}>${s.label}</a>`)}
        </nav>
        <section class="ch-settings-main" role="tabpanel">${this.renderSection()}</section>
      </div>
    </div>`;
  }

  private renderSection(): TemplateResult {
    switch (this.section) {
      case 'tools':
        return this.renderTools();
      case 'packages':
        return this.renderPackages();
      case 'mcp':
        return this.renderMcp();
      case 'search':
        return this.renderSearch();
      case 'prompts':
        return this.renderPrompts();
      case 'sandbox':
        return this.renderSandbox();
      case 'device':
        return this.renderDevice();
    }
  }

  // ---- tools ----------------------------------------------------------------------------------

  private renderTools(): TemplateResult {
    const q = this.filter.toLowerCase();
    const list = this.tools.filter((t) => !q || t.name.includes(q) || t.description.toLowerCase().includes(q));
    return html`<h3 class="ch-settings-h">Tools</h3>
      <p class="ch-settings-lead">What the model may call, and when it has to ask. These are the defaults for new chats; a chat's own choices sit in its composer.</p>
      ${this.renderMode()}
      ${this.renderAgents()}
      <label class="field field-search ch-settings-search"><span class="sr-only">Search tools</span>${icon('search')}<input type="search" placeholder="Search tools" .value=${this.filter} @input=${(e: Event) => (this.filter = (e.target as HTMLInputElement).value)} /></label>
      <ul class="ch-tool-list">
        ${list.map(
          (t) => html`<li class="ch-tool ${t.enabled ? '' : 'is-off'}" data-tool=${t.id}>
            <label class="switch"><input type="checkbox" .checked=${t.enabled} @change=${(e: Event) => this.setTool(t.id, { enabled: (e.target as HTMLInputElement).checked })} /><span class="sr-only">${t.name} on</span></label>
            <div class="ch-tool-main"><code>${t.name}</code><span class="ch-tool-desc">${t.description}</span><span class="ch-tool-src">${sourceWord(t)}</span></div>
            <div class="seg ch-policy-seg" role="radiogroup" aria-label=${`policy for ${t.name}`}>
              ${(['ask', 'always', 'never'] as ToolPolicy[]).map((p) => html`<button type="button" class="seg-btn ${t.policy === p ? 'is-on' : ''}" role="radio" aria-checked=${t.policy === p} ?disabled=${!t.enabled} @click=${() => this.setTool(t.id, { policy: p })}>${p}</button>`)}
            </div>
          </li>`,
        )}
      </ul>
      <p class="ch-settings-note">ask: a prompt in the chat each time. always: runs without asking. never: the model sees the tool and the call is refused with a note.</p>
      ${this.renderKits()}`;
  }

  /** a file picker behind a button; `run` gets the file */
  private pick(accept: string, label: string, run: (f: File) => Promise<void>): TemplateResult {
    return html`<label class="btn ch-file-btn ${this.busy === label ? 'is-busy' : ''}">${this.busy === label ? html`<span class="pulse"></span>` : icon('plus')} ${label}<input type="file" class="sr-only" accept=${accept} ?disabled=${!!this.busy} @change=${async (e: Event) => {
      const input = e.target as HTMLInputElement;
      const f = input.files?.[0];
      input.value = '';
      if (!f) return;
      this.busy = label;
      try {
        await run(f);
      } catch (err) {
        this.say((err as Error).message);
      } finally {
        this.busy = null;
      }
    }} /></label>`;
  }

  // ---- Open WebUI tools (D9: Tools files only; filters, pipes and actions are not imported) ---

  private renderKits(): TemplateResult {
    return html`<h4 class="ch-settings-h4">Open WebUI tools</h4>
      <p class="ch-settings-lead">Import a Tools file from Open WebUI's Workspace (the .py, or the JSON export). Its methods become tools that run in the chat's sandbox; its valves are settings here.</p>
      <ul class="ch-mcp-list">${this.kits.map((k) => this.renderKit(k))}</ul>
      ${this.pick('.py,.json,text/x-python,application/json', 'Import a tools file', async (f) => {
        const r = await this.backend!.importOwui(f);
        this.kits = [...this.kits, ...r.toolkits];
        this.tools = await this.backend!.tools();
        this.say(`Imported ${r.toolkits.map((k) => k.title).join(', ')}: ${r.tools.length} tool${r.tools.length === 1 ? '' : 's'}.`);
      })}`;
  }

  private renderKit(k: UserToolkit): TemplateResult {
    const draft = this.valveDraft[k.id] ?? {};
    return html`<li class="ch-mcp ch-kit" data-kit=${k.id}>
      <div class="ch-mcp-row">
        <label class="switch"><input type="checkbox" .checked=${k.enabled} @change=${async (e: Event) => {
          const r = await this.backend!.patchUserToolkit(k.id, { enabled: (e.target as HTMLInputElement).checked });
          this.kits = this.kits.map((x) => (x.id === k.id ? r : x));
        }} /><span class="sr-only">${k.title} on</span></label>
        <span class="ch-mcp-name">${k.title}</span><span class="tag">Open WebUI</span>${k.version ? html`<span class="num muted small">v${k.version}</span>` : nothing}
        <span class="grow"></span>
        <span class="muted small">${k.tools.length} tool${k.tools.length === 1 ? '' : 's'}</span>
        <button type="button" class="btn btn-ghost btn-sm ch-danger" aria-label=${`Remove ${k.title}`} @click=${async () => {
          if (!confirm(`Remove the tools "${k.title}"?`)) return;
          await this.backend!.removeUserToolkit(k.id);
          this.kits = this.kits.filter((x) => x.id !== k.id);
          this.tools = this.tools.filter((t) => !k.tools.includes(t.id));
        }}>${icon('trash')}</button>
      </div>
      ${k.description ? html`<p class="muted small">${k.description}</p>` : nothing}
      ${k.requirements.length ? html`<p class="small">installs in the sandbox: ${k.requirements.map((r) => html`<code>${r}</code> `)}</p>` : nothing}
      ${k.valves.length
        ? html`<form class="ch-valves" @submit=${async (e: Event) => {
            e.preventDefault();
            const r = await this.backend!.patchUserToolkit(k.id, { valves: valveValues(k, draft) });
            this.kits = this.kits.map((x) => (x.id === k.id ? r : x));
            this.valveDraft = { ...this.valveDraft, [k.id]: {} };
            this.say(`Saved the valves of ${k.title}.`);
          }}>
            ${k.valves.map((v) => html`<label class="field"><span>${v.name}${v.description ? html` <span class="muted">${v.description}</span>` : nothing}</span><input type=${v.secret ? 'password' : 'text'} autocomplete="off" placeholder=${v.secret ? (v.set ? 'stored; type to replace' : 'not set') : String(v.default ?? '')} .value=${draft[v.name] ?? (v.secret ? '' : String(v.value ?? ''))} @input=${(e: Event) => (this.valveDraft = { ...this.valveDraft, [k.id]: { ...draft, [v.name]: (e.target as HTMLInputElement).value } })} /></label>`)}
            <div class="ch-add-row"><span class="grow"></span><button type="submit" class="btn btn-sm btn-primary" ?disabled=${!Object.keys(draft).length}>Save valves</button></div>
          </form>`
        : nothing}
    </li>`;
  }

  /** the permission mode for new chats (chat/permission.ts); a chat changes its own in the composer */
  private renderMode(): TemplateResult {
    const cur = modeOf(this.caps?.defaults);
    return html`<h4 class="ch-settings-h4">Before a tool runs</h4>
      <div class="ch-mode-list" role="radiogroup" aria-label="permission mode for new chats">
        ${PERMISSION_MODES.map((m) => html`<label class="ch-mode ${cur === m ? 'is-on' : ''} ${m === 'yolo' ? 'is-yolo' : ''}" data-mode=${m}>
          <input type="radio" name="permission-mode" .checked=${cur === m} ?disabled=${!this.caps} @change=${() => this.fire('ch-defaults-patch', { patch: { permission_mode: m } })} />
          <span class="ch-mode-name">${modeWord(m)}${m === 'auto' ? html` <span class="muted small">default</span>` : nothing}</span>
          <span class="ch-mode-desc">${modeDesc(m)}</span>
        </label>`)}
      </div>`;
  }

  /** VIS-219: Agents mode for new chats, and for the open chat (which asks first once it has
   *  messages: the next message reads the chat once more) */
  private renderAgents(): TemplateResult {
    const c = this.conv;
    const def = !!this.caps?.defaults.agents;
    const on = !!c?.settings.agents;
    const flip = () => {
      if (c?.messages.length && !this.confirmAgents) {
        this.confirmAgents = true;
        return;
      }
      this.confirmAgents = false;
      this.fire('ch-settings-patch', { patch: { agents: !on } });
    };
    return html`<h4 class="ch-settings-h4">Agents</h4>
      <p class="ch-settings-lead">${AGENTS_HINT}: the main agent talks with you and gives each job that writes files to a worker with a fresh context. Only the worker's report and chatd's check of it enter the chat's context.</p>
      <div class="ch-agents-rows">
        <label class="switch ch-agents-row" data-agents="default"><input type="checkbox" .checked=${def} ?disabled=${!this.caps} @change=${(e: Event) => this.fire('ch-defaults-patch', { patch: { agents: (e.target as HTMLInputElement).checked } })} /><span>Agents on for new chats</span></label>
        ${c
          ? html`<label class="switch ch-agents-row" data-agents="chat"><input type="checkbox" .checked=${on} @change=${(e: Event) => ((e.target as HTMLInputElement).checked = on, flip())} /><span>Agents in "${c.title}"</span></label>
            ${this.confirmAgents ? html`<span class="ch-confirm" data-agents-ask><span class="small">${agentsAsk(c.context.used, !!c.run)}</span><button type="button" class="btn btn-ghost btn-sm" @click=${() => (this.confirmAgents = false)}>Cancel</button><button type="button" class="btn btn-sm btn-primary" @click=${flip}>${on ? 'Turn off' : 'Turn on'}</button></span>` : nothing}`
          : nothing}
      </div>`;
  }

  private async setTool(id: string, patch: { enabled?: boolean; policy?: ToolPolicy }): Promise<void> {
    const t = await this.backend!.setTool(id, patch);
    this.tools = this.tools.map((x) => (x.id === id ? t : x));
  }

  // ---- packages -------------------------------------------------------------------------------

  private renderPackages(): TemplateResult {
    return html`<h3 class="ch-settings-h">Skills and packages</h3>
      <p class="ch-settings-lead">The engine's own packages: a skill shapes the answer, a package brings tools. On here means on for new chats.</p>
      <ul class="ch-pkg-list">
        ${this.pkgs.map(
          (p) => html`<li class="ch-pkg ${p.enabled ? '' : 'is-off'}" data-pkg=${p.id}>
            <label class="switch"><input type="checkbox" .checked=${p.enabled} @change=${async (e: Event) => {
              const r = await this.backend!.setPackage(p.id, (e.target as HTMLInputElement).checked);
              this.pkgs = this.pkgs.map((x) => (x.id === p.id ? r : x));
            }} /><span class="sr-only">${p.name} on</span></label>
            <div class="ch-pkg-main"><span class="ch-pkg-name">${p.name}</span><span class="tag">${p.user ? 'yours' : p.kind}</span>${p.version ? html`<span class="num muted small">v${p.version}</span>` : nothing}<span class="ch-pkg-desc">${p.description}</span>${p.tools.length ? html`<span class="ch-pkg-tools">${p.tools.map((t) => html`<code>${t}</code>`)}</span>` : nothing}</div>
            ${p.user
              ? html`<button type="button" class="btn btn-ghost btn-sm ch-danger" aria-label=${`Remove ${p.name}`} @click=${async () => {
                  if (!confirm(`Remove your skill "${p.name}" from the engine's library?`)) return;
                  await this.backend!.removeSkill(p.id);
                  this.pkgs = this.pkgs.filter((x) => x.id !== p.id);
                }}>${icon('trash')}</button>`
              : nothing}
          </li>`,
        )}
      </ul>
      ${this.pick('.zip,application/zip', 'Upload a skill', async (f) => {
        const p = await this.backend!.uploadSkill(f);
        this.pkgs = [...this.pkgs.filter((x) => x.id !== p.id), p];
        this.say(`Installed the skill ${p.name}.`);
      })}
      <p class="ch-settings-note">A skill is a folder with a SKILL.md (frontmatter <code>name</code> and <code>description</code>), as a .zip. It goes into the engine's library, so the harness and the model's library search find it.</p>`;
  }

  // ---- mcp ------------------------------------------------------------------------------------

  private renderMcp(): TemplateResult {
    return html`<h3 class="ch-settings-h">MCP servers</h3>
      <p class="ch-settings-lead">Servers the model can call through the Model Context Protocol. stdio servers run on the box; HTTP servers are reached by URL.</p>
      <ul class="ch-mcp-list">
        ${this.mcp.map((s) => this.renderServer(s))}
      </ul>
      ${this.adding ? this.renderAdd() : html`<button type="button" class="btn" @click=${() => ((this.adding = true), (this.test = { state: 'idle' }))}>${icon('plus')} Add server</button>`}`;
  }

  private renderServer(s: McpServer): TemplateResult {
    const open = this.openServer === s.id;
    return html`<li class="ch-mcp is-${s.status}" data-mcp=${s.id}>
      <div class="ch-mcp-row">
        <span class="ch-mark ch-mark-mcp-${s.status}"><span class="ch-mark-dot" aria-hidden="true"></span>${s.status}</span>
        <span class="ch-mcp-name">${s.name}</span>
        <span class="tag">${s.transport}</span>
        <code class="ch-mcp-where">${s.transport === 'stdio' ? [s.command, ...(s.args ?? [])].join(' ') : s.url}</code>
        <span class="grow"></span>
        <button type="button" class="linkish small" aria-expanded=${open} @click=${() => (this.openServer = open ? null : s.id)}>${s.tools.length} tool${s.tools.length === 1 ? '' : 's'}</button>
        <button type="button" class="btn btn-ghost btn-sm" title="Restart" aria-label=${`Restart ${s.name}`} @click=${async () => {
          const r = await this.backend!.restartMcpServer(s.id);
          this.mcp = this.mcp.map((x) => (x.id === s.id ? r : x));
          setTimeout(() => void this.backend!.mcpServers().then((l) => (this.mcp = l)), 1_500);
        }}>${icon('refresh')}</button>
        <button type="button" class="btn btn-ghost btn-sm ch-danger" title="Remove" aria-label=${`Remove ${s.name}`} @click=${async () => {
          if (!confirm(`Remove the server "${s.name}"?`)) return;
          await this.backend!.removeMcpServer(s.id);
          this.mcp = this.mcp.filter((x) => x.id !== s.id);
        }}>${icon('trash')}</button>
      </div>
      ${s.error ? html`<p class="ch-mcp-error err-ink small">${s.error}</p>` : nothing}
      ${open ? html`<ul class="ch-mcp-tools">${s.tools.length ? s.tools.map((t) => html`<li><code>${t.name}</code><span class="muted small">${t.description}</span></li>`) : html`<li class="muted small">no tools listed</li>`}</ul>` : nothing}
    </li>`;
  }

  private renderAdd(): TemplateResult {
    const d = this.draft;
    const set = (patch: Partial<McpServerDraft>) => ((this.draft = { ...this.draft, ...patch }), (this.test = { state: 'idle' }));
    return html`<form class="ch-add-mcp" @submit=${async (e: Event) => {
      e.preventDefault();
      const s = await this.backend!.addMcpServer(this.draft);
      this.mcp = [...this.mcp, s];
      this.adding = false;
      this.draft = { name: '', transport: 'stdio', command: '', args: [], env: {}, url: '', headers: {} };
      this.say(`Added ${s.name}; it is starting.`);
      setTimeout(() => void this.backend!.mcpServers().then((l) => (this.mcp = l)), 2_500);
    }}>
      <h4>Add a server</h4>
      <label class="field"><span>name</span><input type="text" required .value=${d.name} @input=${(e: Event) => set({ name: (e.target as HTMLInputElement).value })} /></label>
      <div class="seg" role="radiogroup" aria-label="transport">${(['stdio', 'http'] as const).map((t) => html`<button type="button" class="seg-btn ${d.transport === t ? 'is-on' : ''}" role="radio" aria-checked=${d.transport === t} @click=${() => set({ transport: t })}>${t}</button>`)}</div>
      ${d.transport === 'stdio'
        ? html`<label class="field"><span>command</span><input type="text" required placeholder="uvx" .value=${d.command ?? ''} @input=${(e: Event) => set({ command: (e.target as HTMLInputElement).value })} /></label>
            <label class="field"><span>arguments</span><input type="text" placeholder="my-mcp --flag" .value=${(d.args ?? []).join(' ')} @input=${(e: Event) => set({ args: (e.target as HTMLInputElement).value.split(/\s+/).filter(Boolean) })} /></label>
            <label class="field field-block"><span>environment, one KEY=value per line</span><textarea rows="2" .value=${Object.entries(d.env ?? {}).map(([k, v]) => `${k}=${v}`).join('\n')} @input=${(e: Event) => set({ env: parseKv((e.target as HTMLTextAreaElement).value) })}></textarea></label>`
        : html`<label class="field"><span>URL</span><input type="url" required placeholder="http://127.0.0.1:8787/mcp" .value=${d.url ?? ''} @input=${(e: Event) => set({ url: (e.target as HTMLInputElement).value })} /></label>
            <label class="field field-block"><span>headers, one Name: value per line</span><textarea rows="2" .value=${Object.entries(d.headers ?? {}).map(([k, v]) => `${k}: ${v}`).join('\n')} @input=${(e: Event) => set({ headers: parseKv((e.target as HTMLTextAreaElement).value, ':') })}></textarea></label>`}
      <div class="ch-add-row">
        <button type="button" class="btn" ?disabled=${this.test.state === 'running'} @click=${async () => {
          this.test = { state: 'running' };
          const r = await this.backend!.testMcpServer(this.draft);
          this.test = { state: 'done', ...r };
        }}>${this.test.state === 'running' ? html`<span class="pulse"></span> Testing` : 'Test'}</button>
        ${this.test.state === 'done' ? html`<span class="ch-test-result ${this.test.ok ? 'is-ok' : 'is-bad'}" role="status">${this.test.ok ? icon('check') : icon('warn')} ${this.test.message}</span>` : nothing}
        <span class="grow"></span>
        <button type="button" class="btn btn-ghost" @click=${() => (this.adding = false)}>Cancel</button>
        <button type="submit" class="btn btn-primary" ?disabled=${!d.name.trim()}>Add server</button>
      </div>
      ${this.test.state === 'done' && this.test.ok && this.test.tools?.length ? html`<ul class="ch-mcp-tools">${this.test.tools.map((t) => html`<li><code>${t.name}</code><span class="muted small">${t.description}</span></li>`)}</ul>` : nothing}
    </form>`;
  }

  // ---- web search (VIS-157) -------------------------------------------------------------------

  private renderSearch(): TemplateResult {
    const head = html`<h3 class="ch-settings-h">Web search</h3>
      <p class="ch-settings-lead">The model's <code>web_search</code> tool. It runs on this box and reaches only the provider's host. DuckDuckGo needs no setup; SearXNG is an instance you name by its URL.</p>`;
    const v = this.ws;
    const d = this.wsDraft;
    if (!v || !d) return html`${head}<p class="ch-settings-note">This chat server has no web search routes; update chatd.</p>`;
    const errs = draftErrors(d, v.settings.searxng_auth_set);
    const patch = draftPatch(v.settings, d);
    const dirty = Object.keys(patch).length > 0;
    const set = (p: Partial<WebSearchDraft>) => ((this.wsDraft = { ...d, ...p }), (this.wsTest = { state: 'idle' }));
    const err = (k: string) => (errs[k] ? html`<span class="ch-ws-err err-ink small" role="alert">${errs[k]}</span>` : nothing);
    const num = (e: Event) => Math.round(Number((e.target as HTMLInputElement).value));
    const sel = (e: Event) => (e.target as HTMLSelectElement).value;
    const langs = LANGUAGES.some((l) => l.id === d.language) ? LANGUAGES : [...LANGUAGES, { id: d.language, label: d.language }];
    return html`${head}
      <form class="ch-ws" data-ws-form @submit=${(e: Event) => (e.preventDefault(), void this.saveSearch())}>
        <div class="ch-ws-grid">
          <label class="field field-block ch-ws-field"><span>provider</span><select data-ws="provider" @change=${(e: Event) => set({ provider: sel(e) as WebSearchProvider })}>${PROVIDERS.map((p) => html`<option value=${p} ?selected=${d.provider === p}>${providerWord(p)}</option>`)}</select></label>
          <label class="field field-block ch-ws-field"><span>results</span><input type="number" inputmode="numeric" min="1" max=${MAX_RESULTS} data-ws="max_results" .value=${String(d.max_results)} @input=${(e: Event) => set({ max_results: num(e) })} />${err('max_results')}</label>
          <label class="field field-block ch-ws-field"><span>safe search</span><select data-ws="safe_search" @change=${(e: Event) => set({ safe_search: sel(e) as SafeSearch })}>${SAFE_SEARCH.map((x) => html`<option value=${x} ?selected=${d.safe_search === x}>${x}</option>`)}</select></label>
          <label class="field field-block ch-ws-field"><span>language and region</span><select data-ws="language" @change=${(e: Event) => set({ language: sel(e) })}>${langs.map((l) => html`<option value=${l.id} ?selected=${d.language === l.id}>${l.label}</option>`)}</select>${err('language')}</label>
          <label class="field field-block ch-ws-field"><span>time range</span><select data-ws="time_range" @change=${(e: Event) => set({ time_range: sel(e) as TimeRange })}>${TIME_RANGES.map((t) => html`<option value=${t} ?selected=${d.time_range === t}>${timeWord(t)}</option>`)}</select></label>
          <label class="field field-block ch-ws-field"><span>timeout, seconds</span><input type="number" inputmode="numeric" min="2" max="30" data-ws="timeout_s" .value=${String(d.timeout_s)} @input=${(e: Event) => set({ timeout_s: num(e) })} />${err('timeout_s')}</label>
        </div>
        ${d.provider === 'searxng'
          ? html`<fieldset class="ch-fieldset ch-ws-sx" data-ws-searxng><legend>SearXNG instance</legend>
              <label class="field field-block ch-ws-field"><span>URL</span><input type="url" inputmode="url" autocomplete="off" spellcheck="false" placeholder="https://search.example.org" data-ws="searxng_url" .value=${d.searxng_url} @input=${(e: Event) => set({ searxng_url: (e.target as HTMLInputElement).value.trim() })} />${err('searxng_url')}</label>
              <div class="ch-ws-grid">
                <label class="field field-block ch-ws-field"><span>auth header <span class="muted">optional</span></span><input type="text" autocomplete="off" spellcheck="false" placeholder="Authorization or X-API-Key" data-ws="searxng_auth_header" .value=${d.searxng_auth_header} @input=${(e: Event) => set({ searxng_auth_header: (e.target as HTMLInputElement).value.trim() })} />${err('searxng_auth_header')}</label>
                <label class="field field-block ch-ws-field"><span>value</span><input type=${SECRET_INPUT} autocomplete="off" data-ws="searxng_auth_value" placeholder=${v.settings.searxng_auth_set && !d.clear_auth ? 'stored; type to replace' : 'not set'} .value=${d.searxng_auth_value} @input=${(e: Event) => set({ searxng_auth_value: (e.target as HTMLInputElement).value, clear_auth: false })} />${err('searxng_auth_value')}</label>
              </div>
              ${v.settings.searxng_auth_set ? html`<label class="ch-check"><input type="checkbox" data-ws="clear_auth" .checked=${d.clear_auth} @change=${(e: Event) => set({ clear_auth: (e.target as HTMLInputElement).checked, searxng_auth_value: '' })} /><span>remove the stored value</span></label>` : nothing}
              <p class="ch-settings-note">The instance needs <code>json</code> in <code>search.formats</code> (settings.yml). The value is stored encrypted and never shown again.</p>
            </fieldset>`
          : nothing}
        <div class="ch-add-row">
          <button type="button" class="btn" data-ws-test ?disabled=${this.wsTest.state === 'running' || d.provider === 'off' || !!Object.keys(errs).length} @click=${() => void this.testSearch()}>${this.wsTest.state === 'running' ? html`<span class="pulse"></span> Testing` : 'Test'}</button>
          ${this.wsTest.state === 'done' ? html`<span class="ch-test-result ${this.wsTest.r?.ok ? 'is-ok' : 'is-bad'}" role="status" data-ws-result>${this.wsTest.r?.ok ? icon('check') : icon('warn')} ${this.wsTest.r?.message ?? this.wsTest.error}</span>` : nothing}
          <span class="grow"></span>
          <button type="button" class="btn btn-ghost" ?disabled=${!dirty} @click=${() => ((this.wsDraft = draftOf(v.settings)), (this.wsTest = { state: 'idle' }))}>Revert</button>
          <button type="submit" class="btn btn-primary" data-ws-save ?disabled=${!dirty || this.wsSaving || !!Object.keys(errs).length}>Save</button>
        </div>
        ${this.wsTest.state === 'done' && this.wsTest.r?.results.length ? html`<ul class="ch-ws-sample">${this.wsTest.r.results.map((r) => html`<li><a href=${r.url} target="_blank" rel="noopener noreferrer">${r.title}</a> <span class="muted small">${hostOf(r.url)}</span></li>`)}</ul>` : nothing}
      </form>
      ${this.renderSearchChat()}`;
  }

  /** a chat's own provider, over the default above */
  private renderSearchChat(): TemplateResult {
    const c = this.conv;
    const def = providerWord(this.ws?.settings.provider);
    if (!c) return html`<p class="ch-settings-note">Open a chat to give it its own provider; new chats use the one above (${def}).</p>`;
    const cur = overrideOf(c.settings);
    return html`<h4 class="ch-settings-h4">This chat: "${c.title}"</h4>
      <div class="ch-add-row">
        <label class="field ch-ws-field"><span class="sr-only">web search in this chat</span><select data-ws="chat" @change=${(e: Event) => this.fire('ch-settings-patch', { patch: { web_search: overridePatch(c.settings.web_search, (e.target as HTMLSelectElement).value as '' | WebSearchProvider) } })}>
          <option value="" ?selected=${cur === ''}>the default (${def})</option>
          ${PROVIDERS.map((p) => html`<option value=${p} ?selected=${cur === p}>${providerWord(p)}</option>`)}
        </select></label>
        <span class="small muted">off removes web_search from this chat's tools</span>
      </div>`;
  }

  private async testSearch(): Promise<void> {
    const v = this.ws;
    const d = this.wsDraft;
    if (!v || !d) return;
    this.wsTest = { state: 'running' };
    try {
      this.wsTest = { state: 'done', r: await this.backend!.testWebSearch({ ...draftPatch(v.settings, d), provider: d.provider }) };
    } catch (e) {
      this.wsTest = { state: 'done', error: (e as Error).message };
    }
  }

  private async saveSearch(): Promise<void> {
    const v = this.ws;
    const d = this.wsDraft;
    if (!v || !d) return;
    this.wsSaving = true;
    try {
      const next = await this.backend!.saveWebSearch(draftPatch(v.settings, d));
      this.ws = next;
      this.wsDraft = draftOf(next.settings);
      this.fire('ch-web-search-saved', { provider: next.settings.provider });
      this.say(`Saved: web search with ${providerWord(next.settings.provider)}.`);
    } catch (e) {
      this.say((e as Error).message);
    } finally {
      this.wsSaving = false;
    }
  }

  // ---- prompts and presets --------------------------------------------------------------------

  private renderPrompts(): TemplateResult {
    return html`<h3 class="ch-settings-h">Prompts and presets</h3>
      <p class="ch-settings-lead">A prompt is a saved system text with a slash command. A preset bundles a model, an effort, tools and a prompt, for one click in the composer.</p>
      <div class="ch-two">
        <section>
          <h4>Prompts</h4>
          <ul class="ch-prompt-list">
            ${this.prompts.map((p) => html`<li class="ch-prompt" data-prompt=${p.id}><div class="ch-prompt-row"><code>/${p.command}</code><span class="ch-prompt-title">${p.title}</span><span class="grow"></span><button type="button" class="btn btn-ghost btn-sm" aria-label=${`Edit ${p.title}`} @click=${() => (this.editPrompt = { ...p })}>${icon('edit')}</button><button type="button" class="btn btn-ghost btn-sm ch-danger" aria-label=${`Delete ${p.title}`} @click=${async () => {
              if (!confirm(`Delete the prompt "${p.title}"?`)) return;
              await this.backend!.deletePrompt(p.id);
              this.prompts = this.prompts.filter((x) => x.id !== p.id);
            }}>${icon('trash')}</button></div><p class="ch-prompt-body">${p.body}</p></li>`)}
          </ul>
          ${this.editPrompt
            ? html`<form class="ch-edit-form" @submit=${async (e: Event) => {
                e.preventDefault();
                const p = this.editPrompt!;
                const saved = await this.backend!.savePrompt({ id: p.id, title: p.title ?? '', command: (p.command ?? '').replace(/^\//, ''), body: p.body ?? '' });
                this.prompts = p.id ? this.prompts.map((x) => (x.id === saved.id ? saved : x)) : [...this.prompts, saved];
                this.editPrompt = null;
                this.say(`Saved /${saved.command}.`);
              }}>
                <label class="field"><span>title</span><input type="text" required .value=${this.editPrompt.title ?? ''} @input=${(e: Event) => (this.editPrompt = { ...this.editPrompt, title: (e.target as HTMLInputElement).value })} /></label>
                <label class="field"><span>command</span><input type="text" required pattern="[a-z0-9-]+" placeholder="review" .value=${this.editPrompt.command ?? ''} @input=${(e: Event) => (this.editPrompt = { ...this.editPrompt, command: (e.target as HTMLInputElement).value })} /></label>
                <label class="field field-block"><span>text</span><textarea rows="4" required .value=${this.editPrompt.body ?? ''} @input=${(e: Event) => (this.editPrompt = { ...this.editPrompt, body: (e.target as HTMLTextAreaElement).value })}></textarea></label>
                <div class="ch-add-row"><span class="grow"></span><button type="button" class="btn btn-ghost" @click=${() => (this.editPrompt = null)}>Cancel</button><button type="submit" class="btn btn-primary">Save prompt</button></div>
              </form>`
            : html`<button type="button" class="btn" @click=${() => (this.editPrompt = { title: '', command: '', body: '' })}>${icon('plus')} New prompt</button>`}
        </section>
        <section>
          <h4>Presets</h4>
          <ul class="ch-preset-list">
            ${this.presets.map((p) => html`<li class="ch-preset" data-preset=${p.id}><div class="ch-prompt-row"><span class="ch-prompt-title">${p.name}</span><span class="grow"></span><button type="button" class="btn btn-sm" ?disabled=${!this.conv} title="Apply to this chat" @click=${() => this.fire('ch-apply-preset', { preset: p })}>Use</button><button type="button" class="btn btn-ghost btn-sm" aria-label=${`Edit ${p.name}`} @click=${() => (this.editPreset = { ...p })}>${icon('edit')}</button><button type="button" class="btn btn-ghost btn-sm ch-danger" aria-label=${`Delete ${p.name}`} @click=${async () => {
              if (!confirm(`Delete the preset "${p.name}"?`)) return;
              await this.backend!.deletePreset(p.id);
              this.presets = this.presets.filter((x) => x.id !== p.id);
            }}>${icon('trash')}</button></div><p class="ch-preset-sub">${this.caps?.models.find((m) => m.id === p.model)?.name ?? p.model}, ${p.thinking ? effortWord(p.effort) : 'thinking off'}, ${p.tools.length} tool${p.tools.length === 1 ? '' : 's'}${p.prompt_id ? html`, /${this.prompts.find((x) => x.id === p.prompt_id)?.command ?? 'prompt'}` : ''}</p></li>`)}
          </ul>
          ${this.editPreset
            ? html`<form class="ch-edit-form" @submit=${async (e: Event) => {
                e.preventDefault();
                const p = this.editPreset!;
                const saved = await this.backend!.savePreset({ id: p.id, name: p.name ?? '', model: p.model ?? this.caps?.models[0]?.id ?? 'TandemLLM', profile: null, effort: p.effort ?? 'high', thinking: p.thinking ?? true, tools: p.tools ?? [], prompt_id: p.prompt_id ?? null });
                this.presets = p.id ? this.presets.map((x) => (x.id === saved.id ? saved : x)) : [...this.presets, saved];
                this.editPreset = null;
                this.say(`Saved ${saved.name}.`);
              }}>
                <label class="field"><span>name</span><input type="text" required .value=${this.editPreset.name ?? ''} @input=${(e: Event) => (this.editPreset = { ...this.editPreset, name: (e.target as HTMLInputElement).value })} /></label>
                <label class="field"><span>model</span><select @change=${(e: Event) => (this.editPreset = { ...this.editPreset, model: (e.target as HTMLSelectElement).value })}>${(this.caps?.models ?? []).map((m) => html`<option value=${m.id} ?selected=${(this.editPreset?.model ?? this.caps?.models[0]?.id) === m.id}>${m.name}</option>`)}</select></label>
                <label class="field"><span>effort</span><select .value=${this.editPreset.effort ?? 'high'} @change=${(e: Event) => (this.editPreset = { ...this.editPreset, effort: (e.target as HTMLSelectElement).value as Preset['effort'] })}>${EFFORTS.map((ef) => html`<option value=${ef} ?selected=${(this.editPreset?.effort ?? 'high') === ef}>${effortWord(ef)}</option>`)}</select></label>
                <label class="switch"><input type="checkbox" .checked=${this.editPreset.thinking ?? true} @change=${(e: Event) => (this.editPreset = { ...this.editPreset, thinking: (e.target as HTMLInputElement).checked })} /><span>thinking</span></label>
                <label class="field"><span>prompt</span><select @change=${(e: Event) => (this.editPreset = { ...this.editPreset, prompt_id: (e.target as HTMLSelectElement).value || null })}><option value="">none</option>${this.prompts.map((pr) => html`<option value=${pr.id} ?selected=${this.editPreset?.prompt_id === pr.id}>/${pr.command}</option>`)}</select></label>
                <fieldset class="ch-fieldset"><legend>tools</legend>${this.tools.filter((t) => t.enabled).map((t) => html`<label class="ch-check"><input type="checkbox" .checked=${(this.editPreset?.tools ?? []).includes(t.id)} @change=${(e: Event) => {
                  const on = (e.target as HTMLInputElement).checked;
                  const cur = this.editPreset?.tools ?? [];
                  this.editPreset = { ...this.editPreset, tools: on ? [...cur, t.id] : cur.filter((x) => x !== t.id) };
                }} /><code>${t.name}</code></label>`)}</fieldset>
                <div class="ch-add-row"><span class="grow"></span><button type="button" class="btn btn-ghost" @click=${() => (this.editPreset = null)}>Cancel</button><button type="submit" class="btn btn-primary">Save preset</button></div>
              </form>`
            : html`<button type="button" class="btn" @click=${() => (this.editPreset = { name: '', effort: 'high', thinking: true, tools: [] })}>${icon('plus')} New preset</button>`}
        </section>
      </div>`;
  }

  // ---- this device: notifications and the installed app (VIS-109) ------------------------------

  private renderDevice(): TemplateResult {
    const st = notifyState();
    const prefs = notifyPrefs();
    const installed = matchMedia('(display-mode: standalone)').matches;
    return html`<h3 class="ch-settings-h">This device</h3>
      <p class="ch-settings-lead">Notifications stay on this box: the dashboard shows them itself, no push service carries them. They name the chat and what happened, never what was written.</p>
      <div class="ch-add-row" data-notify=${st}>
        <span class="ch-mark ch-mark-${st === 'granted' ? 'running' : 'idle'}"><span class="ch-mark-dot" aria-hidden="true"></span>${st === 'granted' ? 'notifications on' : st === 'denied' ? 'blocked in the browser settings' : st === 'unsupported' ? 'not available here' : 'not asked yet'}</span>
        ${st === 'default' ? html`<button type="button" class="btn btn-primary btn-sm" @click=${async () => (await Notification.requestPermission(), this.requestUpdate())}>Allow notifications</button>` : nothing}
      </div>
      ${st === 'unsupported' ? html`<p class="ch-settings-note">This browser offers no notifications on this page: on an iPhone, add the dashboard to the home screen and open it from there; elsewhere, open it over https.</p>` : nothing}
      <fieldset class="ch-fieldset"><legend>notify when a chat</legend>${(Object.keys(NOTIFY_WORDS) as NotifyKind[]).map((k) => html`<label class="ch-check"><input type="checkbox" .checked=${prefs[k]} ?disabled=${st !== 'granted'} @change=${(e: Event) => (setNotifyPrefs({ ...notifyPrefs(), [k]: (e.target as HTMLInputElement).checked }), this.requestUpdate())} /><span>${NOTIFY_WORDS[k]}</span></label>`)}</fieldset>
      <p class="ch-settings-note">${installed ? 'Running as the installed app.' : 'Add the dashboard to the home screen (Share, then Add to Home Screen; or the install button in the address bar) for a full-screen app with its own badge.'} The badge counts the chats that wait for you or have an unread answer.</p>`;
  }

  // ---- sandbox --------------------------------------------------------------------------------

  private renderSandbox(): TemplateResult {
    const c = this.conv;
    if (!c) return html`<h3 class="ch-settings-h">Sandbox</h3><p class="ch-settings-lead">Open a chat to see its sandbox. Every chat gets one: a container on the box where the tools run.</p>`;
    const s = c.sandbox;
    return html`<h3 class="ch-settings-h">Sandbox for "${c.title}"</h3>
      <p class="ch-settings-lead">A container on the box where this chat's tools run. Files persist with the chat; a reset wipes them.</p>
      <div class="ch-sb-state"><span class="ch-mark ch-mark-sb-${s.state}"><span class="ch-mark-dot" aria-hidden="true"></span>${s.state}</span>${s.error ? html`<span class="err-ink small">${s.error}</span>` : nothing}</div>
      <dl class="kv kv-grid ch-sb-grid">
        <div><dt>image</dt><dd>${s.image}</dd></div>
        <div><dt>working directory</dt><dd><code>${s.cwd}</code></dd></div>
        <div><dt>CPUs</dt><dd class="num">${s.cpus}</dd></div>
        <div><dt>uptime</dt><dd class="num">${s.state === 'running' ? duration(s.uptime_s) : '—'}</dd></div>
        <div><dt>network</dt><dd>${s.network === 'on' ? 'on: the internet, never the LAN' : s.network === 'allowlist' ? html`allow-list: ${s.allowlist.map((h) => html`<code>${h}</code> `)}` : 'off'}</dd></div>
        ${s.stopped_reason && s.state === 'stopped' ? html`<div><dt>stopped because</dt><dd>${s.stopped_reason === 'memory_guard' ? 'the memory guard: the engine needed the RAM' : s.stopped_reason === 'idle' ? 'it sat idle' : 'you stopped it'}</dd></div>` : nothing}
        ${s.ports?.length ? html`<div><dt>ports</dt><dd>${s.ports.map((p) => html`<code>${p.port}</code> ${p.label ?? ''} `)}</dd></div>` : nothing}
      </dl>
      <div class="ch-add-row">
        <span class="small muted">network for new runs</span>
        <div class="seg" role="radiogroup" aria-label="sandbox network">
          ${(['off', 'on'] as const).map((n) => html`<button type="button" class="seg-btn ${c.settings.sandbox_network === n ? 'is-on' : ''}" role="radio" aria-checked=${c.settings.sandbox_network === n} @click=${() => this.fire('ch-settings-patch', { patch: { sandbox_network: n } })}>${n}</button>`)}
          <button type="button" class="seg-btn" role="radio" aria-checked="false" disabled title="an allow-list of hosts comes later">allow-list</button>
        </div>
        <span class="ch-net-later">on means the internet without the LAN; the allow-list comes later</span>
      </div>
      <div class="meter-block"><div class="meter-head"><span>memory</span><span class="num">${s.memory_bytes != null ? bytes(s.memory_bytes, 0) : s.state === 'running' ? 'not reported' : '0 B'} of ${bytes(s.memory_limit_bytes, 0)}</span></div><div class="meter"><span class="meter-fill" style="width:${(((s.memory_bytes ?? 0) / s.memory_limit_bytes) * 100).toFixed(1)}%"></span></div></div>
      <div class="meter-block"><div class="meter-head"><span>disk</span><span class="num">${bytes(s.disk_bytes, 0)} of ${bytes(s.disk_limit_bytes, 0)}</span></div><div class="meter"><span class="meter-fill" style="width:${((s.disk_bytes / s.disk_limit_bytes) * 100).toFixed(1)}%"></span></div></div>
      <div class="ch-add-row">
        <button type="button" class="btn" @click=${() => this.fire('ch-open-pane', { pane: { kind: 'terminal', ref: 'shell', title: 'Terminal' } })}>${icon('terminal')} Open terminal</button>
        <button type="button" class="btn" @click=${() => this.fire('ch-open-pane', { pane: { kind: 'files', ref: '', title: 'Files' } })}>${icon('folder')} Files</button>
        <span class="grow"></span>
        ${this.confirmReset
          ? html`<span class="ch-confirm"><span class="small">Wipe the files and start a fresh container?</span><button type="button" class="btn btn-ghost btn-sm" @click=${() => (this.confirmReset = false)}>Keep it</button><button type="button" class="btn btn-sm ch-danger-solid" @click=${() => (this.fire('ch-reset-sandbox'), (this.confirmReset = false))}>Reset sandbox</button></span>`
          : html`<button type="button" class="btn ch-danger" @click=${() => (this.confirmReset = true)}>${icon('trash')} Reset sandbox</button>`}
      </div>`;
  }
}

function sourceWord(t: ToolInfo): string {
  return t.source.kind === 'builtin' ? 'built in' : t.source.kind === 'package' ? `package ${t.source.id}` : t.source.kind === 'user' ? 'Open WebUI tool' : `MCP ${t.source.id}`;
}


function parseKv(text: string, sep = '='): Record<string, string> {
  const out: Record<string, string> = {};
  for (const line of text.split('\n')) {
    const i = line.indexOf(sep);
    if (i > 0) out[line.slice(0, i).trim()] = line.slice(i + 1).trim();
  }
  return out;
}
