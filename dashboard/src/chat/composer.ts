// <qse-chat-composer>: the text area and one row of controls. Left: attach, the model and pack
// pill, the effort pill, the tools pill, each with a small popover. Right: Send, or Stop while a
// run is on; Enter then queues the message. A slash at the start opens the command list from the
// server's capabilities and the saved prompts. Files drop anywhere on the thread. The Agents chip
// (VIS-219) switches a main agent with workers on or off; in a chat with messages it asks first.

import { html, nothing, type TemplateResult } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import { stripEdge, type Edge } from '../lib/phone';
import { LightElement } from '../ui/base';
import { icon } from '../ui/bits';
import type { Attachment, Capabilities, ChatSettings, EffortChoice, ModelInfo, Prompt, SlashCommand, PackageInfo, ToolInfo } from './types';
import { effortsFor, effortWord, nearestEffort, promptArgs, localId, toolLabel } from './types';
import { modeDesc, modeOf, modeWord, PERMISSION_MODES } from './permission';
import { chatProvider, providerWord } from './websearch';
import { attachmentWords } from './context-words';
import { AGENTS_HINT, agentsAsk } from './tasks';

const MAX_PREVIEW_BYTES = 400_000;

@customElement('qse-chat-composer')
export class QseChatComposer extends LightElement {
  @property({ attribute: false }) settings: ChatSettings | null = null;
  @property({ attribute: false }) caps: Capabilities | null = null;
  @property({ attribute: false }) tools: ToolInfo[] = [];
  @property({ attribute: false }) packages: PackageInfo[] = [];
  /** MCP resources the user can attach */
  @property({ attribute: false }) resources: { server: string; serverName: string; uri: string; name: string }[] = [];
  @property({ attribute: false }) prompts: Prompt[] = [];
  @property({ type: Boolean }) running = false;
  @property({ type: Boolean }) disabled = false;
  @property({ type: String }) draft = '';
  /** uploads a file as soon as it is added, so its chip shows the tokens and whether it goes
   *  as a file (SRV-487, VIS-169); without it the view uploads at the send */
  @property({ attribute: false }) upload: ((f: File) => Promise<Attachment>) | null = null;
  /** the chat's context now, what a switch of Agents re-reads; null: no messages, nothing to ask */
  @property({ attribute: false }) rereadTokens: number | null = null;
  /** what the last switch of Agents did (the view's, from chatd's answer) */
  @property({ attribute: false }) agentsNote: string | null = null;
  @state() private text = '';
  @state() private files: Attachment[] = [];
  /** the File behind each attachment chip, for the upload before a send (CONTRACT 8, request 9) */
  private blobs = new Map<string, File>();
  @state() private pop: 'none' | 'model' | 'effort' | 'tools' | 'mode' | 'agents' = 'none';
  @state() private slash: { items: SlashCommand[]; index: number } | null = null;
  @state() private dragging = false;
  @state() private enterSends = !(typeof matchMedia === 'function' && matchMedia('(pointer: coarse)').matches);
  /** the phone (the shell's one breakpoint): shorter placeholders, the text beside the buttons */
  @state() private phone = typeof matchMedia === 'function' && matchMedia('(max-width: 768px)').matches;
  private mqPhone = typeof matchMedia === 'function' ? matchMedia('(max-width: 768px)') : null;
  private onMedia = () => (this.phone = !!this.mqPhone?.matches);
  /** the phone's pill strip: which edge has more behind it, so that edge fades */
  @state() private edge: Edge = 'none';

  connectedCallback(): void {
    super.connectedCallback();
    this.text = this.draft;
    document.addEventListener('click', this.onDocClick);
    this.mqPhone?.addEventListener('change', this.onMedia);
  }
  disconnectedCallback(): void {
    document.removeEventListener('click', this.onDocClick);
    this.mqPhone?.removeEventListener('change', this.onMedia);
    super.disconnectedCallback();
  }
  updated(changed: Map<string, unknown>): void {
    if (changed.has('draft') && this.draft !== this.text && !this.text) this.text = this.draft;
    // after every render (a width change on a phone is a rotation, which re-renders through the
    // media query), the strip's edges follow its scroll position
    if (!changed.has('edge')) this.measureStrip();
  }
  private measureStrip(): void {
    const strip = this.querySelector<HTMLElement>('.ch-pills');
    if (!strip) return;
    const e = stripEdge(strip.scrollLeft, strip.scrollWidth, strip.clientWidth);
    if (e !== this.edge) this.edge = e;
  }

  private onDocClick = (e: Event) => {
    if (this.pop !== 'none' && !(e.target as HTMLElement).closest('.ch-pop, .ch-pill')) this.pop = 'none';
  };

  private fire(name: string, detail: unknown = {}): void {
    this.dispatchEvent(new CustomEvent(name, { detail, bubbles: true, composed: true }));
  }

  /** the view reads the draft to keep it per chat */
  get value(): string {
    return this.text;
  }
  setText(t: string): void {
    this.text = t;
    this.updateComplete.then(() => this.area()?.focus());
  }
  focusArea(): void {
    this.area()?.focus();
  }
  /** the File of an attachment this composer made (the view uploads it before the send) */
  fileOf(id: string): File | undefined {
    return this.blobs.get(id);
  }
  /** a send that could not go out: its text and files come back to the composer */
  restore(text: string, files: Attachment[]): void {
    this.text = text;
    this.files = files;
    this.fire('ch-draft', { text });
  }
  private area(): HTMLTextAreaElement | null {
    return this.querySelector<HTMLTextAreaElement>('#ch-input');
  }

  private patch(p: Partial<ChatSettings>): void {
    this.fire('ch-settings-patch', { patch: p });
  }

  // ---- send, stop, queue ----------------------------------------------------------------------

  private send(): void {
    const text = this.text.trim();
    if (!text && !this.files.length) return;
    if (text.startsWith('/') && !text.includes('\n')) {
      const [cmd, ...rest] = text.slice(1).split(/\s+/);
      const known = this.caps?.commands.find((c) => c.command === cmd);
      if (known) {
        if (known.mcp_prompt) {
          const mp = known.mcp_prompt;
          this.fire('ch-mcp-prompt', { serverId: mp.server_id, name: mp.name, args: promptArgs(rest.join(' '), (mp.arguments ?? []).map((a) => a.name)) });
          this.slash = null;
          return;
        }
        if (known.prompt_id) {
          const p = this.prompts.find((x) => x.id === known.prompt_id);
          if (p) {
            this.text = `${p.body}\n\n${rest.join(' ')}`.trim();
            this.slash = null;
            return;
          }
        }
        this.fire('ch-command', { command: cmd, args: rest.join(' ') });
        this.text = '';
        this.slash = null;
        this.fire('ch-draft', { text: '' });
        return;
      }
    }
    this.fire('ch-send', { text, attachments: this.files });
    this.text = '';
    this.files = [];
    this.slash = null;
    this.fire('ch-draft', { text: '' });
  }

  private onInput(e: Event): void {
    this.text = (e.target as HTMLTextAreaElement).value;
    this.fire('ch-draft', { text: this.text });
    const m = /^\/([\w-]*)$/.exec(this.text);
    if (m && this.caps) {
      const q = m[1].toLowerCase();
      const items = this.caps.commands.filter((c) => c.command.startsWith(q)).slice(0, 8);
      this.slash = items.length ? { items, index: 0 } : null;
    } else this.slash = null;
    const ta = e.target as HTMLTextAreaElement;
    ta.style.height = 'auto';
    ta.style.height = Math.min(ta.scrollHeight, 320) + 'px';
  }

  private onKey(e: KeyboardEvent): void {
    if (this.slash) {
      if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        e.preventDefault();
        const n = this.slash.items.length;
        this.slash = { ...this.slash, index: (this.slash.index + (e.key === 'ArrowDown' ? 1 : n - 1)) % n };
        return;
      }
      if (e.key === 'Tab' || (e.key === 'Enter' && !e.shiftKey)) {
        e.preventDefault();
        this.pickSlash(this.slash.items[this.slash.index]);
        return;
      }
      if (e.key === 'Escape') {
        this.slash = null;
        return;
      }
    }
    if (e.key === 'Escape' && this.running) {
      this.fire('ch-stop');
      return;
    }
    if (e.key !== 'Enter') return;
    if (e.metaKey || e.ctrlKey || (this.enterSends && !e.shiftKey)) {
      e.preventDefault();
      this.send();
    }
  }

  private pickSlash(c: SlashCommand): void {
    this.slash = null;
    if (c.prompt_id) {
      const p = this.prompts.find((x) => x.id === c.prompt_id);
      this.text = p ? p.body + '\n\n' : `/${c.command} `;
    } else if (c.command === 'clear' || c.command === 'compact' || c.command === 'mcp') {
      this.fire('ch-command', { command: c.command, args: '' });
      this.text = '';
    } else if (c.command === 'model' || c.command === 'effort') {
      this.text = '';
      this.pop = c.command;
    } else if (c.command === 'pane') this.text = '/pane ';
    else this.text = `/${c.command} `;
    this.fire('ch-draft', { text: this.text });
    this.area()?.focus();
  }

  // ---- files ----------------------------------------------------------------------------------

  private async addFiles(list: FileList | File[]): Promise<void> {
    const out: Attachment[] = [];
    for (const f of Array.from(list)) {
      const kind: Attachment['kind'] = f.type.startsWith('image/') ? 'image' : f.type === 'application/pdf' ? 'pdf' : f.type.startsWith('text/') || /\.(md|txt|py|ts|js|json|yaml|yml|toml|csv)$/i.test(f.name) ? 'text' : 'file';
      let preview: string | null = null;
      if (kind === 'image' && f.size <= MAX_PREVIEW_BYTES) preview = await readDataUrl(f);
      const att: Attachment = { id: localId('att'), name: f.name, kind, bytes: f.size, mime: f.type || 'application/octet-stream', preview };
      this.blobs.set(att.id, f);
      out.push(att);
    }
    this.files = [...this.files, ...out];
    for (const att of out) void this.uploadEarly(att);
  }

  /** the server's attachment in place of the local chip; a failure leaves the File for the send */
  private async uploadEarly(att: Attachment): Promise<void> {
    const f = this.blobs.get(att.id);
    if (!this.upload || !f) return;
    try {
      const up = await this.upload(f);
      if (!this.files.some((x) => x.id === att.id)) return;
      this.files = this.files.map((x) => (x.id === att.id ? { ...up, preview: att.preview ?? null } : x));
      this.blobs.delete(att.id);
    } catch {
      /* the send uploads it again and says so if it fails */
    }
  }

  /** an attachment the server holds already (an MCP resource) */
  attach(a: Attachment): void {
    this.files = [...this.files, a];
  }

  /** the view forwards drops and pastes from the whole thread */
  acceptFiles(list: FileList | File[]): void {
    void this.addFiles(list);
  }

  private onPaste(e: ClipboardEvent): void {
    const files = Array.from(e.clipboardData?.files ?? []);
    if (files.length) {
      e.preventDefault();
      void this.addFiles(files);
    }
  }

  // ---- render ---------------------------------------------------------------------------------

  render() {
    const s = this.settings;
    const caps = this.caps;
    const model = caps?.models.find((m) => m.id === s?.model) ?? caps?.models[0];
    const toolsOn = s ? s.tools.length : 0;
    const mcpOn = s ? s.mcp.length : 0;
    return html`<form class="ch-composer ${this.dragging ? 'is-drag' : ''} ${this.running ? 'is-running' : ''}" @submit=${(e: Event) => (e.preventDefault(), this.send())}
      @dragover=${(e: DragEvent) => (e.preventDefault(), (this.dragging = true))} @dragleave=${() => (this.dragging = false)} @drop=${(e: DragEvent) => {
        e.preventDefault();
        this.dragging = false;
        if (e.dataTransfer?.files.length) void this.addFiles(e.dataTransfer.files);
      }}>
      ${this.slash ? this.renderSlash() : nothing}
      ${this.agentsNote ? html`<p class="ch-agents-note small" role="status">${this.agentsNote}<button type="button" class="btn btn-ghost btn-sm" aria-label="Dismiss" @click=${() => this.fire('ch-agents-note-close')}>${icon('x')}</button></p>` : nothing}
      ${this.files.length ? html`<div class="ch-attachments ch-attachments-draft">${this.files.map((a) => html`<span class="ch-chip">${a.kind === 'image' && a.preview ? html`<img src=${a.preview} alt="" />` : icon('file')}<span class="ch-chip-name">${a.name}</span><span class="ch-chip-size num">${attachmentWords(a).size}</span>${a.inline === false ? html`<span class="ch-chip-warn warn-ink">as a file</span>` : nothing}<button type="button" class="ch-chip-x" aria-label=${`Remove ${a.name}`} @click=${() => (this.files = this.files.filter((x) => x.id !== a.id))}>${icon('x')}</button></span>`)}</div>${this.files.map((a) => attachmentWords(a).warn).filter(Boolean).map((w) => html`<p class="ch-attach-warn warn-ink small" role="status">${w}</p>`)}` : nothing}
      <label class="sr-only" for="ch-input">Message</label>
      <textarea id="ch-input" rows="1" placeholder=${this.running ? (this.phone ? 'Steer the run' : 'Steer: goes in after the current step') : this.phone ? 'Message TandemLLM' : 'Message TandemLLM, or / for a command'} .value=${this.text} ?disabled=${this.disabled} @input=${this.onInput} @keydown=${this.onKey} @paste=${this.onPaste}></textarea>
      <div class="ch-composer-row">
        <label class="btn btn-ghost btn-icon ch-attach-btn" title="Attach files" aria-label="Attach files"><input type="file" multiple class="sr-only" @change=${(e: Event) => {
          const inp = e.target as HTMLInputElement;
          if (inp.files?.length) void this.addFiles(inp.files);
          inp.value = '';
        }} />${icon('plus')}</label>
        <span class="ch-pills" data-edge=${this.edge} @scroll=${() => this.measureStrip()}>
          <span class="ch-pillwrap">
            <button type="button" class="ch-pill" aria-haspopup="menu" aria-expanded=${this.pop === 'model'} @click=${() => (this.pop = this.pop === 'model' ? 'none' : 'model')} title="model"><span class="ch-pill-main">${model?.name ?? '…'}</span></button>
            ${this.pop === 'model' && caps ? this.renderModelPop(caps, s) : nothing}
          </span>
          <span class="ch-pillwrap">
            <button type="button" class="ch-pill ch-pill-effort" aria-haspopup="menu" aria-expanded=${this.pop === 'effort'} @click=${() => (this.pop = this.pop === 'effort' ? 'none' : 'effort')} title="thinking effort">${effortPill(s?.thinking === false ? 'no thinking' : effortWord(nearestEffort(s?.effort ?? 'high', model?.backend !== 'engine' ? model?.capabilities?.effort_levels : null)))}</button>
            ${this.pop === 'effort' && s ? this.renderEffortPop(s, model) : nothing}
          </span>
          <span class="ch-pillwrap">
            <button type="button" class="ch-pill" aria-haspopup="menu" aria-expanded=${this.pop === 'tools'} @click=${() => (this.pop = this.pop === 'tools' ? 'none' : 'tools')} title="tools, skills and MCP servers"><span class="ch-pill-main">${toolsOn} tool${toolsOn === 1 ? '' : 's'}</span>${mcpOn ? html`<span class="ch-pill-sub">${mcpOn} MCP</span>` : nothing}</button>
            ${this.pop === 'tools' && s ? this.renderToolsPop(s) : nothing}
          </span>
          <span class="ch-pillwrap">
            <button type="button" class="ch-pill ch-pill-mode ${modeOf(s) === 'yolo' ? 'is-yolo' : ''}" aria-haspopup="menu" aria-expanded=${this.pop === 'mode'} ?disabled=${!s} title="permissions: who allows a tool call" @click=${() => (this.pop = this.pop === 'mode' ? 'none' : 'mode')}><span class="ch-pill-main">${modeWord(modeOf(s))}</span></button>
            ${this.pop === 'mode' && s ? this.renderModePop(s) : nothing}
          </span>
          <span class="ch-pillwrap">
            <button type="button" class="ch-pill ch-pill-agents ${s?.agents ? 'is-on' : ''}" aria-pressed=${!!s?.agents} ?disabled=${!s || model?.capabilities?.tools === false} title=${`Agents: ${AGENTS_HINT.toLowerCase()}`} @click=${() => s && this.toggleAgents(s)}><span class="ch-pill-main">agents</span>${s?.agents ? html`<span class="ch-pill-sub">workers</span>` : nothing}</button>
            ${this.pop === 'agents' && s ? this.renderAgentsPop(s) : nothing}
          </span>
          <button type="button" class="ch-pill ${s?.plan_mode ? 'is-on' : ''}" aria-pressed=${!!s?.plan_mode} ?disabled=${!s || model?.capabilities?.tools === false} title="Plan mode: the model reads and writes a plan, changes nothing until you approve it" @click=${() => s && this.patch({ plan_mode: !s.plan_mode })}><span class="ch-pill-main">plan</span>${s?.plan_mode ? html`<span class="ch-pill-sub">reads only</span>` : nothing}</button>
        </span>
        <span class="grow"></span>
        <label class="switch ch-enter" title="Enter sends; Shift+Enter is a new line. Off: Enter is a new line, Cmd/Ctrl+Enter sends."><input type="checkbox" .checked=${this.enterSends} @change=${(e: Event) => (this.enterSends = (e.target as HTMLInputElement).checked)} /><span>Enter sends</span></label>
        ${this.running
          ? html`<span class="ch-send-group"><button type="submit" class="btn btn-sm ch-queue-btn" ?disabled=${!this.text.trim()} title="Queue: goes in after the current step">${icon('send')} <span class="ch-btn-word">Queue</span></button><button type="button" class="btn ch-stop" aria-label="Stop" title="Stop the run" @click=${() => this.fire('ch-stop')}>${icon('stop')} <span class="ch-btn-word">Stop</span></button></span>`
          : html`<button type="submit" class="btn btn-primary ch-send" aria-label="Send" ?disabled=${this.disabled || (!this.text.trim() && !this.files.length)}>${icon('send')} <span class="ch-btn-word">Send</span></button>`}
      </div>
    </form>`;
  }

  private renderSlash(): TemplateResult {
    const s = this.slash!;
    return html`<div class="ch-slash" role="listbox" aria-label="commands">
      ${s.items.map((c, i) => html`<button type="button" role="option" aria-selected=${i === s.index} class="ch-slash-item ${i === s.index ? 'is-on' : ''}" @mousedown=${(e: Event) => e.preventDefault()} @click=${() => this.pickSlash(c)}><code>/${c.command}</code><span>${c.description}</span></button>`)}
    </div>`;
  }

  /** VIS-203: models only, one row each; the Chat tab runs under its own conductor profile */
  private renderModelPop(caps: Capabilities, s: ChatSettings | null): TemplateResult {
    const cur = caps.models.find((m) => m.id === s?.model) ?? caps.models[0];
    return html`<div class="ch-pop ch-pop-models" role="menu">
      ${caps.models.map((m) => html`<button type="button" role="menuitemradio" aria-checked=${m === cur} class="ch-pop-item ${m === cur ? 'is-on' : ''}" data-model=${m.id} @click=${() => (this.patch(modelPatch(m)), (this.pop = 'none'))}><span class="ch-pop-name">${m.name}</span><span class="ch-pop-desc">${modelDesc(m)}</span></button>`)}
    </div>`;
  }

  private renderEffortPop(s: ChatSettings, model?: ModelInfo): TemplateResult {
    const def = model?.backend !== 'engine' ? model?.capabilities?.effort_default : null;
    return html`<div class="ch-pop" role="menu">
      <div class="ch-pop-group"><div class="ch-pop-title">Thinking</div>
        <label class="switch ch-pop-switch"><input type="checkbox" .checked=${s.thinking} @change=${(e: Event) => this.patch({ thinking: (e.target as HTMLInputElement).checked })} /><span>${s.thinking ? 'on' : 'off'}</span></label>
      </div>
      <div class="ch-pop-group"><div class="ch-pop-title">Effort</div>
        ${effortsFor(this.caps, model?.id).map((e) => html`<button type="button" role="menuitemradio" aria-checked=${s.effort === e} class="ch-pop-item ${s.effort === e ? 'is-on' : ''}" ?disabled=${!s.thinking} @click=${() => (this.patch({ effort: e }), (this.pop = 'none'))}><span class="ch-pop-name">${effortWord(e)}</span><span class="ch-pop-desc">${effortDesc(e, def)}</span></button>`)}
      </div>
    </div>`;
  }

  private renderModePop(s: ChatSettings): TemplateResult {
    const cur = modeOf(s);
    return html`<div class="ch-pop" role="menu">
      <div class="ch-pop-group"><div class="ch-pop-title">Before a tool runs</div>
        ${PERMISSION_MODES.map((m) => html`<button type="button" role="menuitemradio" aria-checked=${cur === m} class="ch-pop-item ${cur === m ? 'is-on' : ''} ${m === 'yolo' ? 'is-yolo' : ''}" @click=${() => (this.patch({ permission_mode: m }), (this.pop = 'none'))}><span class="ch-pop-name">${modeWord(m)}</span><span class="ch-pop-desc">${modeDesc(m)}</span></button>`)}
      </div>
    </div>`;
  }

  /** VIS-219: a new chat switches at once; a chat with messages asks first, since the switch
   *  changes the prompt head and the next message reads the chat once more */
  private toggleAgents(s: ChatSettings): void {
    if (this.rereadTokens == null) {
      this.patch({ agents: !s.agents });
      return;
    }
    this.pop = this.pop === 'agents' ? 'none' : 'agents';
  }

  private renderAgentsPop(s: ChatSettings): TemplateResult {
    const on = !s.agents;
    return html`<div class="ch-pop ch-pop-agents" role="dialog" aria-label=${on ? 'Turn Agents on' : 'Turn Agents off'}>
      <div class="ch-pop-group"><div class="ch-pop-title">${on ? 'Agents on' : 'Agents off'}</div>
        <p class="ch-pop-desc">${on ? `${AGENTS_HINT}. Only their reports enter this chat's context.` : 'One agent again: it writes the files itself.'}</p>
        <p class="ch-pop-text" data-agents-ask>${agentsAsk(this.rereadTokens ?? 0, this.running)}</p>
        <div class="ch-pop-actions"><button type="button" class="btn btn-ghost btn-sm" @click=${() => (this.pop = 'none')}>Cancel</button><button type="button" class="btn btn-primary btn-sm" @click=${() => (this.patch({ agents: on }), (this.pop = 'none'))}>${on ? 'Turn on' : 'Turn off'}</button></div>
      </div>
    </div>`;
  }

  /** VIS-157: the provider this chat searches with, and the way to the settings */
  private renderSearchWord(s: ChatSettings): TemplateResult {
    const p = chatProvider(this.caps, s);
    return html`<span class="ch-pop-desc" data-search-provider=${p ?? ''}>${p ? html`with <strong>${providerWord(p)}</strong>${s.web_search?.provider ? ' in this chat' : ''}` : 'web search'} <button type="button" class="linkish" @click=${(e: Event) => (e.preventDefault(), this.fire('ch-open-settings', { section: 'search' }), (this.pop = 'none'))}>change</button></span>`;
  }

  private renderToolsPop(s: ChatSettings): TemplateResult {
    const builtin = this.tools.filter((t) => t.source.kind !== 'mcp' && t.enabled);
    const servers = [...new Set(this.tools.filter((t) => t.source.kind === 'mcp').map((t) => (t.source as { id: string }).id))];
    const toggle = (id: string) => this.patch({ tools: s.tools.includes(id) ? s.tools.filter((x) => x !== id) : [...s.tools, id] });
    const toggleMcp = (id: string) => this.patch({ mcp: s.mcp.includes(id) ? s.mcp.filter((x) => x !== id) : [...s.mcp, id] });
    return html`<div class="ch-pop ch-pop-wide" role="menu">
      <div class="ch-pop-group"><div class="ch-pop-title">Tools in this chat</div>
        ${builtin.map((t) => html`<label class="ch-pop-check" data-pop-tool=${t.id}><input type="checkbox" .checked=${s.tools.includes(t.id)} @change=${() => toggle(t.id)} /><code>${toolLabel(t.name)}</code>${t.id === 'web_search' ? this.renderSearchWord(s) : html`<span class="ch-pop-desc">${t.description}</span>`}<span class="ch-policy">${s.policies[t.id] ?? t.policy}</span></label>`)}
      </div>
      <div class="ch-pop-group"><div class="ch-pop-title">Checks</div>
        <label class="ch-pop-check" data-pop-verify><input type="checkbox" .checked=${s.verify !== false} @change=${() => this.patch({ verify: s.verify === false })} /><code>verify apps</code><span class="ch-pop-desc">a run that builds or changes a page or app loads it in a browser before it ends, and the model reads what the check found</span></label>
      </div>
      <div class="ch-pop-group"><div class="ch-pop-title">MCP servers</div>
        ${servers.map((id) => html`<label class="ch-pop-check"><input type="checkbox" .checked=${s.mcp.includes(id)} @change=${() => toggleMcp(id)} /><code>${id}</code><span class="ch-pop-desc">${this.tools.filter((t) => t.source.kind === 'mcp' && (t.source as { id: string }).id === id).length} tools</span></label>`)}
        ${this.resources.map((r) => html`<button type="button" class="ch-pop-item" @click=${() => (this.fire('ch-mcp-resource', { serverId: r.server, uri: r.uri }), (this.pop = 'none'))}><span class="ch-pop-name">${icon('file')} ${r.name}</span><span class="ch-pop-desc">attach, from ${r.serverName}</span></button>`)}
      </div>
      ${this.packages.length
        ? html`<div class="ch-pop-group"><div class="ch-pop-title">Skills</div>
            <div class="seg" role="radiogroup" aria-label="skill mode">${(['auto', 'manual', 'model'] as const).map((m) => html`<button type="button" class="seg-btn ${s.packages_mode === m ? 'is-on' : ''}" role="radio" aria-checked=${s.packages_mode === m} title=${m === 'auto' ? 'the harness picks, plus your pins' : m === 'manual' ? 'your pins only' : 'the model reads the skill index and picks'} @click=${() => this.patch({ packages_mode: m })}>${m}</button>`)}</div>
            ${this.packages.filter((p) => p.enabled).map((p) => html`<label class="ch-pop-check"><input type="checkbox" .checked=${s.packages.includes(p.id)} @change=${() => this.patch({ packages: s.packages.includes(p.id) ? s.packages.filter((x) => x !== p.id) : [...s.packages, p.id] })} /><code>${p.name}</code><span class="ch-pop-desc">pin: ${p.description}</span></label>`)}
          </div>`
        : nothing}
      <div class="ch-pop-foot"><button type="button" class="linkish" @click=${() => (this.fire('ch-open-settings', { section: 'tools' }), (this.pop = 'none'))}>All tools and policies</button><button type="button" class="linkish" @click=${() => (this.fire('ch-open-settings', { section: 'mcp' }), (this.pop = 'none'))}>MCP servers</button></div>
    </div>`;
  }
}

/** a pick sets the model and clears any profile: the engine maps the Chat tab to chat-app */
export function modelPatch(m: ModelInfo): Partial<ChatSettings> {
  return { model: m.id, profile: null };
}

/** "EXL3 4.0, 256k window": the parts a backend reports, none left empty */
export function modelDesc(m: ModelInfo): string {
  return [m.weights, m.max_len ? `${Math.round(m.max_len / 1024)}k window` : '', m.capabilities?.vision ? 'reads images' : ''].filter(Boolean).join(', ');
}

/** "auto effort" as the value and its word, the word in the quieter ink (VIS-185) */
function effortPill(word: string): TemplateResult {
  const m = / effort$/.exec(word);
  return m ? html`<span class="ch-pill-main">${word.slice(0, m.index)}</span><span class="ch-pill-sub">effort</span>` : html`<span class="ch-pill-main">${word}</span>`;
}

export function effortDesc(e: EffortChoice, providerDefault?: string | null): string {
  if (e === 'auto') return providerDefault ? `the provider's default (${providerDefault})` : 'the harness decides per turn';
  if (e === 'max') return "the provider's deepest thinking, slowest";
  return e === 'low' ? 'a short think, quick answers' : e === 'medium' ? 'a middle budget' : e === 'high' ? 'long thinking for code and plans' : 'the largest budget, slow';
}

function readDataUrl(f: File): Promise<string | null> {
  return new Promise((resolve) => {
    const r = new FileReader();
    r.onload = () => resolve(typeof r.result === 'string' ? r.result : null);
    r.onerror = () => resolve(null);
    r.readAsDataURL(f);
  });
}
