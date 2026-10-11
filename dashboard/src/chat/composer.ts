// <qse-chat-composer>: the text area and one row of controls. Left: attach, the model and pack
// pill, the effort pill, the tools pill, each with a small popover. Right: Send, or Stop while a
// run is on; Enter then queues the message. A slash at the start opens the command list from the
// server's capabilities and the saved prompts. Files drop anywhere on the thread.

import { html, nothing, type TemplateResult } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import { bytes } from '../lib/format';
import { LightElement } from '../ui/base';
import { icon } from '../ui/bits';
import type { Attachment, Capabilities, ChatSettings, EffortChoice, Prompt, SlashCommand, ToolInfo } from './types';
import { EFFORTS, effortWord, localId, toolLabel } from './types';

const MAX_PREVIEW_BYTES = 400_000;

@customElement('qse-chat-composer')
export class QseChatComposer extends LightElement {
  @property({ attribute: false }) settings: ChatSettings | null = null;
  @property({ attribute: false }) caps: Capabilities | null = null;
  @property({ attribute: false }) tools: ToolInfo[] = [];
  @property({ attribute: false }) prompts: Prompt[] = [];
  @property({ type: Boolean }) running = false;
  @property({ type: Boolean }) disabled = false;
  @property({ type: String }) draft = '';
  @state() private text = '';
  @state() private files: Attachment[] = [];
  @state() private pop: 'none' | 'model' | 'effort' | 'tools' = 'none';
  @state() private slash: { items: SlashCommand[]; index: number } | null = null;
  @state() private dragging = false;
  @state() private enterSends = !(typeof matchMedia === 'function' && matchMedia('(pointer: coarse)').matches);

  connectedCallback(): void {
    super.connectedCallback();
    this.text = this.draft;
    document.addEventListener('click', this.onDocClick);
  }
  disconnectedCallback(): void {
    document.removeEventListener('click', this.onDocClick);
    super.disconnectedCallback();
  }
  updated(changed: Map<string, unknown>): void {
    if (changed.has('draft') && this.draft !== this.text && !this.text) this.text = this.draft;
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
      out.push({ id: localId('att'), name: f.name, kind, bytes: f.size, mime: f.type || 'application/octet-stream', preview });
    }
    this.files = [...this.files, ...out];
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
    const profile = model?.profiles.find((p) => p.id === s?.profile);
    const toolsOn = s ? s.tools.length : 0;
    const mcpOn = s ? s.mcp.length : 0;
    return html`<form class="ch-composer ${this.dragging ? 'is-drag' : ''} ${this.running ? 'is-running' : ''}" @submit=${(e: Event) => (e.preventDefault(), this.send())}
      @dragover=${(e: DragEvent) => (e.preventDefault(), (this.dragging = true))} @dragleave=${() => (this.dragging = false)} @drop=${(e: DragEvent) => {
        e.preventDefault();
        this.dragging = false;
        if (e.dataTransfer?.files.length) void this.addFiles(e.dataTransfer.files);
      }}>
      ${this.slash ? this.renderSlash() : nothing}
      ${this.files.length ? html`<div class="ch-attachments ch-attachments-draft">${this.files.map((a) => html`<span class="ch-chip">${a.kind === 'image' && a.preview ? html`<img src=${a.preview} alt="" />` : icon('file')}<span class="ch-chip-name">${a.name}</span><span class="ch-chip-size num">${bytes(a.bytes, 0)}</span><button type="button" class="ch-chip-x" aria-label=${`Remove ${a.name}`} @click=${() => (this.files = this.files.filter((x) => x.id !== a.id))}>${icon('x')}</button></span>`)}</div>` : nothing}
      <label class="sr-only" for="ch-input">Message</label>
      <textarea id="ch-input" rows="1" placeholder=${this.running ? 'Steer: goes in after the current step' : 'Message TandemLLM, or / for a command'} .value=${this.text} ?disabled=${this.disabled} @input=${this.onInput} @keydown=${this.onKey} @paste=${this.onPaste}></textarea>
      <div class="ch-composer-row">
        <label class="btn btn-ghost btn-icon ch-attach-btn" title="Attach files" aria-label="Attach files"><input type="file" multiple class="sr-only" @change=${(e: Event) => {
          const inp = e.target as HTMLInputElement;
          if (inp.files?.length) void this.addFiles(inp.files);
          inp.value = '';
        }} />${icon('plus')}</label>
        <span class="ch-pills">
          <span class="ch-pillwrap">
            <button type="button" class="ch-pill" aria-haspopup="menu" aria-expanded=${this.pop === 'model'} @click=${() => (this.pop = this.pop === 'model' ? 'none' : 'model')} title="model and profile"><span class="ch-pill-main">${model?.name ?? '…'}</span>${profile ? html`<span class="ch-pill-sub">${profile.name}</span>` : nothing}</button>
            ${this.pop === 'model' && caps ? this.renderModelPop(caps, s) : nothing}
          </span>
          <span class="ch-pillwrap">
            <button type="button" class="ch-pill" aria-haspopup="menu" aria-expanded=${this.pop === 'effort'} @click=${() => (this.pop = this.pop === 'effort' ? 'none' : 'effort')} title="thinking effort"><span class="ch-pill-main">${s?.thinking === false ? 'no thinking' : effortWord(s?.effort ?? 'high')}</span></button>
            ${this.pop === 'effort' && s ? this.renderEffortPop(s) : nothing}
          </span>
          <span class="ch-pillwrap">
            <button type="button" class="ch-pill" aria-haspopup="menu" aria-expanded=${this.pop === 'tools'} @click=${() => (this.pop = this.pop === 'tools' ? 'none' : 'tools')} title="tools, skills and MCP servers"><span class="ch-pill-main">${toolsOn} tool${toolsOn === 1 ? '' : 's'}</span>${mcpOn ? html`<span class="ch-pill-sub">${mcpOn} MCP</span>` : nothing}</button>
            ${this.pop === 'tools' && s ? this.renderToolsPop(s) : nothing}
          </span>
        </span>
        <span class="grow"></span>
        <label class="switch ch-enter" title="Enter sends; Shift+Enter is a new line. Off: Enter is a new line, Cmd/Ctrl+Enter sends."><input type="checkbox" .checked=${this.enterSends} @change=${(e: Event) => (this.enterSends = (e.target as HTMLInputElement).checked)} /><span>Enter sends</span></label>
        ${this.running
          ? html`<span class="ch-send-group"><button type="submit" class="btn btn-sm ch-queue-btn" ?disabled=${!this.text.trim()} title="Queue: goes in after the current step">${icon('send')} Queue</button><button type="button" class="btn ch-stop" @click=${() => this.fire('ch-stop')}>${icon('stop')} Stop</button></span>`
          : html`<button type="submit" class="btn btn-primary ch-send" ?disabled=${this.disabled || (!this.text.trim() && !this.files.length)}>${icon('send')} Send</button>`}
      </div>
    </form>`;
  }

  private renderSlash(): TemplateResult {
    const s = this.slash!;
    return html`<div class="ch-slash" role="listbox" aria-label="commands">
      ${s.items.map((c, i) => html`<button type="button" role="option" aria-selected=${i === s.index} class="ch-slash-item ${i === s.index ? 'is-on' : ''}" @mousedown=${(e: Event) => e.preventDefault()} @click=${() => this.pickSlash(c)}><code>/${c.command}</code><span>${c.description}</span></button>`)}
    </div>`;
  }

  private renderModelPop(caps: Capabilities, s: ChatSettings | null): TemplateResult {
    return html`<div class="ch-pop" role="menu">
      ${caps.models.map((m) => html`<div class="ch-pop-group"><div class="ch-pop-title">${m.name} <span class="muted small">${m.weights}, ${(m.max_len / 1024).toFixed(0)}k window</span></div>
        ${m.profiles.map((p) => html`<button type="button" role="menuitemradio" aria-checked=${s?.model === m.id && s?.profile === p.id} class="ch-pop-item ${s?.model === m.id && s?.profile === p.id ? 'is-on' : ''}" @click=${() => (this.patch({ model: m.id, profile: p.id }), (this.pop = 'none'))}><span class="ch-pop-name">${p.name}</span><span class="ch-pop-desc">${p.description}</span></button>`)}
      </div>`)}
    </div>`;
  }

  private renderEffortPop(s: ChatSettings): TemplateResult {
    return html`<div class="ch-pop" role="menu">
      <div class="ch-pop-group"><div class="ch-pop-title">Thinking</div>
        <label class="switch ch-pop-switch"><input type="checkbox" .checked=${s.thinking} @change=${(e: Event) => this.patch({ thinking: (e.target as HTMLInputElement).checked })} /><span>${s.thinking ? 'on' : 'off'}</span></label>
      </div>
      <div class="ch-pop-group"><div class="ch-pop-title">Effort</div>
        ${(this.caps?.efforts ?? EFFORTS).map((e) => html`<button type="button" role="menuitemradio" aria-checked=${s.effort === e} class="ch-pop-item ${s.effort === e ? 'is-on' : ''}" ?disabled=${!s.thinking} @click=${() => (this.patch({ effort: e }), (this.pop = 'none'))}><span class="ch-pop-name">${effortWord(e)}</span><span class="ch-pop-desc">${effortDesc(e)}</span></button>`)}
      </div>
    </div>`;
  }

  private renderToolsPop(s: ChatSettings): TemplateResult {
    const builtin = this.tools.filter((t) => t.source.kind !== 'mcp' && t.enabled);
    const servers = [...new Set(this.tools.filter((t) => t.source.kind === 'mcp').map((t) => (t.source as { id: string }).id))];
    const toggle = (id: string) => this.patch({ tools: s.tools.includes(id) ? s.tools.filter((x) => x !== id) : [...s.tools, id] });
    const toggleMcp = (id: string) => this.patch({ mcp: s.mcp.includes(id) ? s.mcp.filter((x) => x !== id) : [...s.mcp, id] });
    return html`<div class="ch-pop ch-pop-wide" role="menu">
      <div class="ch-pop-group"><div class="ch-pop-title">Tools in this chat</div>
        ${builtin.map((t) => html`<label class="ch-pop-check"><input type="checkbox" .checked=${s.tools.includes(t.id)} @change=${() => toggle(t.id)} /><code>${toolLabel(t.name)}</code><span class="ch-pop-desc">${t.description}</span><span class="ch-policy">${s.policies[t.id] ?? t.policy}</span></label>`)}
      </div>
      <div class="ch-pop-group"><div class="ch-pop-title">MCP servers</div>
        ${servers.map((id) => html`<label class="ch-pop-check"><input type="checkbox" .checked=${s.mcp.includes(id)} @change=${() => toggleMcp(id)} /><code>${id}</code><span class="ch-pop-desc">${this.tools.filter((t) => t.source.kind === 'mcp' && (t.source as { id: string }).id === id).length} tools</span></label>`)}
      </div>
      <div class="ch-pop-foot"><button type="button" class="linkish" @click=${() => (this.fire('ch-open-settings', { section: 'tools' }), (this.pop = 'none'))}>All tools and policies</button><button type="button" class="linkish" @click=${() => (this.fire('ch-open-settings', { section: 'mcp' }), (this.pop = 'none'))}>MCP servers</button></div>
    </div>`;
  }
}

function effortDesc(e: EffortChoice): string {
  return e === 'auto' ? 'the harness decides per turn' : e === 'low' ? 'a short think, quick answers' : e === 'medium' ? 'a middle budget' : e === 'high' ? 'long thinking for code and plans' : 'the largest budget, slow';
}

function readDataUrl(f: File): Promise<string | null> {
  return new Promise((resolve) => {
    const r = new FileReader();
    r.onload = () => resolve(typeof r.result === 'string' ? r.result : null);
    r.onerror = () => resolve(null);
    r.readAsDataURL(f);
  });
}
