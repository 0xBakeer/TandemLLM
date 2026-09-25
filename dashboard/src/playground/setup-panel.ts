// <qse-pg-setup>: the setup column of the Playground — system prompt, parameters (VIS-20),
// tools (VIS-21), presets (VIS-20). It owns no state of its own: every change goes up as a
// `pg-change` event with the new Setup; preset actions go up as `pg-preset`.

import { html, nothing, type TemplateResult } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import { LightElement } from '../ui/base';
import { icon } from '../ui/bits';
import { defaultLabel, NOT_A_PARAM, PARAM_GROUPS, PARAMS, type Defaults, type ParamDef, type ParamKey } from './params';
import { appendTemplate, removeTool, toolChoiceKind, toolChoiceOf, TOOL_TEMPLATES, type ToolsValidation } from './tools';
import type { ParamValues, Preset, Setup } from './types';

export type PresetAction = { action: 'save'; name: string } | { action: 'load'; name: string } | { action: 'delete'; name: string } | { action: 'rename'; name: string; to: string } | { action: 'export' } | { action: 'import'; text: string };

@customElement('qse-pg-setup')
export class QsePgSetup extends LightElement {
  @property({ attribute: false }) setup!: Setup;
  @property({ attribute: false }) defaults: Defaults = {};
  @property({ attribute: false }) presets: Preset[] = [];
  @property({ attribute: false }) presetName: string | null = null;
  @property({ type: Boolean }) presetDirty = false;
  @property({ attribute: false }) tools: ToolsValidation = { tools: [], errors: [] };
  @property({ attribute: false }) paramErrors: Partial<Record<ParamKey, string>> = {};
  @property({ type: Boolean }) defaultsReady = false;
  @property({ type: Boolean }) storageOk = true;

  @state() private open: Record<string, boolean> = { system: true, params: true, tools: false, presets: false };
  @state() private stopDraft = '';
  @state() private presetDraft = '';
  @state() private renaming: string | null = null;
  @state() private renameTo = '';

  private emit(patch: Partial<Setup>): void {
    this.dispatchEvent(new CustomEvent<Setup>('pg-change', { detail: { ...this.setup, ...patch }, bubbles: true, composed: true }));
  }
  private setParam(key: ParamKey, value: unknown): void {
    const params: ParamValues = { ...this.setup.params };
    if (value === undefined) delete params[key];
    else (params as Record<string, unknown>)[key] = value;
    this.emit({ params });
  }
  private preset(a: PresetAction): void {
    this.dispatchEvent(new CustomEvent<PresetAction>('pg-preset', { detail: a, bubbles: true, composed: true }));
  }
  private toggle(id: string): void {
    this.open = { ...this.open, [id]: !this.open[id] };
  }

  /** Open a section from outside (the toolbar's "Save preset" lands in Presets). */
  show(id: string): void {
    this.open = { ...this.open, [id]: true };
  }

  private section(id: string, title: string, badge: TemplateResult | string | typeof nothing, body: TemplateResult): TemplateResult {
    const isOpen = !!this.open[id];
    return html`<section class="pg-section ${isOpen ? 'is-open' : ''}" data-section=${id}>
      <button type="button" class="pg-section-head" aria-expanded=${isOpen} aria-controls="pg-sec-${id}" @click=${() => this.toggle(id)}>
        <span class="pg-section-chev" aria-hidden="true"></span>
        <span class="pg-section-title">${title}</span>
        <span class="pg-section-badge">${badge}</span>
      </button>
      <div class="pg-section-body" id="pg-sec-${id}" ?hidden=${!isOpen}>${body}</div>
    </section>`;
  }

  // ---- system prompt ---------------------------------------------------------------------
  private renderSystem(): TemplateResult {
    const n = this.setup.system.length;
    return this.section(
      'system',
      'System prompt',
      n ? html`<span class="num">${n.toLocaleString('en-US')} chars</span>` : 'none',
      html`<label class="field field-block">
        <span class="sr-only">System prompt</span>
        <textarea id="pg-system" rows="5" placeholder="Empty: no system message is sent." .value=${this.setup.system} @input=${(e: Event) => this.emit({ system: (e.target as HTMLTextAreaElement).value })}></textarea>
      </label>`,
    );
  }

  // ---- parameters ------------------------------------------------------------------------
  private changedCount(): number {
    return PARAMS.filter((p) => this.setup.params[p.key] !== undefined).length;
  }

  private renderParam(p: ParamDef): TemplateResult {
    const v = this.setup.params[p.key];
    const changed = v !== undefined;
    const err = this.paramErrors[p.key];
    const def = this.defaultsReady ? defaultLabel(p, this.defaults) : 'default';
    const id = `pg-param-${p.key}`;
    let control: TemplateResult;
    switch (p.kind) {
      case 'number':
      case 'integer': {
        const max = p.key === 'max_tokens' ? (this.defaults.max_len ?? p.max) : p.max;
        control = html`<input id=${id} type="number" inputmode="decimal" min=${p.min ?? nothing} max=${max ?? nothing} step=${p.step ?? nothing} placeholder=${def} .value=${changed ? String(v) : ''} aria-invalid=${err ? 'true' : nothing}
          @input=${(e: Event) => {
            const raw = (e.target as HTMLInputElement).value.trim();
            if (raw === '') return this.setParam(p.key, undefined);
            const n = Number(raw);
            this.setParam(p.key, Number.isFinite(n) ? n : Number.NaN);
          }} />`;
        break;
      }
      case 'switch': {
        const on = changed ? (v as boolean) : (this.defaults[p.key] as boolean | undefined) ?? true;
        control = html`<label class="switch"><input id=${id} type="checkbox" .checked=${on} @change=${(e: Event) => this.setParam(p.key, (e.target as HTMLInputElement).checked)} /><span>${on ? 'on' : 'off'}</span></label>`;
        break;
      }
      case 'select':
        control = html`<select id=${id} .value=${changed ? String(v) : ''} @change=${(e: Event) => this.setParam(p.key, (e.target as HTMLSelectElement).value || undefined)}>
          <option value="">default (${def})</option>
          ${p.options!.map((o) => html`<option value=${o} ?selected=${v === o}>${o}</option>`)}
        </select>`;
        break;
      case 'list': {
        const list = (v as string[] | undefined) ?? [];
        control = html`<div class="pg-chips">
          ${list.map((s, i) => html`<span class="pg-chip"><code>${s}</code><button type="button" class="pg-chip-x" aria-label="Remove stop ${s}" @click=${() => this.setParam(p.key, list.length > 1 ? list.filter((_x, k) => k !== i) : undefined)}>${icon('x')}</button></span>`)}
          <input id=${id} type="text" class="pg-chip-input" placeholder=${list.length ? 'add' : 'add a stop string'} .value=${this.stopDraft} @input=${(e: Event) => (this.stopDraft = (e.target as HTMLInputElement).value)}
            @keydown=${(e: KeyboardEvent) => {
              if (e.key === 'Enter' && this.stopDraft) {
                e.preventDefault();
                this.setParam(p.key, [...list, this.stopDraft]);
                this.stopDraft = '';
              }
            }} />
        </div>`;
        break;
      }
    }
    return html`<div class="pg-param ${changed ? 'is-changed' : ''} ${err ? 'is-invalid' : ''}" data-param=${p.key}>
      <label class="pg-param-label" for=${id} title=${p.help}>${p.label}</label>
      <div class="pg-param-control">${control}</div>
      <div class="pg-param-meta">
        ${changed ? html`<button type="button" class="pg-reset" aria-label="Reset ${p.label} to the server default" title="Reset to the server default" @click=${() => this.setParam(p.key, undefined)}>${icon('refresh')}</button>` : nothing}
        <span class="pg-param-default" title="the server's default: ${def}">${changed ? `default ${def}` : 'default'}</span>
      </div>
      ${err ? html`<div class="pg-param-error" role="alert">${err}</div>` : nothing}
    </div>`;
  }

  private renderParams(): TemplateResult {
    const n = this.changedCount();
    return this.section(
      'params',
      'Parameters',
      n ? html`<span class="pg-badge-changed"><span class="num">${n}</span> changed</span>` : 'defaults',
      html`<p class="pg-help">A field left at its default is not sent; the engine's own flags decide. Only what you change goes into the request.</p>
        ${PARAM_GROUPS.map(
          (g) => html`<div class="pg-param-group">
            <h4 class="pg-param-group-title">${g.label}</h4>
            ${PARAMS.filter((p) => p.group === g.id).map((p) => this.renderParam(p))}
          </div>`,
        )}
        <p class="pg-help pg-help-foot">${NOT_A_PARAM}</p>`,
    );
  }

  // ---- tools -----------------------------------------------------------------------------
  private renderTools(): TemplateResult {
    const names = this.tools.tools.map((t) => t.function.name);
    const kind = toolChoiceKind(this.setup.tool_choice);
    const namedName = kind === 'named' ? (this.setup.tool_choice as { function: { name: string } }).function.name : names[0] ?? '';
    return this.section(
      'tools',
      'Tools',
      names.length ? html`<span class="num">${names.length}</span> defined` : this.tools.errors.length ? html`<span class="err-ink">invalid</span>` : 'none',
      html`<div class="pg-templates">
          <span class="muted small">Add from a template</span>
          ${TOOL_TEMPLATES.map((t) => html`<button type="button" class="btn btn-sm" ?disabled=${names.includes(t.def.function.name)} @click=${() => this.emit({ toolsText: appendTemplate(this.setup.toolsText, t) })}>${t.label}</button>`)}
        </div>
        <label class="field field-block">
          <span class="sr-only">Tools JSON</span>
          <textarea id="pg-tools" class="pg-tools-editor" rows="10" spellcheck="false" placeholder='[{"type": "function", "function": {"name": "…", "description": "…", "parameters": {"type": "object", "properties": {}}}}]' .value=${this.setup.toolsText} aria-invalid=${this.tools.errors.length ? 'true' : nothing} @input=${(e: Event) => this.emit({ toolsText: (e.target as HTMLTextAreaElement).value })}></textarea>
        </label>
        ${this.tools.errors.length ? html`<ul class="pg-errors" role="alert">${this.tools.errors.map((e) => html`<li>${e}</li>`)}</ul>` : nothing}
        ${names.length
          ? html`<ul class="pg-tool-list">
              ${names.map((n) => html`<li><code>${n}</code><button type="button" class="btn btn-ghost btn-sm" aria-label="Remove ${n}" @click=${() => this.emit({ toolsText: removeTool(this.setup.toolsText, n) })}>${icon('trash')}</button></li>`)}
            </ul>`
          : nothing}
        <div class="pg-toolchoice">
          <span class="pg-param-label">tool_choice</span>
          <div class="seg" role="radiogroup" aria-label="tool_choice">
            ${(['auto', 'none', 'required', 'named'] as const).map((k) => html`<button type="button" role="radio" aria-checked=${kind === k} class="seg-btn ${kind === k ? 'is-on' : ''}" ?disabled=${k === 'named' && !names.length} @click=${() => this.emit({ tool_choice: toolChoiceOf(k, namedName) })}>${k}</button>`)}
          </div>
          ${kind === 'named'
            ? html`<select aria-label="named tool" .value=${namedName} @change=${(e: Event) => this.emit({ tool_choice: toolChoiceOf('named', (e.target as HTMLSelectElement).value) })}>
                ${names.map((n) => html`<option value=${n} ?selected=${n === namedName}>${n}</option>`)}
              </select>`
            : nothing}
        </div>
        <p class="pg-help">"none" keeps the tools in the request; the engine drops them itself. A named function forces that call.</p>`,
    );
  }

  // ---- presets ---------------------------------------------------------------------------
  private renderPresets(): TemplateResult {
    return this.section(
      'presets',
      'Presets',
      this.presets.length ? html`<span class="num">${this.presets.length}</span> saved` : 'none',
      html`<form class="pg-preset-save" @submit=${(e: Event) => {
        e.preventDefault();
        const name = this.presetDraft.trim();
        if (!name) return;
        this.preset({ action: 'save', name });
        this.presetDraft = '';
      }}>
          <label class="field grow"><span class="sr-only">Preset name</span><input id="pg-preset-name" type="text" placeholder=${this.presetName ? `save as… (loaded: ${this.presetName})` : 'name this setup…'} .value=${this.presetDraft} @input=${(e: Event) => (this.presetDraft = (e.target as HTMLInputElement).value)} /></label>
          <button type="submit" class="btn btn-primary btn-sm" ?disabled=${!this.presetDraft.trim()}>Save preset</button>
        </form>
        ${this.presetName ? html`<p class="pg-help">Loaded: <strong>${this.presetName}</strong>${this.presetDirty ? html` — changed since; <button type="button" class="pg-link" @click=${() => this.preset({ action: 'save', name: this.presetName! })}>save over it</button>` : ''}</p>` : nothing}
        ${!this.storageOk ? html`<p class="pg-help warn-ink">Presets cannot be saved in this browser (storage is blocked or full); they stay for this page only.</p>` : nothing}
        ${this.presets.length
          ? html`<ul class="pg-preset-list">
              ${this.presets.map(
                (p) => html`<li class="${p.name === this.presetName ? 'is-on' : ''}">
                  ${this.renaming === p.name
                    ? html`<form class="pg-preset-rename" @submit=${(e: Event) => {
                        e.preventDefault();
                        const to = this.renameTo.trim();
                        if (to && to !== p.name) this.preset({ action: 'rename', name: p.name, to });
                        this.renaming = null;
                      }}>
                        <input type="text" aria-label="New name" .value=${this.renameTo} @input=${(e: Event) => (this.renameTo = (e.target as HTMLInputElement).value)} />
                        <button type="submit" class="btn btn-sm">Rename</button>
                        <button type="button" class="btn btn-ghost btn-sm" @click=${() => (this.renaming = null)}>Cancel</button>
                      </form>`
                    : html`<button type="button" class="pg-preset-load" @click=${() => this.preset({ action: 'load', name: p.name })}><span class="pg-preset-name">${p.name}</span><span class="muted small">${p.savedAt.slice(0, 10)}</span></button>
                      <button type="button" class="btn btn-ghost btn-sm" aria-label="Rename ${p.name}" @click=${() => ((this.renaming = p.name), (this.renameTo = p.name))}>rename</button>
                      <button type="button" class="btn btn-ghost btn-sm" aria-label="Delete ${p.name}" @click=${() => this.preset({ action: 'delete', name: p.name })}>${icon('trash')}</button>`}
                </li>`,
              )}
            </ul>`
          : html`<p class="pg-help">A preset is the system prompt, the changed parameters, the tools and tool_choice — saved in this browser only.</p>`}
        <div class="pg-preset-io">
          <button type="button" class="btn btn-sm" ?disabled=${!this.presets.length} @click=${() => this.preset({ action: 'export' })}>${icon('down')} Export all</button>
          <label class="btn btn-sm pg-file"><input type="file" accept="application/json,.json" @change=${(e: Event) => this.importFile(e)} />Import…</label>
        </div>`,
    );
  }

  private async importFile(e: Event): Promise<void> {
    const input = e.target as HTMLInputElement;
    const f = input.files?.[0];
    if (!f) return;
    const text = await f.text();
    input.value = '';
    this.preset({ action: 'import', text });
  }

  render() {
    return html`<div class="pg-setup-inner">${this.renderSystem()}${this.renderParams()}${this.renderTools()}${this.renderPresets()}</div>`;
  }
}
