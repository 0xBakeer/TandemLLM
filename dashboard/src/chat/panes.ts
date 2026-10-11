// <qse-chat-panes>: the sub-windows beside or below the thread. A tab strip (the open panes, a
// plus menu, the dock toggle, close), and one pane at a time: Terminal (the screen in terminal.ts,
// an input line, attached to the sandbox's shell or a tool call's output), Preview (a sandboxed
// iframe for an html or svg artifact, with a 390 px and a 1200 px width), Files (the sandbox's tree
// and a viewer), Diff (unified, coloured), Markdown and Image. On a phone the view wraps this in a
// full-screen sheet.

import { html, nothing, type TemplateResult } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import { bytes } from '../lib/format';
import { renderMarkdown } from '../playground/markdown';
import { unsafeHTML } from 'lit/directives/unsafe-html.js';
import { LightElement } from '../ui/base';
import { copyButton, icon } from '../ui/bits';
import type { ChatBackend, TerminalSession } from './backend';
import { diffCounts, parseDiff, TerminalScreen, type DiffRow } from './terminal';
import { paneIcon, paneWord } from './thread';
import type { Conversation, PaneKind, PaneRef, SandboxFile } from './types';
import './context-panel';

export interface OpenPane extends PaneRef {
  id: string;
  /** a code block from the thread: its content travels with the pane */
  inline?: { content: string; lang: string } | null;
}

export type Dock = 'bottom' | 'right';

@customElement('qse-chat-panes')
export class QseChatPanes extends LightElement {
  @property({ attribute: false }) panes: OpenPane[] = [];
  @property({ type: String }) active: string | null = null;
  @property({ type: String }) dock: Dock = 'bottom';
  @property({ attribute: false }) backend: ChatBackend | null = null;
  @property({ type: String }) convId: string | null = null;
  @property({ type: Boolean }) sheet = false;
  @state() private plus = false;
  @state() private artifacts = new Map<string, { kind: string; content: string; mime: string; title: string }>();
  @state() private files = new Map<string, SandboxFile[]>();
  @state() private file: { path: string; content: string; mime: string; bytes: number; etag?: string } | null = null;
  @state() private previewWidth: 'fit' | 'phone' | 'desktop' = 'fit';
  @state() private termTick = 0;
  @state() private editText = '';
  @state() private editNote: { ok: boolean; text: string } | null = null;
  private loadingFile = false;
  /** for the context pane */
  @property({ attribute: false }) conv: Conversation | null = null;
  private screens = new Map<string, { screen: TerminalScreen; session: TerminalSession | null }>();
  private termLine = '';

  disconnectedCallback(): void {
    for (const s of this.screens.values()) s.session?.close();
    this.screens.clear();
    super.disconnectedCallback();
  }

  private fire(name: string, detail: unknown = {}): void {
    this.dispatchEvent(new CustomEvent(name, { detail, bubbles: true, composed: true }));
  }

  updated(changed: Map<string, unknown>): void {
    const p = this.panes.find((x) => x.id === this.active);
    if (!p || !this.backend || !this.convId) return;
    if (p.kind === 'preview' || p.kind === 'diff' || p.kind === 'markdown' || p.kind === 'image') {
      if (!p.inline && !this.artifacts.has(p.ref)) void this.loadArtifact(p.ref);
    } else if (p.kind === 'files') {
      if (!this.files.has('/work/sync')) void this.loadDir('/work/sync');
      if (p.ref && p.ref.startsWith('/') && this.file?.path !== p.ref && changed.has('active')) void this.openFile(p.ref);
    } else if (p.kind === 'terminal') {
      this.ensureTerminal(p);
      const el = this.querySelector<HTMLElement>('.ch-term-screen');
      if (el) el.scrollTop = el.scrollHeight;
    }
  }

  private async loadArtifact(ref: string): Promise<void> {
    try {
      const a = await this.backend!.artifact(this.convId!, ref);
      this.artifacts = new Map(this.artifacts).set(ref, a);
    } catch (e) {
      this.artifacts = new Map(this.artifacts).set(ref, { kind: 'error', content: (e as Error).message, mime: 'text/plain', title: 'not found' });
    }
  }
  private async loadDir(path: string): Promise<void> {
    const list = await this.backend!.sandboxFiles(this.convId!, path).catch(() => []);
    this.files = new Map(this.files).set(path, list);
  }
  private async openFile(path: string): Promise<void> {
    try {
      const f = await this.backend!.readFile(this.convId!, path);
      this.file = { path, ...f };
      this.editText = f.content;
    } catch (e) {
      this.file = { path, content: (e as Error).message, mime: 'text/plain', bytes: 0 };
      this.editText = '';
    }
    this.editNote = null;
  }

  private ensureTerminal(p: OpenPane): void {
    if (this.screens.has(p.id)) return;
    const screen = new TerminalScreen();
    const entry = { screen, session: null as TerminalSession | null };
    this.screens.set(p.id, entry);
    entry.session = this.backend!.terminal(this.convId!, p.ref || 'shell', (data) => {
      screen.feed(data);
      this.termTick++;
    });
  }

  // ---- render ---------------------------------------------------------------------------------

  render() {
    const p = this.panes.find((x) => x.id === this.active) ?? this.panes[0];
    return html`<div class="ch-panes ${this.sheet ? 'is-sheet' : ''}" data-dock=${this.dock}>
      <div class="ch-pane-tabs" role="tablist" aria-label="panes">
        ${this.panes.map((x) => html`<span class="ch-pane-tab ${x.id === p?.id ? 'is-on' : ''}" role="tab" aria-selected=${x.id === p?.id}><button type="button" class="ch-pane-tab-btn" @click=${() => this.fire('ch-pane-activate', { id: x.id })}>${icon(paneIcon(x.kind))}<span>${x.title ?? paneWord(x.kind)}</span></button><button type="button" class="ch-pane-tab-x" aria-label=${`Close ${x.title ?? paneWord(x.kind)}`} @click=${() => this.fire('ch-pane-close', { id: x.id })}>${icon('x')}</button></span>`)}
        <span class="ch-pane-plus">
          <button type="button" class="btn btn-ghost btn-sm" aria-label="Open a pane" aria-haspopup="menu" aria-expanded=${this.plus} @click=${() => (this.plus = !this.plus)}>${icon('plus')}</button>
          ${this.plus ? html`<div class="ch-pop ch-pane-menu" role="menu">${(['terminal', 'files', 'preview', 'diff', 'plan', 'context'] as PaneKind[]).map((k) => html`<button type="button" role="menuitem" class="ch-pop-item" @click=${() => (this.fire('ch-pane-open', { pane: defaultPane(k) }), (this.plus = false))}>${icon(paneIcon(k))}<span class="ch-pop-name">${paneWord(k)}</span></button>`)}</div>` : nothing}
        </span>
        <span class="grow"></span>
        ${this.sheet ? nothing : html`<button type="button" class="btn btn-ghost btn-sm ch-dock-btn" title=${this.dock === 'bottom' ? 'Dock to the right' : 'Dock below'} aria-label=${this.dock === 'bottom' ? 'Dock to the right' : 'Dock below'} @click=${() => this.fire('ch-pane-dock', { dock: this.dock === 'bottom' ? 'right' : 'bottom' })}>${icon(this.dock === 'bottom' ? 'dock-right' : 'dock-bottom')}</button>`}
        <button type="button" class="btn btn-ghost btn-sm" aria-label="Close panes" title="Close" @click=${() => this.fire('ch-panes-close')}>${icon('x')}</button>
      </div>
      <div class="ch-pane-body" data-kind=${p?.kind ?? ''}>${p ? this.renderPane(p) : html`<p class="muted small ch-pane-empty">No pane open. Press + to open the terminal, the files, or a preview.</p>`}</div>
    </div>`;
  }

  private renderPane(p: OpenPane): TemplateResult {
    switch (p.kind) {
      case 'terminal':
        return this.renderTerminal(p);
      case 'preview':
        return this.renderPreview(p);
      case 'files':
        return this.renderFiles(p);
      case 'diff':
        return this.renderDiff(p);
      case 'markdown':
        return this.renderMarkdown(p);
      case 'image':
        return this.renderImage(p);
      case 'plan':
        return this.renderMarkdown(p);
      case 'editor':
        return this.renderEditor(p);
      case 'context':
        return html`<div class="ch-ctxpane"><qse-chat-context .conv=${this.conv} .now=${Date.now()}></qse-chat-context></div>`;
      case 'pdf':
        return this.renderPdf(p);
    }
  }

  private renderEditor(p: OpenPane): TemplateResult {
    if (this.file?.path !== p.ref) {
      if (p.ref && !this.loadingFile) {
        this.loadingFile = true;
        void this.openFile(p.ref).finally(() => (this.loadingFile = false));
      }
      return html`<p class="muted small ch-pane-empty">loading</p>`;
    }
    const f = this.file;
    return html`<div class="ch-editor">
      <div class="ch-pane-bar"><code class="small">${f.path}</code><span class="muted small num">${bytes(this.editText.length, 0)}</span><span class="grow"></span>${this.editNote ? html`<span class="small ${this.editNote.ok ? 'muted' : 'err-ink'}" role="status">${this.editNote.text}</span>` : nothing}<button type="button" class="btn btn-sm btn-primary" ?disabled=${this.editText === f.content} @click=${async () => {
        try {
          await this.backend!.writeFile(this.convId!, f.path, this.editText, f.etag);
          this.file = { ...f, content: this.editText, bytes: this.editText.length, etag: String(this.editText.length) };
          this.editNote = { ok: true, text: 'saved' };
        } catch (e) {
          this.editNote = { ok: false, text: (e as Error).message };
        }
      }}>Save</button></div>
      <textarea class="ch-editor-area" spellcheck="false" aria-label=${`edit ${f.path}`} .value=${this.editText} @input=${(e: Event) => ((this.editText = (e.target as HTMLTextAreaElement).value), (this.editNote = null))}></textarea>
    </div>`;
  }

  private renderPdf(p: OpenPane): TemplateResult {
    const url = p.ref.startsWith('/') || p.ref.startsWith('http') ? p.ref : `/v1/chat-app/attachments/${p.ref}`;
    return html`<div class="ch-pdf"><div class="ch-pane-bar"><span class="muted small">${p.title ?? 'PDF'}</span><span class="grow"></span><a class="btn btn-ghost btn-sm" href=${url} target="_blank" rel="noopener">${icon('external')} Open</a></div><iframe class="ch-pdf-frame" title=${p.title ?? 'PDF'} src=${url}></iframe></div>`;
  }

  private renderTerminal(p: OpenPane): TemplateResult {
    const entry = this.screens.get(p.id);
    const screen = entry?.screen;
    const shell = !p.ref || p.ref === 'shell';
    void this.termTick;
    return html`<div class="ch-term" @click=${() => this.querySelector<HTMLInputElement>('.ch-term-input')?.focus()}>
      <div class="ch-term-screen" aria-live="off" aria-label="terminal">
        ${screen
          ? screen.lines.map((l, i) => html`<div class="ch-term-line">${l.runs.map((r) => html`<span class=${`${r.fg != null ? `tc-${r.fg}` : ''} ${r.bold ? 'tc-b' : ''} ${r.dim ? 'tc-d' : ''}`}>${r.text}</span>`)}${i === screen.lines.length - 1 ? html`<span class="ch-term-cursor" aria-hidden="true"></span>` : nothing}</div>`)
          : html`<div class="ch-term-line muted">connecting to the sandbox</div>`}
      </div>
      ${shell ? html`<input class="ch-term-input sr-only" aria-label="terminal input" autocomplete="off" @keydown=${(e: KeyboardEvent) => this.onTermKey(e, p)} />` : html`<div class="ch-term-foot muted small">a tool call's output; the shell tab takes input</div>`}
    </div>`;
  }

  private onTermKey(e: KeyboardEvent, p: OpenPane): void {
    const s = this.screens.get(p.id)?.session;
    if (!s) return;
    if (e.key === 'Enter') s.write('\r');
    else if (e.key === 'Backspace') s.write('\x7f');
    else if (e.key.length === 1 && !e.metaKey && !e.ctrlKey) s.write(e.key);
    else if (e.key === 'l' && e.ctrlKey) s.write('clear\r');
    else return;
    e.preventDefault();
    this.termLine = '';
  }

  private renderPreview(p: OpenPane): TemplateResult {
    const a = p.inline ? { kind: 'preview', content: p.inline.content, mime: p.inline.lang === 'svg' ? 'image/svg+xml' : 'text/html', title: p.title ?? 'preview' } : this.artifacts.get(p.ref);
    if (!a) return html`<p class="muted small ch-pane-empty">loading</p>`;
    if (a.kind === 'error') return html`<p class="ch-pane-empty err-ink">${a.content}</p>`;
    const doc = a.mime === 'image/svg+xml' ? `<!doctype html><html><body style="margin:0;display:grid;place-items:center;min-height:100vh;background:#fff">${a.content}</body></html>` : a.content;
    const w = this.previewWidth === 'phone' ? 390 : this.previewWidth === 'desktop' ? 1200 : null;
    return html`<div class="ch-preview">
      <div class="ch-pane-bar">
        <span class="muted small">${a.title}</span>
        <span class="grow"></span>
        <div class="seg" role="radiogroup" aria-label="preview width">
          ${(['fit', 'phone', 'desktop'] as const).map((k) => html`<button type="button" class="seg-btn ${this.previewWidth === k ? 'is-on' : ''}" role="radio" aria-checked=${this.previewWidth === k} @click=${() => (this.previewWidth = k)}>${k === 'fit' ? 'fit' : k === 'phone' ? '390' : '1200'}</button>`)}
        </div>
        <button type="button" class="btn btn-ghost btn-sm" title="Reload" aria-label="Reload" @click=${() => {
          const f = this.querySelector<HTMLIFrameElement>('.ch-preview iframe');
          if (f) f.srcdoc = f.srcdoc;
        }}>${icon('refresh')}</button>
        <button type="button" class="btn btn-ghost btn-sm" title="Open in a new tab" aria-label="Open in a new tab" @click=${() => {
          const url = URL.createObjectURL(new Blob([doc], { type: 'text/html' }));
          window.open(url, '_blank', 'noopener');
          setTimeout(() => URL.revokeObjectURL(url), 10_000);
        }}>${icon('external')}</button>
      </div>
      <div class="ch-preview-frame ${w ? 'is-sized' : ''}"><iframe title=${a.title} sandbox="allow-scripts" referrerpolicy="no-referrer" .srcdoc=${doc} style=${w ? `width:${w}px` : ''}></iframe></div>
    </div>`;
  }

  private renderFiles(p: OpenPane): TemplateResult {
    void p;
    const root = '/work/sync';
    const tree = (dir: string, depth: number): TemplateResult[] => {
      const list = this.files.get(dir);
      if (!list) return [html`<li class="muted small" style="padding-left:${depth * 14 + 8}px">loading</li>`];
      return list.map((f) => {
        const name = f.path.split('/').pop()!;
        if (f.kind === 'dir') {
          const open = this.files.has(f.path);
          return html`<li><button type="button" class="ch-file ${open ? 'is-open' : ''}" style="padding-left:${depth * 14 + 8}px" aria-expanded=${open} @click=${() => {
            if (open) {
              const m = new Map(this.files);
              m.delete(f.path);
              this.files = m;
            } else void this.loadDir(f.path);
          }}>${icon('folder')}<span>${name}</span></button>${open ? html`<ul class="ch-file-list">${tree(f.path, depth + 1)}</ul>` : nothing}</li>`;
        }
        return html`<li><button type="button" class="ch-file ${this.file?.path === f.path ? 'is-on' : ''}" style="padding-left:${depth * 14 + 8}px" @click=${() => void this.openFile(f.path)}>${icon('file')}<span>${name}</span><span class="num muted ch-file-size">${bytes(f.bytes, 0)}</span></button></li>`;
      });
    };
    return html`<div class="ch-files">
      <ul class="ch-file-list ch-file-root" aria-label="sandbox files"><li><div class="ch-file is-root">${icon('folder')}<span>${root}</span></div><ul class="ch-file-list">${tree(root, 1)}</ul></li></ul>
      <div class="ch-file-view">
        ${this.file
          ? html`<div class="ch-pane-bar"><code class="small">${this.file.path}</code><span class="muted small num">${bytes(this.file.bytes, 0)}</span><span class="grow"></span><button type="button" class="btn btn-ghost btn-sm" title="Edit" @click=${() => this.fire('ch-pane-open', { pane: { kind: 'editor', ref: this.file!.path, title: this.file!.path.split('/').pop() } })}>${icon('edit')} Edit</button>${copyButton(this.file.content, 'Copy file')}</div><pre class="ch-code"><code>${this.file.content}</code></pre>`
          : html`<p class="muted small ch-pane-empty">Pick a file.</p>`}
      </div>
    </div>`;
  }

  private renderDiff(p: OpenPane): TemplateResult {
    const a = p.inline ? { content: p.inline.content, title: p.title ?? 'diff' } : this.artifacts.get(p.ref);
    if (!a) return html`<p class="muted small ch-pane-empty">loading</p>`;
    const rows = parseDiff(a.content);
    const n = diffCounts(rows);
    return html`<div class="ch-diff">
      <div class="ch-pane-bar"><span class="muted small">${a.title}</span><span class="ch-diff-counts num"><span class="ch-add">+${n.add}</span> <span class="ch-del">-${n.del}</span>${n.files ? html` <span class="muted">${n.files} file${n.files === 1 ? '' : 's'}</span>` : nothing}</span><span class="grow"></span>${copyButton(a.content, 'Copy diff')}</div>
      <div class="ch-diff-scroll"><table class="ch-diff-table"><tbody>${rows.map((r) => this.renderDiffRow(r))}</tbody></table></div>
    </div>`;
  }

  private renderDiffRow(r: DiffRow): TemplateResult {
    if (r.kind === 'file' || r.kind === 'hunk' || r.kind === 'meta') return html`<tr class="ch-diff-${r.kind}"><td class="ch-diff-n"></td><td class="ch-diff-n"></td><td class="ch-diff-t"><code>${r.text}</code></td></tr>`;
    return html`<tr class="ch-diff-${r.kind}"><td class="ch-diff-n num">${r.old ?? ''}</td><td class="ch-diff-n num">${r.new ?? ''}</td><td class="ch-diff-t"><code>${r.kind === 'add' ? '+' : r.kind === 'del' ? '-' : ' '}${r.text}</code></td></tr>`;
  }

  private renderMarkdown(p: OpenPane): TemplateResult {
    const a = p.inline ? { content: p.inline.content, title: p.title ?? 'markdown' } : this.artifacts.get(p.ref);
    if (!a) return html`<p class="muted small ch-pane-empty">loading</p>`;
    return html`<div class="ch-mdpane"><div class="ch-pane-bar"><span class="muted small">${a.title}</span><span class="grow"></span>${copyButton(a.content, 'Copy Markdown')}</div><div class="ch-md ch-md-pane">${unsafeHTML(renderMarkdown(a.content))}</div></div>`;
  }

  private renderImage(p: OpenPane): TemplateResult {
    const a = this.artifacts.get(p.ref);
    if (!a) return html`<p class="muted small ch-pane-empty">loading</p>`;
    const src = a.mime === 'image/svg+xml' ? `data:image/svg+xml;charset=utf-8,${encodeURIComponent(a.content)}` : a.content;
    return html`<div class="ch-imgpane"><div class="ch-pane-bar"><span class="muted small">${a.title}</span></div><div class="ch-img-frame"><img src=${src} alt=${a.title} /></div></div>`;
  }
}

export function defaultPane(k: PaneKind): PaneRef {
  switch (k) {
    case 'terminal':
      return { kind: 'terminal', ref: 'shell', title: 'Terminal' };
    case 'files':
      return { kind: 'files', ref: '', title: 'Files' };
    case 'preview':
      return { kind: 'preview', ref: 'preview:shop-page', title: 'Preview' };
    case 'diff':
      return { kind: 'diff', ref: 'diff:worker', title: 'Diff' };
    case 'image':
      return { kind: 'image', ref: 'image:acceptance', title: 'Image' };
    case 'plan':
      return { kind: 'plan', ref: 'md:plan', title: 'Plan' };
    case 'context':
      return { kind: 'context', ref: '', title: 'Context' };
    case 'editor':
      return { kind: 'editor', ref: '/work/sync/src/sync/worker.py', title: 'worker.py' };
    case 'pdf':
      return { kind: 'pdf', ref: '', title: 'PDF' };
    default:
      return { kind: 'markdown', ref: 'md:meal-plan', title: 'Markdown' };
  }
}
