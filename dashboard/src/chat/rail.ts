// <qse-chat-rail>: the conversation list. New chat first, a search box, then pinned, today, this
// week, older, and the folders. Each row: the title, the time, and a state mark with a word
// (running, waiting, error). A row's menu: pin, rename (inline), move to a folder, delete.

import { html, nothing, type TemplateResult } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import { ago } from '../lib/format';
import { LightElement } from '../ui/base';
import { icon } from '../ui/bits';
import type { ConversationSummary, Folder } from './types';

type Group = { key: string; label: string; items: ConversationSummary[] };

export function groupConversations(list: ConversationSummary[], folders: Folder[], filter: string, now = Date.now()): Group[] {
  const q = filter.trim().toLowerCase();
  const match = (c: ConversationSummary) => !q || c.title.toLowerCase().includes(q) || c.preview.toLowerCase().includes(q);
  const items = list.filter(match).sort((a, b) => b.updated_at.localeCompare(a.updated_at));
  const groups: Group[] = [];
  const pinned = items.filter((c) => c.pinned);
  if (pinned.length) groups.push({ key: 'pinned', label: 'Pinned', items: pinned });
  const rest = items.filter((c) => !c.pinned && !c.folder_id);
  const day = 86_400_000;
  const today = rest.filter((c) => now - Date.parse(c.updated_at) < day);
  const week = rest.filter((c) => now - Date.parse(c.updated_at) >= day && now - Date.parse(c.updated_at) < 7 * day);
  const older = rest.filter((c) => now - Date.parse(c.updated_at) >= 7 * day);
  if (today.length) groups.push({ key: 'today', label: 'Today', items: today });
  if (week.length) groups.push({ key: 'week', label: 'This week', items: week });
  if (older.length) groups.push({ key: 'older', label: 'Older', items: older });
  for (const f of folders) {
    const inF = items.filter((c) => !c.pinned && c.folder_id === f.id);
    if (inF.length || !q) groups.push({ key: `f:${f.id}`, label: f.name, items: inF });
  }
  return groups;
}

@customElement('qse-chat-rail')
export class QseChatRail extends LightElement {
  @property({ attribute: false }) list: ConversationSummary[] = [];
  @property({ attribute: false }) folders: Folder[] = [];
  @property({ type: String }) active: string | null = null;
  @property({ type: Boolean }) mock = false;
  /** chats whose answer finished since this device last looked */
  @property({ attribute: false }) unread = new Set<string>();
  @state() private filter = '';
  @state() private menu: string | null = null;
  @state() private renaming: { id: string; title: string } | null = null;
  @state() private newFolder = false;
  @state() private collapsed = new Set<string>();

  connectedCallback(): void {
    super.connectedCallback();
    document.addEventListener('click', this.onDocClick);
  }
  disconnectedCallback(): void {
    document.removeEventListener('click', this.onDocClick);
    super.disconnectedCallback();
  }
  private onDocClick = (e: Event) => {
    if (this.menu && !(e.target as HTMLElement).closest('.ch-row-menu, .ch-row-more')) this.menu = null;
  };

  private fire(name: string, detail: unknown = {}): void {
    this.dispatchEvent(new CustomEvent(name, { detail, bubbles: true, composed: true }));
  }

  render() {
    const groups = groupConversations(this.list, this.folders, this.filter);
    return html`<div class="ch-rail-inner">
      <div class="ch-rail-top">
        <button type="button" class="btn btn-primary ch-new" @click=${() => this.fire('ch-new')}>${icon('plus')} New chat</button>
        <button type="button" class="btn btn-ghost btn-icon ch-rail-close" aria-label="Close the list" @click=${() => this.fire('ch-close-rail')}>${icon('x')}</button>
      </div>
      <label class="field field-search ch-search"><span class="sr-only">Search chats</span>${icon('search')}<input type="search" placeholder="Search chats" .value=${this.filter} @input=${(e: Event) => (this.filter = (e.target as HTMLInputElement).value)} /></label>
      <nav class="ch-groups" aria-label="chats">
        ${groups.length ? groups.map((g) => this.renderGroup(g)) : html`<p class="ch-rail-empty muted small">${this.filter ? `Nothing matches "${this.filter}".` : 'No chats yet. Start one above.'}</p>`}
        <div class="ch-folder-new">
          ${this.newFolder
            ? html`<form class="ch-inline-form" @submit=${(e: Event) => {
                e.preventDefault();
                const inp = (e.target as HTMLFormElement).querySelector('input')!;
                if (inp.value.trim()) this.fire('ch-folder-new', { name: inp.value.trim() });
                this.newFolder = false;
              }}><input type="text" placeholder="Folder name" aria-label="Folder name" autofocus @keydown=${(e: KeyboardEvent) => e.key === 'Escape' && (this.newFolder = false)} /><button type="submit" class="btn btn-sm">Add</button></form>`
            : html`<button type="button" class="linkish small" @click=${() => (this.newFolder = true)}>${icon('folder')} New folder</button>`}
        </div>
      </nav>
      <div class="ch-rail-foot">
        <button type="button" class="btn btn-ghost ch-rail-settings" @click=${() => this.fire('ch-open-settings', { section: 'tools' })}>${icon('sliders')} Settings</button>
        ${this.mock ? html`<span class="ch-mock-badge" title="the server has no chat routes yet; this is the in-browser mock">mock data</span>` : nothing}
      </div>
    </div>`;
  }

  private renderGroup(g: Group): TemplateResult {
    const isFolder = g.key.startsWith('f:');
    const closed = this.collapsed.has(g.key);
    return html`<section class="ch-group ${isFolder ? 'is-folder' : ''}" data-group=${g.key}>
      <button type="button" class="ch-group-h" aria-expanded=${!closed} @click=${() => {
        const s = new Set(this.collapsed);
        if (s.has(g.key)) s.delete(g.key);
        else s.add(g.key);
        this.collapsed = s;
      }}>${isFolder ? icon('folder') : nothing}<span>${g.label}</span><span class="num muted">${g.items.length}</span></button>
      ${closed ? nothing : html`<ul class="ch-rows">${g.items.map((c) => this.renderRow(c))}${!g.items.length ? html`<li class="ch-row-empty muted small">empty</li>` : nothing}</ul>`}
    </section>`;
  }

  private renderRow(c: ConversationSummary): TemplateResult {
    const on = c.id === this.active;
    const renaming = this.renaming?.id === c.id;
    return html`<li class="ch-row ${on ? 'is-on' : ''} is-${c.status} ${this.unread.has(c.id) ? 'is-unread' : ''}" data-cid=${c.id} data-status=${c.status}>
      ${renaming
        ? html`<form class="ch-inline-form" @submit=${(e: Event) => {
            e.preventDefault();
            this.fire('ch-rename', { id: c.id, title: this.renaming!.title });
            this.renaming = null;
          }}><input type="text" .value=${this.renaming!.title} aria-label="Title" autofocus @input=${(e: Event) => (this.renaming = { id: c.id, title: (e.target as HTMLInputElement).value })} @keydown=${(e: KeyboardEvent) => e.key === 'Escape' && (this.renaming = null)} /><button type="submit" class="btn btn-sm">Save</button></form>`
        : html`<a class="ch-row-link" href=${`#/chat/${c.id}`} aria-current=${on ? 'page' : nothing} @click=${(e: Event) => {
            e.preventDefault();
            this.fire('ch-open', { id: c.id });
          }}>
            <span class="ch-row-title">${c.pinned ? html`<span class="ch-pin" aria-hidden="true">${icon('pin')}</span>` : nothing}${c.title}</span>
            <span class="ch-row-sub">${c.status !== 'idle' ? html`<span class="ch-mark ch-mark-${c.status}"><span class="ch-mark-dot" aria-hidden="true"></span>${c.status === 'running' ? (c.now ?? 'running') : c.status === 'waiting' ? 'waiting for you' : 'error'}</span>` : html`<span class="ch-row-preview">${c.preview}</span>`}<span class="ch-row-time num">${ago(c.updated_at)}</span></span>
          </a>`}
      <button type="button" class="btn btn-ghost btn-sm ch-row-more" aria-label=${`Actions for ${c.title}`} aria-haspopup="menu" aria-expanded=${this.menu === c.id} @click=${(e: Event) => (e.stopPropagation(), (this.menu = this.menu === c.id ? null : c.id))}>${icon('more')}</button>
      ${this.menu === c.id
        ? html`<div class="ch-row-menu" role="menu">
            <button type="button" role="menuitem" class="btn btn-ghost btn-sm" @click=${() => (this.fire('ch-pin', { id: c.id, pinned: !c.pinned }), (this.menu = null))}>${icon('pin')} ${c.pinned ? 'Unpin' : 'Pin'}</button>
            <button type="button" role="menuitem" class="btn btn-ghost btn-sm" @click=${() => ((this.renaming = { id: c.id, title: c.title }), (this.menu = null))}>${icon('edit')} Rename</button>
            <label class="ch-menu-move">${icon('folder')} <select aria-label="Move to a folder" @change=${(e: Event) => {
              this.fire('ch-move', { id: c.id, folder_id: (e.target as HTMLSelectElement).value || null });
              this.menu = null;
            }}><option value="" ?selected=${!c.folder_id}>No folder</option>${this.folders.map((f) => html`<option value=${f.id} ?selected=${c.folder_id === f.id}>${f.name}</option>`)}</select></label>
            <button type="button" role="menuitem" class="btn btn-ghost btn-sm ch-danger" @click=${() => {
              this.menu = null;
              if (confirm(`Delete "${c.title}"? The chat and its sandbox go away.`)) this.fire('ch-delete', { id: c.id });
            }}>${icon('trash')} Delete</button>
          </div>`
        : nothing}
    </li>`;
  }
}
