// Conductor · Memory · Notes (2026-10-05): the model's memory as files. The notes the engine keeps
// for a user (server/conductor/notes.py), one fact a file under the engine's home on the box: about
// the user, about a project (a working directory), or true for one conversation. The tree on the
// left has the three scopes with their counts (a dot for provisional notes); the list shows a node's
// notes with their kind, status, writer, last change and how often they went into a request in 7
// days. A note opens as a file: its frontmatter, its body raw or rendered, its changelog; a person
// can edit it (a revision, ratified), ratify it, reject or archive it with a reason, pin a ratified
// rule to every request, delete it; its history shows each revision as a diff against the next. The
// search box runs the runtime's own search with its scores, so a surprising injection can be
// reproduced. Below: the requests the notes went into (the journal), the writes and refusals, the
// export as a zip and the per-user delete behind a typed confirmation. Every route needs the admin
// token, reads included: these are the user's own memories. On a phone the tree is a picker above
// the list and the note takes the width.

import { html, nothing, type PropertyValues, type TemplateResult } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import { describeError } from '../../api/client';
import { conductor, type NoteFull, type NoteHit, type NoteInjectedRow, type NoteKind, type NoteMeta, type NoteOne, type NotesList, type NoteWriteRow, type SettingsDoc } from '../../api/conductor';
import { setIn } from '../../lib/conductor';
import { dateTimeShort, exact } from '../../lib/format';
import { filterNotes, KIND_WORDS, KINDS, lineDiff, NO_FILTER, nodeLabel, notesSettings, usesLine, WRITER_WORDS, writeLine, writerRun, type NoteFilter } from '../../lib/notes';
import { setParams } from '../../lib/router';
import { LightElement } from '../../ui/base';
import { emptyState, icon, panel, skeleton, statTile } from '../../ui/bits';
import { confirmDialog, DocEditor, errorsList, modeSeg, saveBar, withToken } from './shared';

const when = (iso: string | null | undefined) => (iso ? dateTimeShort(iso) : '—');
const today = () => new Date().toISOString().slice(0, 10);

@customElement('qse-cond-notes')
export class QseCondNotes extends LightElement {
  /** the open note's id (`note` in the address) */
  @property({ attribute: false }) selected: string | null = null;
  /** the tree node (`nnode`) */
  @property({ attribute: false }) node: string | null = null;
  @state() private user = 'owner';
  @state() private list: NotesList | null = null;
  @state() private one: NoteOne | null = null;
  @state() private error: string | null = null;
  @state() private oneError: string | null = null;
  @state() private filter: NoteFilter = { ...NO_FILTER };
  @state() private query = '';
  @state() private hits: NoteHit[] | null = null;
  @state() private injected: NoteInjectedRow[] | null = null;
  @state() private writes: NoteWriteRow[] | null = null;
  @state() private busy: string | null = null;
  @state() private editing: { title: string; body: string; kind: NoteKind; tags: string; verify_by: string } | null = null;
  @state() private raw = false;
  @state() private diffAt: number | null = null;
  @state() private confirmUser = '';
  @state() private flash: string | null = null;
  private settings = new DocEditor<SettingsDoc>(this, 'settings');
  private loadedFor: string | null = null;

  connectedCallback(): void {
    super.connectedCallback();
    void this.reload();
    void this.settings.load();
  }

  protected willUpdate(changed: PropertyValues): void {
    if (changed.has('node')) this.filter = { ...this.filter, node: this.node };
    const want = this.selected ? `${this.user}#${this.selected}` : null;
    if (want !== this.loadedFor) {
      this.loadedFor = want;
      this.one = null;
      this.oneError = null;
      this.editing = null;
      this.diffAt = null;
      if (this.selected) void this.loadOne(this.selected);
    }
  }

  private async reload() {
    try {
      const [list, inj, wr] = await withToken(() => Promise.all([conductor.notes(this.user), conductor.notesInjected(this.user, 30), conductor.notesWrites(30)]));
      this.list = list;
      this.injected = inj.rows;
      this.writes = wr.rows;
      this.error = null;
    } catch (e) {
      this.error = describeError(e).message;
    }
  }

  private async loadOne(id: string) {
    try {
      const r = await withToken(() => conductor.note(id, this.user));
      if (id === this.selected) this.one = r;
    } catch (e) {
      if (id === this.selected) this.oneError = describeError(e).message;
    }
  }

  private async act(what: string, fn: () => Promise<unknown>, after?: string) {
    this.busy = what;
    this.flash = null;
    try {
      await withToken(fn);
      if (after) this.flash = after;
      await this.reload();
      if (this.selected) await this.loadOne(this.selected);
    } catch (e) {
      this.oneError = describeError(e).message;
    } finally {
      this.busy = null;
    }
  }

  private async search() {
    const q = this.query.trim();
    if (!q) {
      this.hits = null;
      return;
    }
    try {
      this.hits = (await withToken(() => conductor.notesSearch(q, this.user, 10))).hits;
    } catch (e) {
      this.error = describeError(e).message;
    }
  }

  private async download() {
    try {
      const blob = await withToken(() => conductor.notesExport(this.user));
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `memory-${this.user}.zip`;
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    } catch (e) {
      this.error = describeError(e).message;
    }
  }

  private open(id: string) {
    setParams({ note: id });
  }

  render() {
    if (this.error && !this.list) return html`<div class="state state-error" role="alert"><div class="state-title">${icon('warn')} ${this.error}</div></div>`;
    if (!this.list) return skeleton(6);
    const l = this.list;
    return html`
      ${this.renderTiles(l)}
      ${this.flash ? html`<p class="ok-ink small" role="status" data-notes-flash>${icon('check')} ${this.flash}</p>` : nothing}
      ${this.selected ? this.renderOne() : nothing}
      ${this.renderSearch()}
      ${this.renderBrowser(l)}
      ${this.renderInjected()}
      ${this.renderRuns()}
      ${this.renderWrites()}
      ${this.renderSettings()}
      ${this.renderDanger(l)}
    `;
  }

  // ---------------------------------------------------------------- tiles
  private renderTiles(l: NotesList): TemplateResult {
    const byKind = new Map<string, number>();
    for (const n of l.notes) byKind.set(n.kind, (byKind.get(n.kind) ?? 0) + 1);
    const prov = l.notes.filter((n) => n.status === 'provisional').length;
    const day = today();
    const injToday = (this.injected ?? []).filter((r) => new Date(r.ts_ms).toISOString().slice(0, 10) === day).length;
    const refusedToday = (this.writes ?? []).filter((r) => new Date(r.ts_ms).toISOString().slice(0, 10) === day && /^refused/.test(writeLine(r))).length;
    const kinds = [...byKind.entries()].sort((a, b) => b[1] - a[1]).map(([k, v]) => `${v} ${k.replace('_', ' ')}`).join(' · ');
    return html`<div class="stats notes-stats" data-notes-tiles>
      ${statTile({ label: 'live notes', value: exact(l.notes.length), sub: kinds || 'none yet' })}
      ${statTile({ label: 'provisional', value: exact(prov), sub: html`<span class="num">${exact(l.notes.length - prov)}</span> ratified by you`, warn: prov > 0 && prov === l.notes.length && l.notes.length > 5 })}
      ${statTile({ label: 'injected today', value: exact(injToday), unit: 'requests', sub: html`<span class="num">${exact(l.memory.index_notes + l.memory.body_notes)}</span> notes since the start` })}
      ${statTile({ label: 'refused today', value: exact(refusedToday), sub: html`<span class="num">${exact(l.memory.taken)}</span> blocks written, <span class="num">${exact(l.memory.failed)}</span> refused since the start` })}
    </div>`;
  }

  // ---------------------------------------------------------------- search
  private renderSearch(): TemplateResult {
    return panel(
      'Search',
      this.hits === null
        ? html`<p class="muted small">The runtime's own search (BM25 over title, tags and body, every scope of ${this.user}): the scores are the ones that decide what goes into a request.</p>`
        : this.hits.length
          ? html`<ol class="notes-hits" data-notes-hits>${this.hits.map(
              (h) => html`<li><button class="link-btn" @click=${() => this.open(h.id)} data-hit=${h.id}>${h.title}</button>
                <span class="badge">${h.kind}</span> <span class="badge">${h.scope}</span> <span class="num small" data-score>${h.score.toFixed(2)}</span>
                ${h.status === 'provisional' ? html`<span class="badge badge-shadow">provisional</span>` : nothing}</li>`,
            )}</ol>`
          : emptyState('No match', 'No note shares a word with the query.'),
      {
        tools: html`<form class="toolbar notes-search" style="margin:0" @submit=${(e: Event) => {
          e.preventDefault();
          void this.search();
        }}><label class="field field-search">${icon('search')}<input placeholder="words, as the runtime would search them" .value=${this.query} @input=${(e: Event) => (this.query = (e.target as HTMLInputElement).value)} aria-label="search the notes" data-notes-query /></label>
          <button class="btn btn-sm" type="submit" data-notes-search>Search</button>${this.hits ? html`<button class="btn btn-sm btn-ghost" type="button" @click=${() => ((this.hits = null), (this.query = ''))}>Clear</button>` : nothing}</form>`,
        id: 'notes-search',
      },
    );
  }

  // ---------------------------------------------------------------- tree and list
  private renderBrowser(l: NotesList): TemplateResult {
    const rows = filterNotes(l.notes, this.filter);
    const pick = (node: string | null) => setParams({ nnode: node, note: null });
    const tree = html`<nav class="notes-tree" aria-label="memory scopes" data-notes-tree>
      <button class="notes-node ${!this.node ? 'is-on' : ''}" @click=${() => pick(null)} data-node="all"><span>All notes</span><span class="num muted">${l.notes.length}</span></button>
      ${l.tree.map(
        (n) => html`<button class="notes-node ${this.node === n.node ? 'is-on' : ''}" @click=${() => pick(n.node)} data-node=${n.node} title=${n.label ?? n.node}>
          <span class="notes-node-label">${n.provisional ? html`<span class="dot dot-shadow" aria-label="has provisional notes"></span>` : nothing}${nodeLabel(n)}</span>
          <span class="num muted">${n.count}</span></button>`,
      )}
    </nav>`;
    const picker = html`<label class="field notes-picker"><span>scope</span><select @change=${(e: Event) => pick((e.target as HTMLSelectElement).value || null)} aria-label="memory scope" data-notes-picker>
        <option value="" ?selected=${!this.node}>All notes (${l.notes.length})</option>
        ${l.tree.map((n) => html`<option value=${n.node} ?selected=${this.node === n.node}>${nodeLabel(n)} (${n.count})</option>`)}
      </select></label>`;
    const f = this.filter;
    const set = (patch: Partial<NoteFilter>) => (this.filter = { ...this.filter, ...patch });
    const filters = html`<div class="toolbar notes-filters" style="margin:0 0 8px">
      <select aria-label="kind" @change=${(e: Event) => set({ kind: (e.target as HTMLSelectElement).value as NoteKind | '' })} data-filter="kind"><option value="">every kind</option>${KINDS.map((k) => html`<option value=${k} ?selected=${f.kind === k}>${k.replace('_', ' ')}</option>`)}</select>
      <select aria-label="status" @change=${(e: Event) => set({ status: (e.target as HTMLSelectElement).value as NoteFilter['status'] })} data-filter="status"><option value="">every status</option><option value="provisional">provisional</option><option value="ratified">ratified</option></select>
      <select aria-label="writer" @change=${(e: Event) => set({ writer: (e.target as HTMLSelectElement).value })} data-filter="writer"><option value="">every writer</option>${Object.entries(WRITER_WORDS).map(([k, v]) => html`<option value=${k}>${v}</option>`)}</select>
      <label class="switch small"><input type="checkbox" .checked=${f.stale} @change=${(e: Event) => set({ stale: (e.target as HTMLInputElement).checked })} data-filter="stale" /> stale</label>
      <label class="field field-search">${icon('search')}<input placeholder="filter by title or tag" .value=${f.q} @input=${(e: Event) => set({ q: (e.target as HTMLInputElement).value })} aria-label="filter the list" /></label>
    </div>`;
    const status = (n: NoteMeta) => html`<span class="badge ${n.status === 'ratified' ? 'badge-ok' : 'badge-shadow'}">${n.status}</span>${n.pinned ? html` <span class="badge badge-on">pinned</span>` : nothing}${n.stale ? html` <span class="badge badge-warn">stale</span>` : nothing}`;
    const table = html`<div class="table-scroll notes-table"><table class="table" data-notes-list>
        <thead><tr><th>title</th><th>kind</th><th>status</th><th>updated</th><th>writer</th><th class="num-col">uses 7d</th></tr></thead>
        <tbody>${rows.map(
          (n) => html`<tr class="req-row ${n.id === this.selected ? 'is-open' : ''}" data-note-row=${n.id} @click=${() => this.open(n.id)}>
            <td class="cell-wrap">${n.title}</td><td><span class="badge" title=${KIND_WORDS[n.kind] ?? ''}>${n.kind.replace('_', ' ')}</span></td><td>${status(n)}</td>
            <td class="num small">${when(n.updated)}</td><td class="small">${WRITER_WORDS[n.writer] ?? n.writer}</td><td class="num num-col">${exact(n.uses_7d ?? 0)}</td></tr>`,
        )}</tbody></table></div>
      <div class="mem-cards notes-cards" role="list">${rows.map(
        (n) => html`<div class="mem-card ${n.id === this.selected ? 'is-open' : ''}" role="listitem" data-note-card=${n.id} @click=${() => this.open(n.id)}>
          <div class="mem-card-head"><span class="small">${n.title}</span></div>
          <div class="small">${n.kind.replace('_', ' ')} · ${status(n)}</div>
          <div class="small muted">${when(n.updated)} · ${WRITER_WORDS[n.writer] ?? n.writer} · ${exact(n.uses_7d ?? 0)} uses in 7 days</div>
        </div>`,
      )}</div>`;
    const empty = l.notes.length
      ? emptyState('Nothing here', 'No note in this scope matches the filters.')
      : emptyState('No notes yet', l.store.refused ? `The memory directory is inside a git work tree (${l.store.refused}), so nothing is written.` : 'The model writes a note when you ask it to remember something, correct it, or decide something; with the profile\'s memory switch on or in shadow.');
    return panel(
      'Notes',
      html`<div class="notes-browser">${tree}<div class="notes-main">${picker}${filters}${rows.length ? table : empty}</div></div>`,
      {
        sub: html`<span data-notes-root>${exact(l.notes.length)} live · ${exact(l.archived.length)} archived${l.duplicates.length ? html` · <span class="warn-ink">${l.duplicates.length} possible duplicate${l.duplicates.length === 1 ? '' : 's'}</span>` : nothing} · <span class="mono">${l.store.root}/${this.user}</span></span>`,
        tools: html`${l.users.length > 1 ? html`<select aria-label="user" @change=${(e: Event) => ((this.user = (e.target as HTMLSelectElement).value), setParams({ note: null, nnode: null }), void this.reload())} data-notes-user>${l.users.map((u) => html`<option ?selected=${u === this.user}>${u}</option>`)}</select>` : nothing}
          <button class="btn btn-sm" @click=${() => this.download()} data-notes-export>${icon('down')} Export</button>
          <button class="btn btn-sm" @click=${() => this.reload()} aria-label="reload the notes">${icon('refresh')}</button>`,
        id: 'notes',
      },
    );
  }

  // ---------------------------------------------------------------- one note
  private renderOne(): TemplateResult {
    if (this.oneError && !this.one) return html`<div class="state state-error" role="alert" style="margin-bottom:16px"><div class="state-title">${icon('warn')} ${this.oneError}</div></div>`;
    const one = this.one;
    if (!one) return skeleton(4);
    const n = one.note;
    const archived = Boolean(n.archived);
    const canPin = n.status === 'ratified' && (n.kind === 'rule' || n.kind === 'ruling');
    const busy = (w: string) => this.busy === w;
    const id = n.id;
    const buttons = archived
      ? html`<button class="btn btn-sm btn-danger" ?disabled=${busy('delete')} @click=${() => this.del(n)} data-note-act="delete">${icon('trash')} Delete now</button>`
      : html`<button class="btn btn-sm" @click=${() => this.startEdit(n)} data-note-act="edit">${icon('edit')} Edit</button>
          ${n.status === 'provisional' ? html`<button class="btn btn-sm" ?disabled=${busy('ratify')} @click=${() => this.act('ratify', () => conductor.noteStatus(id, { status: 'ratified' }, this.user), 'ratified')} data-note-act="ratify">${icon('check')} Ratify</button>` : nothing}
          ${n.status === 'provisional' ? html`<button class="btn btn-sm" ?disabled=${busy('reject')} @click=${() => this.act('reject', () => conductor.noteArchive(id, 'rejected in the dashboard', this.user), 'rejected and archived')} data-note-act="reject">Reject</button>` : nothing}
          ${canPin ? html`<button class="btn btn-sm" ?disabled=${busy('pin')} @click=${() => this.act('pin', () => conductor.noteStatus(id, { pinned: !n.pinned }, this.user), n.pinned ? 'unpinned' : 'pinned to every request')} data-note-act="pin">${n.pinned ? 'Unpin' : 'Pin to every request'}</button>` : nothing}
          <button class="btn btn-sm" ?disabled=${busy('archive')} @click=${() => this.archive(n)} data-note-act="archive">Archive</button>
          <button class="btn btn-sm btn-danger" ?disabled=${busy('delete')} @click=${() => this.del(n)} data-note-act="delete">${icon('trash')} Delete</button>`;
    const meta = html`<dl class="notes-meta" data-note-meta>
      <dt>kind</dt><dd><span class="badge">${n.kind.replace('_', ' ')}</span> <span class="muted small">${KIND_WORDS[n.kind] ?? ''}</span></dd>
      <dt>scope</dt><dd>${n.scope}${n.project ? html` · <span class="mono small">${n.project}</span>` : nothing}${n.session ? html` · <a class="mono small" href=${`#/conductor?screen=memory&memory=${encodeURIComponent(n.session)}`}>${n.session}</a>` : nothing}</dd>
      <dt>status</dt><dd>${n.status}${n.pinned ? ' · pinned' : ''}${n.stale ? html` · <span class="warn-ink">stale since ${n.verify_by}</span>` : nothing}${archived ? html` · <span class="warn-ink">archived ${when(n.archived)}: ${n.reason}</span>` : nothing}</dd>
      <dt>writer</dt><dd>${WRITER_WORDS[n.writer] ?? n.writer}</dd>
      <dt>tags</dt><dd>${n.tags.length ? n.tags.map((t) => html`<span class="badge">${t}</span> `) : html`<span class="muted">—</span>`}</dd>
      ${n.verify_by ? html`<dt>verify by</dt><dd class="num">${n.verify_by}</dd>` : nothing}
      <dt>created</dt><dd class="num">${when(n.created)}</dd><dt>updated</dt><dd class="num">${when(n.updated)}</dd>
      <dt>sources</dt><dd>${n.sources.length ? n.sources.map((s) => html`<span class="mono small">${s}</span> `) : html`<span class="muted">—</span>`}</dd>
      <dt>used</dt><dd data-note-uses>${usesLine(one.uses, today())}</dd>
    </dl>`;
    const body = this.editing ? this.renderEditor() : html`<div class="notes-body">
        <div class="toolbar" style="margin:0 0 6px">${modeSeg(['rendered', 'raw'] as const, this.raw ? 'raw' : 'rendered', (v) => (this.raw = v === 'raw'), { small: true, label: 'body view' })}</div>
        ${this.raw ? html`<pre class="mem-text" data-note-raw>${n.body}</pre>` : html`<div class="notes-rendered" data-note-body>${renderMd(n.body)}</div>`}
      </div>`;
    return html`<section class="panel notes-one" style="margin-bottom:16px" data-note=${id}>
      <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">${n.title}</h2><p class="panel-sub mono">${id}</p></div>
        <div class="panel-tools notes-actions">${buttons}<button class="btn btn-ghost btn-icon" aria-label="Close the note" @click=${() => setParams({ note: null })}>${icon('x')}</button></div></header>
      ${this.oneError ? html`<p class="err-ink small" role="alert">${icon('warn')} ${this.oneError}</p>` : nothing}
      <div class="notes-file">${meta}${body}</div>
      ${this.renderHistory(one)}
    </section>`;
  }

  private startEdit(n: NoteFull) {
    this.editing = { title: n.title, body: n.body, kind: n.kind, tags: n.tags.join(', '), verify_by: n.verify_by ?? '' };
  }

  private renderEditor(): TemplateResult {
    const e = this.editing!;
    const up = (patch: Partial<typeof e>) => (this.editing = { ...e, ...patch });
    const words = e.body.trim() ? e.body.trim().split(/\s+/).length : 0;
    const over = e.title.length > 120 || words > 400 || !e.body.trim();
    const save = () =>
      this.act('edit', async () => {
        await conductor.noteEdit(this.selected!, { title: e.title, body: e.body, kind: e.kind, tags: e.tags.split(',').map((t) => t.trim()).filter(Boolean), verify_by: e.verify_by || null }, this.user);
        this.editing = null;
      }, 'saved as a new revision, ratified');
    return html`<div class="notes-editor" data-note-editor>
      <label class="field"><span>title <span class="muted num">${e.title.length}/120</span></span><input .value=${e.title} @input=${(ev: Event) => up({ title: (ev.target as HTMLInputElement).value })} data-edit="title" /></label>
      <div class="form-grid">
        <label class="field"><span>kind</span><select @change=${(ev: Event) => up({ kind: (ev.target as HTMLSelectElement).value as NoteKind })} data-edit="kind">${KINDS.map((k) => html`<option value=${k} ?selected=${k === e.kind}>${k.replace('_', ' ')}</option>`)}</select></label>
        <label class="field"><span>tags (comma)</span><input .value=${e.tags} @input=${(ev: Event) => up({ tags: (ev.target as HTMLInputElement).value })} data-edit="tags" /></label>
        <label class="field"><span>verify by</span><input type="date" .value=${e.verify_by} @input=${(ev: Event) => up({ verify_by: (ev.target as HTMLInputElement).value })} data-edit="verify_by" /></label>
      </div>
      <label class="field"><span>body <span class="muted num">${words}/400 words</span></span><textarea rows="8" .value=${e.body} @input=${(ev: Event) => up({ body: (ev.target as HTMLTextAreaElement).value })} data-edit="body"></textarea></label>
      <div class="toolbar" style="margin:0"><button class="btn btn-primary btn-sm" ?disabled=${over || this.busy === 'edit'} @click=${save} data-edit-save>Save as a revision</button><button class="btn btn-sm btn-ghost" @click=${() => (this.editing = null)}>Cancel</button>
        ${over ? html`<span class="warn-ink small">a title of at most 120 characters, a body of 1 to 400 words</span>` : html`<span class="muted small">Your edit ratifies the note; the old text stays in its history.</span>`}</div>
    </div>`;
  }

  private renderHistory(one: NoteOne): TemplateResult {
    const revs = one.revisions;
    const n = one.note;
    // the texts in order: each revision, then the live note
    const texts = [...revs.map((r) => ({ label: `revision ${r.n} · ${when(r.updated)} · ${WRITER_WORDS[r.writer] ?? r.writer}`, title: r.title, body: r.body })), { label: `now · ${when(n.updated)}`, title: n.title, body: n.body }];
    const at = this.diffAt;
    let diff: TemplateResult | typeof nothing = nothing;
    if (at !== null && at + 1 < texts.length) {
      const a = texts[at];
      const b = texts[at + 1];
      const ops = lineDiff(`# ${a.title}\n${a.body}`, `# ${b.title}\n${b.body}`);
      diff = html`<pre class="notes-diff" data-note-diff>${ops.map((o) => html`<span class="diff-${o.op === '+' ? 'add' : o.op === '-' ? 'del' : 'eq'}">${o.op === '=' ? ' ' : o.op} ${o.text}\n</span>`)}</pre>`;
    }
    return html`<details class="sec" ?open=${revs.length > 0} data-note-history>
      <summary>History <span class="sec-sub">${n.changelog.length} change${n.changelog.length === 1 ? '' : 's'}, ${revs.length} earlier text${revs.length === 1 ? '' : 's'}</span></summary>
      <div class="sec-body">
        <ul class="notes-changelog">${n.changelog.map((c) => html`<li class="small">${c}</li>`)}</ul>
        ${revs.length ? html`<ol class="notes-revs">${texts.slice(0, -1).map((t, i) => html`<li><button class="link-btn ${at === i ? 'is-on' : ''}" @click=${() => (this.diffAt = at === i ? null : i)} data-rev=${i}>${t.label}</button> <span class="muted small">to the next</span></li>`)}</ol>${diff}` : html`<p class="muted small">No earlier text: this is the note as it was written.</p>`}
      </div>
    </details>`;
  }

  private async archive(n: NoteFull) {
    const reason = window.prompt('Why is this note archived? (kept with it for 30 days)', 'no longer true');
    if (reason === null) return;
    await this.act('archive', () => conductor.noteArchive(n.id, reason || 'archived in the dashboard', this.user), 'archived');
  }

  private async del(n: NoteFull) {
    const ok = await confirmDialog('Delete this note?', [html`<strong>${n.title}</strong>`, 'The file, its history and any archived copy go at once. This cannot be undone.'], 'Delete');
    if (!ok) return;
    await this.act('delete', () => conductor.noteDelete(n.id, this.user), 'deleted');
    setParams({ note: null });
  }

  // ---------------------------------------------------------------- the journal
  private renderInjected(): TemplateResult {
    const rows = this.injected ?? [];
    return html`<details class="sec" style="margin-top:16px" data-notes-injected>
      <summary>Injected <span class="sec-sub">the last requests the notes went into (the index once a session, the matching notes on a new message)</span></summary>
      <div class="sec-body">${rows.length
        ? html`<div class="table-scroll"><table class="table"><thead><tr><th>when</th><th>session</th><th>what</th><th>notes</th></tr></thead><tbody>${rows.map(
            (r) => html`<tr><td class="num small">${dateTimeShort(new Date(r.ts_ms).toISOString())}</td>
              <td class="mono small cell-wrap">${r.session_key ? html`<a href=${`#/conductor?screen=memory&memory=${encodeURIComponent(r.session_key)}`}>${r.session_key}</a>` : '—'}</td>
              <td class="small">${r.index ? 'index' : 'bodies'}</td>
              <td class="cell-wrap">${r.ids.map((i) => html`<button class="link-btn mono small" @click=${() => this.open(i)}>${i}</button> `)}</td></tr>`,
          )}</tbody></table></div>`
        : html`<p class="muted small">No note went into a request yet.</p>`}</div>
    </details>`;
  }

  private renderRuns(): TemplateResult {
    const runs = (this.writes ?? []).filter((r) => r.task === 'writer').map(writerRun);
    return html`<details class="sec" data-notes-runs ?open=${runs.length > 0}>
      <summary>Writer runs <span class="sec-sub">the model's hidden turns that read a session and write its notes: after a compaction, after a give-up, at the session's end</span></summary>
      <div class="sec-body">${runs.length
        ? html`<div class="table-scroll"><table class="table"><thead><tr><th>when</th><th>trigger</th><th>Laya's gate</th><th>outcome</th><th class="num-col">seconds</th><th class="num-col">tokens</th><th>notes</th><th>refused</th></tr></thead><tbody>${runs.map(
            (r) => html`<tr data-run=${r.trigger}><td class="num small">${dateTimeShort(new Date(r.when).toISOString())}</td><td>${r.trigger.replace('_', ' ')}</td><td class="small">${r.gate}</td><td class="small">${r.outcome}</td>
              <td class="num num-col">${r.seconds ?? '—'}</td><td class="num num-col">${r.tokens ?? '—'}</td><td class="small">${r.ops}</td><td class="small warn-ink">${r.refused.join(', ')}</td></tr>`,
          )}</tbody></table></div>`
        : html`<p class="muted small">No run yet. The writer is ${this.list?.settings.writer?.mode ?? 'off'} (settings below).</p>`}</div>
    </details>`;
  }

  private renderWrites(): TemplateResult {
    const rows = (this.writes ?? []).filter((r) => r.task !== 'writer');
    return html`<details class="sec" data-notes-writes>
      <summary>Writes <span class="sec-sub">every block the model wrote, the script calls, your edits, the nightly job, and what was refused and why</span></summary>
      <div class="sec-body">${rows.length
        ? html`<ul class="notes-writes">${rows.map((r) => html`<li class="small ${/^refused/.test(writeLine(r)) ? 'warn-ink' : ''}"><span class="num muted">${dateTimeShort(new Date(r.ts_ms).toISOString())}</span> ${writeLine(r)}</li>`)}</ul>`
        : html`<p class="muted small">Nothing written yet.</p>`}</div>
    </details>`;
  }

  // ---------------------------------------------------------------- settings and the user's delete
  private renderSettings(): TemplateResult | typeof nothing {
    const ed = this.settings;
    if (!ed.draft) return ed.loadError ? html`<p class="err-ink small">${ed.loadError}</p>` : nothing;
    const t = notesSettings(ed.draft);
    const upd = (k: keyof typeof t, v: unknown) => ed.set(setIn(ed.draft!, ['memory', 'notes'], { ...t, [k]: v }));
    const num = (k: 'index_tokens' | 'body_tokens' | 'pinned_tokens' | 'bodies' | 'min_words', label: string, min: number, max: number) =>
      html`<label class="field"><span>${label}</span><input type="number" min=${min} max=${max} .value=${String(t[k])} data-notes-setting=${k} @input=${(e: Event) => upd(k, Math.round(Number((e.target as HTMLInputElement).value)))} /></label>`;
    return html`<details class="sec" data-sec="notes-settings">
      <summary>Notes settings <span class="sec-sub">what goes into a request, the duplicate rule, Laya's remember question</span></summary>
      <div class="sec-body">
        <div class="form-grid">
          ${num('index_tokens', 'the index, once a session (tokens)', 0, 8000)}
          ${num('body_tokens', 'the matching notes on a new message (tokens)', 0, 4000)}
          ${num('pinned_tokens', 'pinned rules (tokens)', 0, 2000)}
          ${num('bodies', 'matching notes at most', 0, 10)}
          ${num('min_words', 'words a note shares with the message at least', 1, 10)}
          <label class="field"><span>a second note this similar is a revision</span><input type="number" min="0" max="1" step="0.05" .value=${String(t.dedupe_threshold)} data-notes-setting="dedupe_threshold" @input=${(e: Event) => upd('dedupe_threshold', Number((e.target as HTMLInputElement).value))} /></label>
        </div>
        <div class="toolbar" style="margin:0" data-notes-intent><span class="muted small">Laya asks "remember, forget or recall?"</span>${modeSeg(['off', 'shadow', 'on'] as const, t.intent, (v) => upd('intent', v), { small: true, label: 'intent question' })}</div>
        <label class="switch small"><input type="checkbox" .checked=${t.provisional} @change=${(e: Event) => upd('provisional', (e.target as HTMLInputElement).checked)} /> provisional notes go into requests too (marked)</label>
        ${this.renderWriterSettings(t, upd)}
        <p class="muted small">A profile's <code>memory</code> switch turns the notes on (off, shadow: the model may write and nothing is read back, on). Every note passes the secret-pattern refusal; a rule about you needs your own message in the turn; the model never ratifies, pins or deletes.</p>
        ${errorsList(ed)}
        ${saveBar(ed, 'the settings')}
      </div>
    </details>`;
  }

  private renderWriterSettings(t: ReturnType<typeof notesSettings>, upd: (k: keyof ReturnType<typeof notesSettings>, v: unknown) => void): TemplateResult {
    const w = t.writer!;
    const set = (k: keyof typeof w, v: unknown) => upd('writer', { ...w, [k]: v });
    const num = (k: 'idle_s' | 'min_new_tokens' | 'max_tokens', label: string, min: number, max: number) =>
      html`<label class="field"><span>${label}</span><input type="number" min=${min} max=${max} .value=${String(w[k])} data-writer-setting=${k} @input=${(e: Event) => set(k, Number((e.target as HTMLInputElement).value))} /></label>`;
    return html`<h4 class="small" style="margin:12px 0 4px">The background writer</h4>
      <div class="toolbar" style="margin:0" data-writer-mode><span class="muted small">writer</span>${modeSeg(['off', 'shadow', 'on'] as const, w.mode, (v) => set('mode', v), { small: true, label: 'writer mode' })}
        <span class="muted small">after a compaction</span>${modeSeg(['off', 'idle', 'turn'] as const, w.after_compaction, (v) => set('after_compaction', v), { small: true, label: 'after a compaction' })}</div>
      <div class="form-grid">
        ${num('idle_s', "a session's end: seconds without a request", 10, 86400)}
        ${num('min_new_tokens', 'new tokens before the next pass', 0, 1000000)}
        ${num('max_tokens', 'the turn writes at most (tokens)', 64, 4096)}
      </div>
      <label class="switch small"><input type="checkbox" .checked=${w.gate} @change=${(e: Event) => set('gate', (e.target as HTMLInputElement).checked)} /> Laya decides first whether the session has anything worth keeping (about 20 ms; a no skips the turn)</label>
      <p class="muted small">shadow: the turn runs and its notes are counted, nothing is stored. <code>turn</code> after a compaction writes before the next request gets the engine (the next turn waits up to ${w.turn_deadline_s} s); <code>idle</code> waits for an idle moment, which opencode rarely leaves right after a compaction.</p>`;
  }

  private renderDanger(l: NotesList): TemplateResult {
    const ok = this.confirmUser === this.user;
    return html`<details class="sec" data-notes-danger>
      <summary>Delete ${this.user}'s memory <span class="sec-sub">every note, revision and archive of this user, at once</span></summary>
      <div class="sec-body">
        <p class="small">Type <code>${this.user}</code> to delete ${exact(l.notes.length + l.archived.length)} notes. Export them first if you want a copy.</p>
        <div class="toolbar" style="margin:0"><label class="field"><span>user</span><input .value=${this.confirmUser} @input=${(e: Event) => (this.confirmUser = (e.target as HTMLInputElement).value)} data-notes-confirm aria-label="type the user's name" /></label>
          <button class="btn btn-sm btn-danger" ?disabled=${!ok || this.busy === 'user'} @click=${() => this.act('user', () => conductor.notesDeleteUser(this.user, this.confirmUser), `${this.user}'s memory deleted`).then(() => (this.confirmUser = ''))} data-notes-delete-user>${icon('trash')} Delete everything</button></div>
      </div>
    </details>`;
  }
}

/** A note body as light markdown: paragraphs, `- ` lists, `code`, ```fences```. */
export function renderMd(text: string): TemplateResult {
  const blocks: TemplateResult[] = [];
  const lines = (text ?? '').split('\n');
  let i = 0;
  const inline = (s: string) => s.split(/(`[^`]+`)/).map((p) => (p.startsWith('`') && p.endsWith('`') && p.length > 1 ? html`<code>${p.slice(1, -1)}</code>` : p));
  while (i < lines.length) {
    const ln = lines[i];
    if (ln.startsWith('```')) {
      const code: string[] = [];
      i++;
      while (i < lines.length && !lines[i].startsWith('```')) code.push(lines[i++]);
      i++;
      blocks.push(html`<pre class="mem-text">${code.join('\n')}</pre>`);
    } else if (/^\s*[-*] /.test(ln)) {
      const items: string[] = [];
      while (i < lines.length && /^\s*[-*] /.test(lines[i])) items.push(lines[i++].replace(/^\s*[-*] /, ''));
      blocks.push(html`<ul>${items.map((t) => html`<li>${inline(t)}</li>`)}</ul>`);
    } else if (!ln.trim()) {
      i++;
    } else {
      const para: string[] = [];
      while (i < lines.length && lines[i].trim() && !lines[i].startsWith('```') && !/^\s*[-*] /.test(lines[i])) para.push(lines[i++]);
      blocks.push(html`<p>${inline(para.join(' '))}</p>`);
    }
  }
  return html`${blocks}`;
}
