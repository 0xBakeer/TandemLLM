// Conductor · Packages (G2, 2026-10-03; VIS-E6 inside the harness, R1's library): every package
// and skill the router may put into a prompt in one list -- the class packages, the task and test
// skills, the cards and the library -- with its kind, source, switch, general flag, the tokens of
// the text that goes in (the engine's tokenizer, GET /v1/conductor/packages/catalog), its limit,
// the classes that list it and what the journal says it was used for in the last 24 h and 7 d.
// A search (the library through the server's own index, GET /v1/library/search; the rest here)
// and filters. A package opens its detail: the text rendered or raw, the frontmatter and the
// reference, where it is attached, its recent injections and a library skill's BM25 neighbours.
// Inline packages are edited here (a Markdown editor with the token counter and the kind's
// limit); a file on the box is read-only, or copied to an inline package that overrides it. Every
// save is checked first (POST .../check) and shows its diff, then becomes a new version with a
// note; attach and detach edit the router document (general packages first, K1). The Library tab
// is each profile's retrieval switch with what its shadow would have injected (library.ts).

import { html, nothing, type PropertyValues, type TemplateResult } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import { unsafeHTML } from 'lit/directives/unsafe-html.js';
import { describeError } from '../../api/client';
import { conductor, type LibraryHit, type PackageCatalog, type PackageCategory, type PackageDoc, type PackageItem, type PackagesDoc, type PackageUsage, type PackageUse, type ProfileTile, type RouterDoc, type SkillFound } from '../../api/conductor';
import { setIn } from '../../lib/conductor';
import { dateTimeShort, exact } from '../../lib/format';
import { attach, CATEGORY_ORDER, CATEGORY_WORDS, CLASS_PACKAGES_MAX, categoryOf, copyToInline, detach, estimateTokens, filterRows, isOver, limitOf, newInline, newPackageName, NO_FILTERS, packageRows, splitReference, type Filters, type PackageRow } from '../../lib/packages';
import { setParams } from '../../lib/router';
import { renderMarkdown } from '../../playground/markdown';
import { LightElement } from '../../ui/base';
import { copyButton, icon, skeleton, statTile } from '../../ui/bits';
import { clsChip, diffList, DocEditor, errorsList, modeSeg, versionsPanel } from './shared';
import './library';
import './proposals';

type Pane = 'text' | 'edit' | 'classes' | 'usage';
const PANES: { id: Pane; label: string }[] = [
  { id: 'text', label: 'Text' },
  { id: 'edit', label: 'Edit' },
  { id: 'classes', label: 'Classes' },
  { id: 'usage', label: 'Usage' },
];
type DocKey = 'packages' | 'router';

@customElement('qse-cond-packages')
export class QseCondPackages extends LightElement {
  @property({ attribute: false }) ids: string[] = [];
  @property({ attribute: false }) tiles: ProfileTile[] = [];
  /** the open package (`pkg`), the tab (`ptab`: null or library), the detail's pane (`pview`) */
  @property({ attribute: false }) selected: string | null = null;
  @property({ attribute: false }) tab: string | null = null;
  @property({ attribute: false }) pane: string | null = null;
  @state() private catalog: PackageCatalog | null = null;
  @state() private catError: string | null = null;
  @state() private usage: PackageUsage | null = null;
  @state() private filters: Filters = NO_FILTERS;
  @state() private hits: LibraryHit[] | null = null;
  @state() private libError: string | null = null;
  @state() private item: PackageItem | null = null;
  @state() private itemError: string | null = null;
  @state() private recent: PackageUse[] | null = null;
  @state() private raw = false;
  @state() private editView: 'write' | 'preview' | 'split' = 'write';
  @state() private counted: { text: string; n: number; how: 'engine' | 'estimate' } | null = null;
  @state() private review: DocKey | null = null;
  @state() private attachError: string | null = null;
  @state() private found: SkillFound[] | null = null;
  @state() private scanError: string | null = null;
  private pk = new DocEditor<PackagesDoc>(this, 'packages');
  private rt = new DocEditor<RouterDoc>(this, 'router');
  private seq = 0;
  private loadedFor: string | null = null;
  private countTimer: ReturnType<typeof setTimeout> | null = null;
  private countFor: string | null = null;

  connectedCallback(): void {
    super.connectedCallback();
    this.pk.onSaved = () => void this.refresh();
    this.rt.onSaved = () => void this.refresh();
    void this.pk.load();
    void this.rt.load();
    void this.refresh();
  }

  disconnectedCallback(): void {
    if (this.countTimer) clearTimeout(this.countTimer);
    super.disconnectedCallback();
  }

  protected willUpdate(): void {
    if (this.selected !== this.loadedFor) {
      this.loadedFor = this.selected;
      this.item = null;
      this.itemError = null;
      this.recent = null;
      this.attachError = null;
      if (this.selected) void this.loadItem(this.selected);
    }
  }

  protected updated(_changed: PropertyValues): void {
    // the editor's counter: the engine's tokenizer on the text that goes in, a moment after typing
    const p = this.selected ? this.pk.draft?.packages[this.selected] : null;
    if (!p || p.source !== 'inline') return;
    const text = splitReference(p.body, !!p.library)[0];
    if (this.counted?.text === text || this.countFor === text) return;
    this.countFor = text;
    if (this.countTimer) clearTimeout(this.countTimer);
    this.countTimer = setTimeout(() => {
      void conductor
        .packagesTokens([text])
        .then((r) => (this.counted = { text, n: r.tokens[0], how: r.tokenizer }))
        .catch(() => undefined);
    }, 400);
  }

  private async refresh(): Promise<void> {
    try {
      const [cat, use] = await Promise.all([conductor.packagesCatalog(), conductor.packagesUsage(null, 1)]);
      [this.catalog, this.usage, this.catError] = [cat, use, null];
    } catch (e) {
      this.catError = describeError(e).message;
    }
    if (this.selected) await this.loadItem(this.selected);
  }

  private async loadItem(name: string): Promise<void> {
    try {
      const [item, use] = await Promise.all([this.pk.base?.doc.packages[name] || !this.pk.base ? conductor.packageItem(name) : Promise.resolve(null), conductor.packagesUsage(name, 20)]);
      if (name !== this.selected) return;
      [this.item, this.recent, this.itemError] = [item, use.recent, null];
    } catch (e) {
      if (name === this.selected) this.itemError = describeError(e).message;
    }
  }

  private async search(q: string): Promise<void> {
    this.filters = { ...this.filters, q };
    const mine = ++this.seq;
    if (!q.trim() || !Object.values(this.pk.draft?.packages ?? {}).some((p) => p.library)) {
      [this.hits, this.libError] = [null, null];
      return;
    }
    try {
      const hits = await conductor.librarySearch(q.trim(), 20);
      if (mine === this.seq) [this.hits, this.libError] = [hits, null];
    } catch (e) {
      if (mine === this.seq) [this.hits, this.libError] = [[], `the library's search: ${describeError(e).message}`];
    }
  }

  private open(name: string | null, pane: Pane | null = null): void {
    setParams({ pkg: name, pview: pane });
  }

  private upd(path: (string | number)[], v: unknown): void {
    if (this.pk.draft) this.pk.set(setIn(this.pk.draft, path, v));
  }

  private toggleAttach(cls: string, name: string, on: boolean): void {
    const r = this.rt.draft;
    if (!r) return;
    if (!on) {
      this.attachError = null;
      this.rt.set(detach(r, cls, name));
      return;
    }
    const got = attach(r, cls, name, this.pk.draft);
    this.attachError = got.error;
    if (!got.error) this.rt.set(got.router);
  }

  private addInline(): void {
    const d = this.pk.draft!;
    const name = newPackageName(d);
    this.pk.set(setIn(d, ['packages', name], newInline(Number(name.slice('package-'.length)))));
    this.open(name, 'edit');
  }

  private copyInline(name: string): void {
    const d = this.pk.draft!;
    const it = this.item;
    if (!it || it.name !== name || !it.found) return;
    this.pk.set(setIn(d, ['packages', name], copyToInline(d.packages[name], it.body, it.reference)));
    this.open(name, 'edit');
  }

  private removePackage(name: string): void {
    const d = this.pk.draft!;
    const rest = { ...d.packages };
    delete rest[name];
    this.pk.set({ ...d, packages: rest });
    this.open(null);
  }

  private async scan(): Promise<void> {
    this.scanError = null;
    try {
      this.found = (await conductor.scan()).skills;
    } catch (e) {
      this.scanError = describeError(e).message;
    }
  }

  private register(s: SkillFound): void {
    const pkg: PackageDoc = { title: s.name.replace(/-/g, ' '), kind: 'skill', source: `skill:${s.name}`, description: s.description.slice(0, 600) || `The ${s.name} skill.`, body: '', max_tokens: 4000, enabled: true };
    this.pk.set(setIn(this.pk.draft!, ['packages', s.name], pkg));
    this.open(s.name);
  }

  // ---------------------------------------------------------------- check, diff, save
  private async startReview(ed: DocEditor<unknown>, key: DocKey): Promise<void> {
    await ed.check();
    this.review = key;
  }

  private async confirmSave(ed: DocEditor<unknown>): Promise<void> {
    if (await ed.save()) this.review = null;
  }

  private bar(ed: DocEditor<unknown>, key: DocKey, what: string): TemplateResult {
    const meta = ed.base?.meta;
    const edited = !!ed.draft && !!ed.base && JSON.stringify(ed.draft) !== JSON.stringify(ed.base.doc);
    return html`
      ${ed.showVersions ? versionsPanel(ed, what) : nothing}
      ${this.review === key
        ? html`<section class="panel pk-review" data-review=${key}>
            <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">${ed.errors.length ? html`${icon('warn')} The server refuses ${what}` : html`${icon('check')} The server accepts ${what}`}</h2>
              <p class="panel-sub">POST /v1/conductor/${key}/check against live v${meta?.version ?? 0}: ${ed.changes.length} change${ed.changes.length === 1 ? '' : 's'}. Saving makes a new version with your note; the next request uses it.</p></div></header>
            ${errorsList(ed)}
            ${diffList(ed.changes)}
            <div class="toolbar pk-review-tools">
              <button class="btn btn-primary" data-review-save=${key} ?disabled=${ed.errors.length > 0 || !ed.note.trim() || ed.saving || !ed.changes.length} @click=${() => this.confirmSave(ed)}>${ed.saving ? 'Saving…' : 'Save as a new version'}</button>
              <button class="btn btn-ghost" @click=${() => (this.review = null)}>Back to editing</button>
              ${!ed.note.trim() ? html`<span class="muted small">Write the change note first.</span>` : nothing}
            </div>
          </section>`
        : nothing}
      <div class="savebar ${edited || this.review === key ? '' : 'is-idle'}" data-savebar=${key}>
        <span class="status num" title=${meta?.note ?? ''}>${what} · live v${meta?.version ?? 0}${edited ? html` · <strong>${ed.changes.length || '…'} change${ed.changes.length === 1 ? '' : 's'}</strong>` : nothing}</span>
        <input class="note-input" placeholder="one-line change note (required)" .value=${ed.note} @input=${(e: Event) => {
          ed.note = (e.target as HTMLInputElement).value;
          this.requestUpdate();
        }} aria-label=${`change note for ${what}`} maxlength="200" data-note=${key} />
        <button class="btn btn-primary" ?disabled=${!edited || ed.saving} @click=${() => this.startReview(ed, key)} data-check-save=${key}>Check and save…</button>
        <button class="btn btn-ghost" ?disabled=${!edited} @click=${() => {
          ed.reset();
          this.review = null;
        }}>Reset</button>
        <button class="btn btn-ghost ${ed.showVersions ? 'is-on' : ''}" @click=${() => {
          ed.showVersions = !ed.showVersions;
          if (ed.showVersions) void ed.loadVersions();
          else this.requestUpdate();
        }} data-versions-btn=${key}>Versions</button>
        ${ed.saved ? html`<span class="status" role="status" data-saved=${key}>${icon('check')} ${ed.saved}</span>` : nothing}
        ${ed.saveError ? html`<span class="err-ink small" role="alert">${ed.saveError}</span>` : nothing}
      </div>`;
  }

  // ---------------------------------------------------------------- the screen
  render() {
    const tabs = html`<nav class="mem-tabs pk-tabs-top" aria-label="packages tabs">
      ${[
        { id: null, label: 'Packages and skills' },
        { id: 'library', label: 'Library retrieval' },
        { id: 'proposals', label: 'Proposals' },
      ].map(
        (t) => html`<a class="cond-tab ${(this.tab ?? null) === t.id ? 'is-on' : ''}" href="#" data-pk-top=${t.id ?? 'all'} aria-current=${(this.tab ?? null) === t.id ? 'page' : nothing} @click=${(e: Event) => {
          e.preventDefault();
          setParams({ ptab: t.id, pkg: null, pview: null });
        }}>${t.label}</a>`,
      )}
    </nav>`;
    if (this.tab === 'proposals') {
      return html`${tabs}<qse-cond-proposals .onChanged=${() => this.refresh()}></qse-cond-proposals>`;
    }
    if (this.tab === 'library') {
      return html`${tabs}<qse-cond-library .ids=${this.ids} .tiles=${this.tiles} .catalog=${this.catalog} .usage=${this.usage} .onChanged=${() => this.refresh()}></qse-cond-library>`;
    }
    const ed = this.pk;
    if (ed.loading && !ed.draft) return html`${tabs}${skeleton(6)}`;
    if (ed.loadError) return html`${tabs}<div class="state state-error" role="alert"><div class="state-title">${icon('warn')} ${ed.loadError}</div></div>`;
    const d = ed.draft!;
    const all = packageRows(d, this.catalog, this.usage?.packages ?? null, this.rt.draft);
    const rows = filterRows(all, this.filters, this.hits, this.catalog);
    const sel = this.selected && d.packages[this.selected] ? this.selected : null;
    return html`
      ${tabs}
      ${this.catError ? html`<div class="guard-line" role="alert">${icon('warn')} ${this.catError}</div>` : nothing}
      <div class="pk-layout ${sel ? 'has-detail' : ''}">
        ${this.renderList(rows, all)}
        ${sel ? this.renderDetail(sel, d.packages[sel], all.find((r) => r.name === sel)!) : this.selected ? html`<section class="panel" data-pk-gone><p class="muted">No package ${this.selected} in this version.</p><button class="btn btn-sm" @click=${() => this.open(null)}>Back to the list</button></section>` : nothing}
      </div>
      ${this.renderFound(d)}
      <details class="sec" style="margin-top:16px" data-sec="dirs">
        <summary>Skill directories on the box <span class="sec-sub">searched in order for <code>skill:&lt;name&gt;</code></span></summary>
        <div class="sec-body"><label class="field field-block"><span>one directory per line</span><textarea rows="3" .value=${d.skill_dirs.join('\n')} @input=${(e: Event) => this.upd(['skill_dirs'], (e.target as HTMLTextAreaElement).value.split('\n').map((x) => x.trim()).filter(Boolean))}></textarea></label></div>
      </details>
      ${this.rt.draft && (this.rt.showVersions || this.rt.saved || this.review === 'router' || JSON.stringify(this.rt.draft) !== JSON.stringify(this.rt.base?.doc)) ? this.bar(this.rt, 'router', 'the router') : nothing}
      ${this.bar(this.pk, 'packages', 'the packages')}
    `;
  }

  private renderList(rows: PackageRow[], all: PackageRow[]): TemplateResult {
    const f = this.filters;
    const over = all.filter((r) => isOver(r, this.catalog)).length;
    const set = (patch: Partial<Filters>) => (this.filters = { ...this.filters, ...patch });
    return html`<section class="panel pk-list" data-pk-list>
      <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">Packages and skills <span class="muted" data-pk-count-all>· ${rows.length} of ${all.length}</span></h2>
        <p class="panel-sub">What the router may put into a prompt. Tokens are the text that goes in, ${this.catalog?.tokenizer === 'estimate' ? 'estimated at 3.5 characters a token (no tokenizer on this server)' : "on the engine's tokenizer"}${over ? html`; <span class="warn-ink" data-pk-over-count>${over} over the limit</span>` : nothing}.</p></div>
        <div class="panel-tools"><button class="btn btn-sm" @click=${() => this.addInline()} data-pk-new>${icon('plus')} New package</button><button class="btn btn-sm" @click=${() => this.scan()} data-scan>${icon('search')} Skills on the box</button></div></header>
      <div class="pk-filters">
        <label class="field field-search pk-search">${icon('search')}<input type="search" data-pk-search placeholder="search packages, or the library's text" aria-label="search packages and skills" .value=${f.q} @input=${(e: Event) => void this.search((e.target as HTMLInputElement).value)} /></label>
        <label class="field"><span>kind</span><select data-pk-filter="kind" @change=${(e: Event) => set({ category: (e.target as HTMLSelectElement).value as Filters['category'] })}>
          <option value="all" ?selected=${f.category === 'all'}>all</option>
          ${CATEGORY_ORDER.map((c) => html`<option value=${c} ?selected=${f.category === c}>${CATEGORY_WORDS[c]}</option>`)}
        </select></label>
        <label class="field"><span>class</span><select data-pk-filter="class" @change=${(e: Event) => set({ cls: (e.target as HTMLSelectElement).value })}>
          <option value="" ?selected=${!f.cls}>any</option>
          ${this.ids.map((c) => html`<option value=${c} ?selected=${f.cls === c}>${c}</option>`)}
        </select></label>
        <label class="field"><span>enabled</span><select data-pk-filter="enabled" @change=${(e: Event) => set({ enabled: (e.target as HTMLSelectElement).value as Filters['enabled'] })}>
          ${(['all', 'on', 'off'] as const).map((v) => html`<option value=${v} ?selected=${f.enabled === v}>${v}</option>`)}
        </select></label>
        <label class="switch"><input type="checkbox" data-pk-filter="over" .checked=${f.over} @change=${(e: Event) => set({ over: (e.target as HTMLInputElement).checked })} /> over the limit</label>
      </div>
      ${this.libError ? html`<p class="err-ink small">${this.libError}</p>` : nothing}
      <div class="pk-head" aria-hidden="true"><span>package</span><span>tokens / limit</span><span>used 24 h · 7 d</span></div>
      <div class="pk-rows" role="list">${rows.map((r) => this.row(r))}</div>
      ${rows.length ? nothing : html`<p class="muted small" data-pk-empty>${f.q.trim() ? `Nothing matches “${f.q.trim()}”${this.hits ? ', and the library shares no word with it' : ''}.` : 'No package matches these filters.'}</p>`}
    </section>`;
  }

  private row(r: PackageRow): TemplateResult {
    const lim = limitOf(r.name, r.pkg, r.category, this.catalog);
    const n = r.entry ? r.entry.tokens : estimateTokens(splitReference(r.pkg.body, !!r.pkg.library)[0]);
    const over = isOver(r, this.catalog);
    const u = r.usage;
    const on = this.selected === r.name;
    return html`<button class="pk-row ${on ? 'is-open' : ''} ${r.pkg.enabled ? '' : 'is-off'}" role="listitem" data-pk-row=${r.name} data-category=${r.category} aria-current=${on ? 'true' : nothing} @click=${() => this.open(r.name)}>
      <span class="pk-main">
        <span class="pk-name"><strong>${r.name}</strong>
          <span class="badge pk-kind pk-kind-${r.category}">${CATEGORY_WORDS[r.category]}</span>
          ${r.pkg.general ? html`<span class="badge badge-on" title="goes in whole and first on every turn of a class that lists it">general</span>` : nothing}
          ${r.pkg.enabled ? nothing : html`<span class="badge">off</span>`}
          ${r.entry ? nothing : html`<span class="badge badge-shadow">not saved</span>`}
          ${r.entry && !r.entry.available && r.entry.error ? html`<span class="badge badge-warn" title=${r.entry.path ?? ''}>${r.entry.error.replace(/_/g, ' ')}</span>` : nothing}
          ${r.score != null ? html`<span class="pk-score num" data-score title="the library's BM25 score">${r.score.toFixed(2)}</span>` : nothing}
        </span>
        <span class="pk-desc muted small">${r.pkg.description}</span>
        <span class="pk-meta"><span class="mono small muted">${r.pkg.source}</span>${r.classes.map((c) => clsChip(c, this.ids))}${r.category === 'library' && r.entry?.fits?.length ? html`<span class="muted small">fits</span>${r.entry.fits.map((c) => clsChip(c, this.ids))}` : nothing}</span>
      </span>
      <span class="pk-tokens num ${over ? 'is-over' : ''}" data-pk-tokens title=${r.entry ? `${r.entry.tokens} tokens go in; the whole text is ${r.entry.body_tokens}` : 'estimated: not saved yet'}>
        <span>${r.entry ? '' : '≈'}${exact(n)} <span class="muted">/ ${exact(lim.limit)}</span></span>
        <span class="pk-bar"><span style="width:${Math.min(100, (100 * n) / Math.max(1, lim.limit))}%"></span></span>
        ${over ? html`<span class="badge badge-warn">over</span>` : nothing}
      </span>
      <span class="pk-usage num small" data-pk-usage>
        ${u && (u.injected.week || u.shadow.week || u.pick.week)
          ? html`${u.injected.week ? html`<span>${u.injected.day} · ${u.injected.week} in</span>` : nothing}${u.shadow.week ? html`<span class="muted">${u.shadow.day} · ${u.shadow.week} shadow</span>` : nothing}${u.pick.week ? html`<span>${u.pick.day} · ${u.pick.week} picks</span>` : nothing}`
          : html`<span class="muted">—</span>`}
      </span>
    </button>`;
  }

  private renderFound(d: PackagesDoc): TemplateResult | typeof nothing {
    if (this.scanError) return html`<p class="err-ink small">${this.scanError}</p>`;
    if (!this.found) return nothing;
    return html`<section class="panel" style="margin-top:16px" data-found>
      <header class="panel-head"><h2 class="panel-title">Skills in ${d.skill_dirs.join(', ')}</h2><div class="panel-tools"><button class="btn btn-ghost btn-icon" aria-label="Close" @click=${() => (this.found = null)}>${icon('x')}</button></div></header>
      ${this.found.length
        ? html`<div class="table-scroll"><table class="table"><tbody>${this.found.map(
            (s) => html`<tr><td><strong>${s.name}</strong><div class="muted small cell-wrap" style="white-space:normal">${s.description}</div></td><td class="num">${exact(s.chars)} chars</td>
              <td>${s.registered || d.packages[s.name] ? html`<span class="badge badge-ok">registered</span>` : html`<button class="btn btn-sm" @click=${() => this.register(s)} data-register=${s.name}>Register</button>`}</td></tr>`,
          )}</tbody></table></div>`
        : html`<p class="muted small">No SKILL.md in those directories on the box. Copy the skills there (for opencode's: <code>rsync -a ~/.config/opencode/skills/ &lt;box&gt;:~/.config/opencode/skills/</code>).</p>`}
    </section>`;
  }

  // ---------------------------------------------------------------- the detail
  private renderDetail(name: string, p: PackageDoc, row: PackageRow): TemplateResult {
    const entry = row.entry;
    const category = categoryOf(p, entry);
    const lim = limitOf(name, p, category, this.catalog);
    const pane = (PANES.some((x) => x.id === this.pane) ? this.pane : 'text') as Pane;
    const inline = p.source === 'inline';
    const used = row.classes;
    const tok = this.tokensNow(name, p, entry);
    const over = tok.n > lim.limit || (!!entry?.cut && !(inline && this.dirtyBody(name)));
    return html`<section class="panel pk-detail" data-pk-detail=${name}>
      <header class="pk-detail-head">
        <div class="pk-title">
          <h2 class="panel-title">${name}</h2>
          <span class="muted">${p.title}</span>
        </div>
        <button class="btn btn-ghost btn-sm pk-close" @click=${() => this.open(null)} data-pk-close aria-label="Close the package">${icon('x')}<span class="pk-close-word">All packages</span></button>
      </header>
      <div class="pk-badges">
        <span class="badge pk-kind pk-kind-${category}" data-pk-category>${CATEGORY_WORDS[category]}</span>
        <span class="badge">${inline ? 'inline' : p.source.startsWith('skill:') ? 'skill on the box' : 'file on the box'}</span>
        ${p.general ? html`<span class="badge badge-on">general</span>` : nothing}
        ${p.deliverable ? html`<span class="badge" data-pk-deliverable title="stays out of a turn when an offered tool delivers its result (SRV-615)">deliverable: ${p.deliverable}</span>` : nothing}
        ${p.enabled ? html`<span class="badge badge-ok">enabled</span>` : html`<span class="badge">disabled</span>`}
        ${entry ? nothing : html`<span class="badge badge-shadow">not saved</span>`}
        ${entry && !entry.available && entry.error ? html`<span class="badge badge-warn">${entry.error.replace(/_/g, ' ')}</span>` : nothing}
      </div>
      <div class="toolbar pk-switches">
        <label class="switch"><input type="checkbox" data-pk-enabled .checked=${p.enabled} @change=${(e: Event) => this.upd(['packages', name, 'enabled'], (e.target as HTMLInputElement).checked)} /> enabled</label>
        ${category === 'library' ? nothing : html`<label class="switch" title="goes in whole and first on every turn of a class that lists it (K1)"><input type="checkbox" data-pk-general .checked=${!!p.general} @change=${(e: Event) => this.upd(['packages', name, 'general'], (e.target as HTMLInputElement).checked)} /> general</label>`}
        ${inline ? nothing : html`<button class="btn btn-sm" data-pk-copy-inline ?disabled=${!this.item?.found} title=${this.item?.found ? 'an inline copy of the file overrides it' : 'the file is not on the box'} @click=${() => this.copyInline(name)}>${icon('copy')} Copy to inline</button>`}
        <button class="btn btn-sm btn-ghost" data-pk-remove ?disabled=${used.length > 0} title=${used.length ? `${used.join(', ')} list it; detach it first` : 'remove it from the packages document'} @click=${() => this.removePackage(name)}>${icon('trash')} Remove</button>
      </div>
      <div class="pk-meter ${over ? 'is-over' : ''}" data-pk-meter>
        <div class="pk-bar"><span style="width:${Math.min(100, (100 * tok.n) / Math.max(1, lim.limit))}%"></span></div>
        <span class="num small" data-pk-count>${tok.approx ? '≈ ' : ''}${exact(tok.n)} tokens of ${exact(lim.limit)} <span class="muted">(${tok.why}; limit: ${lim.source === 'house' ? (category === 'library' ? 'the house rule for a library skill' : 'the house rule for the library card') : 'max_tokens, where the server cuts'})</span></span>
        ${over ? html`<span class="badge badge-warn" data-pk-limit-warn>over the limit</span>` : nothing}
      </div>
      <nav class="mem-tabs pk-panes" aria-label="package detail">
        ${PANES.map(
          (x) => html`<a class="cond-tab ${x.id === pane ? 'is-on' : ''}" href="#" data-pk-pane=${x.id} aria-current=${x.id === pane ? 'page' : nothing} @click=${(e: Event) => {
            e.preventDefault();
            setParams({ pview: x.id === 'text' ? null : x.id });
          }}>${x.label}</a>`,
        )}
      </nav>
      <div class="pk-pane" data-pk-pane-body=${pane}>
        ${pane === 'text' ? this.paneText(name, p, entry) : pane === 'edit' ? this.paneEdit(name, p, category, lim) : pane === 'classes' ? this.paneClasses(name, p, category) : this.paneUsage(name, category)}
      </div>
    </section>`;
  }

  private dirtyBody(name: string): boolean {
    return this.pk.draft?.packages[name]?.body !== this.pk.base?.doc.packages[name]?.body;
  }

  /** The tokens the meter shows: an inline draft counted on the engine's tokenizer a moment after
   *  typing (an estimate meanwhile), a file the catalog's count of its whole text. */
  private tokensNow(name: string, p: PackageDoc, entry: PackageRow['entry']): { n: number; approx: boolean; why: string } {
    if (p.source === 'inline' && (this.dirtyBody(name) || !entry)) {
      const text = splitReference(p.body, !!p.library)[0];
      if (this.counted?.text === text) return { n: this.counted.n, approx: this.counted.how === 'estimate', why: this.counted.how === 'engine' ? "the draft, on the engine's tokenizer" : 'the draft, estimated' };
      return { n: estimateTokens(text), approx: true, why: 'the draft, estimated until the engine counts it' };
    }
    if (entry) return { n: entry.body_tokens, approx: this.catalog?.tokenizer === 'estimate', why: entry.cut ? `the whole text; ${exact(entry.tokens)} go in after the cut` : this.catalog?.tokenizer === 'estimate' ? 'estimated' : "on the engine's tokenizer" };
    return { n: 0, approx: true, why: 'not counted yet' };
  }

  private md(text: string): TemplateResult {
    if (!text) return html`<p class="muted small">(empty)</p>`;
    return this.raw ? html`<pre class="note-text pk-raw" data-pk-raw>${text}</pre>` : html`<div class="pg-md pk-md" data-pk-md>${unsafeHTML(renderMarkdown(text))}</div>`;
  }

  private paneText(name: string, p: PackageDoc, entry: PackageRow['entry']): TemplateResult {
    const inline = p.source === 'inline';
    const it = this.item?.name === name ? this.item : null;
    if (!inline && !it) return this.itemError ? html`<p class="err-ink small">${this.itemError}</p>` : skeleton(4);
    const [body, ref] = inline ? splitReference(p.body, !!p.library) : [it!.body, it!.reference];
    const front = inline ? {} : it!.frontmatter;
    return html`
      <div class="toolbar pk-text-tools">${modeSeg(['rendered', 'raw'] as const, this.raw ? 'raw' : 'rendered', (v) => (this.raw = v === 'raw'), { small: true, label: 'text view' })}
        ${!inline && it?.path ? html`<span class="mono small muted pk-path" data-pk-path>${it.path}</span>` : nothing}</div>
      ${!inline && it && !it.found ? html`<p class="note" data-pk-missing>${icon('warn')} The file is not on the box (<code>${it.path ?? p.source}</code>): nothing goes in.</p>` : nothing}
      ${Object.keys(front).length
        ? html`<h3 class="pk-sub">Frontmatter <span class="muted small">the client's metadata; never injected</span></h3>
            ${this.raw ? html`<pre class="note-text pk-raw" data-pk-front>${it!.frontmatter_raw}</pre>` : html`<dl class="kv pk-front" data-pk-front>${Object.entries(front).map(([k, v]) => html`<div><dt>${k}</dt><dd>${v}</dd></div>`)}</dl>`}`
        : nothing}
      <h3 class="pk-sub">The text that goes in ${body ? copyButton(body, 'Copy the text') : nothing}</h3>
      ${this.md(body)}
      ${entry?.cut && !this.dirtyBody(name) ? html`<p class="note" data-pk-cut>${icon('warn')} The server cuts this text at ${exact(p.max_tokens)} tokens × 3.5 characters, at a line end, and adds “[... cut to the package's max_tokens]”: ${exact(entry.tokens)} tokens go in of ${exact(entry.body_tokens)}.</p>` : nothing}
      ${ref
        ? html`<details class="sec pk-ref" data-pk-ref>
            <summary>Reference <span class="sec-sub">${exact(it?.tokens.reference ?? estimateTokens(ref))} tokens; only a read gets it (GET /v1/library/${name})</span></summary>
            <div class="sec-body">${this.md(ref)}</div>
          </details>`
        : nothing}`;
  }

  private paneEdit(name: string, p: PackageDoc, category: PackageCategory, lim: { limit: number; source: 'house' | 'max_tokens' }): TemplateResult {
    const inline = p.source === 'inline';
    return html`
      <div class="form-grid">
        <label class="field"><span>title</span><input data-pk-title .value=${p.title} @input=${(e: Event) => this.upd(['packages', name, 'title'], (e.target as HTMLInputElement).value)} /></label>
        <label class="field"><span>max tokens</span><input type="number" min="16" max="32768" data-pk-max .value=${String(p.max_tokens)} @input=${(e: Event) => this.upd(['packages', name, 'max_tokens'], Math.round(Number((e.target as HTMLInputElement).value)))} /><span class="hint">the server cuts a longer text at a line end</span></label>
        ${inline ? nothing : html`<label class="field field-wide"><span>source</span><input class="mono" .value=${p.source} readonly /><span class="hint">${this.item?.path ?? 'a file on the box'}</span></label>`}
      </div>
      <label class="field field-block"><span>description (the package question reads it)</span><textarea rows="2" data-pk-description .value=${p.description} @input=${(e: Event) => this.upd(['packages', name, 'description'], (e.target as HTMLTextAreaElement).value)}></textarea></label>
      ${inline
        ? this.editor(name, p, category, lim)
        : html`<div class="note pk-readonly" data-pk-readonly>${icon('info')} The text is the file on the box${this.item?.path ? html` at <code>${this.item.path}</code>` : nothing}, read when first needed and again when it changes. Editing files on the box is not done here. <button class="btn btn-sm" ?disabled=${!this.item?.found} @click=${() => this.copyInline(name)} data-pk-copy-inline-edit>Copy to inline</button> makes an inline copy of its text that overrides the file.</div>`}`;
  }

  private editor(name: string, p: PackageDoc, category: PackageCategory, lim: { limit: number; source: 'house' | 'max_tokens' }): TemplateResult {
    const [part, ref] = splitReference(p.body, !!p.library);
    const tok = this.tokensNow(name, p, this.catalog?.packages[name] ?? null);
    const over = tok.n > lim.limit;
    const view = this.editView;
    return html`<div class="pk-editor pk-view-${view}" data-pk-editor>
      <div class="toolbar pk-editor-tools">
        ${modeSeg(['write', 'preview', 'split'] as const, view, (v) => (this.editView = v), { small: true, label: 'editor view' })}
        <span class="pk-count num small ${over ? 'is-over' : ''}" data-pk-editor-count>${tok.approx ? '≈ ' : ''}${exact(tok.n)} of ${exact(lim.limit)} tokens${p.library && ref ? html` <span class="muted">· reference ≈ ${exact(estimateTokens(ref))}</span>` : nothing}</span>
        ${over ? html`<span class="badge badge-warn" data-pk-editor-warn>over ${lim.source === 'house' ? `the ${CATEGORY_WORDS[category]}'s ${exact(lim.limit)}` : 'max_tokens: the server will cut it'}</span>` : nothing}
      </div>
      <div class="pk-editor-panes">
        ${view !== 'preview' ? html`<textarea class="code pk-body" rows="18" spellcheck="false" data-pk-body aria-label="the package's text (Markdown)" .value=${p.body} @input=${(e: Event) => this.upd(['packages', name, 'body'], (e.target as HTMLTextAreaElement).value)}></textarea>` : nothing}
        ${view !== 'write' ? html`<div class="pg-md pk-md pk-preview" data-pk-preview>${unsafeHTML(renderMarkdown(part))}</div>` : nothing}
      </div>
      <p class="hint">${p.library ? 'A library skill: the text before “## Reference” goes in; the reference only on a read. ' : ''}${lim.source === 'house' ? `The house rule keeps ${CATEGORY_WORDS[category]} texts at ${exact(lim.limit)} tokens or fewer.` : `The server cuts the text at max_tokens × 3.5 characters, at a line end.`} Markdown; the preview is the dashboard's renderer, the model gets the text as written.</p>
    </div>`;
  }

  private paneClasses(name: string, p: PackageDoc, category: PackageCategory): TemplateResult {
    const entry = this.catalog?.packages[name] ?? null;
    if (category === 'library') {
      return html`<p class="muted small" data-pk-library-classes>A library skill is searched, not listed by a class: on a new user message the router searches the library and keeps the skills INDEX.md fits to the turn's class.</p>
        <div class="pk-meta">${entry?.fits?.length ? html`<span class="muted small">INDEX.md fits</span>${entry.fits.map((c) => clsChip(c, this.ids))}` : html`<span class="muted small">INDEX.md does not list it: it stays in for every class.</span>`}</div>
        <p class="small"><a href="#/conductor?screen=packages&ptab=library">Each profile's retrieval switch and its picks</a></p>`;
    }
    const router = this.rt.draft;
    if (!router) return skeleton(3);
    return html`
      <p class="muted small">A class's packages go in by K1's order: the general ones whole and first, then the router asks the package question among the rest and puts in the one that fits. A class lists at most ${CLASS_PACKAGES_MAX}. Attaching edits the router document; it is saved with its own check below.</p>
      ${this.attachError ? html`<p class="err-ink small" role="alert" data-pk-attach-error>${this.attachError}</p>` : nothing}
      <div class="pk-classes" data-pk-classes>
        ${router.classes.map((c) => {
          const list = c.policy.packages ?? [];
          const on = list.includes(name);
          return html`<div class="pk-class ${on ? 'is-on' : ''}" data-pk-class=${c.id}>
            <label class="switch"><input type="checkbox" .checked=${on} data-pk-attach=${c.id} aria-label=${`attach ${name} to ${c.id}`} @change=${(e: Event) => this.toggleAttach(c.id, name, (e.target as HTMLInputElement).checked)} /> ${clsChip(c.id, this.ids, c.label)}</label>
            <span class="pk-order small" data-pk-order=${c.id}>${list.length
              ? list.map((n, i) => html`<span class="pk-order-item ${n === name ? 'is-me' : ''}">${i + 1}. ${n}${this.pk.draft?.packages[n]?.general ? html` <span class="muted">general</span>` : nothing}</span>`)
              : html`<span class="muted">no packages</span>`}</span>
            <span class="muted small num">${list.length}/${CLASS_PACKAGES_MAX}</span>
          </div>`;
        })}
      </div>
      ${entry && Object.keys(entry.profiles).length
        ? html`<p class="small muted" data-pk-profiles>A profile's own routing also lists it: ${Object.entries(entry.profiles).map(([pn, cl]) => `${pn} (${cl.join(', ')})`).join('; ')}. A profile's routing replaces the router's policy for that class (the Profiles screen).</p>`
        : nothing}
      <div class="toolbar"><button class="btn btn-sm btn-ghost" data-pk-router-versions @click=${() => {
        this.rt.showVersions = true;
        void this.rt.loadVersions();
      }}>Router versions</button></div>`;
  }

  private paneUsage(name: string, category: PackageCategory): TemplateResult {
    const u = this.usage?.packages[name] ?? null;
    const zero = { day: 0, week: 0 };
    const tile = (label: string, s: { day: number; week: number }) => statTile({ label, value: exact(s.day), unit: 'in 24 h', sub: `${exact(s.week)} in 7 days` });
    const it = this.item?.name === name ? this.item : null;
    const picks = category === 'library' || (u?.pick.week ?? 0) > 0;
    return html`
      <div class="stats pk-stats" style="--pk-tiles:${picks ? 3 : 2}" data-pk-usage-tiles>
        ${tile('injected', u?.injected ?? zero)}${tile('shadow (would have)', u?.shadow ?? zero)}${picks ? tile('library picks', u?.pick ?? zero) : nothing}
      </div>
      <h3 class="pk-sub">Recent rows <span class="muted small">the journal, newest first (7 days)</span></h3>
      ${this.recent == null
        ? skeleton(3)
        : this.recent.length
          ? html`<div class="pk-uses" data-pk-recent>${this.recent.map(
              (r) => html`<div class="pk-use" data-pk-use=${r.id}>
                <span class="num small">${dateTimeShort(new Date(r.ts_ms).toISOString())}</span>
                <span class="badge ${r.how === 'injected' ? 'badge-on' : r.how === 'shadow' ? 'badge-shadow' : 'badge-warn'}">${r.how}</span>
                ${clsChip(r.class_id, this.ids)}
                <span class="muted small">${r.profile ?? '—'}${r.client_kind ? ` · ${r.client_kind}` : ''}</span>
                ${r.session_key ? html`<a class="mono small pk-session" href=${`#/conductor?screen=sessions&session=${encodeURIComponent(r.session_key)}`}>${r.session_key.length > 28 ? `${r.session_key.slice(0, 28)}…` : r.session_key}</a>` : nothing}
              </div>`,
            )}</div>`
          : html`<p class="muted small" data-pk-recent-empty>No router row names it in the last 7 days.</p>`}
      ${category === 'library'
        ? html`<h3 class="pk-sub">Neighbours <span class="muted small">the library's BM25 on this skill's name, description and text</span></h3>
            ${it?.neighbours
              ? it.neighbours.length
                ? html`<div class="pk-neighbours" data-pk-neighbours>${it.neighbours.map(
                    (h) => html`<button class="pk-neighbour" data-pk-neighbour=${h.name} @click=${() => this.open(h.name)}><strong>${h.name}</strong> <span class="num muted small">${h.score.toFixed(2)}</span><span class="muted small pk-desc">${h.description}</span></button>`,
                  )}</div>`
                : html`<p class="muted small">No other skill shares a word with it.</p>`
              : skeleton(2)}`
        : nothing}`;
  }
}
