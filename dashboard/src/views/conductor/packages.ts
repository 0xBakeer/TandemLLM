// Conductor · Packages: the skills and packages the router may put into a prompt (VIS-E6 inside
// the harness). Each has a description (what the package question reads), a source (inline text
// edited here, `skill:<name>` = a SKILL.md in the skill directories on the box, or `file:<path>`),
// a token cap and a switch. Import registers a skill found on the box. Which classes use a
// package is set on the Routing screen. The package library (R1) has its own panel: its skills are
// searched (by the router for a turn, by the model through GET /v1/library/search), not listed by a
// class; the search box asks the server's own index, so it ranks as they do.

import { html, nothing, type TemplateResult } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import { describeError } from '../../api/client';
import { conductor, type LibraryHit, type PackageDoc, type PackagesDoc, type RouterDoc, type SkillFound } from '../../api/conductor';
import { libraryNames, libraryRows, setIn } from '../../lib/conductor';
import { exact } from '../../lib/format';
import { LightElement } from '../../ui/base';
import { icon, skeleton } from '../../ui/bits';
import { clsChip, DocEditor, errorsList, saveBar } from './shared';

@customElement('qse-cond-packages')
export class QseCondPackages extends LightElement {
  @property({ attribute: false }) ids: string[] = [];
  @state() private router: RouterDoc | null = null;
  @state() private sel: string | null = null;
  @state() private found: SkillFound[] | null = null;
  @state() private scanError: string | null = null;
  // R1: the library's search, its answer (null: no query) and each skill's classes (INDEX.md)
  @state() private q = '';
  @state() private hits: LibraryHit[] | null = null;
  @state() private libError: string | null = null;
  @state() private classes: Record<string, string[] | null> = {};
  private seq = 0;
  private ed = new DocEditor<PackagesDoc>(this, 'packages');

  connectedCallback(): void {
    super.connectedCallback();
    void this.ed.load();
    void conductor.doc<RouterDoc>('router').then((r) => (this.router = r.doc)).catch(() => undefined);
    void conductor.library().then((l) => (this.classes = Object.fromEntries(l.skills.map((s) => [s.name, s.classes])))).catch(() => undefined);
  }

  private async search(q: string) {
    this.q = q;
    const mine = ++this.seq;
    if (!q.trim()) {
      this.hits = null;
      this.libError = null;
      return;
    }
    try {
      const hits = await conductor.librarySearch(q.trim(), 20);
      if (mine === this.seq) [this.hits, this.libError] = [hits, null];
    } catch (e) {
      if (mine === this.seq) this.libError = describeError(e).message;
    }
  }

  private usedBy(name: string): string[] {
    return (this.router?.classes ?? []).filter((c) => (c.policy.packages ?? []).includes(name)).map((c) => c.id);
  }

  private upd(path: (string | number)[], v: unknown) {
    if (this.ed.draft) this.ed.set(setIn(this.ed.draft, path, v));
  }

  private async scan() {
    this.scanError = null;
    try {
      this.found = (await conductor.scan()).skills;
    } catch (e) {
      this.scanError = describeError(e).message;
    }
  }

  private register(s: SkillFound) {
    const d = this.ed.draft!;
    const pkg: PackageDoc = { title: s.name.replace(/-/g, ' '), kind: 'skill', source: `skill:${s.name}`, description: s.description.slice(0, 600) || `The ${s.name} skill.`, body: '', max_tokens: 4000, enabled: true };
    this.ed.set(setIn(d, ['packages', s.name], pkg));
    this.sel = s.name;
  }

  private addInline() {
    const d = this.ed.draft!;
    let n = 1;
    while (d.packages[`package-${n}`]) n++;
    const name = `package-${n}`;
    this.ed.set(setIn(d, ['packages', name], { title: `Package ${n}`, kind: 'package', source: 'inline', description: 'What this package is for, in one sentence.', body: '# Instructions\n\n', max_tokens: 3000, enabled: true } satisfies PackageDoc));
    this.sel = name;
  }

  private removePackage(name: string) {
    const d = this.ed.draft!;
    const rest = { ...d.packages };
    delete rest[name];
    this.ed.set({ ...d, packages: rest });
    this.sel = null;
  }

  render() {
    const ed = this.ed;
    if (ed.loading && !ed.draft) return skeleton(6);
    if (ed.loadError) return html`<div class="state state-error" role="alert"><div class="state-title">${icon('warn')} ${ed.loadError}</div></div>`;
    const d = ed.draft!;
    const status = ed.base?.status ?? {};
    return html`
      <section class="panel" data-packages>
        <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">Packages and skills</h2><p class="panel-sub">The router injects one into the prompt when the turn's class names it and the client did not load it itself.</p></div>
          <div class="panel-tools"><button class="btn btn-sm" @click=${() => this.addInline()} data-add-package>${icon('plus')} New package</button><button class="btn btn-sm" @click=${() => this.scan()} data-scan>${icon('search')} Skills on the box</button></div></header>
        <div class="table-scroll"><table class="table">
          <thead><tr><th>name</th><th>source</th><th>on the box</th><th class="num-col">tokens</th><th>classes</th><th>enabled</th></tr></thead>
          <tbody>${Object.entries(d.packages).filter(([, p]) => !p.library).map(([name, p]) => {
            const st = status[name];
            return html`<tr class="req-row ${this.sel === name ? 'is-open' : ''}" data-package-row=${name} @click=${() => (this.sel = name)}>
              <td><strong>${name}</strong><div class="muted small">${p.title}</div></td>
              <td class="mono small">${p.source}</td>
              <td>${st ? (st.available ? html`<span class="badge badge-ok">found</span>` : html`<span class="badge badge-warn" title=${st.path ?? ''}>${st.error ?? 'missing'}</span>`) : html`<span class="badge">not saved</span>`}</td>
              <td class="num num-col">${st ? exact(st.tokens_est) : '—'}</td>
              <td>${this.usedBy(name).map((c) => clsChip(c, this.ids))}</td>
              <td>${p.enabled ? 'yes' : 'no'}</td>
            </tr>`;
          })}</tbody></table></div>
      </section>
      ${this.renderLibrary(d, status)}
      ${this.found
        ? html`<section class="panel" style="margin-top:16px" data-found>
            <header class="panel-head"><h2 class="panel-title">Skills in ${d.skill_dirs.join(', ')}</h2></header>
            ${this.found.length
              ? html`<div class="table-scroll"><table class="table"><tbody>${this.found.map(
                  (s) => html`<tr><td><strong>${s.name}</strong><div class="muted small cell-wrap" style="white-space:normal">${s.description}</div></td><td class="num">${exact(s.chars)} chars</td>
                    <td>${s.registered || d.packages[s.name] ? html`<span class="badge badge-ok">registered</span>` : html`<button class="btn btn-sm" @click=${() => this.register(s)} data-register=${s.name}>Register</button>`}</td></tr>`,
                )}</tbody></table></div>`
              : html`<p class="muted small">No SKILL.md in those directories on the box. Copy the skills there (for opencode's: <code>rsync -a ~/.config/opencode/skills/ spark:~/.config/opencode/skills/</code>).</p>`}
          </section>`
        : this.scanError
          ? html`<p class="err-ink small">${this.scanError}</p>`
          : nothing}
      ${this.sel && d.packages[this.sel] ? this.renderEditor(this.sel, d.packages[this.sel]) : nothing}
      <details class="sec" style="margin-top:16px" data-sec="dirs">
        <summary>Skill directories on the box <span class="sec-sub">searched in order for <code>skill:&lt;name&gt;</code></span></summary>
        <div class="sec-body"><label class="field field-block"><span>one directory per line</span><textarea rows="3" .value=${d.skill_dirs.join('\n')} @input=${(e: Event) => this.upd(['skill_dirs'], (e.target as HTMLTextAreaElement).value.split('\n').map((x) => x.trim()).filter(Boolean))}></textarea></label></div>
      </details>
      ${errorsList(ed)}
      ${saveBar(ed, 'the packages')}
    `;
  }

  private renderLibrary(d: PackagesDoc, status: Record<string, { tokens_est: number }>): TemplateResult | typeof nothing {
    const n = libraryNames(d).length;
    if (!n) return nothing;
    const rows = libraryRows(d, this.hits);
    return html`<section class="panel" style="margin-top:16px" data-library>
      <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">Library <span class="muted">· ${n} skills</span></h2><p class="panel-sub">Searched, not listed by a class: the router picks one for a new message (a profile's library switch), the model looks one up with <code>GET /v1/library/search</code>. The search ranks as they see it (the server's index: BM25 over name, description and text). The text before <code>## Reference</code> goes in; the reference only on a read.</p></div></header>
      <label class="field field-block"><span>search the library</span><input type="search" data-library-search placeholder="canvas game collisions" .value=${this.q} @input=${(e: Event) => void this.search((e.target as HTMLInputElement).value)} /></label>
      ${this.libError ? html`<p class="err-ink small">${this.libError}</p>` : nothing}
      ${rows.length
        ? html`<div class="table-scroll"><table class="table">
            <thead><tr><th>skill · the classes it fits</th><th class="num-col">tokens</th>${this.hits ? html`<th class="num-col">score</th>` : nothing}<th>enabled</th></tr></thead>
            <tbody>${rows.map(({ name, score }) => {
              const p = d.packages[name];
              return html`<tr class="req-row ${this.sel === name ? 'is-open' : ''}" data-library-row=${name} @click=${() => (this.sel = name)}>
                <td><strong>${name}</strong><div class="muted small cell-wrap" style="white-space:normal">${p.description}</div><div data-library-classes>${(this.classes[name] ?? []).map((c) => clsChip(c, this.ids))}</div></td>
                <td class="num num-col">${status[name] ? exact(status[name].tokens_est) : '—'}</td>
                ${this.hits ? html`<td class="num num-col" data-score>${score?.toFixed(2)}</td>` : nothing}
                <td>${p.enabled ? 'yes' : 'no'}</td>
              </tr>`;
            })}</tbody></table></div>`
        : html`<p class="muted small" data-library-empty>No skill shares a word with “${this.q.trim()}”.</p>`}
    </section>`;
  }

  private renderEditor(name: string, p: PackageDoc): TemplateResult {
    const inline = p.source === 'inline';
    const used = this.usedBy(name);
    return html`<section class="panel" style="margin-top:16px" data-package-editor=${name}>
      <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">${name}</h2><p class="panel-sub">${used.length ? html`used by ${used.join(', ')}` : 'no class uses it yet (add it on the Routing screen)'}</p></div>
        <div class="panel-tools"><button class="btn btn-sm btn-ghost" ?disabled=${used.length > 0} title=${used.length ? 'a class uses it' : ''} @click=${() => this.removePackage(name)}>${icon('trash')} Remove</button></div></header>
      <div class="form-grid">
        <label class="field"><span>title</span><input .value=${p.title} @input=${(e: Event) => this.upd(['packages', name, 'title'], (e.target as HTMLInputElement).value)} /></label>
        <label class="field"><span>source</span><input .value=${p.source} @input=${(e: Event) => this.upd(['packages', name, 'source'], (e.target as HTMLInputElement).value)} /><span class="hint">inline, skill:&lt;name&gt; or file:&lt;path on the box&gt;</span></label>
        <label class="field"><span>max tokens</span><input type="number" .value=${String(p.max_tokens)} @input=${(e: Event) => this.upd(['packages', name, 'max_tokens'], Math.round(Number((e.target as HTMLInputElement).value)))} /><span class="hint">longer text is cut at a line end</span></label>
        <label class="switch"><input type="checkbox" .checked=${p.enabled} @change=${(e: Event) => this.upd(['packages', name, 'enabled'], (e.target as HTMLInputElement).checked)} /> enabled</label>
      </div>
      <label class="field field-block"><span>description (the package question reads it)</span><textarea rows="2" .value=${p.description} @input=${(e: Event) => this.upd(['packages', name, 'description'], (e.target as HTMLTextAreaElement).value)}></textarea></label>
      ${inline
        ? html`<label class="field field-block"><span>text the model gets</span><textarea rows="14" data-package-body .value=${p.body} @input=${(e: Event) => this.upd(['packages', name, 'body'], (e.target as HTMLTextAreaElement).value)}></textarea></label>`
        : html`<p class="muted small">The text is the file on the box, read when first needed and again when it changes; its YAML front matter is left out.</p>`}
    </section>`;
  }
}
