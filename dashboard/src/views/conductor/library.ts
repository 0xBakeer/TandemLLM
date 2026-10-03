// Conductor · Packages · Library retrieval (G2, 2026-10-03; R1's library): each profile's switch
// for the router's own search of the library (off, shadow, on) beside what its retrieval chose in
// the journal's last rows -- the candidates with their BM25 scores, the pick and how it was made,
// and whether it went in (on) or would have (shadow) -- so a shadow run can be judged before a
// profile is switched on. A switch is a new version of the profile: checked first, its diff shown
// in the confirmation, saved with a note (the admin token when the page has no session).

import { html, nothing, type TemplateResult } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import { describeError } from '../../api/client';
import { conductor, type Mode3, type PackageCatalog, type PackageUsage, type ProfileDoc, type ProfileTile, type Retrieval } from '../../api/conductor';
import { dateTimeShort, exact } from '../../lib/format';
import { libraryModeWords, retrievalsOf } from '../../lib/packages';
import { setParams } from '../../lib/router';
import { LightElement } from '../../ui/base';
import { emptyState, icon, skeleton, statTile } from '../../ui/bits';
import { clsChip, confirmDialog, diffList, modeSeg, withToken } from './shared';

@customElement('qse-cond-library')
export class QseCondLibrary extends LightElement {
  @property({ attribute: false }) ids: string[] = [];
  @property({ attribute: false }) tiles: ProfileTile[] = [];
  @property({ attribute: false }) catalog: PackageCatalog | null = null;
  @property({ attribute: false }) usage: PackageUsage | null = null;
  @property({ attribute: false }) onChanged: () => void = () => undefined;
  @state() private profile = '';
  @state() private busy: string | null = null;
  @state() private msg: string | null = null;
  @state() private error: string | null = null;

  private async setMode(name: string, mode: Mode3): Promise<void> {
    this.busy = name;
    [this.msg, this.error] = [null, null];
    try {
      const cur = await conductor.doc<ProfileDoc>('profiles', name);
      const next: ProfileDoc = { ...cur.doc, library: mode };
      const chk = await conductor.check<ProfileDoc>('profiles', name, next);
      if (chk.errors.length) {
        this.error = `${name}: ${chk.errors.join('; ')}`;
        return;
      }
      const ok = await confirmDialog(`Library ${mode} for ${name}?`, [`With ${mode}, ${libraryModeWords(mode)}.`, html`The server checked the change (live v${cur.version}):${diffList(chk.changes)}`, `Saving makes a new version of the profile with the note “library to ${mode}”.`], 'Save the new version');
      if (!ok) return;
      const r = await withToken(() => conductor.put<ProfileDoc>('profiles', name, next, `library to ${mode}`, cur.version));
      this.msg = `${name}: library ${mode}, version ${r.version}`;
      this.onChanged();
    } catch (e) {
      this.error = describeError(e).message;
    } finally {
      this.busy = null;
    }
  }

  render() {
    const cat = this.catalog;
    if (!cat || !this.usage) return skeleton(5);
    const lib = this.usage.library;
    const skills = Object.values(cat.packages).filter((e) => e.category === 'library').length;
    const picksBy = (p: string) => lib.rows.filter((r) => r.profile === p && r.chosen.length).length;
    const names = Object.keys(cat.library.profiles).sort();
    return html`
      <section class="panel" data-lib-profiles>
        <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">Retrieval per profile <span class="muted">· ${skills} library skills</span></h2>
          <p class="panel-sub">On a new user message, and on a low-quality or stuck outcome, the router searches the library with the message, asks the package question among the best matches and, with <strong>on</strong>, puts the skill that fits at the prompt's tail. <strong>shadow</strong> only journals the pick. Judge the picks below before a profile goes on.</p></div></header>
        ${this.error ? html`<p class="err-ink small" role="alert" data-lib-error>${this.error}</p>` : nothing}
        ${this.msg ? html`<p class="status small" role="status" data-lib-saved>${icon('check')} ${this.msg}</p>` : nothing}
        <div class="lib-profiles">
          ${names.map((n) => {
            const tile = this.tiles.find((t) => t.name === n);
            const mode = cat.library.profiles[n];
            return html`<div class="lib-profile" data-lib-profile=${n}>
              <span class="lib-profile-name"><strong>${n}</strong>${tile?.locked ? html` <span class="badge badge-lock">locked</span>` : nothing}<span class="muted small">${tile?.clients.join(', ') || 'no client'}${tile ? ` · profile ${tile.mode}` : ''}</span></span>
              ${modeSeg(['off', 'shadow', 'on'] as const, mode, (v) => void this.setMode(n, v), { small: true, label: `library of ${n}`, disabled: !!tile?.locked || this.busy === n })}
              <span class="muted small num">${picksBy(n)} pick${picksBy(n) === 1 ? '' : 's'} in the rows below</span>
            </div>`;
          })}
        </div>
      </section>
      <section class="panel" data-lib-shadow>
        <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">What the harness picked</h2>
          <p class="panel-sub">The router's retrieval in the journal (extra.library), newest first: the candidates the search found, the skill the package question picked, and whether it went in (on) or would have (shadow). A pick opens the skill.</p></div>
          <div class="panel-tools"><label class="field"><span>profile</span><select data-lib-filter @change=${(e: Event) => (this.profile = (e.target as HTMLSelectElement).value)}>
            <option value="" ?selected=${!this.profile}>all</option>${names.map((n) => html`<option value=${n} ?selected=${this.profile === n}>${n}</option>`)}
          </select></label></div></header>
        <div class="stats" data-lib-tiles>
          ${statTile({ label: 'messages searched', value: exact(lib.searched.day), unit: 'in 24 h', sub: `${exact(lib.searched.week)} in 7 days` })}
          ${statTile({ label: 'with a pick', value: exact(lib.picked.day), unit: 'in 24 h', sub: `${exact(lib.picked.week)} in 7 days` })}
          ${statTile({ label: 'pick rate', value: lib.searched.week ? `${Math.round((100 * lib.picked.week) / lib.searched.week)}` : '—', unit: lib.searched.week ? '%' : '', sub: '7 days' })}
          ${statTile({ label: 'most picked', value: lib.top[0]?.name ?? '—', sub: lib.top.length ? lib.top.slice(0, 3).map((t) => `${t.name} ${t.picks}`).join(' · ') : 'no pick yet' })}
        </div>
        ${this.rows(retrievalsOf(lib.rows, this.profile))}
      </section>`;
  }

  private rows(rows: Retrieval[]): TemplateResult {
    if (!rows.length) return emptyState('No retrieval in the journal', 'A profile with the library in shadow or on journals each search here; with every profile off nothing is searched.');
    const open = (name: string) => setParams({ ptab: null, pkg: name, pview: null });
    return html`<div class="lib-rows" data-lib-rows>${rows.map(
      (r) => html`<div class="lib-row" data-lib-row=${r.id}>
        <div class="lib-row-head">
          <span class="num small">${dateTimeShort(new Date(r.ts_ms).toISOString())}</span>
          <span class="badge ${r.mode === 'on' ? 'badge-on' : 'badge-shadow'}">${r.mode ?? '—'}</span>
          ${clsChip(r.class_id, this.ids)}
          <span class="muted small">${r.profile ?? '—'} · ${r.trigger ?? '—'}${r.query_chars != null ? ` · ${exact(r.query_chars)} characters searched` : ''}${r.cached ? ' · asked before' : ''}</span>
        </div>
        <div class="lib-cands">${r.candidates.length
          ? r.candidates.map((c) => html`<span class="lib-cand ${c.name && r.chosen.includes(c.name) ? 'is-chosen' : ''}">${c.name}<span class="num muted">${c.score != null ? c.score.toFixed(2) : ''}</span></span>`)
          : html`<span class="muted small">no candidate shares a word with the message</span>`}</div>
        <div class="lib-pick small">${r.chosen.length
          ? html`${r.mode === 'on' && r.applied?.length ? 'went in:' : 'would have gone in:'} ${r.chosen.map((n) => html`<button class="btn btn-sm lib-open" data-lib-open=${n} @click=${() => open(n)}>${n}</button>`)} <span class="muted">(${r.how ?? '—'})</span>`
          : html`<span class="muted">no pick (${(r.how ?? 'none').replace(/_/g, ' ')})</span>`}</div>
      </div>`,
    )}</div>`;
  }
}
