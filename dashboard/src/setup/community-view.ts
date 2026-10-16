// <qse-community>: the "Community weights" section under our own profiles on the weights step
// (CONTRACT.md 2026-10-13 and its three additions). One card per community pack: the maker on top
// (GitHub avatar, @login, the networks the entry lists), the pack's name and what it is a fine-tune
// of, the readout (format, bits a weight, download), the modalities it holds (greyed where
// TandemLLM cannot run one yet, with the reason), the drafters, the speeds we measured when it is
// verified, the license, the Hugging Face repository as owner/name (linked) and Use, which selects
// the pack like one of our radio cards: the memory bar, Review and Finish are the same flow. A pack
// one of the model's own profiles serves (VIS-175: turboderp's Flash-Next branches) has a card here
// too, and its Use selects that profile.
//
// The list is fetched from GitHub when the step opens and on Refresh (the parent does the request;
// this element shows the state). Every string of an entry is untrusted: lit renders it as text, and
// community.ts makes a link only from an https URL or a handle that passes its network's check.

import { html, nothing, svg, type TemplateResult } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import type { CommunitySync, SetupDevice, SetupModel } from '../api/types';
import { ago, dateLong } from '../lib/format';
import { LightElement } from '../ui/base';
import { icon } from '../ui/bits';
import { boxState, cardsOf, countText, DOCS_URL, httpsUrl, isSortKey, searchCards, SORT_KEYS, SORT_LABEL, sortCards, type CommunityCard, type SortKey } from './community';
import { S } from './strings';

const C = S.community;

/** the section's own glyphs, drawn like ui/bits.ts (20 by 20, currentColor); the networks' marks
 *  are reduced to their simplest shapes so they read at 16 px */
const GLYPH: Record<string, TemplateResult> = {
  text: svg`<path d="M4 5.5h12M4 9.5h12M4 13.5h7" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/>`,
  mic: svg`<rect x="7.25" y="3" width="5.5" height="9" rx="2.75" fill="none" stroke="currentColor" stroke-width="1.6"/><path d="M4.5 10a5.5 5.5 0 0 0 11 0M10 15.5v2M7.5 17.5h5" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>`,
  video: svg`<rect x="2.5" y="5" width="10.5" height="10" rx="1.5" fill="none" stroke="currentColor" stroke-width="1.6"/><path d="m13 9 4.5-2.5v7L13 11z" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/>`,
  drafters: svg`<circle cx="4.5" cy="10" r="1.8" fill="currentColor"/><circle cx="10" cy="10" r="1.8" fill="currentColor" opacity=".7"/><circle cx="15.5" cy="10" r="1.8" fill="currentColor" opacity=".45"/><path d="M6.3 10h1.9M11.8 10h1.9" stroke="currentColor" stroke-width="1.4"/>`,
  x: svg`<path d="M4 3.5h3.6l8.4 13H12.4z" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/><path d="M15.5 3.5 4.5 16.5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>`,
  huggingface: svg`<circle cx="10" cy="10.5" r="6.5" fill="none" stroke="currentColor" stroke-width="1.6"/><circle cx="7.6" cy="9" r="1" fill="currentColor"/><circle cx="12.4" cy="9" r="1" fill="currentColor"/><path d="M7 12.2c.8 1.2 1.8 1.8 3 1.8s2.2-.6 3-1.8" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/><path d="M4.2 7.2 3 6.2M15.8 7.2 17 6.2" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>`,
  bluesky: svg`<path d="M10 8.6C9 6.8 6.8 4.2 4.6 3.7 3.3 3.5 2.8 4.3 2.8 5.4c0 1.6.6 4.3 1.9 5 .9.4 2 .2 2.7 0-2 .5-2.9 1.4-2.1 2.9.9 1.8 3.2 2.5 4.7-.4 1.5 2.9 3.8 2.2 4.7.4.8-1.5-.1-2.4-2.1-2.9.7.2 1.8.4 2.7 0 1.3-.7 1.9-3.4 1.9-5 0-1.1-.5-1.9-1.8-1.7C13.2 4.2 11 6.8 10 8.6Z" fill="currentColor"/>`,
  mastodon: svg`<path d="M4 15.5c-.9-1-1.5-2.6-1.5-5V8.3C2.5 5 4.6 3.5 10 3.5s7.5 1.5 7.5 4.8v1.5c0 2.6-1.4 3.9-4.2 4.1-1.9.2-3.5 0-4.3-.3v.9c0 1.4 2.3 1.5 5 1" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/><path d="M6.3 11.8V8.4c0-1.1.6-1.7 1.6-1.7s1.6.6 1.6 1.7v1.9M13.7 11.8V8.4c0-1.1-.6-1.7-1.6-1.7s-1.6.6-1.6 1.7v1.9" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>`,
  website: svg`<circle cx="10" cy="10" r="7" fill="none" stroke="currentColor" stroke-width="1.6"/><path d="M3 10h14M10 3c2.2 2.1 3.3 4.4 3.3 7s-1.1 4.9-3.3 7c-2.2-2.1-3.3-4.4-3.3-7S7.8 5.1 10 3Z" fill="none" stroke="currentColor" stroke-width="1.4"/>`,
};
const glyph = (name: string, size = 16) => html`<svg class="icon" viewBox="0 0 20 20" aria-hidden="true" width=${size} height=${size}>${GLYPH[name]}</svg>`;
const modalityGlyph = (id: string) => (id === 'text' ? glyph('text') : id === 'image' ? icon('image') : id === 'audio' ? glyph('mic') : glyph('video'));
const upper = (f: string) => f.toUpperCase();

@customElement('qse-community')
export class QseCommunity extends LightElement {
  @property({ attribute: false }) model: SetupModel | null = null;
  /** where the list came from (GET models/{id} community_sync, then each sync's answer) */
  @property({ attribute: false }) sync: CommunitySync | null = null;
  /** the selected profile id (community/<slug> when a pack is chosen) */
  @property({ attribute: false }) selected: string | null = null;
  @property({ attribute: false }) device: SetupDevice | null = null;
  /** the parent's sync with GitHub runs: the cards dim a little, the head says so */
  @property({ type: Boolean }) syncing = false;
  @property({ attribute: false }) onUse: (id: string) => void = () => {};
  @property({ attribute: false }) onRefresh: () => void = () => {};

  @state() private query = '';
  @state() private sort: SortKey = 'newest';
  @state() private droppedOpen = false;
  /** logins whose avatar did not load: the initials stand in */
  @state() private noAvatar = new Set<string>();

  render() {
    const m = this.model;
    if (!m) return nothing;
    const all = cardsOf(m);
    const sync = this.sync;
    if (!all.length && !sync) return nothing; // a server without community weights
    const shown = sortCards(searchCards(all, this.query), this.sort);
    const docs = DOCS_URL;
    const add = httpsUrl(sync?.add_url) ?? docs;
    return html`<section class="cw" aria-labelledby="cw-h" data-community data-syncing=${this.syncing ? '' : nothing}>
      <header class="cw-head">
        <div class="cw-title">
          <h2 class="cw-h" id="cw-h">${C.heading}</h2>
          ${this.renderSource()}
        </div>
        <p class="cw-intro">${C.intro} <a href=${docs} target="_blank" rel="noopener noreferrer">${C.addYoursHow} ${icon('external')}</a></p>
        ${all.length
          ? html`<div class="cw-tools">
              <label class="cw-search">
                ${icon('search')}
                <input type="search" placeholder=${C.search} aria-label=${C.searchLabel} .value=${this.query} autocomplete="off" autocapitalize="off" spellcheck="false" @input=${(e: Event) => (this.query = (e.target as HTMLInputElement).value)} />
              </label>
              <label class="cw-sort"><span class="muted small">Sort</span>
                <select class="sel" aria-label=${C.sortLabel} @change=${(e: Event) => this.setSort((e.target as HTMLSelectElement).value)}>${SORT_KEYS.map((k) => html`<option value=${k} ?selected=${this.sort === k}>${SORT_LABEL[k]}</option>`)}</select>
              </label>
              <span class="cw-count num muted small" data-count>${countText(shown.length, all.length)}</span>
              <a class="btn btn-sm cw-add" href=${add} target="_blank" rel="noopener noreferrer">${icon('plus')} ${C.addYours}</a>
            </div>`
          : nothing}
      </header>
      ${sync?.dropped.length ? this.renderDropped(sync.dropped) : nothing}
      ${!all.length
        ? html`<div class="cw-empty" data-community-empty>
            <p class="cw-empty-t">${C.empty}</p>
            <p class="muted">${C.emptyBody}</p>
            <p class="cw-empty-acts"><a class="btn btn-primary btn-sm" href=${add} target="_blank" rel="noopener noreferrer">${icon('plus')} ${C.addYours}</a> <a class="btn btn-sm" href=${docs} target="_blank" rel="noopener noreferrer">${C.addYoursHow} ${icon('external')}</a></p>
          </div>`
        : shown.length
          ? html`<div class="cw-grid" role="list">${shown.map((c) => this.renderCard(c))}</div>`
          : html`<div class="cw-empty" data-community-nomatch role="status"><p class="cw-empty-t">${C.noMatch(this.query.trim())}</p><p><button class="btn btn-sm" @click=${() => (this.query = '')}>${icon('x')} ${C.clearSearch}</button></p></div>`}
    </section>`;
  }

  private setSort(v: string): void {
    if (isSortKey(v)) this.sort = v;
  }

  /** where the list came from, and Refresh */
  private renderSource(): TemplateResult {
    const idx = this.sync;
    const src = idx?.source ?? 'bundled';
    // a sync that just finished says so; later the usual "12 m ago"
    const when = (iso: string | null | undefined) => (iso && Date.now() - Date.parse(iso) < 10_000 ? 'just now' : ago(iso));
    const text = this.syncing ? C.syncing : src === 'github' ? C.fromGithub(when(idx?.fetched_at)) : src === 'cache' ? C.fromCache(when(idx?.fetched_at)) : C.bundled;
    return html`<div class="cw-src" data-source=${this.syncing ? 'syncing' : src}>
      <span class="cw-src-dot" aria-hidden="true"></span>
      <span class="cw-src-text small" role="status">${text}</span>
      <button class="btn btn-ghost btn-sm cw-refresh" ?disabled=${this.syncing} aria-busy=${this.syncing} @click=${() => this.onRefresh()}>${icon('refresh')} ${C.refresh}</button>
      ${idx?.error && !this.syncing ? html`<span class="cw-src-err small" role="alert">${icon('warn')} ${C.syncFailed(idx.error.replace(/\.?$/, '.'))}</span>` : nothing}
      ${idx?.note && !idx.error && !this.syncing ? html`<span class="cw-src-note small muted">${idx.note}</span>` : nothing}
    </div>`;
  }

  private renderDropped(dropped: { path: string; reason: string }[]): TemplateResult {
    return html`<div class="cw-dropped small" data-dropped>
      <p>${icon('info')} <span>${C.dropped(dropped.length)}</span> <button class="linkish" aria-expanded=${this.droppedOpen} @click=${() => (this.droppedOpen = !this.droppedOpen)}>${this.droppedOpen ? C.droppedHide : C.droppedDetails}</button></p>
      ${this.droppedOpen ? html`<ul>${dropped.map((d) => html`<li><code>${d.path}</code> <span class="muted">${d.reason}</span></li>`)}</ul>` : nothing}
    </div>`;
  }

  private renderCard(c: CommunityCard): TemplateResult {
    const on = this.selected === c.id;
    const box = boxState(c.profile, this.device);
    const broken = c.status === 'broken';
    const a = c.author;
    const avatar = a.avatar && !this.noAvatar.has(a.login);
    return html`<article class="ccard ${on ? 'is-selected' : ''} ${broken ? 'is-broken' : ''}" role="listitem" data-community-card=${c.slug} data-field=${on ? 'weights.profile' : nothing} aria-labelledby=${`cc-${c.slug}`}>
      <header class="ccard-maker">
        <span class="ccard-avatar" aria-hidden="true">
          ${avatar
            ? html`<img src=${a.avatar!} alt="" width="40" height="40" loading="lazy" decoding="async" referrerpolicy="no-referrer" @error=${() => this.avatarFailed(a.login)} />`
            : html`<span class="ccard-initials">${a.initials}</span>`}
        </span>
        <span class="ccard-who">
          <span class="ccard-handles">
            ${a.url ? html`<a class="ccard-login" href=${a.url} target="_blank" rel="noopener noreferrer" title=${C.github(a.login)}>@${a.login}</a>` : html`<span class="ccard-login">@${a.login}</span>`}
            ${a.social.length ? html`<span class="ccard-social">${a.social.map((s) => html`<a class="ccard-net tip-host" href=${s.href} target="_blank" rel="noopener noreferrer" aria-label=${s.label} data-tip=${s.label} data-net=${s.kind}>${glyph(s.kind)}</a>`)}</span>` : nothing}
          </span>
          ${a.name && a.name.toLowerCase() !== a.login.toLowerCase() ? html`<span class="ccard-author">${a.name}</span>` : nothing}
        </span>
        <span class="ccard-badges">
          ${box === 'running' ? html`<span class="sbadge sbadge-now">${S.activeNow}</span>` : nothing}
          ${this.statusBadge(c)}
        </span>
      </header>
      <h3 class="ccard-name" id=${`cc-${c.slug}`}>${c.name}</h3>
      <p class="ccard-base">${c.finetuneOf ? html`${C.finetuneOf} ${c.finetuneUrl ? html`<a href=${c.finetuneUrl} target="_blank" rel="noopener noreferrer">${c.finetuneOf}</a>` : html`<span>${c.finetuneOf}</span>`}` : C.stock}</p>
      ${c.description ? html`<p class="ccard-desc">${c.description}</p>` : nothing}
      <dl class="ccard-readout num">
        <div><dt>format</dt><dd>${upper(c.format)}</dd></div>
        <div><dt>${C.bpw}</dt><dd>${c.bpw != null ? c.bpw.toFixed(2) : '–'}</dd></div>
        <div><dt>${C.download}</dt><dd>${box === 'download' ? `${c.sizeGb.toFixed(1)} GB` : S.onDisk}</dd></div>
      </dl>
      <ul class="ccard-modes" aria-label="Modalities">
        ${c.modalities.map((md) => html`<li class="ccard-mode tip-host ${md.runs ? 'is-on' : 'is-off'}" tabindex="0" data-tip=${md.why} aria-label=${md.why} data-modality=${md.id}>${modalityGlyph(md.id)}<span>${md.label}</span></li>`)}
      </ul>
      <p class="ccard-drafters">${glyph('drafters')} <span>${c.drafters}</span></p>
      ${c.verified && c.verified.speeds.length
        ? html`<div class="ccard-speed" data-speed>
            ${c.verified.speeds.map((s) => html`<span class="ccard-sp"><span class="ccard-sp-n num">${s.tok_s.toFixed(1)}</span><span class="ccard-sp-k">${s.task}</span></span>`)}
            <span class="ccard-sp-u"><span class="num">tok/s</span><span class="ccard-sp-note">${C.speedNote(c.verified.device, c.verified.build, dateLong(c.verified.date))}</span></span>
          </div>`
        : html`<p class="ccard-speed-none muted small">${C.notMeasured}</p>`}
      <footer class="ccard-foot">
        <span class="ccard-links">
          ${c.licenseUrl
            ? html`<a class="sbadge ccard-license" href=${c.licenseUrl} target="_blank" rel="noopener noreferrer" title=${`${C.license}: ${c.license}`}>${c.license}</a>`
            : html`<span class="sbadge ccard-license" title=${`${C.license}: ${c.license}`}>${c.license}</span>`}
          ${c.hfUrl ? html`<a class="ccard-hf" href=${c.hfUrl} target="_blank" rel="noopener noreferrer" title=${C.onHf(c.hfRepo)} data-hf-repo>${glyph('huggingface')} <span class="ccard-repo">${c.hfRepo}</span> ${icon('external')}</a>` : html`<span class="ccard-hf" data-hf-repo>${glyph('huggingface')} <span class="ccard-repo">${c.hfRepo}</span></span>`}
        </span>
        <button class="btn ${on ? '' : 'btn-primary'} ccard-use" ?disabled=${broken} aria-pressed=${on} title=${broken ? C.brokenWhy : nothing} @click=${() => !on && this.onUse(c.id)}>${on ? html`${icon('check')} ${C.selected}` : C.use}</button>
      </footer>
    </article>`;
  }

  private statusBadge(c: CommunityCard): TemplateResult {
    if (c.status === 'verified' && c.verified) return html`<span class="sbadge sbadge-ok tip-host" tabindex="0" data-tip=${C.verifiedWhy(c.verified.device, c.verified.build, dateLong(c.verified.date))}>${icon('check')} ${C.verified}</span>`;
    if (c.status === 'broken') return html`<span class="sbadge sbadge-err tip-host" tabindex="0" data-tip=${C.brokenWhy}>${icon('warn')} ${C.broken}</span>`;
    return html`<span class="sbadge tip-host" tabindex="0" data-tip=${C.submittedWhy}>${C.submitted}</span>`;
  }

  private avatarFailed(login: string): void {
    const next = new Set(this.noAvatar);
    next.add(login);
    this.noAvatar = next;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'qse-community': QseCommunity;
  }
}
