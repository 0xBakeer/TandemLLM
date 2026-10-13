// Conductor · Memory (VIS-48): the session transcripts the engine keeps on the box (SRV-137). The
// Sessions tab lists them by their last request -- client, profile, project, turns, acts, the last
// checker, the size, and a keep switch the 04:00 prune honours -- and opens one transcript: its
// blocks as cards, 200 a page (the newest page first), the user's and tool's messages, the answer
// or its calls, the hidden notes and acts highlighted with a jump to their journal row, the
// compaction blocks folded, an export as Markdown or JSON. Below, the transcript settings (what is
// written, how long it is kept, where tool output is clipped), a versioned settings document like
// the others. Every memory route follows the admin rule: open while no password is set on the
// server (AU1, 2026-10-06), the admin token once one is. On a phone the list is cards and the viewer takes the whole width.

import { html, nothing, type PropertyValues, type TemplateResult } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import { describeError } from '../../api/client';
import { conductor, type DecisionRow, type MemoryBlock, type MemorySession, type MemorySessions, type MemorySessionRow, type MonitorEvent, type SettingsDoc, type TranscriptScope } from '../../api/conductor';
import { setIn } from '../../lib/conductor';
import { bytes, dateTimeShort, exact } from '../../lib/format';
import { parseBlock, sessionLine, shortKey, transcriptSettings, type RuntimeLine, type Section } from '../../lib/memory';
import { setParams } from '../../lib/router';
import { LightElement } from '../../ui/base';
import { emptyState, icon, panel, skeleton } from '../../ui/bits';
import { clsChip, DocEditor, errorsList, modeSeg, saveBar } from './shared';
import './disk';
import './notes';

const SCOPE_WORDS: Record<TranscriptScope, string> = {
  tools: 'requests with tools in full (and compactions); chats without tools as counters only',
  all: 'every request in full',
  none: 'nothing',
};

@customElement('qse-cond-memory')
export class QseCondMemory extends LightElement {
  @property({ attribute: false }) ids: string[] = [];
  @property({ attribute: false }) selected: string | null = null;
  @property({ attribute: false }) page: number | null = null;
  /** the tab: the transcripts (null), the memory notes ('notes', 2026-10-05) or the disk tier
   *  ('disk', C1) */
  @property({ attribute: false }) tab: string | null = null;
  /** the Notes tab's open note and tree node (`note`, `nnode` in the address) */
  @property({ attribute: false }) note: string | null = null;
  @property({ attribute: false }) nnode: string | null = null;
  @state() private list: MemorySessions | null = null;
  @state() private one: MemorySession | null = null;
  @state() private error: string | null = null;
  @state() private oneError: string | null = null;
  @state() private query = '';
  @state() private busy: string | null = null;
  @state() private row: DecisionRow | null = null;
  @state() private ev: MonitorEvent | null = null;
  @state() private rowError: string | null = null;
  private settings = new DocEditor<SettingsDoc>(this, 'settings');
  private loadedFor: string | null = null;

  connectedCallback(): void {
    super.connectedCallback();
    if (this.tab === 'disk' || this.tab === 'notes') return;
    void this.loadList();
    void this.settings.load();
  }

  protected updated(changed: PropertyValues): void {
    // back from the disk tab: the transcripts load the first time they are shown
    const away = (t: unknown) => t === 'disk' || t === 'notes';
    if (away(changed.get('tab')) && !away(this.tab) && !this.list && !this.error) {
      void this.loadList();
      if (!this.settings.draft) void this.settings.load();
    }
  }

  protected willUpdate(): void {
    const want = this.selected ? `${this.selected}#${this.page ?? ''}` : null;
    if (want !== this.loadedFor) {
      this.loadedFor = want;
      this.one = null;
      this.oneError = null;
      if (this.selected) void this.loadOne(this.selected, this.page);
    }
  }

  private async loadList() {
    try {
      this.list = await conductor.memorySessions();
      this.error = null;
    } catch (e) {
      this.error = describeError(e).message;
    }
  }

  private async loadOne(key: string, page: number | null) {
    try {
      const r = await conductor.memorySession(key, page);
      if (key === this.selected) this.one = r;
    } catch (e) {
      if (key === this.selected) this.oneError = describeError(e).message;
    }
  }

  private async keep(s: MemorySessionRow, keep: boolean) {
    if (!s.session) return;
    this.busy = s.dir;
    try {
      await conductor.memoryKeep(s.session!, keep);
      await this.loadList();
      if (this.one && this.one.session.session === s.session) this.one = { ...this.one, session: { ...this.one.session, keep } };
    } catch (e) {
      this.error = describeError(e).message;
    } finally {
      this.busy = null;
    }
  }

  private async download(key: string, format: 'md' | 'json') {
    try {
      const blob = await conductor.memoryExport(key, format);
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `transcript-${key.replace(/[^a-z0-9]+/gi, '-')}.${format}`;
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    } catch (e) {
      this.oneError = describeError(e).message;
    }
  }

  /** The journal row a hidden note is in, in the drawer of its kind: a router decision or a
   *  monitor event (the journal answers both by id). */
  private async openRow(id: string) {
    this.rowError = null;
    try {
      const r = (await conductor.decision(id)).decision as DecisionRow;
      if ((r as { kind?: string }).kind === 'monitor') this.ev = r as unknown as MonitorEvent;
      else this.row = r;
    } catch (e) {
      this.rowError = `journal row ${id}: ${describeError(e).message} (the journal keeps 30 days)`;
    }
  }

  render() {
    const disk = this.tab === 'disk';
    const cur = this.tab === 'disk' || this.tab === 'notes' ? this.tab : null;
    const tab = (id: string | null, label: string) =>
      html`<a class="cond-tab ${id === cur ? 'is-on' : ''}" href="#" aria-current=${id === cur ? 'page' : nothing} data-mem-tab=${id ?? 'sessions'} @click=${(e: Event) => {
        e.preventDefault();
        setParams({ mtab: id, memory: null, mpage: null, note: null, nnode: null });
      }}>${label}</a>`;
    const tabs = html`<nav class="mem-tabs" aria-label="memory tabs">${tab(null, 'Sessions')}${tab('notes', 'Notes')}${tab('disk', 'Disk cache')}</nav>`;
    if (disk) return html`${tabs}<qse-cond-disk></qse-cond-disk>`;
    if (cur === 'notes') return html`${tabs}<qse-cond-notes .selected=${this.note} .node=${this.nnode}></qse-cond-notes>`;
    return html`
      ${tabs}
      ${this.selected ? this.renderOne(this.selected) : nothing}
      ${this.renderList()}
      ${this.renderSettings()}
      ${this.row ? html`<qse-cond-decision .row=${this.row} .ids=${this.ids} .close=${() => (this.row = null)}></qse-cond-decision>` : nothing}
      ${this.ev ? html`<qse-cond-event .ev=${this.ev} .close=${() => (this.ev = null)}></qse-cond-event>` : nothing}
    `;
  }

  private renderList(): TemplateResult {
    if (this.error && !this.list) return html`<div class="state state-error" role="alert"><div class="state-title">${icon('warn')} ${this.error}</div></div>`;
    if (!this.list) return skeleton(5);
    const w = this.list.writer;
    const q = this.query.toLowerCase();
    const rows = this.list.sessions.filter((s) => !q || `${s.session ?? s.dir} ${s.client ?? ''} ${s.project ?? ''}`.toLowerCase().includes(q));
    const when = (s: MemorySessionRow) => (s.last_ms ? dateTimeShort(new Date(s.last_ms).toISOString()) : '—');
    const keepSwitch = (s: MemorySessionRow) =>
      html`<label class="switch" title="kept sessions are exempt from the ${w.settings.retention_days}-day prune"><input type="checkbox" .checked=${s.keep} ?disabled=${!s.session || this.busy === s.dir} data-keep=${s.dir} @click=${(e: Event) => e.stopPropagation()} @change=${(e: Event) => this.keep(s, (e.target as HTMLInputElement).checked)} aria-label=${`keep ${shortKey(s.session, s.dir)}`} /> keep</label>`;
    const checker = (s: MemorySessionRow) =>
      s.checkers?.last ? html`<span class="badge ${/ PASS$/.test(s.checkers.last) ? 'badge-ok' : 'badge-warn'}">${s.checkers.last}</span>${s.checkers.PASS || s.checkers.FAIL ? html` <span class="muted small num">${s.checkers.PASS ?? 0} pass · ${s.checkers.FAIL ?? 0} fail</span>` : nothing}` : html`<span class="muted">—</span>`;
    const open = (s: MemorySessionRow) => s.session && setParams({ memory: s.session, mpage: null });
    return panel(
      'Session transcripts',
      rows.length
        ? html`<div class="table-scroll mem-table"><table class="table" data-memory-sessions>
              <thead><tr><th>last request</th><th>session</th><th>client</th><th>profile</th><th>project</th><th class="num-col">turns</th><th class="num-col">acts</th><th>checker</th><th class="num-col">size</th><th>keep</th></tr></thead>
              <tbody>${rows.map(
                (s) => html`<tr class="req-row ${s.session && s.session === this.selected ? 'is-open' : ''}" data-memory-row=${s.dir} @click=${() => open(s)}>
                  <td class="num">${when(s)}</td><td class="mono small cell-wrap">${shortKey(s.session, s.dir)}</td><td>${s.client ?? '—'}</td><td>${s.profile ?? '—'}</td>
                  <td class="small cell-wrap">${s.project ?? '—'}</td>
                  <td class="num num-col">${s.counts ? exact(s.counts.blocks) : '?'}</td><td class="num num-col">${s.counts ? exact(s.counts.acts) : '?'}</td>
                  <td>${checker(s)}</td><td class="num num-col">${bytes(s.bytes ?? 0)}</td><td>${keepSwitch(s)}</td></tr>`,
              )}</tbody></table></div>
            <div class="mem-cards" role="list">${rows.map(
              (s) => html`<div class="mem-card ${s.session && s.session === this.selected ? 'is-open' : ''}" role="listitem" data-memory-card=${s.dir} @click=${() => open(s)}>
                <div class="mem-card-head"><span class="mono small">${shortKey(s.session, s.dir)}</span>${keepSwitch(s)}</div>
                <div class="small">${when(s)} · ${s.client ?? '—'} · ${s.profile ?? '—'}</div>
                <div class="small muted">${sessionLine(s)}</div>
                ${s.project ? html`<div class="small muted cell-wrap">${s.project}</div>` : nothing}
                <div>${checker(s)}</div>
              </div>`,
            )}</div>`
        : emptyState('No transcripts yet', w.refused ? `The memory directory ${w.root} is inside a git work tree, so nothing is written there.` : `TandemLLM requests with tools are written to ${w.root} when their stream closes; chats without tools are counted only.`),
      {
        sub: html`<span data-writer>${exact(w.blocks)} block${w.blocks === 1 ? '' : 's'} written since the start · ${exact(w.redactions)} secret${w.redactions === 1 ? '' : 's'} redacted · ${exact(w.failed)} failed · ${exact(w.dropped)} dropped · ${bytes(w.bytes)} · <span class="mono">${w.root}</span></span>`,
        tools: html`<label class="field field-search">${icon('search')}<input placeholder="search a session, client or project" .value=${this.query} @input=${(e: Event) => (this.query = (e.target as HTMLInputElement).value)} aria-label="search transcripts" /></label>
          <button class="btn btn-sm" @click=${() => this.loadList()} aria-label="reload the list">${icon('refresh')}</button>`,
        id: 'cond-memory',
      },
    );
  }

  private renderOne(key: string): TemplateResult {
    if (this.oneError) return html`<div class="state state-error" role="alert" style="margin-bottom:16px"><div class="state-title">${icon('warn')} ${this.oneError}</div></div>`;
    const one = this.one;
    if (!one) return skeleton(4);
    const s = one.session;
    const pages = one.pages ?? 1;
    const page = one.page ?? 1;
    const pager = pages > 1
      ? html`<div class="toolbar mem-pager" data-pager><button class="btn btn-sm" ?disabled=${page <= 1} @click=${() => setParams({ mpage: String(page - 1) })}>Older</button>
          <span class="small num">page ${page} of ${pages} · ${exact(one.total ?? one.blocks.length)} blocks, ${one.per_page} a page</span>
          <button class="btn btn-sm" ?disabled=${page >= pages} @click=${() => setParams({ mpage: String(page + 1) })}>Newer</button></div>`
      : nothing;
    return html`<section class="panel" style="margin-bottom:16px" data-transcript=${key}>
      <header class="panel-head"><div class="panel-titles"><h2 class="panel-title mono">${shortKey(s.session, s.dir)}${s.keep ? html` <span class="badge badge-on">kept</span>` : nothing}</h2>
        <p class="panel-sub">${s.client ?? '—'} · ${s.profile ?? '—'}${s.project ? ` · ${s.project}` : ''} · ${sessionLine(s)}${s.first_ms ? ` · since ${dateTimeShort(new Date(s.first_ms).toISOString())}` : ''}</p></div>
        <div class="panel-tools">
          <button class="btn btn-sm" @click=${() => this.download(key, 'md')} data-mem-export="md">${icon('down')} Markdown</button>
          <button class="btn btn-sm" @click=${() => this.download(key, 'json')} data-mem-export="json">${icon('down')} JSON</button>
          <a class="btn btn-sm" href=${`#/conductor?screen=sessions&session=${encodeURIComponent(key)}`} data-mem-timeline>Timeline</a>
          <button class="btn btn-ghost btn-icon" aria-label="Close the transcript" @click=${() => setParams({ memory: null, mpage: null })}>${icon('x')}</button></div></header>
      ${this.rowError ? html`<p class="warn-ink small" role="alert">${icon('warn')} ${this.rowError}</p>` : nothing}
      ${pager}
      ${one.blocks.length ? html`<div class="mem-blocks">${one.blocks.map((b) => this.renderBlock(b))}</div>` : emptyState('No blocks', 'This session was counted but not written: its requests carried no tools.')}
      ${pager}
    </section>`;
  }

  private renderBlock(b: MemoryBlock): TemplateResult {
    const p = parseBlock(b.text);
    const hidden = p.runtime.filter((r) => r.kind === 'hidden').length;
    const acts = p.runtime.find((r) => r.kind === 'acts');
    const chips = html`<span class="mem-turn num">turn ${b.turn ?? '?'}</span>
      <span class="muted small num">${b.time ?? ''}</span>
      ${b.class_id ? clsChip(b.class_id, this.ids) : nothing}
      ${b.effort ? html`<span class="badge">${b.effort}</span>` : nothing}
      ${b.reasoning_tokens ? html`<span class="muted small num">${exact(b.reasoning_tokens)} reasoning tokens</span>` : nothing}
      ${hidden ? html`<span class="badge badge-shadow" data-hidden-count=${hidden}>${hidden} hidden</span>` : nothing}
      ${acts ? html`<span class="badge badge-warn">${acts.text.replace(/^acts: /, 'acts ')}</span>` : nothing}`;
    if (b.compaction) {
      return html`<details class="mem-block is-compaction" data-block=${b.turn ?? ''} data-compaction>
        <summary><span class="mem-block-head">${chips}<span class="badge">compaction</span><span class="small">${p.compaction?.replace(/^compaction: /, '') ?? ''}</span></span></summary>
        ${p.sections.map((x) => this.renderSection(x))}${this.renderRuntime(p.runtime)}
      </details>`;
    }
    return html`<article class="mem-block" data-block=${b.turn ?? ''}>
      <div class="mem-block-head">${chips}</div>
      ${p.sections.map((x) => this.renderSection(x))}
      ${this.renderRuntime(p.runtime)}
    </article>`;
  }

  private renderSection(x: Section): TemplateResult {
    const long = x.body.length > 600 || x.body.split('\n').length > 12;
    const body = html`<pre class="mem-text">${x.body}</pre>`;
    if (x.kind === 'tool_result' || (x.kind === 'tool_call' && long) || x.kind === 'reasoning') {
      return html`<details class="mem-sec mem-${x.kind}" ?open=${!long && x.kind !== 'reasoning'} data-sec-kind=${x.kind}>
        <summary class="small">${x.title}${x.clipped ? html` <span class="badge badge-warn">clipped</span>` : nothing}</summary>${body}</details>`;
    }
    return html`<div class="mem-sec mem-${x.kind}" data-sec-kind=${x.kind}><div class="small muted">${x.title}</div>${body}</div>`;
  }

  private renderRuntime(lines: RuntimeLine[]): TemplateResult | typeof nothing {
    if (!lines.length) return nothing;
    return html`<ul class="mem-runtime" aria-label="what the runtime did">${lines.map(
      (r) => html`<li class="mem-rt mem-rt-${r.kind} ${r.verdict ? `is-${r.verdict.toLowerCase()}` : ''}" data-rt=${r.kind}>
        ${r.kind === 'hidden' ? icon('info') : nothing}<span>${r.text.replace(/ · journal [A-Za-z0-9_-]+$/, '')}</span>
        ${r.journal ? html`<button class="btn btn-ghost btn-sm" @click=${() => this.openRow(r.journal!)} data-journal=${r.journal} aria-label=${`open journal row ${r.journal}`}>journal row</button>` : nothing}
      </li>`,
    )}</ul>`;
  }

  private renderSettings(): TemplateResult | typeof nothing {
    const ed = this.settings;
    if (!ed.draft) return ed.loadError ? html`<p class="err-ink small">${ed.loadError}</p>` : nothing;
    const t = transcriptSettings(ed.draft);
    const upd = (k: keyof typeof t, v: unknown) => ed.set(setIn(ed.draft!, ['memory', 'transcripts'], { ...t, [k]: v }));
    const num = (k: 'retention_days' | 'clip_chars' | 'loop_chars', label: string, min: number, max: number) =>
      html`<label class="field"><span>${label}</span><input type="number" min=${min} max=${max} .value=${String(t[k])} data-mem-setting=${k} @input=${(e: Event) => upd(k, Math.round(Number((e.target as HTMLInputElement).value)))} /></label>`;
    return html`<details class="sec" style="margin-top:16px" data-sec="transcripts">
      <summary>Transcript settings <span class="sec-sub">what the engine writes to the box, how long it keeps it, where tool output is clipped</span></summary>
      <div class="sec-body">
        <div class="toolbar" style="margin:0" data-mem-scope><span class="muted small">write</span>${modeSeg(['tools', 'all', 'none'] as TranscriptScope[], t.scope, (v) => upd('scope', v), { small: true, label: 'transcript scope' })}<span class="small">${SCOPE_WORDS[t.scope]}</span></div>
        <div class="form-grid">
          ${num('retention_days', 'keep a session for (days; kept sessions are exempt)', 1, 400)}
          ${num('clip_chars', 'clip a tool result or a call at (characters)', 200, 100000)}
          ${num('loop_chars', 'a loop the D1 guard cut keeps its first and last (characters)', 0, 4000)}
        </div>
        <p class="muted small">Every block passes the secret-pattern refusal before it is written: a key, token or password becomes <code>[redacted:kind]</code> and the journal counts it, never the value. There is no switch for it. The files are <code>0600</code> under the engine's home on the box, never in a repository, and the ledger's backup never holds them. By design, chats without tools are counters only, unless the scope is widened here.</p>
        ${errorsList(ed)}
        ${saveBar(ed, 'the settings')}
      </div>
    </details>`;
  }
}
