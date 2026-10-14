// <qse-chat-thread>: the conversation. The Now line while a run is on (the phase in words, the
// figure, the attachment state), user bubbles with attachments and a branch pager, assistant turns
// as parts in order: a thinking block that opens while it is the only thing streaming, Markdown
// with "Open in pane" on code blocks, tool cards with the permission prompt (allow once, always,
// deny), errors with Retry, the figures line, and the queued messages above the composer.

import { html, nothing, type TemplateResult } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import { repeat } from 'lit/directives/repeat.js';
import { unsafeHTML } from 'lit/directives/unsafe-html.js';
import { bytes, exact, fixed, ms, pct } from '../lib/format';
import { renderMarkdown } from '../playground/markdown';
import { LightElement } from '../ui/base';
import { copyButton, icon } from '../ui/bits';
import type { QseWrenFigure } from '../setup/figure';
import type { FigureState } from '../setup/figure-state';
import { attachLabel, type Attachment as Attach } from './connection';
import { argsSummary, dividersByMessage, dividerWords, messageText } from './store';
import { planItems, toolLabel, type Compaction, type Conversation, type Message, type PaneRef, type Part, type ToolCallPart, type TurnStats } from './types';
import { approvedTitle, approvedWord } from './permission';
import { callMedia, callPane, loadMediaView } from './file-panes';
import { hostOf, resultsWord, searchResults } from './websearch';

const PANE_LANGS: Record<string, PaneRef['kind']> = { html: 'preview', svg: 'preview', markdown: 'markdown', md: 'markdown', diff: 'diff', patch: 'diff' };

@customElement('qse-chat-thread')
export class QseChatThread extends LightElement {
  @property({ attribute: false }) conv: Conversation | null = null;
  @property({ attribute: false }) attach: Attach | null = null;
  /** the browser's clock, ticked by the view once a second while a run is on */
  @property({ type: Number }) now = Date.now();
  @state() private editing: { id: string; text: string } | null = null;
  @state() private openCalls = new Set<string>();
  @state() private openDividers = new Set<string>();
  /** the plan's steps while the user edits them before Approve */
  @state() private planEdit: string | null = null;
  private stickToEnd = true;
  private lastDeltaAt = 0;

  updated(changed: Map<string, unknown>): void {
    if (changed.has('conv') && this.stickToEnd) {
      const el = this.querySelector<HTMLElement>('.ch-scroll');
      if (el) el.scrollTop = el.scrollHeight;
      // Wren's mouth moves on a delta, at most every 140 ms (the figure throttles itself)
      const fig = this.querySelector<QseWrenFigure>('qse-wren-figure');
      if (fig && this.conv?.run && (this.conv.run.phase === 'writing' || this.conv.run.phase === 'thinking') && performance.now() - this.lastDeltaAt > 60) {
        this.lastDeltaAt = performance.now();
        fig.blip();
      }
    }
  }

  private fire(name: string, detail: unknown): void {
    this.dispatchEvent(new CustomEvent(name, { detail, bubbles: true, composed: true }));
  }

  /** the whole thread scrolls to its end on a new message, and stays where the reader put it */
  private onScroll(e: Event): void {
    const el = e.currentTarget as HTMLElement;
    this.stickToEnd = el.scrollHeight - el.scrollTop - el.clientHeight < 48;
  }

  render() {
    const c = this.conv;
    if (!c) return html`<div class="ch-scroll"></div>`;
    const dividers = dividersByMessage(c);
    // the Now line sits in its own row above the scrolling thread: it takes its height there and
    // never covers a message or a tool card (as a sticky card inside the scroll it did)
    return html`${c.run ? html`<div class="ch-now-dock">${this.renderNow(c)}</div>` : nothing}<div class="ch-scroll" role="log" aria-live="polite" aria-relevant="additions" @scroll=${this.onScroll}>
      <div class="ch-col">
        ${repeat(
          c.messages,
          (m) => m.id,
          (m, i) => html`${(dividers.get(m.id) ?? []).map((cp) => this.renderDivider(cp))}${this.renderMessage(c, m, i)}`,
        )}
        ${c.queue.length ? html`<div class="ch-queue" aria-label="queued messages">${c.queue.map((q) => html`<div class="ch-queued" data-queued=${q.id}><span class="ch-queued-word">queued</span><span class="ch-queued-text">${q.text}</span><button type="button" class="btn btn-ghost btn-sm" title="Send now: stops the current answer, keeps its text" @click=${() => this.fire('ch-send-now', { id: q.id })}>${icon('send')} Send now</button><button type="button" class="btn btn-ghost btn-sm" aria-label="Remove from the queue" title="Remove" @click=${() => this.fire('ch-unqueue', { id: q.id })}>${icon('x')}</button></div>`)}<p class="ch-queue-hint">${c.run ? 'goes in after the current step' : 'goes in next'}</p></div>` : nothing}
      </div>
    </div>`;
  }

  /** Chat 7: where the model's history now starts. The messages above stay; the divider opens to
   *  the summary that replaced them for the model. */
  private renderDivider(cp: Compaction): TemplateResult {
    const open = this.openDividers.has(cp.id);
    const toggle = () => {
      const s = new Set(this.openDividers);
      if (open) s.delete(cp.id);
      else s.add(cp.id);
      this.openDividers = s;
    };
    return html`<div class="ch-divider" data-compaction=${cp.id} role="note">
      <button type="button" class="ch-divider-btn" aria-expanded=${open} @click=${toggle}>${icon('doc')}<span>${dividerWords(cp)}</span><span class="ch-divider-more">${open ? 'hide summary' : 'show summary'}</span></button>
      ${open ? html`<p class="ch-cp-summary">${cp.summary}</p>` : nothing}
    </div>`;
  }

  // ---- the Now line -------------------------------------------------------------------------

  private renderNow(c: Conversation): TemplateResult {
    const r = c.run!;
    const started = Date.parse(r.started_at);
    const elapsed = Math.max(0, Math.round((this.now - started) / 1000));
    const label = this.attach ? attachLabel(this.attach, this.now) : null;
    const fig: FigureState = r.phase === 'tool' ? 'acting' : r.phase === 'permission' ? 'listening' : r.phase === 'thinking' || r.phase === 'prefill' || r.phase === 'queued' ? 'thinking' : 'talking';
    const pf = r.prefill;
    return html`<div class="ch-now ${r.phase === 'permission' ? 'is-permission' : ''}" data-phase=${r.phase} role="status">
      <qse-wren-figure class="ch-now-fig" .state=${fig} .size=${30}></qse-wren-figure>
      <div class="ch-now-main">
        <div class="ch-now-line">
          <span class="ch-now-words">${r.now}</span>
          <span class="ch-now-nums num">
            ${r.decode_tps != null ? html`<span title="decode tok/s now">${fixed(r.decode_tps, 1)} tok/s</span>` : nothing}
            ${r.tokens ? html`<span title="completion tokens so far">${exact(r.tokens)} tok</span>` : nothing}
            <span title="time since the run began">${elapsedWords(elapsed)}</span>
          </span>
        </div>
        ${pf ? html`<div class="ch-now-bar" role="progressbar" aria-valuemin="0" aria-valuemax=${pf.total} aria-valuenow=${pf.done} aria-label="prefill progress"><span style="width:${Math.min(100, (pf.done / Math.max(1, pf.total)) * 100).toFixed(1)}%"></span></div>` : nothing}
        <div class="ch-now-sub">
          ${r.started_by ? html`<span>started ${agoWords(elapsed)} on ${r.started_by}</span>` : nothing}
          ${r.phase === 'permission' ? html`<span class="warn-ink">a tool is waiting for your answer</span>` : nothing}
        </div>
      </div>
      ${label && label.text ? html`<span class="ch-attach is-${label.tone}" data-attach=${this.attach?.state ?? ''}><span class="ch-attach-dot" aria-hidden="true"></span>${label.text}${this.attach?.state === 'lost' ? html` <button type="button" class="linkish" @click=${() => this.fire('ch-refresh', {})}>Refresh</button>` : nothing}</span>` : nothing}
    </div>`;
  }

  // ---- messages -------------------------------------------------------------------------------

  private renderMessage(c: Conversation, m: Message, i: number): TemplateResult {
    const live = m.status === 'streaming' && !!c.run && c.run.message_id === m.id;
    if (m.role === 'user') return this.renderUser(c, m, live);
    const isLast = i === c.messages.length - 1;
    return html`<article class="ch-msg ch-assistant ${live ? 'is-live' : ''}" data-mid=${m.id} data-role="assistant" data-status=${m.status}>
      ${m.parts.map((p, k) => this.renderPart(c, m, p, k, live))}
      ${!m.parts.length && live ? html`<p class="ch-waiting"><span class="pulse"></span> ${c.run?.phase === 'prefill' ? 'reading the prompt' : c.run?.phase === 'queued' ? (c.run.queue_position ? `in line, ${c.run.queue_position} ahead` : 'in line') : 'first token'}</p>` : nothing}
      ${m.status === 'interrupted' ? html`<div class="ch-interrupted" role="status" data-part="interrupted">${icon('warn')}<span>The engine restarted while this answer was being written. The text so far stays.</span><button type="button" class="btn btn-primary btn-sm" ?disabled=${!!c.run} @click=${() => this.fire('ch-continue', { messageId: m.id })}>${icon('play')} Continue</button></div>` : nothing}
      ${isLast && c.settings.plan_mode && !c.run && m.status === 'done' ? this.renderPlanBar(m) : nothing}
      ${m.status !== 'streaming' ? this.renderFoot(m, isLast, c) : nothing}
    </article>`;
  }

  private renderUser(c: Conversation, m: Message, live: boolean): TemplateResult {
    const editing = this.editing?.id === m.id;
    const text = messageText(m);
    return html`<article class="ch-msg ch-user ${editing ? 'is-editing' : ''}" data-mid=${m.id} data-role="user">
      ${m.attachments?.length
        ? html`<div class="ch-attachments">${m.attachments.map((a) => html`<span class="ch-chip" title=${`${a.name}, ${bytes(a.bytes)}`}>${a.kind === 'image' && a.preview ? html`<img src=${a.preview} alt="" />` : icon('file')}<span class="ch-chip-name">${a.name}</span><span class="ch-chip-size num">${bytes(a.bytes, 0)}</span></span>`)}</div>`
        : nothing}
      ${editing
        ? html`<div class="ch-bubble ch-edit">
            <textarea rows="3" .value=${this.editing!.text} @input=${(e: Event) => (this.editing = { id: m.id, text: (e.target as HTMLTextAreaElement).value })} @keydown=${(e: KeyboardEvent) => {
              if (e.key === 'Escape') this.editing = null;
              if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) this.saveEdit();
            }}></textarea>
            <div class="ch-edit-row"><span class="muted small">a new branch from here</span><span class="grow"></span><button type="button" class="btn btn-ghost btn-sm" @click=${() => (this.editing = null)}>Cancel</button><button type="button" class="btn btn-primary btn-sm" ?disabled=${!!c.run} @click=${() => this.saveEdit()}>${icon('send')} Send</button></div>
          </div>`
        : html`<div class="ch-bubble"><div class="ch-text">${text}</div></div>`}
      <div class="ch-user-foot">
        ${m.siblings ? this.renderPager(m) : nothing}
        <span class="ch-tools">
          ${copyButton(text, 'Copy message')}
          <button type="button" class="btn btn-ghost btn-sm" aria-label="Edit and branch" title="Edit" ?disabled=${live || !!c.run} @click=${() => (this.editing = { id: m.id, text })}>${icon('edit')}</button>
          <button type="button" class="btn btn-ghost btn-sm" aria-label="Rewind to here" title="Rewind to here: the files go back to before this message, and the message comes back to the composer" @click=${() => {
            if (confirm('Rewind to this message? The files go back to how they were before it, and the message returns to the composer. Nothing is deleted: the old branch stays.')) this.fire('ch-rewind', { messageId: m.id });
          }}>${icon('left')}</button>
        </span>
      </div>
    </article>`;
  }

  /** SRV-363: under a plan answer, Approve (its steps become the todo list) or Edit plan */
  private renderPlanBar(m: Message): TemplateResult {
    const approve = (items?: string[]) => {
      this.planEdit = null;
      this.fire('ch-approve-plan', { items });
    };
    return html`<div class="ch-planbar" data-part="plan" role="group" aria-label="the plan">
      ${this.planEdit != null
        ? html`<label class="sr-only" for="ch-plan-edit">the plan's steps, one a line</label><textarea id="ch-plan-edit" rows="5" .value=${this.planEdit} @input=${(e: Event) => (this.planEdit = (e.target as HTMLTextAreaElement).value)}></textarea>
            <div class="ch-edit-row"><span class="muted small">one step a line</span><span class="grow"></span><button type="button" class="btn btn-ghost btn-sm" @click=${() => (this.planEdit = null)}>Cancel</button><button type="button" class="btn btn-primary btn-sm" ?disabled=${!this.planEdit.trim()} @click=${() => approve(this.planEdit!.split('\n').map((x) => x.trim()).filter(Boolean))}>${icon('check')} Approve</button></div>`
        : html`<span class="small">${planItems(messageText(m)).length} step plan. Approve it to leave plan mode and carry it out.</span><span class="grow"></span><button type="button" class="btn btn-sm" @click=${() => (this.planEdit = planItems(messageText(m)).join('\n'))}>${icon('edit')} Edit plan</button><button type="button" class="btn btn-primary btn-sm" @click=${() => approve()}>${icon('check')} Approve plan</button>`}
    </div>`;
  }

  private saveEdit(): void {
    const e = this.editing;
    if (!e || !e.text.trim()) return;
    this.editing = null;
    this.fire('ch-edit', { messageId: e.id, text: e.text.trim() });
  }

  private renderPager(m: Message): TemplateResult {
    const s = m.siblings!;
    return html`<span class="ch-pager" aria-label="versions">
      <button type="button" class="btn btn-ghost btn-sm" aria-label="Previous version" ?disabled=${s.index === 0} @click=${() => this.fire('ch-branch', { messageId: m.id, index: s.index - 1 })}>${icon('left')}</button>
      <span class="num">${s.index + 1} / ${s.count}</span>
      <button type="button" class="btn btn-ghost btn-sm" aria-label="Next version" ?disabled=${s.index >= s.count - 1} @click=${() => this.fire('ch-branch', { messageId: m.id, index: s.index + 1 })}>${icon('right')}</button>
    </span>`;
  }

  private renderFoot(m: Message, isLast: boolean, c: Conversation): TemplateResult {
    const s = m.stats;
    return html`<footer class="ch-foot">
      <span class="ch-figs num" data-testid="turn-figures">${s ? figsLine(s) : m.status === 'stopped' ? 'stopped' : m.status === 'interrupted' ? 'interrupted' : ''}${m.status === 'stopped' && s ? html` <span class="warn-ink">stopped</span>` : nothing}</span>
      <span class="ch-tools">
        ${copyButton(messageText(m), 'Copy answer')}
        <button type="button" class="btn btn-ghost btn-sm" aria-label="Regenerate" title="Regenerate" ?disabled=${!!c.run} @click=${() => this.fire('ch-regenerate', { messageId: m.id })}>${icon('refresh')}</button>
        ${m.siblings ? this.renderPager(m) : nothing}
        ${isLast ? nothing : nothing}
      </span>
    </footer>`;
  }

  // ---- parts ----------------------------------------------------------------------------------

  private renderPart(c: Conversation, m: Message, p: Part, k: number, live: boolean): TemplateResult {
    const lastPart = k === m.parts.length - 1;
    switch (p.type) {
      case 'thinking': {
        const only = live && !p.done;
        return html`<details class="ch-think" ?open=${only} data-part="thinking">
          <summary>${only ? html`<span class="pulse"></span>` : nothing}<span>Thinking${p.tokens != null ? html`<span class="num muted">, ${exact(p.tokens)} tokens</span>` : only ? html`<span class="num muted">, ${exact(Math.round(p.text.length / 4))} so far</span>` : nothing}</span></summary>
          <div class="ch-think-body">${p.text}</div>
        </details>`;
      }
      case 'text':
        return html`<div class="ch-md ${live && lastPart ? 'is-streaming' : ''}" data-part="text" @click=${(e: Event) => this.onMdClick(e, p.text)}>${unsafeHTML(decorateCode(renderMarkdown(p.text)))}</div>`;
      case 'tool_call':
        return this.renderCall(c, m, p, live && lastPart);
      case 'error':
        return html`<div class="ch-error" role="alert" data-part="error">${icon('warn')}<span>${p.message}</span>${p.retryable ? html`<button type="button" class="btn btn-sm" ?disabled=${!!c.run} @click=${() => this.fire('ch-regenerate', { messageId: m.id })}>${icon('refresh')} Retry</button>` : nothing}</div>`;
      case 'todos':
        return html``;
      case 'artifact':
        return html`<button type="button" class="ch-artifact" data-part="artifact" @click=${() => this.fire('ch-open-pane', { pane: { kind: p.mime.startsWith('image/') ? 'image' : p.mime === 'text/html' ? 'preview' : 'markdown', ref: p.id, title: p.title } as PaneRef })}>${icon('preview')}<span class="ch-artifact-title">${p.title}</span><span class="muted small">v${p.version}, open in a pane</span></button>`;
    }
  }

  /** a click on a code block's own buttons (copy, open in pane), drawn by decorateCode() */
  private onMdClick(e: Event, _text: string): void {
    const btn = (e.target as HTMLElement).closest<HTMLElement>('[data-code-action]');
    if (!btn) return;
    const box = btn.closest<HTMLElement>('.md-code');
    const code = box?.querySelector('code')?.textContent ?? '';
    const lang = box?.dataset.lang ?? '';
    if (btn.dataset.codeAction === 'copy') {
      void navigator.clipboard.writeText(code.replace(/\n$/, '')).then(() => {
        btn.classList.add('is-done');
        setTimeout(() => btn.classList.remove('is-done'), 1200);
      });
    } else if (btn.dataset.codeAction === 'pane') {
      const kind = PANE_LANGS[lang] ?? 'markdown';
      this.fire('ch-open-pane', { pane: { kind, ref: `inline:${lang}:${hash(code)}`, title: `${lang || 'text'} block` } as PaneRef, inline: { content: code, lang } });
    }
  }

  private renderCall(c: Conversation, m: Message, t: ToolCallPart, live: boolean): TemplateResult {
    const open = this.openCalls.has(t.id) || t.status === 'asking' || (live && t.status === 'running' && !!t.output);
    const args = argsSummary(t);
    // the page or Markdown file a write or an edit left opens from the card, folded or not
    const pane = callPane(t);
    // a video or an audio file the call presented plays in the card (VIS-159)
    const media = callMedia(t, c.id);
    if (media) void loadMediaView();
    const dur = t.started_at && t.ended_at ? ms(Date.parse(t.ended_at) - Date.parse(t.started_at)) : t.started_at && t.status === 'running' ? ms(this.now - Date.parse(t.started_at)) : null;
    // web_search: the results as links under the head, open or folded (SRV-469, VIS-157)
    const found = searchResults(t);
    const word = t.status === 'asking' ? 'asking' : t.status === 'done' && found ? resultsWord(found.length) : t.status === 'done' && t.output?.exit_code ? `exit ${t.output.exit_code}` : t.status;
    return html`<section class="ch-call is-${t.status}" data-call=${t.id} data-tool=${t.name}>
      <button type="button" class="ch-call-head" aria-expanded=${open} @click=${() => this.toggleCall(t.id)}>
        <span class="ch-call-dot" aria-hidden="true"></span>
        <code class="ch-call-name">${toolLabel(t.name)}</code>${t.risk ? html`<span class="ch-risk ch-risk-${t.risk}" title=${`${t.risk} tool`}>${t.risk}</span>` : nothing}${t.approved && t.approved !== 'user' ? html`<span class="ch-approved is-${t.approved}" title=${approvedTitle(t.approved)}>${approvedWord(t.approved)}</span>` : nothing}
        <span class="ch-call-args num">${args}</span>
        <span class="grow"></span>
        ${dur ? html`<span class="ch-call-dur num">${dur}</span>` : nothing}
        <span class="ch-call-word">${word}</span>
        <span class="ch-call-chev" aria-hidden="true"></span>
      </button>
      ${found?.length ? renderResults(found) : nothing}
      ${t.name === 'web_search' && t.status === 'failed' && t.output?.content ? html`<p class="ch-call-row ch-search-err err-ink small" role="alert">${t.output.content.split('\n')[0]}</p>` : nothing}
      ${media ? html`<div class="ch-call-row ch-call-media"><qse-media-view kind=${media.kind} src=${media.src} name=${media.name} compact></qse-media-view></div>` : nothing}
      ${pane ? html`<div class="ch-call-row ch-call-open"><button type="button" class="btn btn-sm" @click=${() => this.fire('ch-open-pane', { pane })}>${icon(paneIcon(pane.kind))} Open in ${paneWord(pane.kind)}</button></div>` : nothing}
      ${open
        ? html`<div class="ch-call-body">
            ${t.status === 'asking' && t.permission
              ? html`<div class="ch-perm" role="group" aria-label="permission">
                  <p class="ch-perm-text">${t.permission.summary}</p>
                  <div class="ch-perm-row">
                    <button type="button" class="btn btn-primary btn-sm" @click=${() => this.fire('ch-permission', { callId: t.id, decision: 'once' })}>Allow once</button>
                    ${t.permission.suggested_rule?.matches?.length === 0
                      ? nothing
                      : html`<button type="button" class="btn btn-sm" @click=${() => this.fire('ch-permission', { callId: t.id, decision: 'always' })}>${t.permission.suggested_rule?.matches ? html`Always allow <code class="small">${t.permission.suggested_rule.matches.join(', ')}</code>` : html`Always allow ${toolLabel(t.permission.suggested_rule?.tool ?? t.name)}${t.permission.suggested_rule?.match ? html` <code class="small">${t.permission.suggested_rule.match}</code>` : nothing}`}</button>`}
                    <button type="button" class="btn btn-sm ch-deny" @click=${() => this.fire('ch-permission', { callId: t.id, decision: 'deny' })}>Deny</button>
                  </div>
                </div>`
              : nothing}
            ${c.settings.policies[t.name] === 'always' && t.status !== 'asking' ? html`<p class="ch-call-note">always allowed in this chat</p>` : nothing}
            <div class="ch-call-io">
              <div class="ch-call-col"><div class="ch-call-label">input${copyButton(t.input, 'Copy input')}</div><pre class="ch-call-pre ${live && t.status === 'running' && !t.output ? '' : ''}">${prettyJson(t.input) || (live ? '' : '{}')}</pre></div>
              ${t.output || t.status === 'running'
                ? html`<div class="ch-call-col"><div class="ch-call-label">output${t.output ? html`<span class="num muted">${bytes(t.output.bytes, 0)}</span>${copyButton(t.output.content, 'Copy output')}` : nothing}</div><pre class="ch-call-pre ch-call-out ${t.output?.is_error ? 'is-error' : ''} ${t.status === 'running' ? 'is-streaming' : ''}">${stripAnsi(t.output?.content ?? '')}</pre></div>`
                : nothing}
            </div>
          </div>`
        : nothing}
    </section>`;
  }

  private toggleCall(id: string): void {
    const s = new Set(this.openCalls);
    if (s.has(id)) s.delete(id);
    else s.add(id);
    this.openCalls = s;
  }
}

// ---- words and helpers ------------------------------------------------------------------------

/** a web search's results: the title links out (a new tab, no referrer), the host and snippet */
function renderResults(rs: ReturnType<typeof searchResults> & object): TemplateResult {
  return html`<ol class="ch-call-row ch-search" aria-label="search results">${rs.map(
    (r) => html`<li class="ch-search-item"><a class="ch-search-title" href=${r.url} target="_blank" rel="noopener noreferrer">${r.title}</a><span class="ch-search-meta"><span class="ch-search-host">${hostOf(r.url)}</span>${r.date ? html`<span class="num">${r.date}</span>` : nothing}${r.engine ? html`<span>${r.engine}</span>` : nothing}</span>${r.snippet ? html`<span class="ch-search-snip">${r.snippet}</span>` : nothing}</li>`,
  )}</ol>`;
}

export function figsLine(s: TurnStats): string {
  const parts: string[] = [];
  parts.push(s.finish === 'stop' ? 'done' : s.finish);
  if (s.ttft_ms != null) parts.push(`${ms(s.ttft_ms)} to first token`);
  if (s.decode_tps != null) parts.push(`${fixed(s.decode_tps, 1)} tok/s`);
  parts.push(`${exact(s.completion_tokens)} tok${s.reasoning_tokens ? ` (${exact(s.reasoning_tokens)} thinking)` : ''}`);
  if (s.tokens_per_round != null) parts.push(`${fixed(s.tokens_per_round, 1)} per round`);
  if (s.acceptance != null) parts.push(`${pct(s.acceptance, 0)} accepted`);
  return parts.join(' · ');
}

export function elapsedWords(sec: number): string {
  if (sec < 60) return `${sec} s`;
  const m = Math.floor(sec / 60);
  return `${m} min ${String(sec - m * 60).padStart(2, '0')} s`;
}

export function agoWords(sec: number): string {
  if (sec < 60) return `${sec} s ago`;
  const m = Math.round(sec / 60);
  return m < 60 ? `${m} min ago` : `${Math.round(m / 60)} h ago`;
}

/** the one argument that names the call: the command, the path, the query */
export function prettyJson(s: string): string {
  try {
    return JSON.stringify(JSON.parse(s), null, 2);
  } catch {
    return s;
  }
}

export function stripAnsi(s: string): string {
  return s.replace(/\x1b\[[0-9;]*[A-Za-z]/g, '');
}

const PANE_WORDS: Record<PaneRef['kind'], [string, string]> = {
  terminal: ['Terminal', 'terminal'],
  preview: ['Preview', 'preview'],
  files: ['Files', 'folder'],
  editor: ['Editor', 'edit'],
  diff: ['Diff', 'diff'],
  markdown: ['Markdown', 'doc'],
  image: ['Image', 'image'],
  pdf: ['PDF', 'doc'],
  video: ['Video', 'play'],
  audio: ['Audio', 'play'],
  binary: ['File', 'file'],
  context: ['Context', 'info'],
  plan: ['Plan', 'doc'],
};
export function paneWord(k: PaneRef['kind']): string {
  return PANE_WORDS[k]?.[0] ?? 'Pane';
}
export function paneIcon(k: PaneRef['kind']): string {
  return PANE_WORDS[k]?.[1] ?? 'doc';
}

/** The code blocks from the Markdown renderer get a toolbar: copy, and "Open in pane" for the
 *  kinds a pane can show. The renderer escapes every character; this only adds markup of its own. */
export function decorateCode(htmlText: string): string {
  return htmlText.replace(/<div class="md-code( is-open)?"( data-lang="([^"]*)")?>/g, (_m, open: string | undefined, _d, lang: string | undefined) => {
    const l = lang ?? '';
    const paneable = l in PANE_LANGS;
    return `<div class="md-code${open ?? ''}"${lang != null ? ` data-lang="${l}"` : ''}><span class="md-code-tools"><button type="button" class="btn btn-ghost btn-sm" data-code-action="copy" aria-label="Copy code" title="Copy"><svg class="icon" viewBox="0 0 20 20" aria-hidden="true" width="18" height="18"><rect x="7" y="7" width="10" height="10" rx="1.5" fill="none" stroke="currentColor" stroke-width="1.6"/><path d="M13 7V4.5a1.5 1.5 0 0 0-1.5-1.5h-7A1.5 1.5 0 0 0 3 4.5v7A1.5 1.5 0 0 0 4.5 13H7" fill="none" stroke="currentColor" stroke-width="1.6"/></svg><span class="copy-done"><svg class="icon" viewBox="0 0 20 20" aria-hidden="true" width="18" height="18"><path d="m4 10.5 4 4 8-9" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg></span></button>${paneable ? `<button type="button" class="btn btn-ghost btn-sm" data-code-action="pane" title="Open in pane">Open in pane</button>` : ''}</span>`;
  });
}

function hash(s: string): string {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619);
  return (h >>> 0).toString(36);
}

