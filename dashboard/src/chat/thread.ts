// <qse-chat-thread>: the conversation. The Now line while a run is on (the phase in words, the
// figure, the attachment state), user bubbles with attachments and a branch pager, assistant turns
// as parts in order: a thinking block that opens while it is the only thing streaming, Markdown
// with "Open in pane" on code blocks, tool cards with the permission prompt (allow once, always,
// deny), errors with Retry, the figures line, and the queued messages above the composer. A
// `task` call is a worker's Task card (VIS-217): its live state, report and chatd's check, and the
// worker's transcript drawn with the same part renderers.

import { html, nothing, type TemplateResult } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import { repeat } from 'lit/directives/repeat.js';
import { unsafeHTML } from 'lit/directives/unsafe-html.js';
import { bytes, exact, fixed, money, ms, pct } from '../lib/format';
import { renderMarkdown } from '../playground/markdown';
import { LightElement } from '../ui/base';
import { copyButton, icon } from '../ui/bits';
import type { QseWrenFigure } from '../setup/figure';
import type { FigureState } from '../setup/figure-state';
import { attachLabel, type Attachment as Attach } from './connection';
import { argsSummary, dividersByMessage, dividerWords, interruptCard, interruptWords, messageText, taskForCall } from './store';
import { planItems, toolLabel, type Compaction, type Conversation, type Message, type PaneRef, type Part, type Task, type ToolCallPart, type TurnStats } from './types';
import { askingCall, callsWord, checkView, isUnmatched, reportView, shotsOf, sourceWord, summaryLine, taskSeconds, taskStatus, tokensWord } from './tasks';
import { approvedTitle, approvedWord } from './permission';
import { callMedia, callPane, loadMediaView } from './file-panes';
import { hostOf, resultsWord, searchResults } from './websearch';
import { attachmentWords, sandboxCallWords } from './context-words';
import { figsParts, workerFigures } from './turn-figures';
import { actingPhase, isHarnessCall, verifyBlock, verifyTone } from './verify';

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
  /** VIS-217: the Task cards the user opened, and the ones whose worker transcript is open */
  @state() private openTasks = new Set<string>();
  @state() private openTx = new Set<string>();
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
    const fig: FigureState = actingPhase(r.phase) ? 'acting' : r.phase === 'permission' ? 'listening' : r.phase === 'thinking' || r.phase === 'prefill' || r.phase === 'queued' ? 'thinking' : 'talking';
    const pf = r.prefill;
    return html`<div class="ch-now ${r.phase === 'permission' ? 'is-permission' : ''}" data-phase=${r.phase} role="status">
      <qse-wren-figure class="ch-now-fig" .state=${fig} .size=${30}></qse-wren-figure>
      <div class="ch-now-main">
        <div class="ch-now-line">
          <span class="ch-now-words" title=${r.now}>${r.now}</span>
          <span class="ch-now-nums num">
            ${r.phase === 'prefill' && pf
              ? html`<span title="prompt tokens read of this call's prompt">${exact(pf.done)} / ${exact(pf.total)}</span>
                ${pf.tps != null ? html`<span title="prefill tok/s now">${fixed(pf.tps, 0)} tok/s</span>` : nothing}`
              : r.decode_tps != null ? html`<span title="decode tok/s now">${fixed(r.decode_tps, 1)} tok/s</span>` : nothing}
            ${r.tokens ? html`<span title="tokens the run has written so far">${exact(r.tokens)} written</span>` : nothing}
            <span title="time since the run began">run ${elapsedWords(elapsed)}</span>
          </span>
        </div>
        ${pf ? html`<div class="ch-now-bar" role="progressbar" aria-valuemin="0" aria-valuemax=${pf.total} aria-valuenow=${pf.done} aria-label="prefill progress"><span style="width:${Math.min(100, (pf.done / Math.max(1, pf.total)) * 100).toFixed(1)}%"></span></div>` : nothing}
        <div class="ch-now-sub">
          ${r.started_by ? html`<span>started ${agoWords(elapsed)} on ${r.started_by}</span>` : nothing}
          ${r.phase === 'permission' ? html`<span class="warn-ink">a tool is waiting for your answer</span>` : nothing}
          ${taskAsking(c) ? html`<span class="warn-ink" data-task-asks>${taskAsking(c)}</span>` : nothing}
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
    const cut = interruptCard(m, c.run?.message_id);
    return html`<article class="ch-msg ch-assistant ${live ? 'is-live' : ''}" data-mid=${m.id} data-role="assistant" data-status=${m.status}>
      ${(cut?.notice ? m.parts.slice(0, -1) : m.parts).map((p, k) => this.renderPart(c, m, p, k, live))}
      ${!m.parts.length && live ? html`<p class="ch-waiting"><span class="pulse"></span> ${c.run?.phase === 'prefill' ? 'reading the prompt' : c.run?.phase === 'queued' ? (c.run.queue_position ? `in line, ${c.run.queue_position} ahead` : 'in line') : 'first token'}</p>` : nothing}
      ${cut ? html`<div class="ch-interrupted" role="status" data-part="interrupted">${icon('warn')}<span>${cut.text}</span><button type="button" class="btn btn-primary btn-sm" ?disabled=${!!c.run} @click=${() => this.fire('ch-continue', { messageId: m.id })}>${icon('play')} Continue</button></div>` : nothing}
      ${isLast && c.settings.plan_mode && !c.run && m.status === 'done' ? this.renderPlanBar(m) : nothing}
      ${m.status !== 'streaming' ? this.renderFoot(m, isLast, c) : nothing}
    </article>`;
  }

  private renderUser(c: Conversation, m: Message, live: boolean): TemplateResult {
    const editing = this.editing?.id === m.id;
    const text = messageText(m);
    return html`<article class="ch-msg ch-user ${editing ? 'is-editing' : ''}" data-mid=${m.id} data-role="user">
      ${m.attachments?.length
        ? html`<div class="ch-attachments">${m.attachments.map((a) => html`<span class="ch-chip" title=${`${a.name}, ${bytes(a.bytes)}`}>${a.kind === 'image' && a.preview ? html`<img src=${a.preview} alt="" />` : icon('file')}<span class="ch-chip-name">${a.name}</span><span class="ch-chip-size num">${attachmentWords(a).size}</span>${a.inline === false ? html`<span class="ch-chip-warn warn-ink" title=${attachmentWords(a).warn ?? ''}>as a file</span>` : nothing}</span>`)}</div>${m.attachments.map((a) => attachmentWords(a).warn).filter(Boolean).map((w) => html`<p class="ch-attach-warn warn-ink small">${w}</p>`)}`
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
    const w = s ? workerFigures(s) : [];
    return html`${w.length ? html`<p class="ch-wfigs num" data-testid="turn-workers">${w.map((f) => html`<span class="ch-tf ch-tf-${f.k}">${f.w}<b>${f.n}</b>${f.sub ? html` ${f.sub}` : nothing}</span>`)}</p>` : nothing}<footer class="ch-foot">
      <span class="ch-figs num" data-testid="turn-figures" title=${s ? figsLine(s) : nothing}>${s ? figsParts(s).map((f) => html`<span class="ch-tf ch-tf-${f.k}">${f.pre ? f.w : nothing}${f.n ? html`<b>${f.n}</b>` : nothing}${f.pre ? nothing : f.w}${f.sub ? html`<span class="ch-tf-sub"> ${f.sub}</span>` : nothing}</span>`) : m.status === 'stopped' ? 'stopped' : m.status === 'interrupted' ? 'interrupted' : ''}${m.status === 'stopped' && s ? html`<span class="ch-tf warn-ink">stopped</span>` : nothing}</span>
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
        return p.name === 'task' && !isHarnessCall(p) ? this.renderTask(c, m, p, live && lastPart) : this.renderCall(c, m, p, live && lastPart);
      case 'error':
        return html`<div class="ch-error" role="alert" data-part="error">${icon('warn')}<span>${p.message}</span>${p.retryable ? html`<button type="button" class="btn btn-sm" ?disabled=${!!c.run} @click=${() => this.fire('ch-regenerate', { messageId: m.id })}>${icon('refresh')} Retry</button>` : nothing}</div>`;
      case 'todos':
        return html``;
      case 'verify': {
        const b = verifyBlock(p);
        return html`<div class="ch-verify is-${verifyTone(p.state)}" role="note" data-part="verify" data-state=${p.state}>
          <p class="ch-verify-lead">${icon(p.state === 'passed' ? 'check' : 'warn')}<span>${b.lead}</span></p>
          ${b.checked.length ? html`<p class="ch-verify-k">Checked</p><ul class="ch-verify-list">${b.checked.map((x) => html`<li>${x}</li>`)}</ul>` : nothing}
          ${b.unchecked.length ? html`<p class="ch-verify-k">Not checked</p><ul class="ch-verify-list is-not">${b.unchecked.map((x) => html`<li>${x}</li>`)}</ul>` : nothing}
        </div>`;
      }
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

  // ---- the Task card (VIS-217) ------------------------------------------------------------------

  /** a worker's card in place of its `task` call: "Task 2", the title, the status chip, the calls,
   *  tokens and time, the Now line while it runs (the summary once it ended), its permission prompt;
   *  opened, the brief, the report and chatd's check with their screenshots, and the transcript. A
   *  call chatd refused before any worker started stays a tool card with the refusal. */
  private renderTask(c: Conversation, m: Message, t: ToolCallPart, live: boolean): TemplateResult {
    const task = taskForCall(c, t.id);
    if (!task) {
      if (t.status !== 'running' && t.status !== 'asking') return this.renderCall(c, m, t, live);
      return html`<section class="ch-task is-running" data-call=${t.id} data-status="starting">
        <div class="ch-task-head is-static"><span class="ch-call-dot" aria-hidden="true"></span><span class="ch-task-n">Task</span><span class="ch-task-title">${taskTitle(t.input)}</span><span class="grow"></span><span class="ch-task-chip is-live">starting</span></div>
      </section>`;
    }
    const st = taskStatus(task);
    const open = this.openTasks.has(task.id);
    const ask = askingCall(task);
    const running = task.status === 'running';
    const summary = running ? '' : summaryLine(task);
    return html`<section class="ch-task is-${st.key} ${running ? 'is-running' : ''}" data-task=${task.id} data-call=${t.id} data-status=${st.key} aria-label=${`Task ${task.n}: ${task.title}, ${st.word}`}>
      <button type="button" class="ch-task-head" aria-expanded=${open} @click=${() => this.toggleTask(task)}>
        <span class="ch-call-dot" aria-hidden="true"></span>
        <span class="ch-task-n">Task ${task.n}</span>
        <span class="ch-task-title">${task.title}</span>
        <span class="grow"></span>
        <span class="ch-task-chip is-${st.tone}" data-chip=${st.key}>${st.word}</span>
        <span class="ch-call-chev" aria-hidden="true"></span>
      </button>
      <div class="ch-task-meta num">
        <span data-fig="calls">${callsWord(task)}</span>
        <span data-fig="tokens" title="the worker's prompt tokens in, completion tokens out">${tokensWord(task)}</span>
        <span data-fig="elapsed">${elapsedWords(taskSeconds(task, this.now))}</span>
        <span class="grow"></span>
        ${running ? html`<button type="button" class="btn btn-ghost btn-sm ch-task-stop" title="Stop the whole run: the worker and the main agent" @click=${() => this.fire('ch-stop', {})}>${icon('stop')} Stop</button>` : nothing}
        ${task.status === 'interrupted' ? html`<button type="button" class="btn btn-primary btn-sm" ?disabled=${!!c.run} title="Resume the worker, then the main agent" @click=${() => this.fire('ch-continue', { messageId: m.id })}>${icon('play')} Continue</button>` : nothing}
      </div>
      ${running && task.now ? html`<p class="ch-task-now" data-now><span class="pulse" aria-hidden="true"></span><span>${task.now}</span></p>` : nothing}
      ${summary ? html`<p class="ch-task-sum" data-summary>${summary}</p>` : nothing}
      ${ask ? html`<div class="ch-task-ask" role="group" aria-label=${`Task ${task.n} asks`}><p class="ch-task-ask-h">Task ${task.n} asks</p>${this.renderCall(c, ask.message, ask.call, true)}</div>` : nothing}
      ${open ? this.renderTaskBody(c, task) : nothing}
    </section>`;
  }

  /** opening a finished card loads its transcript (lazily, once): the screenshots come from it */
  private toggleTask(task: Task): void {
    this.openTasks = toggled(this.openTasks, task.id);
    if (this.openTasks.has(task.id) && !task.messages) this.fire('ch-task-load', { taskId: task.id });
  }

  private renderTaskBody(c: Conversation, task: Task): TemplateResult {
    const rep = task.report ? reportView(task.report) : null;
    const chk = checkView(task.check);
    const shots = shotsOf(task, c.id);
    const tx = this.openTx.has(task.id);
    const calls = (task.messages ?? []).reduce((a, m) => a + m.parts.filter((p) => p.type === 'tool_call').length, 0);
    const toggleTx = () => {
      this.openTx = toggled(this.openTx, task.id);
    };
    const src = sourceWord(task);
    return html`<div class="ch-task-body">
      <details class="ch-task-brief"><summary>Brief</summary><pre class="ch-call-pre">${task.brief}${task.expect ? `\nThe report must also contain: ${task.expect}` : ''}</pre></details>
      ${rep
        ? html`<div class="ch-task-report" data-part="report">
            <p class="ch-verify-k">Report: ${rep.status}${src ? html` <span class="ch-task-flag" data-source=${task.report_source ?? ''}>${src}</span>` : nothing}</p>
            ${rep.summary ? html`<p class="ch-task-summary">${rep.summary}</p>` : nothing}
            <p class="ch-verify-k">Changed files</p>${rep.changed.length ? html`<ul class="ch-verify-list">${rep.changed.map((f) => html`<li><code>${f.path}</code>${f.change ? html`: ${f.change}` : nothing}</li>`)}</ul>` : html`<p class="muted small">none</p>`}
            <p class="ch-verify-k">Checks</p>${rep.checks.length ? html`<ul class="ch-verify-list">${rep.checks.map((k) => html`<li>${k.text}${isUnmatched(k, chk) ? html` <span class="ch-task-flag" data-flag="unmatched">no matching call</span>` : nothing}</li>`)}</ul>` : html`<p class="muted small">none</p>`}
            ${rep.expected.length ? html`<p class="ch-verify-k">Expected</p><ul class="ch-verify-list">${rep.expected.map((x) => html`<li>${x}</li>`)}</ul>` : nothing}
            <p class="ch-verify-k">Open problems</p>${rep.open.length ? html`<ul class="ch-verify-list">${rep.open.map((x) => html`<li>${x}</li>`)}</ul>` : html`<p class="muted small">none</p>`}
          </div>`
        : task.status === 'running' ? nothing : html`<p class="muted small">No report.${task.error ? ` ${task.error}` : ''}</p>`}
      ${task.check
        ? html`<div class="ch-task-check" data-part="check">
            <p class="ch-verify-k">chatd's check</p>
            ${chk.changed.length ? html`<p class="small">Changed in /workspace</p><ul class="ch-verify-list">${chk.changed.map((f) => html`<li data-path=${f.path}><code>${f.path}</code>${f.delta ? html` <span class="num muted">${f.delta}</span>` : nothing}${f.unreported ? html` <span class="ch-task-flag" data-flag="unreported">changed but not in the report</span>` : nothing}</li>`)}</ul>` : html`<p class="small">Changed in /workspace: none</p>`}
            ${chk.ran.length ? html`<p class="small">Checks that ran</p><ul class="ch-verify-list">${chk.ran.map((r) => html`<li>${r.text}</li>`)}</ul>` : nothing}
            ${chk.unmatched.length ? html`<p class="small warn-ink">Claimed in Checks with no matching call</p><ul class="ch-verify-list">${chk.unmatched.map((u) => html`<li>${u}</li>`)}</ul>` : nothing}
            ${chk.rest.map((l) => html`<p class="small">${l}</p>`)}
          </div>`
        : nothing}
      ${shots.length ? html`<div class="ch-task-shots" aria-label="screenshots">${shots.map((sh) => html`<button type="button" class="ch-task-shot" data-shot=${sh.id} title=${`Open ${sh.id}`} @click=${() => this.fire('ch-open-pane', { pane: sh.pane })}><img src=${sh.src} alt=${`screenshot ${sh.id}`} loading="lazy" @error=${(e: Event) => ((e.target as HTMLImageElement).hidden = true)} /><span class="num">${sh.id}</span></button>`)}</div>` : nothing}
      <button type="button" class="btn btn-sm ch-task-txbtn" aria-expanded=${tx} @click=${toggleTx}>${icon('doc')} Worker transcript${task.messages ? html`<span class="num muted">${calls} call${calls === 1 ? '' : 's'}</span>` : nothing}</button>
      ${tx ? this.renderTranscript(c, task, toggleTx) : nothing}
    </div>`;
  }

  /** the worker's own messages, drawn as the main thread draws its parts; on a phone a sheet */
  private renderTranscript(c: Conversation, task: Task, close: () => void): TemplateResult {
    const ms = (task.messages ?? []).filter((m) => m.role !== 'user');
    return html`<div class="ch-task-scrim" @click=${close}></div><div class="ch-task-tx" role="region" aria-label=${`Task ${task.n} transcript`} data-transcript=${task.id}>
      <div class="ch-task-tx-head"><span>Task ${task.n}: ${task.title}</span><span class="grow"></span><button type="button" class="btn btn-ghost btn-sm" aria-label="Close the transcript" @click=${close}>${icon('x')}</button></div>
      ${!task.messages
        ? html`<p class="ch-waiting"><span class="pulse"></span> loading the transcript</p>`
        : ms.length
          ? ms.map((m) => html`<div class="ch-task-tx-msg" data-mid=${m.id}>${m.parts.map((p, k) => this.renderPart(c, m, p, k, m.status === 'streaming'))}</div>`)
          : html`<p class="muted small">${task.status === 'running' ? 'The worker has not written anything yet.' : 'The worker wrote nothing.'}</p>`}
    </div>`;
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
        ${isHarnessCall(t) ? html`<code class="ch-call-name">verify</code><span class="ch-approved is-harness" title="the harness checked the app in a browser before the run ends; the model reads the result">harness</span>` : html`<code class="ch-call-name">${toolLabel(t.name)}</code>${t.risk ? html`<span class="ch-risk ch-risk-${t.risk}" title=${`${t.risk} tool`}>${t.risk}</span>` : nothing}`}${t.approved && t.approved !== 'user' ? html`<span class="ch-approved is-${t.approved}" title=${approvedTitle(t.approved)}>${approvedWord(t.approved)}</span>` : nothing}
        <span class="ch-call-args num">${args}</span>
        <span class="grow"></span>
        ${dur ? html`<span class="ch-call-dur num">${dur}</span>` : nothing}
        <span class="ch-call-word ${word === 'done' ? 'sr-only' : ''}">${word}</span>
        <span class="ch-call-chev" aria-hidden="true"></span>
      </button>
      ${found?.length ? renderResults(found) : nothing}
      ${t.name === 'web_search' && t.status === 'failed' && t.output?.content ? html`<p class="ch-call-row ch-search-err err-ink small" role="alert">${t.output.content.split('\n')[0]}</p>` : nothing}
      ${t.output?.data?.sandbox ? html`<p class="ch-call-row ch-search-err err-ink small" role="alert" data-sandbox=${t.output.data.sandbox.reason}>${sandboxCallWords(t.output.data.sandbox)}</p>` : nothing}
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
  for (const f of workerFigures(s)) parts.push(`${f.w}${f.n}${f.sub ? ` ${f.sub}` : ''}`);
  if (s.tokens_per_round != null) parts.push(`${fixed(s.tokens_per_round, 1)} per round`);
  if (s.acceptance != null) parts.push(`${pct(s.acceptance, 0)} accepted`);
  if (s.cost_usd != null) parts.push(money(Math.round(s.cost_usd * 1e6)));
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

function toggled(set: Set<string>, id: string): Set<string> {
  const s = new Set(set);
  if (s.has(id)) s.delete(id);
  else s.add(id);
  return s;
}

/** a task call's title while its arguments still stream in */
export function taskTitle(input: string): string {
  try {
    const t = (JSON.parse(input) as { title?: unknown }).title;
    if (typeof t === 'string' && t.trim()) return t;
  } catch {
    /* the JSON is not whole yet */
  }
  return /"title"\s*:\s*"((?:[^"\\]|\\.)*)/.exec(input)?.[1] ?? 'giving a task to a worker';
}

/** the Now line's word for a worker that waits for permission: "Task 2 asks: bash wants to run ..." */
export function taskAsking(c: Pick<Conversation, 'tasks'>): string | null {
  for (const t of c.tasks ?? []) {
    if (t.status !== 'running') continue;
    const a = askingCall(t);
    if (a || t.phase === 'permission') return `Task ${t.n} asks${a?.call.permission?.summary ? `: ${a.call.permission.summary}` : ''}`;
  }
  return null;
}

function hash(s: string): string {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619);
  return (h >>> 0).toString(36);
}

