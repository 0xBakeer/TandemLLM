// <qse-install-progress>: a setup job, live (D4 step Install, VIS-78; reused by reset level 2,
// VIS-82). GET /v1/setup/jobs/{id} as SSE: the step list (waiting, running, done, failed, skipped),
// a log tail, and the end: installed with the way out, or failed / rolled back with the way back.
// The engine restarts in the middle of a job; the stream then drops and the page says so, retries
// with back-off and resumes the same job (Last-Event-ID), which the server keeps on disk. A job that
// generated an admin password comes back behind it: the stream's 401 asks the page for the sign-in
// and the same stream resumes after it.

import { html, nothing } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import type { SetupJob, SetupJobEvent, SetupJobStep } from '../api/types';
import { ResumableStream, type StreamState } from '../lib/sse';
import { LightElement } from '../ui/base';
import { icon } from '../ui/bits';
import { apiBase } from '../api/client';
import { rememberJob, rememberStatusPort, restarting } from '../lib/restart';
import './restart-card';
import { setupApi, setupUnauthorized } from './api';
import { appendLog, applyJobEvent, changeText, copyText, dedupes, finishedLines, isFinal, type JobLog, needsJob, STALE_MS, streamAction, streamFinished } from './job-view';
import { S } from './strings';
import { shortTokens } from './window';

@customElement('qse-install-progress')
export class QseInstallProgress extends LightElement {
  @property() jobId: string | null = null;
  @property({ attribute: false }) initial: SetupJob | null = null;
  /** the profile the box ran before (for "rolled back to …") */
  @property({ attribute: false }) previousProfile: string | null = null;
  @property({ attribute: false }) onBack: (() => void) | null = null;
  /** level 2 of the reset shows no Playground button */
  @property({ type: Boolean }) compact = false;

  @state() private job: SetupJob | null = null;
  @state() private log: JobLog = { lines: [], dropped: 0 };
  @state() private conn: StreamState = 'closed';
  @state() private awaySince: number | null = null;
  @state() private lost = false;
  @state() private signIn = false;
  @state() private tick = 0;
  /** VIS-211: no job in hand for STALE_MS: the page says it cannot read the install */
  @state() private stale = false;
  /** VIS-213: Retry and Copy diagnostics on a failed job */
  @state() private retrying = false;
  @state() private retryError: string | null = null;
  @state() private copy: 'idle' | 'busy' | 'copied' | 'shown' | 'error' = 'idle';
  @state() private diag: string | null = null;
  private stream: ResumableStream | null = null;
  private connectedAt = 0;
  private reading = false;
  private healthAt = 0;
  private timer: ReturnType<typeof setInterval> | null = null;
  private seenIds = new Set<string>();
  private finalSent = false;

  connectedCallback(): void {
    super.connectedCallback();
    this.timer = setInterval(() => this.everySecond(), 1000);
    window.addEventListener('qse-signed-in', this.onSignedIn);
  }
  disconnectedCallback(): void {
    publishActive(null);
    window.removeEventListener('qse-signed-in', this.onSignedIn);
    this.stream?.close();
    if (this.timer) clearInterval(this.timer);
    super.disconnectedCallback();
  }

  protected willUpdate(changed: Map<string, unknown>): void {
    if (changed.has('initial') && this.initial && !this.job) this.job = this.initial;
    if (changed.has('jobId')) this.connect();
  }

  private onSignedIn = (): void => {
    if (!this.signIn) return;
    this.signIn = false;
    this.stream?.start(); // from its Last-Event-ID
  };

  /** VIS-211: once a second: the away clock; a page with no job in hand for STALE_MS says so; while
   *  the engine is away or no job is in hand, /health every 3 s, and when it answers the job is read
   *  by id and the stream opened again, so the true end shows within seconds of the engine's return */
  private everySecond(): void {
    if (this.awaySince != null) this.tick++;
    if (!this.jobId || isFinal(this.job)) return;
    if (!this.job && Date.now() - this.connectedAt > STALE_MS) this.stale = true;
    if ((this.awaySince != null || !this.job || this.lost) && Date.now() - this.healthAt >= 3000) {
      this.healthAt = Date.now();
      void fetch(apiBase() + '/health', { cache: 'no-store', credentials: 'same-origin' })
        .then(async (r) => {
          if (!r.ok) return;
          rememberStatusPort(await r.json().catch(() => null));
          void this.readJob(true);
        })
        .catch(() => undefined);
    }
  }

  /** VIS-211: the job by id (GET ?format=json): a page that lost the thread re-attaches. `reopen`:
   *  the stream starts again from its last id too (the engine is back). */
  private async readJob(reopen = false): Promise<void> {
    if (!this.jobId || this.reading) return;
    this.reading = true;
    const id = this.jobId;
    try {
      const j = await setupApi.job(id);
      if (id !== this.jobId) return;
      this.job = isFinal(this.job) && !isFinal(j) ? this.job : j;
      this.stale = false;
      this.lost = false;
      this.awaySince = null;
      this.final();
      if (reopen && !isFinal(this.job) && this.stream && this.stream.state !== 'open') this.stream.start();
    } catch {
      /* not readable now: the clock decides when to say so */
    } finally {
      this.reading = false;
    }
  }

  private final(): void {
    publishActive(this.job);
    if (isFinal(this.job) && !this.finalSent) {
      this.finalSent = true;
      this.stream?.close();
      this.dispatchEvent(new CustomEvent('job-final', { detail: this.job, bubbles: true }));
    }
  }

  private retryRead = (): void => {
    this.stale = false;
    this.connectedAt = Date.now();
    void this.readJob(true);
    if (!this.stream || this.stream.state !== 'open') this.stream?.start();
  };

  private connect(): void {
    this.stream?.close();
    this.stream = null;
    this.lost = false;
    this.stale = false;
    this.seenIds.clear();
    this.log = { lines: [], dropped: 0 };
    this.finalSent = false;
    this.connectedAt = Date.now();
    if (!this.jobId) return;
    const id = this.jobId;
    this.stream = new ResumableStream({
      url: () => setupApi.jobUrl(id),
      minBackoffMs: 1000,
      maxBackoffMs: 8000,
      onEvent: (ev) => {
        let data: SetupJobEvent;
        try {
          data = JSON.parse(ev.data);
        } catch {
          return;
        }
        // VIS-211: only log lines are deduped by id; the job's state events are idempotent
        if (ev.id && dedupes(data)) {
          if (this.seenIds.has(ev.id)) return;
          this.seenIds.add(ev.id);
        }
        if (data.type === 'log') {
          this.log = appendLog(this.log, { line: data.line, ts: data.ts });
          void this.updateComplete.then(() => {
            const el = this.querySelector('.inst-log');
            if (el) el.scrollTop = el.scrollHeight;
          });
          return;
        }
        if (needsJob(this.job, data)) void this.readJob();
        this.job = applyJobEvent(this.job, data);
        if (this.job) this.stale = false;
        this.final();
        // VIS-151: read on to the server's done event, which follows the log and the final job
        if (streamFinished(data, this.job)) this.stream?.close();
      },
      onState: (s, detail) => {
        this.conn = s;
        const act = streamAction(s, detail, this.job);
        if (act === 'signin') {
          this.signIn = true;
          this.awaySince = null;
          setupUnauthorized();
        } else if (act === 'open') {
          this.awaySince = null;
          if (!this.job) void this.readJob();
        } else if (act === 'lost') {
          this.lost = true;
          this.awaySince = null;
          this.stream?.close();
        } else if (act === 'close') {
          this.stream?.close();
          if (!isFinal(this.job)) void this.readJob();
        }
        else if (act === 'away') this.awaySince ??= Date.now();
      },
    });
    this.stream.start();
  }

  private stepIcon(s: SetupJobStep) {
    if (s.state === 'done') return icon('check');
    if (s.state === 'failed') return icon('x');
    if (s.state === 'running') return html`<span class="inst-spin" aria-hidden="true"></span>`;
    if (s.state === 'waiting') return html`<span class="inst-dot is-waiting" aria-hidden="true"></span>`;
    return html`<span class="inst-dot" aria-hidden="true"></span>`;
  }

  /** VIS-213: a failed or rolled back job's way on: what tandem doctor repaired, Retry from the
   *  failed step (a new job; the download is not repeated), Copy diagnostics, Back to Review */
  private failedEnd(j: SetupJob) {
    const failed = j.steps.find((s) => s.state === 'failed');
    const fixed = j.doctor?.repaired ?? [];
    return html`<div class="inst-fail" data-job-failed-end>
      ${failed ? html`<p class="inst-fail-step" data-failed-step=${failed.id}>${S.failedAt(failed.label)}</p>` : nothing}
      ${fixed.length
        ? html`<div class="inst-fixed" data-doctor-repaired><p class="small">${S.doctorRepaired}</p><ul>${fixed.map((t) => html`<li>${icon('check')} <span>${t}</span></li>`)}</ul></div>`
        : nothing}
      <div class="inst-actions inst-fail-actions">
        <button class="btn btn-primary" data-job-retry ?disabled=${this.retrying} @click=${() => void this.retry(j)}>${this.retrying ? S.retrying : failed && failed.id !== 'check' ? S.retryFrom(failed.label) : S.retryJob}</button>
        <button class="btn" data-copy-diagnostics ?disabled=${this.copy === 'busy'} @click=${() => void this.copyDiagnostics(j)}>${this.copy === 'copied' ? S.diagCopied : S.copyDiagnostics}</button>
        ${this.onBack ? html`<button class="btn btn-ghost" @click=${() => this.onBack?.()}>${S.backToReview}</button>` : nothing}
      </div>
      ${this.retryError ? html`<p class="rv-warn" role="alert" data-retry-error>${this.retryError}</p>` : nothing}
      ${this.copy === 'error' ? html`<p class="rv-warn" role="alert">${S.diagError}</p>` : nothing}
      ${this.copy === 'shown' && this.diag
        ? html`<div class="inst-diag"><p class="small muted">${S.diagSelect}</p><textarea class="inst-diag-text" readonly rows="10" data-diagnostics-text @focus=${(e: Event) => (e.target as HTMLTextAreaElement).select()}>${this.diag}</textarea></div>`
        : nothing}
    </div>`;
  }

  private async retry(j: SetupJob): Promise<void> {
    this.retrying = true;
    this.retryError = null;
    try {
      const res = await setupApi.retry(j.id);
      this.dispatchEvent(new CustomEvent('job-retry', { detail: res.job, bubbles: true }));
    } catch (e) {
      this.retryError = S.retryFailed((e as Error).message);
    } finally {
      this.retrying = false;
    }
  }

  private async copyDiagnostics(j: SetupJob): Promise<void> {
    this.copy = 'busy';
    try {
      this.diag = await setupApi.diagnostics(j.id);
    } catch {
      this.copy = 'error';
      return;
    }
    this.copy = (await copyText(this.diag)) ? 'copied' : 'shown';
  }

  /** VIS-209: when the restart began: the restart step's start, else when the page saw it go */
  private restartSince(j: SetupJob): number {
    const st = j.steps.find((s) => (s.id === 'switch' || s.id === 'rollback') && (s.state === 'running' || s.state === 'waiting'));
    return st?.started_at ? st.started_at * 1000 : (this.awaySince ?? Date.now());
  }

  /** VIS-155: the plan's settings under the check step, old -> new, collapsed with the count */
  private changeList(j: SetupJob) {
    return html`<details class="inst-changes" data-job-changes>
      <summary class="muted small">${S.jobChanges(j.changes.length)}</summary>
      <ul>
        ${j.changes.map((c) => {
          const t = changeText(c, S.reviewUnset);
          return html`<li data-env=${c.key} class=${c.removed ? 'is-removed' : ''}><code>${c.key}</code> <span class="num">${t.from} → ${t.to}</span></li>`;
        })}
      </ul>
    </details>`;
  }

  /** VIS-211: a finished job's screen: what is ready, what it set up, how long it took, and the way
   *  on (Start chatting, the Setup Guide) */
  private finished(j: SetupJob) {
    const f = finishedLines(j, shortTokens);
    return html`<div class="inst-done" data-job-finished>
      <span class="inst-badge" aria-hidden="true">${icon(j.warnings?.length ? 'warn' : 'check')}</span>
      <h2 class="inst-done-title">${f.title}</h2>
      <ul class="inst-done-facts">${f.lines.map((l) => html`<li>${l}</li>`)}</ul>
      ${j.warnings?.length || j.notes?.length
        ? html`<ul class="inst-roles" data-job-warnings>
            ${(j.warnings ?? []).map((w) => html`<li class="is-warn">${icon('warn')} <span>${w}</span></li>`)}
            ${(j.notes ?? []).map((w) => html`<li>${icon('info')} <span>${w}</span></li>`)}
          </ul>`
        : nothing}
      <div class="inst-done-actions">
        <a class="btn btn-primary btn-lg" href="#/chat" data-start-chatting @click=${() => safeMoment()}>${S.startChatting}</a>
        <a class="btn" href="#/setup" data-back-to-guide @click=${() => {
          safeMoment();
          this.onBack?.();
        }}>${S.backToGuide}</a>
        ${this.compact ? nothing : html`<a class="btn btn-ghost" href="#/performance">${S.openDashboard}</a>`}
      </div>
    </div>`;
  }

  render() {
    void this.tick;
    const j = this.job;
    const final = isFinal(j);
    if (j?.state === 'done') {
      return html`<div class="inst is-done" data-job-state="done">
        ${this.finished(j)}
        <details class="inst-logbox">
          <summary>${S.logTail} <span class="muted small num" data-log-count>${this.log.lines.length + this.log.dropped}</span></summary>
          <ol class="inst-steps">${j.steps.map((st) => html`<li class="inst-step is-${st.state}" data-step=${st.id}><span class="inst-icon">${this.stepIcon(st)}</span><span class="inst-label">${st.label}</span><span class="inst-state muted small">${S.stepState[st.state]}</span></li>`)}</ol>
          <pre class="inst-log" tabindex="0">${this.log.lines.map((l) => l.line).join('\n')}</pre>
        </details>
      </div>`;
    }
    const away = this.awaySince != null && !final;
    // VIS-209: the restart is one card; the steps and the log fold under it
    const card = !final && !!j && (restarting(j) || away);
    const stateWord = j ? (j.kind === 'reset' ? S.resetJobState : S.jobState)[j.state] : S.installStarting;
    const failedStep = j?.steps.find((s) => s.state === 'failed');
    const stopped = card && j!.steps.some((s) => (s.id === 'switch' || s.id === 'rollback') && (s.state === 'running' || s.state === 'waiting'));
    return html`<div class="inst ${j ? `is-${j.state}` : ''}" data-job-state=${j?.state ?? 'starting'}>
      <div class="inst-head" role="status" aria-live="polite" ?hidden=${stopped}>
        <span class="inst-badge" aria-hidden="true">${final || (this.stale && !j) ? icon('warn') : html`<span class="inst-spin"></span>`}</span>
        <div>
          <p class="inst-title">${stateWord}</p>
          ${j?.state === 'rolled_back' ? html`<p class="inst-sub" data-rolled-back>${S.rolledBackTo(this.previousProfile ?? S.previousSetup)}</p>` : nothing}
          ${(j?.state === 'failed' || j?.state === 'rolled_back') && j.message ? html`<p class="inst-sub" data-job-reason>${S.jobReason(j.message)}</p>` : nothing}
        </div>
      </div>
      ${card ? html`<qse-restart-card .job=${j} .since=${this.restartSince(j!)} @engine-up=${() => void this.readJob(true)}></qse-restart-card>` : nothing}
      ${away && !card
        ? html`<div class="inst-away" role="status" data-restarting>
            <span class="inst-spin" aria-hidden="true"></span>
            <div><p>${S.restarting}</p><p class="muted small num">${S.restartingSince((Date.now() - this.awaySince!) / 1000)}</p></div>
          </div>`
        : nothing}
      ${this.lost ? html`<div class="inst-away is-lost" role="alert">${icon('warn')} ${S.jobLost}</div>` : nothing}
      ${this.stale && !this.job
        ? html`<div class="inst-away is-lost" role="alert" data-job-stale>${icon('warn')}
            <div><p>${S.jobStale}</p><div class="inst-actions"><button class="btn btn-sm btn-primary" @click=${this.retryRead}>${S.retry}</button><button class="btn btn-sm" @click=${() => location.reload()}>${S.refreshPage}</button></div></div>
          </div>`
        : nothing}
      ${this.signIn ? html`<div class="inst-away" role="status" data-job-signin>${icon('warn')} <span>${S.jobSignIn}</span> <a class="btn btn-sm" href="#/setup">${S.signIn.go}</a></div>` : nothing}
      <ol class="inst-steps">
        ${(j?.steps ?? []).map(
          (s) => html`<li class="inst-step is-${s.state}" data-step=${s.id}>
            <span class="inst-icon">${this.stepIcon(s)}</span>
            <span class="inst-label">${s.label}</span>
            <span class="inst-state muted small">${S.stepState[s.state]}${s.started_at && s.ended_at ? html` <span class="num">${Math.max(0, s.ended_at - s.started_at).toFixed(1)} s</span>` : nothing}</span>
            ${s.detail && !(s.id === 'check' && j?.changes?.length) ? html`<span class="inst-detail muted small">${s.detail}</span>` : nothing}
            ${s.id === 'check' && j?.changes?.length ? this.changeList(j) : nothing}
          </li>`,
        )}
      </ol>
      <details class="inst-logbox" ?open=${!!failedStep || !final}>
        <summary>${S.logTail} <span class="muted small num" data-log-count>${this.log.lines.length + this.log.dropped}</span></summary>
        ${this.log.dropped ? html`<p class="muted small" data-log-dropped>${S.logDropped(this.log.dropped)}</p>` : nothing}
        <pre class="inst-log" tabindex="0">${this.log.lines.map((l) => l.line).join('\n')}</pre>
      </details>
      ${final ? this.failedEnd(j!) : nothing}
    </div>`;
  }
}

/** VIS-211: the page-wide state of a setup job (the app hides the new-version banner while one runs) */
function publishActive(job: SetupJob | null): void {
  rememberJob(job);
  window.dispatchEvent(new CustomEvent('qse-setup-job', { detail: { active: !!job && !isFinal(job), id: job?.id ?? null } }));
}

/** VIS-211: a moment when a waiting new build of the dashboard can load without losing anything */
function safeMoment(): void {
  window.dispatchEvent(new CustomEvent('qse-safe-moment'));
}
