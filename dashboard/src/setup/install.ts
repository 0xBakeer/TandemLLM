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
import { setupApi, setupUnauthorized } from './api';
import { applyJobEvent, isFinal } from './job-view';
import { S } from './strings';

const LOG_KEEP = 300;

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
  @state() private log: { line: string; ts: number }[] = [];
  @state() private conn: StreamState = 'closed';
  @state() private awaySince: number | null = null;
  @state() private lost = false;
  @state() private signIn = false;
  @state() private tick = 0;
  private stream: ResumableStream | null = null;
  private timer: ReturnType<typeof setInterval> | null = null;
  private seenIds = new Set<string>();

  connectedCallback(): void {
    super.connectedCallback();
    this.timer = setInterval(() => {
      if (this.awaySince != null) this.tick++;
    }, 1000);
    window.addEventListener('qse-signed-in', this.onSignedIn);
  }
  disconnectedCallback(): void {
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

  private connect(): void {
    this.stream?.close();
    this.stream = null;
    this.lost = false;
    this.seenIds.clear();
    if (!this.jobId) return;
    const id = this.jobId;
    this.stream = new ResumableStream({
      url: () => setupApi.jobUrl(id),
      minBackoffMs: 1000,
      maxBackoffMs: 8000,
      onEvent: (ev) => {
        if (ev.id) {
          if (this.seenIds.has(ev.id)) return;
          this.seenIds.add(ev.id);
        }
        let data: SetupJobEvent;
        try {
          data = JSON.parse(ev.data);
        } catch {
          return;
        }
        if (data.type === 'log') {
          const next = [...this.log, { line: data.line, ts: data.ts }];
          this.log = next.length > LOG_KEEP ? next.slice(-LOG_KEEP) : next;
          void this.updateComplete.then(() => {
            const el = this.querySelector('.inst-log');
            if (el) el.scrollTop = el.scrollHeight;
          });
          return;
        }
        this.job = applyJobEvent(this.job, data);
        if (isFinal(this.job)) {
          this.stream?.close();
          this.dispatchEvent(new CustomEvent('job-final', { detail: this.job, bubbles: true }));
        }
      },
      onState: (s, detail) => {
        this.conn = s;
        if (s === 'unauthorized') {
          this.signIn = true;
          this.awaySince = null;
          setupUnauthorized();
        } else if (s === 'open') this.awaySince = null;
        else if ((s === 'reconnecting' || s === 'error') && !isFinal(this.job)) {
          this.awaySince ??= Date.now();
          if (detail && /HTTP 404/.test(detail)) {
            this.lost = true;
            this.stream?.close();
          }
        }
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

  render() {
    void this.tick;
    const j = this.job;
    const final = isFinal(j);
    const away = this.awaySince != null && !final;
    const stateWord = j ? (j.kind === 'reset' ? S.resetJobState : S.jobState)[j.state] : S.installStarting;
    const failedStep = j?.steps.find((s) => s.state === 'failed');
    return html`<div class="inst ${j ? `is-${j.state}` : ''}" data-job-state=${j?.state ?? 'starting'}>
      <div class="inst-head" role="status" aria-live="polite">
        <span class="inst-badge" aria-hidden="true">${j?.state === 'done' ? icon('check') : j && (j.state === 'failed' || j.state === 'rolled_back') ? icon('warn') : html`<span class="inst-spin"></span>`}</span>
        <div>
          <p class="inst-title">${stateWord}</p>
          ${j?.state === 'rolled_back' ? html`<p class="inst-sub" data-rolled-back>${S.rolledBackTo(this.previousProfile ?? S.previousSetup)}</p>` : nothing}
          ${(j?.state === 'failed' || j?.state === 'rolled_back') && j.message ? html`<p class="inst-sub" data-job-reason>${S.jobReason(j.message)}</p>` : nothing}
          ${j?.state === 'done' && j.message ? html`<p class="inst-sub">${j.message}</p>` : nothing}
        </div>
      </div>
      ${away
        ? html`<div class="inst-away" role="status" data-restarting>
            <span class="inst-spin" aria-hidden="true"></span>
            <div><p>${S.restarting}</p><p class="muted small num">${S.restartingSince((Date.now() - this.awaySince!) / 1000)}</p></div>
          </div>`
        : nothing}
      ${this.lost ? html`<div class="inst-away is-lost" role="alert">${icon('warn')} ${S.jobLost}</div>` : nothing}
      ${this.signIn ? html`<div class="inst-away" role="status" data-job-signin>${icon('warn')} <span>${S.jobSignIn}</span> <a class="btn btn-sm" href="#/setup">${S.signIn.go}</a></div>` : nothing}
      <ol class="inst-steps">
        ${(j?.steps ?? []).map(
          (s) => html`<li class="inst-step is-${s.state}" data-step=${s.id}>
            <span class="inst-icon">${this.stepIcon(s)}</span>
            <span class="inst-label">${s.label}</span>
            <span class="inst-state muted small">${S.stepState[s.state]}${s.started_at && s.ended_at ? html` <span class="num">${Math.max(0, s.ended_at - s.started_at).toFixed(1)} s</span>` : nothing}</span>
            ${s.detail ? html`<span class="inst-detail muted small">${s.detail}</span>` : nothing}
          </li>`,
        )}
      </ol>
      <details class="inst-logbox" ?open=${!!failedStep || !final}>
        <summary>${S.logTail} <span class="muted small num">${this.log.length}</span></summary>
        <pre class="inst-log" tabindex="0">${this.log.map((l) => l.line).join('\n')}</pre>
      </details>
      ${final
        ? html`<div class="inst-actions">
            ${j!.state === 'done'
              ? html`<a class="btn btn-primary" href="#/performance">${S.openDashboard}</a>${this.compact ? nothing : html`<a class="btn" href="#/playground">${S.openPlayground}</a>`}`
              : html`${this.onBack ? html`<button class="btn btn-primary" @click=${() => this.onBack?.()}>${S.backToReview}</button>` : nothing}`}
          </div>`
        : nothing}
    </div>`;
  }
}
