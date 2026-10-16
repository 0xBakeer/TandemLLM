// <qse-setup>: the Setup Guide at #/setup (D1, D4, D5, D10, VIS-78). Tiles for every model in the
// registry, then eight steps with a rail, Next and Previous, deep links #/setup/<model>/<step>, the
// sticky memory bar, and Wren docked beside it (wren.ts). Every change goes through setupReducer
// (state.ts), from a click or from one of Wren's tool calls, and is saved to tandem.setup.v1.
//
// The memory bar is computed in the page on every change (estimate.ts) and confirmed by
// POST /v1/setup/estimate (150 ms debounce, the stale request aborted); the server's numbers win
// when they differ by more than 0.05 GiB. Review asks POST /v1/setup/plan for the serve.env diff and
// the commands. Finish puts Wren to sleep, posts /v1/setup/apply and follows the job.
//
// A 401 from any setup route (a box with an admin password) puts the sign-in card in place of the
// steps (signin.ts); the progress stays, and after the sign-in every reading is asked again.

import { html, nothing, render as litRender, type TemplateResult } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import { ApiError, describeError } from '../api/client';
import type { CommunitySync, SetupDevice, SetupEstimate, SetupField, SetupJob, SetupModel, SetupModelsList, SetupPlan, SetupProfile, SetupSelection, SetupStepId, SetupToolCall, SetupValue } from '../api/types';
import { poll, type Poller } from '../lib/poll';
import { LightElement, type Loadable } from '../ui/base';
import { emptyState as emptyPanel, errorState, icon, skeleton } from '../ui/bits';
import { setupApi } from './api';
import { buildPageState, howToFix, overSentence, planFixStep, stepIssues, tokenBlocked, type StepIssues } from './checks';
import { communityProfiles, ownProfiles, packNoImage, sourceLine, withCommunityProfiles } from './community';
import './community-view';
import { availability, isOn } from './deps';
import { estimateLoose as localEstimate, overBy, profileOf } from './estimate';
import { fieldError, isStepId, STEP_IDS, stepIndex } from './fields';
import { MINI_WREN, reducedMotion } from './figure';
import './install';
import { changeText } from './job-view';
import './measure-table';
import './memory-bar';
import './reset';
import './signin';
import { clearSaved, emptyState, readSaved, safeStorage, setupReducer, windowMax, writeSaved, type LoadProblem, type ParallelKey, type SetupAction, type WizardState } from './state';
import { parseSlots, slotsBlocker, slotsRange, slotsVerdict, stepSlots, type SlotsParse } from './slots';
import { S, STEP_INTRO, STEP_LABEL, STEP_SHORT, STEP_TITLE } from './strings';
import { fieldKey, planToolCall } from './tools';
import './wren';
import '../styles/setup.css';
import type { QseWren, ToolOutcome, WrenHost } from './wren';

const WINDOW_PRESETS = [8_192, 32_768, 65_536, 131_072, 262_144];
const kTok = (n: number) => (n >= 1024 ? `${Math.round(n / 1024)}k` : String(n));

type SecChoice = 'keep' | 'generate';

@customElement('qse-setup')
export class QseSetup extends LightElement {
  /** the route below #/setup: [] tiles, [model], [model, step] */
  @property({ attribute: false }) path: string[] = [];

  @state() private wiz: WizardState = emptyState();
  @state() private storageProblem: LoadProblem = null;
  @state() private storageFailed = false;
  @state() private list: Loadable<SetupModelsList> = { state: 'loading' };
  @state() private model: Loadable<SetupModel> | null = null;
  @state() private device: SetupDevice | null = null;
  @state() private serverEst: { key: string; est: SetupEstimate } | null = null;
  @state() private estLocal = false;
  @state() private plan: Loadable<SetupPlan> | null = null;
  @state() private confirmOpen = false;
  @state() private applying = false;
  @state() private applyError: string | null = null;
  @state() private jobInitial: SetupJob | null = null;
  @state() private previousProfile: string | null = null;
  @state() private resetOpen = false;
  @state() private unauthorized = false;
  @state() private winNarrow = false;
  /** VIS-199: what is typed in the parallel-chats field while it is not a count the engine takes
   *  (empty, 0, 2.5, 40): kept as typed with the reason, never clamped; null when the field shows
   *  the selection's count */
  @state() private slotsDraft: string | null = null;
  /** Next was pressed while the count stops it: the reason shows beside Next until it is fixed */
  @state() private nextRefused = false;
  /** the community list's sync with GitHub (CONTRACT.md addition 3: on the weights step's open and on Refresh) */
  @state() private communitySyncing = false;

  private store: Storage | null = null;
  private devPoll: Poller | null = null;
  private estTimer: ReturnType<typeof setTimeout> | null = null;
  private estCtrl: AbortController | null = null;
  private planTimer: ReturnType<typeof setTimeout> | null = null;
  private planCtrl: AbortController | null = null;
  private commCtrl: AbortController | null = null;
  private warnedDiff = false;
  private touchedAt = new Map<string, number>();
  private loadingModel: string | null = null;

  connectedCallback(): void {
    super.connectedCallback();
    this.store = safeStorage();
    const r = readSaved(this.store);
    this.wiz = r.state;
    this.storageProblem = r.problem;
    if (r.problem === 'corrupt' || r.problem === 'version') clearSaved(this.store);
    void this.loadList();
    this.devPoll = poll(() => this.loadDevice(), 5000);
    this.devPoll.start();
    window.addEventListener('qse-setup-reset', this.onReset);
    window.addEventListener('qse-setup-unauthorized', this.onUnauthorized);
    window.addEventListener('qse-signed-in', this.onSignedIn);
  }
  disconnectedCallback(): void {
    this.devPoll?.stop();
    this.estCtrl?.abort();
    this.planCtrl?.abort();
    this.commCtrl?.abort();
    window.removeEventListener('qse-setup-reset', this.onReset);
    window.removeEventListener('qse-setup-unauthorized', this.onUnauthorized);
    window.removeEventListener('qse-signed-in', this.onSignedIn);
    super.disconnectedCallback();
  }

  private onUnauthorized = (): void => {
    this.unauthorized = true;
  };

  /** signed in: everything the 401 stopped is asked again */
  private onSignedIn = (): void => {
    this.unauthorized = false;
    if (this.list.state !== 'ready') void this.loadList();
    const id = this.path[0];
    if (id && this.model?.state !== 'ready') void this.loadModel(id);
    void this.loadDevice();
    if (this.wiz.selection) this.queueEstimate();
    if (this.wiz.step === 'review') this.queuePlan();
    if (this.wiz.step === 'weights') void this.syncCommunity();
  };

  private onReset = (): void => {
    // the reset panel stays open: its dialog shows the result
    this.wiz = emptyState();
    this.plan = null;
    this.jobInitial = null;
    if (location.hash !== '#/setup') location.hash = '#/setup';
  };

  // ---- data ---------------------------------------------------------------------------------

  private async loadList(): Promise<void> {
    this.list = { state: 'loading' };
    try {
      const l = await setupApi.models();
      this.list = l.models.length || l.errors.length ? { state: 'ready', data: l } : { state: 'empty', note: S.modelsEmpty };
    } catch (e) {
      const d = describeError(e);
      this.list = { state: 'error', endpoint: d.endpoint, message: d.status === 401 ? S.unauthorized : d.message };
    }
  }

  private async loadModel(id: string): Promise<void> {
    if (this.loadingModel === id) return;
    this.loadingModel = id;
    this.model = { state: 'loading' };
    try {
      const m = withCommunityProfiles(await setupApi.model(id));
      this.model = { state: 'ready', data: m };
      await this.openModel(m, this.routeStep() ?? undefined);
    } catch (e) {
      const d = describeError(e);
      this.model = { state: 'error', endpoint: d.endpoint, message: d.status === 404 ? `${S.modelsError} (${id})` : d.message };
    } finally {
      this.loadingModel = null;
    }
  }

  private async loadDevice(): Promise<void> {
    try {
      this.device = await setupApi.device();
    } catch {
      /* the bar keeps its last reading */
    }
  }

  /** open a model: this browser's progress for it, else what the box runs now (the server's
   *  reading of serve.env), else the recommended setup */
  private async openModel(m: SetupModel, step: SetupStepId | undefined): Promise<void> {
    let seed: SetupSelection | null = null;
    if (!(this.wiz.model === m.id && this.wiz.selection)) {
      try {
        seed = (await setupApi.current(m.id)).selection;
      } catch {
        seed = null; // an older server, or none: the recommended setup
      }
    }
    this.dispatch({ type: 'open', model: m, step, seed }, false);
    if (this.wiz.step === 'weights') void this.syncCommunity();
  }

  private get m(): SetupModel | null {
    return this.model?.state === 'ready' ? this.model.data : null;
  }

  private routeStep(): SetupStepId | null {
    const s = this.path[1];
    return isStepId(s) ? s : null;
  }

  // ---- routing: the hash and the state agree ------------------------------------------------

  protected willUpdate(changed: Map<string, unknown>): void {
    if (changed.has('path')) this.syncFromRoute();
  }

  private syncFromRoute(): void {
    const [id, stepRaw] = this.path;
    if (!id) {
      if (this.wiz.step !== 'models') this.wiz = { ...this.wiz, step: 'models' };
      return;
    }
    const m = this.m;
    if (!m || m.id !== id) {
      void this.loadModel(id);
      return;
    }
    if (this.wiz.model !== id || !this.wiz.selection) {
      void this.openModel(m, isStepId(stepRaw) ? stepRaw : undefined);
      return;
    }
    let step: SetupStepId = isStepId(stepRaw) ? stepRaw : 'overview';
    if (step === 'install' && !this.wiz.job) step = 'review';
    if (step === 'models') step = 'overview';
    if (step !== this.wiz.step) this.dispatch({ type: 'go', step }, false);
  }

  private hashFor(st: WizardState): string {
    return st.step === 'models' || !st.model ? '#/setup' : `#/setup/${st.model}/${st.step}`;
  }

  /** every change: reduce, save, keep the hash in step (a push, so Back walks the steps) */
  private dispatch(a: SetupAction, push = true): void {
    const next = setupReducer(this.wiz, a, this.m);
    if (next === this.wiz) return;
    const stepChanged = next.step !== this.wiz.step;
    this.wiz = next;
    if (this.store) this.storageFailed = !writeSaved(this.store, next);
    const h = this.hashFor(next);
    if (location.hash !== h) {
      if (push) location.hash = h;
      else history.replaceState(null, '', h);
    }
    if (stepChanged || a.type === 'recommended' || (a.type === 'parallel' && a.key === 'slots')) {
      this.slotsDraft = null;
      this.nextRefused = false;
    }
    if (stepChanged) {
      if (push) window.scrollTo({ top: 0, behavior: reducedMotion() ? 'auto' : 'smooth' });
      if (next.step === 'review') this.queuePlan();
      if (next.step === 'weights') void this.syncCommunity();
    }
    if (a.type !== 'go' && a.type !== 'next' && a.type !== 'previous' && a.type !== 'chat' && a.type !== 'dismiss') {
      this.queueEstimate();
      if (next.step === 'review') this.queuePlan();
    }
  }

  /** a user's own change: recorded, so a tool call that arrives later for the same field is stale */
  private user(step: SetupStepId, field: string, a: SetupAction): void {
    this.touchedAt.set(fieldKey(step, field), Date.now());
    this.dispatch(a);
  }

  // ---- the estimate and the plan ------------------------------------------------------------

  private selKey(sel: SetupSelection): string {
    return JSON.stringify(sel);
  }

  private queueEstimate(): void {
    if (this.estTimer) clearTimeout(this.estTimer);
    this.estTimer = setTimeout(() => void this.confirmEstimate(), 150);
  }

  private async confirmEstimate(): Promise<void> {
    const sel = this.wiz.selection;
    if (!sel) return;
    this.estCtrl?.abort();
    const ctrl = (this.estCtrl = new AbortController());
    const key = this.selKey(sel);
    try {
      const est = await setupApi.estimate(sel, ctrl.signal);
      if (ctrl.signal.aborted || key !== this.selKey(this.wiz.selection!)) return;
      this.serverEst = { key, est };
      this.estLocal = false;
      const local = this.m ? localEstimate(this.m, sel) : null;
      if (local && Math.abs(local.total_gb - est.total_gb) > 0.05 && !this.warnedDiff) {
        this.warnedDiff = true;
        console.warn(`setup: the server's estimate (${est.total_gb} GiB) and the page's (${local.total_gb} GiB) differ; showing the server's`);
      }
    } catch (e) {
      if ((e as Error).name === 'AbortError' || ctrl.signal.aborted) return;
      this.estLocal = true;
    }
  }

  /** the estimate the page shows: the server's for this exact selection when it differs, else the page's */
  private currentEstimate(): SetupEstimate | null {
    const m = this.m;
    const sel = this.wiz.selection;
    if (!m || !sel) return null;
    const local = localEstimate(m, sel, this.device?.mem_available_gb ?? null);
    const srv = this.serverEst;
    if (srv && srv.key === this.selKey(sel) && (!local || Math.abs(srv.est.total_gb - local.total_gb) > 0.05)) return { ...srv.est, available_gb: this.device?.mem_available_gb ?? srv.est.available_gb };
    return local;
  }

  private queuePlan(): void {
    if (this.planTimer) clearTimeout(this.planTimer);
    this.planTimer = setTimeout(() => void this.loadPlan(), 200);
  }

  private async loadPlan(): Promise<void> {
    const sel = this.wiz.selection;
    if (!sel) return;
    this.planCtrl?.abort();
    const ctrl = (this.planCtrl = new AbortController());
    if (!this.plan || this.plan.state !== 'ready') this.plan = { state: 'loading' };
    try {
      const p = await setupApi.plan(sel, ctrl.signal);
      if (!ctrl.signal.aborted) this.plan = { state: 'ready', data: p };
    } catch (e) {
      if ((e as Error).name === 'AbortError' || ctrl.signal.aborted) return;
      const d = describeError(e);
      this.plan = { state: 'error', endpoint: d.endpoint, message: d.message };
    }
  }

  // ---- the community list -------------------------------------------------------------------

  /** fetch the community packs from GitHub now (POST /v1/setup/community/sync: one conditional
   *  GET; the server keeps its cache and the bundled copy for when GitHub does not answer); the
   *  list from GET models/{id} stays on screen meanwhile. A pack the selection named that is gone
   *  falls back to the recommended weights, with the note. */
  private async syncCommunity(): Promise<void> {
    const m = this.m;
    if (!m) return;
    this.commCtrl?.abort();
    const ctrl = (this.commCtrl = new AbortController());
    this.communitySyncing = true;
    try {
      const r = await setupApi.communitySync(ctrl.signal);
      if (ctrl.signal.aborted || this.m?.id !== m.id) return;
      const cur = this.m!;
      // the body covers every model: this model's entries, its dropped ones, its own Add yours link
      const sync: CommunitySync = { ...r.sync, dropped: r.sync.dropped.filter((d) => d.model === m.id), add_url: cur.community_sync?.add_url || r.sync.add_url };
      const fresh = withCommunityProfiles({ ...cur, community_sync: sync }, r.models[m.id] ?? []);
      this.model = { state: 'ready', data: fresh };
      const sel = this.wiz.selection;
      if (sel && !fresh.profiles.some((p) => p.id === sel.profile)) this.dispatch({ type: 'open', model: fresh }, false);
      else this.queueEstimate();
    } catch (e) {
      if ((e as Error).name === 'AbortError' || ctrl.signal.aborted) return;
      // an older server (404) has no list to sync; a failed sync keeps the list and says why
      const cur = this.m;
      if (cur?.community_sync && !(e instanceof ApiError && e.status === 404)) this.model = { state: 'ready', data: { ...cur, community_sync: { ...cur.community_sync, error: describeError(e).message, checked_at: new Date().toISOString() } } };
    } finally {
      if (!ctrl.signal.aborted) this.communitySyncing = false;
    }
  }

  // ---- Wren's side --------------------------------------------------------------------------

  private host: WrenHost = {
    pageState: () => buildPageState(this.m, this.wiz.step, this.wiz.selection, this.currentEstimate(), { errors: this.visibleErrors() }),
    execute: (call, t) => this.executeTool(call, t),
    saveChat: (chat) => this.dispatch({ type: 'chat', chat }),
    memAvailable: () => this.device?.mem_available_gb ?? null,
  };

  /** what the step shows as wrong: its own checks, and on Review the plan's errors */
  private visibleErrors(): string[] {
    const own = this.issues()[this.wiz.step] ?? [];
    if (this.wiz.step !== 'review' || this.plan?.state !== 'ready') return own;
    return [...own, ...this.plan.data.errors.filter((e) => !own.includes(e))];
  }

  private async executeTool(call: SetupToolCall, turnStartedAt: number): Promise<ToolOutcome> {
    const plan = planToolCall(call, { model: this.m, selection: this.wiz.selection, step: this.wiz.step, touchedAt: this.touchedAt, turnStartedAt });
    if (!plan.ok) return { ok: false, result: plan.error, summary: null, undo: null, stale: plan.stale };
    if (plan.kind === 'read_device') {
      try {
        const d = await setupApi.device();
        this.device = d;
        const brief = { mode: d.mode, mem_total_gb: d.mem_total_gb, mem_available_gb: d.mem_available_gb, gpu: d.gpu, disk: d.disk, engine: d.engine, active_profile: d.active_profile, profiles_on_disk: d.profiles_on_disk, password_set: d.password_set };
        return { ok: true, result: JSON.stringify(brief), summary: null, undo: null };
      } catch (e) {
        return { ok: false, result: describeError(e).message, summary: null, undo: null };
      }
    }
    if (plan.kind === 'explain') {
      if (plan.target.step === this.wiz.step) void this.highlight(plan.target.step, plan.target.field);
      return { ok: true, result: plan.result, summary: null, undo: null };
    }
    const { target } = plan;
    // show the field Wren changes: open its step first
    if (target.field && target.step !== this.wiz.step) this.dispatch({ type: 'go', step: target.step });
    for (const a of plan.actions) this.dispatch(a);
    await this.updateComplete;
    if (target.field) void this.highlight(target.step, target.field);
    const undo = plan.undo.length
      ? () => {
          for (const a of plan.undo) this.dispatch(a);
          if (target.field) void this.highlight(target.step, target.field);
        }
      : null;
    // the phone's Show me: the sheet is closed by then; open the field's step if the user left it
    const reveal = target.field
      ? () => {
          if (target.step !== this.wiz.step) this.dispatch({ type: 'go', step: target.step });
          void this.highlight(target.step, target.field);
        }
      : null;
    return { ok: true, result: plan.result, summary: plan.actions.length ? plan.summary : null, undo, reveal };
  }

  /** the field pulses for 1.2 s and the mini gold box hops onto it; offscreen, it scrolls into view */
  private async highlight(step: SetupStepId, field: string | null): Promise<void> {
    if (!field) return;
    await this.updateComplete;
    const el = this.querySelector<HTMLElement>(`[data-field="${CSS.escape(fieldKey(step, field))}"]`);
    if (!el) return;
    const r = el.getBoundingClientRect();
    if (r.top < 120 || r.bottom > innerHeight - 80) el.scrollIntoView({ block: 'center', behavior: reducedMotion() ? 'auto' : 'smooth' });
    el.setAttribute('data-wren-acting', '');
    el.querySelector(':scope > .wren-mini')?.remove();
    const mini = document.createElement('span');
    mini.className = 'wren-mini';
    mini.setAttribute('aria-hidden', 'true');
    el.append(mini);
    litRender(MINI_WREN, mini);
    setTimeout(() => {
      el.removeAttribute('data-wren-acting');
      mini.remove();
    }, 1600);
  }

  // ---- Finish -------------------------------------------------------------------------------

  private issues(): StepIssues {
    const m = this.m;
    const sel = this.wiz.selection;
    if (!m || !sel) return {};
    return stepIssues(m, sel, this.device, this.currentEstimate());
  }

  private finishBlockers(): string[] {
    const m = this.m;
    const sel = this.wiz.selection;
    if (!m || !sel) return [];
    const out: string[] = [];
    const est = this.currentEstimate();
    if (est && !est.fits) out.push(S.reviewOver(overBy(est)));
    if (tokenBlocked(m, sel, this.device)) out.push(S.blockedToken);
    const bad = (Object.keys(this.issues()) as SetupStepId[]).filter((s) => s !== 'review');
    if (bad.length) out.push(S.blockedInvalid(bad.map((s) => STEP_TITLE[s].toLowerCase())));
    // the plan's own errors show in its card; here only what is not said above already
    if (this.plan?.state === 'ready' && !this.plan.data.ok && !out.length) out.push(...this.plan.data.errors);
    return out;
  }

  private async finish(): Promise<void> {
    const sel = this.wiz.selection;
    if (!sel || this.applying || this.finishBlockers().length) return;
    this.applying = true;
    this.applyError = null;
    this.previousProfile = this.device?.active_profile ?? null;
    this.querySelector<QseWren>('qse-wren')?.sleepForInstall();
    const selection: SetupSelection = sel;
    try {
      const res = await setupApi.apply({ selection });
      this.jobInitial = res.job;
      this.confirmOpen = false;
      this.dispatch({ type: 'job', id: res.job.id });
    } catch (e) {
      this.applyError = this.applyErrorText(e);
    } finally {
      this.applying = false;
    }
  }

  /** apply's refusals (server/setup_api.py apply_route): another job, requests in flight, or the
   *  plan's own errors (409 over_budget, 412 missing_token, ...), whose plan comes along and
   *  replaces the review's */
  private applyErrorText(e: unknown): string | null {
    const d = describeError(e);
    if (d.status === 401) return null; // the sign-in card takes over
    const body = e instanceof ApiError ? (e.body as { error?: { running?: number }; plan?: SetupPlan } | undefined) : undefined;
    if (body?.plan) {
      this.plan = { state: 'ready', data: body.plan };
      this.confirmOpen = false;
      return body.plan.errors.join(' ') || d.message;
    }
    if (e instanceof ApiError && e.type === 'job_running') return S.blockedBusy;
    if (e instanceof ApiError && e.type === 'busy') return S.blockedRequests(body?.error?.running ?? 1);
    return `${S.installError} ${d.message}`;
  }

  // ---- render -------------------------------------------------------------------------------

  render() {
    const step = this.wiz.step;
    const m = this.m;
    const inModel = this.path.length > 0;
    return html`<div class="view view-setup" data-step=${inModel ? step : 'models'}>
      <div class="setup">
        <div class="setup-main">
          ${this.renderBanners()}
          ${this.unauthorized ? html`<qse-setup-signin></qse-setup-signin>` : inModel ? this.renderModelArea() : this.renderTiles()}
          <footer class="setup-foot">
            <button class="btn btn-ghost btn-sm" aria-expanded=${this.resetOpen} aria-controls="setup-reset" @click=${() => (this.resetOpen = !this.resetOpen)}>${icon('trash')} ${S.resetLink}</button>
          </footer>
          ${this.resetOpen ? html`<section class="panel setup-reset" id="setup-reset"><header class="panel-head"><h2 class="panel-title">${S.resetHeading}</h2></header><qse-reset inSetup></qse-reset></section>` : nothing}
        </div>
        <qse-wren class="setup-wren" .host=${this.host} .step=${inModel ? step : 'models'} .chat=${this.wiz.chat} .finished=${step === 'install' && !!this.wiz.job}></qse-wren>
      </div>
      ${m ? this.renderConfirm(m) : nothing}
    </div>`;
  }

  private renderBanners(): TemplateResult {
    const p = this.storageProblem;
    const msgs: string[] = [];
    if (p === 'unavailable' || this.storageFailed) msgs.push(S.storageOff);
    if (p === 'corrupt') msgs.push(S.storageCorrupt);
    if (p === 'version') msgs.push(S.storageOld);
    if (this.wiz.notice && this.wiz.step !== 'components' && this.wiz.step !== 'parallelism') msgs.push(this.wiz.notice);
    return html`${msgs.map((t) => html`<div class="banner setup-banner" role="status" data-setup-notice>${icon('info')} <span>${t}</span></div>`)}`;
  }

  private renderTiles(): TemplateResult {
    const l = this.list;
    const dev = this.device;
    const resume = this.wiz.model && this.wiz.selection ? this.wiz : null;
    return html`<section class="setup-tiles">
      <header class="setup-hero">
        <h1 class="setup-h1">${STEP_TITLE.models}</h1>
        <p class="setup-lead">${STEP_INTRO.models}</p>
        ${dev?.mem_total_gb != null ? html`<p class="setup-device muted num" data-device-line>${S.deviceLine(dev.mem_total_gb, dev.mem_available_gb)}</p>` : nothing}
      </header>
      ${l.state === 'loading' ? skeleton(4) : nothing}
      ${l.state === 'error' ? errorState(l.endpoint, `${S.modelsError} ${l.message}`, () => void this.loadList()) : nothing}
      ${l.state === 'empty' ? emptyPanel(S.modelsEmpty) : nothing}
      ${l.state === 'ready'
        ? html`<div class="mtiles">
              ${l.data.models.map((t) => {
                const resuming = resume?.model === t.id;
                const href = resuming ? `#/setup/${t.id}/${resume!.step === 'models' ? 'overview' : resume!.step}` : `#/setup/${t.id}/overview`;
                return html`<a class="mtile ${t.recommended ? 'is-rec' : ''}" href=${href} data-model=${t.id}>
                  <span class="mtile-top"><span class="mtile-family">${t.family}</span>${t.recommended ? html`<span class="sbadge sbadge-rec">${S.tileRecommended}</span>` : nothing}</span>
                  <span class="mtile-name">${t.name}</span>
                  <span class="mtile-params">${S.tileParams(t.params)}</span>
                  <span class="mtile-pitch">${t.pitch}</span>
                  ${t.community_count ? html`<span class="mtile-comm">${S.community.tileCount(t.community_count)}</span>` : nothing}
                  ${t.best_speed
                    ? html`<span class="mtile-speed" title=${S.tileSpeedTitle}><span class="mtile-speed-n num">${t.best_speed.tok_s.toFixed(1)}</span><span class="mtile-speed-u">tok/s</span><span class="mtile-speed-note">${t.best_speed.task}, ${t.best_speed.profile}</span></span>`
                    : nothing}
                  <span class="mtile-go"><span class="btn btn-primary">${resuming ? `${S.tileOpen}: ${STEP_LABEL[resume!.step]}` : S.tileOpen}</span></span>
                </a>`;
              })}
            </div>
            ${l.data.errors.length
              ? html`<div class="setup-load-errors" role="status"><p>${S.modelLoadErrors(l.data.errors.length)}</p><ul>${l.data.errors.map((e) => html`<li><code>${e.path}</code> <span class="muted">${e.message}</span></li>`)}</ul></div>`
              : nothing}`
        : nothing}
      <section class="guide" aria-labelledby="guide-h">
        <h2 class="guide-h" id="guide-h">${S.guideSteps}</h2>
        <ol class="guide-list">${STEP_IDS.slice(1).map((id) => html`<li><span class="guide-t">${STEP_LABEL[id]}</span><span class="guide-d">${STEP_SHORT[id]}</span></li>`)}</ol>
      </section>
    </section>`;
  }

  private renderModelArea(): TemplateResult {
    const lm = this.model;
    if (!lm || lm.state === 'loading') return skeleton(8);
    if (lm.state === 'error') return errorState(lm.endpoint, lm.message, () => void this.loadModel(this.path[0]));
    if (lm.state !== 'ready' || !this.wiz.selection) return skeleton(8);
    const m = lm.data;
    const sel = this.wiz.selection;
    const est = this.currentEstimate();
    const step = this.wiz.step;
    const over = est && !est.fits ? overSentence(m, sel, est) : null;
    return html`
      <div class="setup-sticky">
        <qse-memory-bar .model=${m} .estimate=${est} .available=${this.device?.mem_available_gb ?? null} .local=${this.estLocal} .overText=${over}></qse-memory-bar>
        ${this.estLocal ? html`<p class="setup-local" role="status" data-local-estimate>${S.memLocal}</p>` : nothing}
      </div>
      ${this.renderRail(m)}
      <section class="setup-step" data-step=${step} aria-labelledby="setup-step-title">
        <header class="step-head">
          <p class="step-kicker muted num">${S.stepOf(stepIndex(step) + 1, STEP_IDS.length)}</p>
          <h1 class="step-title" id="setup-step-title" tabindex="-1">${STEP_TITLE[step]}</h1>
          <p class="step-intro">${STEP_INTRO[step]}</p>
        </header>
        ${this.renderStep(m, sel, est)}
      </section>
      ${this.renderNav()}
    `;
  }

  private renderRail(m: SetupModel): TemplateResult {
    const cur = this.wiz.step;
    const issues = this.issues();
    const i = stepIndex(cur);
    return html`<nav class="rail-steps" aria-label="Setup steps">
      <div class="rail-phone" aria-hidden="true">
        <span class="num">${S.stepOf(i + 1, STEP_IDS.length)}</span><span class="rail-phone-label">${STEP_TITLE[cur]}</span>
        <span class="rail-phone-bar"><span style="width:${((i + 1) / STEP_IDS.length) * 100}%"></span></span>
      </div>
      <ol class="steps">
        ${STEP_IDS.map((id, n) => {
          const visited = this.wiz.visited.includes(id);
          const bad = visited && (issues[id]?.length ?? 0) > 0;
          const isCur = id === cur;
          const done = visited && !bad && n < i;
          const reachable = id !== 'install' || !!this.wiz.job;
          const href = id === 'models' ? '#/setup' : `#/setup/${m.id}/${id}`;
          return html`<li class="steps-item ${isCur ? 'is-current' : ''} ${done ? 'is-done' : ''} ${bad ? 'is-bad' : ''} ${visited ? 'is-visited' : ''}" data-rail=${id}>
            ${reachable
              ? html`<a href=${href} aria-current=${isCur ? 'step' : nothing} title=${bad ? issues[id]!.join(' ') : STEP_TITLE[id]}>
                  <span class="steps-node num" aria-hidden="true">${bad ? '!' : done ? icon('check') : n + 1}</span>
                  <span class="steps-label">${STEP_LABEL[id]}</span>
                  ${bad ? html`<span class="sr-only">(needs a fix)</span>` : nothing}
                </a>`
              : html`<span class="steps-link is-locked" aria-disabled="true"><span class="steps-node num" aria-hidden="true">${n + 1}</span><span class="steps-label">${STEP_LABEL[id]}</span></span>`}
          </li>`;
        })}
      </ol>
    </nav>`;
  }

  private renderNav(): TemplateResult | typeof nothing {
    const step = this.wiz.step;
    if (step === 'install') return nothing;
    const i = stepIndex(step);
    const why = this.nextRefused ? this.nextBlocker() : null;
    return html`<nav class="setup-nav ${why ? 'has-why' : ''}" aria-label="Step navigation">
      ${why ? html`<p class="setup-nav-why" id="setup-nav-why" role="alert" data-next-refused>${icon('info')} <span>${S.nextRefused(why)}</span></p>` : nothing}
      <button class="btn setup-prev" @click=${() => (i <= 1 ? this.dispatch({ type: 'tiles' }) : this.dispatch({ type: 'previous' }))}>${S.previous}</button>
      <span class="setup-nav-mid muted small num">${S.stepOf(i + 1, STEP_IDS.length)}</span>
      ${step === 'review'
        ? html`<button class="btn btn-primary setup-finish" ?disabled=${this.finishBlockers().length > 0 || this.plan?.state !== 'ready'} @click=${() => (this.confirmOpen = true)}>${S.finish}</button>`
        : html`<button class="btn btn-primary setup-next" aria-describedby=${why ? 'setup-nav-why' : nothing} @click=${() => this.next()}><span>${S.next}<span class="setup-next-to">: ${STEP_LABEL[STEP_IDS[i + 1]]}</span></span></button>`}
    </nav>`;
  }

  /** VIS-199: Next refuses a parallel-chats count the engine does not take or that does not fit,
   *  says why beside it and puts the focus on the field */
  private nextBlocker(): string | null {
    const m = this.m;
    const sel = this.wiz.selection;
    if (this.wiz.step !== 'parallelism' || !m || !sel) return null;
    return slotsBlocker(slotsVerdict(m, sel), this.slotsParse(m));
  }

  private slotsParse(m: SetupModel): SlotsParse | null {
    return this.slotsDraft == null ? null : parseSlots(this.slotsDraft, slotsRange(m));
  }

  private next(): void {
    if (this.nextBlocker()) {
      this.nextRefused = true;
      this.querySelector<HTMLInputElement>('#pf-slots-in')?.focus();
      return;
    }
    this.nextRefused = false;
    this.dispatch({ type: 'next' });
  }

  private renderStep(m: SetupModel, sel: SetupSelection, est: SetupEstimate | null): TemplateResult {
    switch (this.wiz.step) {
      case 'overview':
        return this.stepOverview(m, sel);
      case 'weights':
        return this.stepWeights(m, sel);
      case 'components':
        return this.stepComponents(m, sel);
      case 'parallelism':
        return this.stepParallel(m, sel, est);
      case 'methods':
        return this.stepMethods(m, sel);
      case 'review':
        return this.stepReview(m, sel, est);
      case 'install':
        return this.stepInstall();
      default:
        return html``;
    }
  }

  // ---- step 1: the model --------------------------------------------------------------------

  private stepOverview(m: SetupModel, sel: SetupSelection): TemplateResult {
    // SRV-337: the selected profile's largest window, or that it fits none
    const prof = m.profiles.find((p) => p.id === sel.profile);
    const longest = windowMax(m, sel.profile);
    return html`<div class="ov">
      <div class="ov-facts">
        <p class="ov-arch">${m.architecture}</p>
        <dl class="ov-dl">
          <div><dt>${S.colWeights}</dt><dd>${m.profiles.length} sizes, ${m.profiles.map((p) => p.label.replace(/ bpw$/, '')).join(', ')}</dd></div>
          <div><dt>${S.window}</dt><dd class="num">${prof?.max_window === null ? S.windowNone(prof.label) : `${longest.toLocaleString('en-US')} ${S.tokensUnit}`}</dd></div>
          <div><dt>${S.license}</dt><dd>${m.license}</dd></div>
        </dl>
        <ul class="ov-links">${m.links.map((l) => html`<li><a href=${l.url} target="_blank" rel="noopener noreferrer">${l.label} ${icon('external')}</a></li>`)}</ul>
      </div>
      <section class="panel ov-table" aria-label="Measured speed">
        <qse-measure-table .model=${m} .selected=${sel.profile}></qse-measure-table>
      </section>
    </div>`;
  }

  // ---- step 2: weights ----------------------------------------------------------------------

  private stepWeights(m: SetupModel, sel: SetupSelection): TemplateResult {
    const dev = this.device;
    const blocked = tokenBlocked(m, sel, dev);
    const rows = m.measurements.rows;
    const fastest = Math.max(...rows.filter((r) => r.comparable).map((r) => r.cells.greedy_512?.code?.tok_s ?? 0));
    return html`
      ${blocked ? html`<div class="banner banner-offline" role="alert" data-token-blocked>${icon('warn')} ${S.privateNoToken}</div>` : nothing}
      ${!blocked && dev?.hf_token_present && m.profiles.some((p) => p.hf_private && !dev.profiles_on_disk?.[p.id]) ? html`<p class="wnote muted small">${icon('info')} ${S.privateRepo}</p>` : nothing}
      <div class="wcards" role="radiogroup" aria-label=${STEP_TITLE.weights}>
        ${ownProfiles(m).map((p) => {
          const on = sel.profile === p.id;
          const row = rows.find((r) => r.profile === p.id);
          const g = row?.cells.greedy_512;
          const here = dev?.profiles_on_disk?.[p.id];
          const active = dev?.active_profile === p.id;
          const top1 = p.quality.top1_pct;
          return html`<label class="wcard ${on ? 'is-selected' : ''} ${p.recommended ? 'is-rec' : ''}" data-profile-card=${p.id} data-field=${on ? 'weights.profile' : nothing}>
            <input class="sr-only" type="radio" name="profile" .checked=${on} value=${p.id} @change=${() => this.user('weights', 'profile', { type: 'profile', id: p.id })} />
            <span class="wcard-head">
              <span class="wcard-radio" aria-hidden="true"></span>
              <span class="wcard-name">${p.label}</span>
              ${active ? html`<span class="sbadge sbadge-now">${S.activeNow}</span>` : nothing}
            </span>
            ${p.badges.length ? html`<span class="wcard-badges">${p.badges.map((b) => html`<span class="sbadge ${b === 'recommended' ? 'sbadge-rec' : ''}">${b === 'recommended' ? S.recommended : b}</span>`)}</span>` : nothing}
            <span class="wcard-meters">
              <span class="wmeter" title=${p.quality.source}>
                <span class="wmeter-k">${S.weightsTop1}</span>
                <span class="wmeter-v num">${top1 != null ? `${top1.toFixed(1)} %` : '–'}</span>
                <span class="wmeter-bar q"><span style="width:${top1 != null ? Math.max(4, ((top1 - 80) / 20) * 100) : 0}%"></span></span>
              </span>
              <span class="wmeter">
                <span class="wmeter-k">${S.weightsSpeed}</span>
                <span class="wmeter-v num">${g?.prose && g?.code ? html`${g.prose.tok_s.toFixed(1)} / ${g.code.tok_s.toFixed(1)}` : html`<span class="muted">${row && !row.comparable ? S.notComparable : S.notMeasured}</span>`}</span>
                <span class="wmeter-bar s"><span style="width:${g?.code && row?.comparable ? (g.code.tok_s / fastest) * 100 : 0}%"></span></span>
              </span>
            </span>
            <span class="wcard-facts num">
              <span><span class="wcard-k">${S.weightsKl}</span> ${p.quality.kl_vs_bf16 != null ? p.quality.kl_vs_bf16.toFixed(4) : '–'}</span>
              <span><span class="wcard-k">${S.weightsMemory}</span> ${p.weights_gib.toFixed(1)} GiB</span>
              <span><span class="wcard-k">${S.weightsDisk}</span> ${here ? S.onDisk : S.toDownload(p.disk_gb)}</span>
            </span>
            ${this.renderSource(p)}
            ${p.note ? html`<span class="wcard-note">${p.note}</span>` : nothing}
            ${p.hf_private && !here && dev?.hf_token_present === false ? html`<span class="wcard-private is-blocked" data-private>${icon('warn')} ${S.privateNoTokenShort}</span>` : nothing}
          </label>`;
        })}
      </div>
      ${communityProfiles(m).length || m.community
        ? html`<qse-community .model=${m} .sync=${m.community_sync ?? null} .selected=${sel.profile} .device=${dev} .syncing=${this.communitySyncing} .onUse=${(id: string) => this.user('weights', 'profile', { type: 'profile', id })} .onRefresh=${() => void this.syncCommunity()}></qse-community>`
        : nothing}`;
  }

  /** VIS-175: the weights' Hugging Face repository as owner/name, linked, and who made them */
  private renderSource(p: SetupProfile): TemplateResult {
    const src = sourceLine(p);
    return html`<span class="wcard-src" data-weights-src=${src.kind}>
      <span class="sbadge wsrc-kind wsrc-${src.kind}">${src.label}</span>
      <span class="wsrc-by">${S.madeBy} ${src.byUrl ? html`<a href=${src.byUrl} target="_blank" rel="noopener noreferrer" title=${S.community.github(src.by.slice(1))}>${src.by}</a>` : src.by}</span>
      ${src.repoUrl ? html`<a class="wsrc-repo" href=${src.repoUrl} target="_blank" rel="noopener noreferrer" title=${S.community.onHf(src.repo)}>${src.repo} ${icon('external')}</a>` : html`<span class="wsrc-repo">${src.repo}</span>`}
    </span>`;
  }

  // ---- step 3: components -------------------------------------------------------------------

  private stepComponents(m: SetupModel, sel: SetupSelection): TemplateResult {
    const av = availability(m, sel.components);
    if (av.vision && packNoImage(m, sel.profile)) av.vision = { disabled: true, reason: S.community.visionOff, required: false };
    const groups = [...new Set(m.components.map((c) => c.group))];
    const prof = profileOf(m, sel.profile);
    return html`
      <div class="step-tools">
        <button class="btn btn-sm" @click=${() => this.dispatch({ type: 'recommended', scope: 'components' })}>${icon('refresh')} ${S.resetToRecommended}</button>
      </div>
      <p class="comp-notice" role="status" aria-live="polite" data-cascade>${this.wiz.notice ?? ''}</p>
      ${groups.map(
        (g) => html`<section class="comp-group" aria-labelledby=${`cg-${g}`}>
          <h2 class="comp-group-title" id=${`cg-${g}`}>${S.groups[g] ?? g}</h2>
          <ul class="comp-list">
            ${m.components
              .filter((c) => c.group === g)
              .map((c) => {
                const on = isOn(c, sel.components);
                const a = av[c.id];
                const mem = c.id === 'drafters' && prof.drafters_gib != null ? prof.drafters_gib : c.memory_gib;
                return html`<li>
                  <label class="comp ${on ? 'is-on' : ''} ${a.disabled ? 'is-disabled' : ''} ${a.required ? 'is-required' : ''}" data-field=${`components.${c.id}`} data-component=${c.id}>
                    <input type="checkbox" class="comp-input" .checked=${on} ?disabled=${a.disabled} aria-describedby=${`cw-${c.id}${a.reason ? ` cr-${c.id}` : ''}`} @change=${(e: Event) => this.user('components', c.id, { type: 'component', id: c.id, on: (e.target as HTMLInputElement).checked })} />
                    <span class="comp-box" aria-hidden="true">${icon('check')}</span>
                    <span class="comp-main">
                      <span class="comp-label">${c.label}${c.default && !c.required ? html` <span class="comp-rec">${S.recommendedValue}</span>` : nothing}</span>
                      <span class="comp-what" id=${`cw-${c.id}`}>${c.what}</span>
                      ${a.reason ? html`<span class="comp-reason" id=${`cr-${c.id}`} data-reason>${a.reason}</span>` : nothing}
                    </span>
                    <span class="comp-cost num">
                      <span class="comp-mem">${S.memoryCost(mem)}</span>
                      ${c.memory_max_gib != null ? html`<span class="muted">${S.memoryUpTo(c.memory_max_gib)}</span>` : nothing}
                      ${c.disk_gb > 0 ? html`<span class="muted">${S.diskCost(c.disk_gb)}</span>` : nothing}
                    </span>
                  </label>
                </li>`;
              })}
          </ul>
        </section>`,
      )}`;
  }

  // ---- step 4: parallelism and memory -------------------------------------------------------

  private stepParallel(m: SetupModel, sel: SetupSelection, est: SetupEstimate | null): TemplateResult {
    const p = m.parallelism;
    const side = sel.side_window ?? sel.window;
    const fit = est?.context_tokens_fit ?? 0;
    const perSlot = Math.floor(fit / sel.slots);
    const set = (key: ParallelKey, value: number | boolean) => this.user('parallelism', key, { type: 'parallel', key, value });
    // the main window's presets above the profile's largest stay, disabled with the reason (SRV-337);
    // the side window's above the main window are left out
    const prof = m.profiles.find((x) => x.id === sel.profile);
    const winCtl = (key: 'window' | 'side_window', value: number, max: number) => html`<div class="win-ctl">
      <div class="seg seg-wrap" role="radiogroup" aria-label=${key === 'window' ? S.window : S.sideWindow}>
        ${WINDOW_PRESETS.filter((w) => w >= p[key].min && w <= (key === 'window' ? p.window.max : max)).map((w) => {
          const over = w > max;
          return html`<button class="seg-btn ${value === w ? 'is-on' : ''}" role="radio" aria-checked=${value === w} ?disabled=${over} title=${over && prof ? S.windowAboveProfile(prof.label, max) : nothing} @click=${() => set(key, w)}>${kTok(w)}</button>`;
        })}
      </div>
      <input class="num-input num" type="number" inputmode="numeric" min=${p[key].min} max=${max} step=${p[key].step} .value=${String(value)} aria-label=${`${key === 'window' ? S.window : S.sideWindow} (${S.tokensUnit})`} @change=${(e: Event) => set(key, Number((e.target as HTMLInputElement).value))} />
      <span class="muted small">${S.tokensUnit}</span>
    </div>`;
    return html`<div class="spar">
      <div class="spar-fields">
        ${this.slotsField(m, sel)}
        <div class="pfield" data-field="parallelism.window">
          <div class="pfield-k"><span class="pfield-label">${S.window}</span><span class="pfield-what">${S.windowWhat}</span></div>
          ${winCtl('window', sel.window, windowMax(m, sel.profile))}
        </div>
        ${sel.slots > 1
          ? html`<div class="pfield" data-field="parallelism.side_window">
              <div class="pfield-k"><span class="pfield-label">${S.sideWindow}</span><span class="pfield-what">${S.sideWindowWhat(sel.slots)}</span></div>
              ${winCtl('side_window', side, sel.window)}
            </div>`
          : nothing}
        <div class="pfield" data-field="parallelism.kv_pool">
          <div class="pfield-k"><span class="pfield-label" id="pf-pool">${S.kvPool}</span><span class="pfield-what">${S.kvPoolWhat}</span></div>
          <label class="switch"><input type="checkbox" role="switch" .checked=${sel.kv_pool} aria-labelledby="pf-pool" @change=${(e: Event) => set('kv_pool', (e.target as HTMLInputElement).checked)} /><span>${sel.kv_pool ? 'on' : 'off'}</span></label>
        </div>
        <div class="pfield" data-field="parallelism.floor_gb">
          <div class="pfield-k"><span class="pfield-label" id="pf-floor">${S.floor}</span><span class="pfield-what">${S.floorWhat}</span></div>
          <div class="floor-ctl">
            <input type="range" min=${p.floor_gb.min} max=${p.floor_gb.max} step="1" .value=${String(sel.floor_gb)} aria-labelledby="pf-floor" @input=${(e: Event) => set('floor_gb', Number((e.target as HTMLInputElement).value))} />
            <output class="num">${sel.floor_gb} GiB</output>
          </div>
        </div>
        <div class="step-tools"><button class="btn btn-sm" @click=${() => this.dispatch({ type: 'recommended', scope: 'parallelism' })}>${icon('refresh')} ${S.resetToRecommended}</button></div>
        ${this.wiz.notice ? html`<p class="comp-notice" role="status">${this.wiz.notice}</p>` : nothing}
      </div>
      <aside class="spar-readout panel" aria-live="polite" data-fit-tokens=${fit}>
        <p class="spar-big num">${fit.toLocaleString('en-US')}</p>
        <p class="spar-big-u">${S.fitsUnit}</p>
        <div class="spar-slots" aria-hidden="true">
          ${[
            { n: '1', w: sel.window, on: true },
            { n: sel.slots > 2 ? `2–${sel.slots}` : '2', w: side, on: sel.slots > 1 },
          ].map(
            (r) => html`<div class="spar-slot ${r.on ? '' : 'is-off'}">
              <span class="spar-slot-n num">${r.n}</span>
              <span class="spar-slot-track"><span class="spar-slot-win" style="width:${(r.w / p.window.max) * 100}%"></span><span class="spar-slot-fit" style="width:${(Math.min(r.w, perSlot) / p.window.max) * 100}%"></span></span>
              <span class="spar-slot-k num">${kTok(r.w)}</span>
            </div>`,
          )}
        </div>
        <p class="small">${S.fitsPerSlot(perSlot, sel.slots)}</p>
        ${est ? html`<p class="small muted">${S.worstCase(est.kv_worst_gb, est.fits_worst ?? false)}</p>` : nothing}
      </aside>
    </div>`;
  }

  /** VIS-199: parallel chats as a stepper, − / a number field / +. Typing, the buttons and the arrow
   *  keys change the selection at once, so the memory bar and the readout follow on every change.
   *  No card cap: above the card's measured range is a quiet note, a count that does not fit or that
   *  the engine does not take says why here, in the bar, and beside Next, which refuses it. */
  private slotsField(m: SetupModel, sel: SetupSelection): TemplateResult {
    const r = slotsRange(m);
    const v = slotsVerdict(m, sel);
    const draft = this.slotsParse(m);
    const typedBad = draft && !draft.ok ? draft.why : null;
    const err = typedBad ?? (v.over ? S.slotsOver(v.n, v.overGb, v.fitMax) : null);
    const set = (n: number) => this.user('parallelism', 'slots', { type: 'parallel', key: 'slots', value: n });
    const step = (d: number) => {
      const base = draft?.ok ? draft.n : sel.slots;
      const n = stepSlots(base, d, r);
      if (n !== sel.slots) set(n);
      else this.slotsDraft = null;
    };
    const onInput = (e: Event) => {
      const raw = (e.target as HTMLInputElement).value;
      const p = parseSlots(raw, r);
      if (p.ok) {
        this.slotsDraft = null;
        if (p.n !== sel.slots) set(p.n);
      } else this.slotsDraft = raw;
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
        e.preventDefault();
        step(e.key === 'ArrowUp' ? 1 : -1);
      }
    };
    const shown = this.slotsDraft ?? String(sel.slots);
    const atMax = sel.slots >= r.max && !typedBad;
    return html`<div class="pfield pfield-slots ${err ? 'is-bad' : ''}" data-field="parallelism.slots">
      <div class="pfield-k"><label class="pfield-label" id="pf-slots" for="pf-slots-in">${S.slots}</label><span class="pfield-what">${S.slotsWhat}</span></div>
      <div class="stepper ${err ? 'is-bad' : ''}" role="group" aria-labelledby="pf-slots">
        <button class="stepper-btn" type="button" aria-label=${S.slotsFewer} aria-controls="pf-slots-in" ?disabled=${sel.slots <= r.min && !typedBad} @click=${() => step(-1)}>
          <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3.5 8h9" /></svg>
        </button>
        <input
          id="pf-slots-in"
          class="stepper-in num"
          type="text"
          inputmode="numeric"
          enterkeyhint="done"
          autocomplete="off"
          spellcheck="false"
          role="spinbutton"
          aria-valuemin=${r.min}
          aria-valuemax=${r.max}
          aria-valuenow=${draft && !draft.ok ? nothing : sel.slots}
          aria-valuetext=${draft && !draft.ok ? nothing : `${sel.slots} ${sel.slots === 1 ? 'chat' : 'chats'}`}
          aria-invalid=${err ? 'true' : 'false'}
          aria-describedby="pf-slots-say"
          maxlength="3"
          .value=${shown}
          @input=${onInput}
          @keydown=${onKey}
          @focus=${(e: FocusEvent) => (e.target as HTMLInputElement).select()}
        />
        <button class="stepper-btn" type="button" aria-label=${S.slotsMore} aria-controls="pf-slots-in" ?disabled=${atMax} title=${atMax ? S.slotsAtMax(r.max) : nothing} @click=${() => step(1)}>
          <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3.5 8h9M8 3.5v9" /></svg>
        </button>
      </div>
      <div class="slots-say" id="pf-slots-say">
        ${err ? html`<p class="slots-err" role="alert" data-slots-error>${err}</p>` : nothing}
        <p class="slots-cost" data-slots-cost>${S.slotsCost(v.cost.now, v.cost.full, v.cost.window)}</p>
        ${v.untested && !typedBad ? html`<p class="slots-note" data-slots-untested>${S.slotsUntested(r.measured)}</p>` : nothing}
      </div>
    </div>`;
  }

  // ---- step 5: methods and advanced ---------------------------------------------------------

  private fieldControl(group: 'methods' | 'advanced', f: SetupField, value: SetupValue): TemplateResult {
    const set = (v: SetupValue) => this.user('methods', f.id, { type: 'field', group, id: f.id, value: v });
    const id = `mf-${f.id}`;
    if (f.type === 'enum') {
      const opts = f.options ?? [];
      if (opts.length <= 3)
        return html`<div class="seg seg-wrap" role="radiogroup" aria-labelledby=${`${id}-l`}>${opts.map((o) => html`<button class="seg-btn ${value === o.value ? 'is-on' : ''}" role="radio" aria-checked=${value === o.value} @click=${() => set(o.value)}>${o.label}</button>`)}</div>`;
      return html`<select id=${id} class="sel" @change=${(e: Event) => set((e.target as HTMLSelectElement).value)}>${opts.map((o) => html`<option value=${o.value} ?selected=${value === o.value}>${o.label}</option>`)}</select>`;
    }
    if (f.type === 'bool') return html`<label class="switch"><input id=${id} type="checkbox" role="switch" .checked=${value === true} @change=${(e: Event) => set((e.target as HTMLInputElement).checked)} /><span>${value ? 'on' : 'off'}</span></label>`;
    if (f.type === 'int' || f.type === 'number')
      return html`<input id=${id} class="num-input num" type="number" inputmode=${f.type === 'int' ? 'numeric' : 'decimal'} min=${f.min ?? nothing} max=${f.max ?? nothing} step=${f.type === 'int' ? 1 : 0.05} .value=${String(value)} @change=${(e: Event) => {
        const raw = (e.target as HTMLInputElement).value;
        set(raw === '' ? NaN : Number(raw));
      }} />`;
    return html`<input id=${id} class="text-input" type="text" .value=${String(value)} spellcheck="false" autocapitalize="off" @change=${(e: Event) => set((e.target as HTMLInputElement).value)} />`;
  }

  private fieldRow(group: 'methods' | 'advanced', f: SetupField, sel: SetupSelection): TemplateResult {
    const value = sel[group][f.id] ?? f.recommended;
    const why = fieldError(f, value);
    const rec = f.options?.find((o) => o.value === f.recommended)?.label ?? String(f.recommended);
    return html`<div class="mfield ${why ? 'is-bad' : ''}" data-field=${`methods.${f.id}`}>
      <div class="pfield-k">
        <label class="pfield-label" id=${`mf-${f.id}-l`} for=${`mf-${f.id}`}>${f.label}</label>
        <span class="pfield-what">${f.what}</span>
        <span class="mfield-rec muted small">${S.recommendedValue}: <span class="num">${rec}</span>${value !== f.recommended ? html` <button class="linkish" @click=${() => this.user('methods', f.id, { type: 'field', group, id: f.id, value: f.recommended })}>${S.resetToRecommended.toLowerCase()}</button>` : nothing}</span>
      </div>
      <div class="mfield-ctl">${this.fieldControl(group, f, value)}${why ? html`<p class="mfield-err" role="alert">${S.invalidValue(why)}</p>` : nothing}</div>
    </div>`;
  }

  private stepMethods(m: SetupModel, sel: SetupSelection): TemplateResult {
    const advBad = m.advanced.some((f) => fieldError(f, sel.advanced[f.id] ?? f.recommended));
    return html`<div class="methods">
      ${m.methods.map((f) => this.fieldRow('methods', f, sel))}
      <details class="adv" ?open=${advBad}>
        <summary><span class="adv-title">${S.advanced}</span> <span class="muted small">${S.advancedNote}</span></summary>
        ${m.advanced.map((f) => this.fieldRow('advanced', f, sel))}
      </details>
      <div class="step-tools"><button class="btn btn-sm" @click=${() => this.dispatch({ type: 'recommended', scope: 'methods' })}>${icon('refresh')} ${S.resetToRecommended}</button></div>
    </div>`;
  }

  // ---- step 6: review -----------------------------------------------------------------------

  private stepReview(m: SetupModel, sel: SetupSelection, est: SetupEstimate | null): TemplateResult {
    const plan = this.plan;
    const step = this.wiz.step;
    const fixes = est && !est.fits ? howToFix(m, sel) : [];
    const blockers = this.finishBlockers();
    const dev = this.device;
    const prof = profileOf(m, sel.profile);
    const ready = plan?.state === 'ready' ? plan.data : null;
    return html`<div class="review">
      <section class="rv-card rv-memory ${est && !est.fits ? 'is-over' : 'is-ok'}" data-review-memory>
        <h2 class="rv-h">${S.reviewMemory}</h2>
        ${est ? html`<p class="rv-say">${est.fits ? S.reviewFits(est.total_gb, est.budget_gb ?? est.device_gb - est.floor_gb) : overSentence(m, sel, est)}</p>` : nothing}
        ${fixes.length
          ? html`<h3 class="rv-sub">${S.howToFix}</h3>
              <ul class="rv-fixes">${fixes.map((f) => html`<li><span>${f.text}</span><span class="muted small num">−${f.saves.toFixed(1)} GiB</span><a class="btn btn-sm" href=${`#/setup/${m.id}/${f.step}`}>${S.fixGo}: ${STEP_LABEL[f.step]}</a></li>`)}</ul>`
          : nothing}
      </section>

      <section class="rv-card">
        <h2 class="rv-h">${S.reviewWeights}</h2>
        <p>${dev?.active_profile === prof.id ? S.reviewWeightsSame(prof.label) : S.reviewWeightsSwitch(dev?.active_profile ? (m.profiles.find((p) => p.id === dev.active_profile)?.label ?? dev.active_profile) : null, prof.label)}</p>
        ${ready && !ready.profile.on_disk ? html`<p class="muted">${S.reviewDownload(ready.profile.download_gb)}</p>` : nothing}
        ${ready?.downloads?.length
          ? html`<ul class="rv-downloads" data-review-downloads>${ready.downloads.map((d) => html`<li data-download=${d.id}><span>${d.label}</span> <span class="muted small">${d.for}</span> <span class="num">${d.gb.toFixed(2)} GB</span></li>`)}</ul>`
          : nothing}
        ${ready ? html`<p class="muted small">${ready.restart ? S.reviewRestart : S.reviewNoRestart}</p>` : nothing}
      </section>

      <section class="rv-card rv-changes">
        <h2 class="rv-h">${S.reviewChanges}</h2>
        ${plan == null || plan.state === 'loading' ? skeleton(4) : nothing}
        ${plan?.state === 'error' ? html`<p class="rv-warn" role="alert">${icon('warn')} ${S.planError} <span class="muted">${plan.message}</span> <button class="btn btn-sm" @click=${() => void this.loadPlan()}>${S.retry}</button></p>` : nothing}
        ${ready
          ? ready.changes.length
            ? html`<div class="table-scroll"><table class="table rv-table">
                <thead><tr><th scope="col">${S.reviewKey}</th><th scope="col">${S.reviewNow}</th><th scope="col">${S.reviewAfter}</th><th scope="col">${S.reviewFrom}</th></tr></thead>
                <tbody>${ready.changes.map((c) => html`<tr data-env=${c.key}><th scope="row"><code>${c.key}</code></th><td class="rv-old">${c.current ?? html`<span class="muted">${S.reviewUnset}</span>`}</td><td class="rv-new">${c.removed ? html`<span class="rv-removed" data-removed>${changeText(c, S.reviewUnset).to}</span>` : c.next ?? html`<span class="muted">${S.reviewUnset}</span>`}</td><td class="muted small">${c.from}</td></tr>`)}</tbody>
              </table></div>`
            : html`<p class="muted">${S.reviewNoChanges}</p>`
          : nothing}
        ${ready && (ready.errors.length || ready.warnings.length)
          ? html`<ul class="rv-notes">
              ${ready.errors.map((e, i) => {
                const code = ready.error_codes?.[i];
                const fix = planFixStep(code);
                return html`<li class="is-err" data-error-code=${code ?? nothing}>${icon('warn')} <span>${e}</span>${fix && fix !== step ? html` <a class="btn btn-sm" href=${`#/setup/${m.id}/${fix}`}>${S.planFix}: ${STEP_LABEL[fix]}</a>` : nothing}</li>`;
              })}
              ${ready.warnings.map((w) => html`<li>${icon('info')} ${w}</li>`)}
            </ul>`
          : nothing}
      </section>

      ${ready?.commands.length
        ? html`<section class="rv-card">
            <h2 class="rv-h">${S.reviewCommands}</h2>
            <p class="muted small">${S.reviewCommandsNote}</p>
            <pre class="rv-cmds">${ready.commands.join('\n')}</pre>
          </section>`
        : nothing}

      ${dev && !dev.password_set ? this.securityCard() : dev?.password_set ? html`<p class="rv-secure muted">${icon('check')} ${S.securitySet}</p>` : nothing}

      ${blockers.length ? html`<ul class="rv-blockers" role="status" data-finish-blocked>${blockers.map((b) => html`<li>${icon('warn')} ${b}</li>`)}</ul>` : nothing}
      ${this.applyError ? html`<p class="rv-warn" role="alert">${icon('warn')} ${this.applyError}</p>` : nothing}
    </div>`;
  }

  private get sec(): SecChoice {
    return this.wiz.selection?.advanced.admin_password === 'generate' ? 'generate' : 'keep';
  }

  /** the model has the Dashboard login field and it is off: after a generated password the setup
   *  routes want a session, and POST /v1/dashboard/session exists only with the login on */
  private loginOff(): boolean {
    const f = this.m?.advanced.find((x) => x.id === 'dashboard_login');
    return !!f && (this.wiz.selection?.advanced.dashboard_login ?? f.recommended) !== 'on';
  }

  private securityCard(): TemplateResult {
    const opt = (v: SecChoice, label: string, note?: string) => html`<label class="sec-opt ${this.sec === v ? 'is-on' : ''}">
      <input type="radio" name="sec" .checked=${this.sec === v} @change=${() => this.user('methods', 'admin_password', { type: 'field', group: 'advanced', id: 'admin_password', value: v })} />
      <span><span class="sec-opt-label">${label}</span>${note ? html`<span class="muted small">${note}</span>` : nothing}</span>
    </label>`;
    return html`<section class="rv-card rv-security" data-security-warning role="note">
      <h2 class="rv-h">${icon('warn')} ${S.securityTitle}</h2>
      <p>${S.securityBody}</p>
      <div class="sec-opts" role="radiogroup" aria-label=${S.securityChoice}>
        ${opt('keep', S.securityKeep)} ${opt('generate', S.securityGenerate, S.securityGenerateNote)}
      </div>
      ${this.sec === 'generate' && this.loginOff()
        ? html`<p class="sec-login" data-login-off>${icon('info')} <span>${S.securityLoginOff}</span>
            <button class="btn btn-sm" @click=${() => this.user('methods', 'dashboard_login', { type: 'field', group: 'advanced', id: 'dashboard_login', value: 'on' })}>${S.securityLoginOn}</button></p>`
        : nothing}
    </section>`;
  }

  private renderConfirm(m: SetupModel): TemplateResult {
    const restart = this.plan?.state === 'ready' ? this.plan.data.restart : true;
    return html`<dialog
      class="setup-confirm"
      aria-labelledby="sc-t"
      aria-describedby="sc-b"
      @cancel=${(e: Event) => this.applying && e.preventDefault()}
      @close=${() => (this.confirmOpen = false)}
    >
      ${this.confirmOpen
        ? html`<h2 id="sc-t">${S.confirmTitle}</h2>
            <p id="sc-b">${S.confirmBody(restart)}</p>
            <p class="muted small">${profileOf(m, this.wiz.selection!.profile).label}</p>
            ${this.applyError ? html`<p class="rv-warn" role="alert">${this.applyError}</p>` : nothing}
            <div class="setup-confirm-foot">
              <button class="btn" ?disabled=${this.applying} @click=${() => (this.confirmOpen = false)}>${S.cancel}</button>
              <button class="btn btn-primary setup-confirm-go" autofocus ?disabled=${this.applying} @click=${() => void this.finish()}>${this.applying ? S.installStarting : S.confirmGo}</button>
            </div>`
        : nothing}
    </dialog>`;
  }

  protected updated(): void {
    const d = this.querySelector<HTMLDialogElement>('dialog.setup-confirm');
    if (!d) return;
    if (this.confirmOpen && !d.open) d.showModal();
    else if (!this.confirmOpen && d.open) d.close();
  }

  // ---- step 7: install ----------------------------------------------------------------------

  private stepInstall(): TemplateResult {
    const id = this.wiz.job;
    if (!id) return html`<p class="muted">${S.installStarting}</p>`;
    return html`<qse-install-progress .jobId=${id} .initial=${this.jobInitial} .previousProfile=${this.previousProfile} .onBack=${() => this.dispatch({ type: 'job', id: null })}></qse-install-progress>`;
  }
}

