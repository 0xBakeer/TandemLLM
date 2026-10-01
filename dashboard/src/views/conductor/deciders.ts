// Conductor · Deciders (VIS-46): the decision models the engine serves itself (engine/decision,
// stage A of the decision-model design; Laya's checkpoints first). What each model is doing, Load
// and Unload through the residency planner (a refusal says why, with its numbers), what a load now
// would get, which model answers each question id the conductor asks (the `deciders` document,
// versioned like every other), and a test box for one raw /v1/systemone question. Questions run
// only while no request holds the engine in this stage, which the screen says.

import { html, nothing, type TemplateResult } from 'lit';
import { customElement, state } from 'lit/decorators.js';
import { ApiError, describeError } from '../../api/client';
import { conductor, type DeciderLoad, type DeciderModel, type DecidersDoc, type DecidersStatus, type SystemOneResult } from '../../api/conductor';
import { setIn } from '../../lib/conductor';
import { choiceQuestion, limitsWords, picksWords, plannerWords, probsOf, refusalWords, STATE_WORDS, stateBadge } from '../../lib/deciders';
import { exact, ms } from '../../lib/format';
import { poll, type Poller } from '../../lib/poll';
import { gib, gibOf } from '../../lib/residency';
import { LightElement } from '../../ui/base';
import { icon, panel, skeleton, statTile } from '../../ui/bits';
import { confirmDialog, DocEditor, errorsList, saveBar, withToken } from './shared';

const LOADS: DeciderLoad[] = ['on_start', 'on_demand', 'off'];

@customElement('qse-cond-deciders')
export class QseCondDeciders extends LightElement {
  @state() private st: DecidersStatus | null = null;
  @state() private error: string | null = null;
  @state() private busy: string | null = null;
  @state() private said: Record<string, string> = {};
  @state() private tState = 'The build fails since this morning: TypeError in src/parser.ts line 41 after the refactor.';
  @state() private tQuestion = 'What kind of work does this request ask for?';
  @state() private tOptions = 'debug, code_edit, plan, chat';
  @state() private tModel = 'auto';
  @state() private tOut: SystemOneResult | null = null;
  @state() private tErr: string | null = null;
  @state() private asking = false;
  private ed = new DocEditor<DecidersDoc>(this, 'deciders');
  private poller: Poller | null = null;

  connectedCallback(): void {
    super.connectedCallback();
    void this.ed.load();
    this.poller = poll(() => this.load(), 3_000);
    this.poller.start();
  }
  disconnectedCallback(): void {
    this.poller?.stop();
    super.disconnectedCallback();
  }

  private async load(): Promise<void> {
    try {
      this.st = await conductor.deciders();
      this.error = null;
    } catch (e) {
      const d = describeError(e);
      this.error = `${d.endpoint}: ${d.message}`;
    }
  }

  private async act(m: DeciderModel, what: 'load' | 'unload') {
    if (what === 'unload') {
      const ok = await confirmDialog(`Unload ${m.name}?`, [
        'Its weights and its memory pool go. Questions routed to it fall back to another loaded model of its family, or fail open (the conductor passes to Qwen).',
        'Loading it again goes through the residency planner.',
      ], 'Unload');
      if (!ok) return;
    }
    this.busy = m.name;
    this.said = { ...this.said, [m.name]: '' };
    try {
      const r = await withToken(() => (what === 'load' ? conductor.deciderLoad(m.name) : conductor.deciderUnload(m.name)));
      this.said = { ...this.said, [m.name]: what === 'load' ? `load started (${r.state ?? 'loading'})` : 'unloaded' };
    } catch (e) {
      const msg = e instanceof ApiError && e.status === 409 && what === 'load' ? `refused by the residency planner: ${e.message}` : describeError(e).message;
      this.said = { ...this.said, [m.name]: msg };
    } finally {
      this.busy = null;
      await this.load();
    }
  }

  private setLoad(name: string, load: DeciderLoad) {
    const d = this.ed.draft;
    if (!d) return;
    const i = d.models.findIndex((x) => x.name === name);
    const models = i >= 0 ? d.models.map((x, j) => (j === i ? { ...x, load } : x)) : [...d.models, { name, load }];
    this.ed.set({ ...d, models });
  }

  private async ask() {
    const q = choiceQuestion(this.tQuestion, this.tOptions);
    if (typeof q === 'string') {
      this.tErr = q;
      return;
    }
    this.asking = true;
    this.tErr = null;
    try {
      this.tOut = await conductor.deciderTest(this.tState, { q }, this.tModel);
    } catch (e) {
      this.tOut = null;
      this.tErr = describeError(e).message;
    } finally {
      this.asking = false;
    }
  }

  render() {
    const s = this.st;
    if (this.error && !s) return html`<div class="state state-error" role="alert"><div class="state-title">${icon('warn')} ${this.error}</div></div>`;
    if (!s) return skeleton(6);
    if (!s.enabled) {
      return panel(
        'Decision models',
        html`<div class="state" data-deciders-off>
          <div class="state-title">${icon('warn')} This engine serves no decision model</div>
          <p class="muted small">${s.why}</p>
          <p class="muted small">With models named, the conductor's decider role (endpoint <code>self</code>) asks them in process, <code>/v1/systemone</code> answers on the engine port behind the admin token, and this screen loads, unloads and tests them.</p>
        </div>`,
        { id: 'deciders-off' },
      );
    }
    const loaded = s.models.filter((m) => m.state === 'loaded');
    return html`
      <div class="stats" data-decider-tiles>
        ${statTile({ label: 'loaded', value: `${loaded.length} / ${s.models.length}`, sub: loaded.length ? loaded.map((m) => m.name).join(', ') : 'none: questions fail open', warn: !loaded.length })}
        ${statTile({ label: 'on the device', value: gib(loaded.reduce((a, m) => a + m.device_bytes, 0)), sub: `${s.device ?? 'no device'} · own stream and memory pool` })}
        ${statTile({ label: 'a load now', value: s.planner?.allowed ? 'fits' : 'refused', sub: plannerWords(s.planner), warn: !s.planner?.allowed })}
        ${statTile({ label: 'when they run', value: 'idle', sub: 'only while no request holds the engine (stage A)' })}
      </div>
      ${this.renderModels(s)}
      ${this.renderMemory(s)}
      ${this.renderQuestions(s)}
      ${this.renderTest(s)}
    `;
  }

  private renderModels(s: DecidersStatus): TemplateResult {
    const d = this.ed.draft;
    const policy = (n: string) => d?.models.find((x) => x.name === n)?.load ?? 'on_demand';
    return panel(
      'Models',
      html`<div class="table-scroll"><table class="table" data-deciders>
        <thead><tr><th>model</th><th>state</th><th>on start</th><th>limits <span class="muted small">ctx / question / options</span></th><th>precision</th><th class="num-col">questions</th><th class="num-col">forward p50 / p95</th><th class="num-col">errors</th><th></th></tr></thead>
        <tbody>${s.models.map((m) => {
          const refused = refusalWords(m.refusal);
          const say = this.said[m.name];
          const canUnload = m.state === 'loaded' || m.state === 'failed' || m.state === 'warming';
          return html`<tr data-decider-row=${m.name}>
              <td><strong>${m.name}</strong><div class="muted small">${m.family} · ${m.contract}${m.aliases.length ? html` · ${m.aliases.join(', ')}` : nothing}</div></td>
              <td><span class="badge ${stateBadge(m.state)}" data-decider-state=${m.state}>${STATE_WORDS[m.state]}</span>${m.warm === false ? html` <span class="muted small">not warmed</span>` : nothing}</td>
              <td><select aria-label=${`load policy of ${m.name}`} data-decider-load=${m.name} @change=${(e: Event) => this.setLoad(m.name, (e.target as HTMLSelectElement).value as DeciderLoad)}>
                ${LOADS.map((l) => html`<option value=${l} ?selected=${policy(m.name) === l}>${l.replace('_', ' ')}</option>`)}
              </select></td>
              <td class="num">${limitsWords(m)}</td>
              <td class="small">${m.precision ?? html`<span class="muted">—</span>`}<div class="muted small mono" title=${m.weight_hash ?? ''}>${m.weight_hash ? m.weight_hash.slice(0, 12) : ''}</div></td>
              <td class="num num-col">${exact(m.questions)}</td>
              <td class="num num-col">${m.stats?.forward_ms_p50 != null ? `${ms(m.stats.forward_ms_p50)} / ${ms(m.stats.forward_ms_p95)}` : '—'}</td>
              <td class="num num-col ${m.stats?.errors ? 'err-ink' : ''}">${m.stats ? exact(m.stats.errors + m.stats.deadline_missed) : '—'}</td>
              <td>${canUnload
                ? html`<button class="btn btn-sm" ?disabled=${this.busy === m.name} @click=${() => this.act(m, 'unload')} data-unload=${m.name}>Unload</button>`
                : html`<button class="btn btn-sm btn-primary" ?disabled=${this.busy === m.name || m.state === 'loading' || m.state === 'unloading'} @click=${() => this.act(m, 'load')} data-load=${m.name}>Load</button>`}</td>
            </tr>
            ${refused || m.error || say
              ? html`<tr class="is-warn" data-decider-note=${m.name}><td colspan="9" class="small">
                  ${refused ? html`<span class="warn-ink" data-refusal=${m.name}>${icon('warn')} ${refused}</span> ` : nothing}
                  ${m.error ? html`<span class="err-ink">${m.error}</span> ` : nothing}
                  ${say ? html`<span role="status">${say}</span>` : nothing}
                </td></tr>`
              : nothing}`;
        })}</tbody>
      </table></div>
      ${s.unserved?.length ? html`<p class="muted small">In the document but not served by this engine: ${s.unserved.join(', ')} (<code>QWEN38_DECIDER_MODELS</code> names what it serves).</p>` : nothing}
      <p class="muted small">"On start": loaded after the engine is healthy and its page cache is dropped, one at a time through the planner; "on demand": when a question first needs it; "off": only by Load here. Saved with the question mapping below.</p>`,
      { id: 'deciders-models', sub: 'the engine’s own decision models: own CUDA stream and memory pool, nothing shared with Qwen' },
    );
  }

  private renderMemory(s: DecidersStatus): TemplateResult {
    const p = s.planner;
    const ref = s.last_refusal;
    return panel(
      'Memory',
      html`<div class="stats residency-stats" data-decider-memory>
          <div class="stat"><div class="stat-label">MemAvailable</div><div class="stat-value num">${gibOf(p?.mem_available_gib)}</div></div>
          ${p?.engine_growth_gib != null ? html`<div class="stat"><div class="stat-label">engine growth to its peak</div><div class="stat-value num">${gibOf(p.engine_growth_gib)}</div></div>` : nothing}
          <div class="stat"><div class="stat-label">decider claim</div><div class="stat-value num">${gibOf(p?.claim_gib)}</div></div>
          <div class="stat"><div class="stat-label">floor + margin</div><div class="stat-value num">${gibOf(p?.floor_gib)} + ${gibOf(p?.margin_gib)}</div></div>
          <div class="stat ${p?.allowed ? '' : 'is-warn'}"><div class="stat-label">a load now</div><div class="stat-value num small">${plannerWords(p)}</div></div>
        </div>
        <p class="small" data-decider-last-refusal>${ref
          ? html`Last refused engine load: ${refusalWords(ref)} <span class="muted">(${ref.source ?? ''})</span>`
          : html`<span class="muted">No refused engine load in the planner's journal.</span>`}</p>
        <p class="muted small">A load is allowed when MemAvailable, less what the engine may still take up to its declared peak, minus the claim stays at or above floor + margin, never during a hold, never beside another load. <a href="#/conductor?screen=profiles">Edit the claim and the limits</a> (Profiles, Residency).</p>`,
      { id: 'deciders-memory' },
    );
  }

  private renderQuestions(s: DecidersStatus): TemplateResult {
    const ed = this.ed;
    const d = ed.draft;
    const served = s.models.map((m) => m.name);
    const rows = d ? Object.entries(d.questions) : [];
    return html`${panel(
        'Questions',
        d
          ? html`<div class="table-scroll"><table class="table" data-decider-questions>
              <thead><tr><th>question id</th><th>answered by</th><th>last calls</th></tr></thead>
              <tbody>${rows.map(([qid, model]) => {
                const live = s.questions.find((x) => x.id === qid);
                return html`<tr data-question=${qid}>
                  <td class="mono">${qid}</td>
                  <td><select aria-label=${`model for ${qid}`} data-question-model=${qid} @change=${(e: Event) => ed.set(setIn(d, ['questions', qid], (e.target as HTMLSelectElement).value) as DecidersDoc)}>
                    ${['auto', ...served].map((m) => html`<option value=${m} ?selected=${m === model}>${m === 'auto' ? 'auto (the role\u2019s model; the router picks)' : m}</option>`)}
                    ${model !== 'auto' && !served.includes(model) ? html`<option value=${model} selected>${model} (not served)</option>` : nothing}
                  </select></td>
                  <td class="small">${picksWords(live?.picks ?? {})}</td>
                </tr>`;
              })}</tbody></table></div>
              <p class="muted small">The conductor asks <code>classify</code> (the task class of a new message) and <code>packages</code> (which skill fits); the session monitor's Tier 1 questions join this list when they land.</p>`
          : skeleton(3),
        { id: 'deciders-questions', sub: 'which model answers each question the conductor asks' },
      )}
      ${errorsList(ed)}
      ${saveBar(ed, 'the deciders document')}`;
  }

  private renderTest(s: DecidersStatus): TemplateResult {
    const out = this.tOut;
    const a = out?.answers.q;
    return panel(
      'Test box',
      html`<div class="form-grid" data-decider-test>
          <label class="field field-block"><span>state (the text the model reads)</span><textarea rows="3" .value=${this.tState} @input=${(e: Event) => (this.tState = (e.target as HTMLTextAreaElement).value)} data-test-state></textarea></label>
          <label class="field"><span>question</span><input .value=${this.tQuestion} @input=${(e: Event) => (this.tQuestion = (e.target as HTMLInputElement).value)} data-test-question /></label>
          <label class="field"><span>options, comma separated</span><input .value=${this.tOptions} @input=${(e: Event) => (this.tOptions = (e.target as HTMLInputElement).value)} data-test-options /></label>
          <label class="field"><span>model</span><select @change=${(e: Event) => (this.tModel = (e.target as HTMLSelectElement).value)} data-test-model>
            ${['auto', ...s.models.map((m) => m.name)].map((m) => html`<option value=${m} ?selected=${m === this.tModel}>${m}</option>`)}
          </select></label>
        </div>
        <div class="savebar"><button class="btn btn-primary" ?disabled=${this.asking} @click=${() => this.ask()} data-test-ask>${icon('send')} ${this.asking ? 'Asking…' : 'Ask'}</button>
          ${this.tErr ? html`<span class="err-ink small" role="alert" data-test-error>${this.tErr}</span>` : nothing}</div>
        ${out && a
          ? html`<div class="decider-answer" data-test-out>
              <p><strong data-test-choice>${a.type === 'choice' ? a.choice : a.type === 'noul' ? (a.noul ?? 0).toFixed(4) : (a.score ?? 0).toFixed(4)}</strong>
                <span class="muted small num">confidence ${a.confidence.toFixed(2)} · ${out.model} · ${ms(out.latency_ms)} · ${out.path} · ${exact(out.usage.input_tokens)} tokens</span></p>
              <div class="prob-list">${probsOf(a).map(([k, p]) => html`<div class="prob-row" data-prob=${k}><span class="mono small">${k}</span><span class="conf-bar"><span style="width:${Math.round(p * 100)}%"></span></span><span class="num small">${p.toFixed(4)}</span></div>`)}</div>
              <p class="muted small" data-test-routing>routing: ${out.routing.reason}${out.routing.requested ? ` (wanted ${out.routing.requested})` : ''}</p>
            </div>`
          : nothing}`,
      { id: 'deciders-test', sub: 'one choice question, as /v1/systemone answers it (it runs in the idle window like any other)' },
    );
  }
}
