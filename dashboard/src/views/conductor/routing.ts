// Conductor · Routing: the task router's whole document, editable. The classes (id, label, the
// description a classifier reads, and the policy: effort, candidate packages, rules, a note), the
// classifier stages in order with their thresholds and timeouts, stickiness and the structural
// rules, the effort levels (thinking, budget, the sentence it adds), and the note's wrapper and
// token caps. A test box runs the whole cascade on a pasted prompt and shows the exact note.
// Every save is a new router version; the server loads it for the next request.

import { html, nothing, type TemplateResult } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import { describeError } from '../../api/client';
import { conductor, type Effort, type Explain, type Meta, type PackagesDoc, type RouterDoc, type TaskClass } from '../../api/conductor';
import { setIn } from '../../lib/conductor';
import { ms } from '../../lib/format';
import { LightElement } from '../../ui/base';
import { icon, skeleton } from '../../ui/bits';
import { clsChip, confView, DocEditor, errorsList, modeSeg, saveBar, sourceWords } from './shared';

const EFFORTS: Effort[] = ['off', 'low', 'medium', 'high', 'xhigh'];

@customElement('qse-cond-routing')
export class QseCondRouting extends LightElement {
  @property({ attribute: false }) ids: string[] = [];
  @property({ attribute: false }) meta: Meta | null = null;
  @property({ attribute: false }) active = 'coding';
  @property({ attribute: false }) onSaved: () => void = () => undefined;
  @state() private packages: string[] = [];
  @state() private testText = '';
  @state() private testProfile: string | null = null;
  @state() private testTools = true;
  @state() private testing = false;
  @state() private result: Explain | null = null;
  @state() private testError: string | null = null;
  private ed = new DocEditor<RouterDoc>(this, 'router');

  connectedCallback(): void {
    super.connectedCallback();
    this.ed.onSaved = () => this.onSaved();
    void this.ed.load();
    void conductor.doc<PackagesDoc>('packages').then((p) => (this.packages = Object.keys(p.doc.packages))).catch(() => undefined);
  }

  private upd(path: (string | number)[], v: unknown) {
    if (!this.ed.draft) return;
    this.ed.set(setIn(this.ed.draft, path, v));
  }

  private async test() {
    if (!this.testText.trim()) return;
    this.testing = true;
    this.testError = null;
    try {
      this.result = await conductor.classify(this.testText, this.testProfile ?? this.active, this.testTools);
    } catch (e) {
      this.testError = describeError(e).message;
    } finally {
      this.testing = false;
    }
  }

  render() {
    const ed = this.ed;
    if (ed.loading && !ed.draft) return skeleton(8);
    if (ed.loadError) return html`<div class="state state-error" role="alert"><div class="state-title">${icon('warn')} ${ed.loadError}</div></div>`;
    const d = ed.draft!;
    const ids = d.classes.map((c) => c.id);
    return html`
      ${this.renderTest(ids)}
      <details class="sec" open data-sec="classes">
        <summary>Task classes <span class="sec-sub">${d.classes.length} classes; a classifier sees only the descriptions</span></summary>
        <div class="sec-body">
          ${d.classes.map((c, i) => this.renderClass(c, i, d.classes.length, ids))}
          <button class="btn" data-add-class @click=${() => this.addClass()}>${icon('plus')} Add a class</button>
        </div>
      </details>
      <details class="sec" data-sec="classifier">
        <summary>Classifier stages <span class="sec-sub">asked in order until one answers at its threshold</span></summary>
        <div class="sec-body">
          ${d.classifier.stages.map(
            (s, i) => html`<div class="row-card" data-stage=${s.kind}>
              <div class="row-card-head">
                <strong>${i + 1}. ${s.kind === 'decider' ? 'decider (role decider: any model behind the Jev question shape)' : 'Qwen, thinking off (role classifier)'}</strong>
                <span class="toolbar" style="margin:0">
                  <label class="switch"><input type="checkbox" .checked=${s.enabled} @change=${(e: Event) => this.upd(['classifier', 'stages', i, 'enabled'], (e.target as HTMLInputElement).checked)} /> enabled</label>
                  <button class="btn btn-ghost btn-sm" ?disabled=${i === 0} @click=${() => this.moveStage(i, -1)} aria-label="earlier">↑</button>
                  <button class="btn btn-ghost btn-sm" ?disabled=${i === d.classifier.stages.length - 1} @click=${() => this.moveStage(i, 1)} aria-label="later">↓</button>
                </span>
              </div>
              <div class="form-grid">
                ${this.num('accept at (probability)', s.accept, (v) => this.upd(['classifier', 'stages', i, 'accept'], v), { step: 0.01, min: 0, max: 1, hint: 'below it, the next stage is asked' })}
                ${this.num('timeout (ms)', s.timeout_ms, (v) => this.upd(['classifier', 'stages', i, 'timeout_ms'], Math.round(v)), { step: 50, min: 1, hint: 'a slower answer counts as no answer' })}
              </div>
            </div>`,
          )}
          <div class="form-grid">
            ${this.text('question to the decider', d.classifier.question, (v) => this.upd(['classifier', 'question'], v))}
            ${this.num('digest (characters)', d.classifier.digest_chars, (v) => this.upd(['classifier', 'digest_chars'], Math.round(v)), { step: 100, min: 200, hint: 'the user message is cut to this, head and tail; the decider cuts again to its own limits' })}
            ${this.text('package question', d.classifier.package_question, (v) => this.upd(['classifier', 'package_question'], v))}
            ${this.num('package accept at', d.classifier.package_accept, (v) => this.upd(['classifier', 'package_accept'], v), { step: 0.01, min: 0, max: 1 })}
          </div>
          <label class="field field-block"><span>Qwen's instruction (<code>{classes}</code> becomes the id and description list)</span>
            <textarea rows="5" .value=${d.classifier.qwen_prompt} @input=${(e: Event) => this.upd(['classifier', 'qwen_prompt'], (e.target as HTMLTextAreaElement).value)}></textarea></label>
        </div>
      </details>
      <details class="sec" data-sec="stickiness">
        <summary>Stickiness and structure <span class="sec-sub">when a session changes class, and what the request shape decides</span></summary>
        <div class="sec-body">
          <div class="form-grid">
            ${this.num('switch class at (probability)', d.stickiness.switch, (v) => this.upd(['stickiness', 'switch'], v), { step: 0.01, min: 0, max: 1, hint: 'a session with a class moves to another only at this confidence' })}
            <label class="field"><span>opencode's compaction request</span>
              <select @change=${(e: Event) => this.upd(['structural', 'compaction'], (e.target as HTMLSelectElement).value)}>
                ${['keep', 'classify', ...ids].map((v) => html`<option value=${v} ?selected=${v === d.structural.compaction}>${v === 'keep' ? 'keep the session class' : v === 'classify' ? 'ask a classifier' : `class ${v}`}</option>`)}
              </select><span class="hint">the turn itself; the session keeps its class after it</span></label>
            <label class="field"><span>a turn that only carries tool results</span>
              <select @change=${(e: Event) => this.upd(['structural', 'tool_turn'], (e.target as HTMLSelectElement).value)}>
                <option value="keep" ?selected=${d.structural.tool_turn === 'keep'}>keep the session class (no model call)</option>
                <option value="classify" ?selected=${d.structural.tool_turn === 'classify'}>ask a classifier every turn</option>
              </select></label>
          </div>
        </div>
      </details>
      <details class="sec" data-sec="effort">
        <summary>Effort levels <span class="sec-sub">a budget and a sentence at the prompt's end; never the template's reasoning_effort</span></summary>
        <div class="sec-body">
          <p class="muted small">The template renders <code>reasoning_effort</code> as the first system tokens, so changing it per turn would cost a cold prefill. A level here sets the reasoning budget (0 = no cap below the backstop), thinking on or off (off uses <code>enable_thinking: false</code> only when the server's effort is medium, else a budget of 1), and a sentence in the tail note.</p>
          ${EFFORTS.map(
            (lv) => html`<div class="row-card" data-effort=${lv}>
              <div class="row-card-head"><strong class="mono">${lv}</strong>
                <label class="switch"><input type="checkbox" .checked=${d.effort.levels[lv].thinking} @change=${(e: Event) => this.upd(['effort', 'levels', lv, 'thinking'], (e.target as HTMLInputElement).checked)} /> thinking</label></div>
              <div class="form-grid">${this.num('budget (tokens, 0 = no cap)', d.effort.levels[lv].budget, (v) => this.upd(['effort', 'levels', lv, 'budget'], Math.round(v)), { step: 256, min: 0 })}</div>
              <label class="field field-block"><span>sentence in the note</span><textarea rows="2" .value=${d.effort.levels[lv].note} @input=${(e: Event) => this.upd(['effort', 'levels', lv, 'note'], (e.target as HTMLTextAreaElement).value)}></textarea></label>
            </div>`,
          )}
          <label class="switch"><input type="checkbox" .checked=${d.effort.raise_on_fail} @change=${(e: Event) => this.upd(['effort', 'raise_on_fail'], (e.target as HTMLInputElement).checked)} /> a failing checker (RESULT: FAIL) raises the effort one level for that turn</label>
        </div>
      </details>
      <details class="sec" data-sec="injection">
        <summary>The note <span class="sec-sub">where hidden text goes and how much</span></summary>
        <div class="sec-body">
          <p class="muted small">The note goes into the last turn, right before its <code>&lt;|im_end|&gt;</code>. Later turns replay it at the same token position, so the resident prefix keeps matching. <code>{text}</code> is replaced by the effort sentence, the package and the rules.</p>
          <label class="field field-block"><span>wrapper</span><textarea rows="6" data-wrapper .value=${d.injection.wrapper} @input=${(e: Event) => this.upd(['injection', 'wrapper'], (e.target as HTMLTextAreaElement).value)}></textarea></label>
          <div class="form-grid">
            ${this.num('max tokens per note', d.injection.max_note_tokens, (v) => this.upd(['injection', 'max_note_tokens'], Math.round(v)), { step: 500, min: 16 })}
            ${this.num('max tokens replayed per session', d.injection.max_replay_tokens, (v) => this.upd(['injection', 'max_replay_tokens'], Math.round(v)), { step: 1000, min: 0 })}
            <label class="switch"><input type="checkbox" .checked=${d.injection.replay} @change=${(e: Event) => this.upd(['injection', 'replay'], (e.target as HTMLInputElement).checked)} /> replay notes on later turns</label>
          </div>
        </div>
      </details>
      ${errorsList(ed)}
      ${saveBar(ed, 'the router')}
    `;
  }

  private renderTest(ids: string[]): TemplateResult {
    const r = this.result;
    return html`<section class="panel" data-testbox>
      <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">Test a prompt</h2><p class="panel-sub">Runs the live cascade and policy on the text, as a new session, and shows the exact note. Nothing is journaled; the Qwen stage uses the engine for about a second.</p></div></header>
      <label class="field field-block"><span>prompt</span><textarea rows="3" data-test-text .value=${this.testText} @input=${(e: Event) => (this.testText = (e.target as HTMLTextAreaElement).value)} placeholder="Refactor the parser so every token carries its line number."></textarea></label>
      <div class="toolbar" style="margin:10px 0 0">
        <label class="field"><span>profile</span><input style="width:120px" .value=${this.testProfile ?? this.active} @input=${(e: Event) => (this.testProfile = (e.target as HTMLInputElement).value || null)} /></label>
        <label class="switch"><input type="checkbox" .checked=${this.testTools} @change=${(e: Event) => (this.testTools = (e.target as HTMLInputElement).checked)} /> the request offers tools</label>
        <button class="btn btn-primary" ?disabled=${this.testing || !this.testText.trim()} @click=${() => this.test()} data-run-test>${this.testing ? 'Running…' : 'Run'}</button>
        ${this.testError ? html`<span class="err-ink small" role="alert">${this.testError}</span>` : nothing}
      </div>
      ${r
        ? r.active
          ? html`<div class="grid-2 grid-2-tight" style="margin-top:14px" data-test-result>
              <div>
                <dl class="kv">
                  <div><dt>class</dt><dd>${clsChip(r.class_id ?? null, ids)} ${confView(r.confidence)}</dd></div>
                  <div><dt>decided by</dt><dd>${sourceWords(r.source ?? null)}${r.kept ? html` <span class="badge badge-warn">${r.kept}</span>` : nothing} <span class="muted num">${ms(r.classifier_ms)}</span></dd></div>
                  <div><dt>effort</dt><dd>${r.effort ?? 'unchanged'} · thinking ${r.thinking ?? 'on'}${r.budget != null ? ` · budget ${r.budget}` : ''}</dd></div>
                  <div><dt>profile</dt><dd>${r.profile} (${r.mode})</dd></div>
                </dl>
                <div class="stage-list" style="margin-top:10px">${(r.stages ?? []).map(
                  (s) => html`<div class="stage ${s.error ? 'is-error' : !s.below ? 'is-accepted' : ''}"><strong>${s.stage === 'decider' ? 'decider' : 'Qwen'}</strong><span>${s.error ?? html`${s.label} ${confView(s.confidence, s.accept ?? 0.6)}`}</span><span class="num muted">${ms(s.ms)}</span></div>`,
                )}</div>
              </div>
              <div>${r.note ? html`<p class="muted small">The model would see this at the end of the last turn (${(r.note_parts ?? []).map((p) => `${p.kind} ${p.name}`).join(', ')}):</p><pre class="note-text" data-test-note>${r.note}</pre>` : html`<p class="muted small">No note: ${r.shadow ? 'the decisions are in shadow' : 'this class adds nothing'}.</p>`}</div>
            </div>`
          : html`<p class="muted small" style="margin-top:12px">Profile ${r.profile} is raw (${r.mode}): the conductor does nothing for it.</p>`
        : nothing}
    </section>`;
  }

  private renderClass(c: TaskClass, i: number, n: number, ids: string[]): TemplateResult {
    const p = c.policy;
    const isNew = !this.ed.base?.doc.classes.some((x) => x.id === c.id);
    return html`<details class="row-card" data-class-row=${c.id} ?open=${isNew}>
      <summary>
        <span>${clsChip(c.id, ids, c.label)} <span class="muted small mono">${p.effort ?? 'medium'}${(p.packages ?? []).length ? ` · ${(p.packages ?? []).join(', ')}` : ''}</span></span>
        <span class="toolbar" style="margin:0" @click=${(e: Event) => e.preventDefault()}>
          <button class="btn btn-ghost btn-sm" ?disabled=${i === 0} @click=${() => this.moveClass(i, -1)} aria-label="up">↑</button>
          <button class="btn btn-ghost btn-sm" ?disabled=${i === n - 1} @click=${() => this.moveClass(i, 1)} aria-label="down">↓</button>
          <button class="btn btn-ghost btn-sm" ?disabled=${n <= 1} @click=${() => this.removeClass(i)} aria-label=${`remove ${c.id}`}>${icon('trash')}</button>
        </span>
      </summary>
      <div class="form-grid">
        <label class="field"><span>id</span><input .value=${c.id} ?readonly=${!isNew} @input=${(e: Event) => this.upd(['classes', i, 'id'], (e.target as HTMLInputElement).value)} data-class-id /><span class="hint">${isNew ? 'lowercase, digits, _ and -' : 'fixed once saved; the journal keys on it'}</span></label>
        <label class="field"><span>label</span><input .value=${c.label} @input=${(e: Event) => this.upd(['classes', i, 'label'], (e.target as HTMLInputElement).value)} /></label>
      </div>
      <label class="field field-block"><span>description (what a classifier reads)</span><textarea rows="2" .value=${c.description} @input=${(e: Event) => this.upd(['classes', i, 'description'], (e.target as HTMLTextAreaElement).value)} data-class-desc></textarea></label>
      <div class="toolbar" style="margin:0"><span class="muted small">effort</span>${modeSeg(EFFORTS, (p.effort ?? 'medium') as Effort, (v) => this.upd(['classes', i, 'policy', 'effort'], v), { small: true, label: `effort of ${c.id}` })}</div>
      <div><span class="muted small">packages (a class with several asks a second question and injects the one that fits)</span>
        <div class="pkg-picks">${this.packages.map((name) => {
          const on = (p.packages ?? []).includes(name);
          return html`<button class="pick ${on ? 'is-on' : ''}" aria-pressed=${on ? 'true' : 'false'} @click=${() => this.upd(['classes', i, 'policy', 'packages'], on ? (p.packages ?? []).filter((x) => x !== name) : [...(p.packages ?? []), name])}>${name}</button>`;
        })}</div></div>
      <label class="field field-block"><span>rules (one per line; they go into the note)</span><textarea rows="2" .value=${(p.rules ?? []).join('\n')} @input=${(e: Event) => this.upd(['classes', i, 'policy', 'rules'], (e.target as HTMLTextAreaElement).value.split('\n').map((x) => x.trim()).filter(Boolean))}></textarea></label>
    </details>`;
  }

  private num(label: string, value: number, set: (v: number) => void, o: { step?: number; min?: number; max?: number; hint?: string } = {}): TemplateResult {
    return html`<label class="field"><span>${label}</span><input type="number" .value=${String(value)} step=${o.step ?? 1} min=${o.min ?? ''} max=${o.max ?? ''} @input=${(e: Event) => {
      const v = Number((e.target as HTMLInputElement).value);
      if (Number.isFinite(v)) set(v);
    }} />${o.hint ? html`<span class="hint">${o.hint}</span>` : nothing}</label>`;
  }
  private text(label: string, value: string, set: (v: string) => void): TemplateResult {
    return html`<label class="field"><span>${label}</span><input .value=${value} @input=${(e: Event) => set((e.target as HTMLInputElement).value)} /></label>`;
  }

  private addClass() {
    const d = this.ed.draft!;
    let n = 1;
    while (d.classes.some((c) => c.id === `class_${n}`)) n++;
    const c: TaskClass = { id: `class_${n}`, label: `Class ${n}`, description: 'Describe the kind of request, in the words a user would use.', policy: { effort: 'medium', packages: [], rules: [] } };
    this.ed.set({ ...d, classes: [...d.classes, c] });
  }
  private removeClass(i: number) {
    const d = this.ed.draft!;
    this.ed.set({ ...d, classes: d.classes.filter((_, j) => j !== i) });
  }
  private moveClass(i: number, by: number) {
    const d = this.ed.draft!;
    const cs = [...d.classes];
    const [c] = cs.splice(i, 1);
    cs.splice(i + by, 0, c);
    this.ed.set({ ...d, classes: cs });
  }
  private moveStage(i: number, by: number) {
    const d = this.ed.draft!;
    const st = [...d.classifier.stages];
    const [s] = st.splice(i, 1);
    st.splice(i + by, 0, s);
    this.ed.set(setIn(d, ['classifier', 'stages'], st));
  }
}
