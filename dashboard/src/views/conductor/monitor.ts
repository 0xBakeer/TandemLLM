// Conductor · Monitor: the session monitor's journal and its rules (SRV-113, SRV-115).
//
// The journal: every decision the monitor made, newest first, every 3 s -- what it did or would
// have done in shadow, why, the hidden text the model saw, where it went and what came of it --
// with the drawer (and its right / wrong labels) and a filter by task and mode. It is the same
// journal the router writes; a session's timeline and export show both (Sessions).
//
// The rules: the `monitor` document, versioned like the others (version 0 is the server's
// ops/supervisor.yaml). Every task with its rule and the rule's parameters -- the tool checks'
// risk floors, project rules, reminders and limits; the reasoning arms; the empty-turn nudges --
// rendered from the policy's own JSON Schema, so a rule registered on the server shows up here
// without a dashboard change. Caps, texts, the journal setting and the chat template's shapes
// below. A profile sets each task's mode, the tool-gate packs and its thresholds on top (Profiles).

import { html, nothing, type TemplateResult } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import { describeError } from '../../api/client';
import { conductor, type ConfigInfo, type JsonSchema, type Meta, type MonitorDoc, type MonitorEvent, type MonitorTask, type TaskMode } from '../../api/conductor';
import { linesToList, monitorTaskOrder, ruleSchema, setIn } from '../../lib/conductor';
import { poll, type Poller } from '../../lib/poll';
import { LightElement } from '../../ui/base';
import { icon, panel, skeleton } from '../../ui/bits';
import { DocEditor, errorsList, eventRow, modeSeg, saveBar } from './shared';

const TASK_MODES: TaskMode[] = ['off', 'shadow', 'act'];
const LONG = /text|texts|close|nudge|nudges|turn|note|pattern|root|home|requires|suffix|give_up/;

@customElement('qse-cond-monitor')
export class QseCondMonitor extends LightElement {
  @property({ attribute: false }) meta: Meta | null = null;
  @property({ attribute: false }) onSaved: () => void = () => undefined;
  @state() private config: ConfigInfo | null = null;
  private ed = new DocEditor<MonitorDoc>(this, 'monitor');

  connectedCallback(): void {
    super.connectedCallback();
    this.ed.onSaved = () => this.onSaved();
    void this.ed.load();
    void conductor.config().then((c) => (this.config = c)).catch(() => undefined);
  }

  private upd(path: (string | number)[], v: unknown) {
    if (this.ed.draft) this.ed.set(setIn(this.ed.draft, path, v));
  }

  render() {
    const ed = this.ed;
    const mode = this.config?.server.global_mode;
    return html`
      <qse-cond-journal></qse-cond-journal>
      ${ed.loading && !ed.draft
        ? skeleton(8)
        : ed.loadError
          ? html`<div class="state state-error" role="alert"><div class="state-title">${icon('warn')} ${ed.loadError}</div></div>`
          : html`
            ${mode === 'raw' || mode == null
              ? html`<div class="guard-line" data-monitor-off>${icon('warn')} The session monitor does not run on this server (<code>--conductor ${mode ?? 'raw'}</code>): the rules below are kept and take effect once it runs.</div>`
              : nothing}
            <p class="muted small" style="margin:12px 0">The rules of every task. Each profile sets a task's mode (off, shadow, act), the tool-gate packs and the thresholds on top, on the Profiles screen; the mode here is what a server without a profile uses.</p>
            ${this.renderTasks(ed.draft!)}
            ${this.renderRest(ed.draft!)}
            ${errorsList(ed)}
            ${saveBar(ed, 'the monitor')}`}`;
  }

  private renderTasks(d: MonitorDoc): TemplateResult {
    const order = monitorTaskOrder(d);
    const tool = order.filter((t) => d.tasks[t] && /^C\d/.test(t));
    const rest = order.filter((t) => !tool.includes(t));
    return html`
      <details class="sec" open data-sec="toolchecks">
        <summary>Tool checks <span class="sec-sub">${tool.join(', ') || 'none'}: held calls are checked at the end of the answer, before any goes out</span></summary>
        <div class="sec-body">${tool.map((t) => this.renderTask(t, d.tasks[t]))}</div>
      </details>
      <details class="sec" data-sec="othertasks">
        <summary>Thinking and end of turn <span class="sec-sub">${rest.join(', ')}</span></summary>
        <div class="sec-body">${rest.map((t) => this.renderTask(t, d.tasks[t]))}</div>
      </details>`;
  }

  private renderTask(tid: string, task: MonitorTask): TemplateResult {
    const sch = ruleSchema(this.meta?.schemas?.monitor, task.rule);
    const tmeta = this.meta?.tasks?.[tid];
    return html`<div class="row-card" data-monitor-task=${tid}>
      <div class="row-card-head">
        <strong>${tid}${tmeta ? html` · ${tmeta.label}` : nothing} <span class="muted small mono">${task.rule}</span></strong>
        ${modeSeg(TASK_MODES, task.mode, (m) => this.upd(['tasks', tid, 'mode'], m), { small: true, label: `${tid} mode without a profile` })}
      </div>
      ${tmeta?.why ? html`<p class="muted small">${tmeta.why}</p>` : nothing}
      ${sch?.properties
        ? html`<div class="form-stack">${Object.entries(sch.properties).map(([k, s]) => this.field(k, s, task[k], ['tasks', tid, k]))}</div>`
        : html`<p class="muted small">This rule's parameters have no schema here; edit them as JSON.</p>${this.jsonField('parameters', task, ['tasks', tid])}`}
    </div>`;
  }

  /** One parameter from its JSON Schema: a number, a switch, a choice, a text, a list of texts,
   * a list of objects (the risk floors), or a group (the reasoning arms). */
  private field(key: string, s: JsonSchema, value: unknown, path: (string | number)[]): TemplateResult {
    const t = Array.isArray(s.type) ? s.type[0] : s.type;
    const label = key.replace(/_/g, ' ');
    const hint = s.description ? html`<span class="hint">${s.description}</span>` : nothing;
    if (s.enum) {
      return html`<label class="field"><span>${label}</span><select data-param=${key} @change=${(e: Event) => this.upd(path, (e.target as HTMLSelectElement).value)}>${s.enum.map((v) => html`<option value=${String(v)} ?selected=${v === value}>${String(v)}</option>`)}</select>${hint}</label>`;
    }
    if (t === 'boolean') {
      return html`<label class="switch" data-param=${key}><input type="checkbox" .checked=${!!value} @change=${(e: Event) => this.upd(path, (e.target as HTMLInputElement).checked)} /> ${label}</label>${s.description ? html`<p class="hint">${s.description}</p>` : nothing}`;
    }
    if (t === 'integer' || t === 'number') {
      return html`<label class="field"><span>${label}</span><input type="number" data-param=${key} step=${t === 'integer' ? '1' : '0.01'} min=${s.minimum ?? nothing} max=${s.maximum ?? nothing} .value=${value == null ? '' : String(value)} @input=${(e: Event) => {
        const raw = (e.target as HTMLInputElement).value;
        this.upd(path, raw === '' ? null : t === 'integer' ? Math.round(Number(raw)) : Number(raw));
      }} />${hint}</label>`;
    }
    if (t === 'string') {
      const v = String(value ?? '');
      return LONG.test(key) || v.length > 60
        ? html`<label class="field field-block"><span>${label}</span><textarea rows=${Math.min(6, Math.max(2, Math.ceil(v.length / 90)))} class=${/pattern|root|home|requires|suffix/.test(key) ? 'mono' : ''} data-param=${key} .value=${v} @input=${(e: Event) => this.upd(path, (e.target as HTMLTextAreaElement).value)}></textarea>${hint}</label>`
        : html`<label class="field"><span>${label}</span><input data-param=${key} .value=${v} @input=${(e: Event) => this.upd(path, (e.target as HTMLInputElement).value)} />${hint}</label>`;
    }
    if (t === 'array' && s.items && (Array.isArray(s.items.type) ? s.items.type[0] : s.items.type) === 'string') {
      const list = Array.isArray(value) ? (value as string[]) : [];
      return html`<label class="field field-block"><span>${label} <span class="muted small">(one a line)</span></span><textarea rows=${Math.min(8, Math.max(2, list.length + 1))} data-param=${key} .value=${list.join('\n')} @input=${(e: Event) => this.upd(path, linesToList((e.target as HTMLTextAreaElement).value))}></textarea>${hint}</label>`;
    }
    if (t === 'array' && s.items?.properties) {
      const list = Array.isArray(value) ? (value as Record<string, unknown>[]) : [];
      const item = s.items;
      return html`<fieldset class="fieldset" data-param=${key}><legend>${label} <span class="muted small">${list.length}</span></legend>${hint}
        ${list.map((it, i) => html`<div class="row-card row-card-sub" data-item=${String(it.id ?? i)}>
          <div class="row-card-head"><strong class="mono small">${String(it.id ?? `#${i + 1}`)}</strong>
            <button class="btn btn-ghost btn-sm" @click=${() => this.upd(path, list.filter((_, j) => j !== i))} aria-label=${`remove ${String(it.id ?? i + 1)}`}>${icon('trash')}</button></div>
          <div class="form-stack">${Object.entries(item.properties!).map(([k, ss]) => this.field(k, ss, it[k], [...path, i, k]))}</div></div>`)}
        <button class="btn btn-sm" data-add-item=${key} @click=${() => this.upd(path, [...list, blank(item)])}>${icon('plus')} Add</button>
      </fieldset>`;
    }
    if (t === 'object' && s.properties) {
      const obj = (value ?? {}) as Record<string, unknown>;
      return html`<fieldset class="fieldset" data-param=${key}><legend>${label}</legend>${hint}
        <div class="form-stack">${Object.entries(s.properties).map(([k, ss]) => obj[k] === undefined && !(s.required ?? []).includes(k)
          ? html`<button class="btn btn-ghost btn-sm" @click=${() => this.upd([...path, k], blank(ss))}>${icon('plus')} ${k.replace(/_/g, ' ')}</button>`
          : this.field(k, ss, obj[k], [...path, k]))}</div></fieldset>`;
    }
    return this.jsonField(label, value, path);
  }

  private jsonField(label: string, value: unknown, path: (string | number)[]): TemplateResult {
    return html`<label class="field field-block"><span>${label} (JSON)</span><textarea class="mono" rows="6" .value=${JSON.stringify(value, null, 2)} @change=${(e: Event) => {
      try {
        const v = JSON.parse((e.target as HTMLTextAreaElement).value);
        if (path.length === 2 && path[0] === 'tasks') {
          const cur = this.ed.draft!.tasks[path[1] as string];
          this.upd(path, { ...v, rule: cur.rule, mode: cur.mode });
        } else this.upd(path, v);
      } catch {
        this.ed.errors = [`${label}: not JSON`];
        this.requestUpdate();
      }
    }}></textarea></label>`;
  }

  private renderRest(d: MonitorDoc): TemplateResult {
    return html`
      <details class="sec" data-sec="caps">
        <summary>Caps, texts and the journal <span class="sec-sub">beyond a cap an act becomes GIVE_UP; a profile sets its own caps</span></summary>
        <div class="sec-body">
          <div class="form-grid">
            ${this.field('acts_per_request', { type: 'integer', minimum: 0 }, d.caps.acts_per_request, ['caps', 'acts_per_request'])}
            ${this.field('acts_per_session_hour', { type: 'integer', minimum: 0 }, d.caps.acts_per_session_hour, ['caps', 'acts_per_session_hour'])}
            ${this.field('same_action_in_a_row', { type: 'boolean', description: 'off: the same hidden action twice in a row on one turn ends in GIVE_UP' }, d.caps.same_action_in_a_row, ['caps', 'same_action_in_a_row'])}
            ${this.field('tool_passes', { type: 'boolean', description: 'one journal row for every answer whose calls passed every check (findings are always journaled, passes always counted)' }, d.journal.tool_passes, ['journal', 'tool_passes'])}
          </div>
          ${Object.entries(d.texts).map(([k, v]) => this.field(k, { type: 'string', description: k === 'give_up' ? 'the one visible sentence of a GIVE_UP' : undefined }, v, ['texts', k]))}
          <div class="form-grid">
            ${this.field('lineage_tokens', { type: 'integer', minimum: 1, description: 'prompt tokens hashed into the session key when the client sends no id' }, d.session.lineage_tokens, ['session', 'lineage_tokens'])}
            ${this.field('join_window_s', { type: 'number', minimum: 0 }, d.session.join_window_s, ['session', 'join_window_s'])}
            ${this.field('max_sessions', { type: 'integer', minimum: 1 }, d.session.max_sessions, ['session', 'max_sessions'])}
          </div>
        </div>
      </details>
      <details class="sec" data-sec="templates">
        <summary>Chat template shapes <span class="sec-sub">how a hidden turn, a reminder and a note are written in the model's own format</span></summary>
        <div class="sec-body">${Object.entries(d.templates).map(([k, v]) => this.field(k, { type: 'string' }, v, ['templates', k]))}</div>
      </details>`;
  }
}

/** An empty value of a schema, for "Add". */
function blank(s: JsonSchema): unknown {
  const t = Array.isArray(s.type) ? s.type[0] : s.type;
  if (s.enum) return s.enum[0];
  if (t === 'boolean') return false;
  if (t === 'integer' || t === 'number') return s.minimum ?? 0;
  if (t === 'array') return [];
  if (t === 'object') return Object.fromEntries(Object.entries(s.properties ?? {}).filter(([k]) => (s.required ?? []).includes(k)).map(([k, ss]) => [k, blank(ss)]));
  return '';
}

// ---------------------------------------------------------------- the journal
@customElement('qse-cond-journal')
export class QseCondJournal extends LightElement {
  @property({ attribute: false }) session: string | null = null;
  @state() private events: MonitorEvent[] | null = null;
  @state() private monitor = true;
  @state() private error: string | null = null;
  @state() private task = '';
  @state() private kind: '' | 'acted' | 'shadow' = '';
  @state() private paused = false;
  @state() private open: MonitorEvent | null = null;
  private poller: Poller | null = null;

  connectedCallback(): void {
    super.connectedCallback();
    this.poller = poll(() => this.load(), 3_000);
    this.poller.start();
  }
  disconnectedCallback(): void {
    this.poller?.stop();
    super.disconnectedCallback();
  }

  private async load() {
    if (this.paused) return;
    try {
      const r = await conductor.events({ limit: 100, session: this.session, task: this.task || null, mode: this.kind === 'shadow' ? 'shadow' : null, acted: this.kind === 'acted' });
      this.events = r.events;
      this.monitor = r.monitor;
      this.error = null;
    } catch (e) {
      const d = describeError(e);
      this.error = `${d.endpoint}: ${d.message}`;
    }
  }

  render() {
    const evs = this.events;
    const tasks = [...new Set((evs ?? []).map((e) => e.task ?? '').filter(Boolean))].sort();
    if (this.task && !tasks.includes(this.task)) tasks.push(this.task);
    return html`${panel(
      html`Monitor journal ${this.paused ? html`<span class="badge">paused</span>` : html`<span class="badge badge-on">every 3 s</span>`}`,
      this.error && !evs
        ? html`<p class="err-ink small" role="alert">${this.error}</p>`
        : evs == null
          ? skeleton(4)
          : evs.length
            ? html`<div class="feed" data-journal>${evs.map((e) => eventRow(e, (x) => (this.open = x)))}</div>`
            : html`<div class="feed-empty">${this.monitor ? 'No decision of the session monitor yet. Every check that finds something, every act and every shadow decision shows up here with its exact hidden text.' : 'The session monitor does not run on this server (--conductor raw).'}</div>`,
      {
        id: 'cond-journal',
        sub: 'the session monitor\'s decisions, newest first; the router\'s are on the Overview, both in a session\'s timeline and export',
        tools: html`<label class="field"><span>task</span><select aria-label="task" @change=${(e: Event) => {
            this.task = (e.target as HTMLSelectElement).value;
            void this.load();
          }}><option value="">all</option>${tasks.map((t) => html`<option value=${t} ?selected=${t === this.task}>${t}</option>`)}</select></label>
          <label class="field"><span>show</span><select aria-label="kind" @change=${(e: Event) => {
            this.kind = (e.target as HTMLSelectElement).value as '' | 'acted' | 'shadow';
            void this.load();
          }}><option value="">every decision</option><option value="acted" ?selected=${this.kind === 'acted'}>acted</option><option value="shadow" ?selected=${this.kind === 'shadow'}>shadow</option></select></label>
          <button class="btn btn-ghost btn-sm" @click=${() => (this.paused = !this.paused)}>${icon(this.paused ? 'play' : 'pause')} ${this.paused ? 'resume' : 'pause'}</button>`,
      },
    )}
    ${this.open ? html`<qse-cond-event .ev=${this.open} .close=${() => (this.open = null)}></qse-cond-event>` : nothing}`;
  }
}
