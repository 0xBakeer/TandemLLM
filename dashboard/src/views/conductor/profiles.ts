// Conductor · Profiles: the tiles (name, mode, clients, counts, version, last change; raw and
// bench with a lock) and the editor of one profile: mapping and default class, the four routing
// decision types (off, shadow, on), per-class overrides of the router's policy, the session
// monitor's tasks (mode, thresholds with units and their defaults, cap, the evidence line),
// thinking, the tool gate, context and speed hints; then the roles every profile shares, each
// with a test button and its contract (SRV-133: kind, window, preemptible, latency budget,
// fallback, writes, the tokenizer hash and the forms of sharing it allows); then the residency
// planner's limits (OPS-41): the MemAvailable floor and the
// margin a role load must leave, the settle time after a stop, the load order and each role's
// memory claim with its basis. Saving writes a new version; "Try in Playground" opens it there.

import { html, nothing, type TemplateResult } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import { describeError } from '../../api/client';
import type { ResidencyBasis, ResidencyDoc } from '../../api/types';
import { conductor, type Decision, type Effort, type Meta, type Mode3, type PackagesDoc, type ProfileDoc, type ProfileTile, type RoleDoc, type RoleFallback, type RoleHealth, type RoleKind, type RolesContract, type RoleWindow, type RoleWrites, type RouterDoc, type TaskMode, type ToolTurnEffort } from '../../api/conductor';
import { BUDGET_LEVELS, BUDGET_MAX, budgetText, BY_SIZE_DEFAULT, classPackages, defaultsByRole, levelBudget, progressOf, ROLE_KINDS, ROLE_WINDOWS, ROLE_WRITES, setIn, sharingOptions, STATE_KINDS, statesChannel, TOOL_TURN_EFFORTS, toolTurnEffort, toolTurnText, WINDOW_WORDS, windowRefusal, withBudget, writesRefusal } from '../../lib/conductor';
import { dateTimeShort, ms } from '../../lib/format';
import { setParams } from '../../lib/router';
import { LightElement } from '../../ui/base';
import { icon, skeleton } from '../../ui/bits';
import { clsChip, DocEditor, errorsList, modeSeg, saveBar } from './shared';

const EFFORTS: Effort[] = ['off', 'low', 'medium', 'high', 'xhigh'];
const DECISION_WORDS: Record<Decision, string> = {
  classify: 'classify: ask a model which task class the turn is',
  effort: 'effort: the thinking budget and the effort sentence',
  packages: 'packages: inject the class\'s skill or package, hidden',
  rules: 'rules: the class\'s working rules in the note',
};

// the session tasks whose numbers the Context section sets (hook.Conductor.CONTEXT on the server)
const CONTEXT_TASKS = new Set(['S1', 'S4']);

// tool-gate pack -> the task it switches and what it checks
const PACKS = [
  ['risk', 'C1', 'risk floors'],
  ['project', 'C2', 'project rules'],
  ['same_call', 'C3', 'same call thrice'],
  ['malformed', 'C4', 'malformed calls'],
] as const;

@customElement('qse-cond-profiles')
export class QseCondProfiles extends LightElement {
  @property({ attribute: false }) ids: string[] = [];
  @property({ attribute: false }) meta: Meta | null = null;
  @property({ attribute: false }) tiles: ProfileTile[] = [];
  @property({ attribute: false }) selected: string | null = null;
  @property({ attribute: false }) onSaved: () => void = () => undefined;
  @state() private router: RouterDoc | null = null;
  @state() private packages: string[] = [];
  @state() private roleHealth: Record<string, RoleHealth> = {};
  @state() private roleContract: RolesContract | null = null;
  @state() private roleTest: Record<string, string> = {};
  private ed: DocEditor<ProfileDoc> | null = null;
  private roles = new DocEditor<Record<string, RoleDoc>>(this, 'roles');
  private residency = new DocEditor<ResidencyDoc>(this, 'residency');
  @state() private newClaim = '';
  private editing: string | null = null;

  connectedCallback(): void {
    super.connectedCallback();
    this.roles.onSaved = () => void this.loadHealth();
    void this.roles.load();
    void this.residency.load();
    void this.loadHealth();
    void conductor.doc<RouterDoc>('router').then((r) => (this.router = r.doc)).catch(() => undefined);
    void conductor.doc<PackagesDoc>('packages').then((p) => (this.packages = classPackages(p.doc))).catch(() => undefined);
  }

  protected willUpdate(): void {
    if (this.selected !== this.editing) {
      this.editing = this.selected;
      this.ed = this.selected ? new DocEditor<ProfileDoc>(this, 'profiles', this.selected) : null;
      if (this.ed) {
        this.ed.onSaved = () => this.onSaved();
        void this.ed.load();
      }
    }
  }

  private async loadHealth() {
    try {
      const r = await conductor.roles();
      this.roleHealth = r.health;
      this.roleContract = r.contract ?? null;
    } catch {
      /* the roles editor shows its own error */
    }
  }

  private async testRole(role: string) {
    this.roleTest = { ...this.roleTest, [role]: 'testing…' };
    try {
      const r = await conductor.roleTest(role);
      this.roleTest = { ...this.roleTest, [role]: r.ok ? `ok in ${ms(r.ms)}: "${r.answer?.label}" at ${r.answer?.confidence.toFixed(2)}` : `failed: ${r.error}` };
    } catch (e) {
      this.roleTest = { ...this.roleTest, [role]: describeError(e).message };
    }
    void this.loadHealth();
  }

  render() {
    return html`
      <div class="tiles" data-profile-tiles>
        ${this.tiles.map(
          (t) => html`<button class="tile ${t.name === this.selected ? 'is-on' : ''}" data-profile=${t.name} @click=${() => setParams({ edit: t.name })}>
            <div class="tile-head"><span class="tile-name">${t.name}</span>${t.locked ? html`<span class="badge badge-lock">${icon('warn')} locked</span>` : html`<span class="badge ${t.mode === 'on' ? 'badge-on' : t.mode === 'shadow' ? 'badge-shadow' : ''}">${t.mode}</span>`}</div>
            <div class="tile-meta">${t.locked ?? t.description}</div>
            <div class="tile-meta">clients: ${t.clients.length ? t.clients.join(', ') : '— (header or default)'} · default class ${t.default_class}</div>
            <div class="tile-meta num">decisions on ${t.counts.decisions.on ?? 0} · shadow ${t.counts.decisions.shadow ?? 0} · tasks act ${t.counts.tasks.act ?? 0} · shadow ${t.counts.tasks.shadow ?? 0}</div>
            <div class="tile-meta num">v${t.version} · ${t.last_change.note ?? ''}${t.last_change.created_at ? ` · ${dateTimeShort(t.last_change.created_at)}` : ''}</div>
          </button>`,
        )}
      </div>
      ${this.ed ? this.renderEditor(this.ed) : html`<p class="muted small" style="margin-top:14px">Pick a profile to edit it.</p>`}
      ${this.renderRoles()}
      ${this.renderResidency()}
    `;
  }

  private upd(path: (string | number)[], v: unknown) {
    if (!this.ed?.draft) return;
    this.ed.set(setIn(this.ed.draft, path, v));
  }

  /** A context number, and the task threshold that mirrors it (the server reads the context). */
  private ctx(key: 's1_runs' | 's4_minutes', tid: string, threshold: string, v: number) {
    if (!this.ed?.draft || !Number.isFinite(v)) return;
    let d = setIn(this.ed.draft, ['context', key], v);
    if (d.tasks[tid]?.thresholds && threshold in d.tasks[tid].thresholds) d = setIn(d, ['tasks', tid, 'thresholds', threshold], v);
    this.ed.set(d);
  }

  /** TandemLLM's steps as lines the client shows (`ui.states`), per kind, where they go and the
   *  progress lines (S1). */
  private renderUi(d: ProfileDoc, ro: boolean): TemplateResult {
    const ui = d.ui ?? { states: 'off' as const };
    const kinds = ui.kinds ?? {};
    const off = ro || ui.states !== 'on';
    const channel = statesChannel(ui);
    const prog = progressOf(ui);
    const progNum = (key: 'prefill_every_s' | 'thinking_every_s' | 'thinking_every_tokens', label: string, min: number, max: number, step: number) =>
      html`<label class="field"><span>${label}</span><input type="number" inputmode="numeric" min=${min} max=${max} step=${step} ?disabled=${off || prog.mode !== 'on'} data-ui-progress=${key} .value=${String(prog[key])}
        @input=${(e: Event) => { const v = Math.round(Number((e.target as HTMLInputElement).value)); if (Number.isFinite(v) && v >= min && v <= max) this.upd(['ui'], { ...ui, progress: { ...(ui.progress ?? {}), mode: prog.mode, [key]: v } }); }} /></label>`;
    return html`<details class="sec" data-sec="ui">
      <summary>What the user sees <span class="sec-sub">TandemLLM's own steps as lines the client shows</span> <span class="badge ${ui.states === 'on' ? 'badge-on' : ''}" data-ui-badge>${ui.states}</span></summary>
      <div class="sec-body">
        <div class="toolbar" style="margin:0"><span class="muted small">state lines</span>${modeSeg(['off', 'on'] as const, ui.states, (v) => this.upd(['ui'], { ...ui, states: v }), { disabled: ro, small: true, label: 'state lines' })}</div>
        <p class="muted small">Each step goes out at the moment it happens as one line, <code>› deciding: code_edit, high effort</code>. The server takes them out of the reasoning and the content a client sends back, so Qwen never sees them; they are not tokens and not in the usage. Only in mode on.</p>
        <div class="toolbar" style="margin:0" data-ui-channel><span class="muted small">where</span>${modeSeg(['reasoning', 'content', 'inline'] as const, channel, (v) => this.upd(['ui'], { ...ui, states_channel: v }), { disabled: off, small: true, label: 'where the lines go' })}</div>
        <p class="muted small">${channel === 'inline'
          ? html`Each line opens a thinking block of its own and is its title, in time order: opencode shows <code>Thought: › thinking 2 min, 4,096 tokens</code> between Qwen's thinking blocks, and the answer after them. Needs <code>reasoning_content</code>; with <code>&lt;think&gt;</code> in the content (Hermes) the lines go as in the content.`
          : channel === 'content'
          ? html`In the content, outside the thinking: before the reasoning, between its parts, after it. opencode shows them in its text part; with <code>&lt;think&gt;</code> in the content (Hermes) a line said inside the block goes out right after <code>&lt;/think&gt;</code>.`
          : html`In the reasoning opencode and Hermes show (a <code>reasoning_content</code> delta, or inside <code>&lt;think&gt;</code>). A profile saved before this choice existed keeps them here.`}</p>
        <div class="pkg-picks" data-ui-kinds>${STATE_KINDS.map(({ id, example }) => {
          const on = kinds[id] !== false;
          return html`<button class="pick ${on ? 'is-on' : ''}" data-ui-kind=${id} title=${example} aria-pressed=${on ? 'true' : 'false'} ?disabled=${off} @click=${() => this.upd(['ui'], { ...ui, kinds: { ...kinds, [id]: !on } })}>${id}</button>`;
        })}</div>
        <div class="toolbar" style="margin:0" data-ui-progress-mode><span class="muted small">progress</span>${modeSeg(['off', 'on'] as const, prog.mode, (v) => this.upd(['ui'], { ...ui, progress: { ...(ui.progress ?? {}), mode: v } }), { disabled: off, small: true, label: 'progress lines' })}</div>
        <p class="muted small">The numbers of a wait, templated: <code>waiting: opencode's title request runs first, position 1</code> while the request waits, <code>reading 35,189 tokens at ~750 tok/s, about 47 s left</code> during a long prefill, <code>thinking 4 min, 9,800 tokens</code> during a long thinking. Streams only; a request that waits opens its stream at its first waiting line.</p>
        <div class="form-grid">
          ${progNum('prefill_every_s', 'a prefill line every (s)', 2, 600, 1)}
          ${progNum('thinking_every_s', 'a thinking line every (s)', 10, 3600, 10)}
          ${progNum('thinking_every_tokens', 'or every (tokens)', 256, 262144, 256)}
        </div>
        <div class="toolbar" style="margin:0" data-ui-narrator><span class="muted small">narrator</span>${modeSeg(['off', 'on', 'thoughts'] as const, ui.narrator ?? 'off', (v) => this.upd(['ui'], { ...ui, narrator: v }), { disabled: off, small: true, label: 'narrator' })}</div>
        <p class="muted small">A small model inside the engine (the <code>narrator</code> role, Qwen3.5-0.8B) writes in plain words. <b>thoughts</b>: after each stretch of thinking (and at the long-thinking marks) one line with what Qwen thought about and decided, e.g. “Thought 1m 43s: planned the level layout …”, read from the thought alone; the other lines stay templates. <b>on</b>: it writes every line instead of the template. Qwen waits while it writes, at most 3 % of its time (the rest at the turn’s end); a line it cannot write in time is the template. It loads on its first line; the Roles section below holds its budget and mode.</p>
      </div>
    </details>`;
  }

  /** F2: the effort of a tool-result turn for this profile; "Routing's" keeps the router's. */
  private renderToolTurn(d: ProfileDoc, ro: boolean): TemplateResult {
    const own = d.thinking.tool_turn_effort;
    const router = toolTurnEffort(this.router, null).level;
    const now = toolTurnEffort(this.router, d);
    return html`<label class="field ${own ? '' : 'is-inherited'}" data-tool-turn-effort><span>effort of a tool-result turn</span>
      <select aria-label="effort of a tool-result turn" ?disabled=${ro} @change=${(e: Event) => {
        const v = (e.target as HTMLSelectElement).value;
        if (!this.ed?.draft) return;
        const th: ProfileDoc['thinking'] = { ...this.ed.draft.thinking, tool_turn_effort: v as ToolTurnEffort };
        if (!v) delete th.tool_turn_effort;
        this.ed.set(setIn(this.ed.draft, ['thinking'], th));
      }}>
        <option value="" ?selected=${!own}>Routing's: ${router === 'inherit' ? "the class's" : router}</option>
        ${TOOL_TURN_EFFORTS.map((lv) => html`<option value=${lv} ?selected=${own === lv}>${lv === 'inherit' ? "the class's (inherit)" : lv}</option>`)}
      </select>
      <span class="hint">${toolTurnText(now.level, this.router, d)}${now.source === 'profile' ? ' (this profile)' : now.source === 'router' ? ' (Routing)' : ''}: a turn whose newest message is a tool result thinks at this level, its switch and cap; the first turn and every new user message keep the class's</span></label>`;
  }

  /** SRV-159: each level's reasoning cap for this profile; an empty field keeps the router's. */
  private renderBudgets(d: ProfileDoc, ro: boolean): TemplateResult {
    return html`<div data-budgets>
      <div class="muted small">Thinking budget per level <span class="muted">(reasoning tokens; empty keeps Routing's level, 0 is no cap below the backstop)</span></div>
      <div class="form-grid">
        ${BUDGET_LEVELS.map((lv) => {
          const own = d.thinking.budgets?.[lv];
          const router = levelBudget(lv, this.router, null).cap;
          const now = levelBudget(lv, this.router, d);
          return html`<label class="field ${own == null ? 'is-inherited' : ''}" data-budget=${lv}><span class="mono">${lv}</span>
            <input type="number" inputmode="numeric" min="0" max=${BUDGET_MAX} step="1024" ?disabled=${ro} placeholder=${router == null ? '' : String(router)} .value=${own == null ? '' : String(own)}
              @change=${(e: Event) => {
                const raw = (e.target as HTMLInputElement).value.trim();
                if (this.ed?.draft) this.ed.set(setIn(this.ed.draft, ['thinking'], withBudget(this.ed.draft.thinking, lv, raw === '' ? null : Number(raw))));
              }} />
            <span class="hint">${budgetText(now.cap)}${now.source === 'router' ? ' (Routing)' : now.source === 'profile' ? ' (this profile)' : ''}</span></label>`;
        })}
      </div>
    </div>`;
  }

  private renderEditor(ed: DocEditor<ProfileDoc>): TemplateResult {
    if (ed.loading && !ed.draft) return skeleton(8);
    if (ed.loadError) return html`<div class="state state-error" role="alert"><div class="state-title">${icon('warn')} ${ed.loadError}</div></div>`;
    const d = ed.draft!;
    const ro = !!ed.locked;
    const kinds = this.meta?.client_kinds ?? ['open-webui', 'openai-sdk', 'opencode', 'curl', 'dashboard', 'other'];
    const classes = this.router?.classes ?? [];
    return html`<div style="margin-top:16px" data-editor=${ed.name}>
      <div class="toolbar"><h2 class="panel-title">${ed.name} <span class="muted small num">v${ed.base?.version}</span></h2>
        <a class="btn btn-sm" href=${`#/playground?profile=${encodeURIComponent(ed.name ?? '')}`} data-try>${icon('playground')} Try in Playground</a></div>
      ${ro ? html`<div class="note" data-lock-reason>${icon('warn')} ${ed.locked}. Every control below is read-only.</div>` : nothing}
      <details class="sec" open data-sec="mapping">
        <summary>Mapping <span class="sec-sub">who gets this profile, and the class it starts from</span></summary>
        <div class="sec-body">
          <label class="field field-block"><span>description</span><input .value=${d.description} ?disabled=${ro} @input=${(e: Event) => this.upd(['description'], (e.target as HTMLInputElement).value)} /></label>
          <div class="toolbar" style="margin:0"><span class="muted small">mode (the header switch sets this too)</span>${modeSeg(['raw', 'shadow', 'on'] as const, d.mode, (v) => this.upd(['mode'], v), { disabled: ro, small: true, label: 'profile mode' })}</div>
          <div><span class="muted small">client kinds that resolve to this profile (model <code>TandemLLM</code>, no <code>X-Tandem-Profile</code> header)</span>
            <div class="pkg-picks">${kinds.map((k) => {
              const on = d.clients.includes(k);
              return html`<button class="pick ${on ? 'is-on' : ''}" ?disabled=${ro} aria-pressed=${on ? 'true' : 'false'} @click=${() => this.upd(['clients'], on ? d.clients.filter((x) => x !== k) : [...d.clients, k])}>${k}</button>`;
            })}</div></div>
          <label class="field"><span>default class (a new session nobody could classify)</span>
            <select ?disabled=${ro} @change=${(e: Event) => this.upd(['default_class'], (e.target as HTMLSelectElement).value)}>${classes.map((c) => html`<option value=${c.id} ?selected=${c.id === d.default_class}>${c.id}</option>`)}</select></label>
        </div>
      </details>
      <details class="sec" open data-sec="decisions">
        <summary>Routing decisions <span class="sec-sub">off, shadow (journal only) or on, each on its own</span></summary>
        <div class="sec-body">
          ${(Object.keys(DECISION_WORDS) as Decision[]).map(
            (k) => html`<div class="toolbar" style="margin:0" data-decision-mode=${k}>${modeSeg(['off', 'shadow', 'on'] as Mode3[], d.decisions[k], (v) => this.upd(['decisions', k], v), { disabled: ro, small: true, label: k })}<span class="small">${DECISION_WORDS[k]}</span></div>`,
          )}
        </div>
      </details>
      <details class="sec" data-sec="overrides">
        <summary>Per-class overrides <span class="sec-sub">this profile's own effort, packages or rules for a class; grey = the router's</span></summary>
        <div class="sec-body">
          ${classes.map((c) => {
            const ov = d.routing[c.id] ?? {};
            const has = Object.keys(ov).length > 0;
            return html`<div class="row-card" data-override=${c.id}>
              <div class="row-card-head">${clsChip(c.id, this.ids, c.label)}
                ${has ? html`<button class="btn btn-ghost btn-sm" ?disabled=${ro} @click=${() => {
                  const r = { ...d.routing };
                  delete r[c.id];
                  this.upd(['routing'], r);
                }}>use the router's</button>` : html`<span class="muted small">router: ${c.policy.effort ?? 'medium'}${(c.policy.packages ?? []).length ? ` · ${(c.policy.packages ?? []).join(', ')}` : ''}</span>`}</div>
              <div class="toolbar" style="margin:0"><span class="muted small">effort</span>${modeSeg(EFFORTS, (ov.effort ?? c.policy.effort ?? 'medium') as Effort, (v) => this.upd(['routing', c.id], { ...ov, effort: v }), { disabled: ro, small: true, label: `effort override of ${c.id}`, cls: ov.effort ? '' : 'is-inherited' })}</div>
              <div class="pkg-picks">${this.packages.map((name) => {
                const cur = ov.packages ?? c.policy.packages ?? [];
                const on = cur.includes(name);
                return html`<button class="pick ${on ? 'is-on' : ''}" ?disabled=${ro} @click=${() => this.upd(['routing', c.id], { ...ov, packages: on ? cur.filter((x) => x !== name) : [...cur, name] })}>${name}</button>`;
              })}</div>
            </div>`;
          })}
        </div>
      </details>
      <details class="sec" data-sec="tasks">
        <summary>Session monitor tasks <span class="sec-sub">the frame (SRV-109) runs these; act needs the role online</span></summary>
        <div class="sec-body">${Object.entries(d.tasks).map(([tid, t]) => this.renderTask(tid, t.mode, t.cap, t.thresholds, ro))}</div>
      </details>
      <details class="sec" data-sec="thinking">
        <summary>Thinking <span class="sec-sub">the effort when no class sets one</span></summary>
        <div class="sec-body">
          <div class="toolbar" style="margin:0"><span class="muted small">default</span>${modeSeg(EFFORTS, d.thinking.default, (v) => this.upd(['thinking', 'default'], v), { disabled: ro, small: true, label: 'default effort' })}</div>
          <label class="switch"><input type="checkbox" ?disabled=${ro} .checked=${d.thinking.hints} @change=${(e: Event) => this.upd(['thinking', 'hints'], (e.target as HTMLInputElement).checked)} /> hints from the last tool result (a checker FAIL raises the effort)</label>
          <div class="toolbar" style="margin:0" data-by-size><span class="muted small">effort by task size</span>${modeSeg(['off', 'shadow', 'on'] as Mode3[], d.thinking.by_size ?? BY_SIZE_DEFAULT, (v) => this.upd(['thinking', 'by_size'], v), { disabled: ro, small: true, label: 'effort by task size', cls: d.thinking.by_size ? '' : 'is-inherited' })}</div>
          <p class="muted small">Effort by task size (SRV-157): a class with levels per size (Routing › the class) thinks at its size's level, never above the class's effort; the size only rises within a task. <strong>off</strong>: the class's effort; <strong>shadow</strong>: the class's effort, and the journal shows the size and the effort it would set; <strong>on</strong>: the size picks.</p>
          ${this.renderToolTurn(d, ro)}
          ${this.renderBudgets(d, ro)}
          <p class="muted small">A client's own <code>max_reasoning_tokens</code> or <code>thinking_budget</code> always wins (fixed).</p>
        </div>
      </details>
      <details class="sec" data-sec="library">
        <summary>Package library <span class="sec-sub">a searched skill for a new message or a stuck run (R1)</span></summary>
        <div class="sec-body">
          <div class="toolbar" style="margin:0" data-library-mode><span class="muted small">library</span>${modeSeg(['off', 'shadow', 'on'] as Mode3[], d.library ?? 'off', (v) => this.upd(['library'], v), { disabled: ro, small: true, label: 'package library', cls: d.library ? '' : 'is-inherited' })}</div>
          <p class="muted small">On a new user message, and on a low-quality or stuck outcome, the router searches the library (the <a href="#/conductor?screen=packages">Packages</a> screen) with the message, asks the package question among the best matches and puts the skill that fits at the prompt's tail. <strong>off</strong>: nothing; <strong>shadow</strong>: the journal shows the pick, nothing goes in; <strong>on</strong>: the skill and the card that tells the model how to search the library itself.</p>
        </div>
      </details>
      <details class="sec" data-sec="toolgate">
        <summary>Tool gate <span class="sec-sub">rule packs on held tool calls (SRV-113); a pack that is off turns its task off</span></summary>
        <div class="sec-body"><div class="form-grid">
          ${PACKS.map(([k, tid, words]) => html`<label class="switch" data-pack=${k}><input type="checkbox" ?disabled=${ro} .checked=${d.toolgate[k]} @change=${(e: Event) => this.upd(['toolgate', k], (e.target as HTMLInputElement).checked)} /> ${words} <span class="muted small">(${tid}, ${d.toolgate[k] ? d.tasks[tid]?.mode ?? 'off' : 'off'})</span></label>`)}
          <label class="field"><span>Tier 1 timeout (ms)</span><input type="number" ?disabled=${ro} .value=${String(d.toolgate.tier1_timeout_ms)} @input=${(e: Event) => this.upd(['toolgate', 'tier1_timeout_ms'], Math.round(Number((e.target as HTMLInputElement).value)))} /></label>
        </div>
        <p class="muted small">With a pack on, its task runs in the mode set above; with one of them able to act, an answer's tool calls are held to its end and released whole. The floors, rules and reminders are on the <a href="#/conductor?screen=monitor">Monitor</a> screen.</p></div>
      </details>
      <details class="sec" data-sec="context">
        <summary>Context <span class="sec-sub">the session tasks S1, S2, S4 (SRV-121): their numbers and the push</span></summary>
        <div class="sec-body"><div class="form-grid">
          <label class="field"><span>S1: failed runs of one gate before the blocked-stage note <span class="hint">count, default 3</span></span><input type="number" min="1" max="20" ?disabled=${ro} data-context="s1_runs" .value=${String(d.context.s1_runs)} @input=${(e: Event) => this.ctx('s1_runs', 'S1', 'fails', Math.round(Number((e.target as HTMLInputElement).value)))} /></label>
          <label class="field"><span>S4: minutes without progress before the note, and of silence before the client counts as gone <span class="hint">minutes, default 20</span></span><input type="number" min="1" max="600" ?disabled=${ro} data-context="s4_minutes" .value=${String(d.context.s4_minutes)} @input=${(e: Event) => this.ctx('s4_minutes', 'S4', 'minutes', Math.round(Number((e.target as HTMLInputElement).value)))} /></label>
          <label class="switch" data-context="notify"><input type="checkbox" ?disabled=${ro} .checked=${d.context.notify} @change=${(e: Event) => this.upd(['context', 'notify'], (e.target as HTMLInputElement).checked)} /> NOTIFY: a push through ntfy to the phone when the client has gone <span class="badge ${d.context.notify ? 'badge-warn' : ''}">${d.context.notify ? 'on' : 'off'}</span></label>
        </div>
        <p class="muted small" data-notify-rule>Allowed by Khaled on 2026-09-30: off by default, on in the overnight profile. Only S4 in act pushes, never in shadow, once a turn end; the token is a file on the box. S2 (re-read after a compaction) has no number here: its texts and state files are on the <a href="#/conductor?screen=monitor">Monitor</a> screen.</p></div>
      </details>
      ${this.renderUi(d, ro)}
      <details class="sec" data-sec="hints">
        <summary>Speed hints <span class="badge badge-warn">not gated</span></summary>
        <div class="sec-body">
          <p class="warn-ink small">No SPD measurement is in the journal yet (SPD-73); the router's ≤ 1 ulp text rule and the never-worse gate come first.</p>
          <label class="switch"><input type="checkbox" ?disabled=${ro} .checked=${d.hints.stair_width} @change=${(e: Event) => this.upd(['hints', 'stair_width'], (e.target as HTMLInputElement).checked)} /> StairCut width hint</label>
          <label class="switch"><input type="checkbox" ?disabled=${ro} .checked=${d.hints.prefix_retention} @change=${(e: Event) => this.upd(['hints', 'prefix_retention'], (e.target as HTMLInputElement).checked)} /> prefix retention hint</label>
        </div>
      </details>
      ${errorsList(ed)}
      ${saveBar(ed, `profile ${ed.name}`)}
    </div>`;
  }

  private renderTask(tid: string, mode: TaskMode, cap: number, thresholds: Record<string, number>, ro: boolean): TemplateResult {
    const m = this.meta?.tasks[tid];
    const needRole = m?.role && m.tier === '1';
    const roleUp = !needRole || !!this.roleHealth[m!.role!]?.up;
    return html`<div class="task-row" data-task=${tid}>
      <div><div class="task-id">${tid}</div><div class="muted small">${m?.family ?? ''}</div></div>
      <div>${m?.label ?? tid}<div class="task-why">${m?.why ?? ''}</div>${needRole && !roleUp ? html`<div class="warn-ink small">${icon('warn')} the ${m!.role} role is offline: act is refused</div>` : nothing}</div>
      <div class="task-mode">${modeSeg(['off', 'shadow', 'act'] as TaskMode[], mode, (v) => this.upd(['tasks', tid, 'mode'], v), { disabled: ro, small: true, label: `${tid} mode` })}</div>
      <div class="thresholds">
        ${CONTEXT_TASKS.has(tid) && Object.keys(thresholds).length ? html`<p class="muted small" data-in-context=${tid}>${Object.entries(thresholds).map(([k, v]) => `${k.replace(/_/g, ' ')} ${v}`).join(' · ')}: set in Context below</p>` : nothing}
        ${Object.entries(CONTEXT_TASKS.has(tid) ? {} : thresholds).map(([k, v]) => {
          const def = m?.thresholds[k];
          return html`<label class="field ${def && def.default !== v ? 'is-changed' : ''}"><span>${k.replace(/_/g, ' ')} <span class="hint">${def?.unit ?? ''}${def ? `, default ${def.default}` : ''}</span></span>
            <input type="number" ?disabled=${ro} .value=${String(v)} step="any" @input=${(e: Event) => {
              const x = Number((e.target as HTMLInputElement).value);
              if (Number.isFinite(x)) this.upd(['tasks', tid, 'thresholds', k], x);
            }} /></label>`;
        })}
        <label class="field"><span>cap <span class="hint">acts a request</span></span><input type="number" ?disabled=${ro} min="0" max="10" .value=${String(cap)} @input=${(e: Event) => this.upd(['tasks', tid, 'cap'], Math.round(Number((e.target as HTMLInputElement).value)))} /></label>
      </div>
    </div>`;
  }

  private renderRoles(): TemplateResult | typeof nothing {
    const ed = this.roles;
    if (!ed.draft) return ed.loadError ? html`<p class="err-ink small">${ed.loadError}</p>` : nothing;
    const d = ed.draft;
    const upd = (path: (string | number)[], v: unknown) => ed.set(setIn(d, path, v));
    const defaults = defaultsByRole(ed.base?.meta.defaults);
    const engineSha = this.roleContract?.engine_tokenizer_sha256 ?? null;
    const enums = this.meta?.roles;
    const kinds = enums?.kinds ?? ROLE_KINDS;
    const windows = enums?.windows ?? ROLE_WINDOWS;
    const fallbacks: RoleFallback[] = enums?.fallbacks ?? ['PASS', 'stale', 'skip', 'next-role'];
    const writes = ROLE_WRITES.filter((w) => !enums || enums.writes.includes(w));
    return html`<details class="sec" style="margin-top:16px" data-sec="roles">
      <summary>Roles <span class="sec-sub">shared by every profile; a role is a kind of question, only this names models</span></summary>
      <div class="sec-body">
        ${ed.base?.meta.defaults?.length ? html`<div class="note" data-role-defaults>${icon('warn')} Version ${ed.base.version} was stored before the role contract and loaded with ${ed.base.meta.defaults.length} default${ed.base.meta.defaults.length === 1 ? '' : 's'} (marked <span class="badge">default</span> below; the journal has the list). Save the roles to make them explicit.</div>` : nothing}
        <p class="muted small">The contract (SRV-133): when a role may use the GPU, whether a Qwen request cuts it, how long its caller waits before the fallback, what it may write, and the tokenizer it reads, which decides what it may share with Qwen. Engine tokenizer: <code class="mono" data-engine-sha>${engineSha ? `${engineSha.slice(0, 12)}…` : 'not recorded'}</code></p>
        ${Object.entries(d).map(([name, r]) => {
          const h = this.roleHealth[name];
          const dflt = new Set((defaults[name] ?? []).map((x) => x.split('=')[0]));
          const tag = (f: string) => (dflt.has(f) ? html` <span class="badge" data-default=${`${name}.${f}`}>default</span>` : nothing);
          const winWhy = windowRefusal(r.kind, r.window);
          const wrWhy = writesRefusal(r.kind, r.writes);
          const shares = sharingOptions(r, engineSha);
          return html`<div class="row-card" data-role-row=${name}>
            <div class="row-card-head"><strong>${name}</strong>
              <span class="toolbar" style="margin:0">${h ? html`<span class="badge ${h.up ? 'badge-ok' : r.mode === 'off' ? '' : 'badge-warn'}">${h.up ? `up ${ms(h.ms)}` : r.mode === 'off' ? 'off' : `down: ${h.error ?? '?'}`}</span>` : nothing}
                <button class="btn btn-sm" @click=${() => this.testRole(name)} data-role-test=${name}>Test</button></span></div>
            ${this.roleTest[name] ? html`<p class="small" role="status">${this.roleTest[name]}</p>` : nothing}
            <div class="form-grid">
              <label class="field"><span>endpoint (self = this server)</span><input .value=${r.endpoint} @input=${(e: Event) => upd([name, 'endpoint'], (e.target as HTMLInputElement).value)} /></label>
              <label class="field"><span>model</span><input .value=${r.model} @input=${(e: Event) => upd([name, 'model'], (e.target as HTMLInputElement).value)} /></label>
              <label class="field"><span>timeout (ms)</span><input type="number" .value=${String(r.timeout_ms)} @input=${(e: Event) => upd([name, 'timeout_ms'], Math.round(Number((e.target as HTMLInputElement).value)))} /></label>
              <label class="field"><span>memory claim (GiB, its peak)${tag('memory_claim_gib')}</span><input type="number" step="0.1" min="0" data-role-field="memory_claim_gib" .value=${String(r.memory_claim_gib)} @input=${(e: Event) => upd([name, 'memory_claim_gib'], Number((e.target as HTMLInputElement).value))} /></label>
            </div>
            <div class="form-grid" data-role-contract=${name}>
              <label class="field"><span>kind${tag('kind')}</span><select data-role-field="kind" @change=${(e: Event) => upd([name, 'kind'], (e.target as HTMLSelectElement).value as RoleKind)}>${kinds.map((k) => html`<option value=${k} ?selected=${k === r.kind}>${k}</option>`)}</select></label>
              <label class="field"><span>window${tag('window')}</span><select data-role-field="window" @change=${(e: Event) => upd([name, 'window'], (e.target as HTMLSelectElement).value as RoleWindow)}>${windows.map((w) => html`<option value=${w} ?selected=${w === r.window} ?disabled=${w !== r.window && !!windowRefusal(r.kind, w)} title=${WINDOW_WORDS[w]}>${w}</option>`)}</select></label>
              <label class="field"><span>latency budget (ms)${tag('latency_budget_ms')}</span><input type="number" min="1" max="600000" data-role-field="latency_budget_ms" .value=${String(r.latency_budget_ms)} @input=${(e: Event) => upd([name, 'latency_budget_ms'], Math.round(Number((e.target as HTMLInputElement).value)))} /></label>
              <label class="field"><span>fallback (off, slow, unsure or refused)${tag('fallback')}</span><select data-role-field="fallback" @change=${(e: Event) => upd([name, 'fallback'], (e.target as HTMLSelectElement).value as RoleFallback)}>${fallbacks.map((f) => html`<option value=${f} ?selected=${f === r.fallback}>${f}</option>`)}</select></label>
              <label class="field"><span>writes${tag('writes')}</span><select data-role-field="writes" @change=${(e: Event) => upd([name, 'writes'], (e.target as HTMLSelectElement).value as RoleWrites)}>${writes.map((w) => html`<option value=${w} ?selected=${w === r.writes} ?disabled=${w !== r.writes && !!writesRefusal(r.kind, w)}>${w.replace(/_/g, ' ')}</option>`)}</select></label>
              <label class="field field-wide"><span>tokenizer sha256 (the engine records its own; none: text only)${tag('tokenizer_sha256')}</span><input class="mono" data-role-field="tokenizer_sha256" spellcheck="false" .value=${r.tokenizer_sha256} @input=${(e: Event) => upd([name, 'tokenizer_sha256'], (e.target as HTMLInputElement).value.trim().toLowerCase() || 'none')} /></label>
            </div>
            <div class="toolbar" style="margin:0">
              <label class="switch"><input type="checkbox" data-role-field="preemptible" .checked=${r.preemptible} @change=${(e: Event) => upd([name, 'preemptible'], (e.target as HTMLInputElement).checked)} /> preemptible: a Qwen request cuts it, it resumes later${tag('preemptible')}</label>
            </div>
            <div><span class="muted small">sharing with Qwen${tag('sharing')}</span>
              <div class="pkg-picks" data-role-sharing=${name}>${shares.map((o) => {
                const on = r.sharing.includes(o.form);
                return html`<button class="pick ${on ? 'is-on' : ''}" aria-pressed=${on ? 'true' : 'false'} ?disabled=${!on && !o.allowed} title=${o.why ?? ''} data-share=${o.form} @click=${() => upd([name, 'sharing'], on ? r.sharing.filter((x) => x !== o.form) : [...r.sharing, o.form])}>${o.label}</button>`;
              })}</div>
              ${shares.filter((o) => !o.allowed).map((o) => html`<p class="muted small" data-share-why=${o.form}>${o.label}: ${o.why}</p>`)}
            </div>
            ${winWhy ? html`<p class="warn-ink small" data-role-why="window">${icon('warn')} ${winWhy}</p>` : nothing}
            ${wrWhy ? html`<p class="warn-ink small" data-role-why="writes">${icon('warn')} ${wrWhy}</p>` : nothing}
            <div class="toolbar" style="margin:0"><span class="muted small">mode</span>${modeSeg(['off', 'shadow', 'on'] as Mode3[], r.mode, (v) => upd([name, 'mode'], v), { small: true, label: `${name} mode` })}<span class="muted small mono">${r.contract}</span></div>
          </div>`;
        })}
        ${errorsList(ed)}
        ${saveBar(ed, 'the roles')}
      </div>
    </details>`;
  }

  private renderResidency(): TemplateResult | typeof nothing {
    const ed = this.residency;
    if (!ed.draft) return ed.loadError ? html`<p class="err-ink small">${ed.loadError}</p>` : nothing;
    const d = ed.draft;
    const upd = (path: (string | number)[], v: unknown) => ed.set(setIn(d, path, v));
    const num = (path: string[], step = '0.1', int = false) => html`<input type="number" step=${step} min="0" .value=${String((d as unknown as Record<string, number>)[path[0]])} @input=${(e: Event) => { const v = Number((e.target as HTMLInputElement).value); upd(path, int ? Math.round(v) : v); }} data-res=${path[0]} />`;
    const bases: ResidencyBasis[] = ['measured', 'peak', 'declared'];
    return html`<details class="sec" style="margin-top:16px" data-sec="residency">
      <summary>Residency <span class="sec-sub">what may load beside the engine: a role loads when MemAvailable minus its claim stays at or above floor + margin, one at a time, never during a hold</span></summary>
      <div class="sec-body">
        <div class="form-grid">
          <label class="field"><span>floor (GiB of MemAvailable)</span>${num(['floor_gib'])}</label>
          <label class="field"><span>margin above the floor (GiB)</span>${num(['margin_gib'])}</label>
          <label class="field"><span>settle after a stop (s)</span>${num(['settle_s'], '1', true)}</label>
          <label class="field"><span>load order (roles, first loads first)</span><input .value=${d.order.join(', ')} @change=${(e: Event) => upd(['order'], (e.target as HTMLInputElement).value.split(/[\s,]+/).filter(Boolean))} data-res="order" /></label>
        </div>
        ${Object.entries(d.claims).map(
          ([role, c]) => html`<div class="row-card" data-claim=${role}>
            <div class="row-card-head"><strong>${role}</strong><button class="btn btn-ghost btn-sm" @click=${() => { const { [role]: _drop, ...rest } = d.claims; upd(['claims'], rest); }} aria-label=${`remove the claim of ${role}`}>${icon('trash')}</button></div>
            <div class="form-grid">
              <label class="field"><span>claim (GiB, its peak)</span><input type="number" step="0.1" min="0" .value=${String(c.gib)} @input=${(e: Event) => upd(['claims', role, 'gib'], Number((e.target as HTMLInputElement).value))} data-claim-gib=${role} /></label>
              <label class="field"><span>basis</span><select @change=${(e: Event) => upd(['claims', role, 'basis'], (e.target as HTMLSelectElement).value)}>${bases.map((b) => html`<option value=${b} ?selected=${b === c.basis}>${b}</option>`)}</select></label>
              <label class="field field-wide"><span>note (where the number comes from)</span><input .value=${c.note} @input=${(e: Event) => upd(['claims', role, 'note'], (e.target as HTMLInputElement).value)} /></label>
            </div>
          </div>`,
        )}
        <div class="toolbar"><label class="field"><span>a claim for another role</span><input placeholder="role id, e.g. librarian" .value=${this.newClaim} @input=${(e: Event) => (this.newClaim = (e.target as HTMLInputElement).value.trim())} /></label>
          <button class="btn btn-sm" ?disabled=${!/^[a-z][a-z0-9_-]{0,39}$/.test(this.newClaim) || this.newClaim in d.claims} @click=${() => { upd(['claims', this.newClaim], { gib: 0, basis: 'declared', note: '' }); this.newClaim = ''; }}>${icon('plus')} Add</button></div>
        <p class="muted small">The claims are peaks, not idle figures, and MemAvailable, not nvidia-smi (arbiter: 5.1 GB there, 8.2 GiB of MemAvailable). What fits now, the engine's own split and the last refusal are on the <a href="#/system">System view</a>.</p>
        ${errorsList(ed)}
        ${saveBar(ed, 'the residency limits')}
      </div>
    </details>`;
  }
}
