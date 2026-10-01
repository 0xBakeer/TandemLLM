// Conductor · Profiles: the tiles (name, mode, clients, counts, version, last change; raw and
// bench with a lock) and the editor of one profile: mapping and default class, the four routing
// decision types (off, shadow, on), per-class overrides of the router's policy, the session
// monitor's tasks (mode, thresholds with units and their defaults, cap, the evidence line),
// thinking, the tool gate, context and speed hints; then the roles every profile shares, each
// with a test button; then the residency planner's limits (OPS-41): the MemAvailable floor and the
// margin a role load must leave, the settle time after a stop, the load order and each role's
// memory claim with its basis. Saving writes a new version; "Try in Playground" opens it there.

import { html, nothing, type TemplateResult } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import { describeError } from '../../api/client';
import type { ResidencyBasis, ResidencyDoc } from '../../api/types';
import { conductor, type Decision, type Effort, type Meta, type Mode3, type PackagesDoc, type ProfileDoc, type ProfileTile, type RoleDoc, type RoleHealth, type RouterDoc, type TaskMode } from '../../api/conductor';
import { setIn } from '../../lib/conductor';
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
    void conductor.doc<PackagesDoc>('packages').then((p) => (this.packages = Object.keys(p.doc.packages))).catch(() => undefined);
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
      this.roleHealth = (await conductor.roles()).health;
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
          <p class="muted small">A client's own <code>max_reasoning_tokens</code> or <code>thinking_budget</code> always wins (fixed).</p>
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
    return html`<details class="sec" style="margin-top:16px" data-sec="roles">
      <summary>Roles <span class="sec-sub">shared by every profile; a role is a kind of question, only this names models</span></summary>
      <div class="sec-body">
        ${Object.entries(d).map(([name, r]) => {
          const h = this.roleHealth[name];
          return html`<div class="row-card" data-role-row=${name}>
            <div class="row-card-head"><strong>${name}</strong>
              <span class="toolbar" style="margin:0">${h ? html`<span class="badge ${h.up ? 'badge-ok' : r.mode === 'off' ? '' : 'badge-warn'}">${h.up ? `up ${ms(h.ms)}` : r.mode === 'off' ? 'off' : `down: ${h.error ?? '?'}`}</span>` : nothing}
                <button class="btn btn-sm" @click=${() => this.testRole(name)} data-role-test=${name}>Test</button></span></div>
            ${this.roleTest[name] ? html`<p class="small" role="status">${this.roleTest[name]}</p>` : nothing}
            <div class="form-grid">
              <label class="field"><span>endpoint (self = this server)</span><input .value=${r.endpoint} @input=${(e: Event) => upd([name, 'endpoint'], (e.target as HTMLInputElement).value)} /></label>
              <label class="field"><span>model</span><input .value=${r.model} @input=${(e: Event) => upd([name, 'model'], (e.target as HTMLInputElement).value)} /></label>
              <label class="field"><span>timeout (ms)</span><input type="number" .value=${String(r.timeout_ms)} @input=${(e: Event) => upd([name, 'timeout_ms'], Math.round(Number((e.target as HTMLInputElement).value)))} /></label>
              <label class="field"><span>memory claim (GB)</span><input type="number" step="0.1" .value=${String(r.memory_gb)} @input=${(e: Event) => upd([name, 'memory_gb'], Number((e.target as HTMLInputElement).value))} /></label>
            </div>
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
