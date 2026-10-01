// Conductor · Compare: whether a decision earned "on", and whether TandemLLM beats the raw engine.
//   * precision per profile and decision type in shadow (from the "right / wrong" labels), the
//     promotion rule in words, and "Promote to on", enabled only when the rule holds; the session
//     monitor's tasks the same way once its journal is on the server;
//   * the classifier report (tools/conductor_eval.py): accuracy per stage on the labelled set,
//     the confusion table, latency;
//   * the suite table (OPS-43): raw and TandemLLM per suite, "not comparable" in orange unless both
//     arms are from the same build and night; the speed guard of the build that serves now;
//   * A/B of two versions of a profile: what changed and how many decisions each version made.

import { html, nothing, type TemplateResult } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import { describeError } from '../../api/client';
import { conductor, type Change, type DecisionRow, type EvalReport, type ProfileDoc, type ProfileTile, type Stats, type VersionRow } from '../../api/conductor';
import { promotion, PROMOTE_RULE, readablePath, setIn, short, speedGuard, suitePairs, type SuiteRow } from '../../lib/conductor';
import { dateTimeShort, exact, ms, pct } from '../../lib/format';
import { LightElement } from '../../ui/base';
import { emptyState, icon, panel, skeleton } from '../../ui/bits';
import { clsChip, confirmDialog, withToken } from './shared';

interface StageReport {
  n: number;
  answered: number;
  correct: number;
  accuracy: number;
  accuracy_answered?: number | null;
  p50_ms: number | null;
  p95_ms: number | null;
  confusion?: Record<string, Record<string, number>>;
}

@customElement('qse-cond-compare')
export class QseCondCompare extends LightElement {
  @property({ attribute: false }) ids: string[] = [];
  @property({ attribute: false }) tiles: ProfileTile[] = [];
  @property({ attribute: false }) onSaved: () => void = () => undefined;
  @state() private stats: Stats | null = null;
  @state() private reports: EvalReport[] = [];
  @state() private engine: { version: string | null; code_sha: string | null } = { version: null, code_sha: null };
  @state() private error: string | null = null;
  @state() private msg: string | null = null;
  @state() private stage = 'cascade';
  @state() private abProfile = 'coding';
  @state() private abVersions: VersionRow[] = [];
  @state() private abA: number | null = null;
  @state() private abB: number | null = null;
  @state() private abChanges: Change[] | null = null;
  @state() private abCounts: Record<number, number> = {};

  connectedCallback(): void {
    super.connectedCallback();
    void this.load();
    void this.loadAb();
  }

  private async load() {
    try {
      const [s, e] = await Promise.all([conductor.stats(168), conductor.evals()]);
      this.stats = s;
      this.reports = e.reports;
      this.engine = e.engine;
      this.error = null;
    } catch (e) {
      const d = describeError(e);
      this.error = `${d.endpoint}: ${d.message}`;
    }
  }

  private async promote(profile: string, what: 'decision' | 'task', id: string, precision: number, labelled: number) {
    const ok = await confirmDialog(`Promote ${id} to ${what === 'task' ? 'act' : 'on'} in ${profile}?`, [
      what === 'task' ? `The session monitor will act on ${id} from the next request of this profile.` : `From the next request of ${profile}, the router applies its ${id} decisions instead of only journaling them: ${id === 'effort' ? 'budgets and thinking off go to the engine' : id === 'packages' ? 'skills and packages go into the prompt' : id === 'rules' ? 'rules go into the note' : 'the class it picks sets the policy'}.`,
      `Precision ${precision.toFixed(2)} over ${labelled} labelled decisions. A new profile version records it.`,
    ], 'Promote');
    if (!ok) return;
    try {
      const cur = await conductor.doc<ProfileDoc>('profiles', profile);
      const doc = what === 'task' ? setIn(cur.doc, ['tasks', id, 'mode'], 'act') : setIn(cur.doc, ['decisions', id], 'on');
      const r = await withToken(() => conductor.put('profiles', profile, doc, `${id} to ${what === 'task' ? 'act' : 'on'}, precision ${precision.toFixed(2)} / ${labelled}`, cur.version));
      this.msg = `${profile} v${r.version}: ${id} promoted`;
      this.onSaved();
    } catch (e) {
      this.msg = describeError(e).message;
    }
  }

  private async loadAb() {
    try {
      this.abVersions = (await conductor.versions('profiles', this.abProfile)).versions;
      this.abA = this.abVersions[1]?.version ?? this.abVersions[0]?.version ?? 0;
      this.abB = this.abVersions[0]?.version ?? 0;
      const rows: DecisionRow[] = (await conductor.decisions({ limit: 500 })).decisions.filter((r) => r.profile === this.abProfile);
      const counts: Record<number, number> = {};
      for (const r of rows) if (r.profile_version != null) counts[r.profile_version] = (counts[r.profile_version] ?? 0) + 1;
      this.abCounts = counts;
      await this.runAb();
    } catch {
      this.abVersions = [];
    }
  }
  private async runAb() {
    if (this.abA == null || this.abB == null) return;
    try {
      this.abChanges = (await conductor.diff('profiles', this.abProfile, this.abA, this.abB)).changes;
    } catch (e) {
      this.msg = describeError(e).message;
    }
  }

  render() {
    if (this.error && !this.stats) return html`<div class="state state-error" role="alert"><div class="state-title">${icon('warn')} ${this.error}</div></div>`;
    if (!this.stats) return skeleton(8);
    return html`
      ${this.msg ? html`<div class="note note-info" role="status">${this.msg}</div>` : nothing}
      ${this.renderPromotion(this.stats)}
      ${this.renderClassifier()}
      ${this.renderSuites()}
      ${this.renderAb()}
    `;
  }

  private renderPromotion(s: Stats): TemplateResult {
    const modeOf = (profile: string | null, d: string) => this.tiles.find((t) => t.name === profile)?.decisions[d as keyof ProfileTile['decisions']] ?? '—';
    // promotion is about the shadow decisions of a task; its acts have nothing to promote
    const tasks = (s.monitor?.precision ?? []).filter((t) => t.mode === 'shadow');
    return panel(
      'Shadow precision and promotion',
      html`<p class="muted small">${PROMOTE_RULE}. Precision is the share of labelled decisions a person marked right in the drawer; label them from the Overview feed or a session.</p>
        ${s.precision.length || tasks.length
          ? html`<div class="table-scroll"><table class="table" data-precision>
              <thead><tr><th>profile</th><th>decision</th><th>mode now</th><th class="num-col">decisions</th><th class="num-col">labelled</th><th class="num-col">precision</th><th>rule</th><th></th></tr></thead>
              <tbody>
                ${s.precision.map((p) => {
                  const pr = promotion(p);
                  const mode = modeOf(p.profile, p.decision);
                  return html`<tr data-precision-row=${`${p.profile}/${p.decision}`}>
                    <td>${p.profile}</td><td class="mono">${p.decision}</td><td><span class="badge ${mode === 'on' ? 'badge-on' : mode === 'shadow' ? 'badge-shadow' : ''}">${mode}</span></td>
                    <td class="num num-col">${exact(p.decisions)}</td><td class="num num-col">${exact(p.labelled)}</td><td class="num num-col">${p.precision == null ? '—' : p.precision.toFixed(2)}</td>
                    <td class="small ${pr.allowed ? '' : 'muted'}">${pr.why}</td>
                    <td><button class="btn btn-sm ${pr.allowed ? 'btn-primary' : ''}" ?disabled=${!pr.allowed || mode !== 'shadow'} title=${pr.allowed ? '' : PROMOTE_RULE} data-promote=${`${p.profile}/${p.decision}`} @click=${() => this.promote(p.profile ?? '', 'decision', p.decision, p.precision ?? 0, p.labelled)}>Promote to on</button></td></tr>`;
                })}
                ${tasks.map((t) => {
                  const pr = promotion(t);
                  return html`<tr data-precision-row=${`task/${t.task}`}><td>—</td><td class="mono">${t.task}</td><td><span class="badge ${t.mode === 'act' ? 'badge-on' : 'badge-shadow'}">${t.mode}</span></td>
                    <td class="num num-col">${exact(t.decisions)}</td><td class="num num-col">${exact(t.labelled)}</td><td class="num num-col">${t.precision == null ? '—' : t.precision.toFixed(2)}</td>
                    <td class="small ${pr.allowed ? '' : 'muted'}">${pr.why}</td>
                    <td><button class="btn btn-sm ${pr.allowed ? 'btn-primary' : ''}" ?disabled=${!pr.allowed || t.mode !== 'shadow'} data-promote=${`task/${t.task}`} @click=${() => this.promote(this.tiles.find((x) => !x.locked && x.name === 'coding')?.name ?? 'coding', 'task', t.task, t.precision ?? 0, t.labelled)}>Promote to act</button></td></tr>`;
                })}
              </tbody></table></div>`
          : emptyState('Nothing in shadow this week', 'Set a decision type to shadow in a profile; its decisions show up here with their labels.')}
        ${s.monitor ? nothing : html`<p class="muted small">The session monitor's tasks (D1, E1, C3…) appear here once its journal runs on this server (SRV-115).</p>`}`,
      { id: 'cond-precision' },
    );
  }

  private renderClassifier(): TemplateResult {
    const rep = this.reports.find((r) => r.kind === 'classifier');
    if (!rep) {
      return panel('Classifier accuracy', emptyState('No report yet', 'tools/conductor_eval.py runs the labelled set against each stage and writes the report the box keeps under the conductor directory (evals/).'), { id: 'cond-classifier' });
    }
    const stages = (rep.stages ?? {}) as Record<string, StageReport>;
    const set = (rep.set ?? {}) as { name?: string; n?: number; classes?: string[] };
    const cur = stages[this.stage] ?? Object.values(stages)[0];
    const cm = cur?.confusion ?? {};
    const labels = set.classes ?? Object.keys(cm);
    return panel(
      html`Classifier accuracy <span class="muted small num">${set.name ?? ''} · ${set.n ?? '?'} prompts · ${rep.created_at ? dateTimeShort(String(rep.created_at)) : ''}</span>`,
      html`<div class="table-scroll"><table class="table" data-classifier-report>
          <thead><tr><th>stage</th><th class="num-col">accuracy</th><th class="num-col">answered</th><th class="num-col">accuracy when it answered</th><th class="num-col">p50</th><th class="num-col">p95</th></tr></thead>
          <tbody>${Object.entries(stages).map(
            ([k, v]) => html`<tr class="req-row ${k === this.stage ? 'is-open' : ''}" @click=${() => (this.stage = k)}><td class="mono">${k}</td><td class="num num-col">${pct(v.accuracy, 1)}</td><td class="num num-col">${v.answered} / ${v.n}</td><td class="num num-col">${v.accuracy_answered == null ? '—' : pct(v.accuracy_answered, 1)}</td><td class="num num-col">${ms(v.p50_ms)}</td><td class="num num-col">${ms(v.p95_ms)}</td></tr>`,
          )}</tbody></table></div>
        ${labels.length && Object.keys(cm).length
          ? html`<p class="muted small" style="margin-top:12px">Confusion of <strong>${this.stage}</strong>: rows are the true class, columns what the stage answered (— = no answer).</p>
            <div class="table-scroll"><table class="table cm">
              <thead><tr><th></th>${[...labels, '—'].map((l) => html`<th class="mono small">${l}</th>`)}</tr></thead>
              <tbody>${labels.map(
                (t) => html`<tr><th>${clsChip(t, this.ids)}</th>${[...labels, '—'].map((p) => {
                  const v = cm[t]?.[p] ?? 0;
                  return html`<td class="cm-cell ${t === p ? 'cm-diag' : v ? 'cm-miss' : ''}">${v || ''}</td>`;
                })}</tr>`,
              )}</tbody></table></div>`
          : html`<p class="muted small" style="margin-top:12px">No confusion table for <strong>${this.stage}</strong>; pick another stage.</p>`}`,
      { id: 'cond-classifier' },
    );
  }

  private renderSuites(): TemplateResult {
    const suite = this.reports.filter((r) => r.kind === 'suite');
    const rows: SuiteRow[] = suite.flatMap((r) => ((r.rows ?? []) as SuiteRow[]));
    const guard = speedGuard(this.reports, this.engine);
    const pairs = suitePairs(rows);
    return panel(
      'TandemLLM against the raw engine',
      html`${guard ? html`<div class="guard-line" data-guard-detail>${icon('warn')} ${guard}</div>` : html`<p class="small">${icon('check')} the never-worse gate ran for build ${this.engine.version}.</p>`}
        ${pairs.length
          ? html`<div class="table-scroll" style="margin-top:12px"><table class="table" data-suites>
              <thead><tr><th>suite</th><th>arm</th><th class="num-col">gates passed</th><th class="num-col">minutes to PASS</th><th class="num-col">tokens</th><th class="num-col">interventions</th><th>build · night</th><th class="num-col">n</th></tr></thead>
              <tbody>${pairs.flatMap((p) =>
                [p.raw, p.tandem].filter(Boolean).map(
                  (r, i) => html`<tr data-suite=${p.suite}>
                    <td>${i === 0 ? p.suite : ''}${i === 0 && !p.cmp.ok ? html` <span class="badge badge-warn" data-not-comparable>not comparable</span><div class="warn-ink small">${p.cmp.why}</div>` : nothing}</td>
                    <td class="mono">${r!.arm}</td><td class="num num-col">${r!.gates_passed ?? '—'}${r!.gates_total ? ` / ${r!.gates_total}` : ''}</td>
                    <td class="num num-col">${r!.minutes_to_pass ?? '—'}</td><td class="num num-col">${r!.tokens != null ? exact(r!.tokens) : '—'}</td>
                    <td class="num num-col">${r!.interventions ?? '—'}</td><td class="small">${short(r!.build, 16)} · ${r!.night ?? '—'}</td><td class="num num-col">${r!.n ?? '—'}</td></tr>`,
                ),
              )}</tbody></table></div>
              <p class="muted small">Better means a pass rate not lower and minutes to PASS not higher on every suite, with the raw row unchanged; n under 10 is indicative only.</p>`
          : emptyState('No suite runs yet', 'OPS-43 (tools/tandem_eval.py) writes both arms per suite into the box\'s evals/ directory.')}`,
      { id: 'cond-suites' },
    );
  }

  private renderAb(): TemplateResult {
    return panel(
      'A/B of two profile versions',
      html`<div class="toolbar" style="margin-top:0">
          <label class="field"><span>profile</span><select @change=${(e: Event) => {
            this.abProfile = (e.target as HTMLSelectElement).value;
            void this.loadAb();
          }}>${this.tiles.filter((t) => !t.locked).map((t) => html`<option value=${t.name} ?selected=${t.name === this.abProfile}>${t.name}</option>`)}</select></label>
          ${(['abA', 'abB'] as const).map(
            (k) => html`<label class="field"><span>${k === 'abA' ? 'A' : 'B'}</span><select @change=${(e: Event) => {
              this[k] = Number((e.target as HTMLSelectElement).value);
              void this.runAb();
            }}>${this.abVersions.map((v) => html`<option value=${v.version} ?selected=${v.version === this[k]}>v${v.version} ${short(v.note, 30)}</option>`)}</select></label>`,
          )}
        </div>
        <p class="small num">decisions in the last 500: A ${this.abCounts[this.abA ?? -1] ?? 0} · B ${this.abCounts[this.abB ?? -1] ?? 0}</p>
        ${this.abChanges ? (this.abChanges.length ? html`<div class="diff" data-ab>${this.abChanges.map((c) => html`<div class="diff-row"><span>${readablePath(c.path)}</span><span class="diff-before">${short(c.before, 80)}</span><span class="diff-after">${short(c.after, 80)}</span></div>`)}</div>` : html`<p class="muted small">The two versions are the same.</p>`) : nothing}`,
      { id: 'cond-ab' },
    );
  }
}
