// Conductor · Compare: whether a decision earned "on", and whether TandemLLM beats the raw engine.
//   * precision per profile and decision type in shadow (from the "right / wrong" labels), the
//     promotion rule in words, and "Promote to on", enabled only when the rule holds; the session
//     monitor's tasks the same way once its journal is on the server;
//   * the classifier report (tools/conductor_eval.py): accuracy per stage on the labelled set,
//     the confusion table, latency;
//   * the suite report (OPS-43, tools/tandem_eval.py): its verdict with the indicative label when
//     n < 10 and the raw row's status, then raw and TandemLLM per suite -- n, pass rate, gates,
//     minutes to PASS, wall minutes, tokens, decode tok/s, empty stops, the continuations the
//     watch-only driver would have sent, interventions, gave-ups, cached share -- "not comparable"
//     in orange unless both arms are from the same build and night; older nights in a picker; the
//     speed guard of the build that serves now;
//   * A/B of two versions of a profile: what changed and how many decisions each version made.

import { html, nothing, type TemplateResult } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import { describeError } from '../../api/client';
import { conductor, type Change, type DecisionRow, type EvalReport, type EvalVerdict, type ProfileDoc, type ProfileTile, type Stats, type SuiteComparison, type VersionRow } from '../../api/conductor';
import { armName, indicative, INDICATIVE_N, promotion, PROMOTE_RULE, readablePath, setIn, short, speedGuard, suitePairs, type SuiteRow } from '../../lib/conductor';
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
  @state() private suiteFile: string | null = null;

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
    // the newest suite report first (the box keeps them newest first); a picker for older nights
    const suite = this.reports.filter((r) => r.kind === 'suite');
    const rep = suite.find((r) => r.file === this.suiteFile) ?? suite[0];
    const rows: SuiteRow[] = ((rep?.rows ?? []) as SuiteRow[]);
    const guard = speedGuard(this.reports, this.engine);
    const pairs = suitePairs(rows);
    const v = rep?.verdict as EvalVerdict | undefined;
    const comp = (rep?.comparison ?? []) as SuiteComparison[];
    const rawRow = rep?.raw_row as { status: string; source: string | null } | undefined;
    const build = rep?.build as { version?: string | null; code_sha?: string | null } | undefined;
    const ind = v?.indicative ?? indicative(rows);
    const cell = (x: number | null | undefined, digits = 1) => (x == null ? '—' : Number.isInteger(x) ? exact(x) : x.toFixed(digits));
    return panel(
      html`TandemLLM against the raw engine ${rep ? html`<span class="muted small num">night ${rep.night ?? '—'}${build ? ` · ${build.version ?? '?'} ${short(build.code_sha, 12)}` : ''}${rep.profile ? ` · profile ${rep.profile}` : ''}</span>` : nothing}`,
      html`${guard ? html`<div class="guard-line" data-guard-detail>${icon('warn')} ${guard}</div>` : html`<p class="small">${icon('check')} the never-worse gate ran for build ${this.engine.version}.</p>`}
        ${suite.length > 1
          ? html`<label class="field"><span>report</span><select aria-label="suite report" data-suite-report @change=${(e: Event) => (this.suiteFile = (e.target as HTMLSelectElement).value)}>${suite.map((r) => html`<option value=${String(r.file)} ?selected=${r === rep}>${String(r.night ?? r.file)} · ${String(r.file)}</option>`)}</select></label>`
          : nothing}
        ${v
          ? html`<div class="note" data-eval-verdict>
              <span class="badge ${v.better ? 'badge-ok' : v.better === false ? 'badge-warn' : ''}"><strong>${v.better ? 'Better' : v.better === false ? 'Not better' : 'Not shown yet'}</strong></span>${ind ? html` <span class="badge badge-warn" data-indicative>indicative: n &lt; 10</span>` : nothing} — ${v.why}
              <div class="muted small" data-raw-row>raw never-worse row: ${rawRow?.status ?? '—'}${rawRow?.source ? ` (${rawRow.source})` : ''}${rep?.driver ? ` · driver ${(rep.driver as { mode: string }).mode}` : ''}${rep?.isolation ? ` · ${(rep.isolation as { store_clears: unknown[] }).store_clears.length} store clears between arms` : ''}</div>
            </div>`
          : nothing}
        ${pairs.length
          ? html`<div class="table-scroll" style="margin-top:12px"><table class="table" data-suites>
              <thead><tr><th>suite</th><th>arm</th><th class="num-col">n</th><th class="num-col">pass rate</th><th class="num-col">gates passed</th><th class="num-col">minutes to PASS</th><th class="num-col">wall min</th><th class="num-col">tokens</th><th class="num-col">decode tok/s</th><th class="num-col">empty stops</th><th class="num-col">continuations</th><th class="num-col">interventions</th><th class="num-col">gave up</th><th class="num-col">cached</th><th>build · night</th></tr></thead>
              <tbody>${pairs.flatMap((p) => {
                const c = comp.find((x) => x.suite === p.suite || x.suite_id === (p.raw ?? p.tandem)?.suite_id);
                return [p.raw, p.tandem].filter(Boolean).map(
                  (r, i) => html`<tr data-suite=${p.suite} data-arm=${r!.arm}>
                    <td>${i === 0 ? p.suite : ''}${i === 0 && !p.cmp.ok ? html` <span class="badge badge-warn" data-not-comparable>not comparable</span><div class="warn-ink small">${p.cmp.why}</div>` : nothing}${i === 0 && c ? html` <span class="badge ${c.verdict === 'better' ? 'badge-ok' : c.verdict === 'worse' || c.verdict === 'mixed' ? 'badge-warn' : ''}" data-suite-verdict=${c.verdict}>${c.verdict}</span>` : nothing}${i === 0 && (r!.indicative ?? (r!.n ?? 0) < INDICATIVE_N) ? html` <span class="badge" data-row-indicative>indicative</span>` : nothing}</td>
                    <td class="mono">${armName(r!.arm)}${r!.pristine === false ? html` <span class="badge badge-warn" title="a checker or reference file changed during a run">not pristine</span>` : nothing}</td>
                    <td class="num num-col">${r!.n ?? '—'}</td>
                    <td class="num num-col">${r!.pass_rate == null ? '—' : pct(r!.pass_rate, 0)}</td>
                    <td class="num num-col">${r!.gates_passed ?? '—'}${r!.gates_total ? ` / ${r!.gates_total}` : ''}</td>
                    <td class="num num-col">${cell(r!.minutes_to_pass)}</td><td class="num num-col">${cell(r!.wall_minutes)}</td>
                    <td class="num num-col">${r!.tokens != null ? exact(Math.round(r!.tokens)) : '—'}</td><td class="num num-col">${cell(r!.decode_tps)}</td>
                    <td class="num num-col">${r!.empty_stops ?? '—'}</td><td class="num num-col">${r!.continuations ?? '—'}</td>
                    <td class="num num-col">${r!.interventions ?? '—'}</td><td class="num num-col">${r!.gave_ups ?? '—'}</td>
                    <td class="num num-col">${r!.cache_hit == null ? '—' : pct(r!.cache_hit, 0)}</td>
                    <td class="small">${short(r!.build, 16)} · ${r!.night ?? '—'}</td></tr>`,
                );
              })}</tbody></table></div>
              <p class="muted small">Better means a pass rate not lower and minutes to PASS not higher on every suite, with the raw row unchanged; n under 10 is indicative only. Continuations are the ones the watch-only driver would have sent; interventions are the conductor's acts.</p>`
          : emptyState('No suite runs yet', 'OPS-43 (tools/tandem_eval.py) writes both arms per suite and keeps the report in the box\'s evals/ directory (--publish).')}`,
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
