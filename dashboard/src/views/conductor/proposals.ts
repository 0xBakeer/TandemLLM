// Conductor · Packages · Proposals (2026-10-05, P-M3): the changes to packages and skills the model
// proposed with its `<tandem_memory op="propose">` block (server/conductor/proposals.py). Nothing
// the model proposes goes live by itself: each proposal shows its reason, its evidence (the session's
// J1 outcome rows that showed a gap, or "no evidence"), and the diff against the live text; Apply
// stores the text as a new version of the packages document (the package becomes inline; the old
// version stays and the versions panel restores it), Reject keeps the proposal with its reason. A
// replacement of a text that changed since it was proposed cannot be applied.

import { html, nothing, type TemplateResult } from 'lit';
import { customElement, state } from 'lit/decorators.js';
import { describeError } from '../../api/client';
import { conductor, type Proposal, type ProposalView } from '../../api/conductor';
import { dateTimeShort } from '../../lib/format';
import { lineDiff } from '../../lib/notes';
import { LightElement } from '../../ui/base';
import { emptyState, icon, panel, skeleton } from '../../ui/bits';
import { confirmDialog } from './shared';

const when = (iso: string | null | undefined) => (iso ? dateTimeShort(iso) : '—');

@customElement('qse-cond-proposals')
export class QseCondProposals extends LightElement {
  /** called after an Apply, so the Packages tab reloads the document */
  onChanged: (() => void) | null = null;
  @state() private list: Proposal[] | null = null;
  @state() private open: ProposalView | null = null;
  @state() private error: string | null = null;
  @state() private flash: string | null = null;
  @state() private busy = false;
  @state() private status: 'open' | 'all' = 'open';

  connectedCallback(): void {
    super.connectedCallback();
    void this.load();
  }

  private async load() {
    try {
      this.list = (await conductor.proposals()).proposals;
      this.error = null;
    } catch (e) {
      this.error = describeError(e).message;
    }
  }

  private async show(p: Proposal) {
    try {
      this.open = (await conductor.proposal(p.name, p.n)).proposal;
    } catch (e) {
      this.error = describeError(e).message;
    }
  }

  private async apply(p: ProposalView) {
    const ok = await confirmDialog(`Apply the proposal to ${p.name}?`, [p.title, 'The text becomes a new version of the packages document; every session gets it from its next note. The versions panel restores the old text.'], 'Apply');
    if (!ok) return;
    this.busy = true;
    try {
      const r = await conductor.proposalApply(p.name, p.n);
      this.flash = `${p.name}: packages version ${r.version} (was ${r.previous})`;
      this.onChanged?.();
      await this.load();
      await this.show(p);
    } catch (e) {
      this.error = describeError(e).message;
    } finally {
      this.busy = false;
    }
  }

  private async reject(p: ProposalView) {
    const reason = window.prompt('Why is this proposal rejected?', 'not what the skill needs');
    if (reason === null) return;
    this.busy = true;
    try {
      await conductor.proposalReject(p.name, p.n, reason || 'rejected in the dashboard');
      this.flash = `${p.name}#${p.n} rejected`;
      await this.load();
      await this.show(p);
    } catch (e) {
      this.error = describeError(e).message;
    } finally {
      this.busy = false;
    }
  }

  render(): TemplateResult {
    if (this.error && !this.list) return html`<div class="state state-error" role="alert"><div class="state-title">${icon('warn')} ${this.error}</div></div>`;
    if (!this.list) return skeleton(4);
    const rows = this.list.filter((p) => this.status === 'all' || p.status === 'open');
    const badge = (s: string) => html`<span class="badge ${s === 'open' ? 'badge-shadow' : s === 'applied' ? 'badge-ok' : ''}">${s}</span>`;
    return html`
      ${this.flash ? html`<p class="ok-ink small" role="status" data-prop-flash>${icon('check')} ${this.flash}</p>` : nothing}
      ${this.error ? html`<p class="err-ink small" role="alert">${icon('warn')} ${this.error}</p>` : nothing}
      ${this.open ? this.renderOne(this.open) : nothing}
      ${panel(
        'Proposals',
        rows.length
          ? html`<div class="table-scroll"><table class="table" data-proposals><thead><tr><th>package</th><th>change</th><th>mode</th><th>status</th><th>evidence</th><th>proposed</th></tr></thead><tbody>${rows.map(
              (p) => html`<tr class="req-row ${this.open && this.open.name === p.name && this.open.n === p.n ? 'is-open' : ''}" data-proposal=${`${p.name}#${p.n}`} @click=${() => this.show(p)}>
                <td class="mono small">${p.name}#${p.n}</td><td class="cell-wrap">${p.title}</td><td class="small">${p.mode}</td><td>${badge(p.status)}</td>
                <td class="small">${p.evidence.length ? `${p.evidence.length} outcome row${p.evidence.length === 1 ? '' : 's'}` : html`<span class="muted">no evidence</span>`}</td><td class="num small">${when(p.created)}</td></tr>`,
            )}</tbody></table></div>`
          : emptyState(this.status === 'open' ? 'No open proposal' : 'No proposal yet', 'The model proposes a change to a package with a hidden block when a session showed a gap; it waits here for your click.'),
        {
          sub: 'changes to packages and skills the model proposed; none goes live without Apply',
          tools: html`<label class="switch small"><input type="checkbox" .checked=${this.status === 'all'} @change=${(e: Event) => (this.status = (e.target as HTMLInputElement).checked ? 'all' : 'open')} data-prop-all /> applied and rejected too</label>
            <button class="btn btn-sm" @click=${() => this.load()} aria-label="reload the proposals">${icon('refresh')}</button>`,
          id: 'proposals',
        },
      )}`;
  }

  private renderOne(p: ProposalView): TemplateResult {
    const ops = p.live !== null ? lineDiff(p.live, p.proposed) : [];
    return html`<section class="panel" style="margin-bottom:16px" data-proposal-open=${`${p.name}#${p.n}`}>
      <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">${p.name}#${p.n}: ${p.title}</h2>
        <p class="panel-sub">${p.mode === 'append' ? 'adds a section' : 'replaces the text'} · ${p.status}${p.applied_version ? ` as version ${p.applied_version}` : ''}${p.rejected_reason ? `: ${p.rejected_reason}` : ''} · ${p.evidence_word}${p.session ? html` · <a class="mono" href=${`#/conductor?screen=memory&memory=${encodeURIComponent(p.session)}`}>${p.session}</a>` : nothing}</p></div>
        <div class="panel-tools">
          ${p.status === 'open' ? html`<button class="btn btn-primary btn-sm" ?disabled=${!p.applies || this.busy} @click=${() => this.apply(p)} data-prop-apply>Apply</button>
            <button class="btn btn-sm" ?disabled=${this.busy} @click=${() => this.reject(p)} data-prop-reject>Reject</button>` : nothing}
          <button class="btn btn-ghost btn-icon" aria-label="Close the proposal" @click=${() => (this.open = null)}>${icon('x')}</button></div></header>
      ${p.reason ? html`<p class="small"><span class="muted">why:</span> ${p.reason}</p>` : nothing}
      ${p.stale && p.status === 'open' ? html`<p class="warn-ink small">${icon('warn')} ${p.name} changed since this was proposed${p.mode === 'append' ? '; the section is added to the text as it is now' : '; a replacement cannot be applied'}.</p>` : nothing}
      ${p.live === null ? html`<p class="warn-ink small">No package ${p.name} with a text any more.</p>` : html`<pre class="notes-diff" data-prop-diff>${ops.map((o) => html`<span class="diff-${o.op === '+' ? 'add' : o.op === '-' ? 'del' : 'eq'}">${o.op === '=' ? ' ' : o.op} ${o.text}\n</span>`)}</pre>`}
    </section>`;
  }
}
