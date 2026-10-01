// Conductor · Sessions: every session the conductor saw (key, client, profile, its class now,
// turns, notes, the monitor's acts and gave-ups, the interventions, the last checker result), and
// one session's timeline: a strip of turns, width by time, colour by class, a filled marker below
// where the router put a note in (hollow: shadow) and above where the session monitor acted
// (hollow: shadow); the checker numbers per gate across turns (a stalled gate is a flat line);
// the turns as a feed with the drawer; the monitor's journal of the session; and the export, JSON
// or Markdown: every hidden text the model saw that the client's transcript does not hold, the
// router's and the monitor's in one record.

import { html, nothing, type TemplateResult } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import { describeError } from '../../api/client';
import { conductor, type DecisionRow, type LiveSession, type MonitorEvent, type SessionRow } from '../../api/conductor';
import { classColor, didWhat, gateSeries, isAct, short } from '../../lib/conductor';
import { ago, dateTimeShort, exact } from '../../lib/format';
import { poll, type Poller } from '../../lib/poll';
import { setParams } from '../../lib/router';
import { LightElement } from '../../ui/base';
import { emptyState, icon, panel, skeleton } from '../../ui/bits';
import { clsChip, feedRow } from './shared';

@customElement('qse-cond-sessions')
export class QseCondSessions extends LightElement {
  @property({ attribute: false }) ids: string[] = [];
  @property({ attribute: false }) selected: string | null = null;
  @state() private list: SessionRow[] | null = null;
  @state() private live: LiveSession[] = [];
  @state() private turns: DecisionRow[] | null = null;
  @state() private events: MonitorEvent[] = [];
  @state() private error: string | null = null;
  @state() private query = '';
  @state() private client = '';
  @state() private open: DecisionRow | null = null;
  private listPoll: Poller | null = null;
  private loadedFor: string | null = null;

  connectedCallback(): void {
    super.connectedCallback();
    this.listPoll = poll(() => this.loadList(), 10_000);
    this.listPoll.start();
  }
  disconnectedCallback(): void {
    this.listPoll?.stop();
    super.disconnectedCallback();
  }

  protected willUpdate(): void {
    if (this.selected !== this.loadedFor) {
      this.loadedFor = this.selected;
      this.turns = null;
      if (this.selected) void this.loadSession(this.selected);
    }
  }

  private async loadList() {
    try {
      const r = await conductor.sessions();
      this.list = r.sessions;
      this.live = r.live;
      this.error = null;
      if (this.selected) void this.loadSession(this.selected);
    } catch (e) {
      const d = describeError(e);
      this.error = `${d.endpoint}: ${d.message}`;
    }
  }

  private async loadSession(key: string) {
    try {
      const r = await conductor.session(key);
      if (key === this.selected) {
        this.turns = r.turns;
        this.events = r.events ?? [];
      }
    } catch (e) {
      if (key === this.selected) this.turns = [];
      this.error = describeError(e).message;
    }
  }

  render() {
    if (this.error && !this.list) return html`<div class="state state-error" role="alert"><div class="state-title">${icon('warn')} ${this.error}</div></div>`;
    if (!this.list) return skeleton(6);
    const clients = [...new Set(this.list.map((s) => s.client_kind ?? 'other'))];
    const rows = this.list.filter((s) => (!this.query || s.key.toLowerCase().includes(this.query.toLowerCase())) && (!this.client || s.client_kind === this.client));
    return html`
      ${this.selected ? this.renderSession(this.selected) : nothing}
      ${panel(
        'Sessions, last 7 days',
        rows.length
          ? html`<div class="table-scroll"><table class="table" data-sessions>
              <thead><tr><th>session</th><th>client</th><th>profile</th><th>class now</th><th class="num-col">turns</th><th class="num-col">notes</th><th class="num-col">monitor acts</th><th class="num-col">interventions</th><th>last checker</th><th>last turn</th></tr></thead>
              <tbody>${rows.map(
                (s) => html`<tr class="req-row ${s.key === this.selected ? 'is-open' : ''}" data-session=${s.key} @click=${() => setParams({ session: s.key })}>
                  <td class="mono small cell-wrap">${short(s.key, 28)}</td><td>${s.client_kind ?? '—'}</td><td>${s.profile ?? '—'}</td>
                  <td>${clsChip(s.class_id, this.ids)}</td><td class="num num-col">${exact(s.turns)}</td>
                  <td class="num num-col">${s.notes ? `${s.notes} (${exact(s.note_tokens)} tok)` : '—'}</td>
                  <td class="num num-col">${s.acts ? html`${exact(s.acts)}${s.gave_ups ? html` <span class="badge badge-warn">${s.gave_ups} gave up</span>` : nothing}` : '—'}</td>
                  <td class="num num-col" data-interventions>${exact(s.interventions ?? 0)}</td>
                  <td>${s.last_checker ? html`<span class="badge ${s.last_checker === 'PASS' ? 'badge-ok' : 'badge-warn'}">${s.last_checker}</span>` : '—'}</td>
                  <td class="num muted">${ago(new Date(s.last_ms).toISOString())}</td></tr>`,
              )}</tbody></table></div>`
          : emptyState('No sessions', 'Every TandemLLM request belongs to one: opencode sends its session id, other clients are grouped by their first two messages.'),
        {
          tools: html`<label class="field field-search">${icon('search')}<input placeholder="search a key" .value=${this.query} @input=${(e: Event) => (this.query = (e.target as HTMLInputElement).value)} aria-label="search sessions" /></label>
            <label class="field"><span>client</span><select @change=${(e: Event) => (this.client = (e.target as HTMLSelectElement).value)}><option value="">all</option>${clients.map((c) => html`<option value=${c}>${c}</option>`)}</select></label>`,
          id: 'cond-sessions',
        },
      )}
      ${this.open ? html`<qse-cond-decision .row=${this.open} .ids=${this.ids} .close=${() => (this.open = null)}></qse-cond-decision>` : nothing}
    `;
  }

  private renderSession(key: string): TemplateResult {
    const turns = this.turns;
    const live = this.live.find((s) => s.key === key);
    if (turns == null) return skeleton(4);
    const total = turns.reduce((s, t) => s + (t.outcome?.total_ms ?? 1000), 0) || 1;
    const notes = turns.reduce((s, t) => s + (t.injected?.length ?? 0), 0);
    const acts = this.events.filter((e) => isAct(e)).length;
    const g = gateSeries(turns);
    return html`<section class="panel" style="margin-bottom:16px" data-timeline=${key}>
      <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">${short(key, 44)} ${live ? html`<span class="badge badge-on">live</span>` : nothing}</h2>
        <p class="panel-sub">${turns.length} turns · ${notes} hidden note${notes === 1 ? '' : 's'} · ${acts} monitor act${acts === 1 ? '' : 's'}${live ? ` · class now ${live.class_id} · ${exact(live.note_tokens)} note tokens replayed a turn` : ''}${turns.length ? ` · since ${dateTimeShort(new Date(turns[0].ts_ms).toISOString())}` : ''}</p></div>
        <div class="panel-tools">
          <a class="btn btn-sm" href=${conductor.exportUrl(key, 'md')} download=${`session-${key.replace(/[^a-z0-9]+/gi, '-')}.md`} data-export="md">${icon('down')} Markdown</a>
          <a class="btn btn-sm" href=${conductor.exportUrl(key, 'json')} download=${`session-${key.replace(/[^a-z0-9]+/gi, '-')}.json`} data-export="json">${icon('down')} JSON</a>
          <button class="btn btn-ghost btn-icon" aria-label="Close the session" @click=${() => setParams({ session: null })}>${icon('x')}</button></div></header>
      <div class="strip" role="list" aria-label="turns">
        ${turns.map((t) => {
          const w = Math.max(18, Math.round(((t.outcome?.total_ms ?? 1000) / total) * 900));
          const acted = (t.injected?.length ?? 0) > 0;
          const shadow = !!t.shadow;
          const evs = (t.events ?? []).filter((e) => e.decision !== 'PASS');
          const mon = evs.some((e) => isAct(e)) ? 'act' : evs.length ? 'shadow' : null;
          return html`<button class="strip-turn ${this.open?.id === t.id ? 'is-sel' : ''}" role="listitem" style="width:${w}px; --cls: var(--series-${classColor(t.class_id, this.ids)})" title=${`${t.class_id} · ${t.outcome?.finish_reason ?? 'running'} · ${didWhat(t)}`} @click=${() => (this.open = t)} data-turn=${t.id}>
            ${acted ? html`<span class="mark"></span>` : shadow ? html`<span class="mark is-shadow"></span>` : nothing}
            ${mon ? html`<span class="mark is-monitor ${mon === 'shadow' ? 'is-shadow' : ''}" data-monitor-mark=${mon}></span>` : nothing}
          </button>`;
        })}
      </div>
      <p class="muted small">Width is the turn's time, colour its class. Below: ◆ the router put a hidden note in, ◇ shadow would have. Above, orange: the session monitor acted (◆) or would have (◇).</p>
      ${g.series.length ? html`<qse-ribbon .series=${g.series} .labels=${g.labels} .height=${160} aria-label="checker gate values per turn"></qse-ribbon>` : nothing}
      <div class="feed" style="margin-top:10px">${turns
        .slice()
        .reverse()
        .map((t) => feedRow(t, this.ids, (x) => (this.open = x), didWhat(t) + ((t.events ?? []).some((e) => isAct(e)) ? ' · monitor acted' : '')))}</div>
      ${this.events.length ? html`<div style="margin-top:12px"><qse-cond-journal .session=${key}></qse-cond-journal></div>` : nothing}
    </section>`;
  }
}
