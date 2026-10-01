// Conductor · Overview: four tiles (decisions today and how many acted or ran in shadow, the
// session monitor's acts and gave-ups, the classifier's latency with the fail-open rate, roles
// online), the 24 h ribbon of decisions an hour by class, the live decision feed (every 3 s) with
// the drawer, the monitor's acts an hour by task family and its last 20 actions (each opening the
// event, and its session), the class mix, and which profile each client gets.

import { html, nothing } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import { describeError } from '../../api/client';
import { conductor, type DecisionRow, type MonitorEvent, type ProfileTile, type Stats } from '../../api/conductor';
import { didWhat, eventWhat, hourlySeries, monitorSeries } from '../../lib/conductor';
import { exact, ms, pct, timeShort } from '../../lib/format';
import { poll, type Poller } from '../../lib/poll';
import { setParams } from '../../lib/router';
import { LightElement } from '../../ui/base';
import { emptyState, icon, panel, skeleton, statTile } from '../../ui/bits';
import { clsChip, feedRow } from './shared';

@customElement('qse-cond-overview')
export class QseCondOverview extends LightElement {
  @property({ attribute: false }) ids: string[] = [];
  @property({ attribute: false }) tiles: ProfileTile[] = [];
  @property({ attribute: false }) active = 'chat';
  @state() private stats: Stats | null = null;
  @state() private feed: DecisionRow[] = [];
  @state() private error: string | null = null;
  @state() private open: DecisionRow | null = null;
  @state() private openEvent: MonitorEvent | null = null;
  @state() private paused = false;
  private statsPoll: Poller | null = null;
  private feedPoll: Poller | null = null;

  connectedCallback(): void {
    super.connectedCallback();
    this.statsPoll = poll(() => this.loadStats(), 20_000);
    this.feedPoll = poll(() => this.loadFeed(), 3_000);
    this.statsPoll.start();
    this.feedPoll.start();
  }
  disconnectedCallback(): void {
    this.statsPoll?.stop();
    this.feedPoll?.stop();
    super.disconnectedCallback();
  }

  private async loadStats() {
    try {
      this.stats = await conductor.stats(24);
      this.error = null;
    } catch (e) {
      const d = describeError(e);
      this.error = `${d.endpoint}: ${d.message}`;
    }
  }
  private async loadFeed() {
    if (this.paused) return;
    try {
      this.feed = (await conductor.decisions({ limit: 20 })).decisions;
    } catch {
      /* the stats panel says what failed */
    }
  }

  private async openAction(id: string) {
    try {
      this.openEvent = (await conductor.event(id)).event;
    } catch (e) {
      this.error = describeError(e).message;
    }
  }

  private renderMonitor(m: NonNullable<Stats['monitor']>) {
    const series = monitorSeries(m);
    return html`<div class="grid-2 grid-2-wide">
      ${panel(
        'Monitor acts an hour, last 24 h',
        m.acts
          ? html`<qse-ribbon .series=${series} .labels=${m.hourly.map((h) => new Date(h.start_ms).toISOString().slice(11, 13) + ':00')} unit="/h" .height=${180} aria-label="the session monitor's acts per hour by task family"></qse-ribbon>`
          : emptyState('No act in 24 hours', 'Shadow decisions and passes are counted, not drawn here.'),
        { sub: 'one colour per task family: thinking, end of turn, tool calls, session', id: 'cond-monitor-ribbon' },
      )}
      ${panel(
        'Last actions of the monitor',
        m.last_actions.length
          ? html`<div class="feed" data-last-actions>${m.last_actions.map(
              (a) => html`<button class="feed-row" data-action=${a.id} @click=${() => this.openAction(a.id)}>
                <span class="feed-time">${timeShort(new Date(a.ts_ms).toISOString())}</span>
                <span class="feed-cls"><span class="badge mono">${a.task ?? '—'}</span></span>
                <span class="feed-conf muted small">${a.outcome?.kind ?? 'running'}</span>
                <span class="feed-what">${eventWhat({ task: a.task, decision: a.decision, mode: 'act', trigger: a.trigger, cap: a.cap })}</span>
                <span class="feed-act">${a.session_key ? html`<a class="btn btn-ghost btn-sm" href=${`#/conductor?screen=sessions&session=${encodeURIComponent(a.session_key)}`} @click=${(e: Event) => e.stopPropagation()}>session</a>` : nothing}</span>
              </button>`,
            )}</div>`
          : html`<div class="feed-empty">No act yet. The Monitor screen's journal shows every decision, shadow and passes of the tool checks included.</div>`,
        { id: 'cond-last-actions', tools: html`<a class="btn btn-ghost btn-sm" href="#/conductor?screen=monitor">journal ${icon('external')}</a>` },
      )}
    </div>`;
  }

  render() {
    const s = this.stats;
    if (this.error && !s) return html`<div class="state state-error" role="alert"><div class="state-title">${icon('warn')} ${this.error}</div></div>`;
    if (!s) return skeleton(6);
    const clf = s.classifier;
    const arb = clf.decider;
    const qwen = clf.qwen;
    const lead = arb?.calls ? arb : qwen?.calls ? qwen : arb;
    const roles = Object.entries(s.roles);
    const online = roles.filter(([, r]) => r.up).length;
    const monitor = s.monitor;
    return html`
      <div class="stats" data-tiles>
        ${statTile({ label: 'decisions, 24 h', value: exact(s.decisions), sub: html`<span class="num">${exact(s.acted)}</span> acted · <span class="num">${exact(s.shadow)}</span> in shadow`, title: `${s.injections} notes, ${s.injected_tokens} tokens injected` })}
        ${statTile({ label: 'monitor, 24 h', value: monitor ? exact(monitor.acts) : '—', sub: monitor ? html`acts · <span class="num">${exact(monitor.shadow)}</span> in shadow · <span class="num" data-gave-up>${exact(monitor.gave_up)}</span> gave up` : 'the session monitor does not run on this server (--conductor raw)', na: !monitor, warn: (monitor?.gave_up ?? 0) > 0, title: monitor ? `${exact(monitor.hidden_tokens)} hidden tokens` : undefined })}
        ${statTile({
          // the first stage that was asked leads: the decider, or Qwen when the decider is off
          label: `classifier p50 / p95${!arb?.calls && qwen?.calls ? ' (Qwen)' : ''}`,
          value: lead?.p50_ms != null ? `${Math.round(lead.p50_ms)} / ${Math.round(lead.p95_ms ?? 0)}` : '—',
          unit: lead?.p50_ms != null ? 'ms' : undefined,
          sub: html`decider ${exact(arb?.calls ?? 0)} calls${qwen ? html` · Qwen ${exact(qwen.calls)} (${ms(qwen.p50_ms)})` : nothing} · fail-open ${pct(s.fail_open_rate, 1)}`,
          warn: (s.fail_open_rate ?? 0) > 0.1,
        })}
        ${statTile({
          label: 'roles online',
          value: `${online} / ${roles.length}`,
          sub: html`${roles.map(([n, r]) => html`<span title=${r.error ?? ''} class=${r.up ? '' : r.mode === 'off' ? 'muted' : 'err-ink'} data-role=${n}>${r.up ? '●' : '○'} ${n} </span>`)}`,
          warn: roles.some(([, r]) => !r.up && r.mode !== 'off'),
        })}
      </div>
      <div class="grid-2 grid-2-wide">
        ${panel(
          'Decisions an hour, last 24 h',
          s.decisions
            ? html`<qse-ribbon .series=${hourlySeries(s, this.ids)} .labels=${s.hourly.map((h) => new Date(h.start_ms).toISOString().slice(11, 13) + ':00')} unit="/h" .height=${200} aria-label="conductor decisions per hour by task class"></qse-ribbon>`
            : emptyState('No decisions yet', 'Send a request to the model TandemLLM; the raw engine is never routed.'),
          { sub: 'one colour per task class; tool-result turns count too (they keep the class)', id: 'cond-ribbon' },
        )}
        ${panel(
          'Task classes, 24 h',
          html`<div class="table-scroll"><table class="table" data-class-mix>
            <thead><tr><th>class</th><th class="num-col">decisions</th><th class="num-col">labelled right</th></tr></thead>
            <tbody>${s.classes.map((c) => html`<tr><td>${clsChip(c.class_id, this.ids)}</td><td class="num num-col">${exact(c.decisions)}</td><td class="num num-col">${c.labelled ? `${c.agree} / ${c.labelled}` : '—'}</td></tr>`)}</tbody>
          </table></div>
          <p class="panel-foot">By source: ${Object.entries(s.by_source).map(([k, v]) => `${k} ${v}`).join(' · ') || '—'}. By effort: ${Object.entries(s.by_effort).map(([k, v]) => `${k} ${v}`).join(' · ') || '—'}.</p>`,
          { id: 'cond-classes' },
        )}
      </div>
      <div class="grid-2 grid-2-wide">
        ${panel(
          html`Live decisions ${this.paused ? html`<span class="badge">paused</span>` : html`<span class="badge badge-on">every 3 s</span>`}`,
          this.feed.length
            ? html`<div class="feed" data-feed>${this.feed.map((r) => feedRow(r, this.ids, (x) => (this.open = x), didWhat(r)))}</div>`
            : html`<div class="feed-empty">No decisions yet. Every TandemLLM request shows up here: the class, how sure, who decided, what was set and injected.</div>`,
          {
            id: 'cond-feed',
            tools: html`<button class="btn btn-ghost btn-sm" @click=${() => (this.paused = !this.paused)}>${icon(this.paused ? 'play' : 'pause')} ${this.paused ? 'resume' : 'pause'}</button>
              <a class="btn btn-ghost btn-sm" href="#/conductor?screen=sessions">sessions ${icon('external')}</a>`,
          },
        )}
        ${panel(
          'Profiles by client',
          html`<div class="chips" data-client-chips>
              ${this.tiles.flatMap((t) =>
                t.clients.map((c) => html`<button class="pick" @click=${() => setParams({ screen: 'profiles', edit: t.name })}><span class="muted">${c}</span> → ${t.name} <span class="badge ${t.mode === 'on' ? 'badge-on' : t.mode === 'shadow' ? 'badge-shadow' : ''}">${t.mode}</span></button>`),
              )}
            </div>
            <p class="panel-foot">Other clients get the default profile. A request picks a profile with <code>X-Tandem-Profile</code>; <code>X-Tandem: raw</code> (or the locked <code>bench</code> / <code>raw</code> profile) always serves the engine untouched. Every model name, the old raw one included, is TandemLLM.</p>
            <dl class="kv">${this.tiles.map((t) => html`<div><dt>${t.name}</dt><dd><span class="badge ${t.mode === 'on' ? 'badge-on' : t.mode === 'shadow' ? 'badge-shadow' : ''}">${t.mode}</span> <span class="muted small">v${t.version}${t.locked ? ' · locked' : ''}</span></dd></div>`)}</dl>`,
          { id: 'cond-clients' },
        )}
      </div>
      ${monitor ? this.renderMonitor(monitor) : nothing}
      ${this.open ? html`<qse-cond-decision .row=${this.open} .ids=${this.ids} .close=${() => (this.open = null)}></qse-cond-decision>` : nothing}
      ${this.openEvent ? html`<qse-cond-event .ev=${this.openEvent} .close=${() => (this.openEvent = null)}></qse-cond-event>` : nothing}
    `;
  }
}

