// System — the box at a glance: engine, memory, GPU, queue, inflight counters, caches,
// ledger, disk, plus rolling 30-minute charts kept in the browser (the soak panel).
//
// OPS-41: the memory plan -- where the engine's resident set goes (one bar, each part read off the
// object that holds it, the load stages), and the residency planner beside it: MemAvailable, the
// floor and margin, every role with its claim and whether it fits now, the last refusal. The
// limits and claims are edited in the Conductor view (Profiles, Residency).

import { html, nothing, type TemplateResult } from 'lit';
import { customElement, state } from 'lit/decorators.js';
import { api, describeError } from '../api/client';
import type { ResidencyPlan, SystemInfo } from '../api/types';
import { bytes, duration, exact, pct, timeShort, dateTimeShort, ago } from '../lib/format';
import { poll, type Poller } from '../lib/poll';
import { gib, gibOf, REASON_WORDS, rowWords, splitSegments } from '../lib/residency';
import { LightElement, type Loadable } from '../ui/base';
import { copyButton, icon, panel, skeleton, errorState, emptyState } from '../ui/bits';

const KEEP = 360; // 30 min at 5 s

interface Sample {
  t: number;
  alloc: number | null;
  reserved: number | null;
  avail: number | null;
  temp: number | null;
  power: number | null;
}

@customElement('qse-system')
export class QseSystem extends LightElement {
  @state() private sys: Loadable<SystemInfo> = { state: 'loading' };
  @state() private samples: Sample[] = [];
  /** the planner's view; null before it answered, 'off' on a server without the conductor */
  @state() private plan: ResidencyPlan | 'off' | null = null;
  private poller: Poller | null = null;
  private errorsAtOpen: number | null = null;

  connectedCallback(): void {
    super.connectedCallback();
    this.poller = poll(() => this.load(), 5000);
    this.poller.start();
  }
  disconnectedCallback(): void {
    this.poller?.stop();
    super.disconnectedCallback();
  }

  private async load(): Promise<void> {
    try {
      const s = await api.system();
      this.sys = { state: 'ready', data: s };
      if (this.errorsAtOpen == null) this.errorsAtOpen = s.inflight.errors;
      const next = this.samples.slice(-(KEEP - 1));
      next.push({ t: Date.now(), alloc: s.memory.gpu_allocated_bytes, reserved: s.memory.gpu_reserved_bytes, avail: s.memory.unified_available_bytes, temp: s.gpu.temperature_c, power: s.gpu.power_w });
      this.samples = next;
      void this.loadPlan();
    } catch (e) {
      const d = describeError(e);
      this.sys = { state: 'error', endpoint: d.endpoint, message: d.message };
    }
  }

  private async loadPlan(): Promise<void> {
    try {
      this.plan = await api.residencyPlan();
    } catch (e) {
      if (describeError(e).status === 404) this.plan = 'off';
    }
  }

  render() {
    const s = this.sys;
    if (s.state === 'loading') return html`<div class="view view-system">${skeleton(8)}</div>`;
    if (s.state === 'error') return html`<div class="view view-system">${errorState(s.endpoint, s.message, () => this.poller?.refresh())}</div>`;
    if (s.state === 'empty') return html`<div class="view view-system">${emptyState('No system data')}</div>`;
    const d = s.data;
    const queueNearFull = d.queue.waiting > 6;
    const diskLow = d.disk.state_dir_free_bytes != null && d.disk.state_dir_free_bytes < 20e9;
    const errorsRose = this.errorsAtOpen != null && d.inflight.errors > this.errorsAtOpen;
    const store = (d.caches as { state_store?: Record<string, number> }).state_store;
    const rc = (d.caches as { response_cache?: Record<string, number> | null }).response_cache;
    const suffix = (d.caches as { suffix_store?: Record<string, number | string | boolean> | null }).suffix_store;
    const flagCount = Object.keys(d.flags.env).filter((k) => k.startsWith('QWEN38_')).length;
    const g = d.gpu;
    const gpuNA = g.name == null && g.temperature_c == null;

    return html`<div class="view view-system">
      <div class="grid-3">
        ${panel(
          html`Engine <span class="pill pill-${d.engine.status}">${d.engine.status}</span>`,
          html`<dl class="kv">
            <div><dt>version</dt><dd class="num">${d.engine.version} <span class="muted">${d.engine.git_sha}</span></dd></div>
            <div><dt>code hash</dt><dd class="num">${d.engine.code_sha256.slice(0, 16)}… ${copyButton(d.engine.code_sha256, 'Copy code hash')}</dd></div>
            <div><dt>started</dt><dd class="num">${dateTimeShort(d.engine.started_at)}</dd></div>
            <div><dt>uptime</dt><dd class="num">${duration(d.engine.uptime_s)}</dd></div>
            <div><dt>pid</dt><dd class="num">${d.engine.pid}</dd></div>
            <div><dt>model</dt><dd>${d.engine.model}${d.engine.base_model && d.engine.base_model !== d.engine.model ? html` <span class="muted small" title="the base engine's name, a hidden alias">(base ${d.engine.base_model})</span>` : nothing}</dd></div>
            <div><dt>max_len</dt><dd class="num">${exact(d.engine.max_len)}</dd></div>
            <div><dt>drafter</dt><dd>${d.engine.drafter ?? 'none'}${d.engine.tree ? ', tree' : ''}</dd></div>
            <div><dt>flags</dt><dd><span class="num">${flagCount}</span> QWEN38_* set · <a href="#/dev">see the Dev tab</a></dd></div>
          </dl>`,
          { cls: d.engine.status === 'draining' ? 'is-warn' : '', id: 'card-engine' },
        )}
        ${panel(
          'Memory',
          html`<dl class="kv">
            <div><dt>unified total</dt><dd class="num">${bytes(d.memory.unified_total_bytes)}</dd></div>
            <div><dt>unified available</dt><dd class="num">${bytes(d.memory.unified_available_bytes)}</dd></div>
            <div><dt>GPU allocated</dt><dd class="num">${bytes(d.memory.gpu_allocated_bytes)}</dd></div>
            <div><dt>GPU reserved</dt><dd class="num">${bytes(d.memory.gpu_reserved_bytes)}</dd></div>
            <div><dt>GPU max allocated</dt><dd class="num">${bytes(d.memory.gpu_max_allocated_bytes)}</dd></div>
            <div><dt>process RSS</dt><dd class="num">${bytes(d.memory.process_rss_bytes)}</dd></div>
          </dl>
          <p class="muted small">nvidia-smi reports N/A for memory on GB10; these are the torch allocator and <code>/proc/meminfo</code>.</p>`,
          { id: 'card-memory' },
        )}
        ${panel(
          'GPU',
          gpuNA
            ? emptyState('Not available', 'The engine could not sample nvidia-smi; everything else is unaffected.')
            : html`<dl class="kv">
                <div><dt>name</dt><dd>${g.name ?? '—'}</dd></div>
                <div><dt>temperature</dt><dd class="num">${g.temperature_c == null ? 'not available' : `${g.temperature_c} °C`}</dd></div>
                <div><dt>power</dt><dd class="num">${g.power_w == null ? 'not available' : `${g.power_w.toFixed(1)} W`}</dd></div>
                <div><dt>SM clock</dt><dd class="num">${g.sm_clock_mhz == null ? 'not available' : `${g.sm_clock_mhz} MHz`}</dd></div>
                <div><dt>utilization</dt><dd class="num">${g.utilization == null ? 'not available' : pct(g.utilization, 0)}</dd></div>
                <div><dt>sampled</dt><dd class="num">${g.sampled_at ? `${timeShort(g.sampled_at)} (${ago(g.sampled_at)})` : '—'}</dd></div>
              </dl>`,
          { id: 'card-gpu' },
        )}
        ${panel(
          html`Queue ${queueNearFull ? html`<span class="pill pill-warn">${icon('warn')} queue nearly full</span>` : nothing}`,
          html`<dl class="kv">
            <div><dt>running</dt><dd class="num">${d.queue.running}</dd></div>
            <div><dt>waiting</dt><dd class="num">${d.queue.waiting} <span class="muted">of ${d.queue.max_queue}</span></dd></div>
            <div><dt>queue timeout</dt><dd class="num">${duration(d.queue.queue_timeout_s)}</dd></div>
            <div><dt>request timeout</dt><dd class="num">${duration(d.queue.request_timeout_s)}</dd></div>
          </dl>
          <div class="meter" role="img" aria-label=${`${d.queue.waiting} of ${d.queue.max_queue} waiting`}><span class="meter-fill ${queueNearFull ? 'is-warn' : ''}" style="width:${Math.min(100, (d.queue.waiting / Math.max(1, d.queue.max_queue)) * 100)}%"></span></div>`,
          { cls: queueNearFull ? 'is-warn' : '', id: 'card-queue' },
        )}
        ${panel(
          html`Since start ${errorsRose ? html`<span class="pill pill-warn">${icon('warn')} errors rose</span>` : nothing}`,
          html`<dl class="kv">
            <div><dt>served</dt><dd class="num">${exact(d.inflight.served)}</dd></div>
            <div><dt>refused</dt><dd class="num">${exact(d.inflight.refused)}</dd></div>
            <div><dt>errors</dt><dd class="num">${exact(d.inflight.errors)}</dd></div>
            <div><dt>timeouts</dt><dd class="num">${exact(d.inflight.timeouts)}</dd></div>
            <div><dt>abandoned</dt><dd class="num">${exact(d.inflight.abandoned)}</dd></div>
          </dl>`,
          { cls: errorsRose ? 'is-warn' : '', id: 'card-inflight' },
        )}
        ${panel(
          'Caches',
          html`${store
            ? html`<div class="meter-block">
                <div class="meter-head"><span>state store</span><span class="num">${bytes(store.bytes)} of ${bytes(store.budget)} · ${pct(store.bytes / Math.max(1, store.budget), 0)}</span></div>
                <div class="meter" role="img" aria-label=${`state store ${pct(store.bytes / Math.max(1, store.budget), 0)} of budget`}><span class="meter-fill" style="width:${Math.min(100, (store.bytes / Math.max(1, store.budget)) * 100)}%"></span></div>
              </div>
              <dl class="kv kv-grid">
                <div><dt>entries</dt><dd class="num">${exact(store.entries)}</dd></div>
                <div><dt>hits</dt><dd class="num">${exact(store.hits)}</dd></div>
                <div><dt>misses</dt><dd class="num">${exact(store.misses)}</dd></div>
                <div><dt>evictions</dt><dd class="num">${exact(store.evictions)}</dd></div>
              </dl>`
            : html`<p class="muted">state store off</p>`}
          <dl class="kv kv-grid">
            ${rc ? html`<div><dt>response cache</dt><dd class="num">${exact(rc.entries)} entries, ${exact(rc.hits)} hits</dd></div>` : html`<div><dt>response cache</dt><dd>off</dd></div>`}
            ${suffix ? html`<div><dt>suffix store</dt><dd class="num">${exact(Number(suffix.tokens))} tokens</dd></div>` : nothing}
          </dl>`,
          { id: 'card-caches' },
        )}
        ${panel(
          html`Ledger ${d.ledger.dropped > 0 ? html`<span class="pill pill-warn">${icon('warn')} ${d.ledger.dropped} dropped</span>` : nothing}`,
          html`<dl class="kv">
            <div><dt>enabled</dt><dd>${d.ledger.enabled ? 'yes' : 'no'}</dd></div>
            <div><dt>rows</dt><dd class="num">${exact(d.ledger.rows)}</dd></div>
            <div><dt>size</dt><dd class="num">${bytes(d.ledger.bytes)}</dd></div>
            <div><dt>oldest row</dt><dd class="num">${d.ledger.oldest ? dateTimeShort(d.ledger.oldest) : '—'}</dd></div>
            <div><dt>write queue</dt><dd class="num">${d.ledger.queue}</dd></div>
            <div><dt>dropped</dt><dd class="num">${d.ledger.dropped}</dd></div>
          </dl>`,
          { cls: d.ledger.dropped > 0 ? 'is-warn' : '', id: 'card-ledger' },
        )}
        ${panel(
          html`Disk ${diskLow ? html`<span class="pill pill-warn">${icon('warn')} low</span>` : nothing}`,
          html`<dl class="kv">
            <div><dt>state dir free</dt><dd class="num">${bytes(d.disk.state_dir_free_bytes)}</dd></div>
          </dl>
          ${diskLow ? html`<p class="warn-ink small">Below 20 GB: the ledger, backups and snapshots share this disk.</p>` : html`<p class="muted small">Warns below 20 GB.</p>`}`,
          { cls: diskLow ? 'is-warn' : '', id: 'card-disk' },
        )}
      </div>
      ${this.renderPlan(d)}
      ${this.renderRolling()}
    </div>`;
  }

  /** OPS-41: the engine's split and what may load beside it. Nothing on a server without the split. */
  private renderPlan(d: SystemInfo): TemplateResult | typeof nothing {
    const p = this.plan && this.plan !== 'off' ? this.plan : null;
    const split = p?.engine ?? d.memory.split ?? null;
    if (!split && !p) return nothing;
    const segs = splitSegments(split);
    const pl = p?.plan;
    const ref = p?.last_refusal;
    return panel(
      html`Memory plan <span class="muted small">what the engine holds, what may load beside it</span>`,
      html`${split
        ? html`<div class="split-bar" role="img" aria-label=${segs.map((x) => `${x.label} ${gib(x.bytes)}`).join(', ')} data-split>
              ${segs.map((x) => html`<span class="split-seg" style="flex-grow:${Math.max(1, Math.round(x.share * 1000))};--seg:var(--series-${x.color}, var(--${x.color}))" title=${`${x.label}: ${gib(x.bytes)}`} data-part=${x.key}></span>`)}
            </div>
            <dl class="kv kv-grid split-legend">
              ${segs.map((x) => html`<div data-part-row=${x.key}><dt><span class="split-dot" style="--seg:var(--series-${x.color}, var(--${x.color}))"></span>${x.label}</dt><dd class="num">${gib(x.bytes)}</dd></div>`)}
            </dl>
            <p class="small muted num">allocated ${gib(split.allocated)} · reserved ${gib(split.reserved)} · max_len ${exact(split.max_len)} · ${split.drafter_caches} drafter caches</p>
            ${split.load?.length
              ? html`<details class="sec"><summary>Load stages <span class="sec-sub">allocated / taken from MemAvailable, each stage of the start</span></summary>
                  <div class="table-scroll"><table class="table" data-load-stages><thead><tr><th>stage</th><th class="num-col">allocated</th><th class="num-col">reserved</th><th class="num-col">MemAvailable taken</th><th class="num-col">took</th></tr></thead>
                  <tbody>${split.load.map((x) => html`<tr><td>${x.stage}</td><td class="num num-col">${gib(x.allocated)}</td><td class="num num-col">${gib(x.reserved)}</td><td class="num num-col">${gib(x.mem_available_drop)}</td><td class="num num-col">${x.s} s</td></tr>`)}</tbody></table></div>
                </details>`
              : nothing}`
        : nothing}
      ${pl
        ? html`<div class="stats residency-stats" data-residency>
              <div class="stat"><div class="stat-label">MemAvailable</div><div class="stat-value num">${gibOf(pl.mem_available_gib)}</div></div>
              <div class="stat"><div class="stat-label">floor + margin</div><div class="stat-value num">${gibOf(pl.floor_gib)} + ${gibOf(pl.margin_gib)}</div></div>
              <div class="stat"><div class="stat-label">spendable above it</div><div class="stat-value num ${pl.spendable_gib != null && pl.spendable_gib < 0 ? 'warn-ink' : ''}">${gibOf(pl.spendable_gib)}</div></div>
              <div class="stat"><div class="stat-label">settle after a stop</div><div class="stat-value num">${pl.settle_s} s</div></div>
            </div>
            <div class="table-scroll"><table class="table" data-residency-roles>
              <thead><tr><th>role, in load order</th><th class="num-col">claim</th><th>basis</th><th>now</th></tr></thead>
              <tbody>${pl.rows.map((r) => html`<tr data-residency-role=${r.role} class=${r.allowed ? '' : 'is-warn'}><td>${r.role}</td><td class="num num-col">${gibOf(r.claim_gib)}</td><td title=${r.note ?? ''}>${r.basis ?? '—'}${r.source === 'roles' ? html` <span class="muted small">(roles)</span>` : nothing}</td><td class=${r.allowed ? '' : 'warn-ink'}>${rowWords(r)}</td></tr>`)}</tbody>
            </table></div>
            <p class="small" data-last-refusal>${ref
              ? html`Last refusal: <b>${ref.role}</b>, ${ref.reason ? REASON_WORDS[ref.reason as keyof typeof REASON_WORDS] ?? ref.reason : 'refused'} <span class="muted num">(${dateTimeShort(new Date(ref.ts * 1000).toISOString())}${ref.mem_available_gib != null ? `, MemAvailable ${gibOf(ref.mem_available_gib)}` : ''}${ref.source ? `, ${ref.source}` : ''})</span>`
              : html`<span class="muted">No refusal in the planner's journal.</span>`}</p>
            <p class="muted small">A role loads after the engine, one at a time, when MemAvailable minus its claim stays at or above floor + margin, and never during a hold. <a href="#/conductor?screen=profiles">Edit the limits and claims</a> (Conductor, Profiles, Residency; version ${p!.version}).</p>`
        : this.plan === 'off'
          ? html`<p class="muted small">The residency planner needs the conductor on this server (<code>--conductor-config</code>).</p>`
          : nothing}`,
      { id: 'card-residency', cls: pl && pl.spendable_gib != null && pl.spendable_gib < 0 ? 'is-warn' : '' },
    );
  }

  private renderRolling(): TemplateResult {
    const s = this.samples;
    const labels = s.map((x) => timeShort(new Date(x.t).toISOString()).slice(0, 5));
    const gb = (v: number) => (v / 1e9).toFixed(1);
    return html`<div class="grid-2">
      ${panel(
        'Memory, last 30 minutes',
        html`<qse-ribbon
          .series=${[
            { name: 'GPU allocated', color: 'cobalt', values: s.map((x) => x.alloc) },
            { name: 'GPU reserved', color: 'violet', values: s.map((x) => x.reserved) },
            { name: 'unified available', color: 'aqua', values: s.map((x) => x.avail) },
          ]}
          .labels=${labels}
          .format=${(v: number) => gb(v)}
          unit="GB"
          .height=${220}
          aria-label="GPU allocated, GPU reserved and unified available memory over the last 30 minutes"
        ></qse-ribbon>
        <p class="panel-foot">${s.length} sample${s.length === 1 ? '' : 's'} at 5 s, kept in the browser and reset on reload. Allocated climbing across requests is a leak; reserved climbing alone is fragmentation.</p>`,
        { id: 'chart-memory' },
      )}
      ${panel(
        'GPU temperature and power',
        html`<div class="grid-2 grid-2-tight">
          <qse-ribbon .series=${[{ name: 'temperature', color: 'orange', values: s.map((x) => x.temp) }]} .labels=${labels} .format=${(v: number) => String(Math.round(v))} unit="°C" .height=${160} aria-label="GPU temperature"></qse-ribbon>
          <qse-ribbon .series=${[{ name: 'power', color: 'yellow', values: s.map((x) => x.power) }]} .labels=${labels} .format=${(v: number) => String(Math.round(v))} unit="W" .height=${160} aria-label="GPU power"></qse-ribbon>
        </div>`,
        { id: 'chart-gpu' },
      )}
    </div>`;
  }
}
