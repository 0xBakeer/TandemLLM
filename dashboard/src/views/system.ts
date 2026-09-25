// System — the box at a glance (VIS-17): engine, memory, GPU, queue, inflight counters, caches,
// ledger, disk, plus rolling 30-minute charts kept in the browser (the soak panel).

import { html, nothing, type TemplateResult } from 'lit';
import { customElement, state } from 'lit/decorators.js';
import { api, describeError } from '../api/client';
import type { SystemInfo } from '../api/types';
import { bytes, duration, exact, pct, timeShort, dateTimeShort, ago } from '../lib/format';
import { poll, type Poller } from '../lib/poll';
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
    } catch (e) {
      const d = describeError(e);
      this.sys = { state: 'error', endpoint: d.endpoint, message: d.message };
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
            <div><dt>model</dt><dd>${d.engine.model}</dd></div>
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
      ${this.renderRolling()}
    </div>`;
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
