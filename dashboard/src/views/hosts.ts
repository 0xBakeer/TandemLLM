// P1.5: the System view's hosts -- the machine behind each backend (contract 1.3 `system.hosts`).
// This machine first (the server's own process and every backend on it), then one panel per remote
// collector or node_exporter + DCGM pair. Each panel: memory (unified on a GB10), CPU, the GPUs, the
// disks and network as rates, the process tree behind each backend's port with its RSS and GPU
// memory, and a 30-minute history kept in the browser.

import { html, nothing, type TemplateResult } from 'lit';
import type { HostGpu, HostInfo, HostProcess } from '../api/types';
import { ago, bytes, duration, pct } from '../lib/format';
import { panel } from '../ui/bits';

export const HOST_KEEP = 360; // 30 min at 5 s

export interface HostSample {
  t: number;
  cpu: number | null;
  gpu: number | null;
  avail: number | null;
}

/** one sample per host id, the newest last, at most HOST_KEEP each */
export function pushHostSamples(prev: Map<string, HostSample[]>, hosts: HostInfo[] | undefined, t: number): Map<string, HostSample[]> {
  const next = new Map<string, HostSample[]>();
  for (const h of hosts ?? []) {
    const old = prev.get(h.id) ?? [];
    const gpus = h.gpus.filter((g) => g.utilization != null);
    next.set(h.id, [
      ...old.slice(-(HOST_KEEP - 1)),
      {
        t,
        cpu: h.cpu?.utilization ?? null,
        gpu: gpus.length ? gpus.reduce((a, g) => a + (g.utilization ?? 0), 0) / gpus.length : null,
        avail: h.memory?.available_bytes ?? null,
      },
    ]);
  }
  return next;
}

export function rate(bps: number | null | undefined): string {
  return bps == null ? '—' : `${bytes(bps, 1)}/s`;
}

const SOURCE_WORDS: Record<HostInfo['source'], string> = {
  local: 'this machine',
  collector: 'remote collector',
  exporter: 'node_exporter + DCGM',
};

export function sourceWords(h: HostInfo): string {
  return SOURCE_WORDS[h.source] ?? h.source;
}

/** a GPU's memory: bytes of total, or why there is no figure */
export function gpuMemoryWords(g: HostGpu): string {
  if (g.memory_unified) return 'unified with the CPU';
  if (g.memory_used_bytes == null) return 'not reported';
  return g.memory_total_bytes != null ? `${bytes(g.memory_used_bytes, 1)} of ${bytes(g.memory_total_bytes, 1)}` : bytes(g.memory_used_bytes, 1);
}

/** a process tree's GPU memory as the collector got it */
export function procGpuWords(p: HostProcess): string {
  switch (p.gpu_memory) {
    case 'measured':
      return bytes(p.total_gpu_memory_bytes ?? p.gpu_memory_bytes, 1);
    case 'unified':
      return 'unified (no figure)';
    case 'none':
      return 'not on the GPU';
    default:
      return 'not reported';
  }
}

export function procLabel(p: HostProcess): string {
  if (p.backend === 'engine') return p.label ? `${p.label} (this engine)` : 'this engine';
  if (p.backend === 'proxy') return 'this proxy';
  return p.label && p.label !== p.backend ? `${p.label} (${p.backend})` : (p.backend ?? '—');
}

function stateBadge(h: HostInfo): TemplateResult | typeof nothing {
  if (h.state === 'ok') return nothing;
  const words = h.state === 'pending' ? 'first read pending' : h.state === 'stale' ? `stale, ${duration(h.age_s)} old` : `error: ${h.error ?? 'no answer'}`;
  return html`<span class="pill ${h.state === 'pending' ? '' : 'pill-warn'}" data-host-state=${h.state}>${words}</span>`;
}

function memoryBlock(h: HostInfo): TemplateResult {
  const m = h.memory;
  if (!m) return html`<div class="stat"><div class="stat-label">memory</div><div class="stat-value num">—</div></div>`;
  const share = m.total_bytes && m.used_bytes != null ? m.used_bytes / m.total_bytes : null;
  return html`<div class="stat" data-host-memory>
    <div class="stat-label">${m.unified ? 'unified memory available' : 'memory available'}</div>
    <div class="stat-value num">${bytes(m.available_bytes, 1)}</div>
    <div class="stat-sub muted small num">of ${bytes(m.total_bytes, 1)}${share != null ? ` · ${pct(share, 0)} used` : ''}${m.swap_used_bytes ? ` · swap ${bytes(m.swap_used_bytes, 1)}` : ''}</div>
  </div>`;
}

function cpuBlock(h: HostInfo): TemplateResult {
  const c = h.cpu;
  return html`<div class="stat" data-host-cpu>
    <div class="stat-label">CPU${c?.count ? ` · ${c.count} cores` : ''}</div>
    <div class="stat-value num">${c?.utilization == null ? '—' : pct(c.utilization, 0)}</div>
    <div class="stat-sub muted small num">load ${(c?.load ?? []).map((x) => (x == null ? '—' : x.toFixed(2))).join(' / ') || '—'}${c?.temperature_c != null ? ` · ${c.temperature_c} °C` : ''}</div>
  </div>`;
}

function gpuBlock(g: HostGpu, many: boolean): TemplateResult {
  return html`<div class="stat" data-host-gpu=${g.index ?? 0}>
    <div class="stat-label">${many ? `GPU ${g.index} · ` : ''}${g.name ?? 'GPU'}</div>
    <div class="stat-value num">${g.utilization == null ? '—' : pct(g.utilization, 0)}</div>
    <div class="stat-sub muted small num">${[
      g.power_w != null ? `${g.power_w.toFixed(0)} W${g.power_limit_w != null ? ` of ${g.power_limit_w.toFixed(0)}` : ''}` : null,
      g.temperature_c != null ? `${g.temperature_c} °C` : null,
      g.sm_clock_mhz != null ? `${g.sm_clock_mhz} MHz` : null,
    ]
      .filter(Boolean)
      .join(' · ') || '—'}</div>
    <div class="stat-sub muted small num" data-gpu-memory>memory ${gpuMemoryWords(g)}</div>
  </div>`;
}

function ioBlock(h: HostInfo): TemplateResult | typeof nothing {
  const disks = h.disks.filter((d) => d.read_bps != null || d.write_bps != null || d.nvme);
  if (!disks.length && !h.net.length) return nothing;
  return html`<div class="table-scroll"><table class="table" data-host-io>
    <thead><tr><th>device</th><th class="num-col">read / in</th><th class="num-col">write / out</th><th class="num-col">busy</th></tr></thead>
    <tbody>
      ${disks.map((d) => html`<tr><td>${d.name}${d.nvme ? html` <span class="muted small">NVMe</span>` : nothing}</td><td class="num num-col">${rate(d.read_bps)}</td><td class="num num-col">${rate(d.write_bps)}</td><td class="num num-col">${d.busy == null ? '—' : pct(d.busy, 0)}</td></tr>`)}
      ${h.net.map((n) => html`<tr><td>${n.name} <span class="muted small">net</span></td><td class="num num-col">${rate(n.rx_bps)}</td><td class="num num-col">${rate(n.tx_bps)}</td><td class="num num-col">—</td></tr>`)}
      ${h.fs.map((f) => html`<tr><td class="mono small">${f.path} <span class="muted small">free</span></td><td class="num num-col" colspan="3">${bytes(f.free_bytes, 1)} of ${bytes(f.total_bytes, 1)}</td></tr>`)}
    </tbody>
  </table></div>`;
}

function processTable(h: HostInfo): TemplateResult {
  if (h.processes == null)
    return html`<p class="muted small" data-host-noproc>This source gives no per-process figures (node_exporter and DCGM report per machine and per GPU). Run <code>python -m proxy.hostcollect</code> on that host for the process behind each backend.</p>`;
  if (!h.processes.length) return html`<p class="muted small">No backend on this host.</p>`;
  return html`<div class="table-scroll"><table class="table" data-host-procs>
    <thead><tr><th>backend</th><th>process</th><th class="num-col">RSS</th><th class="num-col">GPU memory</th><th class="num-col">CPU</th><th class="num-col">threads</th><th class="num-col">up</th></tr></thead>
    <tbody>${h.processes.map((p) => html`<tr data-proc=${p.backend ?? ''} class=${p.found ? '' : 'is-warn'}>
      <td>${procLabel(p)}${p.port ? html` <span class="muted small num">:${p.port}</span>` : nothing}</td>
      <td>${p.found
        ? html`<span class="num">${p.pid}</span> ${p.name ?? ''}${p.children.length ? html` <span class="muted small">+${p.children.length} child${p.children.length === 1 ? '' : 'ren'}</span>` : nothing}
            ${p.cmdline || p.children.length
              ? html`<details class="proc-more"><summary class="small muted">details</summary>
                  ${p.cmdline ? html`<div class="mono small proc-cmd">${p.cmdline}</div>` : nothing}
                  ${p.children.map((c) => html`<div class="small num">${c.pid} ${c.name ?? ''} · ${bytes(c.rss_bytes, 1)}${c.gpu_memory_bytes != null ? ` · GPU ${bytes(c.gpu_memory_bytes, 1)}` : ''}</div>`)}
                </details>`
              : nothing}`
        : html`<span class="muted small">${p.note ?? 'not found'}</span>`}</td>
      <td class="num num-col">${bytes(p.total_rss_bytes ?? p.rss_bytes, 1)}</td>
      <td class="num num-col" data-proc-gpu title=${p.note ?? ''}>${procGpuWords(p)}</td>
      <td class="num num-col">${p.cpu == null ? '—' : p.cpu.toFixed(2)}</td>
      <td class="num num-col">${p.threads ?? '—'}</td>
      <td class="num num-col">${duration(p.uptime_s)}</td>
    </tr>`)}</tbody>
  </table></div>
  ${h.memory?.unified && h.processes.some((p) => p.gpu_memory === 'measured') ? html`<p class="muted small">Unified memory: a process's GPU memory comes out of the same pool as the RSS, so the two overlap.</p>` : nothing}`;
}

function footer(h: HostInfo): TemplateResult {
  const src = h.gpu_source === 'nvml' ? 'GPU via NVML' : h.gpu_source === 'nvidia-smi' ? 'GPU via nvidia-smi' : h.gpu_source === 'dcgm' ? 'GPU via DCGM' : 'no GPU source';
  return html`<p class="panel-foot small" data-host-foot>${sourceWords(h)}${h.url ? html` <span class="mono">${h.url}</span>` : nothing} · ${src}${h.sampled_at ? ` · sampled ${ago(h.sampled_at)}` : ''}${h.cost?.last_ms != null
    ? html` · <span title="CPU time the collector itself spent on its last sample and process scan">collector ${h.cost.last_ms.toFixed(1)} ms, scan ${h.cost.last_scan_ms?.toFixed(1) ?? '—'} ms, every ${h.cost.fast_period_s} / ${h.cost.scan_period_s} s</span>`
    : nothing}</p>`;
}

export function renderHosts(hosts: HostInfo[] | undefined, history: Map<string, HostSample[]>): TemplateResult | typeof nothing {
  if (!hosts?.length) return nothing;
  return html`<div class="hosts" data-hosts>
    ${hosts.map((h) => {
      const s = history.get(h.id) ?? [];
      const labels = s.map((x) => new Date(x.t).toTimeString().slice(0, 5));
      return panel(
        html`${h.source === 'local' ? 'This machine' : 'Host'} <span class="muted">${h.label ?? h.id}</span> <span class="pill">${sourceWords(h)}</span> ${stateBadge(h)}`,
        html`<div class="stats host-stats">
            ${memoryBlock(h)}${cpuBlock(h)}${h.gpus.map((g) => gpuBlock(g, h.gpus.length > 1))}
          </div>
          ${processTable(h)}
          ${ioBlock(h)}
          ${s.length > 1
            ? html`<qse-ribbon
                .series=${[
                  { name: 'CPU', color: 'cobalt', values: s.map((x) => (x.cpu == null ? null : x.cpu * 100)) },
                  { name: 'GPU', color: 'orange', values: s.map((x) => (x.gpu == null ? null : x.gpu * 100)) },
                ]}
                .labels=${labels}
                .format=${(v: number) => String(Math.round(v))}
                unit="%"
                .height=${140}
                aria-label=${`CPU and GPU use on ${h.label ?? h.id} over the last 30 minutes`}
              ></qse-ribbon>`
            : nothing}
          ${footer(h)}`,
        { id: `card-host-${h.id}`, cls: h.state === 'error' ? 'is-warn' : '', sub: h.backends.length ? `backends: ${h.backends.map((b) => (b === 'engine' ? 'this engine' : b === 'proxy' ? 'this proxy' : b)).join(', ')}` : 'no backend placed here' },
      );
    })}
  </div>`;
}
