// P2, P3: the vLLM and SGLang backends' own engines at layer L1 (proxy/engines.py). One card per
// backend: requests running and waiting, KV use, the decode rate, the prefix-cache hit rate, TTFT,
// the queue, prefill and decode split, preemptions and speculation. The Performance view fills the
// figures from its own /metrics scrapes (the qse_backend_* families), the System view and the
// Backends screen from the backend rows' `engine` block; both come out of the same arithmetic.
// A number the engine's page does not give says "not reported by this backend".

import { html, nothing, type TemplateResult } from 'lit';
import type { EngineBlock, EngineFigures, PhaseSplit } from '../api/types';
import { compact, ms, NOT_REPORTED, pct, tps } from '../lib/format';
import { panel, statTile } from '../ui/bits';

export interface EngineCard {
  id: string;
  label: string | null;
  kind: string;
  version?: string | null;
  engine?: EngineBlock | null;
  figures: EngineFigures | null;
}

export const ENGINE_KINDS = new Set(['vllm', 'sglang']);
const KIND_WORD: Record<string, string> = { vllm: 'vLLM', sglang: 'SGLang' };

/** What the engine's own flags decide about the per-request counts, in one line. */
export function flagNote(kind: string, e: EngineBlock | null | undefined): string | null {
  if (kind === 'sglang') {
    if (e?.metrics === 'off') return 'No /metrics page: start SGLang with --enable-metrics.';
    if (e?.info?.cache_report === false) return 'Per-request cached tokens need --enable-cache-report.';
    return null;
  }
  if (kind === 'vllm') return 'Per request: cached tokens with --enable-prompt-tokens-details, reasoning tokens with --reasoning-parser.';
  return null;
}

const sec = (v: number | null | undefined) => (v == null ? '—' : ms(v * 1000));

function splitTile(label: string, s: PhaseSplit | null, reported: boolean): TemplateResult {
  if (!reported) return statTile({ label, value: 'not reported', sub: NOT_REPORTED, na: true });
  return statTile({ label: `${label}, p50`, value: s ? sec(s.p50) : 'none', sub: s ? `p90 ${sec(s.p90)} · ${s.count} requests` : 'no request in the window', na: !s });
}

function kvTile(f: EngineFigures): TemplateResult {
  const u = f.kv_usage;
  const tokens = f.kv_tokens != null ? `${compact(f.kv_tokens)}${f.kv_capacity_tokens ? ` of ${compact(f.kv_capacity_tokens)}` : ''} tokens` : null;
  return html`<div class="stat ${u == null ? 'is-na' : ''}" data-engine-kv>
    <div class="stat-label">KV cache in use</div>
    <div class="stat-value"><span class="num">${u == null ? 'not reported' : pct(u, 0)}</span></div>
    ${u != null ? html`<div class="meter" role="img" aria-label=${`KV cache ${pct(u, 0)} in use`}><span class="meter-fill ${u > 0.9 ? 'is-warn' : ''}" style="width:${Math.min(100, u * 100)}%"></span></div>` : nothing}
    ${tokens ? html`<div class="stat-sub">${tokens}</div>` : nothing}
  </div>`;
}

function specTiles(kind: string, f: EngineFigures): TemplateResult {
  const s = f.spec;
  if (!s) return statTile({ label: 'draft acceptance', value: 'not reported', sub: `${NOT_REPORTED}: speculation off`, na: true });
  const perPos = s.per_position?.length
    ? html`<div class="eng-pos" role="img" aria-label=${`acceptance by draft position: ${s.per_position.map((x, i) => `${i + 1}: ${x == null ? '—' : pct(x, 0)}`).join(', ')}`}>
        ${s.per_position.map((x, i) => html`<span class="eng-pos-bar" title=${`position ${i + 1}: ${x == null ? '—' : pct(x, 0)}`} style="height:${Math.max(2, (x ?? 0) * 28)}px"></span>`)}
      </div>`
    : nothing;
  const src = s.source === 'counters' ? 'accepted / drafted over the window' : `${KIND_WORD[kind] ?? kind}'s own last window`;
  return html`${statTile({ label: 'draft acceptance', value: s.acceptance == null ? 'none' : pct(s.acceptance, 1), sub: src, na: s.acceptance == null })}
    <div class="stat ${s.tokens_per_step == null ? 'is-na' : ''}" data-engine-steps>
      <div class="stat-label">tokens per step</div>
      <div class="stat-value"><span class="num">${s.tokens_per_step == null ? 'none' : s.tokens_per_step.toFixed(2)}</span></div>
      <div class="stat-sub">accepted drafts plus the sampled token</div>
      ${perPos}
    </div>`;
}

export function engineCard(c: EngineCard): TemplateResult {
  const e = c.engine ?? null;
  const f = c.figures;
  const state = e?.metrics ?? (f ? 'ok' : 'unknown');
  const version = c.version ?? e?.version ?? null;
  const vllm = c.kind === 'vllm';
  const note = flagNote(c.kind, e);
  const head = html`<header class="eng-head">
    <span class="eng-title">${c.label ?? c.id}</span>
    <span class="badge">${KIND_WORD[c.kind] ?? c.kind}${version ? ` ${version}` : ''}</span>
    <span class="pill eng-state is-${state}" data-state=${state}>${state === 'ok' ? 'metrics read' : state === 'off' ? 'no metrics' : state}</span>
    ${f?.awake === false ? html`<span class="pill is-warn">asleep</span>` : nothing}
    <span class="grow"></span>
    ${e?.scrape_age_s != null ? html`<span class="muted small num">read ${e.scrape_age_s.toFixed(0)} s ago</span>` : nothing}
  </header>`;
  if (!f)
    return html`<article class="eng-card" data-engine=${c.id} data-kind=${c.kind}>${head}
      <p class="muted small">${state === 'off' ? note : state === 'error' ? html`The page could not be read: ${e?.error ?? 'unknown error'}. Tried again with back-off.` : 'Waiting for the first read of the engine’s /metrics.'}</p>
    </article>`;
  return html`<article class="eng-card" data-engine=${c.id} data-kind=${c.kind}>${head}
    <div class="stats eng-stats">
      ${statTile({ label: 'running · waiting', value: `${f.running ?? '—'} · ${f.waiting ?? '—'}`, sub: f.preemptions ? `${f.preemptions} preempted in the window` : 'requests in the engine now' })}
      ${kvTile(f)}
      ${statTile({ label: 'decode', value: f.decode_tps == null ? 'none' : tps(f.decode_tps), sub: f.engine_gen_tps != null ? `engine says ${tps(f.engine_gen_tps)}` : 'all requests together, over the window', na: f.decode_tps == null })}
      ${statTile({ label: 'prefix-cache hits', value: f.prefix_hit_rate == null ? 'none' : pct(f.prefix_hit_rate, 0), sub: f.prefix_hit_rate == null ? 'no prompt in the window' : vllm ? 'hit tokens over looked-up tokens' : 'cached over prompt tokens', na: f.prefix_hit_rate == null })}
      ${statTile({ label: 'time to first token, p50', value: f.ttft_p50_s == null ? 'none' : sec(f.ttft_p50_s), sub: 'the engine’s own histogram', na: f.ttft_p50_s == null })}
      ${splitTile('queue', f.queue_s, true)}
      ${splitTile('prefill', f.prefill_s, vllm)}
      ${splitTile('decode time', f.decode_s, vllm)}
      ${specTiles(c.kind, f)}
    </div>
    ${note ? html`<p class="muted small eng-note">${note}</p>` : nothing}
  </article>`;
}

/** The panel with one card per vLLM or SGLang backend; nothing when there is none. */
export function renderEngines(cards: EngineCard[], sub: string): TemplateResult | typeof nothing {
  const list = cards.filter((c) => ENGINE_KINDS.has(c.kind));
  if (!list.length) return nothing;
  return panel(html`Backend engines <span class="muted small">vLLM and SGLang, read from their own /metrics</span>`, html`<div class="eng-cards">${list.map(engineCard)}</div>`, { id: 'backend-engines', sub });
}
