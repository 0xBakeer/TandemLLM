// P2, P3 in the mock: the proxy's qse_backend_* families for the mock's vLLM and SGLang backends
// (counters that move with the clock, as proxy/engines.py maps the engines' own pages), and the
// `engine` block of their status rows. Mode `nospec` drops vLLM's speculation families.

type J = Record<string, unknown>;

const T0 = Date.now();
const TTFT = [0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 7.5, 10, 20, 40, 80, 160, 640, 2560];
const LAT = [0.3, 0.5, 0.8, 1, 1.5, 2, 2.5, 5, 10, 15, 20, 30, 40, 50, 60, 120, 240, 480, 960, 1920, 7680];

function hist(name: string, backend: string, bounds: number[], n: number, at: (i: number) => number): string[] {
  const vals = Array.from({ length: n }, (_, i) => at(i));
  const out = [`# TYPE qse_backend_${name} histogram`];
  for (const le of bounds) out.push(`qse_backend_${name}_bucket{backend="${backend}",le="${le}"} ${vals.filter((v) => v <= le).length}`);
  out.push(`qse_backend_${name}_bucket{backend="${backend}",le="+Inf"} ${n}`);
  out.push(`qse_backend_${name}_sum{backend="${backend}"} ${vals.reduce((a, b) => a + b, 0).toFixed(3)}`);
  out.push(`qse_backend_${name}_count{backend="${backend}"} ${n}`);
  return out;
}

/** The families, as the proxy's /metrics carries them. */
export function engineFamilies(now: number, ids: { vllm?: string; sglang?: string }, spec = true): string {
  const s = Math.max(1, (now - T0) / 1000);
  const n = Math.floor(s / 4) + 40; // a request every 4 s
  const out: string[] = ['# TYPE qse_backend_info gauge'];
  if (ids.vllm) out.push(`qse_backend_info{backend="${ids.vllm}",kind="vllm",version="0.30.0"} 1`);
  if (ids.sglang) out.push(`qse_backend_info{backend="${ids.sglang}",kind="sglang",version="0.5.21"} 1`);
  const v = ids.vllm;
  if (v) {
    const wave = 0.5 + 0.5 * Math.sin(s / 20);
    out.push(
      `qse_backend_requests_running{backend="${v}"} ${Math.round(1 + 2 * wave)}`,
      `qse_backend_requests_waiting{backend="${v}"} ${wave > 0.9 ? 1 : 0}`,
      `qse_backend_kv_usage{backend="${v}"} ${(0.2 + 0.25 * wave).toFixed(4)}`,
      `qse_backend_awake{backend="${v}"} 1`,
      `qse_backend_prompt_tokens_total{backend="${v}"} ${Math.floor(52000 + s * 70)}`,
      `qse_backend_generation_tokens_total{backend="${v}"} ${Math.floor(9000 + s * 42)}`,
      `qse_backend_prefix_cache_queries_total{backend="${v}"} ${Math.floor(52000 + s * 70)}`,
      `qse_backend_prefix_cache_hits_total{backend="${v}"} ${Math.floor(30000 + s * 53)}`,
      `qse_backend_prompt_tokens_cached_total{backend="${v}"} ${Math.floor(30000 + s * 53)}`,
      `qse_backend_requests_finished_total{backend="${v}",reason="stop"} ${n}`,
      `qse_backend_preemptions_total{backend="${v}"} ${Math.floor(s / 600)}`,
      ...hist('ttft_seconds', v, TTFT, n, (i) => 0.08 + (i % 5) * 0.04),
      ...hist('queue_seconds', v, LAT, n, (i) => 0.2 + (i % 4) * 0.2),
      ...hist('prefill_seconds', v, LAT, n, (i) => 0.3 + (i % 3) * 0.3),
      ...hist('decode_seconds', v, LAT, n, (i) => 2 + (i % 6)),
      ...hist('e2e_seconds', v, LAT, n, (i) => 3 + (i % 6)),
    );
    if (spec) {
      const drafts = Math.floor(3000 + s * 14);
      out.push(
        `qse_backend_spec_drafts_total{backend="${v}"} ${drafts}`,
        `qse_backend_spec_draft_tokens_total{backend="${v}"} ${drafts * 3}`,
        `qse_backend_spec_accepted_tokens_total{backend="${v}"} ${Math.floor(drafts * 2.2)}`,
        `qse_backend_spec_accepted_tokens_per_pos_total{backend="${v}",position="0"} ${Math.floor(drafts * 0.92)}`,
        `qse_backend_spec_accepted_tokens_per_pos_total{backend="${v}",position="1"} ${Math.floor(drafts * 0.74)}`,
        `qse_backend_spec_accepted_tokens_per_pos_total{backend="${v}",position="2"} ${Math.floor(drafts * 0.54)}`,
      );
    }
  }
  const g = ids.sglang;
  if (g) {
    out.push(
      `qse_backend_requests_running{backend="${g}"} 2`,
      `qse_backend_requests_waiting{backend="${g}"} 0`,
      `qse_backend_kv_usage{backend="${g}"} 0.4`,
      `qse_backend_kv_tokens{backend="${g}"} 40000`,
      `qse_backend_kv_capacity_tokens{backend="${g}"} 100000`,
      `qse_backend_gen_tps{backend="${g}"} 118.5`,
      `qse_backend_cache_hit_rate{backend="${g}"} 0.61`,
      `qse_backend_prompt_tokens_total{backend="${g}"} ${Math.floor(40000 + s * 80)}`,
      `qse_backend_prompt_tokens_cached_total{backend="${g}"} ${Math.floor(23000 + s * 56)}`,
      `qse_backend_generation_tokens_total{backend="${g}"} ${Math.floor(7000 + s * 120)}`,
      `qse_backend_requests_finished_total{backend="${g}"} ${n}`,
      `qse_backend_spec_accept_rate{backend="${g}"} 0.7`,
      `qse_backend_spec_accept_length{backend="${g}"} 3.1`,
      ...hist('ttft_seconds', g, [0.1, 0.2, 0.4, 0.6, 0.8, 1, 2, 4], n, (i) => 0.15 + (i % 3) * 0.2),
      ...hist('queue_seconds', g, [0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2], n, (i) => 0.02 + (i % 4) * 0.1),
    );
  }
  return out.join('\n') + '\n';
}

/** The `engine` block of a status row, with fixed figures over a 5-minute window. */
export function engineBlock(kind: string, metricsOff = false): J | null {
  if (kind !== 'vllm' && kind !== 'sglang') return null;
  const split = (p50: number, p90: number, count: number) => ({ p50, p90, count });
  if (kind === 'sglang' && metricsOff)
    return { layer: 'L1', metrics: 'off', error: 'HTTP 404', scrape_age_s: null, version: '0.5.21', verified: '0.5.21', info: { version: '0.5.21', context_length: 32768, speculative: null, cache_report: false, reasoning_parser: null, metrics_enabled: false }, figures: null };
  if (kind === 'vllm')
    return {
      layer: 'L2', metrics: 'ok', error: null, scrape_age_s: 4.2, version: '0.30.0', verified: '0.30.0', info: { version: '0.30.0', context_length: null, speculative: null, cache_report: null, reasoning_parser: null, metrics_enabled: null },
      // P6: the plugin inside this vLLM posts (tandemllm-vllm-plugin)
      plugin: { state: 'l2', age_s: 0.8, via: 'push', plugin: 'tandemllm-vllm-plugin', plugin_version: '0.1.0', engine: 'vllm', engine_version: '0.30.0', joined: 73, approximate: 2, unmatched: 0, untagged: 0, dropped: { requests: 0, steps: 0 }, last: { running: 2, waiting: 0, kv_usage: 0.31 }, window: { samples: 300, preempted: 1, evictions: 4, prefix_hit_rate: 0.76, spec: { acceptance: 0.733, tokens_per_step: 3.2, per_position: [0.92, 0.74, 0.54] } } },
      figures: {
        running: 2, waiting: 0, kv_usage: 0.31, kv_tokens: null, kv_capacity_tokens: null, awake: true, decode_tps: 41.6, prompt_tps: 70.1, engine_gen_tps: null, cached_share: 0.76, prefix_hit_rate: 0.76, engine_cache_hit_rate: null,
        ttft_p50_s: 0.16, queue_s: split(0.42, 0.8, 75), prefill_s: split(0.55, 0.9, 75), decode_s: split(4.4, 6.8, 75), e2e_s: split(5.2, 7.9, 75), preemptions: 0, finished: 75, window_s: 300,
        spec: { source: 'counters', acceptance: 0.733, tokens_per_step: 3.2, per_position: [0.92, 0.74, 0.54], drafts: 4200 },
      },
    };
  return {
    layer: 'L1', metrics: 'ok', error: null, scrape_age_s: 9.8, version: '0.5.21', verified: '0.5.21', info: { version: '0.5.21', context_length: 32768, speculative: { algorithm: 'EAGLE3', num_steps: 3, num_draft_tokens: 4 }, cache_report: true, reasoning_parser: 'qwen3', metrics_enabled: true },
    figures: {
      running: 2, waiting: 0, kv_usage: 0.4, kv_tokens: 40000, kv_capacity_tokens: 100000, awake: null, decode_tps: 120, prompt_tps: 80, engine_gen_tps: 118.5, cached_share: 0.7, prefix_hit_rate: 0.7, engine_cache_hit_rate: 0.61,
      ttft_p50_s: 0.6, queue_s: split(0.5, 1.8, 75), prefill_s: null, decode_s: null, e2e_s: split(3, 3.8, 75), preemptions: 0, finished: 75, window_s: 300,
      spec: { source: 'engine_window', acceptance: 0.7, tokens_per_step: 3.1, per_position: null, drafts: null },
    },
  };
}
