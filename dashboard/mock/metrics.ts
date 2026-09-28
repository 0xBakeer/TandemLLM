// Synthetic /metrics in the Prometheus text format, the engine's qse_* names (server/METRICS.md
// plus the additions). Counters move between scrapes because they are derived from the
// mock engine's live state.

import type { LedgerRow } from './generate.ts';

export interface LiveEngineState {
  startedAt: number;
  running: number;
  waiting: number;
  generationTokens: number; // moves while a request runs
  promptTokens: number;
  cachedPromptTokens: number;
  reasoningTokens: number;
  requestsByFinish: Record<string, number>;
  refusedByReason: Record<string, number>;
  errorsByType: Record<string, number>;
  ttft: number[]; // seconds, recent observations (bounded)
  acceptPerBlock: number[]; // tokens per block observations
  draftTokens: number;
  acceptedTokens: number;
  drafts: number;
  cacheHits: Record<string, number>;
  cacheMisses: Record<string, number>;
  cacheEvictions: number;
  cacheBytes: number;
  cacheEntries: number;
  reused: number;
  forwarded: number;
  gpuAllocated: number;
  gpuReserved: number;
  unifiedFree: number;
  rss: number;
  ledgerRows: number;
  ledgerDropped: number;
  logSubscribers: number;
  httpByRoute: Record<string, number>;
}

export const TTFT_BUCKETS = [0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0, 60.0];
export const ACCEPT_BUCKETS = Array.from({ length: 17 }, (_, i) => i + 1);
export const DECODE_TPS_BUCKETS = [5, 10, 20, 30, 40, 50, 60, 70, 80, 100, 120, 150, 200];

export function initialState(rows: LedgerRow[], startedAt: number): LiveEngineState {
  const st: LiveEngineState = {
    startedAt,
    running: 0,
    waiting: 0,
    generationTokens: 0,
    promptTokens: 0,
    cachedPromptTokens: 0,
    reasoningTokens: 0,
    requestsByFinish: {},
    refusedByReason: {},
    errorsByType: {},
    ttft: [],
    acceptPerBlock: [],
    draftTokens: 0,
    acceptedTokens: 0,
    drafts: 0,
    cacheHits: { 'state,session': 0, 'state,prefix': 0, response: 0 },
    cacheMisses: { state: 0, response: 0 },
    cacheEvictions: 37,
    cacheBytes: 6.1e9,
    cacheEntries: 14,
    reused: 0,
    forwarded: 0,
    gpuAllocated: 61.2e9,
    gpuReserved: 63.9e9,
    unifiedFree: 42.1e9,
    rss: 4.4e9,
    ledgerRows: rows.length,
    ledgerDropped: 0,
    logSubscribers: 0,
    httpByRoute: { chat: 0, completions: 0, models: 3, health: 120, metrics: 240, cache: 2, dashboard: 0, static: 0, other: 1 },
  };
  // The counters since "start": the rows of the current process (last ~6 h of rows).
  const since = startedAt;
  for (const r of rows) if (r.ts_ms >= since) applyRow(st, r);
  return st;
}

export function applyRow(st: LiveEngineState, r: LedgerRow): void {
  st.httpByRoute[r.endpoint === 'chat' ? 'chat' : 'completions'] = (st.httpByRoute[r.endpoint === 'chat' ? 'chat' : 'completions'] ?? 0) + 1;
  if (r.finish_reason === 'refused') {
    const reason = r.status === 429 ? 'rate' : 'queue_full';
    st.refusedByReason[reason] = (st.refusedByReason[reason] ?? 0) + 1;
    return;
  }
  st.requestsByFinish[r.finish_reason ?? 'stop'] = (st.requestsByFinish[r.finish_reason ?? 'stop'] ?? 0) + 1;
  if (r.error_type) st.errorsByType[r.error_type] = (st.errorsByType[r.error_type] ?? 0) + 1;
  st.generationTokens += r.completion_tokens ?? 0;
  st.promptTokens += r.prompt_tokens ?? 0;
  st.cachedPromptTokens += r.cached_tokens ?? 0;
  st.reasoningTokens += r.reasoning_tokens ?? 0;
  if (r.ttft_ms != null) push(st.ttft, r.ttft_ms / 1000);
  if (r.blocks && r.completion_tokens && r.cache_source !== 'response') {
    const tpb = (r.completion_tokens - 1) / r.blocks;
    for (let i = 0; i < Math.min(r.blocks, 40); i++) push(st.acceptPerBlock, tpb, 4000);
    st.drafts += r.blocks;
    st.draftTokens += r.draft_tokens ?? 0;
    st.acceptedTokens += r.draft_accepted ?? 0;
  }
  if (r.cache_source === 'session') st.cacheHits['state,session']++;
  else if (r.cache_source === 'prefix') st.cacheHits['state,prefix']++;
  else if (r.cache_source === 'response') st.cacheHits.response++;
  else st.cacheMisses.state++;
  st.reused += r.cached_tokens ?? 0;
  st.forwarded += (r.prompt_tokens ?? 0) - (r.cached_tokens ?? 0);
  st.ledgerRows++;
}

function push(arr: number[], v: number, cap = 2000): void {
  arr.push(v);
  if (arr.length > cap) arr.splice(0, arr.length - cap);
}

function hist(name: string, help: string, buckets: number[], values: number[]): string {
  const lines = [`# HELP ${name} ${help}`, `# TYPE ${name} histogram`];
  let cum = 0;
  const sorted = [...values].sort((a, b) => a - b);
  let i = 0;
  for (const ub of buckets) {
    while (i < sorted.length && sorted[i] <= ub) {
      cum++;
      i++;
    }
    lines.push(`${name}_bucket{le="${fmt(ub)}"} ${cum}`);
  }
  lines.push(`${name}_bucket{le="+Inf"} ${sorted.length}`);
  lines.push(`${name}_sum ${sorted.reduce((a, b) => a + b, 0).toFixed(4)}`);
  lines.push(`${name}_count ${sorted.length}`);
  return lines.join('\n');
}

function fmt(n: number): string {
  return Number.isInteger(n) ? n.toFixed(1).replace(/\.0$/, n >= 1 ? '.0' : '') : String(n);
}

function counter(name: string, help: string, entries: [string, number][]): string {
  const lines = [`# HELP ${name} ${help}`, `# TYPE ${name} counter`];
  for (const [labels, v] of entries) lines.push(`${name}${labels ? `{${labels}}` : ''} ${v}`);
  return lines.join('\n');
}

function gauge(name: string, help: string, v: number | string, labels = ''): string {
  return `# HELP ${name} ${help}\n# TYPE ${name} gauge\n${name}${labels ? `{${labels}}` : ''} ${v}`;
}

export function renderMetrics(st: LiveEngineState, now: number, opts: { spec?: boolean } = {}): string {
  const spec = opts.spec ?? true;
  const uptime = (now - st.startedAt) / 1000;
  const out: string[] = [];
  out.push(
    counter(
      'qse_requests_total',
      'generations that finished, by the reason they finished',
      Object.entries(st.requestsByFinish)
        .sort()
        .map(([k, v]) => [`finish_reason="${k}"`, v]),
    ),
  );
  out.push(counter('qse_request_success_total', 'the stop and length rows', [['', (st.requestsByFinish.stop ?? 0) + (st.requestsByFinish.length ?? 0)]]));
  out.push(
    counter(
      'qse_requests_refused_total',
      'requests turned away with a Retry-After',
      Object.entries(st.refusedByReason).map(([k, v]) => [`reason="${k}"`, v]),
    ),
  );
  out.push(gauge('qse_requests_running', 'generations the engine is working on', st.running));
  out.push(gauge('qse_requests_waiting', 'callers waiting for the engine lock', st.waiting));
  out.push(counter('qse_prompt_tokens_total', 'prompt tokens accepted', [['', st.promptTokens]]));
  out.push(counter('qse_prompt_tokens_cached_total', 'prompt tokens served from a cache', [['', st.cachedPromptTokens]]));
  out.push(counter('qse_generation_tokens_total', 'tokens the engine wrote', [['', st.generationTokens]]));
  out.push(counter('qse_reasoning_tokens_total', 'tokens inside reasoning blocks', [['', st.reasoningTokens]]));
  out.push(
    counter(
      'qse_errors_total',
      'exceptions that ended a generation',
      Object.entries(st.errorsByType).map(([k, v]) => [`type="${k}"`, v]),
    ),
  );
  out.push(
    counter(
      'qse_http_requests_total',
      'HTTP requests by route and status',
      Object.entries(st.httpByRoute).map(([k, v]) => [`route="${k}",code="200"`, v]),
    ),
  );
  out.push(hist('qse_time_to_first_token_seconds', 'arrival of the request to its first token, queue wait included', TTFT_BUCKETS, st.ttft));
  if (spec) {
    out.push(counter('qse_spec_decode_num_drafts_total', 'blocks the drafter proposed', [['', st.drafts]]));
    out.push(counter('qse_spec_decode_num_draft_tokens_total', 'tokens proposed, the anchor excluded', [['', st.draftTokens]]));
    out.push(counter('qse_spec_decode_num_accepted_tokens_total', 'drafted tokens the target agreed with', [['', st.acceptedTokens]]));
    out.push(hist('qse_spec_accept_per_block', "tokens one block committed, the target's own token included", ACCEPT_BUCKETS, st.acceptPerBlock));
  }
  out.push(
    counter(
      'qse_cache_hits_total',
      'prefix lookups answered from a cache',
      [
        ['cache="state",kind="session"', st.cacheHits['state,session']],
        ['cache="state",kind="prefix"', st.cacheHits['state,prefix']],
        ['cache="response"', st.cacheHits.response],
      ],
    ),
  );
  out.push(
    counter('qse_cache_misses_total', 'lookups that found nothing', [
      ['cache="state"', st.cacheMisses.state],
      ['cache="response"', st.cacheMisses.response],
    ]),
  );
  out.push(counter('qse_cache_evictions_total', 'entries dropped to stay inside the byte budget', [['cache="state"', st.cacheEvictions]]));
  out.push(gauge('qse_cache_bytes', 'bytes held', Math.round(st.cacheBytes), 'cache="state"'));
  out.push(gauge('qse_cache_entries', 'entries held', st.cacheEntries, 'cache="state"'));
  out.push(gauge('qse_state_store_budget_bytes', 'byte budget of the state store', 8e9));
  out.push(counter('qse_prefill_tokens_reused_total', 'prompt tokens restored from a snapshot', [['', st.reused]]));
  out.push(counter('qse_prefill_tokens_forwarded_total', 'prompt tokens run through the 64 layers', [['', st.forwarded]]));
  out.push(gauge('qse_suffix_store_tokens', 'tokens in the suffix store', 1_284_211));
  out.push(gauge('qse_gpu_memory_used_bytes', 'torch.cuda.memory_allocated()', Math.round(st.gpuAllocated)));
  out.push(gauge('qse_gpu_memory_reserved_bytes', 'what the allocator took from the driver', Math.round(st.gpuReserved)));
  out.push(gauge('qse_unified_memory_free_bytes', "free bytes in the board's one pool", Math.round(st.unifiedFree)));
  out.push(gauge('qse_process_resident_memory_bytes', 'RSS of the server process', Math.round(st.rss)));
  out.push(gauge('qse_engine_uptime_seconds', 'seconds since the server started', uptime.toFixed(1)));
  out.push(gauge('qse_engine_start_time_seconds', 'unix time the server started', (st.startedAt / 1000).toFixed(0)));
  out.push(gauge('qse_log_subscribers', 'open log stream subscribers', st.logSubscribers));
  out.push(counter('qse_usage_ledger_rows_total', 'rows written to the usage ledger', [['', st.ledgerRows]]));
  out.push(counter('qse_usage_ledger_dropped_total', 'rows dropped because the write queue was full', [['', st.ledgerDropped]]));
  out.push(
    gauge(
      'qse_engine_info',
      'always 1; the labels are the configuration',
      1,
      'version="0.1.0-rc4",model="qwen38-spark-engine",drafter="LengthRouter",width="8",tree="true",nvfp4="true",fp8_head="true",contract="0.2.0"',
    ),
  );
  out.push(
    gauge(
      'qse_build_info',
      'the build',
      1,
      'version="0.1.0-rc4",git_sha="7d0b176",code_sha256="4c1f0e9a7b2d5e8f3a6c9d0b1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f",flags_sha256="9e8d7c6b5a4f3e2d1c0b9a8f7e6d5c4b3a2f1e0d9c8b7a6f5e4d3c2b1a0f9e8d"',
    ),
  );
  return out.join('\n') + '\n';
}
