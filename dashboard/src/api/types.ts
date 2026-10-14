// Dashboard API contract v1 — the TypeScript view of docs/contract/dashboard-v1/*.schema.json.
// Prose: docs/contract/dashboard-v1/README.md. A change here bumps contract_version.

export type ContractVersion = '1.0' | '1.1' | '1.2' | '1.3';

export interface Totals {
  requests: number;
  errors: number;
  refused: number;
  prompt_tokens: number;
  cached_tokens: number;
  completion_tokens: number;
  reasoning_tokens: number;
  total_tokens: number;
  tool_call_requests: number;
  thinking_requests: number;
  decode_tps_p50: number | null;
  decode_tps_p90: number | null;
  ttft_ms_p50: number | null;
  ttft_ms_p90: number | null;
  prefill_tps_p50: number | null;
  tokens_per_block_mean: number | null;
  draft_acceptance: number | null;
  active_days: number;
  /** 1.3: micro-dollars at the models' prices; null when no row had a price */
  cost_micros?: number | null;
}

export type LiveStatus = 'ok' | 'busy' | 'draining' | 'offline';

export interface Summary {
  contract_version: ContractVersion;
  generated_at: string;
  tz: string;
  windows: { today: Totals; '7d': Totals; '30d': Totals; '365d': Totals; all: Totals };
  streak: { current_days: number; longest_days: number };
  ledger: { enabled: boolean; since: string | null; rows: number; bytes: number; retention_days: number | null };
  live: { status: LiveStatus; running: number; waiting: number; uptime_s: number; last_request_at: string | null; conductor?: MonitorModeBlock | null; backends?: BackendStatus[] };
}

/** 1.3, the proxy: one configured backend's state (summary.live, system.proxy). */
export interface BackendStatus {
  id: string;
  kind: string;
  label: string | null;
  enabled?: boolean;
  cloud?: boolean;
  state: 'up' | 'down' | 'unknown' | 'disabled';
  since?: string | null;
  last_error?: string | null;
  latency_ms?: number | null;
  in_flight: number;
  max_inflight?: number | null;
  models?: number;
  harness_mode?: string;
  cost_today_micros?: number | null;
  scrape_age_s?: number | null;
  plugin_age_s?: number | null;
  /** P2, P3: the backend's own engine read at L1 (vllm, sglang); null for every other kind */
  engine?: EngineBlock | null;
}

/** p50 and p90 of one request phase over the window, from the engine's own histogram */
export interface PhaseSplit {
  p50: number | null;
  p90: number | null;
  count: number;
}

/** The figures of a vLLM or SGLang backend over the last 5 minutes (proxy/engines.py); null where
 *  the engine's page does not give the number, never 0. */
export interface EngineFigures {
  running: number | null;
  waiting: number | null;
  kv_usage: number | null;
  kv_tokens?: number | null;
  kv_capacity_tokens?: number | null;
  awake?: boolean | null;
  decode_tps: number | null;
  prompt_tps?: number | null;
  engine_gen_tps?: number | null;
  cached_share?: number | null;
  prefix_hit_rate: number | null;
  engine_cache_hit_rate?: number | null;
  ttft_p50_s: number | null;
  queue_s: PhaseSplit | null;
  prefill_s: PhaseSplit | null;
  decode_s: PhaseSplit | null;
  e2e_s: PhaseSplit | null;
  preemptions: number | null;
  finished: number | null;
  window_s?: number | null;
  spec: { source: 'counters' | 'engine_window'; acceptance: number | null; tokens_per_step: number | null; per_position?: (number | null)[] | null; drafts?: number | null } | null;
}

export interface EngineInfo {
  version: string | null;
  context_length: number | null;
  speculative: Record<string, unknown> | null;
  cache_report: boolean | null;
  reasoning_parser: string | null;
  metrics_enabled: boolean | null;
}

/** P6: the plugin inside a vLLM or SGLang backend (tandemllm-*-plugin, proxy/collect_api.py) */
export interface PluginBlock {
  /** l2 while it posts; l1_only after 60 s without a batch (an idle plugin posts every 10 s) */
  state: 'l2' | 'l1_only';
  age_s: number;
  via?: 'push' | 'pull';
  plugin: string | null;
  plugin_version: string | null;
  engine: string | null;
  engine_version: string | null;
  /** ledger rows that got the engine's own numbers, by request id or by time and counts */
  joined: number;
  approximate: number;
  unmatched?: number;
  untagged?: number;
  dropped?: { requests?: number; steps?: number };
  last?: Record<string, number> | null;
  window?: {
    samples: number;
    preempted: number;
    evictions: number;
    prefix_hit_rate: number | null;
    spec: { acceptance: number | null; tokens_per_step: number | null; per_position: (number | null)[] | null } | null;
  } | null;
}

/** P2, P3: a backend's engine at layer L1; P6: L2 while its plugin posts */
export interface EngineBlock {
  layer: 'L1' | 'L2';
  plugin?: PluginBlock | null;
  metrics: 'ok' | 'off' | 'error' | 'unknown';
  error?: string | null;
  scrape_age_s?: number | null;
  version?: string | null;
  verified?: string | null;
  info?: EngineInfo | null;
  figures: EngineFigures | null;
}

/** 1.3: the backend a request went to (`engine` in filters: the engine's own rows). */
export interface BackendRef {
  id: string;
  kind: string | null;
  label: string | null;
}

/** VIS-44: the session monitor's mode (`--conductor`; raw reads off), absent on a server without
 *  a monitor policy, so an older server shows no indicator. */
export type MonitorMode = 'off' | 'shadow' | 'on';
export interface MonitorModeBlock {
  mode: MonitorMode;
}

/** VIS-44: the conductor on a finished request (the ledger's columns and the journal); null when
 *  it did not run on it, absent on an older server. */
export interface RequestConductor {
  mode: string | null;
  acts: number;
  shadow: number;
  hidden_tokens: number;
  acted: boolean;
}

export interface UsageBucket {
  start: string;
  requests: number;
  errors: number;
  prompt_tokens: number;
  cached_tokens: number;
  completion_tokens: number;
  reasoning_tokens: number;
  total_tokens: number;
  tool_call_requests: number;
  decode_tps_p50: number | null;
  decode_tps_p90: number | null;
  ttft_ms_p50: number | null;
  ttft_ms_p90: number | null;
  prefill_tps_p50: number | null;
  tokens_per_block_mean: number | null;
  draft_acceptance: number | null;
  cost_micros?: number | null;
}

export interface ClientDim {
  id: string;
  label: string | null;
  kind: ClientKind;
}

/** `internal` (VIS-60): TandemLLM's own calls to itself (the router's question); left out of every view unless asked */
export type ClientKind = 'open-webui' | 'openai-sdk' | 'opencode' | 'hermes' | 'curl' | 'dashboard' | 'chat-app' | 'other' | 'internal';

export interface TopDay {
  date: string;
  total_tokens: number;
  requests: number;
  top_client: string | null;
}

export type Bucket = 'day' | 'hour';

export interface Usage {
  contract_version: ContractVersion;
  from: string;
  to: string;
  bucket: Bucket;
  tz: string;
  filters: { model: string | null; client: string | null; backend?: string | null; user?: string | null };
  buckets: UsageBucket[];
  totals: Totals;
  top_days: TopDay[];
  dimensions: { models: string[]; clients: ClientDim[]; backends?: BackendRef[]; users?: string[] };
}

export type FinishReason = 'stop' | 'length' | 'tool_calls' | 'content_filter' | 'timeout' | 'abandoned' | 'error' | 'refused';
/** `resident`: the resident prompt's pinned head or a kept anchor answered the prefix (turncache). */
export type CacheSource = 'response' | 'session' | 'prefix' | 'resident' | 'disk' | 'spill' | 'end' | 'continuation' | 'prepared' | 'none';
export type Endpoint = 'chat' | 'completions' | 'responses' | 'embeddings';

export interface RequestRow {
  id: number;
  request_id: string;
  ts: string;
  model: string;
  client: ClientDim;
  endpoint: Endpoint;
  stream: boolean;
  status: number;
  finish_reason: FinishReason | null;
  prompt_tokens: number | null;
  cached_tokens: number | null;
  completion_tokens: number | null;
  reasoning_tokens: number | null;
  queue_ms: number | null;
  prompt_ms: number | null;
  ttft_ms: number | null;
  decode_ms: number | null;
  total_ms: number | null;
  decode_tps: number | null;
  prefill_tps: number | null;
  blocks: number | null;
  /** VIS-60: an internal row's request (the turn the router's question served) */
  parent_request?: string | null;
  tokens_per_block: number | null;
  draft_tokens: number | null;
  draft_accepted: number | null;
  tool_calls: number;
  thinking: boolean;
  cache_source: CacheSource | null;
  error_type: string | null;
  conductor?: RequestConductor | null;
  /** served in the side lane between another request's blocks or prefill chunks */
  lane?: SideLane | null;
  /** SRV-197: the engine's sequence slot with parallel requests (0 main, 1+ side), else null */
  slot?: number | null;
  /** 1.3, the proxy: the backend that answered; null for the engine's own request */
  backend?: BackendRef | null;
  user?: string | null;
  /** where the timings come from: the engine's loop, or the proxy (TTFT holds the network) */
  measured_at?: 'engine' | 'proxy';
  backend_model?: string | null;
  usage_source?: 'backend' | 'none' | null;
  cache_write_tokens?: number | null;
  cost_micros?: number | null;
  proxy_ms?: number | null;
  retries?: number | null;
  fallback_from?: string | null;
  upstream_status?: number | null;
  /** P6: the queue, prefill and decode split and the cached tokens came from the engine's plugin */
  l2?: { join: 'exact' | 'approximate'; preemptions?: number | null; engine_finish?: string | null; corrupted?: boolean | null } | null;
}

/** Where a side-lane request was served (server/preempt.py): at a safe point of another
 * request's decode or prefill, or under the lock with none running; `pending` while it waits. */
export type SideLane = 'decode' | 'prefill' | 'self' | 'pending';

export interface SideLaneReport {
  where: 'on' | 'decode';
  prompt_max: number;
  new_max: number;
  waiting: number;
  jobs: number;
  served_decode: number;
  served_prefill: number;
  served_self: number;
  cancelled: number;
  errors: number;
  lent_ms: number;
}

export interface Requests {
  contract_version: ContractVersion;
  next_before: number | null;
  requests: RequestRow[];
}

export interface SystemInfo {
  contract_version: ContractVersion;
  generated_at: string;
  engine: {
    version: string;
    git_sha: string;
    code_sha256: string;
    started_at: string;
    uptime_s: number;
    pid: number;
    /** null on the standalone proxy (no engine) */
    model: string | null;
    /** the base engine's own name; `model` is TandemLLM when the conductor is configured */
    base_model?: string | null;
    max_len: number | null;
    /** the weight profile (VIS-95: the pack, its branch and its largest window) */
    weights?: WeightsInfo | null;
    drafter: string | null;
    tree: boolean;
    reasoning_format: string | null;
    reasoning_effort: string | null;
    status: LiveStatus;
  };
  flags: { args: Record<string, unknown>; env: Record<string, string> };
  memory: {
    gpu_allocated_bytes: number | null;
    gpu_reserved_bytes: number | null;
    gpu_max_allocated_bytes: number | null;
    unified_total_bytes: number | null;
    unified_available_bytes: number | null;
    process_rss_bytes: number | null;
    split?: MemorySplit | null; // OPS-41
    limits?: MemoryLimits | null; // OPS-47
    roles?: RolesReport | null; // 0.6.1: the small models under memory pressure (server/rolemem.py)
  };
  gpu: {
    name: string | null;
    temperature_c: number | null;
    power_w: number | null;
    sm_clock_mhz: number | null;
    utilization: number | null;
    source: string | null;
    sampled_at: string | null;
  };
  caches: Record<string, unknown> | null;
  /** 1.3: the proxy on this server */
  /** `off` (VIS-54): the engine without --backends; /v1/config is a 404 there */
  proxy?: { mode: 'embedded' | 'standalone' | 'off'; backends: BackendStatus[] } | null;
  queue: {
    running: number;
    waiting: number;
    max_queue: number | null;
    queue_timeout_s: number | null;
    request_timeout_s: number | null;
    /** short requests in the side lane now, and the lane's counts (null: off) */
    lane?: number;
    side_lane?: SideLaneReport | null;
  };
  inflight: { served: number; refused: number; errors: number; timeouts: number; abandoned: number };
  ledger: { enabled: boolean; rows: number; bytes: number; oldest: string | null; queue: number; dropped: number };
  disk: { state_dir_free_bytes: number | null };
  /** 1.3, P1.5: the machines behind the backends; this machine first. Absent on older servers
   *  and with --host-collector off. */
  hosts?: HostInfo[];
}

/** P1.5: one process tree behind a backend's port (or the server's own, backend engine/proxy) */
export interface HostProcess {
  backend: string | null;
  label: string | null;
  port: number | null;
  pid: number | null;
  found: boolean;
  via: 'self' | 'psutil' | 'proc' | 'lsof' | 'ss' | null;
  name: string | null;
  cmdline: string | null;
  rss_bytes: number | null;
  threads: number | null;
  open_files: number | null;
  uptime_s: number | null;
  /** cores over the last scan */
  cpu: number | null;
  gpu_memory_bytes: number | null;
  /** measured: bytes from NVML/nvidia-smi; unified: listed, no figure; not_reported: no GPU
   *  source; none: not on the GPU */
  gpu_memory: 'measured' | 'unified' | 'not_reported' | 'none';
  children: { pid: number; name: string | null; rss_bytes: number | null; threads: number | null; cpu: number | null; gpu_memory_bytes: number | null }[];
  total_rss_bytes: number | null;
  total_gpu_memory_bytes: number | null;
  note: string | null;
}

export interface HostGpu {
  index: number | null;
  name: string | null;
  uuid: string | null;
  utilization: number | null;
  memory_used_bytes: number | null;
  memory_total_bytes: number | null;
  /** GB10: one pool with the CPU, no memory figure of the GPU's own */
  memory_unified: boolean;
  power_w: number | null;
  power_limit_w: number | null;
  temperature_c: number | null;
  sm_clock_mhz: number | null;
  mem_clock_mhz: number | null;
  source: 'nvml' | 'nvidia-smi' | 'dcgm';
}

export interface HostInfo {
  id: string;
  label: string | null;
  source: 'local' | 'collector' | 'exporter';
  url: string | null;
  backends: string[];
  state: 'ok' | 'pending' | 'stale' | 'error';
  error: string | null;
  memory: {
    total_bytes: number | null;
    available_bytes: number | null;
    used_bytes: number | null;
    swap_total_bytes: number | null;
    swap_used_bytes: number | null;
    unified: boolean | null;
  } | null;
  cpu: { count: number | null; utilization: number | null; load: (number | null)[]; temperature_c: number | null } | null;
  gpus: HostGpu[];
  gpu_source: 'nvml' | 'nvidia-smi' | 'dcgm' | null;
  disks: { name: string; nvme: boolean; read_bps: number | null; write_bps: number | null; busy: number | null }[];
  fs: { path: string; total_bytes: number; free_bytes: number }[];
  net: { name: string; rx_bps: number | null; tx_bps: number | null }[];
  /** null where the source has no per-process figures (node_exporter + DCGM) */
  processes: HostProcess[] | null;
  sampled_at: string | null;
  age_s: number | null;
  cost: { samples: number; scans: number; cpu_ms_total: number; last_ms: number | null; last_scan_ms: number | null; fast_period_s: number; scan_period_s: number } | null;
}

export type LogLevel = 'debug' | 'info' | 'warning' | 'error';
// the line's bracket tag -- an open set on the engine's side (req, server, cache, drafter, think,
// ledger, body, http, ...; stdout / stderr for an untagged line)
export type LogSource = string;

export interface LogLine {
  seq: number;
  ts: string;
  level: LogLevel;
  source: LogSource;
  msg: string;
  request_id: string | null;
}

export interface LogsJson {
  contract_version: ContractVersion;
  lines: LogLine[];
  last_seq: number;
}

/** GET /v1/dashboard/session (1.1). `login: false`: the server runs with the dashboard login off
 *  or with no admin password, there is nothing to sign in to or out of, and `expires_at` is null.
 *  `password_set` (AU1, absent on older servers): an admin password is set on the server. */
export interface SessionInfo {
  contract_version: '1.1';
  authenticated: boolean;
  login: boolean;
  password_set?: boolean;
  expires_at: string | null;
}

export interface ApiError {
  error: { type: 'unauthorized' | 'bad_request' | 'not_found' | 'too_many'; message: string };
}

// The per-response usage block of (the finish chunk / the include_usage chunk).
export interface ChatUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  /** SRV-165: `hidden_tokens`, the prompt tokens TandemLLM added (its notes); absent when none */
  prompt_tokens_details?: { cached_tokens: number; hidden_tokens?: number };
  completion_tokens_details?: { reasoning_tokens: number };
}
export interface ChatTimings {
  cache_n: number;
  prompt_n: number;
  prompt_ms: number;
  prompt_per_token_ms: number;
  prompt_per_second: number;
  predicted_n: number;
  predicted_ms: number;
  predicted_per_token_ms: number;
  predicted_per_second: number;
  draft_n?: number;
  draft_n_accepted?: number;
  ttft_ms: number;
  queue_ms: number;
  total_ms: number;
  blocks: number;
  tokens_per_block: number;
  reasoning_n: number;
  cache_source: CacheSource;
}
export interface ChatMetrics {
  time_to_first_token_ms: number;
  generation_time_ms: number;
  queue_time_ms: number;
  mean_itl_ms: number;
  tokens_per_second: number;
  speculative_decoding?: { mean_acceptance_length: number; draft_acceptance_rate: number };
}

// The live view: GET /v1/dashboard/live?follow=0, and the data of one `event: live`.
// Definitions in docs/contract/dashboard-v1 (live.schema.json).
export type LivePhase = 'queued' | 'prefill' | 'decode' | 'done';

export interface LiveSample {
  t: number; // unix seconds
  decode_tps: number | null; // tokens over this second, all requests; null when nothing decoded
  prefill_tps: number | null; // a prefill that finished in this second, else null
  running: number;
  waiting: number;
  tokens: number; // the running total the deltas come from
  // VIS-40 (1.1, absent on older servers): the live rates without the warm-up
  decode_live_tps?: number | null; // each request's tokens over this second, from its anchor on
  prefill_live_tps?: number | null; // the in-flight prefills' now-rate (a mostly cached one left out)
}

/** VIS-40: the settings of the live rates, `QSE_LIVE_*` on the server. */
export interface LiveWindows {
  decode_warmup_tokens: number; // a request's live decode rate starts at this many decoded tokens
  prefill_warmup_chunks: number; // the live prefill rate starts after this many chunks
  now_window_s: number;
  mostly_cached_pct: number;
  mostly_cached_tokens: number;
}

export interface LiveRequest {
  request_id: string;
  phase: LivePhase;
  finish_reason: FinishReason | null;
  status: number;
  model: string;
  client: { id: string; kind: ClientKind };
  endpoint: Endpoint;
  stream: boolean;
  thinking: boolean;
  temperature: number | null;
  prompt_tokens: number | null;
  cached_tokens: number | null;
  /** VIS-58 (absent on older servers): the prompt tokens TandemLLM added (its notes); `prompt_tokens` is the client's count */
  hidden_tokens?: number | null;
  forwarded_tokens: number | null;
  tokens: number;
  blocks: number | null;
  tokens_per_block: number | null;
  elapsed_ms: number | null;
  queue_ms: number | null;
  prompt_ms: number | null;
  ttft_ms: number | null;
  decode_ms: number | null;
  prefill_tps: number | null;
  decode_tps: number | null; // running average since the first token; the final number when done
  decode_tps_now: number | null; // the last ~2 s
  lane?: SideLane | null;
  cache_source: CacheSource | null;
  max_tokens: number | null;
  ended_ms_ago: number | null;
  // VIS-40 (1.1, absent on older servers): the decode rates from token 50 on, and the cache share
  decode_live_now_tps?: number | null; // the last ~2 s, never before the anchor; null when done
  decode_live_avg_tps?: number | null; // from the anchor to now (to the last token when done)
  decode_warmup?: { tokens: number; reached: boolean } | null; // decoded tokens so far; null before the first
  mostly_cached?: boolean | null;
  // narrator v3 (absent on older servers, null without state lines): the newest line the client
  // was shown, the voice's "what Qwen is doing right now"
  line?: LiveLine | null;
  // contract 1.1: absent on a 1.0 server, null with `--live-activity off`
  activity?: LiveActivity | null;
  timeline?: LiveTimelineEntry[] | null;
}

/** One state line as the client saw it (narrator v3). */
export interface LiveLine {
  kind: string;
  text: string;
  at_ms: number;
}

// ---- contract 1.1: what the model is doing right now ------------------
// Definitions in docs/contract/dashboard-v1 (live.schema.json, contract 1.1); the words (`label`,
// `sentence`) come from server/activity.py and are shown as sent.
export type ActivityState = 'queued' | 'prefilling' | 'replaying' | 'thinking' | 'closing_reasoning' | 'writing' | 'tool_call' | 'finishing' | 'done';
export type EngineState = 'idle' | 'busy' | 'waiting_for_client' | 'draining' | 'starting';
export type StopReason = 'stop' | 'length' | 'tool_calls' | 'timeout' | 'abandoned' | 'error' | 'refused' | 'rejected' | 'cancelled';
export type FinishingStep = 'flush' | 'saving_state' | 'final_chunk';

export interface LiveStop {
  reason: StopReason;
  detail: string | null; // eos, stop_string, pattern_guard, the error type, queue_full, ...
  state: ActivityState; // the state it happened in
  tokens_sent: number;
  silent_ms: number | null; // since the last token went out (or since the lock, with none)
  client_gone_ms: number | null; // how long before the handler noticed the client had left
  sentence: string; // "abandoned by the client after 73 s of silent prefill, 0 tokens sent"
}

export interface LiveActivity {
  state: ActivityState;
  label: string; // the server's short phrase: "Prefilling 36,864 of 48,210 (76 %)"
  since_ms: number | null; // time in this state
  constrained: 'response_format' | 'tool_choice' | null;
  queue: { place: number | null; wait_ms: number | null; timeout_s: number | null; place_is_estimate: boolean; /** 0.8.2: why it waits, in words; absent on older servers */ reason?: string | null } | null;
  prefill: {
    done: number | null;
    total: number;
    cached: number | null;
    pct: number | null;
    tps_now: number | null;
    tps_avg: number | null;
    eta_ms: number | null;
    progress: 'chunked' | 'single_call' | null;
    at_ms: number | null;
    // VIS-40 (absent on older servers): the rate after the warm-up chunk(s), now and on average
    tps_live?: number | null;
    tps_live_avg?: number | null;
    warmup?: { chunks: number; reached: boolean };
  } | null;
  decode: {
    tokens: number;
    thinking_tokens: number;
    content_tokens: number;
    tool_tokens: number;
    tps_now: number | null;
    tps_avg: number | null;
    rounds: number | null;
    tokens_per_round: number | null;
    tokens_per_round_now: number | null;
    ms_per_round: number | null;
    ms_per_round_now: number | null;
    accept_mean: number | null;
  } | null;
  tool: { index: number; name: string | null; arg_bytes: number | null; calls_done: number } | null;
  reasoning: { closed_by: 'model' | 'budget' | 'stall' | null; tokens: number | null } | null;
  client: { connected: boolean | null; silent_ms: number | null; gone_ms: number | null };
  continues: { request_id: string; gap_ms: number | null; tool_names: string[]; inferred: boolean } | null;
  step: FinishingStep | null;
  stop: LiveStop | null;
  // live 1.2 (SRV-115, absent on older servers): the session monitor on this request; the label
  // says "Supervising: ..." while a hidden forward runs (and a second after it)
  conductor?: LiveConductor | null;
}

/** live 1.2 (SRV-115); `shadow` and the decisions' `score`, `tier`, `at_tokens` are VIS-44's. */
export interface LiveConductor {
  mode: 'shadow' | 'on';
  acts: number;
  shadow?: number;
  hidden_tokens: number;
  gave_up: boolean;
  supervising: boolean;
  action: { task: string; action: string; reason: string | null } | null;
  last: LiveDecision | null;
  last_action: LiveDecision | null;
}
export interface LiveDecision {
  task: string;
  decision: string;
  mode: string;
  reason: string | null;
  ago_ms: number | null;
  score?: number | null;
  tier?: number | null;
  at_tokens?: number | null;
}

export interface LiveTimelineEntry {
  t_ms: number; // from arrival
  state: ActivityState;
  detail?: string;
}

export interface LiveEngine {
  state: EngineState;
  label: string; // "Idle", "Calling tool write", "Waiting for client: running tool bash"
  since_ms: number | null;
  model: string | null;
  version: string | null;
  draining: boolean;
  kv: { length: number; max_len: number } | null;
  memory: { mem_available_gib: number | null; rss_gib: number | null } | null;
  store: { entries: number; bytes_gib: number } | null;
  waiting_for_client: { request_id: string; tool_names: string[]; since_ms: number | null; client_kind: string } | null;
  conductor?: MonitorModeBlock | null; // VIS-44
}

export interface LiveRecent {
  request_id: string;
  ended_at: string;
  client_kind: string;
  path: ActivityState[];
  tokens: number;
  elapsed_ms: number | null;
  ttft_ms: number | null;
  decode_tps: number | null;
  tool_names: string[];
  stop: LiveStop;
  lane?: SideLane | null;
  /** the prompt, what came out of a cache, what was forwarded and from where */
  prompt_tokens?: number | null;
  cached_tokens?: number | null;
  forwarded_tokens?: number | null;
  cache_source?: CacheSource | null;
}

export interface LiveCounts {
  in_flight: number;
  queued: number;
  prefilling: number;
  decoding: number;
  completed_1m: number;
  served: number;
  errors: number;
  refused: number;
}

export interface Live {
  contract_version: ContractVersion;
  seq?: number; // 1.1: rises by one per event, also the SSE id
  generated_at: string;
  interval_s: number;
  engine?: LiveEngine; // 1.1
  counts: LiveCounts;
  now: {
    decode_tps: number | null;
    prefill_tps: number | null; // the last FINISHED prefill's end-to-end rate (1.0)
    prefilling: boolean;
    tokens_per_block: number | null;
    last_prefill_ms_ago: number | null;
    // VIS-40 (absent on older servers): the speed now, without the warm-up
    prefill_live_tps?: number | null;
    prefill_live_count?: number;
    decode_live_tps?: number | null;
    decode_warming?: number;
  };
  windows?: LiveWindows; // VIS-40: present when the server sends the live rates
  requests: LiveRequest[];
  recent?: LiveRecent[] | null; // 1.1: the last 20 finished requests, newest first; null with --live-activity off
  sample: LiveSample | null;
  sampler?: { ticks: number; encodes: number; tick_us_last: number | null; tick_us_mean: number | null; tick_us_max: number | null; tick_cpu_us_mean: number | null; tick_cpu_us_max: number | null }; // 1.1
  history?: LiveSample[]; // the first event and follow=0 only
}

// ---- 0.6.1: the small models under memory pressure (server/rolemem.py) ------------------------

export type RoleState = 'loaded' | 'unloaded_for_memory' | 'reloading' | 'not_loaded' | 'loading' | 'unloading' | 'failed';
export interface RoleRow {
  models: string[];
  kind?: string;
  state: RoleState | string;
  claim_gib?: number;
  reload_at_gib?: number;
  out_since?: number | null; // unix seconds
  out_s?: number | null;
  why?: string | null;
  freed_gib?: number | null;
  unloads?: number;
  reloads?: number;
  hold_s?: number;
  refusal?: string | null;
  /** VIS-153: the load error of a failed role */
  error?: string | null;
  /** VIS-153: the planner's reason when its last load was refused */
  refused?: string | null;
}
export interface RolesReport {
  dynamic: boolean;
  order?: string[];
  reload_order?: string[];
  policy?: { floor_gib: number; margin_gib: number; hold_s: number; thrash_s: number; max_hold_s: number };
  mem_available_gib?: number | null;
  headroom_gib?: number | null;
  quiet_s?: number | null;
  roles: Record<string, RoleRow>;
  stats?: Record<string, number>;
  events?: { kind: string; ts: number; role?: string; freed_gib?: number; why?: string; reason?: string; out_s?: number }[];
  error?: string;
}

// ---- OPS-41: the engine's memory split and the residency planner -------------------------------

/** server/memsplit.py: the resident set in bytes, each part read off the object that holds it. */
/** /health's and /v1/dashboard/system's `weights` (the weight profile). */
export interface WeightsInfo {
  name: string | null;
  format: string | null;
  weights: string | null;
  hf_repo: string | null;
  /** a branch of the repo (a Flash-Next pack) */
  hf_revision?: string;
  drafters: string | null;
  draft_head: string | null;
  /** the registry's largest window for the profile; null: it fits none; absent: the profile names none */
  max_window?: number | null;
}

/** /health's `cache.resident_files`: files the engine keeps on purpose (Flash-Next's n-gram table, SRV-335). */
export interface ResidentFile {
  name: string;
  path: string;
  policy: 'warm' | 'cold';
  file_bytes: number;
  resident_bytes: number | null;
  cache_bytes?: number;
  hits?: number;
  misses?: number;
}

export interface MemorySplit {
  allocated: number | null;
  reserved: number | null;
  parts: Record<string, number>; // weights, kv_pool, gdn_state, drafter_kv, vision, state_store, resident_prefix, deciders
  other: number | null; // what the allocator holds beyond the named parts
  reserve: number | null; // what it keeps from the driver without a tensor in it
  mem_available: number | null;
  max_len: number | null;
  drafter_caches: number;
  /** 0.6.1: the parts held as KV pages the driver maps outside the allocator (the shared KV pool), committed bytes */
  vmm_parts?: string[];
  kv_committed?: number | null; // every slot's committed KV pages
  kv_reserved_va?: number | null; // the address range reserved for them: not memory
  mem_total?: number | null;
  in_use?: number | null; // MemTotal - MemAvailable
  outside?: number | null; // in_use - allocator reserved - KV pages: the CUDA context, host memory, other programs
  /** SRV-335: page cache the engine keeps on purpose (the warm n-gram table), counted as used by every floor check */
  pinned_cache?: number | null;
  load?: { stage: string; allocated: number | null; reserved: number | null; mem_available_drop: number | null; s: number }[];
}

export type ResidencyBasis = 'measured' | 'peak' | 'declared';
/** The conductor's `residency` document. */
export interface ResidencyDoc {
  floor_gib: number;
  margin_gib: number;
  settle_s: number;
  order: string[];
  claims: Record<string, { gib: number; basis: ResidencyBasis; note: string }>;
  /** OPS-47: the engine's memory limits the dashboard overrides, by their serve.env names */
  engine?: Record<string, number>;
}
export type ResidencyReason = 'refused_hold' | 'refused_load_in_progress' | 'refused_memory' | 'refused_unknown_role';
export interface ResidencyRow {
  role: string;
  allowed: boolean;
  reason: ResidencyReason | null;
  mem_available_gib: number | null;
  claim_gib: number | null;
  basis: string | null;
  floor_gib: number;
  margin_gib: number;
  need_gib: number;
  after_gib: number | null;
  note?: string | null;
  source?: string | null;
  /** OPS-47: the engine's declared peak, what it holds now, and what it may still take */
  engine_peak_gib?: number | null;
  engine_now_gib?: number | null;
  engine_growth_gib?: number;
}
export interface ResidencyJournalLine {
  kind: 'check' | 'loaded' | 'memory_shed' | 'memory_refused' | 'memory_estimate_raised';
  ts: number;
  role: string;
  reason?: string | null;
  allowed?: boolean;
  source?: string;
  took_gib?: number;
  claim_gib?: number | null;
  mem_available_gib?: number | null;
}
/** `GET /v1/conductor/residency/plan`. */
export interface ResidencyPlan {
  version: number;
  doc: ResidencyDoc;
  plan: { mem_available_gib: number | null; floor_gib: number; margin_gib: number; need_gib: number; spendable_gib: number | null; settle_s: number; order: string[]; rows: ResidencyRow[]; engine_peak_gib?: number | null; engine_now_gib?: number | null; engine_growth_gib?: number };
  engine: MemorySplit | null;
  limits?: MemoryLimits | null; // OPS-47
  last_refusal: ResidencyJournalLine | null;
  journal: ResidencyJournalLine[];
}

// ---- OPS-47: the engine's memory limits (server/memlimits.py) ----------------------------------

export type LimitUnit = 'GiB' | 'MiB' | 'tokens' | 'on/off';
/** One limit: its value and where it comes from, whether a change applies at the next request or
 *  needs a restart, and what it holds now (`used` of `cap`, bytes for GiB and MiB keys). */
export interface MemoryLimitRow {
  key: string;
  unit: LimitUnit;
  what: string;
  value: number | null;
  flag: number | null;
  source: 'serve.env' | 'dashboard';
  applies: 'next request' | 'restart' | 'locked';
  locked: string | null;
  running: number | null;
  pending_restart: boolean;
  used: number | null;
  cap: number | null;
}
export interface MemoryEvent {
  kind: 'shed' | 'refused' | 'empty' | 'estimate_raised' | 'admit_refused';
  ts: number;
  level?: string;
  why?: string;
  freed?: number;
  ms?: number;
  n?: number;
  need?: number;
  peak?: number;
  nbytes?: number;
  safe?: boolean;
  per_token?: number;
  was?: number;
  workspace?: number;
}
/** engine/cache.py Budget.report(). */
export interface MemoryBudget {
  ceiling: number;
  peak: number;
  slack: number;
  allocated: number;
  reserved: number;
  reserve: number;
  ws_floor: number;
  ws_per_token: number;
  events: MemoryEvent[];
  gates: number;
  refused: number;
  admitted: number;
  admit_refused: number;
  empties: number;
  empty_ms: number;
  empty_freed: number;
  last_empty_ms: number;
  shed: Record<string, number>;
  shed_bytes: Record<string, number>;
  prefill_peak: number;
  observed: { n: number; workspace: number; estimate: number } | null;
  last_gate: { n: number; workspace: number; allocated: number; reserved: number; shed: string[]; ts: number } | null;
  last_prefill_peak: { n: number; peak: number } | null;
}
/** A request that replaced another session's rows, and what it cost. */
export interface SessionSwitch {
  ts: number;
  from: string | null;
  to: string | null;
  tokens: number;
  reused: number;
  reread: number;
  via: string;
  ms: number;
}
export interface MemoryLimits {
  version: number | null;
  rows: MemoryLimitRow[];
  budget: MemoryBudget | null;
  workspace_at_max_len: number | null;
  switch: SessionSwitch | null;
  switches: number;
  park: { owner: string | null; rows: number; base: number; bytes: number } | null;
  owner: string | null;
}

// ---------------------------------------------------------------------------------------------
// The setup guide: /v1/setup/* (server/setup/, docs/setup.md). Contract 1.0, separate from the
// dashboard's. Every memory figure is GiB (2^30 bytes) like the engine's knobs, whatever the field
// is called (`gb`, `*_gb`, `*_gib`). The model payload is models/<id>/model.yaml parsed as JSON.
// ---------------------------------------------------------------------------------------------

export type SetupContract = '1.0';
export type SetupStepId = 'models' | 'overview' | 'weights' | 'components' | 'parallelism' | 'methods' | 'review' | 'install';
export type SetupValue = string | number | boolean;

/** GET /v1/setup/models: one tile per models/<id>/model.yaml. */
export interface SetupModelTile {
  id: string;
  name: string;
  family: string;
  params: string;
  pitch: string;
  recommended: boolean;
  icon: string | null;
  /** the fastest measured cell of the recommended profile (greedy, 512 tokens) */
  best_speed: { tok_s: number; task: string; profile: string } | null;
  /** the community packs for this model (absent on an older server) */
  community_count?: number;
}
export interface SetupModelsList {
  contract_version: SetupContract;
  models: SetupModelTile[];
  /** a model.yaml that did not load: its path and why (the other models still list) */
  errors: { path: string; message: string }[];
}

export interface SetupQuality {
  kl_vs_bf16: number | null;
  top1_pct: number | null;
  ppl: number | null;
  date: string;
  source: string;
}
/** One weight profile: one ops/profiles/<id>.env. */
export interface SetupProfile {
  id: string;
  label: string;
  format: 'exl3' | 'nvfp4' | 'bf16' | 'fp8' | 'gguf';
  bpw: number | null;
  /** download size */
  disk_gb: number;
  /** the target's weights on the device, measured at load */
  weights_gib: number;
  hf_repo: string;
  recommended: boolean;
  /** e.g. "best speed for quality", "closest to BF16" */
  badges: string[];
  quality: SetupQuality;
  note: string | null;
  /** 1.0+: the draft models, the draft head and their buffers on the device (the profile picks them) */
  drafters_gib: number;
  /** 1.0+: the weights' repo needs a Hugging Face token */
  hf_private: boolean;
  /** 1.0+: SetupAsset ids the profile needs besides its own weights */
  assets_required: string[];
  /** SRV-337: the repository branch of the weights */
  hf_revision?: string | null;
  /** SRV-337: the largest window that fits the device preset (null: none fits; absent: the model's max) */
  max_window?: number | null;
  /** SRV-337: a lookup table beside the weights, warm in the page cache or read past it through a row cache */
  table?: { gib: number; policy: 'warm' | 'cold'; cache_gib: number } | null;
  /** SRV-337: a tool's memory under this profile, by component id */
  components_gib?: Record<string, number>;
  /** page-side only (setup/community.ts withCommunityProfiles): the community entry this profile
   *  stands for, so the reducer, the estimate and Wren's select_profile treat a pack like one of
   *  ours; absent on our own profiles and in the API */
  community?: CommunityEntry;
}

// ---- community weights (CONTRACT.md 2026-10-13, "API (as built)") -----------------------------
export type SetupModality = 'text' | 'image' | 'audio' | 'video';
export type SetupCommunityStatus = 'submitted' | 'verified' | 'broken';
export type SetupCommunityDrafters = 'tandem-exl3' | 'dflash2' | 'none';
export type SetupSocialNetwork = 'x' | 'huggingface' | 'bluesky' | 'mastodon' | 'website';
/** One community pack as GET models/{id} lists it in `community` (newest first): the entry as
 *  written in models/<model>/community/<id>.json with every default filled in, plus what the server
 *  computes. Every string is untrusted text (fetched from GitHub): rendered as text, links only
 *  when https. */
export interface CommunityEntry {
  schema: 1;
  id: string;
  name: string;
  base_model: string;
  /** the HF id of the source model when it is not stock Qwen */
  finetune_of: string | null;
  author: { github: string; name: string | null; social: Partial<Record<SetupSocialNetwork, string>> };
  hf: { repo: string; revision: string };
  /** only exl3 passes validation today */
  format: 'exl3';
  bpw: number;
  /** download, 10^9 bytes */
  size_gb: number;
  load: 'standalone' | 'overlay';
  /** what works in TandemLLM today */
  modalities: Record<SetupModality, boolean>;
  /** what the pack contains (greyed when not supported yet) */
  modalities_in_pack: SetupModality[];
  drafters: SetupCommunityDrafters;
  license: string;
  license_url: string | null;
  description: string;
  tags: string[];
  added: string;
  status: SetupCommunityStatus;
  verified: null | {
    build: string;
    device: string;
    date: string;
    tok_s: Partial<Record<'prose' | 'code' | 'chat' | 'plain', number>>;
    tokens_per_round: Partial<Record<'prose' | 'code' | 'chat', number>>;
    notes: string | null;
  };
  /** https URLs, e.g. { pr } */
  links: Record<string, string>;
  // computed by the server
  /** "community/<id>": the selection's profile value */
  profile: string;
  /** https://github.com/<github>.png?size=96 */
  avatar_url: string;
  author_url: string;
  /** ready-made, in a fixed order, [] when none */
  social_links: { network: SetupSocialNetwork; label: string; url: string }[];
  /** https://huggingface.co/<repo>/tree/<revision> */
  hf_url: string;
  /** the estimate's numbers, from our own profile of the same format and nearest bpw (`like`) */
  memory: { weights_gib: number; drafters_gib: number; like: string };
}
/** Where the community list came from (the server fetches models/community-index.json from GitHub
 *  only when the weights section opens or Refresh is pressed; offline: the cache, else the copy
 *  bundled with the release). */
export interface CommunitySync {
  /** github: the fetch this server just did (200 or 304) succeeded */
  source: 'github' | 'cache' | 'bundled';
  /** null when QSE_COMMUNITY_URL=off */
  url: string | null;
  /** the last 200; null when bundled */
  fetched_at: string | null;
  /** the last attempt by this server; null: none yet */
  checked_at: string | null;
  /** why the last attempt failed (shown small, not fatal; one short line, no URL) */
  error: string | null;
  /** a quiet note that is no failure: "no published list yet" (the repo has no index: HTTP 404),
   *  "fetching is off (QSE_COMMUNITY_URL=off)"; absent on an older server */
  note?: string | null;
  /** entries that failed validation */
  dropped: { model: string; id: string; path: string; reason: string }[];
  /** "Add yours": the pre-filled GitHub issue form */
  add_url: string;
}
/** POST /v1/setup/community/sync {} (after one conditional fetch) and GET /v1/setup/community */
export interface CommunityBody {
  contract_version: SetupContract;
  /** `dropped` covers every model */
  sync: CommunitySync;
  /** model id -> entries, newest first */
  models: Record<string, CommunityEntry[]>;
}

/** 1.0+: a file a profile or a tool needs besides the profile's weights. */
export interface SetupAsset {
  id: string;
  label: string;
  /** with ${HOME}, ${REPO} or ${HF_HUB}; may end in a glob */
  path: string;
  also: string[];
  /** something in the repository downloads it */
  fetchable: boolean;
  /** reset level 4 never removes it */
  protected: boolean;
  hf_repo: string | null;
  source: string;
}

/** One cell of the measured table: mean tok/s (and min), ms a round, tokens a round. */
export interface SetupMeasureCell {
  tok_s: number;
  min: number;
  ms_round: number;
  tok_round: number;
  n: number;
  ttft_s?: number;
}
export type SetupMeasureMode = 'greedy_512' | 'greedy_128' | 'sampled_512' | 'sampled_128' | 'long_greedy' | 'long_sampled';
export type SetupMeasureTask = 'prose' | 'chat' | 'code' | 'long32k' | 'long64k';
export interface SetupMeasureRow {
  profile: string;
  build: string;
  date: string;
  /** false: another build or other rows (the NVFP4 row from 0.4.8); show it apart */
  comparable: boolean;
  note: string | null;
  cells: Partial<Record<SetupMeasureMode, Partial<Record<SetupMeasureTask, SetupMeasureCell>>>>;
  /** TTFT with the caches off, by prompt length in tokens ("8192", "32768") */
  ttft_s: Record<string, number>;
  lossless: { identical: number; near_tie: number; other: number; max_near_tie_ulp: number | null } | null;
}
export interface SetupMeasurements {
  build: string;
  date: string;
  method: string;
  source: string;
  rows: SetupMeasureRow[];
}

export type SetupComponentGroup = 'speed' | 'roles' | 'caches' | 'disk' | 'features';
/** A tool that runs beside the model (the narrator, Laya, the caches, ...). */
export interface SetupComponent {
  id: string;
  label: string;
  group: SetupComponentGroup;
  /** one plain sentence for users */
  what: string;
  /** on in the recommended setup */
  default: boolean;
  /** cannot be switched off (the engine always loads it) */
  required: boolean;
  /** memory it holds while loaded */
  memory_gib: number;
  /** what it may grow to (a cache at its cap), when that differs */
  memory_max_gib: number | null;
  /** given back first when memory runs short (caches, the small models) */
  elastic: boolean;
  disk_gb: number;
  depends_on: string[];
  conflicts: string[];
  /** serve.env key -> its value with the component on / off */
  /** null: that state writes nothing (the template's or the profile's value stands; model.yaml) */
  knobs: Record<string, { on: string | null; off: string | null }>;
  docs: string;
  note: string | null;
  /** 1.0+: SetupAsset ids that must be on disk for it to load */
  assets: string[];
}

/** A field of the methods and advanced steps. */
export interface SetupField {
  id: string;
  label: string;
  what: string;
  type: 'enum' | 'int' | 'number' | 'bool' | 'string';
  options?: { value: string; label: string }[];
  min?: number;
  max?: number;
  /** string fields: the characters allowed (a server-side check too) */
  pattern?: string;
  default: SetupValue;
  recommended: SetupValue;
  /** the serve.env key it writes; null: not a serve.env key (handled by its own step) */
  knob: string | null;
  docs: string | null;
}

export interface SetupParallelism {
  slots: { min: number; max: number; default: number };
  window: { min: number; max: number; default: number; step: number };
  side_window: { min: number; max: number; default: number; step: number };
  kv_pool: { default: boolean };
  floor_gb: { min: number; max: number; default: number };
  kv_hard_floor_gb: number;
  /** 106.5 KiB on Qwen3.8-27B: engine attention KV + the two drafter arms + the rotary row */
  kv_bytes_per_token: number;
  /** a side slot's bytes beyond its window (verify graphs off) */
  side_slot_const_gib: number;
  knobs: Record<'slots' | 'window' | 'side_window' | 'kv_pool' | 'floor_gb', string>;
  /** 1.0+: a slot's conversation in the estimate's typical case (32,768) */
  typical_fill_tokens: number;
  /** 1.0+: serve.env keys written with every selection (QSE_SIDE_GRAPHS off, ...) */
  fixed: Record<string, string>;
}

export interface SetupDevicePreset {
  id: string;
  label: string;
  total_gib: number;
  /** the OS, the CUDA context and other programs (outside the engine's allocator) */
  reserved_gib: number;
  note: string | null;
}

/** GET /v1/setup/models/{id}: model.yaml as JSON (schema 1). */
export interface SetupModel {
  contract_version: SetupContract;
  schema: 1;
  id: string;
  name: string;
  family: string;
  params: string;
  pitch: string;
  architecture: string;
  license: string;
  links: { label: string; url: string }[];
  icon: string | null;
  profiles: SetupProfile[];
  measurements: SetupMeasurements;
  components: SetupComponent[];
  parallelism: SetupParallelism;
  methods: SetupField[];
  advanced: SetupField[];
  devices: SetupDevicePreset[];
  /** the engine's fixed parts the estimate adds (drafters, graphs, ...), with their sources */
  memory: {
    parts: { id: string; label: string; gib: number; kind: 'fixed' | 'reserve'; source: string }[];
    sources: string[];
    /** SRV-337: parts that grow with slot 0's window (bytes a token) */
    per_window_parts?: { id: string; label: string; bytes_per_token: number; max_gib?: number; kind: 'fixed' | 'reserve'; source: string }[];
  };
  /** 1.0+: the tile's recommended badge */
  recommended: boolean;
  /** 1.0+ */
  assets: SetupAsset[];
  /** 1.0+: the step schema (the same list GET .../steps returns as `steps`) */
  wizard_fields: SetupStep[];
  /** the community packs (newest first) from the index in use, no fetch; absent on an older server */
  community?: CommunityEntry[];
  /** `dropped` and `add_url` for this model only */
  community_sync?: CommunitySync;
}

/** The wizard's choices: what the estimate, the plan and the apply job take. */
export interface SetupSelection {
  model: string;
  profile: string;
  components: Record<string, boolean>;
  slots: number;
  /** slot 0's window (MAX_LEN) */
  window: number;
  /** each side slot's window (QSE_SIDE_MAX_LEN); default = window */
  side_window?: number;
  kv_pool: boolean;
  floor_gb: number;
  methods: Record<string, SetupValue>;
  advanced: Record<string, SetupValue>;
  /** a SetupDevicePreset id; default the model's first */
  device?: string;
}

/** 1.0+: GET /v1/setup/models/{id}/current: what the box runs now, as a selection (the wizard's
 *  seed); a value the registry cannot express stays recommended and is listed in `unmatched`. */
export interface SetupCurrent {
  contract_version: SetupContract;
  model: string;
  selection: SetupSelection;
  unmatched: { field: string; value: unknown }[];
}

/** POST /v1/setup/estimate {selection} (also the TS copy's input, models/estimate-cases.json). */
export type SetupEstimateRequest = SetupSelection;
export interface SetupEstimatePart {
  id: string;
  label: string;
  gb: number;
  kind: 'fixed' | 'kv' | 'reserve';
  /** kv: every slot at its full window; reserve: a cache at its cap */
  worst_gb?: number;
}
export interface SetupEstimate {
  contract_version: SetupContract;
  /** the parts' sum: the KV part at typical_fill_tokens a slot with the pool, every window without */
  total_gb: number;
  /** live MemAvailable of the box; null in the fixture and off the box */
  available_gb: number | null;
  device_gb: number;
  floor_gb: number;
  parts: SetupEstimatePart[];
  fits: boolean;
  /** tokens of conversation that fit in the room left, over all slots */
  context_tokens_fit: number;
  kv_bytes_per_token: number;
  /** every slot at its full window */
  kv_worst_gb: number;
  /** device - fixed parts - floor: what conversations (and the caches, which give it back) may use */
  room_gb: number;
  notes: string[];
  /** 1.0+: device - floor; fits is total_gb <= budget_gb, both rounded to 0.01 */
  budget_gb: number;
  /** 1.0+: room_gb >= kv_worst_gb (every slot at its full window fits too) */
  fits_worst: boolean;
  /** 1.0+ */
  typical_fill_tokens: number;
}

/** GET /v1/setup/models/{id}/steps: the step schema Wren's tool calls are checked against. */
export interface SetupStepField {
  field: string;
  label: string;
  type: 'enum' | 'int' | 'number' | 'bool' | 'string';
  options?: string[];
  min?: number;
  max?: number;
  pattern?: string;
}
export interface SetupStep {
  index: number;
  id: SetupStepId;
  label: string;
  fields: SetupStepField[];
}
/** read_device: the page's own read-only tool (the server puts the device in Wren's prompt instead) */
export type SetupToolName = 'set_field' | 'toggle' | 'select_profile' | 'go_to_step' | 'next' | 'previous' | 'explain' | 'read_device';
export interface SetupSteps {
  contract_version: SetupContract;
  model: string;
  steps: SetupStep[];
  /** the OpenAI tool definitions Wren is given */
  tools: { type: 'function'; function: { name: SetupToolName; description: string; parameters: Record<string, unknown> } }[];
}

/** GET /v1/setup/device: the box, read only. */
export interface SetupDevice {
  contract_version: SetupContract;
  /** setup: started by `ops/start.sh --setup` (no model loaded); serving: the engine is up */
  mode: 'setup' | 'serving';
  preset: string | null;
  mem_total_gb: number | null;
  mem_available_gb: number | null;
  gpu: { name: string | null; util_pct: number | null; temp_c: number | null; power_w: number | null } | null;
  disk: { path: string; free_gb: number; total_gb: number } | null;
  engine: { version: string; profile: string | null; uptime_s: number; running: number; waiting: number; busy: boolean; mode: 'setup' | 'serving' };
  active_profile: string | null;
  /** profile id -> its weights are on this box */
  profiles_on_disk: Record<string, boolean>;
  /** false: anyone who reaches the port can run setup, apply and reset */
  password_set: boolean;
  assistant: SetupAssistantStatus;
  /** 1.0+: a Hugging Face token is configured (never the token) */
  hf_token_present: boolean;
  /** 1.0+: the same as password_set */
  admin_password_set: boolean;
}

export type SetupAssistantState = 'off' | 'not_loaded' | 'loading' | 'loaded' | 'unloading' | 'failed' | 'refused';
/** GET /v1/setup/assistant/status (and POST load / unload answer the same). */
export interface SetupAssistantStatus {
  name: string;
  /** qwen3.5-2b, or qwen3.5-0.8b when the 2B is not on disk; null when neither is */
  model: string | null;
  fallback: boolean;
  state: SetupAssistantState;
  /** why it is off, refused or failed, in a sentence */
  reason: string | null;
  load_s: number | null;
  device_gib: number | null;
  claim_gib: number | null;
  /** a load is possible now (a model on disk, not loaded or loading) */
  can_load: boolean;
  /** 1.0+: Wren uses the narrator's loaded copy (no memory of its own) */
  borrowed: boolean;
  /** 1.0+: the rows of its cache (8,192 own, the narrator's 4,096 borrowed) */
  window: number;
  /** 1.0+: seconds into a load, while loading */
  loading_s: number | null;
  /** 1.0+: counts (turns, tool_valid, tool_invalid, errors, unload_<why>) */
  stats: Record<string, number>;
}

export interface SetupToolCall {
  id: string;
  name: SetupToolName;
  args: Record<string, unknown>;
}
export interface SetupChatMessage {
  role: 'user' | 'assistant' | 'tool';
  content: string;
  tool_calls?: SetupToolCall[];
  /** role tool: the call it answers; content is the browser's result ("ok", or the error) */
  tool_call_id?: string;
}
/** What the page shows now, in the system prompt each turn. */
export interface SetupPageState {
  model: string | null;
  step: SetupStepId;
  step_index: number;
  selection: Partial<SetupSelection>;
  /** the step's visible content in short (labels, values, errors) */
  visible?: Record<string, unknown>;
  estimate?: Partial<SetupEstimate>;
}
/** POST /v1/setup/assistant/chat: an SSE stream of SetupChatEvent (`data: {...}` frames). */
export interface SetupChatRequest {
  messages: SetupChatMessage[];
  page: SetupPageState;
  /** false: answer in words only (the page's last round); the server renders no tools and drops a call */
  tools?: boolean;
}
export interface SetupSource {
  path: string;
  title: string;
}
export type SetupChatEvent =
  | { type: 'state'; state: 'thinking' | 'loading' | 'talking' }
  | { type: 'text'; delta: string }
  | { type: 'tool_call'; id: string; name: SetupToolName; args: Record<string, unknown> }
  | { type: 'done'; finish: 'stop' | 'length' | 'time' | 'tool_calls'; sources: SetupSource[]; unchanged?: boolean }
  | { type: 'error'; message: string };

/** POST /v1/setup/plan {selection}: the review step's facts, nothing changes. */
export interface SetupEnvChange {
  key: string;
  current: string | null;
  next: string | null;
  /** the component or field that sets it */
  from: string;
  /** VIS-155: the run/setup.env on the box sets it and the new one does not; `next` is what
   *  serve.env and the profile then give (null: the engine's own default) */
  removed?: boolean;
}
export interface SetupPlan {
  contract_version: SetupContract;
  ok: boolean;
  /** what is wrong with the selection (Finish stays off) */
  errors: string[];
  warnings: string[];
  profile: { current: string | null; next: string; on_disk: boolean; download_gb: number };
  /** VIS-153: what the fetch step downloads: the weights, the profile's assets and the models of
   *  the tools the selection turns on, each with its size (GB) and what it is for */
  downloads?: { id: string; label: string; gb: number; for: string }[];
  /** VIS-153: the tools' asset ids the fetch step adds (QSE_FETCH_ASSETS) */
  fetch_assets?: string[];
  changes: SetupEnvChange[];
  restart: boolean;
  /** what the job will run, for reading: the server runs its own fixed steps, never these strings */
  commands: string[];
  estimate: SetupEstimate;
  /** 1.0+: a code per error, in order (over_budget, missing_assets, missing_token, disk_full,
   *  no_overlay_hook, secrets_exist, port_in_use, repo_mismatch) */
  error_codes: string[];
  /** 1.0+: the API port before and after */
  ports: { current: number; next: number };
  /** 1.0+: POST /v1/setup/apply {dry_run: true} answered this (false from POST /v1/setup/plan) */
  dry_run?: boolean;
}

export type SetupJobStepId = 'check' | 'unload_assistant' | 'fetch' | 'write_config' | 'switch' | 'smoke' | 'rollback';
/** waiting: the step waits for the engine's requests to finish before anything stops (1.0+) */
export type SetupJobStepState = 'pending' | 'waiting' | 'running' | 'done' | 'failed' | 'skipped';
export interface SetupJobStep {
  id: SetupJobStepId;
  label: string;
  state: SetupJobStepState;
  started_at: number | null;
  ended_at: number | null;
  detail: string | null;
}
export type SetupJobState = 'queued' | 'running' | 'done' | 'failed' | 'rolled_back';
export interface SetupJob {
  id: string;
  state: SetupJobState;
  created_at: number;
  ended_at: number | null;
  profile: string;
  steps: SetupJobStep[];
  message: string | null;
  changes: SetupEnvChange[];
  /** 1.0+: apply, or reset (reset level 2) */
  kind: 'apply' | 'reset';
  /** 1.0+ */
  rolled_back: boolean;
  failed_step: SetupJobStepId | null;
  /** 1.0+: the dashboard after the job, `<this box>` for the host the browser knows */
  url: string;
  /** VIS-153: the tools the selection turned on that did not load after the start, each with
   *  its reason (the job is still done) */
  warnings?: string[];
  /** VIS-153: tools waiting for memory or not loaded yet */
  notes?: string[];
}
/** POST /v1/setup/apply {selection} -> 202 {job}; 409 while a job runs or requests are in flight. */
export interface SetupApplyRequest {
  selection: SetupSelection;
  /** 1.0+: answer the plan (SetupPlan, dry_run true) and change nothing */
  dry_run?: boolean;
}
export interface SetupApplyResponse {
  contract_version: SetupContract;
  job: SetupJob;
}
/** GET /v1/setup/jobs/{id}: SSE (`id:` = sequence, resumable with Last-Event-ID); ?format=json: the SetupJob. */
export type SetupJobEvent =
  | { type: 'job'; job: SetupJob }
  | { type: 'step'; step: SetupJobStep }
  | { type: 'log'; line: string; ts: number }
  | { type: 'done'; state: SetupJobState; message: string | null; warnings?: string[]; notes?: string[] };

/** GET /v1/setup/reset: the server-side levels (level 1, the browser's progress and chat, is the page's own). */
export interface SetupResetItem {
  path: string;
  label: string;
  bytes: number;
  exists: boolean;
  /** 1.0+: files counted; at_least: the count stopped after 2 s (a huge tree) */
  files: number;
  at_least: boolean;
  /** 1.0+: path, or memory (a cache in RAM) */
  kind: 'path' | 'memory';
  /** 1.0+: remove; engine (the running engine clears it); skip (in use, nothing can clear it now) */
  action: 'remove' | 'engine' | 'skip';
  note: string | null;
  /** 1.0+: the running engine keeps its copy until a restart */
  restart: boolean;
  /** 1.0+: why the server will not remove it (null: it will) */
  refused: string | null;
}
export interface SetupResetLevel {
  level: 2 | 3 | 4 | 5;
  label: string;
  what: string;
  items: SetupResetItem[];
  bytes: number;
  restart: boolean;
  /** the word to type; null: a plain confirm (`confirm: true`) */
  confirm_word: string | null;
}
export interface SetupResetLevels {
  contract_version: SetupContract;
  levels: SetupResetLevel[];
  /** requests in flight: every reset is refused until they end */
  busy: boolean;
}
/** POST /v1/setup/reset {level, confirm}: confirm is the typed word, or true for level 2. */
export interface SetupResetRequest {
  level: 2 | 3 | 4 | 5;
  confirm: string | boolean;
  /** 1.0+: the level (SetupResetLevel, dry_run true) and nothing removed; no confirm needed */
  dry_run?: boolean;
}
export interface SetupResetResult {
  contract_version: SetupContract;
  level: number;
  removed: { path: string; bytes: number }[];
  restart_needed: boolean;
  message: string;
  /** 1.0+: what could not be removed, and why (the rest still went) */
  failed: { path: string; reason: string }[];
  freed_gb: number;
  ok: boolean;
  /** 1.0+: level 2 is a job (202): follow it at GET /v1/setup/jobs/{job_id} */
  job_id?: string;
  job?: SetupJob;
}
