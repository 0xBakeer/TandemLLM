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
  max_inflight?: number;
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
export type ClientKind = 'open-webui' | 'openai-sdk' | 'opencode' | 'hermes' | 'curl' | 'dashboard' | 'other' | 'internal';

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
  /** 0.3.27-test: served in the side lane between another request's blocks or prefill chunks */
  lane?: SideLane | null;
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
  /** `off` (0.3.41-test, VIS-54): the engine without --backends; /v1/config is a 404 there */
  proxy?: { mode: 'embedded' | 'standalone' | 'off'; backends: BackendStatus[] } | null;
  queue: {
    running: number;
    waiting: number;
    max_queue: number | null;
    queue_timeout_s: number | null;
    request_timeout_s: number | null;
    /** 0.3.27-test: short requests in the side lane now, and the lane's counts (null: off) */
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
  queue: { place: number | null; wait_ms: number | null; timeout_s: number | null; place_is_estimate: boolean } | null;
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
  /** 0.3.21-test: the prompt, what came out of a cache, what was forwarded and from where */
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

// ---- OPS-41: the engine's memory split and the residency planner -------------------------------

/** server/memsplit.py: the resident set in bytes, each part read off the object that holds it. */
export interface MemorySplit {
  allocated: number | null;
  reserved: number | null;
  parts: Record<string, number>; // weights, kv_pool, gdn_state, drafter_kv, vision, state_store, resident_prefix, deciders
  other: number | null; // what the allocator holds beyond the named parts
  reserve: number | null; // what it keeps from the driver without a tensor in it
  mem_available: number | null;
  max_len: number | null;
  drafter_caches: number;
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
