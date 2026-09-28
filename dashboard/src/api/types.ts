// Dashboard API contract v1 — the TypeScript view of docs/contract/dashboard-v1/*.schema.json.
// Prose: Memo "Usage & speed metrics — design (2026-09-24)" §3. A change here bumps contract_version.

export type ContractVersion = '1.0' | '1.1';

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
}

export type LiveStatus = 'ok' | 'busy' | 'draining' | 'offline';

export interface Summary {
  contract_version: ContractVersion;
  generated_at: string;
  tz: string;
  windows: { today: Totals; '7d': Totals; '30d': Totals; '365d': Totals; all: Totals };
  streak: { current_days: number; longest_days: number };
  ledger: { enabled: boolean; since: string | null; rows: number; bytes: number; retention_days: number | null };
  live: { status: LiveStatus; running: number; waiting: number; uptime_s: number; last_request_at: string | null };
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
}

export interface ClientDim {
  id: string;
  label: string | null;
  kind: ClientKind;
}

export type ClientKind = 'open-webui' | 'openai-sdk' | 'opencode' | 'curl' | 'dashboard' | 'other';

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
  filters: { model: string | null; client: string | null };
  buckets: UsageBucket[];
  totals: Totals;
  top_days: TopDay[];
  dimensions: { models: string[]; clients: ClientDim[] };
}

export type FinishReason = 'stop' | 'length' | 'tool_calls' | 'timeout' | 'abandoned' | 'error' | 'refused';
export type CacheSource = 'response' | 'session' | 'prefix' | 'none';
export type Endpoint = 'chat' | 'completions';

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
  tokens_per_block: number | null;
  draft_tokens: number | null;
  draft_accepted: number | null;
  tool_calls: number;
  thinking: boolean;
  cache_source: CacheSource | null;
  error_type: string | null;
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
    model: string;
    max_len: number;
    drafter: string | null;
    tree: boolean;
    reasoning_format: string;
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
  caches: Record<string, unknown>;
  queue: { running: number; waiting: number; max_queue: number; queue_timeout_s: number; request_timeout_s: number };
  inflight: { served: number; refused: number; errors: number; timeouts: number; abandoned: number };
  ledger: { enabled: boolean; rows: number; bytes: number; oldest: string | null; queue: number; dropped: number };
  disk: { state_dir_free_bytes: number | null };
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

export interface SessionInfo {
  authenticated: boolean;
  expires_at: string;
}

export interface ApiError {
  error: { type: 'unauthorized' | 'bad_request' | 'not_found' | 'too_many'; message: string };
}

// The per-response usage block of (the finish chunk / the include_usage chunk).
export interface ChatUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  prompt_tokens_details?: { cached_tokens: number };
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
// Definitions in the Memo note "Live speed panel — design (2026-09-26)" §2.
export type LivePhase = 'queued' | 'prefill' | 'decode' | 'done';

export interface LiveSample {
  t: number; // unix seconds
  decode_tps: number | null; // tokens over this second, all requests; null when nothing decoded
  prefill_tps: number | null; // a prefill that finished in this second, else null
  running: number;
  waiting: number;
  tokens: number; // the running total the deltas come from
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
  cache_source: CacheSource | null;
  max_tokens: number | null;
  ended_ms_ago: number | null;
  // contract 1.1: absent on a 1.0 server, null with `--live-activity off`
  activity?: LiveActivity | null;
  timeline?: LiveTimelineEntry[] | null;
}

// ---- contract 1.1: what the model is doing right now ------------------
// Definitions in the Memo note "Live activity design (2026-09-27)" §3-§5; the words (`label`,
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
  prefill: { done: number | null; total: number; cached: number | null; pct: number | null; tps_now: number | null; tps_avg: number | null; eta_ms: number | null; progress: 'chunked' | 'single_call' | null; at_ms: number | null } | null;
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
  now: { decode_tps: number | null; prefill_tps: number | null; prefilling: boolean; tokens_per_block: number | null; last_prefill_ms_ago: number | null };
  requests: LiveRequest[];
  recent?: LiveRecent[] | null; // 1.1: the last 20 finished requests, newest first; null with --live-activity off
  sample: LiveSample | null;
  sampler?: { ticks: number; encodes: number; tick_us_last: number | null; tick_us_mean: number | null; tick_us_max: number | null; tick_cpu_us_mean: number | null; tick_cpu_us_max: number | null }; // 1.1
  history?: LiveSample[]; // the first event and follow=0 only
}
