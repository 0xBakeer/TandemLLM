// Dashboard API contract v1 — the TypeScript view of docs/contract/dashboard-v1/*.schema.json.
// Prose: Memo "Usage & speed metrics — design (2026-09-24)" §3. A change here bumps contract_version.

export type ContractVersion = '1.0';

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
  ledger: { enabled: boolean; since: string | null; rows: number; bytes: number; retention_days: number };
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

export type ClientKind = 'open-webui' | 'openai-sdk' | 'curl' | 'dashboard' | 'other';

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
    drafter: string;
    tree: boolean;
    reasoning_format: string;
    reasoning_effort: string;
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
export type LogSource = 'req' | 'server' | 'cache' | 'drafter' | 'think' | 'http' | 'traceback' | 'other';

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

// The per-response usage block of SRV-27 (the finish chunk / the include_usage chunk).
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
