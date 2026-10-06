// The Backends and Setup screens' API: /v1/config/* (proxy/config_api.py). Reads follow the
// dashboard's rule; writes the admin rule (open while no password is set, AU1). No answer holds a key:
// a backend shows `key_set`, the last 4 characters and a hash prefix; a user's key comes back once,
// in the answer to its creation.

import { request } from './client';
import type { EngineBlock } from './types';

export type BackendKind = 'generic' | 'tandemllm' | 'vllm' | 'sglang' | 'openai' | 'deepseek';
export type FeatureValue = 'inherit' | 'off' | 'shadow' | 'on';

export interface KeyHint {
  key_set: boolean;
  last4: string | null;
  sha256_6: string | null;
  source: 'env' | 'database' | null;
  readable: boolean | null;
}

export interface Price {
  input: number | null;
  cached_input: number | null;
  cache_write: number | null;
  output: number | null;
  currency?: 'USD';
  updated_at?: number;
  /** a provider's price outside its peak hours (DeepSeek); the adapter picks by the hour */
  offpeak?: { input: number | null; cached_input: number | null; cache_write: number | null; output: number | null } | null;
}

export type WriterMode = 'auto' | 'off' | 'shadow' | 'on';

export interface WriterConfig {
  writer: WriterMode;
  /** a paid model's spend cap a day in USD; null: no cap */
  writer_usd_day: number | null;
  writer_runs_day: number;
  writer_prompt_tokens: number;
}

export interface WriterView {
  config: WriterConfig;
  today: { runs: number; cost_micros: number | null };
  settings_mode?: 'off' | 'shadow' | 'on' | null;
  shadow_log?: string | null;
}

/** proxy/memwriter.py DEFAULTS */
export const WRITER_DEFAULTS: WriterConfig = { writer: 'auto', writer_usd_day: 0.25, writer_runs_day: 48, writer_prompt_tokens: 32000 };

export interface ModelRow {
  backend: string;
  model: string;
  public_id: string | null;
  enabled: boolean;
  present: boolean;
  source: 'discovered' | 'list' | 'manual' | 'env';
  discovered_at: number | null;
  context_length: number | null;
  price: Price | null;
  fallback: string[] | null;
  features: Record<string, FeatureValue>;
  locks: { price?: string };
  /** P4: what each switch resolves to for this model (`inherit` read through the backend and the
   * defaults by kind) */
  effective?: Record<string, 'off' | 'shadow' | 'on'>;
  /** P4: the harness's extra requests over the last 24 h, by feature group */
  harness_cost_day?: Record<string, { requests: number; cost_micros: number | null }>;
  /** M3: the background writer's mode for this model and why (auto, paid_cloud, backend, settings) */
  writer?: { mode: 'off' | 'shadow' | 'on'; why: string };
}

export interface Health {
  state: 'up' | 'down' | 'unknown' | 'disabled';
  since: number | null;
  last_check: number | null;
  last_error: string | null;
  latency_ms: number | null;
  models_at: number | null;
}

export interface BackendConfig {
  id: string;
  kind: BackendKind;
  label: string;
  url: string;
  enabled: boolean;
  models_mode: 'discover' | 'list';
  models: string[];
  fallback: string[];
  /** null: no limit from the proxy (the default) */
  max_inflight: number | null;
  timeouts: { connect: number; first_byte: number; idle: number };
  inject_usage: boolean;
  harness: {
    mode: 'off' | 'shadow' | 'on';
    profile: string | null;
    /** M3 (0.3.35): the background memory writer's switches on this backend (absent: the defaults) */
    writer?: WriterMode;
    writer_usd_day?: number | null;
    writer_runs_day?: number;
    writer_prompt_tokens?: number;
  };
  /** M3 (0.3.35): the writer's switches as they resolve (`config`), the day's runs and spend,
   * the settings document's mode and where a shadow run's blocks go (0.3.36); absent without a
   * conductor */
  writer?: WriterView;
  /** P1.5: where the System view reads the backend's machine (absent on an older server) */
  host?: { mode: 'auto' | 'same' | 'collector' | 'exporter' | 'off'; url: string | null; dcgm_url: string | null };
  cloud: boolean;
  created_at: number;
  updated_at: number;
  updated_by: string | null;
  key: KeyHint;
  /** field -> the variable that sets it (the environment wins) */
  locks: Record<string, string>;
  health: Health;
  in_flight: number;
  capabilities: Record<string, unknown>;
  routes: string[];
  /** P2, P3: the engine's own /metrics page and discovery (vllm, sglang); null otherwise */
  engine?: EngineBlock | null;
  /** P4: per feature group, what it costs or loses on this backend, one sentence */
  degradations?: Record<string, string>;
  model_rows?: ModelRow[];
  test?: { state: string; latency_ms: number | null; error: string | null };
}

export interface Rule {
  id?: number;
  position?: number;
  pattern: string;
  backend: string;
  enabled: boolean;
  source?: 'env' | 'gui';
  variable?: string | null;
}

export interface ProxyUser {
  id: number | null;
  label: string;
  key_hint: string | null;
  role: string;
  enabled: boolean;
  created_at: number | null;
  last_seen_at: number | null;
  source: 'database' | 'env';
}

export interface KindInfo {
  cloud: boolean;
  url: string | null;
  inject_usage: boolean;
  timeouts: { connect: number; first_byte: number; idle: number };
}

export interface ConfigOverview {
  contract_version: string;
  setup_needed: boolean;
  key_source: 'env' | 'keychain' | 'file' | null;
  config_path: string;
  kinds: Record<BackendKind, KindInfo>;
  feature_groups: string[];
  backends: BackendConfig[];
  rules: Rule[];
  users: ProxyUser[];
  users_active: boolean;
  engine: {
    model: string; base_model: string | null; embedded: boolean;
    /** PL1: this engine's own parallel requests (never a relayed backend's: those get no limit
     *  from us) and the small models it runs beside Qwen */
    parallel?: { n: number; requested: number; side_max_len: number | null; side_graphs: string | null; floor_gib: number | null };
    roles?: { deciders?: string[]; narrator?: { model: string | null; state: string } };
  } | null;
  generation: number;
  /** P4: whether a conductor runs the harness on this server, its profiles, and the defaults of
   * every feature group by kind */
  harness?: { conductor: boolean; profiles?: string[]; defaults: { local: Record<string, string>; cloud: Record<string, string> } };
}

export interface TestResult {
  ok: boolean;
  status?: number | null;
  latency_ms: number | null;
  error: string | null;
  models: string[];
}

export interface SeedPrices {
  as_of: string;
  note: string;
  providers: Record<string, Record<string, Price>>;
}

const J = { 'Content-Type': 'application/json' };
const body = (b: unknown) => JSON.stringify(b);
const enc = encodeURIComponent;
/** a model id may hold `/` (Qwen/Qwen3.5-4B): each part is encoded, the slashes kept */
const encModel = (m: string) => m.split('/').map(enc).join('/');

export const config = {
  overview: () => request<ConfigOverview>('/v1/config'),
  test: (b: { kind: BackendKind; url?: string; key?: string; id?: string }) => request<TestResult>('/v1/config/test', { method: 'POST', headers: J, body: body(b) }),
  create: (b: Partial<BackendConfig> & { id: string; kind: BackendKind; key?: string }) => request<BackendConfig>('/v1/config/backends', { method: 'POST', headers: J, body: body(b) }),
  update: (id: string, patch: Record<string, unknown>) => request<BackendConfig>(`/v1/config/backends/${enc(id)}`, { method: 'PATCH', headers: J, body: body(patch) }),
  remove: (id: string) => request<void>(`/v1/config/backends/${enc(id)}`, { method: 'DELETE' }, 'none'),
  retest: (id: string) => request<TestResult>(`/v1/config/backends/${enc(id)}/test`, { method: 'POST', headers: J, body: '{}' }),
  model: (id: string, model: string, patch: Record<string, unknown>) => request<ModelRow>(`/v1/config/backends/${enc(id)}/models/${encModel(model)}`, { method: 'PUT', headers: J, body: body(patch) }),
  removeModel: (id: string, model: string) => request<void>(`/v1/config/backends/${enc(id)}/models/${encModel(model)}`, { method: 'DELETE' }, 'none'),
  rules: (rules: Rule[]) => request<{ rules: Rule[] }>('/v1/config/rules', { method: 'PUT', headers: J, body: body({ rules }) }),
  addUser: (label: string) => request<{ user: ProxyUser; key: string; note: string }>('/v1/config/users', { method: 'POST', headers: J, body: body({ label }) }),
  setUser: (id: number, enabled: boolean) => request<ProxyUser>(`/v1/config/users/${id}`, { method: 'PATCH', headers: J, body: body({ enabled }) }),
  removeUser: (id: number) => request<void>(`/v1/config/users/${id}`, { method: 'DELETE' }, 'none'),
  seedPrices: () => request<SeedPrices>('/v1/config/prices/seed'),
};
