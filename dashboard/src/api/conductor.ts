// The conductor's API (/v1/conductor/*, contract docs/contract/conductor-v1): its types and calls.
// Reads go through the shared `request` (a 401 there means the dashboard session ended). Writes
// carry the admin token when one was entered on this device, and a 401 on a write only asks for
// the token: with the dashboard login off the page has no session, and editing needs the token.

import { ApiError, apiBase, request } from './client';

export type Mode3 = 'off' | 'shadow' | 'on';
export type TaskMode = 'off' | 'shadow' | 'act';
export type ProfileMode = 'raw' | 'shadow' | 'on';
export type Effort = 'off' | 'low' | 'medium' | 'high' | 'xhigh';
export type Decision = 'classify' | 'effort' | 'packages' | 'rules';
export type DocKind = 'settings' | 'router' | 'roles' | 'packages' | 'monitor' | 'profiles';

export interface Policy {
  effort?: Effort;
  packages?: string[];
  rules?: string[];
  note?: string;
}
export interface TaskClass {
  id: string;
  label: string;
  description: string;
  policy: Policy;
}
export interface Stage {
  kind: 'decider' | 'qwen';
  enabled: boolean;
  accept: number;
  timeout_ms: number;
}
export interface EffortLevel {
  thinking: boolean;
  budget: number;
  note: string;
}
export interface RouterDoc {
  classes: TaskClass[];
  classifier: { stages: Stage[]; question: string; digest_chars: number; package_question: string; package_accept: number; qwen_prompt: string };
  structural: { compaction: string; tool_turn: 'keep' | 'classify' };
  stickiness: { switch: number };
  effort: { levels: Record<Effort, EffortLevel>; raise_on_fail: boolean };
  injection: { wrapper: string; max_note_tokens: number; max_replay_tokens: number; replay: boolean };
}
export interface TaskSetting {
  mode: TaskMode;
  cap: number;
  thresholds: Record<string, number>;
}
export interface ProfileDoc {
  description: string;
  mode: ProfileMode;
  clients: string[];
  default_class: string;
  decisions: Record<Decision, Mode3>;
  routing: Record<string, Policy>;
  thinking: { default: Effort; hints: boolean; client_override: true };
  tasks: Record<string, TaskSetting>;
  caps: { acts_per_request: number; acts_per_session_hour: number };
  toolgate: { risk: boolean; project: boolean; same_call: boolean; malformed: boolean; tier1_timeout_ms: number };
  context: { s1_runs: number; s4_minutes: number; notify: boolean };
  hints: { stair_width: boolean; prefix_retention: boolean };
  locked?: string;
}
export interface RoleDoc {
  endpoint: string;
  model: string;
  contract: 'systemone' | 'openai-chat' | 'rerank';
  timeout_ms: number;
  memory_gb: number;
  mode: Mode3;
}
export interface PackageDoc {
  title: string;
  kind: 'skill' | 'package';
  source: string;
  description: string;
  body: string;
  max_tokens: number;
  enabled: boolean;
}
export interface PackagesDoc {
  skill_dirs: string[];
  packages: Record<string, PackageDoc>;
}
/** The session monitor's policy (ops/supervisor.yaml is version 0): every task with its rule and
 * the rule's parameters, caps, texts, templates. Its schema is `meta.schemas.monitor`. */
export interface MonitorTask {
  rule: string;
  mode: TaskMode;
  [param: string]: unknown;
}
export interface MonitorDoc {
  version: number;
  caps: { acts_per_request: number; acts_per_session_hour: number; same_action_in_a_row: boolean };
  session: { lineage_tokens: number; join_window_s: number; max_sessions: number };
  texts: Record<string, string>;
  templates: Record<string, string>;
  journal: { tool_passes: boolean };
  tasks: Record<string, MonitorTask>;
}
/** One decision of the session monitor (journal kind `monitor`, SRV-115). */
export interface MonitorEvent {
  id: string;
  ts_ms: number;
  request_id: string | null;
  session_key: string | null;
  kind: 'monitor';
  task: string | null;
  tier: number | null;
  mode: 'shadow' | 'act' | null;
  sp: string | null;
  rule: string | null;
  trigger: string | null;
  score: number | null;
  decision: string | null;
  action: string | null;
  cap: string | null;
  hidden_text_id: string | null;
  hidden_tokens: number | null;
  tokens_at_cut: number | null;
  position: number | null;
  outcome: { kind?: string | null; acts?: number | null; hidden_tokens?: number | null; finish?: string | null } | null;
  next_checker: DecisionRow['next_checker'];
  policy_version: number | null;
  client_kind: string | null;
  profile: string | null;
  detail: Record<string, unknown> | null;
  label: 'right' | 'wrong' | null;
  label_note: string | null;
  labelled_at: number | null;
  text: string | null;
}
export interface Intervention {
  source: 'router' | 'monitor';
  id: string;
  ts_ms: number;
  request_id: string | null;
  task: string | null;
  action: string;
  reason: string | null;
  text: string | null;
  tokens: number;
  pos: number | null;
  hidden: boolean;
  outcome: Record<string, unknown> | null;
  next_checker: DecisionRow['next_checker'];
}
export interface MonitorStats {
  decisions: number;
  acts: number;
  shadow: number;
  gave_up: number;
  hidden_tokens: number;
  by_task: { task: string; family: string; acts: number; shadow: number; pass: number }[];
  by_action: Record<string, number>;
  hourly: { start_ms: number; acts: number; shadow: number; by_family: Record<string, number> }[];
  last_actions: { id: string; ts_ms: number; request_id: string | null; session_key: string | null; task: string | null; decision: string | null; trigger: string | null; outcome: MonitorEvent['outcome']; hidden_tokens: number | null; cap: string | null }[];
  precision: { task: string; decisions: number; labelled: number; agree: number; precision: number | null; mode: string }[];
}
export interface SettingsDoc {
  default_profile: string;
  journal: { retention_days: number; digest: 'tools' | 'all' | 'none' };
}

export interface VersionMeta {
  version: number;
  note: string | null;
  author: string | null;
  created_at: string | null;
  restored_from: number | null;
  loaded_at?: string | null;
  error?: { file?: string; message: string; at?: string } | null;
}
export interface DocResponse<T> {
  contract_version: string;
  kind: DocKind;
  name: string | null;
  version: number;
  doc: T;
  meta: VersionMeta;
  locked?: string | null;
  status?: Record<string, { available: boolean; error: string | null; path: string | null; chars: number; tokens_est: number }>;
}
export interface WriteResult {
  version: number;
  loaded_at: string | null;
  restored_from?: number;
  mode?: string;
}
export interface VersionRow extends VersionMeta {
  current: boolean;
}
export interface Change {
  path: string;
  before: unknown;
  after: unknown;
}
export interface ProfileTile {
  name: string;
  version: number;
  mode: ProfileMode;
  description: string;
  clients: string[];
  default_class: string;
  locked: string | null;
  decisions: Record<Decision, Mode3>;
  counts: { tasks: Record<TaskMode, number>; decisions: Record<Mode3, number> };
  last_change: { version: number; note: string | null; author: string | null; created_at: string | null };
  error: VersionMeta['error'];
}
export interface StageAnswer {
  stage: 'decider' | 'qwen';
  label: string | null;
  confidence: number;
  probs?: Record<string, number>;
  ms: number;
  error: string | null;
  model?: string | null;
  accept?: number;
  below?: boolean;
  cached?: boolean;
}
export interface Injected {
  pos: number;
  at: number;
  tokens: number;
  text: string;
  parts: { kind: 'effort' | 'package' | 'rules'; name: string }[];
  kinds: string[];
  reason: string;
}
export interface DecisionRow {
  id: string;
  ts_ms: number;
  request_id: string | null;
  session_key: string | null;
  client_kind: string | null;
  model: string | null;
  profile: string | null;
  profile_version: number | null;
  router_version: number | null;
  mode: 'shadow' | 'on';
  class_id: string | null;
  confidence: number | null;
  source: 'decider' | 'qwen' | 'sticky' | 'structural' | 'fallback' | 'profile' | null;
  kept: string | null;
  stages: StageAnswer[];
  classifier_ms: number | null;
  effort: Effort | null;
  budget: number | null;
  budget_source: string | null;
  thinking: string | null;
  decisions: Record<Decision, Mode3>;
  injected: Injected[] | null;
  replayed: { count: number; tokens: number; positions?: number[] } | null;
  shadow: Record<string, unknown> | null;
  digest: string | null;
  outcome: { finish_reason?: string | null; tool_calls?: number; tool_names?: string[]; completion_tokens?: number | null; reasoning_tokens?: number | null; ttft_ms?: number | null; total_ms?: number | null; status?: number | null } | null;
  next_checker: { checker: string; target: string; verdict: 'PASS' | 'FAIL'; reasons: string[]; readings: { gate: string; name: string; value: number; op: string; limit: number }[] }[] | null;
  label: 'right' | 'wrong' | null;
  label_note: string | null;
  labelled_at: number | null;
  acted: number | null;
  changed: number | null;
  extra: Record<string, unknown> | null;
  /** the session monitor's decisions in the same request */
  events?: MonitorEvent[];
  hidden_tokens?: number | null;
}
export interface RoleHealth {
  up: boolean;
  ms: number | null;
  mode: string | null;
  error?: string | null;
  detail?: Record<string, unknown>;
  limits?: Record<string, number>;
}
export interface PrecisionRow {
  profile: string | null;
  decision: Decision;
  decisions: number;
  labelled: number;
  agree: number;
  precision: number | null;
}
export interface Stats {
  hours: number;
  decisions: number;
  acted: number;
  shadow: number;
  changed: number;
  injections: number;
  injected_tokens: number;
  by_class: Record<string, number>;
  by_source: Record<string, number>;
  by_effort: Record<string, number>;
  classifier: Record<string, { calls: number; p50_ms: number | null; p95_ms: number | null }>;
  failures: Record<string, number>;
  fail_open_rate: number | null;
  model_decided: number;
  labels: { right: number; wrong: number };
  precision: PrecisionRow[];
  classes: { class_id: string; decisions: number; labelled: number; agree: number }[];
  hourly: { start_ms: number; total: number; by_class: Record<string, number>; acted: number; shadow: number }[];
  monitor: MonitorStats | null;
  roles: Record<string, RoleHealth>;
}
export interface SessionRow {
  key: string;
  client_kind: string | null;
  profile: string | null;
  started_ms: number;
  turns: number;
  notes: number;
  note_tokens: number;
  classes: Record<string, number>;
  class_id: string | null;
  last_ms: number;
  last_checker: string | null;
  labels: number;
  acts: number;
  gave_ups: number;
  shadow: number;
  interventions: number;
}
export interface LiveSession {
  key: string;
  class_id: string | null;
  confidence: number | null;
  source: string | null;
  turns: number;
  notes: number;
  note_tokens: number;
}
export interface Explain {
  profile: string;
  profile_version?: number;
  mode: string;
  active: boolean;
  class_id?: string | null;
  confidence?: number | null;
  source?: string | null;
  kept?: string | null;
  stages?: StageAnswer[];
  classifier_ms?: number | null;
  effort?: Effort | null;
  budget?: number | null;
  budget_source?: string | null;
  thinking?: string | null;
  decisions?: Record<Decision, Mode3>;
  shadow?: Record<string, unknown> | null;
  extra?: Record<string, unknown> | null;
  note?: string | null;
  note_parts?: { kind: string; name: string }[];
  thinking_off?: boolean;
}
export interface ConfigInfo {
  server: { composite: string; raw_model: string | null; reasoning_effort: string | null; global_mode: string | null; monitor_mode?: 'raw' | 'shadow' | 'on'; self_url: string | null };
  store: { root: string; readonly: boolean; writes: number; documents: Record<string, { version: number | null; loaded_at: string | null; error: VersionMeta['error']; note?: string | null }> };
  journal: { path: string | null; ring: number; queue: number; dropped: number; errors: number; retention_days: number; events_ring?: number; monitor_rows?: number; table?: string };
  time: string;
}
export interface TaskMeta {
  label: string;
  family: string;
  tier: string;
  role: string | null;
  why: string;
  thresholds: Record<string, { default: number; unit: string }>;
}
export interface Meta {
  modes: Mode3[];
  task_modes: TaskMode[];
  profile_modes: ProfileMode[];
  efforts: Effort[];
  decisions: Decision[];
  client_kinds: string[];
  locked: Record<string, string>;
  tasks: Record<string, TaskMeta>;
  schemas?: Record<string, JsonSchema>;
}
/** The JSON Schema subset the config documents use (the monitor's form renders from it). */
export interface JsonSchema {
  type?: string | string[];
  description?: string;
  properties?: Record<string, JsonSchema>;
  required?: string[];
  items?: JsonSchema;
  enum?: unknown[];
  const?: unknown;
  minimum?: number;
  maximum?: number;
  minLength?: number;
  minItems?: number;
  oneOf?: JsonSchema[];
  additionalProperties?: boolean | JsonSchema;
}
export interface EvalReport {
  kind?: 'classifier' | 'suite' | 'speed_guard';
  file?: string;
  [k: string]: unknown;
}
export interface SkillFound {
  name: string;
  description: string;
  path: string;
  chars: number;
  registered: boolean;
}

// ---------------------------------------------------------------- the admin token for writes
const TOKEN_KEY = 'qse-conductor-token';

export function adminToken(): string | null {
  try {
    return sessionStorage.getItem(TOKEN_KEY) ?? localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}
export function setAdminToken(token: string | null, remember = false): void {
  try {
    sessionStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(TOKEN_KEY);
    if (token) (remember ? localStorage : sessionStorage).setItem(TOKEN_KEY, token);
  } catch {
    /* storage blocked: the token lives for this call only */
  }
}

/** A write. A 401 is thrown as ApiError type `token_needed` and does not end the session. */
async function write<T>(endpoint: string, method: 'PUT' | 'POST', body: unknown): Promise<T> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  const tok = adminToken();
  if (tok) headers.Authorization = `Bearer ${tok}`;
  let res: Response;
  try {
    res = await fetch(apiBase() + endpoint, { method, headers, body: JSON.stringify(body), credentials: 'same-origin', cache: 'no-store' });
  } catch (e) {
    throw new ApiError(endpoint, 0, `engine unreachable (${(e as Error).message})`, 'offline');
  }
  let payload: { error?: { message?: string; type?: string; errors?: string[] } } & Record<string, unknown> = {};
  try {
    payload = await res.json();
  } catch {
    /* not json */
  }
  if (res.status === 401) throw new ApiError(endpoint, 401, 'editing needs the admin token', 'token_needed');
  if (!res.ok) {
    const err = new ApiError(endpoint, res.status, payload.error?.message ?? `HTTP ${res.status}`, payload.error?.type ?? 'error');
    (err as ApiError & { errors?: string[] }).errors = payload.error?.errors ?? [];
    throw err;
  }
  return payload as T;
}

const enc = encodeURIComponent;
const docPath = (kind: DocKind, name?: string | null) => (kind === 'profiles' ? `/v1/conductor/profiles/${enc(name ?? '')}` : `/v1/conductor/${kind}`);

export const conductor = {
  config: () => request<ConfigInfo>('/v1/conductor/config'),
  meta: () => request<Meta>('/v1/conductor/meta'),
  profiles: () => request<{ default_profile: string; profiles: ProfileTile[] }>('/v1/conductor/profiles'),
  doc: <T>(kind: DocKind, name?: string | null) => request<DocResponse<T>>(docPath(kind, name)),
  versions: (kind: DocKind, name?: string | null) => request<{ versions: VersionRow[] }>(`${docPath(kind, name)}/versions`),
  version: <T>(kind: DocKind, name: string | null, v: number) => request<{ doc: T }>(`${docPath(kind, name)}/versions/${v}`),
  diff: (kind: DocKind, name: string | null, from: number, to: number) => request<{ changes: Change[] }>(`${docPath(kind, name)}/diff?from=${from}&to=${to}`),
  check: <T>(kind: DocKind, name: string | null, doc: T) => request<{ valid: boolean; errors: string[]; changes: Change[]; base_version: number }>(`${docPath(kind, name)}/check`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ doc }) }),
  put: <T>(kind: DocKind, name: string | null, doc: T, note: string, base: number | null) => write<WriteResult>(docPath(kind, name), 'PUT', { doc, note, base_version: base }),
  restore: (kind: DocKind, name: string | null, version: number, note?: string) => write<WriteResult>(`${docPath(kind, name)}/restore`, 'POST', { version, note }),
  setMode: (name: string, mode: ProfileMode, note: string, base: number) => write<WriteResult>(`/v1/conductor/profiles/${enc(name)}/mode`, 'POST', { mode, note, base_version: base }),
  roles: () => request<{ version: number; doc: Record<string, RoleDoc>; meta: VersionMeta; health: Record<string, RoleHealth> }>('/v1/conductor/roles?fresh=1'),
  roleTest: (role: string) => request<{ role: string; ok: boolean; ms?: number; question?: string; answer?: StageAnswer; error?: string }>(`/v1/conductor/roles/${enc(role)}/test`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' }),
  scan: () => request<{ skills: SkillFound[] }>('/v1/conductor/packages/scan'),
  classify: (text: string, profile: string | null, tools = true) => request<Explain>('/v1/conductor/classify', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text, profile, tools }) }),
  decisions: (p: { since?: number; limit?: number; session?: string | null; before?: number } = {}) => {
    const q = new URLSearchParams();
    if (p.since) q.set('since', String(p.since));
    q.set('limit', String(p.limit ?? 100));
    if (p.session) q.set('session', p.session);
    if (p.before) q.set('before', String(p.before));
    return request<{ decisions: DecisionRow[] }>(`/v1/conductor/decisions?${q}`);
  },
  decision: (id: string) => request<{ decision: DecisionRow }>(`/v1/conductor/decisions/${enc(id)}`),
  label: (id: string, label: 'right' | 'wrong' | 'clear', note = '') => write<{ id: string; label: string | null }>(`/v1/conductor/decisions/${enc(id)}/label`, 'POST', { label, note }),
  events: (p: { since?: number; limit?: number; session?: string | null; task?: string | null; mode?: string | null; acted?: boolean } = {}) => {
    const q = new URLSearchParams();
    if (p.since) q.set('since', String(p.since));
    q.set('limit', String(p.limit ?? 100));
    if (p.session) q.set('session', p.session);
    if (p.task) q.set('task', p.task);
    if (p.mode) q.set('mode', p.mode);
    if (p.acted) q.set('acted', '1');
    return request<{ events: MonitorEvent[]; monitor: boolean }>(`/v1/conductor/events?${q}`);
  },
  event: (id: string) => request<{ event: MonitorEvent }>(`/v1/conductor/events/${enc(id)}`),
  stats: (hours = 24) => request<Stats>(`/v1/conductor/stats?hours=${hours}`),
  sessions: () => request<{ sessions: SessionRow[]; live: LiveSession[] }>('/v1/conductor/sessions'),
  session: (key: string) => request<{ session: string; live: LiveSession | null; turns: DecisionRow[]; events: MonitorEvent[] }>(`/v1/conductor/sessions/${enc(key)}`),
  exportJson: (key: string) => request<{ session: string; turns: number; interventions: number; hidden_texts: number; summary: string; injections: Intervention[] }>(`/v1/conductor/sessions/${enc(key)}/export?format=json`),
  exportUrl: (key: string, format: 'json' | 'md') => `${apiBase()}/v1/conductor/sessions/${enc(key)}/export?format=${format}`,
  evals: () => request<{ reports: EvalReport[]; engine: { version: string | null; code_sha: string | null } }>('/v1/conductor/evals'),
};
