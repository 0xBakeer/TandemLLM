// The mock proxy (contract 1.3): /v1/config/* over an in-memory registry, as proxy/config_api.py
// answers it. Two backends to start: a local vLLM and DeepSeek with its URL and key set by the
// environment (locked fields). POST /__mock/backends {"empty": true} empties it (the Setup screen);
// {"off": true} answers /v1/config as an engine without --backends.

import { engineBlock } from './engines.ts';

type J = Record<string, unknown>;

const KINDS = {
  generic: { cloud: false, url: null, inject_usage: true, timeouts: { connect: 5, first_byte: 600, idle: 120 } },
  tandemllm: { cloud: false, url: null, inject_usage: false, timeouts: { connect: 5, first_byte: 600, idle: 120 } },
  vllm: { cloud: false, url: null, inject_usage: true, timeouts: { connect: 5, first_byte: 600, idle: 120 } },
  sglang: { cloud: false, url: null, inject_usage: true, timeouts: { connect: 5, first_byte: 600, idle: 120 } },
  openai: { cloud: true, url: 'https://api.openai.com/v1', inject_usage: true, timeouts: { connect: 10, first_byte: 120, idle: 60 } },
  deepseek: { cloud: true, url: 'https://api.deepseek.com/v1', inject_usage: true, timeouts: { connect: 10, first_byte: 120, idle: 60 } },
} as const;
type Kind = keyof typeof KINDS;
const GROUPS = ['mode', 'class', 'effort', 'packages', 'side', 'loops', 'e1', 'tools', 'compaction', 'states'];

// P4: the harness's defaults by kind (proxy/harness.py DEFAULTS) and what each group costs or loses
const DEFAULTS: Record<'local' | 'cloud', Record<string, string>> = {
  local: { class: 'on', effort: 'on', packages: 'on', side: 'on', loops: 'on', e1: 'on', tools: 'on', compaction: 'on', states: 'on' },
  cloud: { class: 'on', effort: 'on', packages: 'off', side: 'on', loops: 'on', e1: 'off', tools: 'shadow', compaction: 'on', states: 'on' },
};
const RANK: Record<string, number> = { off: 0, shadow: 1, on: 2 };

function effective(b: J, m: J): Record<string, string> {
  const f = (m.features ?? {}) as Record<string, string>;
  const mode = f.mode && f.mode !== 'inherit' ? f.mode : String((b.harness as J).mode);
  const base = DEFAULTS[b.cloud ? 'cloud' : 'local'];
  const out: Record<string, string> = { mode };
  for (const g of GROUPS.slice(1)) {
    const v = f[g] && f[g] !== 'inherit' ? f[g] : base[g];
    out[g] = RANK[v] <= RANK[mode] ? v : mode;
  }
  return out;
}

function degradations(b: J): Record<string, string> {
  const cloud = !!b.cloud;
  const cont = b.kind === 'vllm' || b.kind === 'sglang';
  return {
    class: `One short extra request to this backend per new user message${cloud ? ' (a few hundred prompt tokens, billed)' : ''}.`,
    effort: b.kind === 'openai' ? 'Effort through reasoning_effort, on the o-series and gpt-5 models only; no hard cap.' : b.kind === 'deepseek' ? 'Thinking on or off only (thinking.type); no budget, no cap.' : 'Thinking on or off through chat_template_kwargs; an effort level is a word, never a hard cap.',
    packages: `The note sits at the end of the user message, so the model can quote it; it is replayed at the same message index, so the backend's prompt cache keeps its match${cloud ? ', and it costs input tokens every turn.' : '.'}`,
    side: 'No loss: the cap and thinking off go into the request, which saves tokens.',
    loops: cont ? 'Loops are read from the streamed words (thresholds measured for words); a close is a second request that continues the reasoning.' : 'Loops are read from the streamed words and journaled; this backend cannot continue an assistant message, so a reasoning loop is journaled, never closed. The tool-argument guard stops the stream.',
    e1: `An empty answer costs one more full request${cloud ? ', billed.' : '.'}`,
    tools: `The checks are free; a regenerate is one more full request${cloud ? ', billed' : ''}, and the client never sees the first one.`,
    compaction: "The summary keeps the TandemLLM section; the backend reads the client's whole history cold (the engine's continuation is engine-only).",
    states: "Template lines unless a narrator role names a backend; no prefill progress (the backend's prefill is not visible).",
  };
}

function feats(): Record<string, string> {
  return Object.fromEntries(GROUPS.map((g) => [g, 'inherit']));
}

function model(backend: string, id: string, price: J | null = null, ctx: number | null = 32768): J {
  return { backend, model: id, public_id: null, enabled: true, present: true, source: 'discovered', discovered_at: Date.now(), context_length: ctx, price, fallback: null, features: feats(), locks: {} };
}

function backend(id: string, kind: Kind, url: string, extra: J = {}): J {
  const now = Date.now();
  return {
    id, kind, label: id, url, enabled: true, models_mode: 'discover', models: [], fallback: [], max_inflight: 64,
    timeouts: { ...KINDS[kind].timeouts }, inject_usage: KINDS[kind].inject_usage, harness: { mode: 'off', profile: null }, cloud: KINDS[kind].cloud,
    host: { mode: 'auto', url: null, dcgm_url: null },
    created_at: now - 86_400_000, updated_at: now - 3_600_000, updated_by: 'admin',
    key: { key_set: false, last4: null, sha256_6: null, source: null, readable: null },
    locks: {}, health: { state: 'up', since: now / 1000 - 7200, last_check: now / 1000, last_error: null, latency_ms: 3.1, models_at: now / 1000 },
    in_flight: 0, capabilities: {}, routes: ['/v1/chat/completions', '/v1/completions', '/v1/embeddings', '/v1/responses'], cost_today_micros: null,
    ...extra,
  };
}

export const SEED_PRICES = {
  as_of: '2026-10-04',
  note: 'mock seed',
  providers: { deepseek: { 'deepseek-chat': { input: 0.27, cached_input: 0.07, cache_write: null, output: 1.1 } } },
};

export class MockBackends {
  backends: J[] = [];
  models: J[] = [];
  rules: J[] = [];
  users: J[] = [];
  off = false;
  private nextUser = 1;
  private nextRule = 1;

  constructor() {
    this.reset();
  }

  reset(): void {
    this.off = false;
    this.backends = [
      backend('vllm1', 'vllm', 'http://vllm.local:8000/v1', { label: 'vLLM (Qwen3.5-4B)', host: { mode: 'collector', url: 'http://vllm.local:9108', dcgm_url: null }, harness: { mode: 'on', profile: 'coding' } }),
      backend('sglang1', 'sglang', 'http://127.0.0.1:30000/v1', { label: 'SGLang (Qwen3.5-0.8B)' }),
      backend('deepseek', 'deepseek', 'https://api.deepseek.com/v1', {
        label: 'DeepSeek',
        key: { key_set: true, last4: 'a1b2', sha256_6: '9f86d0', source: 'env', readable: true },
        locks: { kind: 'TANDEM_BACKEND_DEEPSEEK_KIND', url: 'TANDEM_BACKEND_DEEPSEEK_URL', key: 'TANDEM_BACKEND_DEEPSEEK_KEY' },
        cost_today_micros: 41_230,
      }),
    ];
    this.models = [
      model('vllm1', 'Qwen/Qwen3.5-4B'),
      model('sglang1', 'Qwen/Qwen3.5-0.8B'),
      model('deepseek', 'deepseek-chat', { input: 0.27, cached_input: 0.07, cache_write: null, output: 1.1, currency: 'USD', updated_at: Date.now() }, 131072),
      model('deepseek', 'deepseek-reasoner', null, 131072),
    ];
    this.rules = [{ id: 1, position: 0, pattern: 'deepseek-*', backend: 'deepseek', enabled: true, source: 'env', variable: 'TANDEM_ROUTE_1' }];
    this.users = [];
    this.nextUser = 1;
    this.nextRule = 2;
  }

  empty(): void {
    this.backends = [];
    this.models = [];
    this.rules = [];
  }

  status(): J[] {
    return this.backends.map((b) => {
      const h = b.health as J;
      return {
        id: b.id, kind: b.kind, label: b.label, enabled: b.enabled, cloud: b.cloud, state: b.enabled ? h.state : 'disabled',
        since: null, last_error: h.last_error, latency_ms: h.latency_ms, in_flight: b.in_flight, max_inflight: b.max_inflight,
        models: this.models.filter((m) => m.backend === b.id).length, harness_mode: (b.harness as J).mode, cost_today_micros: b.cost_today_micros ?? null,
        scrape_age_s: (engineBlock(b.kind as string)?.scrape_age_s as number | null | undefined) ?? null, plugin_age_s: null, engine: engineBlock(b.kind as string),
      };
    });
  }

  private view(b: J): J {
    const rows = this.models.filter((m) => m.backend === b.id).map((m) => ({
      ...m, effective: effective(b, m),
      harness_cost_day: b.id === 'deepseek' && m.model === 'deepseek-chat' ? { class: { requests: 41, cost_micros: 1_230 } } : {},
    }));
    return { ...b, engine: engineBlock(b.kind as string), degradations: degradations(b), model_rows: rows };
  }

  overview(): J {
    return {
      contract_version: '1.0', setup_needed: this.backends.length === 0, key_source: 'file', config_path: '~/.tandemllm/config.sqlite3', kinds: KINDS, feature_groups: GROUPS,
      backends: this.backends.map((b) => this.view(b)), rules: this.rules, users: this.users, users_active: this.users.some((u) => u.enabled),
      engine: { model: 'TandemLLM', base_model: 'qwen38-spark-engine', embedded: true }, generation: 1,
      harness: { conductor: true, profiles: ['chat', 'coding', 'general', 'hermes', 'overnight'], defaults: DEFAULTS },
    };
  }

  /** One /v1/config request: [status, body] (body null for a 204). */
  handle(method: string, path: string, body: J | null): [number, J | null] {
    if (this.off) return [404, { error: { type: 'not_found', message: 'the proxy is off on this server (--backends off)' } }];
    const parts = path.replace(/^\/v1\/config\/?/, '').split('/').filter(Boolean).map(decodeURIComponent);
    const bad = (message: string, field?: string, variable?: string, status = 422): [number, J] => [status, { error: { type: 'invalid_config', message, ...(field ? { field } : {}), ...(variable ? { variable } : {}) } }];
    if (!parts.length && method === 'GET') return [200, this.overview()];
    if (parts[0] === 'test' && method === 'POST') {
      const url = String(body?.url ?? '');
      if (/down|:1\//.test(url)) return [200, { ok: false, status: null, latency_ms: null, error: 'connect to the host failed: Connection refused', models: [] }];
      return [200, { ok: true, status: 200, latency_ms: 4.2, error: null, models: body?.kind === 'deepseek' ? ['deepseek-chat', 'deepseek-reasoner'] : ['Qwen/Qwen3.5-4B', 'Qwen/Qwen3.5-9B'] }];
    }
    if (parts[0] === 'prices' && parts[1] === 'seed') return [200, SEED_PRICES];
    if (parts[0] === 'rules' && method === 'PUT') {
      const rules = (body?.rules ?? []) as J[];
      for (const r of this.rules.filter((x) => x.source === 'env')) {
        const sent = rules.find((x) => x.id === r.id);
        if (!sent || sent.pattern !== r.pattern || sent.backend !== r.backend) return bad(`the rule set by ${r.variable} cannot change here`, 'rules', String(r.variable));
      }
      this.rules = rules.map((r, i) => {
        const env = this.rules.find((x) => x.id === r.id && x.source === 'env');
        return env ? { ...env, position: i } : { id: this.nextRule++, position: i, pattern: r.pattern, backend: r.backend, enabled: r.enabled !== false, source: 'gui', variable: null };
      });
      return [200, { rules: this.rules }];
    }
    if (parts[0] === 'users') {
      if (parts.length === 1 && method === 'POST') {
        const label = String(body?.label ?? '');
        if (!/^[A-Za-z0-9][A-Za-z0-9 ._@-]{0,63}$/.test(label)) return bad('label does not match', 'label');
        if (this.users.some((u) => u.label === label)) return bad(`a user '${label}' exists already`, 'label', undefined, 409);
        const key = 'tdm-' + Math.random().toString(36).slice(2) + Math.random().toString(36).slice(2);
        const user = { id: this.nextUser++, label, key_hint: key.slice(-4), role: 'user', enabled: true, created_at: Date.now(), last_seen_at: null, source: 'database' };
        this.users.push(user);
        return [201, { user, key, note: 'shown once' }];
      }
      const u = this.users.find((x) => String(x.id) === parts[1]);
      if (!u) return [404, { error: { type: 'not_found', message: 'no user' } }];
      if (method === 'PATCH') {
        u.enabled = body?.enabled !== false;
        return [200, u];
      }
      if (method === 'DELETE') {
        this.users = this.users.filter((x) => x !== u);
        return [204, null];
      }
    }
    if (parts[0] === 'backends') {
      if (parts.length === 1 && method === 'POST') {
        const id = String(body?.id ?? '');
        if (!/^[a-z0-9][a-z0-9_-]{0,31}$/.test(id)) return bad('id does not match ^[a-z0-9][a-z0-9_-]{0,31}$', 'id');
        if (this.backends.some((b) => b.id === id)) return bad(`a backend '${id}' exists already`, 'id', undefined, 409);
        const kind = (body?.kind ?? 'generic') as Kind;
        const url = String(body?.url || KINDS[kind].url || '');
        if (!/^https?:\/\//.test(url)) return bad('url does not match', 'url');
        const key = body?.key ? String(body.key) : null;
        const b = backend(id, kind, url, {
          label: body?.label || id, max_inflight: body?.max_inflight ?? 64, fallback: body?.fallback ?? [], models: body?.models ?? [],
          key: key ? { key_set: true, last4: key.slice(-4), sha256_6: 'c0ffee', source: 'database', readable: true } : { key_set: false, last4: null, sha256_6: null, source: null, readable: null },
        });
        this.backends.push(b);
        this.models.push(model(id, kind === 'deepseek' ? 'deepseek-chat' : 'Qwen/Qwen3.5-9B'));
        return [201, { ...this.view(b), test: { state: 'up', latency_ms: 4.2, error: null } }];
      }
      const b = this.backends.find((x) => x.id === parts[1]);
      if (!b) return [404, { error: { type: 'not_found', message: `no backend '${parts[1]}'` } }];
      const rest = parts.slice(2);
      if (!rest.length && method === 'GET') return [200, this.view(b)];
      if (!rest.length && method === 'PATCH') {
        const locks = b.locks as Record<string, string>;
        for (const [k, v] of Object.entries(body ?? {})) {
          if (locks[k] && (k === 'key' || JSON.stringify(v) !== JSON.stringify(b[k]))) return bad(`${k} is set by ${locks[k]}; change the variable instead`, k, locks[k]);
        }
        for (const [k, v] of Object.entries(body ?? {})) {
          if (k === 'key') b.key = v ? { key_set: true, last4: String(v).slice(-4), sha256_6: 'c0ffee', source: 'database', readable: true } : { key_set: false, last4: null, sha256_6: null, source: null, readable: null };
          else b[k] = v;
        }
        if (!b.label) b.label = b.id;
        b.updated_at = Date.now();
        return [200, this.view(b)];
      }
      if (!rest.length && method === 'DELETE') {
        if (Object.keys(b.locks as J).length) return bad(`backend '${b.id}' is declared by TANDEM_BACKENDS; remove it there`, undefined, 'TANDEM_BACKENDS');
        this.backends = this.backends.filter((x) => x !== b);
        this.models = this.models.filter((m) => m.backend !== b.id);
        this.rules = this.rules.filter((r) => r.backend !== b.id);
        return [204, null];
      }
      if (rest[0] === 'test' && method === 'POST') return [200, { ok: true, latency_ms: 3.0, error: null, models: this.models.filter((m) => m.backend === b.id).map((m) => m.model) }];
      if (rest[0] === 'models' && rest.length > 1) {
        const id = rest.slice(1).join('/');
        let m = this.models.find((x) => x.backend === b.id && x.model === id);
        if (method === 'PUT') {
          if (!m) {
            m = model(String(b.id), id);
            m.source = 'manual';
            this.models.push(m);
          }
          for (const [k, v] of Object.entries(body ?? {})) {
            if (k === 'features') m.features = { ...(m.features as J), ...(v as J) };
            else if (k === 'price') m.price = v ? { ...(v as J), updated_at: Date.now() } : null;
            else m[k] = v;
          }
          return [200, m];
        }
        if (method === 'DELETE' && m) {
          this.models = this.models.filter((x) => x !== m);
          return [204, null];
        }
      }
    }
    return [404, { error: { type: 'not_found', message: `no route ${method} ${path}` } }];
  }
}
