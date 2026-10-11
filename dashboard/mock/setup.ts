// The setup guide's mock server (/v1/setup/*, VIS-78/80/82): the real Qwen3.8-27B registry entry
// (mock/setup/qwen3.8-27b.json, models/qwen3.8-27b/model.yaml as JSON), the box, the estimate from
// the dashboard's own formula, the plan, apply jobs over SSE, Wren's status and a scripted chat
// with tool calls, and the reset levels. Never part of dist/.
//
// POST /__mock/setup switches it (every key optional):
//   scenario: ok | rollback | slow | restart | fail    the next apply job (restart: a gap where the
//                                                     origin does not answer, gap_s long, default 6)
//   assistant: loaded | not_loaded                    Wren's state now
//   refuse: bool (a load is refused for memory), noModel: bool (no Qwen3.5 on disk),
//   loadMs: n (how long a load takes), fallback: bool (the 0.8B)
//   hfToken: bool, password: bool, busy: n (requests running), setupMode: bool (/health mode)
//   resetPartial: bool (two items fail), resetEmpty: bool (nothing to delete at level 3),
//   shadowOn: bool (the shadow reader runs: its sessions are kept, `skip`, and reported in `failed`)
//   waitIdle: bool (the next job's write_config first waits for 2 requests to finish)
//   estimateDown: bool, planDown: bool, models: ok | empty | error | broken
//   active: <profile id> (what the box runs), secretsExist: bool (generate is refused, secrets_exist)
//   unauthorized: bool (an admin password is set: every /v1/setup route answers 401 until
//   POST /v1/dashboard/session {token: QSE_MOCK_TOKEN or "mock"} signs in), loginOff: bool (with
//   unauthorized: the dashboard login is off, so that POST is a 404 as on the server)
//   current: a partial selection: what the box runs (GET /v1/setup/models/{id}/current, the
//   wizard's seed; its profile becomes `active`)
//   reset: true (all of the above back to the start)
//
// In setup mode every other /v1 route answers 503 setup_mode (mock/server.ts), as server/app.py's
// SetupHandler does.

import type { IncomingMessage, ServerResponse } from 'node:http';
import type {
  SetupAssistantStatus,
  SetupChatMessage,
  SetupChatRequest,
  SetupDevice,
  SetupEnvChange,
  SetupJob,
  SetupJobEvent,
  SetupJobStep,
  SetupJobStepId,
  SetupModel,
  SetupPlan,
  SetupResetLevel,
  SetupSelection,
} from '../src/api/types.ts';
import { checkSelection, estimate, SelectionError } from '../src/setup/estimate.ts';
import { defaults } from '../src/setup/state.ts';
import qwen from './setup/qwen3.8-27b.json';

const MODEL = qwen as unknown as SetupModel;
const GIB = 2 ** 30;

type Scenario = 'ok' | 'rollback' | 'slow' | 'restart' | 'fail';

interface MockJob {
  job: SetupJob;
  created: number;
  timeline: { at: number; ev: SetupJobEvent }[];
  gap: { from: number; to: number } | null;
}

interface Options {
  scenario: Scenario;
  gapMs: number;
  assistant: 'loaded' | 'not_loaded' | 'loading';
  refuse: boolean;
  noModel: boolean;
  fallback: boolean;
  loadMs: number;
  hfToken: boolean;
  password: boolean;
  busy: number;
  setupMode: boolean;
  resetPartial: boolean;
  resetEmpty: boolean;
  shadowOn: boolean;
  waitIdle: boolean;
  secretsExist: boolean;
  estimateDown: boolean;
  planDown: boolean;
  models: 'ok' | 'empty' | 'error' | 'broken';
  active: string;
  /** a box with an admin password and no session: every /v1/setup route answers 401 */
  unauthorized: boolean;
  loginOff: boolean;
}

const START: Options = {
  scenario: 'ok',
  gapMs: 6000,
  assistant: 'not_loaded',
  refuse: false,
  noModel: false,
  fallback: false,
  loadMs: 2600,
  hfToken: true,
  password: false,
  busy: 0,
  setupMode: false,
  resetPartial: false,
  resetEmpty: false,
  shadowOn: false,
  waitIdle: false,
  secretsExist: false,
  estimateDown: false,
  planDown: false,
  models: 'ok',
  active: 'exl3-4.5',
  unauthorized: false,
  loginOff: false,
};

const TOKEN = process.env.QSE_MOCK_TOKEN ?? 'mock';
/** server/setup/apply.py CLIENT_FACING: the changes every client notices, warned about in Review */
const CLIENT_FACING: Record<string, string> = {
  HOST: 'the engine listens on another address; clients and the reverse proxy may not reach it any more.',
  PORT: 'clients, the reverse proxy and the watchdog must use the new port.',
  SERVED_MODEL: 'clients that ask for the old model name get an error until they use the new one.',
  QSE_BACKENDS: 'the model names routed to other servers or cloud APIs change with it.',
};
function clientFacing(changes: SetupEnvChange[]): string[] {
  return changes.filter((c) => CLIENT_FACING[c.key]).map((c) => `${c.key} changes from ${c.current ?? 'unset'} to ${c.next ?? 'unset'}: ${CLIENT_FACING[c.key]}`);
}
/** server/setup_api.py PLAN_STATUS: 412 for what the box lacks, 409 for the rest */
const PLAN_STATUS: Record<string, number> = { missing_token: 412, missing_assets: 412, disk_full: 412, no_overlay_hook: 412 };
/** the checkout the mock box runs from (run/setup.env lives there) */
const REPO = '~/tandem-live';

/** server/setup/apply.py STEPS */
const STEP_LABELS: Record<SetupJobStepId, string> = {
  check: 'Check the selection, the disk and the box',
  unload_assistant: 'Unload Wren',
  fetch: 'Download the weights',
  write_config: 'Write the configuration (run/setup.env)',
  switch: 'Restart the engine with the new setup',
  smoke: 'Check that the new engine answers',
  rollback: 'Roll back',
};

export class MockSetup {
  o: Options = { ...START };
  onDisk: Record<string, boolean> = { 'exl3-4.0': true, 'exl3-4.5': true, 'exl3-6.0': true, nvfp4: false, 'exl3-2.5': false, 'exl3-3.0': false, 'exl3-3.5': false };
  loadStartedAt = 0;
  jobs = new Map<string, MockJob>();
  streams = new Set<ServerResponse>();
  /** the origin does not answer until then (an engine restart) */
  awayUntil = 0;
  awayFrom = 0;
  private seq = 0;
  /** dashboard sessions made by POST /v1/dashboard/session while `unauthorized` is on */
  sessions = new Set<string>();
  /** serve.env as it is now on the mock box */
  env: Record<string, string> = {};

  /** what the mock box runs, as GET .../current answers it */
  box: SetupSelection = defaults(MODEL);

  constructor() {
    this.env = this.envFor(this.box);
  }

  reset(): void {
    this.o = { ...START };
    this.jobs.clear();
    this.sessions.clear();
    this.awayUntil = 0;
    this.awayFrom = 0;
    this.box = defaults(MODEL);
    this.env = this.envFor(this.box);
  }

  control(b: Record<string, unknown>): Record<string, unknown> {
    if (b.reset) this.reset();
    const o = this.o as unknown as Record<string, unknown>;
    for (const k of Object.keys(START)) if (k in b) o[k] = b[k];
    if (typeof b.gap_s === 'number') this.o.gapMs = b.gap_s * 1000;
    if (b.assistant === 'loading') this.loadStartedAt = Date.now();
    if (b.current && typeof b.current === 'object') {
      // the box runs another setup than the recommended one (a partial selection over it)
      const c = b.current as Partial<SetupSelection>;
      const d = defaults(MODEL);
      this.box = checkSelection(MODEL, { ...d, ...c, components: { ...d.components, ...c.components }, methods: { ...d.methods, ...c.methods }, advanced: { ...d.advanced, ...c.advanced }, model: MODEL.id });
      this.o.active = this.box.profile;
      this.env = this.envFor(this.box);
    } else if (typeof b.active === 'string') {
      this.box = { ...this.box, profile: b.active };
    }
    return { ...this.o };
  }

  /** an engine restart in progress: nothing answers */
  away(now = Date.now()): boolean {
    return now >= this.awayFrom && now < this.awayUntil;
  }

  // ---- the payloads -------------------------------------------------------------------------

  status(): SetupAssistantStatus {
    const o = this.o;
    const extra = { borrowed: false, window: 8192, loading_s: null as number | null, stats: {} as Record<string, number> };
    if (o.noModel) return { name: 'Wren', model: null, fallback: false, state: 'off', reason: 'neither Qwen3.5-2B nor Qwen3.5-0.8B is in ~/.qwen38-spark-engine/models. Download it with: hf download Qwen/Qwen3.5-2B --local-dir ~/.qwen38-spark-engine/models/qwen3.5-2b', load_s: null, device_gib: null, claim_gib: null, can_load: false, ...extra };
    let state = o.assistant as SetupAssistantStatus['state'];
    if (state === 'loading' && Date.now() - this.loadStartedAt >= o.loadMs) {
      this.o.assistant = 'loaded';
      state = 'loaded';
    }
    return {
      name: 'Wren',
      model: o.fallback ? 'qwen3.5-0.8b' : 'qwen3.5-2b',
      fallback: o.fallback,
      state,
      reason: null,
      load_s: state === 'loaded' ? o.loadMs / 1000 : null,
      device_gib: state === 'loaded' ? 4.4 : null,
      claim_gib: 4.4,
      can_load: state !== 'loaded' && state !== 'loading',
      ...extra,
      loading_s: state === 'loading' ? Math.round((Date.now() - this.loadStartedAt) / 100) / 10 : null,
    };
  }

  memAvailable(): number {
    const used = estimate(MODEL, { ...defaults(MODEL), profile: this.o.active }).total_gb - 13.0 + 7.2;
    const wobble = Math.sin(Date.now() / 7000) * 0.4;
    return Math.round((MODEL.devices[0].total_gib - used + wobble) * 100) / 100;
  }

  device(): SetupDevice {
    const o = this.o;
    return {
      contract_version: '1.0',
      mode: o.setupMode ? 'setup' : 'serving',
      preset: 'dgx-spark',
      mem_total_gb: MODEL.devices[0].total_gib,
      mem_available_gb: this.memAvailable(),
      gpu: { name: 'NVIDIA GB10', util_pct: o.busy ? 92 : 3, temp_c: o.busy ? 71 : 44, power_w: o.busy ? 88 : 14 },
      disk: { path: '/home/bakeer/.qwen38-spark-engine', free_gb: 1412.6, total_gb: 3726.0 },
      engine: { version: '0.6.2', profile: o.setupMode ? null : o.active, uptime_s: 22_841, running: o.busy, waiting: 0, busy: o.busy > 0, mode: o.setupMode ? 'setup' : 'serving' },
      active_profile: o.setupMode ? null : o.active,
      profiles_on_disk: { ...this.onDisk },
      password_set: o.password,
      hf_token_present: o.hfToken,
      admin_password_set: o.password,
      assistant: this.status(),
    };
  }

  envFor(sel: SetupSelection): Record<string, string> {
    const env: Record<string, string> = {};
    for (const c of MODEL.components) {
      const on = c.required || (sel.components[c.id] ?? c.default);
      for (const [k, v] of Object.entries(c.knobs)) {
        const val = on ? v.on : v.off;
        if (val != null) env[k] = val;
      }
    }
    const p = MODEL.parallelism;
    env[p.knobs.slots] = String(sel.slots);
    env[p.knobs.window] = String(sel.window);
    env[p.knobs.side_window] = String(sel.side_window ?? sel.window);
    env[p.knobs.kv_pool] = sel.kv_pool ? 'on' : 'off';
    env[p.knobs.floor_gb] = String(sel.floor_gb);
    for (const [k, v] of Object.entries(p.fixed ?? {})) env[k] = v;
    for (const f of [...MODEL.methods, ...MODEL.advanced]) {
      if (!f.knob) continue;
      const v = (MODEL.methods.includes(f) ? sel.methods : sel.advanced)[f.id] ?? f.recommended;
      env[f.knob] = String(v);
    }
    return env;
  }

  plan(sel: SetupSelection): SetupPlan {
    const next = this.envFor(sel);
    const from = (key: string): string => {
      for (const c of MODEL.components) if (key in c.knobs) return c.label;
      for (const f of [...MODEL.methods, ...MODEL.advanced]) if (f.knob === key) return f.label;
      const pk = Object.entries(MODEL.parallelism.knobs).find(([, v]) => v === key);
      const names: Record<string, string> = { slots: 'Parallel chats', window: 'Longest conversation', side_window: 'Longest conversation in the extra chats', kv_pool: 'Shared memory pool', floor_gb: 'Memory kept free' };
      return pk ? names[pk[0]] : 'Parallel chats';
    };
    const changes: SetupEnvChange[] = [];
    for (const key of new Set([...Object.keys(this.env), ...Object.keys(next)])) {
      const cur = this.env[key] ?? null;
      const nx = next[key] ?? null;
      if (cur !== nx) changes.push({ key, current: cur, next: nx, from: from(key) });
    }
    const est = estimate(MODEL, sel, this.memAvailable());
    const prof = MODEL.profiles.find((p) => p.id === sel.profile)!;
    const onDisk = !!this.onDisk[prof.id];
    const errors: string[] = [];
    const codes: string[] = [];
    if (!est.fits) {
      errors.push(`The selection needs ${est.total_gb.toFixed(2)} GiB, ${(est.total_gb - (est.budget_gb ?? 0)).toFixed(2)} GiB more than the ${(est.budget_gb ?? 0).toFixed(2)} GiB there is.`);
      codes.push('over_budget');
    }
    if (!onDisk && prof.hf_private && !this.o.hfToken) {
      errors.push(`${prof.hf_repo} is private and no Hugging Face token is set. Add one with \`hf auth login\` on the box, or set HF_TOKEN, then try again.`);
      codes.push('missing_token');
    }
    const generate = sel.advanced.admin_password === 'generate';
    if (generate && this.o.secretsExist) {
      errors.push('secrets.env already exists. ops/make-secrets.sh --rotate --admin would replace the metrics token too, so run it on the box yourself.');
      codes.push('secrets_exist');
    }
    const port = Number(sel.advanced.port ?? 8000);
    // server/setup/apply.py plan(): the commands for reading, in the runner's order
    const commands = [
      ...(onDisk ? [] : [`bash ops/use-profile.sh ${prof.id} --fetch-only   (about ${prof.disk_gb} GB from ${prof.hf_repo})`]),
      ...(port !== 8000 ? ['bash ops/stop.sh 120   (the engine on port 8000)'] : []),
      `write run/setup.env (${Object.keys(next).length} keys; the old one kept as a .bak)`,
      ...(generate ? ['bash ops/make-secrets.sh --admin'] : []),
      `bash ops/use-profile.sh ${prof.id}   (stop, start, smoke, roll back on failure)`,
      `smoke: GET http://127.0.0.1:${port}/health and one short chat`,
    ];
    return {
      contract_version: '1.0',
      ok: errors.length === 0,
      errors,
      error_codes: codes,
      warnings: [...clientFacing(changes), ...(this.o.password ? [] : ['No admin password is set: anyone who can reach this port can run the setup, apply it and reset the box. Generate one in the advanced step.']), ...(est.fits && !est.fits_worst ? [est.notes[0]] : [])],
      profile: { current: this.o.setupMode ? null : this.o.active, next: prof.id, on_disk: onDisk, download_gb: onDisk ? 0 : prof.disk_gb },
      changes,
      restart: true,
      commands,
      estimate: est,
      ports: { current: 8000, next: port },
    };
  }

  // ---- the apply job ------------------------------------------------------------------------

  startJob(sel: SetupSelection, scenario: Scenario = this.o.scenario, kind: SetupJob['kind'] = 'apply'): SetupJob {
    // a token_hex(8) as the server makes (apply.JOB_ID)
    const id = (Date.now().toString(16) + (++this.seq).toString(16).padStart(4, '0')).slice(-16).padStart(16, '0');
    const now = Date.now();
    const prof = sel.profile;
    const fetch = !this.onDisk[prof];
    const slow = scenario === 'slow' ? 3 : 1;
    const ids: SetupJobStepId[] = ['check', 'unload_assistant', 'fetch', 'write_config', 'switch', 'smoke', 'rollback'];
    const steps: SetupJobStep[] = ids.map((s) => ({ id: s, label: STEP_LABELS[s], state: 'pending', started_at: null, ended_at: null, detail: null }));
    const job: SetupJob = { id, state: 'queued', created_at: now / 1000, ended_at: null, profile: prof, steps, message: null, changes: this.plan(sel).changes, kind, rolled_back: false, failed_step: null, url: `http://<this box>:${sel.advanced.port ?? 8000}/dashboard/` };
    const tl: { at: number; ev: SetupJobEvent }[] = [];
    const sec = (ms: number) => (now + ms) / 1000;
    let t = 0;
    const step = (sid: SetupJobStepId, dur: number, logs: string[], end: 'done' | 'failed' | 'skipped' = 'done', detail: string | null = null) => {
      if (end === 'skipped') {
        tl.push({ at: t, ev: { type: 'step', step: { id: sid, label: STEP_LABELS[sid], state: 'skipped', started_at: null, ended_at: null, detail } } });
        return;
      }
      const start = t;
      tl.push({ at: t, ev: { type: 'step', step: { id: sid, label: STEP_LABELS[sid], state: 'running', started_at: sec(start), ended_at: null, detail: null } } });
      logs.forEach((l, i) => tl.push({ at: start + ((i + 1) * dur) / (logs.length + 1), ev: { type: 'log', line: l, ts: sec(start + ((i + 1) * dur) / (logs.length + 1)) } }));
      t += dur;
      tl.push({ at: t, ev: { type: 'step', step: { id: sid, label: STEP_LABELS[sid], state: end, started_at: sec(start), ended_at: sec(t), detail } } });
    };
    tl.push({ at: 0, ev: { type: 'job', job: { ...job, state: 'running' } } });
    step('check', 700 * slow, ['[setup] selection valid against models/qwen3.8-27b/model.yaml', `[setup] estimate ${estimate(MODEL, sel).total_gb} GiB of ${MODEL.devices[0].total_gib - sel.floor_gb} GiB`]);
    step('unload_assistant', 600 * slow, ['[wren] unloading qwen3.5-2b (4.4 GiB back)']);
    if (fetch) step('fetch', 2600 * slow, [`[profile] fetching ${prof} from ${MODEL.profiles.find((p) => p.id === prof)?.hf_repo}`, '[hf] 25 % of 18.4 GB', '[hf] 50 %', '[hf] 75 %', '[hf] 100 %, sha256 ok']);
    else step('fetch', 0, [], 'skipped', 'already on the box');
    if (this.o.waitIdle) {
      // apply.py _wait_idle: the step waits for the engine's requests before anything stops
      tl.push({ at: t, ev: { type: 'step', step: { id: 'write_config', label: STEP_LABELS.write_config, state: 'waiting', started_at: null, ended_at: null, detail: 'waiting for 2 requests to finish before the restart' } } });
      t += 2500;
    }
    step('write_config', 600 * slow, kind === 'reset' ? ['[setup] run/setup.env and 1 backup(s) removed'] : ['[setup] run/setup.env.bak-20261010T081200Z kept', `[setup] run/setup.env: ${Object.keys(this.envFor(sel)).length} keys`]);
    const switchStart = t;
    step('switch', 2800 * slow, [`[profile] switching ${this.o.active} -> ${prof}`, '[engine] stopping (draining 0 requests)', '[engine] starting', '[memory] load stages: weights, drafters, graphs', '[engine] ready on :8000']);
    let gap: MockJob['gap'] = null;
    if (scenario === 'restart') {
      gap = { from: now + switchStart + 800, to: now + switchStart + 800 + this.o.gapMs };
      // the rest of the job waits for the engine to come back
      const shift = this.o.gapMs;
      for (const e of tl) if (e.at > switchStart + 800) e.at += shift;
      t += shift;
    }
    if (scenario === 'rollback' || scenario === 'fail') {
      step('smoke', 1400 * slow, ['[smoke] GET /health 200', '[smoke] chat request: 500 internal error (CUDA graph capture failed)'], 'failed', 'the test request failed');
      if (scenario === 'rollback') {
        step('rollback', 1600 * slow, [`[profile] rolling back to ${this.o.active}`, '[engine] ready on :8000', '[smoke] ok']);
        // the job's message is why it failed (apply.py _fail); the previous profile is the page's to say
        tl.push({ at: t + 100, ev: { type: 'done', state: 'rolled_back', message: 'the chat request failed (HTTP 500)' } });
      } else {
        step('rollback', 0, [], 'skipped', null);
        tl.push({ at: t + 100, ev: { type: 'done', state: 'failed', message: 'The engine did not pass its check.' } });
      }
    } else {
      step('smoke', 1400 * slow, ['[smoke] GET /health 200', '[smoke] chat request 200, 41.2 tok/s']);
      step('rollback', 0, [], 'skipped', null);
      tl.push({ at: t + 100, ev: { type: 'done', state: 'done', message: kind === 'reset' ? `The shipped configuration runs, on ${prof}.` : `${prof} is live on port ${sel.advanced.port ?? 8000}.` } });
      const finish = t + 100;
      setTimeout(() => {
        this.o.active = prof;
        this.box = sel;
        this.onDisk[prof] = true;
        this.env = this.envFor(sel);
      }, finish).unref?.();
    }
    tl.sort((a, b) => a.at - b.at);
    this.jobs.set(id, { job, created: now, timeline: tl, gap });
    if (gap) {
      this.awayFrom = gap.from;
      this.awayUntil = gap.to;
      setTimeout(() => {
        for (const s of this.streams) s.end();
        this.streams.clear();
      }, gap.from - now).unref?.();
    }
    return { ...job };
  }

  /** the id of a job that has not ended, or null */
  runningJob(): string | null {
    for (const j of this.jobs.values()) if (!['done', 'failed', 'rolled_back'].includes(this.jobNow(j.job.id)!.state)) return j.job.id;
    return null;
  }

  /** the job as it is now (?format=json) */
  jobNow(id: string): SetupJob | null {
    const j = this.jobs.get(id);
    if (!j) return null;
    let job: SetupJob = j.job;
    const el = Date.now() - j.created;
    for (const e of j.timeline) {
      if (e.at > el) break;
      if (e.ev.type === 'job') job = e.ev.job;
      else if (e.ev.type === 'step') job = { ...job, steps: job.steps.map((s) => (s.id === (e.ev as { step: SetupJobStep }).step.id ? (e.ev as { step: SetupJobStep }).step : s)) };
      else if (e.ev.type === 'done') job = { ...job, state: e.ev.state, message: e.ev.message, ended_at: (j.created + e.at) / 1000 };
    }
    return job;
  }

  streamJob(req: IncomingMessage, res: ServerResponse, id: string): void {
    const j = this.jobs.get(id);
    if (!j) {
      res.writeHead(404, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: { type: 'not_found', message: `no job ${id}` } }));
      return;
    }
    res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-store', Connection: 'keep-alive' });
    const last = Number(req.headers['last-event-id'] ?? 0) || 0;
    let sent = last;
    this.streams.add(res);
    const pump = () => {
      const el = Date.now() - j.created;
      for (let i = sent; i < j.timeline.length; i++) {
        const e = j.timeline[i];
        if (e.at > el) break;
        res.write(`id: ${i + 1}\ndata: ${JSON.stringify(e.ev)}\n\n`);
        sent = i + 1;
        if (e.ev.type === 'done') {
          clearInterval(t);
          this.streams.delete(res);
          res.end();
          return;
        }
      }
    };
    const t = setInterval(pump, 150);
    pump();
    req.on('close', () => {
      clearInterval(t);
      this.streams.delete(res);
    });
  }

  // ---- reset --------------------------------------------------------------------------------

  /** server/setup/reset.py's levels on the mock box: the same labels, items, actions and words */
  levels(): SetupResetLevel[] {
    const E = '~/.qwen38-spark-engine';
    const item = (path: string, label: string, bytes: number, exists = true, more: Partial<SetupResetLevel['items'][number]> = {}): SetupResetLevel['items'][number] => ({ path, label, bytes: exists ? bytes : 0, exists, files: exists ? 12 : 0, at_least: false, kind: 'path', action: 'remove', note: null, restart: false, refused: null, ...more });
    const memory = (name: string, label: string) => item(name, label, 0, true, { kind: 'memory', action: 'engine', files: 0 });
    const empty = this.o.resetEmpty;
    const config = empty ? [] : [item(`${REPO}/run/setup.env`, "The setup guide's configuration", 2150, true, { files: 1 }), item(`${REPO}/run/setup.env.bak-20261010T081200Z`, 'An earlier one, kept as a backup', 2080, true, { files: 1 })];
    const caches = empty
      ? []
      : [
          item(`${E}/cache/disk-${this.o.active}`, "Disk cache and parked chats (the running profile's)", 11.2 * GIB, true, { action: 'engine', note: 'the engine clears it itself' }),
          item(`${E}/cache/disk-${this.o.active === 'exl3-6.0' ? 'exl3-4.5' : 'exl3-6.0'}`, 'Disk cache and parked chats', 3.1 * GIB),
          ...(this.o.shadowOn ? [item(`${E}/cache/shadow`, "The shadow reader's saved sessions", 0.6 * GIB, true, { action: 'skip', note: 'kept while the shadow reader runs' })] : []),
          item(`${E}/suffix`, 'The lookup store', 412 * 2 ** 20, true, { action: 'engine', note: 'the engine empties it itself' }),
          memory('state_store', 'The state store (memory)'),
          memory('response_cache', 'The response cache (memory)'),
        ];
    const weights = MODEL.profiles.filter((p) => this.onDisk[p.id] && p.id !== this.o.active).map((p) => item(`${E}/weights/${p.id}`, `Weights of ${p.id}`, p.disk_gb * 1e9));
    const lv = (level: 2 | 3 | 4 | 5, label: string, what: string, items: SetupResetLevel['items'], restart: boolean, word: string | null): SetupResetLevel => ({ level, label, what, items, bytes: items.reduce((a, i) => a + (i.refused ? 0 : i.bytes), 0), restart: restart || items.some((i) => i.restart), confirm_word: word });
    const rec = MODEL.profiles.find((p) => p.recommended)!.id;
    return [
      lv(2, 'Applied configuration', `The setup guide's configuration goes, and the engine restarts on the recommended weight profile. The recommended profile is ${rec}.`, config, true, null),
      lv(3, 'Caches and stored chats', 'Every cache on the SSD and in memory, and the conversations parked there. Answers stay right; the next long prompts are slower once.', caches, false, 'delete caches'),
      lv(4, 'Downloaded weights', 'The weights of every profile but the active one. They download again when you pick that profile.', weights, false, 'delete weights'),
      lv(5, 'Usage history and conductor data', "The usage history, the conductor's settings and journal, and the memory notes. This cannot be undone.", [item(`${E}/usage/ledger.sqlite3`, 'The usage history', 1.4 * GIB, true, { files: 1 }), item(`${E}/conductor`, "The conductor's settings and journal", 220 * 2 ** 20), item(`${E}/memory`, 'Memory notes and transcripts', 5.1 * GIB, true, { files: 50_000, at_least: true })], true, 'delete history'),
    ];
  }

  // ---- the chat -----------------------------------------------------------------------------

  chat(req: IncomingMessage, res: ServerResponse, body: SetupChatRequest): void {
    const st = this.status();
    if (st.state !== 'loaded') {
      // the server answers 200 with an error event (wren.py _turn)
      res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-store' });
      res.end(`data: ${JSON.stringify({ type: 'error', message: 'Wren is not loaded. Load it first, or use the page without chat.' })}\n\n`);
      return;
    }
    const msgs = body.messages ?? [];
    const last = msgs[msgs.length - 1];
    const lastUser = [...msgs].reverse().find((m) => m.role === 'user')?.content.toLowerCase() ?? '';
    const page = body.page;
    const script = this.script(lastUser, last, msgs, page?.visible as Record<string, unknown> | undefined, page?.step ?? 'models');
    res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-store' });
    let closed = false;
    req.on('close', () => (closed = true));
    const send = (o: unknown) => !closed && res.write(`data: ${JSON.stringify(o)}\n\n`);
    void (async () => {
      const wait = (ms: number) => new Promise((r) => setTimeout(r, ms));
      if (script.errorAfter === -1) {
        send({ type: 'error', message: 'Wren is answering in another tab; try again when that answer is done.' });
        res.end();
        return;
      }
      send({ type: 'state', state: 'thinking' });
      await wait(script.think ?? 650);
      send({ type: 'state', state: 'talking' });
      const words = script.text.split(/(?<= )/);
      for (let i = 0; i < words.length; i++) {
        if (closed) return;
        send({ type: 'text', delta: words[i] });
        if (script.errorAfter != null && i === script.errorAfter) {
          send({ type: 'error', message: 'The model stopped: out of memory while decoding.' });
          res.end();
          return;
        }
        await wait(script.wordMs ?? 34);
      }
      for (const c of script.calls ?? []) send({ type: 'tool_call', id: `call_${++this.seq}`, name: c.name, args: c.args });
      send({ type: 'done', finish: script.calls?.length ? 'tool_calls' : 'stop', sources: script.sources ?? [] });
      res.end();
    })();
  }

  private script(user: string, last: SetupChatMessage | undefined, msgs: SetupChatMessage[], visible: Record<string, unknown> | undefined, step: string): { text: string; calls?: { name: string; args: Record<string, unknown> }[]; sources?: { path: string; title: string }[]; errorAfter?: number; wordMs?: number; think?: number } {
    const title = String(visible?.title ?? step);
    if (last?.role === 'tool') {
      if (/loop/.test(user)) return { text: 'One more step. ', calls: [{ name: 'go_to_step', args: { n: 3 } }] };
      let res: { ok?: boolean; error?: string; result?: string } = {};
      try {
        res = JSON.parse(last.content);
      } catch {
        /* plain text */
      }
      if (res.ok === false) return { text: `That did not work: ${res.error ?? 'the page said no'} I will leave it as it is.` };
      return { text: 'Done. The memory bar at the top shows the new total. Press Undo on the note if you want it back.' };
    }
    if (/other tab/.test(user)) return { text: '', errorAfter: -1 };
    if (/error/.test(user)) return { text: 'Let me look at that for you, it takes a moment to', errorAfter: 6 };
    if (/loop/.test(user)) return { text: 'Opening the steps. ', calls: [{ name: 'go_to_step', args: { n: 2 } }] };
    if (/flux/.test(user)) return { text: 'Turning on flux. ', calls: [{ name: 'toggle', args: { component: 'flux', on: true } }] };
    if (/finish|install it|reset/.test(user)) return { text: 'I will start the install now. ', calls: [{ name: 'finish', args: {} }] };
    if (/vision/.test(user) && /off|disable|turn|no /.test(user) && !/ on\b/.test(user)) return { text: 'Vision lets the model read images and costs about 0.9 GiB. I will turn it off. ', calls: [{ name: 'toggle', args: { component: 'vision', on: false } }] };
    if (/vision/.test(user) && /\bon\b/.test(user)) return { text: 'Turning vision back on. ', calls: [{ name: 'toggle', args: { component: 'vision', on: true } }] };
    if (/2 slots|two slots|use 2|2 chats/.test(user)) return { text: 'Two parallel chats leave more room for each one. Setting it now. ', calls: [{ name: 'set_field', args: { step: 'parallelism', field: 'slots', value: 2 } }] };
    if (/size|memory|best|which/.test(user) && /weight|size|memory/.test(user))
      return { text: `Your box has ${this.memAvailable().toFixed(1)} GiB free right now. **EXL3 4.5** is the recommended size: it fits with room for 4 chats. If you want more speed, **EXL3 4.0** is a little faster and still very close to the original.`, sources: [{ path: 'models/qwen3.8-27b/model.yaml', title: 'Qwen3.8-27B registry' }] };
    if (/narrator/.test(user))
      return {
        text: 'The **narrator** is a small model (Qwen3.5-2B, about 3.6 GiB) that writes short status lines while the big model thinks, so you can see what it is doing.\n\nIt needs the conductor. When memory runs short it unloads first and loads again later.',
        sources: [{ path: 'docs/server.md', title: 'The server: roles' }, { path: 'models/qwen3.8-27b/model.yaml', title: 'Qwen3.8-27B registry' }],
      };
    if (/everything|long/.test(user))
      return {
        wordMs: 60,
        text: Array.from({ length: 12 }, (_, i) => `Point ${i + 1}: every tool on this page is optional except the draft models, and each one says what it costs.`).join(' '),
      };
    if (/det\b|sampled/.test(user)) return { text: '**det** checks a sampled request against the same tree a greedy request uses. It is the fastest, and the answers match what the model would sample alone.', sources: [{ path: 'docs/speculative-decoding.md', title: 'Speculative decoding' }] };
    if (/kl\b|what is kl/.test(user)) return { text: '**KL** measures how far the smaller weights drift from the original model. Lower is closer: 0.02 is very close, 0.2 is noticeably different.', sources: [{ path: 'docs/weight-profiles.md', title: 'Weight profiles' }] };
    return {
      text: `You are on **${title}**. ${step === 'components' ? 'The ticked tools are what we run on our own box. Each one says what it costs in memory, and the bar at the top adds them up.' : step === 'parallelism' ? 'Each slot is one chat at a time. More slots share the GPU, and the window is the longest conversation a slot can hold.' : step === 'weights' ? 'Smaller weights are faster and drift further from the original. 4.5 is the recommended balance.' : 'Ask me what a setting does, or tell me what you want and I will set it.'}`,
      sources: [{ path: 'docs/setup.md', title: 'The setup guide' }],
    };
  }

  // ---- the admin password (unauthorized: on) ------------------------------------------------

  signedIn(req: IncomingMessage): boolean {
    const m = /(?:^|;\s*)qse_dash=([^;]+)/.exec(req.headers.cookie ?? '');
    return !!m && this.sessions.has(decodeURIComponent(m[1]));
  }

  /** /v1/dashboard/session while the mock box has a password for the setup routes (server/app.py
   *  _session): POST signs in (204 and the cookie), 401 on a wrong token; 404 with the login off */
  async session(req: IncomingMessage, res: ServerResponse, readBody: () => Promise<string>): Promise<void> {
    const json = (code: number, body: unknown) => {
      res.writeHead(code, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' });
      res.end(JSON.stringify(body));
    };
    const method = req.method ?? 'GET';
    if (this.o.loginOff) {
      if (method === 'GET') return json(200, { contract_version: '1.1', authenticated: true, login: false, password_set: true, expires_at: null });
      return json(404, { error: { type: 'not_found', message: 'the dashboard login is off' } });
    }
    if (method === 'POST') {
      const b = JSON.parse((await readBody()) || '{}');
      if (b.token !== TOKEN) return json(401, { error: { type: 'unauthorized', message: 'wrong token' } });
      const sid = `s${Date.now().toString(36)}${++this.seq}`;
      this.sessions.add(sid);
      res.writeHead(204, { 'Set-Cookie': `qse_dash=${sid}; HttpOnly; SameSite=Strict; Path=/; Max-Age=43200` });
      res.end();
      return;
    }
    if (method === 'DELETE') {
      this.sessions.clear();
      res.writeHead(204, { 'Set-Cookie': 'qse_dash=; HttpOnly; SameSite=Strict; Path=/; Max-Age=0' });
      res.end();
      return;
    }
    if (!this.signedIn(req)) return json(401, { error: { type: 'unauthorized', message: 'no session' } });
    return json(200, { contract_version: '1.1', authenticated: true, login: true, password_set: true, expires_at: new Date(Date.now() + 43_200_000).toISOString().replace(/\.\d{3}Z$/, 'Z') });
  }

  // ---- the router ---------------------------------------------------------------------------

  /** true when it answered */
  async handle(req: IncomingMessage, res: ServerResponse, path: string, q: URLSearchParams, readBody: () => Promise<string>): Promise<boolean> {
    const json = (code: number, body: unknown) => {
      res.writeHead(code, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' });
      res.end(JSON.stringify(body));
      return true;
    };
    // server/setup_api.py _err: the extras go inside the error
    const err = (code: number, type: string, message: string, extra: Record<string, unknown> = {}) => json(code, { error: { type, code: type, message, ...extra } });
    if (path === '/__mock/setup' && req.method === 'POST') return json(200, this.control(JSON.parse((await readBody()) || '{}')));
    if (!path.startsWith('/v1/setup/')) return false;
    if (this.away()) {
      req.socket.destroy();
      return true;
    }
    if (this.o.unauthorized && !this.signedIn(req)) return err(401, 'unauthorized', 'a bearer token or the dashboard session is needed');
    const method = req.method ?? 'GET';
    const body = method === 'POST' ? JSON.parse((await readBody()) || '{}') : null;
    const o = this.o;

    if (path === '/v1/setup/models') {
      if (o.models === 'error') return err(500, 'internal_error', 'models/: permission denied');
      if (o.models === 'empty') return json(200, { contract_version: '1.0', models: [], errors: [] });
      const rec = MODEL.profiles.find((p) => p.recommended)!;
      const row = MODEL.measurements.rows.find((r) => r.profile === rec.id)!;
      const best = Object.entries(row.cells.greedy_512 ?? {}).sort((a, b) => b[1]!.tok_s - a[1]!.tok_s)[0];
      return json(200, {
        contract_version: '1.0',
        models: [{ id: MODEL.id, name: MODEL.name, family: MODEL.family, params: MODEL.params, pitch: MODEL.pitch, recommended: true, icon: null, best_speed: best ? { tok_s: best[1]!.tok_s, task: best[0], profile: rec.id } : null }],
        errors: o.models === 'broken' ? [{ path: 'models/gemma-4-31b/model.yaml', message: 'models/gemma-4-31b/model.yaml: $.profiles: exactly one recommended profile, not 0' }] : [],
      });
    }
    const mm = path.match(/^\/v1\/setup\/models\/([^/]+)$/);
    if (mm) return decodeURIComponent(mm[1]) === MODEL.id ? json(200, MODEL) : err(404, 'not_found', `no model ${mm[1]}`);
    const mc = path.match(/^\/v1\/setup\/models\/([^/]+)\/current$/);
    if (mc) return decodeURIComponent(mc[1]) === MODEL.id ? json(200, { contract_version: '1.0', model: MODEL.id, selection: this.box, unmatched: [] }) : err(404, 'not_found', `no model ${mc[1]}`);
    if (path === '/v1/setup/device') return json(200, this.device());
    const refuse = (e: unknown) => (e instanceof SelectionError ? err(e.status, e.code, e.message) : err(500, 'internal_error', String(e)));
    if (path === '/v1/setup/estimate' && method === 'POST') {
      if (o.estimateDown) return err(500, 'internal_error', 'estimate failed');
      try {
        return json(200, estimate(MODEL, (body.selection ?? body) as SetupSelection, this.memAvailable()));
      } catch (e) {
        return refuse(e);
      }
    }
    if (path === '/v1/setup/plan' && method === 'POST') {
      if (o.planDown) return err(500, 'internal_error', 'plan failed');
      try {
        return json(200, this.plan(checkSelection(MODEL, body.selection ?? body)));
      } catch (e) {
        return refuse(e);
      }
    }
    if (path === '/v1/setup/apply' && method === 'POST') {
      const extra = Object.keys(body).filter((k) => k !== 'selection' && k !== 'dry_run');
      if (extra.length) return err(422, 'unknown_field', `unknown field '${extra[0]}'; apply takes selection and dry_run`);
      let sel: SetupSelection;
      try {
        sel = checkSelection(MODEL, body.selection);
      } catch (e) {
        return refuse(e);
      }
      const p = this.plan(sel);
      if (body.dry_run) return json(200, { ...p, dry_run: true });
      if (o.busy && !o.setupMode) return err(409, 'busy', `${o.busy} request(s) run and 0 wait; apply when the engine is idle`, { running: o.busy, waiting: 0 });
      if (!p.ok) return json(PLAN_STATUS[p.error_codes[0]] ?? 409, { error: { type: p.error_codes[0], code: p.error_codes[0], message: p.errors.join(' ') }, plan: p });
      const running = this.runningJob();
      if (running) return err(409, 'job_running', `job ${running} runs; one setup job at a time`, { job: running });
      this.o.assistant = 'not_loaded';
      return json(202, { contract_version: '1.0', job: this.startJob(sel) });
    }
    const jm = path.match(/^\/v1\/setup\/jobs\/([^/]+)$/);
    if (jm) {
      const id = decodeURIComponent(jm[1]);
      if (q.get('format') === 'json') {
        const j = this.jobNow(id);
        return j ? json(200, j) : err(404, 'not_found', `no job ${id}`);
      }
      this.streamJob(req, res, id);
      return true;
    }
    if (path === '/v1/setup/reset') {
      if (method === 'GET') return json(200, { contract_version: '1.0', levels: this.levels(), busy: o.busy > 0 });
      // server/setup_api.py reset_route, in its order
      const extra = Object.keys(body).filter((k) => !['level', 'confirm', 'dry_run'].includes(k));
      if (extra.length) return err(422, 'unknown_field', `unknown field '${extra[0]}'; reset takes level, confirm and dry_run`);
      if (body.level === 1) return err(400, 'browser_only', "level 1 (the setup progress and Wren's chat) is the browser's own: the page clears it");
      const level = body.level as 2 | 3 | 4 | 5;
      const lv = this.levels().find((l) => l.level === level);
      if (!lv || typeof level !== 'number') return err(422, 'invalid_level', 'level is 2, 3, 4 or 5');
      if (body.dry_run) return json(200, { ...lv, contract_version: '1.0', dry_run: true });
      const running = this.runningJob();
      if (running) return err(409, 'job_running', `job ${running} runs; one setup job at a time`);
      if (o.busy) return err(409, 'busy', `${o.busy} request(s) run and 0 wait; reset when the engine is idle`, { running: o.busy, waiting: 0 });
      if (lv.confirm_word ? typeof body.confirm !== 'string' || body.confirm.trim().toLowerCase() !== lv.confirm_word : body.confirm !== true)
        return json(400, { error: { type: 'confirm_required', code: 'confirm_required', message: lv.confirm_word ? `type '${lv.confirm_word}' to confirm` : 'confirm: true' }, level: { ...lv, contract_version: '1.0' } });
      if (level === 2) {
        const rec = MODEL.profiles.find((p) => p.recommended)!.id;
        const job = this.startJob({ ...defaults(MODEL), profile: rec }, 'ok', 'reset');
        return json(202, { contract_version: '1.0', level, removed: [], failed: [], freed_gb: 0, restart_needed: true, message: 'The engine restarts with the shipped configuration.', ok: true, job_id: job.id, job });
      }
      // reset.execute: refused and `skip` items stay (in `failed`), `engine` ones the engine clears
      const removed: { path: string; bytes: number }[] = [];
      const failed: { path: string; reason: string }[] = [];
      let busyLeft = o.resetPartial ? 2 : 0;
      for (const it of lv.items) {
        if (!it.exists) continue;
        if (it.refused) failed.push({ path: it.path, reason: it.refused });
        else if (it.action === 'skip') failed.push({ path: it.path, reason: 'the shadow reader runs; reset again in setup mode or after a restart' });
        else if (busyLeft > 0) {
          busyLeft--;
          failed.push({ path: it.path, reason: 'OSError: [Errno 16] Device or resource busy' });
        } else removed.push({ path: it.path, bytes: it.bytes });
      }
      if (level === 4) for (const p of MODEL.profiles) if (p.id !== o.active && removed.some((r) => r.path.endsWith(`/${p.id}`))) this.onDisk[p.id] = false;
      const freed = removed.reduce((a, r) => a + r.bytes, 0);
      const message = `Removed ${removed.length} of ${lv.items.length} items, ${(freed / GIB).toFixed(2)} GiB.` + (failed.length ? ` ${failed.length} could not be removed: see \`failed\`.` : '') + (lv.restart ? ' Restart the engine to finish.' : '');
      return json(200, { contract_version: '1.0', level, removed, failed, freed_gb: Math.round((freed / GIB) * 100) / 100, restart_needed: lv.restart, message, ok: !failed.length });
    }
    if (path === '/v1/setup/assistant/status') return json(200, this.status());
    if (path === '/v1/setup/assistant/load' && method === 'POST') {
      if (o.noModel) return json(200, this.status());
      if (o.refuse) return json(200, { ...this.status(), state: 'refused', can_load: true, reason: 'not enough memory beside the engine: 12.9 GiB would be left, 15 GiB must stay free' });
      if (this.o.assistant !== 'loaded') {
        this.o.assistant = 'loading';
        this.loadStartedAt = Date.now();
      }
      return json(200, this.status());
    }
    if (path === '/v1/setup/assistant/unload' && method === 'POST') {
      this.o.assistant = 'not_loaded';
      return json(200, this.status());
    }
    if (path === '/v1/setup/assistant/chat' && method === 'POST') {
      this.chat(req, res, body as SetupChatRequest);
      return true;
    }
    return err(404, 'not_found', `no route ${path}`);
  }
}
