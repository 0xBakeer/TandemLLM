// The mock engine's conductor (/v1/conductor/*, contract docs/contract/conductor-v1): the server's
// own built-in documents (seeds.json, written by tools/conductor_contract.py), versions and writes
// in memory with the server's validation (the same config schemas through ajv), a day of seeded
// decisions over a handful of sessions -- the router's rows and the session monitor's events in
// one journal (SRV-115) -- and the aggregates the server computes. Dev and preview only; never in
// dist/.
//
// Test switches, POST /__mock/conductor: {requireToken: true} makes writes take only the admin
// bearer token (as a server with the dashboard login off), {guard: true} adds a speed-guard report that passed, {guard: 'single'} one of a short gate
// (one run a side, the noise unknown, as 0.3.4-test's)
// for the serving build, {reset: true} starts over, {roleDown: "decider"} takes a role offline.

import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import Ajv2020 from 'ajv/dist/2020.js';
import { rng } from './generate.ts';
import { diskRoute, diskSeed, longSession, memoryRoute, memorySeed, type MockDisk } from './memory.ts';

const DIR = fileURLToPath(new URL('../../docs/contract/conductor-v1/', import.meta.url));
const SEEDS = JSON.parse(readFileSync(join(DIR, 'seeds.json'), 'utf8'));
const LOCKED: Record<string, string> = SEEDS.meta.locked;
// the tokenizer hash the mock engine "records" (ENG-129); a role that shares token ids must have it
export const ENGINE_SHA = '5f0c'.repeat(16);
// the classifier report measured on the Spark (tools/conductor_eval.py), per-prompt rows left out
const REPORT = JSON.parse(readFileSync(fileURLToPath(new URL('../../tests/fixtures/conductor/classifier-report-2026-10-01.json', import.meta.url)), 'utf8'));
// a night of the eval harness (tools/tandem_eval.py, OPS-43): the contract's example, made-up numbers
const SUITE_REPORT = JSON.parse(readFileSync(fileURLToPath(new URL('../../docs/contract/dashboard-v1/tandem-eval.example.json', import.meta.url)), 'utf8'));
export const ENGINE = { version: '0.3.1-test', code_sha: '4c1f0e9a7b2d' };
// the three Laya checkpoints as the engine serves them (engine/decision/laya), with their limits
const LAYA = [
  { name: 'laya-english', aliases: ['en', 'english'], limits: { context_tokens: 512, question_tokens: 192, max_options: 20 }, bytes: 1_698_000_000, hash: '6f1c2a9e0b7d4c3f8e21' },
  { name: 'laya-multilingual', aliases: ['ml', 'multi', 'multilingual'], limits: { context_tokens: 1024, question_tokens: 256, max_options: 20 }, bytes: 1_297_000_000, hash: 'a83e5d10c6f2b9e74d03' },
  { name: 'laya-typed-decisions', aliases: ['decisions', 'typed', 'typed-decisions', 'typed_decisions'], limits: { context_tokens: 1024, question_tokens: 256, max_options: 20 }, bytes: 1_698_000_000, hash: '0d47b2c9e81a6f35c2e9' },
];

type Doc = Record<string, unknown>;
interface Version {
  version: number;
  note: string;
  author: string;
  created_at: string | null;
  restored_from: number | null;
  doc: Doc;
}
export interface Reply {
  status: number;
  body?: unknown;
  raw?: { type: string; data: string };
}

const iso = (ms: number) => new Date(ms).toISOString().replace(/\.\d{3}Z$/, 'Z');
const clone = <T>(x: T): T => JSON.parse(JSON.stringify(x));

// ---------------------------------------------------------------- validation (the server's rules)
const ajv = new Ajv2020({ allErrors: true, strict: false });
const validators: Record<string, ReturnType<typeof ajv.compile>> = {};
for (const k of ['settings', 'router', 'roles', 'packages', 'monitor', 'profiles', 'residency', 'deciders']) {
  validators[k] = ajv.compile(JSON.parse(readFileSync(join(DIR, `${k}.config.schema.json`), 'utf8')));
}

function diff(a: unknown, b: unknown, path = '$'): { path: string; before: unknown; after: unknown }[] {
  const out: { path: string; before: unknown; after: unknown }[] = [];
  const isObj = (x: unknown) => x != null && typeof x === 'object' && !Array.isArray(x);
  if (isObj(a) && isObj(b)) {
    const ka = a as Record<string, unknown>;
    const kb = b as Record<string, unknown>;
    for (const k of [...new Set([...Object.keys(ka), ...Object.keys(kb)])].sort()) {
      if (!(k in ka)) out.push({ path: `${path}.${k}`, before: null, after: kb[k] });
      else if (!(k in kb)) out.push({ path: `${path}.${k}`, before: ka[k], after: null });
      else out.push(...diff(ka[k], kb[k], `${path}.${k}`));
    }
    return out;
  }
  if (Array.isArray(a) && Array.isArray(b) && [...a, ...b].every((x) => isObj(x) && 'id' in (x as object))) {
    const ia = new Map(a.map((x) => [(x as { id: string }).id, x]));
    const ib = new Map(b.map((x) => [(x as { id: string }).id, x]));
    for (const k of [...ia.keys(), ...[...ib.keys()].filter((x) => !ia.has(x))]) {
      if (!ia.has(k)) out.push({ path: `${path}[${k}]`, before: null, after: ib.get(k) });
      else if (!ib.has(k)) out.push({ path: `${path}[${k}]`, before: ia.get(k), after: null });
      else out.push(...diff(ia.get(k), ib.get(k), `${path}[${k}]`));
    }
    return out;
  }
  if (JSON.stringify(a) !== JSON.stringify(b)) out.push({ path, before: a, after: b });
  return out;
}

// ---------------------------------------------------------------- the seeded day
const CLIENT_PROFILE: Record<string, string> = { opencode: 'coding', 'open-webui': 'chat' };
const PROMPTS: Record<string, string> = {
  code_edit: 'Refactor the lexer so every token carries its line number.',
  debug: 'The tests fail with a KeyError in parse_args; find it and fix it.',
  design_ui: 'Make the settings page calmer: more spacing, one accent colour.',
  drawing: 'Draw this portrait as one spiral line, animated from the centre.',
  three_d: 'Rebuild the fjord photo as a Three.js scene from the hero camera.',
  plan: 'Plan the migration of the billing service to the new queue.',
  research: 'How does the resident prefix cache work in this engine?',
  summarize: 'Summarize this changelog in five bullet points.',
  chat: 'thanks, that works!',
};

export class MockConductor {
  docs = new Map<string, Version[]>();
  rows: Record<string, unknown>[] = [];
  events: Record<string, unknown>[] = [];
  memory = new Map<string, ReturnType<typeof longSession>>();
  disk: MockDisk = diskSeed(Date.now());
  requireToken = false;
  guard: boolean | 'single' = false;
  roleDown: string | null = null;
  seed: number;

  constructor(seed = 99) {
    this.seed = seed;
    this.reset();
  }

  reset() {
    this.docs.clear();
    for (const k of ['settings', 'router', 'roles', 'packages', 'monitor', 'residency', 'deciders']) this.docs.set(k, [this.v0(SEEDS[k])]);
    for (const [n, d] of Object.entries(SEEDS.profiles)) this.docs.set(`profiles/${n}`, [this.v0(d as Doc)]);
    // the chat profile runs effort in shadow in the seeded day (the promotion cases)
    this.put('profiles', 'chat', { ...clone(this.cur('profiles/chat').doc), decisions: { classify: 'on', effort: 'shadow', packages: 'on', rules: 'off' } }, 'effort to shadow', 'seed');
    this.put('profiles', 'coding', { ...clone(this.cur('profiles/coding').doc), decisions: { classify: 'on', effort: 'on', packages: 'shadow', rules: 'shadow' } }, 'packages and rules to shadow', 'seed');
    this.rows = this.day();
    this.events = this.monitorDay();
    // VIS-48: the session transcripts, their hidden notes pointing at rows of this journal
    const noted = this.rows.find((r) => (r.injected as unknown[]).length);
    const acted = this.events.find((e) => e.mode === 'act' && e.decision !== 'PASS');
    this.memory = memorySeed(Date.now(), (noted?.id as string) ?? null, (acted?.id as string) ?? null);
    this.disk = diskSeed(Date.now());
    this.requireToken = false;
    this.guard = false;
    this.roleDown = null;
    this.resetDeciders();
  }

  private v0(doc: Doc): Version {
    return { version: 0, note: 'built-in defaults', author: 'server', created_at: null, restored_from: null, doc: clone(doc) };
  }
  private key(kind: string, name: string | null) {
    return kind === 'profiles' ? `profiles/${name}` : kind;
  }
  cur(key: string): Version {
    return this.docs.get(key)!.at(-1)!;
  }

  check(kind: string, name: string | null, doc: Doc): string[] {
    const v = validators[kind];
    if (!v(doc)) return (v.errors ?? []).map((e) => `$${e.instancePath.replace(/\//g, '.')}: ${e.message}${e.params && 'additionalProperty' in e.params ? ` (${(e.params as { additionalProperty: string }).additionalProperty})` : ''}`);
    const errs: string[] = [];
    if (kind === 'profiles' && name && LOCKED[name]) errs.push(`$: profile '${name}' is locked (${LOCKED[name]})`);
    if (kind === 'router') {
      const ids = (doc.classes as { id: string }[]).map((c) => c.id);
      if (new Set(ids).size !== ids.length) errs.push('$.classes: two classes share an id');
      if (!String((doc.injection as { wrapper: string }).wrapper).includes('{text}')) errs.push('$.injection.wrapper: must contain {text}');
    }
    if (kind === 'profiles') {
      const ids = (this.cur('router').doc.classes as { id: string }[]).map((c) => c.id);
      if (!ids.includes(doc.default_class as string)) errs.push(`$.default_class: no class '${doc.default_class}'`);
      const tasks = doc.tasks as Record<string, { mode: string }>;
      if (this.roleDown === 'decider' && ['D3', 'E2'].some((t) => tasks[t]?.mode === 'act')) errs.push('$.tasks.E2.mode: act needs the decider role, which is offline');
    }
    if (kind === 'roles') {
      // SRV-133: the contract's rules (server/conductor/schema.py role_rules)
      for (const [n, r] of Object.entries(doc as Record<string, { kind: string; window: string; writes: string; sharing: string[]; tokenizer_sha256: string }>)) {
        if (r.writes === 'client_visible' && r.kind !== 'reasoner') errs.push(`$.${n}.writes: only the reasoner writes tokens the client sees (the honesty rule); a ${r.kind} may write hidden_notes, memory_files or nothing`);
        if (r.window === 'in_decode' && r.kind !== 'decider' && r.kind !== 'probe') errs.push(`$.${n}.window: in_decode is for a decider or a probe only; during decode the GPU belongs to Qwen (a ${r.kind} runs idle or any)`);
        if (r.sharing.includes('states') && r.kind !== 'probe') errs.push(`$.${n}.sharing: states (Qwen's own hidden rows) is for a probe inside the engine only`);
        if (r.sharing.includes('tokens') && r.tokenizer_sha256 !== ENGINE_SHA) errs.push(`$.${n}.tokenizer_sha256: ${r.tokenizer_sha256.slice(0, 12)}… is not the engine's tokenizer (${ENGINE_SHA.slice(0, 12)}…); a different tokenizer shares text only`);
      }
    }
    if (kind === 'residency' && new Set(doc.order as string[]).size !== (doc.order as string[]).length) errs.push('$.order: a role is listed twice');
    if (kind === 'residency') {
      const e = (doc.engine ?? {}) as Record<string, number>;
      if (e.MEM_CEILING_GIB != null && e.MEM_PEAK_GIB != null && e.MEM_PEAK_GIB < e.MEM_CEILING_GIB) errs.push("$.engine: MEM_PEAK_GIB is under MEM_CEILING_GIB (the peak is the ceiling plus a prefill's workspace)");
    }
    if (kind === 'deciders') {
      const names = (doc.models as { name: string }[]).map((m) => m.name);
      if (new Set(names).size !== names.length) errs.push('$.models: a model is listed twice');
    }
    return errs;
  }

  put(kind: string, name: string | null, doc: Doc, note: string, author: string, base?: number | null, restored?: number): Reply {
    const k = this.key(kind, name);
    if (kind === 'profiles' && name && LOCKED[name]) return { status: 409, body: { error: { type: 'conflict', message: `profile '${name}' is locked: ${LOCKED[name]}` } } };
    if (!note.trim()) return { status: 422, body: { error: { type: 'invalid_config', message: 'a change note is required', errors: ['$note: missing'] } } };
    const errs = this.check(kind, name, doc);
    if (errs.length) return { status: 422, body: { error: { type: 'invalid_config', message: errs[0], errors: errs } } };
    const list = this.docs.get(k) ?? [];
    const curV = list.at(-1)?.version ?? 0;
    if (base != null && base !== curV) return { status: 409, body: { error: { type: 'conflict', message: `changed to version ${curV} after you opened version ${base}` } } };
    const v = Math.max(0, ...list.map((x) => x.version)) + 1;
    list.push({ version: v, note: note.trim(), author, created_at: iso(Date.now()), restored_from: restored ?? null, doc: clone(doc) });
    this.docs.set(k, list);
    return { status: 200, body: { contract_version: '1.0', kind, name, version: v, loaded_at: iso(Date.now()), ...(restored != null ? { restored_from: restored } : {}) } };
  }

  // ---------------------------------------------------------------- decisions
  private day(): Record<string, unknown>[] {
    const r = rng(this.seed);
    const now = Date.now();
    const router = this.cur('router').doc as { classes: { id: string; policy: { effort: string; packages: string[]; rules: string[] } }[]; effort: { levels: Record<string, { note: string }> }; injection: { wrapper: string } };
    const classes = router.classes.map((c) => c.id);
    const out: Record<string, unknown>[] = [];
    const sessions = [
      { key: 'c:ses_night_pagoda', client: 'opencode', cls: 'three_d' },
      { key: 'c:ses_mona_spiral', client: 'opencode', cls: 'drawing' },
      { key: 'c:ses_lexer_refactor', client: 'opencode', cls: 'code_edit' },
      { key: 'c:ses_settings_page', client: 'opencode', cls: 'design_ui' },
      { key: 'c:ses_parse_bug', client: 'opencode', cls: 'debug' },
      { key: 'h:5f0c1a9e33d2b7a18c4e2f60', client: 'open-webui', cls: 'chat' },
      { key: 'h:a1b2c3d4e5f60718293a4b5c', client: 'open-webui', cls: 'research' },
      { key: 'h:77e1d0c9b8a7f6e5d4c3b2a1', client: 'open-webui', cls: 'summarize' },
    ];
    let t = now - 23 * 3600_000;
    let labelledCoding = 0;
    let labelledChat = 0;
    for (let round = 0; round < 3; round++) {
      for (const s of sessions) {
        const profile = CLIENT_PROFILE[s.client] ?? 'chat';
        const decisions = profile === 'coding' ? { classify: 'on', effort: 'on', packages: 'shadow', rules: 'shadow' } : { classify: 'on', effort: 'shadow', packages: 'on', rules: 'off' };
        const turns = s.client === 'opencode' ? 6 + Math.floor(r() * 6) : 2 + Math.floor(r() * 2);
        for (let i = 0; i < turns; i++) {
          t += 60_000 + Math.floor(r() * 400_000);
          if (t > now - 30_000) t = now - 30_000 - Math.floor(r() * 20_000);
          const first = i === 0;
          const conf = first ? 0.62 + r() * 0.35 : null;
          const src = first ? (r() < 0.85 ? 'decider' : 'qwen') : s.client === 'opencode' ? 'sticky' : r() < 0.7 ? 'decider' : 'qwen';
          const cls = s.cls;
          const pol = router.classes.find((c) => c.id === cls)!.policy;
          const stages = src === 'decider' || src === 'qwen'
            ? [
                src === 'qwen'
                  ? { stage: 'decider', label: null, confidence: 0, probs: {}, ms: 400.5, error: 'decider_timeout', model: null, accept: 0.6 }
                  : { stage: 'decider', label: cls, confidence: conf ?? 0.8, probs: Object.fromEntries(classes.map((c) => [c, c === cls ? (conf ?? 0.8) : (1 - (conf ?? 0.8)) / (classes.length - 1)])), ms: 18 + r() * 30, error: null, model: 'laya-english', accept: 0.6 },
                ...(src === 'qwen' ? [{ stage: 'qwen', label: cls, confidence: conf ?? 0.7, probs: { [cls]: conf ?? 0.7 }, ms: 420 + r() * 500, error: null, model: 'qwen38-spark-engine', accept: 0.55 }] : []),
              ]
            : [];
          const effortOn = decisions.effort === 'on';
          const note = router.effort.levels[pol.effort]?.note ?? '';
          const parts: { kind: string; name: string; text: string }[] = [];
          if (note) parts.push({ kind: 'effort', name: pol.effort, text: note });
          if (pol.packages.length && decisions.packages === 'on') parts.push({ kind: 'package', name: pol.packages[0], text: `# ${pol.packages[0]}\n\n(the package's text on the box)` });
          const injectNow = first && effortOn && parts.length > 0 && (profile === 'coding' ? parts.length > 0 : decisions.packages === 'on' && parts.some((p) => p.kind === 'package'));
          const injected = injectNow
            ? [{ pos: 18_000 + Math.floor(r() * 90_000), at: 18_000, tokens: 40 + Math.floor(r() * 900), text: router.injection.wrapper.replace('{text}', parts.map((p) => p.text).join('\n\n')), parts: parts.map(({ kind, name }) => ({ kind, name })), kinds: [...new Set(parts.map((p) => (p.kind === 'package' ? 'packages' : p.kind)))], reason: `class ${cls} (${src}${conf != null ? `, ${conf.toFixed(2)}` : ''})` }]
            : [];
          const shadow = decisions.packages === 'shadow' || decisions.effort === 'shadow' ? { effort: { effort: pol.effort, budget: null, thinking: pol.effort === 'off' ? 'off' : 'on' }, notes: pol.packages.length ? [{ decision: 'packages', kind: 'package', name: pol.packages[0], chars: 3200 }] : [] } : null;
          let label: string | null = null;
          if (profile === 'coding' && labelledCoding < 35 && r() < 0.8) {
            label = labelledCoding < 33 ? 'right' : 'wrong';
            labelledCoding++;
          } else if (profile === 'chat' && labelledChat < 12) {
            label = labelledChat < 11 ? 'right' : 'wrong';
            labelledChat++;
          }
          const finish = s.client === 'opencode' ? (r() < 0.85 ? 'tool_calls' : 'stop') : 'stop';
          out.push({
            id: (Math.floor(r() * 2 ** 32).toString(16) + Math.floor(r() * 2 ** 32).toString(16)).padStart(16, '0').slice(0, 16),
            ts_ms: t, request_id: `chatcmpl-${Math.floor(r() * 2 ** 40).toString(16)}`, session_key: s.key, client_kind: s.client, model: 'TandemLLM',
            profile, profile_version: 1, router_version: 0, mode: 'on', class_id: cls, confidence: first ? conf : 0.9, source: src, kept: null,
            stages, classifier_ms: stages.reduce((a, x) => a + (x.ms as number), 0), effort: effortOn ? pol.effort : null, budget: effortOn && pol.effort === 'low' ? 1024 : null,
            budget_source: effortOn && pol.effort === 'low' ? 'class' : 'server', thinking: effortOn && pol.effort === 'off' ? 'off' : 'on', decisions,
            injected, replayed: !first && s.client === 'opencode' ? { count: 1, tokens: 600, positions: [18_000] } : null, shadow,
            digest: s.client === 'opencode' ? PROMPTS[cls] : null,
            outcome: { status: 200, finish_reason: finish, tool_calls: finish === 'tool_calls' ? 1 : 0, tool_names: finish === 'tool_calls' ? ['bash'] : [], completion_tokens: 200 + Math.floor(r() * 1800), reasoning_tokens: 100 + Math.floor(r() * 1500), prompt_tokens: 30_000, ttft_ms: 400 + r() * 2000, total_ms: 4000 + r() * 60_000 },
            next_checker: s.key === 'c:ses_night_pagoda' && i > 1 ? [{ checker: 'pagodacheck', target: '~/demos/pagoda-seasons', verdict: i === turns - 1 ? 'PASS' : 'FAIL', reasons: [`(a) summer island colour differs by ${(15 - i * 2).toFixed(1)} > 1.0`], readings: [{ gate: 'a', name: 'summer island colour differs by', value: Math.max(0.6, 15 - i * 2), op: '>', limit: 1 }] }] : null,
            label, label_note: null, labelled_at: label ? t + 60_000 : null, acted: injected.length || (effortOn && pol.effort !== 'medium') ? 1 : 0, changed: injected.length || !first ? 1 : 0, extra: {},
          });
        }
      }
    }
    out.sort((a, b) => (a.ts_ms as number) - (b.ts_ms as number));
    // spread the day over the last 23.5 hours, the last turn half a minute ago
    const lo = out[0].ts_ms as number;
    const hi = out[out.length - 1].ts_ms as number;
    const span = 23.5 * 3600_000 - 30_000;
    for (const r of out) {
      const ts = now - 23.5 * 3600_000 + Math.round((((r.ts_ms as number) - lo) / Math.max(1, hi - lo)) * span);
      if (r.labelled_at) r.labelled_at = ts + 60_000;
      r.ts_ms = ts;
    }
    return out;
  }

  /** The session monitor's day on the router's rows: acts (hidden continues, closes,
   * regenerations, one give-up), shadow findings of the tool checks, the passes of held answers,
   * and the shadow decisions with labels the promotion cases need (D1 41 labelled at 0.93, E2 12). */
  private monitorDay(): Record<string, unknown>[] {
    const r = rng(this.seed + 7);
    const out: Record<string, unknown>[] = [];
    const texts: Record<string, string> = {
      E1: 'You ended your turn without a tool call or an answer. Continue the task now with the next tool call.',
      D1: '\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n',
      C4: 'Your tool call was not run: the <function=NAME> opener is malformed. Write the tool call again, complete, in the same format.',
      C3: 'You ran `bash` with these arguments 2 times and got the same result each time (first line: RESULT: FAIL <- (a) summer island colour differs by 3.5 > 1.0). This call was not run. Do something different.',
    };
    let id = 0;
    const ev = (row: Record<string, unknown>, e: Record<string, unknown>) => {
      const text = (e.text as string | undefined) ?? null;
      out.push({
        id: `ev${(id++).toString(16).padStart(6, '0')}`, ts_ms: (row.ts_ms as number) + 1000 + Math.floor(r() * 20_000), request_id: row.request_id, session_key: row.session_key, kind: 'monitor',
        task: null, tier: 0, mode: 'act', sp: 'SP2', rule: null, trigger: null, score: null, decision: 'PASS', action: null, cap: null,
        hidden_text_id: text ? `t${text.length}` : null, hidden_tokens: 0, tokens_at_cut: null, position: null,
        outcome: { kind: (row.outcome as { tool_calls: number }).tool_calls ? 'tool_call' : 'text', acts: 1, hidden_tokens: 0, finish: (row.outcome as { finish_reason: string }).finish_reason },
        next_checker: row.next_checker ?? null, policy_version: 2, client_kind: row.client_kind, profile: row.profile, detail: {}, label: null, label_note: null, labelled_at: null, text, ...e,
      });
    };
    let d1 = 0;
    let e2 = 0;
    let gaveUp = false;
    // one of each act in a known session, so every screen has something to show
    const sure: Record<string, string> = { 'c:ses_night_pagoda': 'E1', 'c:ses_mona_spiral': 'C4', 'c:ses_lexer_refactor': 'C3', 'c:ses_settings_page': 'C1', 'c:ses_parse_bug': 'C2' };
    const first = new Set<string>();
    for (const row of this.rows) {
      if (row.client_kind !== 'opencode') continue;
      const k = row.session_key as string;
      const forced = sure[k] && !first.has(k) ? sure[k] : null;
      first.add(k);
      const x = forced ? { E1: 0.01, D1: 0.1, C4: 0.15, C3: 0.18, C1: 0.22, C2: 0.28 }[forced]! : r();
      if (x < 0.08) ev(row, { task: 'E1', rule: 'empty_turn', sp: 'SP2', trigger: 'empty_turn(case=B)', decision: 'CONTINUE_HIDDEN', action: 'CONTINUE_HIDDEN', text: texts.E1, hidden_tokens: 31, position: 41_200 + Math.floor(r() * 40_000), detail: { case: 'B', act: 1 } });
      else if (x < 0.13) ev(row, { task: 'D1', rule: 'reasoning_loop', sp: 'SP1', trigger: 'repeat16(share=0.74)', score: 0.74, decision: 'CLOSE_REASONING', action: 'CLOSE_REASONING', text: texts.D1, hidden_tokens: 22, tokens_at_cut: 2_880, position: 52_000, detail: { at: 2880 } });
      else if (x < 0.17) ev(row, { task: 'C4', rule: 'malformed_call', sp: 'SP3', trigger: 'unreadable(the <function=NAME> opener is malformed)', decision: 'REGENERATE_CALL', action: 'REGENERATE_CALL', text: texts.C4, hidden_tokens: 48, position: 61_300, detail: { findings: [{ task: 'C4', rule: 'malformed_call', decision: 'REGENERATE_CALL' }], calls: 1 } });
      else if (x < 0.2) ev(row, { task: 'C3', rule: 'repeated_call', sp: 'SP3', trigger: 'repeated(bash x3)', decision: 'REGENERATE_CALL', action: 'REGENERATE_CALL', text: texts.C3, hidden_tokens: 77, position: 70_100, detail: { name: 'bash', count: 2 } });
      else if (x < 0.26) ev(row, { task: 'C1', rule: 'risk_floor', sp: 'SP3', mode: 'shadow', trigger: 'floor(rm_rf_outside_root)', decision: 'REGENERATE_CALL', text: 'This call was not run: rm -rf outside the project root (rm ~/pago). Do it another way, inside the project.', detail: { floor: 'rm_rf_outside_root', match: 'rm ~/pago' } });
      else if (x < 0.31) ev(row, { task: 'C2', rule: 'project_rules', sp: 'SP3', mode: 'shadow', trigger: 'rule(numbered_copy)', decision: 'REGENERATE_CALL', text: 'This call was not run: it breaks a project rule (numbered_copy: js/creatures/bambooBridge2.js, a copy of bambooBridge.js).', detail: { rule: 'numbered_copy', path: 'js/creatures/bambooBridge2.js' } });
      else if (x < 0.7) ev(row, { task: 'SP3', sp: 'SP3', trigger: 'checked(1)', decision: 'PASS', outcome: null, detail: { calls: 1, blocks: 1 } });
      if (!gaveUp && x > 0.98) {
        gaveUp = true;
        ev(row, { task: 'E1', rule: 'empty_turn', trigger: 'empty_turn(tries=2)', decision: 'GIVE_UP', action: 'GIVE_UP', text: 'No tool call after two tries. Stopping this turn.', outcome: { kind: 'gave_up', acts: 2, hidden_tokens: 62, finish: 'stop' } });
      }
      if (d1 < 44) {
        const lab = d1 < 41 ? (d1 < 38 ? 'right' : 'wrong') : null;
        ev(row, { task: 'D1', rule: 'reasoning_loop', sp: 'SP1', mode: 'shadow', trigger: 'repeat16(share=0.71)', score: 0.71, decision: 'CLOSE_REASONING', text: texts.D1, tokens_at_cut: 2_400, label: lab, labelled_at: lab ? (row.ts_ms as number) + 90_000 : null });
        d1++;
      } else if (e2 < 12) {
        ev(row, { task: 'E2', rule: 'done_without_evidence', mode: 'shadow', trigger: 'claim(0.88)', score: 0.88, decision: 'CONTINUE_HIDDEN', text: 'Run the checker before you say it passes.', label: e2 < 11 ? 'right' : 'wrong', labelled_at: (row.ts_ms as number) + 90_000 });
        e2++;
      }
    }
    // the session tasks (SRV-121) on the pagoda night: the gate note, a re-read note after a
    // compaction, the no-progress note, and the client gone at the end
    const night = this.rows.filter((x) => x.session_key === 'c:ses_night_pagoda');
    if (night.length >= 4) {
      ev(night[1], { task: 'S1', rule: 'gate_stagnation', sp: 'SP4', trigger: 'gate_stagnation(a: 3.5, 12.3, 3.5)', score: 2.5, decision: 'PROMPT_TAIL_NOTE', action: 'PROMPT_TAIL_NOTE', text: 'Gate (a) read 3.5, 12.3, 3.5. Record this stage as blocked in PROGRESS.yaml with the reason and take `next`.', hidden_tokens: 29, position: 88_400, detail: { gate: 'pagodacheck:.:a', values: [3.5, 12.3, 3.5], stage: 'S1-season-state-palette', limit: 1, op: '>' } });
      ev(night[2], { task: 'S2', rule: 'context_health', sp: 'SP4', trigger: 'context_health(no re-read in 2 steps)', decision: 'PROMPT_TAIL_NOTE', action: 'PROMPT_TAIL_NOTE', text: 'You are after a summary. Read PROGRESS.yaml and AGENTS.md before the next change.', hidden_tokens: 21, position: 31_200, detail: { steps: 2, compaction: 1 } });
      ev(night[3], { task: 'S4', rule: 'stall_watchdog', sp: 'SP4', trigger: 'stall_watchdog(21.2 min)', decision: 'PROMPT_TAIL_NOTE', action: 'PROMPT_TAIL_NOTE', text: 'No progress for 20 minutes. Re-read PROGRESS.yaml and take the next stage.', hidden_tokens: 19, position: 64_100, detail: { minutes: 21.2, last_progress: null } });
      ev(night[night.length - 1], { task: 'S4', rule: 'stall_watchdog', sp: 'watch', trigger: 'client_gone(20.4 min)', decision: 'NOTIFY', action: 'NOTIFY', text: null, outcome: null, detail: { idle_minutes: 20.4, requests: night.length, notify: 'sent 200' } });
    }
    if (!gaveUp && out.length) {
      const row = this.rows.find((x) => x.client_kind === 'opencode')!;
      ev(row, { task: 'E1', rule: 'empty_turn', trigger: 'empty_turn(tries=2)', decision: 'GIVE_UP', action: 'GIVE_UP', text: 'No tool call after two tries. Stopping this turn.', outcome: { kind: 'gave_up', acts: 2, hidden_tokens: 62, finish: 'stop' } });
    }
    for (const e of out) if (e.mode === 'act' && e.decision === 'PASS') e.action = null;
    return out.sort((a, b) => (a.ts_ms as number) - (b.ts_ms as number));
  }

  /** OPS-41 / OPS-47: server/conductor/residency.py's plan over the current document, at the box's
   *  2026-10-01 17:50 reading (MemAvailable 37.9 GiB, the engine holding 70 of its declared 77 GiB,
   *  so 7 GiB of growth comes off first) and its engine split; one refusal in the journal. */
  residencyMemGib = 37.9;
  enginePeakGib = 77;
  engineNowGib = 70;
  private engineGrowth(): number {
    return Math.round(Math.max(0, this.enginePeakGib - this.engineNowGib) * 100) / 100;
  }
  residencyPlan() {
    const cur = this.cur('residency');
    const d = cur.doc as { floor_gib: number; margin_gib: number; settle_s: number; order: string[]; claims: Record<string, { gib: number; basis: string; note: string }> };
    const need = d.floor_gib + d.margin_gib;
    const order = [...d.order.filter((r) => r in d.claims), ...Object.keys(d.claims).filter((r) => !d.order.includes(r)).sort()];
    const grow = this.engineGrowth();
    const eng = { engine_peak_gib: this.enginePeakGib, engine_now_gib: this.engineNowGib, engine_growth_gib: grow };
    let left: number | null = Math.round((this.residencyMemGib - grow) * 100) / 100;
    const rows = order.map((role) => {
      const c = d.claims[role];
      const after = left == null ? null : Math.round((left - c.gib) * 100) / 100;
      const allowed = after != null && after >= need;
      const row = { role, allowed, reason: allowed ? null : 'refused_memory', mem_available_gib: left, claim_gib: c.gib, basis: c.basis, floor_gib: d.floor_gib, margin_gib: d.margin_gib, need_gib: need, after_gib: after, note: c.note, source: 'residency', ...eng };
      if (allowed) left = after;
      return row;
    });
    const G = 1024 ** 3;
    const parts = { weights: Math.round(16.35 * G), kv_pool: 16 * G, gdn_state: Math.round(0.3 * G), drafter_kv: 10 * G, vision: 920_000_000, resident_prefix: Math.round(9.4 * G), state_store: Math.round(1.6 * G) };
    const named = Object.values(parts).reduce((a, b) => a + b, 0);
    const reserve = Math.round(1.1 * G);
    const allocated = Math.round(this.engineNowGib * G) - reserve;
    const refusal = { kind: 'check', ts: Date.now() / 1000 - 1800, role: 'reader', allowed: false, reason: 'refused_memory', mem_available_gib: 18.0, claim_gib: 12.0, source: 'ops' };
    return {
      contract_version: '1.0', version: cur.version, doc: d,
      plan: { mem_available_gib: this.residencyMemGib, floor_gib: d.floor_gib, margin_gib: d.margin_gib, need_gib: need, spendable_gib: Math.round((this.residencyMemGib - grow - need) * 100) / 100, settle_s: d.settle_s, order, rows, ...eng },
      engine: { allocated, reserved: allocated + reserve, parts, other: allocated - named, reserve, mem_available: Math.round(this.residencyMemGib * G), max_len: 262144, drafter_caches: 2,
        load: [{ stage: 'weights', allocated: parts.weights, reserved: parts.weights, mem_available_drop: parts.weights + G, s: 61.2 }, { stage: 'engine', allocated: parts.kv_pool + parts.gdn_state, reserved: parts.kv_pool + parts.gdn_state, mem_available_drop: parts.kv_pool + parts.gdn_state, s: 0.4 }] },
      limits: this.memoryLimits(allocated, reserve),
      last_refusal: refusal, journal: [refusal],
    };
  }

  /** OPS-47: server/memlimits.py view(): serve.env's values, the residency document's overrides
   *  (live keys at once, restart keys pending until a restart the mock never does), what each holds. */
  memoryLimits(allocated: number, reserve: number) {
    const G = 1024 ** 3;
    const M = 1024 ** 2;
    const cur = this.cur('residency');
    const over = ((cur.doc as { engine?: Record<string, number> }).engine ?? {}) as Record<string, number>;
    const T: [string, string, boolean, number, string, number | null, number | null][] = [
      // key, unit, live, serve.env value, what it caps, used (bytes or count), cap in the unit's bytes (null: the value's)
      ['MEM_CEILING_GIB', 'GiB', true, 70, 'the engine between requests, allocated plus the allocator\'s reserve', allocated + reserve, null],
      ['MEM_PEAK_GIB', 'GiB', true, 77, 'the engine during a prefill; the residency planner subtracts it', Math.round(74.6 * G), null],
      ['RESERVE_SLACK_GIB', 'GiB', true, 2, 'the allocator\'s reserve kept between requests; over it, it goes back to the driver', reserve, null],
      ['RESIDENT_GB', 'GiB', true, 2.5, 'the resident prefix\'s anchors', Math.round(2.44 * G), null],
      ['RESIDENT_STASH_GB', 'GiB', true, 1.0, 'the rows a short request between two turns may overwrite, copied aside', Math.round(0.86 * G), null],
      ['RESIDENT_GUEST_MAX', 'tokens', true, 8192, 'the longest request that counts as such a guest', null, null],
      ['RESIDENT_PIN_GB', 'GiB', true, 4.0, 'the pinned head of a prompt a long unrelated request replaced', Math.round(3.85 * G), null],
      ['RESIDENT_PARK_GB', 'GiB', true, 4, 'the rows of one other session, swapped back on its next turn', Math.round(2.25 * G), null],
      ['CACHE_GB', 'GiB', true, 2, 'the state store', Math.round(1.6 * G), null],
      ['STORE_ENTRY_GB', 'GiB', true, 1, 'one state store entry', Math.round(0.71 * G), null],
      ['VISION_EMBED_MB', 'MiB', true, 512, 'encoded images', Math.round(412 * M), null],
      ['SUFFIX_STORE_MB', 'MiB', true, 192, 'the suffix store\'s token log (host memory and disk, not the GPU pool)', 124_800_000, null],
      ['PREFIX_CHUNK', 'tokens', false, 1024, 'the prefill\'s chunk and the caches\' grid', null, null],
      ['NARROW_ARM', 'on/off', false, 1, 'the block-8 drafter arm (SPD-75): about 8 GiB at a 262,144-token window', null, null],
      ['MAX_LEN', 'tokens', false, 262144, 'the context window', null, null],
    ];
    const rows = T.map(([key, unit, live, flag, what, used]) => {
      const value = key !== 'MAX_LEN' && key in over ? over[key] : flag;
      const scale = unit === 'GiB' ? G : unit === 'MiB' ? M : 1;
      const cap = used == null ? null : Math.round(Number(value) * scale);
      // a cache over a lowered cap is given back at once (ResidentPrefix.set_limits); the limits
      // on the engine as a whole are read, not held
      const held = used != null && cap != null && used > cap && !key.startsWith('MEM_') && key !== 'RESERVE_SLACK_GIB' ? 0 : used;
      return {
        key, unit, what, value, flag, source: key !== 'MAX_LEN' && key in over ? 'dashboard' : 'serve.env',
        applies: key === 'MAX_LEN' ? 'locked' : live ? 'next request' : 'restart',
        locked: key === 'MAX_LEN' ? "the context window is 262,144 tokens (Khaled's hard rule)" : null,
        running: live ? value : flag, pending_restart: !live && key !== 'MAX_LEN' && value !== flag,
        used: held, cap,
      };
    });
    const now = Date.now() / 1000;
    const val = (k: string) => Number(rows.find((r) => r.key === k)!.value);
    return {
      version: cur.version, rows,
      budget: {
        ceiling: Math.round(val('MEM_CEILING_GIB') * G), peak: Math.round(val('MEM_PEAK_GIB') * G), slack: Math.round(val('RESERVE_SLACK_GIB') * G),
        allocated, reserved: allocated + reserve, reserve, ws_floor: Math.round(1.15 * G), ws_per_token: 30078,
        events: [
          { kind: 'shed', ts: now - 3600, level: 'store', why: 'peak', freed: Math.round(1.9 * G), safe: false },
          { kind: 'empty', ts: now - 600, why: 'idle', freed: Math.round(3.4 * G), ms: 41.2 },
        ],
        gates: 412, refused: 0, admitted: 1870, admit_refused: 3, empties: 37, empty_ms: 1210.4, empty_freed: Math.round(88 * G), last_empty_ms: 41.2,
        shed: { store: 1, anchors: 0, stash: 0, park: 0, pin: 0 }, shed_bytes: { store: Math.round(1.9 * G), anchors: 0, stash: 0, park: 0, pin: 0 },
        prefill_peak: Math.round(74.6 * G), observed: { n: 173855, workspace: Math.round(4.87 * G), estimate: Math.round(4.87 * G) },
        last_gate: { n: 25187, workspace: Math.round(1.15 * G), allocated, reserved: allocated + reserve, shed: [], ts: now - 60 },
        last_prefill_peak: { n: 25187, peak: Math.round(70.9 * G) },
      },
      workspace_at_max_len: Math.round(7.34 * G),
      switch: { ts: now - 900, from: 'ses_a81c2f…', to: 'ses_f0c723…', tokens: 97583, reused: 16384, reread: 81199, via: 'resident', ms: 125200 },
      switches: 4,
      park: { owner: 'ses_a81c2f…', rows: 22528, base: 20480, bytes: Math.round(2.25 * G) },
      owner: 'ses_f0c723…',
    };
  }

  /** SRV-131 / VIS-46: the engine's own decision models (engine/decision) as
   *  /v1/conductor/deciders/* answers: English loaded and warm, the other two not loaded; a load
   *  goes loading, warming, loaded over two seconds, through the planner's byte-math on the
   *  residency document (MemAvailable `deciderMem`, else the plan's 31.0 GiB). */
  decidersOff = false;
  deciderMem: number | null = null;
  deciderState: Record<string, { state: string; until: number; questions: number; loaded_at: number | null; refusal: Record<string, unknown> | null; picks: Record<string, number> }> = {};
  resetDeciders() {
    this.decidersOff = false;
    this.deciderMem = null;
    const now = Date.now() / 1000;
    this.deciderState = {};
    for (const m of LAYA) this.deciderState[m.name] = { state: m.name === 'laya-english' ? 'loaded' : 'not_loaded', until: 0, questions: m.name === 'laya-english' ? 214 : 0, loaded_at: m.name === 'laya-english' ? now - 5400 : null, refusal: null, picks: {} };
    this.deciderState['laya-english'].picks = { 'laya-english': 37 };
  }
  private deciderPlan(): Record<string, unknown> {
    const d = this.cur('residency').doc as { floor_gib: number; margin_gib: number; claims: Record<string, { gib: number; basis: string }> };
    const c = d.claims.decider;
    const mem = this.deciderMem ?? this.residencyMemGib;
    const need = d.floor_gib + d.margin_gib;
    const grow = this.engineGrowth();
    const after = Math.round((mem - grow - c.gib) * 100) / 100;
    const allowed = after >= need;
    return { role: 'decider', allowed, reason: allowed ? null : 'refused_memory', mem_available_gib: mem, claim_gib: c.gib, basis: c.basis, floor_gib: d.floor_gib, margin_gib: d.margin_gib, need_gib: need, after_gib: after, engine_peak_gib: this.enginePeakGib, engine_now_gib: this.engineNowGib, engine_growth_gib: grow };
  }
  private tick() {
    const now = Date.now();
    for (const st of Object.values(this.deciderState)) {
      if (st.state === 'loading' && now >= st.until) {
        st.state = 'warming';
        st.until = now + 700;
      } else if (st.state === 'warming' && now >= st.until) {
        st.state = 'loaded';
        st.loaded_at = now / 1000;
      }
    }
  }
  decidersStatus(): Record<string, unknown> {
    if (this.decidersOff) return { contract_version: '1.0', enabled: false, why: 'this engine serves no decision model: QWEN38_DECIDER_MODELS is empty (set it in ops/serve.env, e.g. laya-english,laya-multilingual,laya-typed-decisions, and restart)', ready: false, device: null, window: 'idle', families: [], models: [], planner: null, last_refusal: null, questions: [] };
    this.tick();
    const doc = this.cur('deciders').doc as { models: { name: string; load: string }[]; questions: Record<string, string>; api?: Record<string, unknown> };
    const pol = Object.fromEntries(doc.models.map((m) => [m.name, m.load]));
    const models = LAYA.map((m) => {
      const st = this.deciderState[m.name];
      const loaded = st.state === 'loaded';
      return {
        name: m.name, family: 'laya', contract: 'systemone', role: { kind: 'decider', memory_claim_gib: 4.4, window: 'idle', preemptible: true, latency_budget_ms: 100, fallback: 'PASS', tokenizer_sha256: 'none', writes: 'nothing', sharing: ['text'] },
        aliases: m.aliases, owned_by: 'convaiinnovations', state: st.state, load: pol[m.name] ?? 'on_demand', limits: m.limits,
        precision: loaded ? 'fp32 parameters under autocast(bfloat16), eager' : null, device: 'cuda', device_bytes: loaded ? m.bytes : 0, weight_hash: loaded ? m.hash : null,
        warm: loaded ? true : null, loaded_at: st.loaded_at, questions: st.questions, error: null, refusal: st.refusal,
        stats: loaded ? { forwards: Math.max(1, Math.round(st.questions * 0.8)), rows: st.questions, errors: 0, deadline_missed: st.questions ? 2 : 0, queue_rows: 0, last_path: 'eager', forward_ms_p50: 19.4, forward_ms_p95: 31.2, rows_per_forward_p50: 1 } : null,
      };
    });
    const picks = this.deciderState['laya-english'].picks;
    const refused = Object.values(this.deciderState).map((x) => x.refusal).filter(Boolean)[0] ?? null;
    return { contract_version: '1.0', enabled: true, why: null, ready: models.some((m) => m.state === 'loaded'), device: 'cuda', window: 'idle', families: ['laya'], models, planner: this.deciderPlan(), last_refusal: refused,
      questions: Object.entries(doc.questions).sort().map(([id, model]) => ({ id, model, picks: id === 'classify' ? picks : {} })), unserved: doc.models.map((m) => m.name).filter((n) => !LAYA.some((x) => x.name === n)),
      // L1: Laya for the model's scripts (the document's api over the server's defaults)
      api: { mode: 'on', rate_per_min: 240, max_texts: 16, max_chars: 100000, deadline_ms: 15000, threshold: 0.8, ...(doc.api ?? {}), stats: { calls: 12, texts: 19, errors: 1, rate_limited: 0, last_ms: 31.4, last_at: 1790990000 } } };
  }
  decidersRoute(method: string, rest: string[], body: Record<string, unknown>): Reply {
    if (rest[0] === 'status' && method === 'GET') return { status: 200, body: this.decidersStatus() };
    if (this.decidersOff) return { status: 404, body: { error: { type: 'not_found', message: 'this engine serves no decision model' } } };
    this.tick();
    if (rest[0] === 'models' && rest.length === 3 && method === 'POST') {
      const st = this.deciderState[rest[1]];
      if (!st) return { status: 404, body: { error: { type: 'not_found', message: `this engine does not serve '${rest[1]}'` } } };
      if (rest[2] === 'unload') {
        if (!['loaded', 'warming', 'failed'].includes(st.state)) return { status: 409, body: { error: { type: 'not_loaded', message: `${rest[1]} is ${st.state}, not loaded` } } };
        Object.assign(st, { state: 'not_loaded', loaded_at: null });
        return { status: 200, body: { contract_version: '1.0', name: rest[1], state: st.state } };
      }
      if (['loaded', 'loading', 'warming'].includes(st.state)) return { status: 202, body: { contract_version: '1.0', name: rest[1], state: st.state, decision: { ...this.deciderPlan(), allowed: true, reason: null, state: st.state } } };
      const d = this.deciderPlan();
      if (!d.allowed) {
        st.refusal = { ...d, at: Date.now() / 1000, state: st.state };
        return { status: 409, body: { contract_version: '1.0', name: rest[1], state: st.state, decision: { ...d, state: st.state }, error: { type: 'refused_memory', message: 'the residency planner refused the load: refused_memory' } } };
      }
      Object.assign(st, { state: 'loading', until: Date.now() + 1300, refusal: null });
      return { status: 202, body: { contract_version: '1.0', name: rest[1], state: 'loading', decision: { ...d, state: 'loading' } } };
    }
    if (rest[0] === 'autoload' && method === 'POST') return { status: 202, body: { contract_version: '1.0', queued: [] } };
    if (rest[0] === 'test' && method === 'POST') {
      const qs = (body.questions ?? {}) as Record<string, { type: string; criteria?: Record<string, unknown> }>;
      const state = String(body.state ?? '');
      if (!Object.keys(qs).length) return { status: 422, body: { error: { type: 'invalid_request_error', message: 'questions must contain at least one question' } } };
      const want = String(body.model ?? 'auto');
      const nonLatin = /[^\u0000-\u024f\s\p{P}\d]/u.test(state);
      const routed = want !== 'auto' ? want : nonLatin ? 'laya-multilingual' : 'laya-english';
      const st = this.deciderState[routed];
      if (!st) return { status: 422, body: { error: { type: 'invalid_request_error', message: `unknown model '${want}'` } } };
      let model = routed;
      const routing: Record<string, unknown> = { model, reason: want !== 'auto' ? `explicit model='${want}'` : nonLatin ? 'non-Latin script; the English checkpoint cannot read it' : 'English Latin text', detection: null, workflow: null };
      if (st.state !== 'loaded') {
        if (want !== 'auto') return { status: 503, body: { error: { type: 'loading', message: `${routed} was not loaded; its load has started` } } };
        model = 'laya-english';
        Object.assign(routing, { model, requested: routed, reason: `${routing.reason}; ${routed} is not loaded here, falling back to laya-english` });
      }
      const answers: Record<string, unknown> = {};
      for (const [qid, q] of Object.entries(qs)) {
        const opts = Object.keys(q.criteria ?? {});
        const pick = opts.find((o) => new RegExp(`\\b${o.replace(/[^a-z0-9_]/gi, '')}\\b`, 'i').test(state)) ?? opts[0];
        const rest2 = opts.length > 1 ? Math.round((0.14 / (opts.length - 1)) * 10000) / 10000 : 0;
        answers[qid] = { type: 'choice', choice: pick, probabilities: Object.fromEntries(opts.map((o) => [o, o === pick ? 0.86 : rest2])), confidence: 0.61, action: { act_probability: 0.42 } };
      }
      this.deciderState[model].questions += Object.keys(qs).length;
      return { status: 200, body: { contract_version: '1.0', model, answers, usage: { input_tokens: 58 + Math.round(state.length / 4), output_tokens: 0 }, routing, path: 'eager', latency_ms: 21.6 } };
    }
    return { status: 404, body: { error: { type: 'not_found', message: `no route ${method} /v1/conductor/deciders/${rest.join('/')}` } } };
  }

  /** VIS-44: the live `supervisor` scenario's decisions, as the journal holds them. */
  addLive(list: { request_id: string; ts_ms: number; finish: string | null; e: { task: string; decision: string; mode: 'act' | 'shadow'; reason: string; score: number | null; atTokens: number | null; hidden: number; tier?: number } }[]) {
    let k = 0;
    for (const x of list) {
      const e = x.e;
      this.events.push({
        id: `live${(k++).toString(16).padStart(4, '0')}${x.request_id.slice(-6)}`, ts_ms: x.ts_ms, request_id: x.request_id, session_key: 'c:ses_supervised', kind: 'monitor',
        task: e.task, tier: e.tier ?? 0, mode: e.mode, sp: e.task.startsWith('C') ? 'SP3' : 'SP2', rule: null, trigger: e.reason, score: e.score, decision: e.decision, action: e.mode === 'act' ? e.decision : null, cap: null,
        hidden_text_id: null, hidden_tokens: e.mode === 'act' ? e.hidden : 0, tokens_at_cut: e.atTokens, position: null,
        outcome: x.finish ? { kind: x.finish === 'tool_calls' ? 'tool_call' : 'text', acts: null, hidden_tokens: null, finish: x.finish } : null,
        next_checker: null, policy_version: 3, client_kind: 'opencode', profile: 'coding', detail: {}, label: null, label_note: null, labelled_at: null, text: null,
      });
    }
    this.events.sort((a, b) => (a.ts_ms as number) - (b.ts_ms as number));
  }

  /**
   * VIS-44's night: one opencode session of 140 turns over ten hours with 3 monitor actions, one
   * compaction (the router classed it by its shape) and 9 checker results, for the timeline test.
   */
  night140() {
    const key = 'c:ses_night_140';
    if (this.rows.some((r) => r.session_key === key)) return;
    const now = Date.now();
    const start = now - 10 * 3600_000;
    const step = (10 * 3600_000 - 60_000) / 140;
    const base = this.rows.find((r) => r.client_kind === 'opencode')!;
    const checkers = new Set([10, 25, 40, 55, 70, 85, 100, 115, 130]);
    const turns: Record<string, unknown>[] = [];
    for (let i = 0; i < 140; i++) {
      const ts = Math.round(start + i * step);
      const comp = i === 64;
      turns.push({
        ...base, id: `n140t${i.toString().padStart(3, '0')}`, ts_ms: ts, request_id: `chatcmpl-n140${i.toString(16).padStart(6, '0')}`, session_key: key, profile: 'overnight',
        class_id: comp ? 'summarize' : 'three_d', source: comp ? 'structural' : i === 0 ? 'decider' : 'sticky', confidence: comp ? 1 : 0.9, injected: [], replayed: null, shadow: null, label: null, labelled_at: null,
        outcome: { status: 200, finish_reason: comp ? 'stop' : 'tool_calls', tool_calls: comp ? 0 : 1, tool_names: comp ? [] : ['bash'], completion_tokens: 600, reasoning_tokens: 300, prompt_tokens: 60_000, ttft_ms: 900, total_ms: Math.round(step * 0.55) },
        next_checker: checkers.has(i) ? [{ checker: 'pagodacheck', target: '~/demos/pagoda-seasons', verdict: i === 130 ? 'PASS' : 'FAIL', reasons: [], readings: [{ gate: 'a', name: 'summer island colour differs by', value: Math.max(0.6, 15 - i / 10), op: '>', limit: 1 }] }] : null,
        acted: 0, changed: 0,
      });
    }
    this.rows.push(...turns);
    this.rows.sort((a, b) => (a.ts_ms as number) - (b.ts_ms as number));
    const acts: [number, string, string, string][] = [[22, 'D1', 'CLOSE_REASONING', 'repeat16(share=0.88)'], [66, 'S2', 'PROMPT_TAIL_NOTE', 'context_health(no re-read in 2 steps)'], [118, 'E1', 'CONTINUE_HIDDEN', 'empty_turn(case=B)']];
    acts.forEach(([i, task, decision, trigger], k) => {
      const t = turns[i];
      this.events.push({
        id: `n140e${k}`, ts_ms: (t.ts_ms as number) + 4000, request_id: t.request_id, session_key: key, kind: 'monitor', task, tier: 0, mode: 'act', sp: task === 'S2' ? 'SP4' : 'SP2', rule: null, trigger, score: task === 'D1' ? 0.88 : null, decision, action: decision, cap: null,
        hidden_text_id: null, hidden_tokens: 24, tokens_at_cut: task === 'D1' ? 2650 : null, position: 40_000, outcome: { kind: 'tool_call', acts: 1, hidden_tokens: 24, finish: 'tool_calls' },
        next_checker: null, policy_version: 3, client_kind: 'opencode', profile: 'overnight', detail: {}, label: null, label_note: null, labelled_at: null, text: `${task}: the hidden text of act ${k + 1}`,
      });
    });
    this.events.sort((a, b) => (a.ts_ms as number) - (b.ts_ms as number));
  }

  private isAct(e: Record<string, unknown>) {
    return e.mode === 'act' && e.decision != null && e.decision !== 'PASS';
  }

  private withEvents(rows: Record<string, unknown>[]) {
    const by = new Map<unknown, Record<string, unknown>[]>();
    for (const e of this.events) by.set(e.request_id, [...(by.get(e.request_id) ?? []), e]);
    return rows.map((r) => ({ ...r, events: by.get(r.request_id) ?? [] }));
  }

  monitorStats(hours: number) {
    const now = Date.now();
    const since = now - hours * 3600_000;
    const evs = this.events.filter((e) => (e.ts_ms as number) >= since);
    const fam = (t: unknown) => (/^D/.test(String(t)) ? 'thinking' : /^E/.test(String(t)) ? 'end of turn' : /^C/.test(String(t)) ? 'tool calls' : /^S\d/.test(String(t)) ? 'session' : 'other');
    const acts = evs.filter((e) => this.isAct(e));
    const shadow = evs.filter((e) => e.mode === 'shadow' && e.decision !== 'PASS');
    const byTask = new Map<string, { task: string; family: string; acts: number; shadow: number; pass: number }>();
    for (const e of evs) {
      const t = byTask.get(String(e.task)) ?? { task: String(e.task), family: fam(e.task), acts: 0, shadow: 0, pass: 0 };
      if (this.isAct(e)) t.acts++;
      else if (e.decision === 'PASS') t.pass++;
      else t.shadow++;
      byTask.set(t.task, t);
    }
    const prec = new Map<string, { task: string; mode: string; decisions: number; labelled: number; agree: number; precision: number | null }>();
    for (const e of evs) {
      if (e.decision === 'PASS') continue;
      const k = `${e.task}/${e.mode}`;
      const p = prec.get(k) ?? { task: String(e.task), mode: String(e.mode), decisions: 0, labelled: 0, agree: 0, precision: null };
      p.decisions++;
      if (e.label) {
        p.labelled++;
        if (e.label === 'right') p.agree++;
      }
      prec.set(k, p);
    }
    for (const p of prec.values()) p.precision = p.labelled ? Math.round((p.agree / p.labelled) * 10000) / 10000 : null;
    return {
      decisions: evs.length, acts: acts.length, shadow: shadow.length, gave_up: acts.filter((e) => e.decision === 'GIVE_UP').length,
      hidden_tokens: acts.reduce((s, e) => s + ((e.hidden_tokens as number) ?? 0), 0),
      by_task: [...byTask.values()].sort((a, b) => a.task.localeCompare(b.task)),
      by_action: acts.reduce<Record<string, number>>((o, e) => ((o[String(e.decision)] = (o[String(e.decision)] ?? 0) + 1), o), {}),
      hourly: Array.from({ length: hours }, (_, h) => {
        const lo = since + h * 3600_000;
        const inh = acts.filter((e) => (e.ts_ms as number) >= lo && (e.ts_ms as number) < lo + 3600_000);
        return { start_ms: lo, acts: inh.length, shadow: shadow.filter((e) => (e.ts_ms as number) >= lo && (e.ts_ms as number) < lo + 3600_000).length, by_family: inh.reduce<Record<string, number>>((o, e) => ((o[fam(e.task)] = (o[fam(e.task)] ?? 0) + 1), o), {}) };
      }),
      last_actions: [...acts].reverse().slice(0, 20).map((e) => ({ id: e.id, ts_ms: e.ts_ms, request_id: e.request_id, session_key: e.session_key, task: e.task, decision: e.decision, trigger: e.trigger, outcome: e.outcome, hidden_tokens: e.hidden_tokens, cap: e.cap })),
      precision: [...prec.values()].sort((a, b) => `${a.task}${a.mode}`.localeCompare(`${b.task}${b.mode}`)),
    };
  }

  stats(hours: number) {
    const now = Date.now();
    const since = now - hours * 3600_000;
    const rows = this.rows.filter((r) => (r.ts_ms as number) >= since);
    const count = (f: (r: Record<string, unknown>) => string) => rows.reduce<Record<string, number>>((o, r) => ((o[f(r)] = (o[f(r)] ?? 0) + 1), o), {});
    const stageMs: Record<string, number[]> = {};
    const failures: Record<string, number> = {};
    for (const r of rows) for (const s of r.stages as { stage: string; ms: number; error: string | null }[]) {
      (stageMs[s.stage] ??= []).push(s.ms);
      if (s.error) failures[s.error] = (failures[s.error] ?? 0) + 1;
    }
    const pctl = (v: number[], q: number) => {
      const s = [...v].sort((a, b) => a - b);
      return s.length ? Math.round(s[Math.min(s.length - 1, Math.round(q * (s.length - 1)))] * 100) / 100 : null;
    };
    const prec = new Map<string, { profile: string; decision: string; decisions: number; labelled: number; agree: number; precision: number | null }>();
    for (const r of rows) for (const [d, m] of Object.entries(r.decisions as Record<string, string>)) {
      if (m !== 'shadow') continue;
      const k = `${r.profile}/${d}`;
      const p = prec.get(k) ?? { profile: r.profile as string, decision: d, decisions: 0, labelled: 0, agree: 0, precision: null };
      p.decisions++;
      if (r.label) {
        p.labelled++;
        if (r.label === 'right') p.agree++;
      }
      prec.set(k, p);
    }
    for (const p of prec.values()) p.precision = p.labelled ? Math.round((p.agree / p.labelled) * 10000) / 10000 : null;
    const classes = new Map<string, { class_id: string; decisions: number; labelled: number; agree: number }>();
    for (const r of rows) {
      const c = classes.get(r.class_id as string) ?? { class_id: r.class_id as string, decisions: 0, labelled: 0, agree: 0 };
      c.decisions++;
      if (r.label) {
        c.labelled++;
        if (r.label === 'right') c.agree++;
      }
      classes.set(c.class_id, c);
    }
    const hourly = Array.from({ length: hours }, (_, h) => {
      const lo = since + h * 3600_000;
      const inh = rows.filter((r) => (r.ts_ms as number) >= lo && (r.ts_ms as number) < lo + 3600_000);
      return { start_ms: lo, total: inh.length, by_class: inh.reduce<Record<string, number>>((o, r) => ((o[r.class_id as string] = (o[r.class_id as string] ?? 0) + 1), o), {}), acted: inh.filter((r) => r.acted).length, shadow: inh.filter((r) => r.shadow).length };
    });
    return {
      contract_version: '1.0', hours, decisions: rows.length, acted: rows.filter((r) => r.acted).length, shadow: rows.filter((r) => r.shadow).length, changed: rows.filter((r) => r.changed).length,
      injections: rows.filter((r) => (r.injected as unknown[]).length).length, injected_tokens: rows.reduce((s, r) => s + (r.injected as { tokens: number }[]).reduce((a, n) => a + n.tokens, 0), 0),
      by_class: count((r) => r.class_id as string), by_source: count((r) => r.source as string), by_effort: count((r) => (r.effort as string) ?? 'none'),
      classifier: Object.fromEntries(Object.entries(stageMs).map(([k, v]) => [k, { calls: v.length, p50_ms: pctl(v, 0.5), p95_ms: pctl(v, 0.95) }])),
      failures, fail_open_rate: rows.length ? Math.round((rows.filter((r) => r.kept === 'fail_open').length / rows.length) * 10000) / 10000 : null,
      model_decided: rows.filter((r) => r.source === 'decider' || r.source === 'qwen').length,
      labels: { right: rows.filter((r) => r.label === 'right').length, wrong: rows.filter((r) => r.label === 'wrong').length },
      precision: [...prec.values()].sort((a, b) => `${a.profile}${a.decision}`.localeCompare(`${b.profile}${b.decision}`)),
      classes: [...classes.values()].sort((a, b) => b.decisions - a.decisions),
      hourly,
      monitor: this.monitorStats(hours),
      roles: this.health(),
    };
  }

  health() {
    const roles = this.cur('roles').doc as Record<string, { mode: string }>;
    return Object.fromEntries(Object.entries(roles).map(([n, r]) => {
      if (r.mode === 'off') return [n, { up: false, mode: 'off', ms: null, error: 'off' }];
      if (this.roleDown === n) return [n, { up: false, mode: r.mode, ms: 3.1, error: 'unreachable' }];
      return [n, { up: true, mode: r.mode, ms: n === 'decider' ? 4.2 : 11.8, detail: { models: n === 'decider' ? ['english', 'multilingual', 'typed-decisions'] : ['TandemLLM'] } }];
    }));
  }

  sessions() {
    interface S {
      key: string;
      client_kind: string;
      profile: string;
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
    const by = new Map<string, S>();
    for (const r of this.rows) {
      const k = r.session_key as string;
      const s: S = by.get(k) ?? { key: k, client_kind: r.client_kind as string, profile: r.profile as string, started_ms: r.ts_ms as number, turns: 0, notes: 0, note_tokens: 0, classes: {}, class_id: null, last_ms: r.ts_ms as number, last_checker: null, labels: 0, acts: 0, gave_ups: 0, shadow: 0, interventions: 0 };
      s.turns++;
      s.last_ms = r.ts_ms as number;
      s.class_id = r.class_id as string;
      s.classes[s.class_id] = (s.classes[s.class_id] ?? 0) + 1;
      for (const n of r.injected as { tokens: number }[]) {
        s.notes++;
        s.interventions++;
        s.note_tokens += n.tokens;
      }
      const nc = r.next_checker as { verdict: string }[] | null;
      if (nc?.length) s.last_checker = nc[nc.length - 1].verdict;
      if (r.label) s.labels++;
      by.set(k, s);
    }
    for (const e of this.events) {
      const s = by.get(e.session_key as string);
      if (!s) continue;
      if (this.isAct(e)) {
        s.acts++;
        s.interventions++;
        if (e.decision === 'GIVE_UP') s.gave_ups++;
      } else if (e.mode === 'shadow' && e.decision !== 'PASS') s.shadow++;
    }
    return [...by.values()].sort((a, b) => b.last_ms - a.last_ms);
  }

  explain(text: string, profile: string | null) {
    const router = this.cur('router').doc as { classes: { id: string; label: string; policy: { effort: string; packages: string[]; rules: string[] } }[]; effort: { levels: Record<string, { note: string }> }; injection: { wrapper: string } };
    const prof = profile ?? 'coding';
    const pdoc = this.docs.get(`profiles/${prof}`)?.at(-1)?.doc as { mode: string; decisions: Record<string, string> } | undefined;
    if (!pdoc) return { status: 400, body: { error: { type: 'bad_request', message: `no profile '${prof}'` } } };
    if (pdoc.mode === 'raw') return { status: 200, body: { contract_version: '1.0', profile: prof, profile_version: 0, mode: 'raw', active: false } };
    // a mock stands in for the models: a class whose id or label is named, else the first
    const low = text.toLowerCase();
    const hit = router.classes.find((c) => low.includes(c.id) || low.includes(c.label.toLowerCase()));
    const cls = hit ?? router.classes[0];
    const conf = hit ? 0.88 : 0.41;
    const pol = cls.policy;
    const stages = [{ stage: 'decider', label: cls.id, confidence: conf, probs: { [cls.id]: conf }, ms: 21.4, error: null, model: 'laya-english', accept: 0.6, ...(hit ? {} : { below: true }) }, ...(hit ? [] : [{ stage: 'qwen', label: 'code_edit', confidence: 0.71, probs: { code_edit: 0.71 }, ms: 612.2, error: null, model: 'qwen38-spark-engine', accept: 0.55 }])];
    const final = hit ? cls : router.classes.find((c) => c.id === 'code_edit')!;
    // SRV-157: a new session is small; the class's levels per size pick under its effort
    const bySize = (pdoc as { thinking?: { by_size?: string } }).thinking?.by_size ?? 'shadow';
    const map = (final.policy as { effort_by_size?: Record<string, string> }).effort_by_size;
    const order = ['off', 'low', 'medium', 'high', 'xhigh'];
    const picked = map?.small && order.indexOf(map.small) <= order.indexOf(final.policy.effort) ? map.small : final.policy.effort;
    const size = bySize === 'off' ? null : { mode: bySize, size: 'small', score: 0, now: 'small', signals: { session_tokens: { value: 0, size: 'small' }, files: { value: 0, size: 'small' }, turns: { value: 1, size: 'small' } }, ...(map ? { ceiling: final.policy.effort, effort: picked, applied: bySize === 'on' } : {}) };
    const effort = size?.applied ? picked : final.policy.effort;
    const parts = [router.effort.levels[size?.applied ? effort : pol.effort]?.note, pol.packages[0] ? `# ${pol.packages[0]}\n\n(the package's text on the box)` : '', pdoc.decisions.rules === 'on' ? pol.rules.map((x) => `- ${x}`).join('\n') : ''].filter(Boolean);
    return {
      status: 200,
      body: {
        contract_version: '1.0', profile: prof, profile_version: this.docs.get(`profiles/${prof}`)!.at(-1)!.version, mode: pdoc.mode, active: true,
        class_id: final.id, confidence: hit ? conf : 0.71, source: hit ? 'decider' : 'qwen', kept: null, stages, classifier_ms: stages.reduce((a, s) => a + s.ms, 0),
        effort, budget: effort === 'low' ? 1024 : null, budget_source: effort === 'low' ? 'class' : 'server', thinking: effort === 'off' ? 'off' : 'on',
        decisions: pdoc.decisions, shadow: size && map && !size.applied ? { size: { size: 'small', score: 0, effort: picked, ceiling: final.policy.effort } } : null, extra: size ? { size } : {}, note: parts.length ? router.injection.wrapper.replace('{text}', parts.join('\n\n')) : null,
        note_parts: parts.length ? [{ kind: 'effort', name: effort }] : [], thinking_off: effort === 'off',
      },
    };
  }

  evals() {
    const reports: Record<string, unknown>[] = [
      { ...REPORT, file: 'classifier-20260930T235236.json' },
      { ...SUITE_REPORT, file: 'suite-2026-10-01-032011.json', supersedes: { file: 'suite-2026-10-01-020000.json', why: 'the reader read every export cut at 64 KiB (OPS-63)' } },
      // the report it replaced, kept as published (the box marks it superseded on GET)
      { ...SUITE_REPORT, file: 'suite-2026-10-01-020000.json', superseded_by: { file: 'suite-2026-10-01-032011.json', why: 'the reader read every export cut at 64 KiB (OPS-63)' } },
      {
        // an older report in the first format: two builds in one pair (not comparable)
        kind: 'suite', file: 'suite-2026-09-30-054000.json', night: '2026-09-30', created_at: '2026-10-01T05:40:00Z',
        rows: [
          { suite: 'spiral portrait', arm: 'raw', gates_passed: 3, gates_total: 4, minutes_to_pass: null, tokens: 212_000, interventions: 0, build: '0.3.1-test', night: '2026-10-01', n: 2 },
          { suite: 'spiral portrait', arm: 'TandemLLM', gates_passed: 4, gates_total: 4, minutes_to_pass: 13, tokens: 164_000, interventions: 5, build: '0.3.1-test', night: '2026-10-01', n: 2 },
          { suite: 'pagoda seasons', arm: 'raw', gates_passed: 2, gates_total: 4, minutes_to_pass: null, tokens: 480_000, interventions: 0, build: '0.3.0', night: '2026-09-30', n: 1 },
          { suite: 'pagoda seasons', arm: 'TandemLLM', gates_passed: 3, gates_total: 4, minutes_to_pass: null, tokens: 405_000, interventions: 9, build: '0.3.1-test', night: '2026-10-01', n: 1 },
        ],
      },
    ];
    if (this.guard === 'single') {
      reports.push({
        kind: 'speed_guard', file: `speed-guard-${ENGINE.version}-${ENGINE.code_sha}.json`, build: ENGINE, created_at: '2026-10-01T11:55:00Z',
        runs_each: 1, noise_estimated: false, verdict: 'no_noise_estimate',
        why: '1 run a side: the run-to-run noise cannot be estimated, so this row cannot say whether any difference is real; the deltas are as measured',
        stats: [
          { stat: 'mean', base: 49.98, cand: 50.54, delta_pct: 1.12, higher_is_better: true, noise_pct: null, verdict: 'no noise estimate' },
          { stat: 'ttft_p50_ms', base: 424.9, cand: 425.6, delta_pct: 0.16, higher_is_better: false, noise_pct: null, verdict: 'no noise estimate' },
          { stat: 'ms_blk', base: 92.84, cand: 92.82, delta_pct: -0.02, higher_is_better: false, noise_pct: null, verdict: 'no noise estimate' },
        ],
      });
    } else if (this.guard) {
      reports.push({
        kind: 'speed_guard', file: 'speed-guard.json', build: ENGINE, created_at: '2026-10-01T04:00:00Z', runs_each: 3, noise_estimated: true, verdict: 'pass',
        why: 'nothing resolved worse (3 runs a side)',
        stats: [{ stat: 'mean', base: 50.04, cand: 50.22, delta_pct: 0.36, higher_is_better: true, noise_pct: 2.1, verdict: 'not resolved' }],
      });
    }
    return { contract_version: '1.0', reports, engine: ENGINE };
  }

  /** What the session tasks hold (SRV-121): the pagoda night with a noted gate and a compaction
   * being watched, and a session whose client has gone with the push on. */
  watch() {
    const now = Date.now() / 1000;
    return [
      { session: 'c:ses_night_pagoda', requests: 214, last_seen: now - 40, ended_at: now - 60, running: true, note_at: now - 900, idle_minutes: null, client_gone: false,
        watch: { task: 'S4', minutes: 20, mode: 'act', profile: 'overnight', client: 'opencode', notify: true },
        gates: [{ task: 'S1', checker: 'pagodacheck', gate: 'a', op: '>', limit: 1, values: [3.5, 12.3, 3.5], noted: true }],
        compaction: { task: 'S2', count: 2, watching: true, steps: 1 },
        progress: { task: 'S4', last: 'edit: changed', minutes: 6.5 } },
      { session: 'c:ses_mona_spiral', requests: 37, last_seen: now - 1500, ended_at: now - 1490, running: false, note_at: null, idle_minutes: 24.8, client_gone: true,
        watch: { task: 'S4', minutes: 20, mode: 'act', profile: 'overnight', client: 'opencode', notify: true },
        gates: [], compaction: null, progress: { task: 'S4', last: 'sketchcheck: PASS', minutes: 25.1 } },
    ];
  }

  // ---------------------------------------------------------------- routing
  handle(method: string, path: string, q: URLSearchParams, body: Record<string, unknown>, who: string): Reply {
    const parts = path.split('/').slice(3).filter(Boolean).map(decodeURIComponent);
    const head = parts[0];
    const doc = (kind: string, name: string | null) => {
      const k = this.key(kind, name);
      const list = this.docs.get(k);
      if (!list) return null;
      return { list, cur: list.at(-1)! };
    };
    if (head === 'config') {
      const documents: Record<string, unknown> = {};
      for (const [k, list] of this.docs) documents[k] = { version: list.at(-1)!.version, loaded_at: iso(Date.now() - 3_600_000), error: null, note: list.at(-1)!.note };
      return { status: 200, body: { contract_version: '1.0', server: { composite: 'TandemLLM', raw_model: 'qwen38-spark-engine', reasoning_effort: 'medium', global_mode: null, monitor_mode: 'on', self_url: 'http://127.0.0.1:8000' }, store: { root: '~/.qwen38-spark-engine/conductor', readonly: false, writes: 2, documents }, journal: { path: '~/.qwen38-spark-engine/usage/ledger.sqlite3', ring: this.rows.length, events_ring: this.events.length, monitor_rows: this.events.length, table: 'conductor_events', queue: 0, dropped: 0, errors: 0, retention_days: 30 }, time: iso(Date.now()) } };
    }
    if (head === 'meta') return { status: 200, body: { contract_version: '1.0', ...SEEDS.meta } };
    if (head === 'memory' && parts[1] === 'disk') {
      const list = this.docs.get('settings')!;
      const d = ((list.at(-1)!.doc as { memory?: { disk_cache?: Record<string, unknown> } }).memory?.disk_cache ?? {});
      return diskRoute(this.disk, method, parts.slice(1), { mode: 'on', budget_gb: 50, ttl_days: 7, admit_after: 2, min_free_gb: 50, ...d }, list.at(-1)!.version);
    }
    if (head === 'memory') {
      const t = ((this.cur('settings').doc as { memory?: { transcripts?: Record<string, unknown> } }).memory?.transcripts ?? {});
      return memoryRoute(this.memory, method, parts.slice(1), q, body, { scope: 'tools', retention_days: 30, clip_chars: 4000, loop_chars: 200, ...t });
    }
    if (head === 'profiles' && parts.length === 1) {
      const settings = this.cur('settings').doc as { default_profile: string };
      const profiles = [...this.docs.keys()].filter((k) => k.startsWith('profiles/')).map((k) => k.slice(9)).sort().map((name) => {
        const { cur } = doc('profiles', name)!;
        const d = cur.doc as { mode: string; description: string; clients: string[]; default_class: string; decisions: Record<string, string>; tasks: Record<string, { mode: string }> };
        const tasks = Object.values(d.tasks).map((t) => t.mode);
        const decs = Object.values(d.decisions);
        return { name, version: cur.version, mode: d.mode, description: d.description, clients: d.clients, default_class: d.default_class, locked: LOCKED[name] ?? null, decisions: d.decisions, counts: { tasks: { off: tasks.filter((x) => x === 'off').length, shadow: tasks.filter((x) => x === 'shadow').length, act: tasks.filter((x) => x === 'act').length }, decisions: { off: decs.filter((x) => x === 'off').length, shadow: decs.filter((x) => x === 'shadow').length, on: decs.filter((x) => x === 'on').length } }, last_change: { version: cur.version, note: cur.note, author: cur.author, created_at: cur.created_at }, error: null };
      });
      return { status: 200, body: { contract_version: '1.0', default_profile: settings.default_profile, profiles } };
    }
    if (head === 'classify' && method === 'POST') return this.explain(String(body.text ?? ''), (body.profile as string) ?? null);
    if (head === 'events') {
      if (parts.length === 2) {
        const e = this.events.find((x) => x.id === parts[1]);
        return e ? { status: 200, body: { contract_version: '1.0', event: e } } : { status: 404, body: { error: { type: 'not_found', message: `no monitor event ${parts[1]}` } } };
      }
      const limit = Number(q.get('limit') ?? 100);
      const now = Date.now();
      // `request` (VIS-44's drawer): one request's decisions, those that have happened by now
      const evs = [...this.events].reverse().filter((e) => (e.ts_ms as number) <= now && (!q.get('session') || e.session_key === q.get('session')) && (!q.get('request') || e.request_id === q.get('request')) && (!q.get('task') || e.task === q.get('task')) && (!q.get('mode') || e.mode === q.get('mode')) && (q.get('acted') !== '1' || this.isAct(e)));
      return { status: 200, body: { contract_version: '1.0', events: evs.slice(0, limit), monitor: true } };
    }
    if (head === 'decisions') {
      if (parts.length === 1) {
        const limit = Number(q.get('limit') ?? 100);
        const session = q.get('session');
        return { status: 200, body: { contract_version: '1.0', decisions: this.withEvents([...this.rows].reverse().filter((r) => !session || r.session_key === session).slice(0, limit)) } };
      }
      const row = this.rows.find((r) => r.id === parts[1]) ?? this.events.find((e) => e.id === parts[1]);
      if (!row) return { status: 404, body: { error: { type: 'not_found', message: `no decision ${parts[1]}` } } };
      if (parts[2] === 'label' && method === 'POST') {
        const l = body.label as string;
        row.label = l === 'clear' ? null : l;
        row.labelled_at = l === 'clear' ? null : Date.now();
        return { status: 200, body: { contract_version: '1.0', id: row.id, label: row.label } };
      }
      return { status: 200, body: { contract_version: '1.0', decision: row } };
    }
    if (head === 'stats') return { status: 200, body: this.stats(Number(q.get('hours') ?? 24)) };
    if (head === 'sessions') {
      if (parts.length === 1) return { status: 200, body: { contract_version: '1.0', sessions: this.sessions(), live: this.sessions().slice(0, 2).map((s) => ({ key: s.key, class_id: s.class_id, confidence: 0.9, source: 'sticky', turns: s.turns, notes: s.notes, note_tokens: s.note_tokens })) } };
      const rows = this.rows.filter((r) => r.session_key === parts[1]);
      const evs = this.events.filter((e) => e.session_key === parts[1]);
      if (parts[2] === 'export') {
        const items = [
          ...rows.flatMap((r) => (r.injected as { text: string; tokens: number; pos: number; reason: string; parts: unknown[] }[]).map((n) => ({ source: 'router', id: r.id, ts_ms: r.ts_ms, request_id: r.request_id, task: 'route', action: 'PROMPT_TAIL_NOTE', reason: n.reason, text: n.text, tokens: n.tokens, pos: n.pos, hidden: true, parts: n.parts, outcome: r.outcome, next_checker: r.next_checker }))),
          ...evs.filter((e) => this.isAct(e)).map((e) => ({ source: 'monitor', id: e.id, ts_ms: e.ts_ms, request_id: e.request_id, task: e.task, action: e.decision, reason: e.trigger, text: e.text, tokens: e.hidden_tokens ?? 0, pos: e.position, hidden: !!e.hidden_tokens, tokens_at_cut: e.tokens_at_cut, cap: e.cap, outcome: e.outcome, next_checker: e.next_checker })),
        ].sort((a, b) => (a.ts_ms as number) - (b.ts_ms as number));
        const summary = `${items.length} intervention${items.length === 1 ? '' : 's'} in ${rows.length} turn${rows.length === 1 ? '' : 's'}`;
        if (q.get('format') === 'md') return { status: 200, raw: { type: 'text/markdown; charset=utf-8', data: `# Session ${parts[1]}\n\n${summary}.\n\n` + items.map((n) => `${n.task} ${n.action}: because ${n.reason}\n\n` + '```text\n' + (n.text ?? '') + '\n```').join('\n\n') } };
        return { status: 200, body: { contract_version: '1.0', session: parts[1], turns: rows.length, interventions: items.length, hidden_texts: items.filter((i) => i.hidden).length, summary, injections: items } };
      }
      if (!rows.length) return { status: 404, body: { error: { type: 'not_found', message: `no session ${parts[1]}` } } };
      return { status: 200, body: { contract_version: '1.0', session: parts[1], live: null, turns: this.withEvents(rows), events: evs } };
    }
    if (head === 'evals' && method === 'POST') {
      const rep = body.report as Record<string, unknown> | undefined;
      if (!rep || rep.kind !== 'suite' || !Array.isArray(rep.rows)) return { status: 422, body: { error: { type: 'invalid_report', message: "report.kind: must be 'suite'" } } };
      return { status: 201, body: { contract_version: '1.0', file: `suite-${String(rep.night)}-000000.json` } };
    }
    if (head === 'evals') return { status: 200, body: this.evals() };
    if (head === 'watch') return { status: 200, body: { contract_version: '1.0', monitor: true, sessions: this.watch() } };
    if (head === 'roles' && parts[2] === 'test') {
      if (this.roleDown === parts[1]) return { status: 200, body: { contract_version: '1.0', role: parts[1], ok: false, error: `${parts[1] === 'decider' ? 'decider' : 'qwen'}_unreachable` } };
      return { status: 200, body: { contract_version: '1.0', role: parts[1], ok: true, ms: 21.7, question: 'Rename the variable `cnt` to `count` in utils.py and run the tests.', answer: { stage: parts[1] === 'decider' ? 'decider' : 'qwen', label: 'code_edit', confidence: 0.9, probs: { code_edit: 0.9 }, ms: 21.7, error: null, model: 'laya-english' } } };
    }
    if (head === 'roles' && parts.length === 1 && method === 'GET') {
      const { cur } = doc('roles', null)!;
      const roles = cur.doc as Record<string, { kind: string; tokenizer_sha256: string }>;
      const contract = {
        engine_tokenizer_sha256: ENGINE_SHA,
        roles: Object.fromEntries(Object.entries(roles).map(([n, r]) => [n, { sharing_allowed: ['text', ...(r.tokenizer_sha256 === ENGINE_SHA ? ['tokens'] : []), ...(r.kind === 'probe' ? ['states'] : [])], tokenizer_match: r.tokenizer_sha256 === 'none' ? null : r.tokenizer_sha256 === ENGINE_SHA }])),
      };
      return { status: 200, body: { contract_version: '1.0', version: cur.version, doc: cur.doc, meta: { version: cur.version, note: cur.note, author: cur.author, created_at: cur.created_at, restored_from: cur.restored_from, loaded_at: iso(Date.now()), error: null }, health: this.health(), contract } };
    }
    if (head === 'packages' && parts[1] === 'scan') {
      const pk = this.cur('packages').doc as { packages: Record<string, unknown> };
      const skills = ['one-line-portrait', 'pencil-sketch', 'photo-to-3d', 'spiral-portrait', 'voxel-seasons', 'watercolour'].map((n) => ({ name: n, description: `Recipe for ${n.replace(/-/g, ' ')}.`, path: `~/.config/opencode/skills/${n}/SKILL.md`, chars: 7000 + n.length * 300, registered: n in pk.packages }));
      return { status: 200, body: { contract_version: '1.0', skills } };
    }
    if (head === 'profiles' && parts[2] === 'mode' && method === 'POST') {
      const d = doc('profiles', parts[1]);
      if (!d) return { status: 404, body: { error: { type: 'not_found', message: `no profile ${parts[1]}` } } };
      const r = this.put('profiles', parts[1], { ...clone(d.cur.doc), mode: body.mode }, String(body.note ?? `mode to ${body.mode}`), who, (body.base_version as number) ?? null);
      if (r.status === 200) (r.body as Record<string, unknown>).mode = body.mode;
      return r;
    }
    if (head === 'residency' && parts[1] === 'plan' && method === 'GET') return { status: 200, body: this.residencyPlan() };
    if (head === 'deciders' && ['status', 'models', 'autoload', 'test'].includes(parts[1])) return this.decidersRoute(method, parts.slice(1), body);
    if (['settings', 'router', 'roles', 'packages', 'monitor', 'profiles', 'residency', 'deciders'].includes(head)) {
      const name = head === 'profiles' ? parts[1] : null;
      const rest = head === 'profiles' ? parts.slice(2) : parts.slice(1);
      const d = doc(head, name);
      if (!d) return { status: 404, body: { error: { type: 'not_found', message: `no ${head} ${name ?? ''}` } } };
      const meta = (v: Version) => ({ version: v.version, note: v.note, author: v.author, created_at: v.created_at, restored_from: v.restored_from, loaded_at: iso(Date.now()), error: null });
      if (method === 'GET' && !rest.length) {
        const out: Record<string, unknown> = { contract_version: '1.0', kind: head, name, version: d.cur.version, doc: d.cur.doc, meta: meta(d.cur) };
        if (head === 'profiles') out.locked = LOCKED[name!] ?? null;
        if (head === 'packages') out.status = Object.fromEntries(Object.entries((d.cur.doc as { packages: Record<string, { source: string; body: string }> }).packages).map(([n, p]) => [n, { available: p.source === 'inline' || n !== 'one-line-portrait', error: p.source === 'inline' || n !== 'one-line-portrait' ? null : 'package_file_missing', path: p.source === 'inline' ? null : `~/.config/opencode/skills/${n}/SKILL.md`, chars: p.source === 'inline' ? p.body.length : 9000, tokens_est: Math.round((p.source === 'inline' ? p.body.length : 9000) / 3.5) }]));
        return { status: 200, body: out };
      }
      if (method === 'PUT' && !rest.length) return this.put(head, name, body.doc as Doc, String(body.note ?? ''), who, (body.base_version as number) ?? null);
      if (method === 'GET' && rest[0] === 'versions' && rest.length === 1) return { status: 200, body: { contract_version: '1.0', kind: head, name, versions: [...d.list].reverse().map((v) => ({ ...meta(v), loaded_at: undefined, error: undefined, current: v === d.cur })).map(({ loaded_at: _l, error: _e, ...v }) => v) } };
      if (method === 'GET' && rest[0] === 'versions') {
        const v = d.list.find((x) => x.version === Number(rest[1]));
        return v ? { status: 200, body: { contract_version: '1.0', kind: head, name, version: v.version, doc: v.doc } } : { status: 404, body: { error: { type: 'not_found', message: `no version ${rest[1]}` } } };
      }
      if (method === 'GET' && rest[0] === 'diff') {
        const a = d.list.find((x) => x.version === Number(q.get('from') ?? d.cur.version));
        const b = d.list.find((x) => x.version === Number(q.get('to') ?? d.cur.version));
        if (!a || !b) return { status: 404, body: { error: { type: 'not_found', message: 'no such version' } } };
        return { status: 200, body: { contract_version: '1.0', kind: head, name, from: a.version, to: b.version, changes: diff(a.doc, b.doc) } };
      }
      if (method === 'POST' && rest[0] === 'check') {
        const errs = this.check(head, name, body.doc as Doc);
        return { status: 200, body: { contract_version: '1.0', valid: !errs.length, errors: errs, changes: diff(d.cur.doc, body.doc), base_version: d.cur.version } };
      }
      if (method === 'POST' && rest[0] === 'restore') {
        const v = d.list.find((x) => x.version === Number(body.version));
        if (!v) return { status: 404, body: { error: { type: 'not_found', message: `no version ${body.version}` } } };
        return this.put(head, name, clone(v.doc), String(body.note ?? `restore of v${v.version}`), who, null, v.version);
      }
    }
    return { status: 404, body: { error: { type: 'not_found', message: `no route ${method} ${path}` } } };
  }
}
