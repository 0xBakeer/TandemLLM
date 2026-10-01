// The mock engine's conductor (/v1/conductor/*, contract docs/contract/conductor-v1): the server's
// own built-in documents (seeds.json, written by tools/conductor_contract.py), versions and writes
// in memory with the server's validation (the same config schemas through ajv), a day of seeded
// decisions over a handful of sessions, and the aggregates the server computes. Dev and preview
// only; never in dist/.
//
// Test switches, POST /__mock/conductor: {requireToken: true} makes writes take only the admin
// bearer token (as a server with the dashboard login off), {guard: true} adds a speed-guard report
// for the serving build, {reset: true} starts over, {roleDown: "decider"} takes a role offline.

import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import Ajv2020 from 'ajv/dist/2020.js';
import { rng } from './generate.ts';

const DIR = fileURLToPath(new URL('../../docs/contract/conductor-v1/', import.meta.url));
const SEEDS = JSON.parse(readFileSync(join(DIR, 'seeds.json'), 'utf8'));
const LOCKED: Record<string, string> = SEEDS.meta.locked;
// the classifier report measured on the Spark (tools/conductor_eval.py), per-prompt rows left out
const REPORT = JSON.parse(readFileSync(fileURLToPath(new URL('../../tests/fixtures/conductor/classifier-report-2026-10-01.json', import.meta.url)), 'utf8'));
export const ENGINE = { version: '0.3.1-test', code_sha: '4c1f0e9a7b2d' };

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
for (const k of ['settings', 'router', 'roles', 'packages', 'profiles']) {
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
  requireToken = false;
  guard = false;
  roleDown: string | null = null;
  seed: number;

  constructor(seed = 99) {
    this.seed = seed;
    this.reset();
  }

  reset() {
    this.docs.clear();
    for (const k of ['settings', 'router', 'roles', 'packages']) this.docs.set(k, [this.v0(SEEDS[k])]);
    for (const [n, d] of Object.entries(SEEDS.profiles)) this.docs.set(`profiles/${n}`, [this.v0(d as Doc)]);
    // the chat profile runs effort in shadow in the seeded day (the promotion cases)
    this.put('profiles', 'chat', { ...clone(this.cur('profiles/chat').doc), decisions: { classify: 'on', effort: 'shadow', packages: 'on', rules: 'off' } }, 'effort to shadow', 'seed');
    this.put('profiles', 'coding', { ...clone(this.cur('profiles/coding').doc), decisions: { classify: 'on', effort: 'on', packages: 'shadow', rules: 'shadow' } }, 'packages and rules to shadow', 'seed');
    this.rows = this.day();
    this.requireToken = false;
    this.guard = false;
    this.roleDown = null;
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
          const src = first ? (r() < 0.85 ? 'arbiter' : 'qwen') : s.client === 'opencode' ? 'sticky' : r() < 0.7 ? 'arbiter' : 'qwen';
          const cls = s.cls;
          const pol = router.classes.find((c) => c.id === cls)!.policy;
          const stages = src === 'arbiter' || src === 'qwen'
            ? [
                src === 'qwen'
                  ? { stage: 'arbiter', label: null, confidence: 0, probs: {}, ms: 400.5, error: 'arbiter_timeout', model: null, accept: 0.6 }
                  : { stage: 'arbiter', label: cls, confidence: conf ?? 0.8, probs: Object.fromEntries(classes.map((c) => [c, c === cls ? (conf ?? 0.8) : (1 - (conf ?? 0.8)) / (classes.length - 1)])), ms: 18 + r() * 30, error: null, model: 'laya-english', accept: 0.6 },
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
      model_decided: rows.filter((r) => r.source === 'arbiter' || r.source === 'qwen').length,
      labels: { right: rows.filter((r) => r.label === 'right').length, wrong: rows.filter((r) => r.label === 'wrong').length },
      precision: [...prec.values()].sort((a, b) => `${a.profile}${a.decision}`.localeCompare(`${b.profile}${b.decision}`)),
      classes: [...classes.values()].sort((a, b) => b.decisions - a.decisions),
      hourly,
      monitor: { gave_up: 1, acts: 14, shadow: 57, precision: [{ task: 'D1', decisions: 44, labelled: 41, agree: 38, precision: 0.9268, mode: 'shadow' }, { task: 'E2', decisions: 12, labelled: 12, agree: 11, precision: 0.9167, mode: 'shadow' }] },
      roles: this.health(),
    };
  }

  health() {
    const roles = this.cur('roles').doc as Record<string, { mode: string }>;
    return Object.fromEntries(Object.entries(roles).map(([n, r]) => {
      if (r.mode === 'off') return [n, { up: false, mode: 'off', ms: null, error: 'off' }];
      if (this.roleDown === n) return [n, { up: false, mode: r.mode, ms: 3.1, error: 'unreachable' }];
      return [n, { up: true, mode: r.mode, ms: n === 'decider' ? 4.2 : 11.8, detail: { models: n === 'decider' ? ['english', 'multilingual', 'typed-decisions'] : ['TandemLLM', 'qwen38-spark-engine'] } }];
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
    }
    const by = new Map<string, S>();
    for (const r of this.rows) {
      const k = r.session_key as string;
      const s: S = by.get(k) ?? { key: k, client_kind: r.client_kind as string, profile: r.profile as string, started_ms: r.ts_ms as number, turns: 0, notes: 0, note_tokens: 0, classes: {}, class_id: null, last_ms: r.ts_ms as number, last_checker: null, labels: 0 };
      s.turns++;
      s.last_ms = r.ts_ms as number;
      s.class_id = r.class_id as string;
      s.classes[s.class_id] = (s.classes[s.class_id] ?? 0) + 1;
      for (const n of r.injected as { tokens: number }[]) {
        s.notes++;
        s.note_tokens += n.tokens;
      }
      const nc = r.next_checker as { verdict: string }[] | null;
      if (nc?.length) s.last_checker = nc[nc.length - 1].verdict;
      if (r.label) s.labels++;
      by.set(k, s);
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
    const parts = [router.effort.levels[pol.effort]?.note, pol.packages[0] ? `# ${pol.packages[0]}\n\n(the package's text on the box)` : '', pdoc.decisions.rules === 'on' ? pol.rules.map((x) => `- ${x}`).join('\n') : ''].filter(Boolean);
    const stages = [{ stage: 'arbiter', label: cls.id, confidence: conf, probs: { [cls.id]: conf }, ms: 21.4, error: null, model: 'laya-english', accept: 0.6, ...(hit ? {} : { below: true }) }, ...(hit ? [] : [{ stage: 'qwen', label: 'code_edit', confidence: 0.71, probs: { code_edit: 0.71 }, ms: 612.2, error: null, model: 'qwen38-spark-engine', accept: 0.55 }])];
    const final = hit ? cls : router.classes.find((c) => c.id === 'code_edit')!;
    return {
      status: 200,
      body: {
        contract_version: '1.0', profile: prof, profile_version: this.docs.get(`profiles/${prof}`)!.at(-1)!.version, mode: pdoc.mode, active: true,
        class_id: final.id, confidence: hit ? conf : 0.71, source: hit ? 'arbiter' : 'qwen', kept: null, stages, classifier_ms: stages.reduce((a, s) => a + s.ms, 0),
        effort: final.policy.effort, budget: final.policy.effort === 'low' ? 1024 : null, budget_source: final.policy.effort === 'low' ? 'class' : 'server', thinking: final.policy.effort === 'off' ? 'off' : 'on',
        decisions: pdoc.decisions, shadow: null, extra: {}, note: parts.length ? router.injection.wrapper.replace('{text}', parts.join('\n\n')) : null,
        note_parts: parts.length ? [{ kind: 'effort', name: final.policy.effort }] : [], thinking_off: final.policy.effort === 'off',
      },
    };
  }

  evals() {
    const reports: Record<string, unknown>[] = [
      { ...REPORT, file: 'classifier-20260930T235236.json' },
      {
        kind: 'suite', file: 'suites-2026-10-01.json', created_at: '2026-10-01T05:40:00Z',
        rows: [
          { suite: 'spiral portrait', arm: 'raw', gates_passed: 3, gates_total: 4, minutes_to_pass: null, tokens: 212_000, interventions: 0, build: '0.3.1-test', night: '2026-10-01', n: 2 },
          { suite: 'spiral portrait', arm: 'TandemLLM', gates_passed: 4, gates_total: 4, minutes_to_pass: 13, tokens: 164_000, interventions: 5, build: '0.3.1-test', night: '2026-10-01', n: 2 },
          { suite: 'pagoda seasons', arm: 'raw', gates_passed: 2, gates_total: 4, minutes_to_pass: null, tokens: 480_000, interventions: 0, build: '0.3.0', night: '2026-09-30', n: 1 },
          { suite: 'pagoda seasons', arm: 'TandemLLM', gates_passed: 3, gates_total: 4, minutes_to_pass: null, tokens: 405_000, interventions: 9, build: '0.3.1-test', night: '2026-10-01', n: 1 },
        ],
      },
    ];
    if (this.guard) reports.push({ kind: 'speed_guard', file: 'speed-guard.json', build: ENGINE, created_at: '2026-10-01T04:00:00Z', verdict: 'pass' });
    return { contract_version: '1.0', reports, engine: ENGINE };
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
      return { status: 200, body: { contract_version: '1.0', server: { composite: 'TandemLLM', raw_model: 'qwen38-spark-engine', reasoning_effort: 'medium', global_mode: null, self_url: 'http://127.0.0.1:8000' }, store: { root: '~/.qwen38-spark-engine/conductor', readonly: false, writes: 2, documents }, journal: { path: '~/.qwen38-spark-engine/conductor/decisions.sqlite3', ring: this.rows.length, queue: 0, dropped: 0, errors: 0, retention_days: 30 }, time: iso(Date.now()) } };
    }
    if (head === 'meta') return { status: 200, body: { contract_version: '1.0', ...SEEDS.meta } };
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
    if (head === 'decisions') {
      if (parts.length === 1) {
        const limit = Number(q.get('limit') ?? 100);
        const session = q.get('session');
        return { status: 200, body: { contract_version: '1.0', decisions: [...this.rows].reverse().filter((r) => !session || r.session_key === session).slice(0, limit) } };
      }
      const row = this.rows.find((r) => r.id === parts[1]);
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
      if (parts[2] === 'export') {
        const notes = rows.flatMap((r) => (r.injected as { text: string }[]).map((n) => ({ ...n, decision: r.id, ts_ms: r.ts_ms })));
        if (q.get('format') === 'md') return { status: 200, raw: { type: 'text/markdown; charset=utf-8', data: `# Session ${parts[1]}\n\n${notes.length} interventions in ${rows.length} turns.\n\n` + notes.map((n) => '```text\n' + n.text + '\n```').join('\n\n') } };
        return { status: 200, body: { contract_version: '1.0', session: parts[1], turns: rows.length, interventions: notes.length, injections: notes } };
      }
      if (!rows.length) return { status: 404, body: { error: { type: 'not_found', message: `no session ${parts[1]}` } } };
      return { status: 200, body: { contract_version: '1.0', session: parts[1], live: null, turns: rows } };
    }
    if (head === 'evals') return { status: 200, body: this.evals() };
    if (head === 'roles' && parts[2] === 'test') {
      if (this.roleDown === parts[1]) return { status: 200, body: { contract_version: '1.0', role: parts[1], ok: false, error: `${parts[1] === 'decider' ? 'arbiter' : 'qwen'}_unreachable` } };
      return { status: 200, body: { contract_version: '1.0', role: parts[1], ok: true, ms: 21.7, question: 'Rename the variable `cnt` to `count` in utils.py and run the tests.', answer: { stage: parts[1] === 'decider' ? 'arbiter' : 'qwen', label: 'code_edit', confidence: 0.9, probs: { code_edit: 0.9 }, ms: 21.7, error: null, model: 'laya-english' } } };
    }
    if (head === 'roles' && parts.length === 1 && method === 'GET') {
      const { cur } = doc('roles', null)!;
      return { status: 200, body: { contract_version: '1.0', version: cur.version, doc: cur.doc, meta: { version: cur.version, note: cur.note, author: cur.author, created_at: cur.created_at, restored_from: cur.restored_from, loaded_at: iso(Date.now()), error: null }, health: this.health() } };
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
    if (['settings', 'router', 'roles', 'packages', 'profiles'].includes(head)) {
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
