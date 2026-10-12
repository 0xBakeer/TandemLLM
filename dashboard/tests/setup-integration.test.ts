// The setup guide on the backend's real behaviour (VIS-78/80/82 on SRV-323..329, OPS-93): the plan's
// error codes lead to the step that fixes them, a 401 from a setup route asks for the sign-in (and a
// wrong password does not), the sign-in's words by status, Wren's page state carries the visible
// fields and the whole estimate, and the mock answers the way server/setup_api.py does (the reset
// levels' items and words, level 2 as a job, the refusals, the session on a box with a password).
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { IncomingMessage, ServerResponse } from 'node:http';
import model from '../mock/setup/qwen3.8-27b.json';
import { MockSetup } from '../mock/setup';
import type { SetupModel, SetupResetLevel, SetupResetLevels, SetupResetResult } from '../src/api/types';
import { setupApi, streamChat } from '../src/setup/api';
import { buildPageState, planFixStep, signInError } from '../src/setup/checks';
import { estimate } from '../src/setup/estimate';
import { defaults } from '../src/setup/state';
import { S } from '../src/setup/strings';

const M = model as unknown as SetupModel;

afterEach(() => vi.unstubAllGlobals());

describe('the plan error codes', () => {
  it('each code the server names leads to the step that fixes it; the box-side ones to none', () => {
    expect(planFixStep('over_budget')).toBe('parallelism');
    expect(planFixStep('missing_token')).toBe('weights');
    expect(planFixStep('disk_full')).toBe('weights');
    expect(planFixStep('missing_assets')).toBe('components');
    expect(planFixStep('secrets_exist')).toBe('methods');
    expect(planFixStep('port_in_use')).toBe('methods');
    for (const code of ['no_overlay_hook', 'repo_mismatch', 'something_new', undefined]) expect(planFixStep(code), String(code)).toBeNull();
  });
});

describe('the sign-in', () => {
  it('says why by status: a wrong password, too many tries, the login off, no answer', () => {
    expect(signInError(401, 'wrong token')).toBe(S.signIn.refused);
    expect(signInError(429, '')).toBe(S.signIn.tooMany);
    expect(signInError(404, 'the dashboard login is off')).toContain('QSE_DASHBOARD_LOGIN=on');
    expect(signInError(0, '')).toBe(S.signIn.offline);
    expect(signInError(500, 'boom')).toBe(`${S.signIn.failed} boom`);
  });

  function stub(status: number, body: unknown = { error: { type: 'unauthorized', message: 'no' } }) {
    const events: string[] = [];
    const w = new EventTarget();
    w.addEventListener('qse-setup-unauthorized', () => events.push('setup-401'));
    vi.stubGlobal('window', w);
    vi.stubGlobal('fetch', async () => new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } }));
    return events;
  }

  it('a 401 from a setup read or write asks the page for the sign-in', async () => {
    const events = stub(401);
    await expect(setupApi.device()).rejects.toMatchObject({ status: 401 });
    await expect(setupApi.reset({ level: 3, confirm: 'delete caches' })).rejects.toMatchObject({ status: 401 });
    expect(events).toEqual(['setup-401', 'setup-401']);
  });

  it("Wren's chat stream's 401 does too", async () => {
    const events = stub(401);
    await expect(streamChat({ messages: [], page: { model: null, step: 'models', step_index: 0, selection: {} } }, () => {}, new AbortController().signal)).rejects.toMatchObject({ status: 401 });
    expect(events).toEqual(['setup-401']);
  });

  it("a wrong password at the sign-in is the sign-in's own error, not another request for it", async () => {
    const events = stub(401, { error: { type: 'unauthorized', message: 'wrong token' } });
    await expect(setupApi.signIn('nope')).rejects.toMatchObject({ status: 401 });
    expect(events).toEqual([]);
  });

  it('the password goes once, as the session route takes it, and nothing else is sent', async () => {
    const calls: { url: string; init: RequestInit }[] = [];
    vi.stubGlobal('window', new EventTarget());
    vi.stubGlobal('fetch', async (url: string, init: RequestInit) => {
      calls.push({ url, init });
      return new Response(null, { status: 204 });
    });
    await setupApi.signIn('s3cret-password-of-the-box');
    expect(calls).toHaveLength(1);
    expect(calls[0].url).toBe('/v1/dashboard/session');
    expect(calls[0].init.method).toBe('POST');
    expect(JSON.parse(String(calls[0].init.body))).toEqual({ token: 's3cret-password-of-the-box' });
  });
});

describe("Wren's page state", () => {
  it('carries the visible fields and the estimate the bar shows: budget, worst case, free memory', () => {
    const sel = defaults(M);
    const est = { ...estimate(M, sel), available_gb: 41.2 };
    const page = buildPageState(M, 'parallelism', sel, est, { errors: ['Over by 2.0 GiB.'] });
    expect((page.visible!.fields as { field: string }[]).map((f) => f.field)).toEqual(expect.arrayContaining(['slots', 'window']));
    expect(page.visible!.errors).toEqual(['Over by 2.0 GiB.']);
    expect(page.estimate).toMatchObject({ total_gb: est.total_gb, budget_gb: est.budget_gb, fits: est.fits, fits_worst: est.fits_worst, kv_worst_gb: est.kv_worst_gb, available_gb: 41.2, context_tokens_fit: est.context_tokens_fit });
    expect(JSON.stringify(page).length).toBeLessThan(6 * 1024);
  });
});

// ---- the mock, held to server/setup_api.py's answers ----------------------------------------

interface Answer {
  status: number;
  headers: Record<string, string>;
  body: any; // eslint-disable-line @typescript-eslint/no-explicit-any
}

async function call(ms: MockSetup, method: string, path: string, body?: unknown, cookie?: string): Promise<Answer> {
  const out: Answer = { status: 0, headers: {}, body: null };
  let text = '';
  const req = { method, headers: cookie ? { cookie } : {}, on: () => req, socket: { destroy: () => {} } } as unknown as IncomingMessage;
  await new Promise<void>((done) => {
    const res = {
      writeHead: (code: number, h: Record<string, string> = {}) => {
        out.status = code;
        out.headers = h;
        return res;
      },
      write: (chunk: string) => {
        text += chunk;
        return true;
      },
      end: (chunk?: string) => {
        if (chunk) text += chunk;
        done();
      },
    } as unknown as ServerResponse;
    const readBody = async () => (body === undefined ? '' : JSON.stringify(body));
    if (path === '/v1/dashboard/session') void ms.session(req, res, readBody);
    else void ms.handle(req, res, path, new URLSearchParams(), readBody);
  });
  out.body = text ? JSON.parse(text) : null;
  return out;
}

describe('the second model (OPS-97)', () => {
  it('the mock lists Flash-Next as a second tile and answers its entry, current and estimate', async () => {
    const ms = new MockSetup();
    const tiles = (await call(ms, 'GET', '/v1/setup/models')).body.models as { id: string; recommended: boolean; best_speed: unknown }[];
    expect(tiles.map((t) => [t.id, t.recommended])).toEqual([['qwen3.8-27b', true], ['qwen3.8-flash-next', false]]);
    expect(tiles[1].best_speed).toBeNull();                       // no measured row yet
    const fn = (await call(ms, 'GET', '/v1/setup/models/qwen3.8-flash-next')).body as SetupModel;
    expect(fn.profiles.map((p) => p.max_window)).toEqual([262_144, 262_144, 212_992, 65_536, null]);
    expect((await call(ms, 'GET', '/v1/setup/models/qwen3.8-flash-next/current')).body.selection).toEqual({ model: 'qwen3.8-flash-next' });
    const est = (await call(ms, 'POST', '/v1/setup/estimate', { selection: { ...defaults(fn), model: fn.id } })).body;
    expect(est.fits).toBe(true);
    expect(est.parts.some((p: { label: string }) => p.label.startsWith('n-gram row cache'))).toBe(true); // 3.05 reads its table cold (SPD-99)
    const over = await call(ms, 'POST', '/v1/setup/estimate', { selection: { ...defaults(fn), model: fn.id, profile: 'flashnext-6.05' } });
    expect(over.body.fits).toBe(false);
    expect(over.body.notes[0]).toContain('does not fit');
  });
});
describe('the mock answers as the server does', () => {
  it('the reset levels: the words, level 2 the configuration files, level 3 with what the engine clears itself', async () => {
    const ms = new MockSetup();
    const r = await call(ms, 'GET', '/v1/setup/reset');
    const lv = (r.body as SetupResetLevels).levels;
    expect(lv.map((l) => [l.level, l.confirm_word])).toEqual([[2, null], [3, 'delete caches'], [4, 'delete weights'], [5, 'delete history']]);
    const two = lv.find((l) => l.level === 2)!;
    expect(two.items.map((i) => i.path)).toEqual(['~/tandem-live/run/setup.env', '~/tandem-live/run/setup.env.bak-20261010T081200Z']);
    expect(two.what).toContain('The recommended profile is exl3-4.5.');
    const three = lv.find((l) => l.level === 3)!;
    // the running disk tier and the lookup store are emptied by the engine itself (SRV-329)
    expect(three.items.map((i) => [i.kind, i.action])).toEqual([['path', 'engine'], ['path', 'remove'], ['path', 'engine'], ['memory', 'engine'], ['memory', 'engine']]);
    expect(three.restart).toBe(false);
    for (const l of lv) for (const i of l.items) expect(Object.keys(i).sort()).toEqual(['action', 'at_least', 'bytes', 'exists', 'files', 'kind', 'label', 'note', 'path', 'refused', 'restart']);
  });

  it('level 3 beside a running shadow reader: its sessions stay and come back in `failed`', async () => {
    const ms = new MockSetup();
    ms.control({ shadowOn: true });
    const three = (await call(ms, 'GET', '/v1/setup/reset')).body.levels.find((l: SetupResetLevel) => l.level === 3) as SetupResetLevel;
    expect(three.items.find((i) => i.action === 'skip')?.note).toBe('kept while the shadow reader runs');
    const r = await call(ms, 'POST', '/v1/setup/reset', { level: 3, confirm: 'delete caches' });
    const res = r.body as SetupResetResult;
    expect(r.status).toBe(200);
    expect(res.ok).toBe(false);
    expect(res.failed.map((f) => f.path)).toEqual(['~/.qwen38-spark-engine/cache/shadow']);
    expect(res.removed).toHaveLength(5);
  });

  it('the refusals in the server order: level 1, a bad level, no word, busy, a job running', async () => {
    const ms = new MockSetup();
    expect((await call(ms, 'POST', '/v1/setup/reset', { level: 1, confirm: true })).body.error.type).toBe('browser_only');
    expect((await call(ms, 'POST', '/v1/setup/reset', { level: 7, confirm: true })).status).toBe(422);
    const noWord = await call(ms, 'POST', '/v1/setup/reset', { level: 3, confirm: 'delete cache' });
    expect([noWord.status, noWord.body.error.code, noWord.body.level.level]).toEqual([400, 'confirm_required', 3]);
    ms.control({ busy: 2 });
    const busy = await call(ms, 'POST', '/v1/setup/reset', { level: 3, confirm: 'delete caches' });
    expect([busy.status, busy.body.error.type, busy.body.error.running]).toEqual([409, 'busy', 2]);
    ms.control({ busy: 0 });
    const job = await call(ms, 'POST', '/v1/setup/reset', { level: 2, confirm: true });
    expect(job.status).toBe(202);
    expect(job.body.job.kind).toBe('reset');
    expect(job.body.job.profile).toBe('exl3-4.5');
    expect(job.body.job_id).toMatch(/^[0-9a-f]{16}$/);
    const again = await call(ms, 'POST', '/v1/setup/reset', { level: 3, confirm: 'delete caches' });
    expect([again.status, again.body.error.type]).toEqual([409, 'job_running']);
  });

  it("apply's refusals carry the plan, with a code per error", async () => {
    const ms = new MockSetup();
    ms.control({ secretsExist: true });
    const sel = { ...defaults(M), advanced: { ...defaults(M).advanced, admin_password: 'generate' } };
    const r = await call(ms, 'POST', '/v1/setup/apply', { selection: sel });
    expect(r.status).toBe(409);
    expect(r.body.error.code).toBe('secrets_exist');
    expect(r.body.plan.error_codes).toEqual(['secrets_exist']);
    expect(r.body.plan.errors).toHaveLength(1);
    expect(r.body.plan.commands).toContain('bash ops/make-secrets.sh --admin');
  });

  it('a box with a password: 401 until the sign-in, the cookie then opens the setup routes; with the login off the sign-in is a 404', async () => {
    const ms = new MockSetup();
    ms.control({ unauthorized: true });
    expect((await call(ms, 'GET', '/v1/setup/device')).status).toBe(401);
    expect((await call(ms, 'POST', '/v1/dashboard/session', { token: 'wrong' })).status).toBe(401);
    const ok = await call(ms, 'POST', '/v1/dashboard/session', { token: 'mock' });
    expect(ok.status).toBe(204);
    const cookie = ok.headers['Set-Cookie'].split(';')[0];
    expect((await call(ms, 'GET', '/v1/setup/device', undefined, cookie)).status).toBe(200);
    ms.control({ loginOff: true });
    const off = await call(ms, 'POST', '/v1/dashboard/session', { token: 'mock' });
    expect([off.status, off.body.error.message]).toEqual([404, 'the dashboard login is off']);
  });

  it('Wren refused for memory is a 200 with state refused (the page shows "cannot load")', async () => {
    const ms = new MockSetup();
    ms.control({ refuse: true });
    const r = await call(ms, 'POST', '/v1/setup/assistant/load', {});
    expect(r.status).toBe(200);
    expect(r.body.state).toBe('refused');
    expect(r.body.reason).toMatch(/not enough memory/);
  });
});
