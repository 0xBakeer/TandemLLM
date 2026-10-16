// The setup guide on the real box (VIS-84's live tier): pw-live.config.ts, PW_LIVE=1, Chromium on the
// box itself, directly on :8000 and through the reverse proxy. The steps are tagged so the operator
// runs them one at a time around the shell's own measurements:
//   @read     the tiles and every step with the box's data, from what the box runs now; the memory
//             bar against the live split; a reload keeps the step (desktop and 390 px)
//   @wren-on  Wren loads (borrowing the narrator's copy where there is one), answers, ticks a box; stays loaded
//   @wren-off Wren goes to sleep from its panel
//   @csrf     a page on another site cannot post to /v1/setup (its request in the browser and the
//             same request's answer); the page's own posts pass
//   @reset    the levels with their sizes, level 3 as a dry run, the real level 1, the refusal while a
//             request runs
//   @apply    PW_TARGET (a profile) through Finish, every step's state on the way, back on that profile
// Nothing here runs against the mock, and only @apply and level 1 change anything.
import { expect, test, type Page } from '@playwright/test';
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { noHorizontalOverflow } from './helpers';

test.skip(process.env.PW_LIVE !== '1', 'the live tier: PW_LIVE=1 on the box');

// the model the box serves (PW_MODEL; the 27B when unset)
const MODEL = process.env.PW_MODEL ?? 'qwen3.8-27b';
const SHOTS = process.env.PW_SHOTS ?? 'e2e-live-shots';
const STEPS = ['overview', 'weights', 'components', 'parallelism', 'methods', 'review'];
const isPhone = (page: Page) => (page.viewportSize()?.width ?? 1440) <= 768;
const via = () => (new URL(process.env.PW_BASE ?? 'http://127.0.0.1:8000/dashboard/').hostname.startsWith('127.') ? 'direct' : 'proxy');

async function shot(page: Page, name: string): Promise<void> {
  mkdirSync(SHOTS, { recursive: true });
  const tag = `${isPhone(page) ? 'phone' : 'desktop'}-${via()}`;
  await page.screenshot({ path: join(SHOTS, `${tag}-${name}.png`), fullPage: !isPhone(page) });
}

function note(name: string, data: unknown): void {
  mkdirSync(SHOTS, { recursive: true });
  writeFileSync(join(SHOTS, `${name}.json`), JSON.stringify(data, null, 1));
}

async function fresh(page: Page): Promise<void> {
  await page.goto('./#/setup');
  await page.evaluate(() => localStorage.removeItem('tandem.setup.v1'));
  await page.goto('./#/setup');
  await expect(page.locator(`.mtile[data-model="${MODEL}"]`)).toBeVisible();
}

async function wren(page: Page) {
  if (isPhone(page) && !(await page.locator('.wren-panel.is-open').count())) await page.locator('.wren-fab').click();
  const p = page.locator('.wren-panel');
  await expect(p).toBeVisible();
  return p;
}

async function askWren(page: Page, text: string): Promise<string> {
  const p = await wren(page);
  // a turn can take more than one round (a tool call, then the words), and the figure rests between
  // them: a turn has ended when the Send button is back, not when the figure is idle
  await expect(p.locator('.wren-send')).toBeVisible({ timeout: 90_000 });
  const users = await p.locator('.wren-msg.is-user').count();
  await p.locator('.wren-input').fill(text);
  await p.locator('.wren-input').press('Enter');
  await expect(p.locator('.wren-msg.is-user')).toHaveCount(users + 1);
  await expect(p.locator('.wren-stop')).toHaveCount(0, { timeout: 90_000 });
  await expect(p.locator('.wren-send')).toBeVisible();
  await expect(page.locator('qse-wren-figure.wren-main')).toHaveAttribute('state', 'idle', { timeout: 30_000 });
  return (await p.locator('.wren-msg.is-assistant').last().innerText()).trim();
}

test('@read the tiles and every step with the box data, from what the box runs now', async ({ page }) => {
  const current = await (await page.request.get(`/v1/setup/models/${MODEL}/current`)).json();
  const model = await (await page.request.get(`/v1/setup/models/${MODEL}`)).json();
  const est = await (await page.request.post('/v1/setup/estimate', { data: current.selection })).json();
  const posted: { url: string; status: number }[] = [];
  page.on('response', (r) => {
    if (r.request().method() === 'POST' && new URL(r.url()).pathname.startsWith('/v1/setup/')) posted.push({ url: new URL(r.url()).pathname, status: r.status() });
  });
  await fresh(page);
  const tile = page.locator(`.mtile[data-model="${MODEL}"]`);
  await expect(tile).toContainText(model.name);
  await shot(page, '0-tiles');
  await tile.click();
  await expect(page.locator('.setup-step')).toHaveAttribute('data-step', 'overview');
  await expect(page.locator('.mt-table tbody tr.mt-row').first()).toBeVisible();
  await expect(page.locator(`.mt-row[data-profile="${current.selection.profile}"]`)).toHaveCount(1);
  // a box that runs another profile than the recommended one: the page says it starts from the box
  // (shown where the model opens; any step change clears it)
  const recommended = (model.profiles as { id: string; recommended: boolean }[]).find((p) => p.recommended)?.id;
  if (current.selection.profile !== recommended) {
    await expect(page.locator('[data-setup-notice]')).toContainText('starts from what this box runs now');
  }
  // the measured table is the registry's rows, number for number (each comparable row's prose, chat, code)
  for (const r of model.measurements.rows as { profile: string; comparable: boolean; cells: { greedy_512?: Record<string, { tok_s: number }> } }[]) {
    if (!r.comparable || !r.cells.greedy_512) continue;
    const vals = await page.locator(`.mt-row[data-profile="${r.profile}"] .mt-val`).allInnerTexts();
    expect(vals, r.profile).toEqual(['prose', 'chat', 'code'].map((t) => r.cells.greedy_512![t].tok_s.toFixed(1)));
  }
  await shot(page, '1-overview');

  await page.locator('.setup-next').click();
  await expect(page.locator('.setup-step')).toHaveAttribute('data-step', 'weights');
  // the box's own profile is the one picked
  await expect(page.locator('.wcard.is-selected')).toHaveAttribute('data-profile-card', current.selection.profile);
  // every size is a card with its registry numbers: the KL and the top-1 agreement
  for (const p of (model.profiles as { id: string; quality: { kl_vs_bf16: number | null; top1_pct: number | null } }[]).filter((x) => !x.id.startsWith('community/'))) {
    const card = page.locator(`[data-profile-card="${p.id}"]`);
    await expect(card, p.id).toBeVisible();
    if (p.quality.kl_vs_bf16 != null) await expect(card.locator('.wcard-facts')).toContainText(p.quality.kl_vs_bf16.toFixed(4));
    if (p.quality.top1_pct != null) await expect(card.locator('.wmeter-v').first()).toHaveText(`${p.quality.top1_pct.toFixed(1)} %`);
  }
  // the bar is the server's estimate of that selection, and its "in use now" is the box's
  await expect(page.locator('.membar')).toHaveAttribute('data-total', est.total_gb.toFixed(2));
  const sys = await (await page.request.get('/v1/dashboard/system')).json();
  const inUse = sys.memory.split.in_use / 2 ** 30;
  const nowText = await page.locator('.membar-now').innerText();
  const shown = Number(/([\d.]+) GiB/.exec(nowText)![1]);
  note(`membar-${isPhone(page) ? 'phone' : 'desktop'}-${via()}`, { estimate_total_gb: est.total_gb, bar_total: await page.locator('.membar').getAttribute('data-total'), bar_now_text: nowText, split_in_use_gib: inUse, split: sys.memory.split.parts });
  expect(Math.abs(shown - inUse)).toBeLessThan(1.5);
  await shot(page, '2-weights');

  await page.locator('.setup-next').click();
  await expect(page.locator('.setup-step')).toHaveAttribute('data-step', 'components');
  for (const [id, on] of Object.entries(current.selection.components as Record<string, boolean>)) {
    const box = page.locator(`[data-component="${id}"] input`);
    if (!(await box.count())) continue;                     // a required tool has no checkbox
    if (on) await expect(box, id).toBeChecked();
    else await expect(box, id).not.toBeChecked();
  }
  await shot(page, '3-tools');

  await page.locator('.setup-next').click();
  await expect(page.locator('.setup-step')).toHaveAttribute('data-step', 'parallelism');
  await expect(page.locator('#pf-slots-in')).toHaveValue(String(current.selection.slots));
  await shot(page, '4-parallelism');

  await page.locator('.setup-next').click();
  await expect(page.locator('.setup-step')).toHaveAttribute('data-step', 'methods');
  await shot(page, '5-methods');

  await page.locator('.setup-next').click();
  await expect(page.locator('.setup-step')).toHaveAttribute('data-step', 'review');
  await expect(page.locator('.rv-changes')).toContainText('Nothing in serve.env changes.');
  await expect(page.locator('[data-security-warning]')).toBeVisible();
  await shot(page, '6-review');
  // the page's own posts (estimate, plan) went through, on this base
  expect(posted.length).toBeGreaterThan(0);
  expect(posted.filter((p) => p.status >= 400), JSON.stringify(posted)).toEqual([]);

  // a reload keeps the step and the selection
  await page.reload();
  await expect(page.locator('.setup-step')).toHaveAttribute('data-step', 'review');
  const saved = await page.evaluate(() => JSON.parse(localStorage.getItem('tandem.setup.v1') ?? 'null'));
  expect(saved).toMatchObject({ v: 1, model: MODEL, step: 'review', selection: { profile: current.selection.profile } });
  if (isPhone(page)) {
    for (const step of STEPS) {
      await page.goto(`./#/setup/${MODEL}/${step}`);
      await expect(page.locator('.setup-step')).toHaveAttribute('data-step', step);
      expect(await noHorizontalOverflow(page), step).toBe(true);
    }
  }
});

test('@wren-on Wren loads (borrowing the narrator where it runs the same model), answers from the docs, changes the page', async ({ page }) => {
  test.skip(isPhone(page), 'one Wren at a time: the desktop run');
  const log: Record<string, unknown> = {};
  await fresh(page);
  await page.goto(`./#/setup/${MODEL}/components`);
  const p = await wren(page);
  await expect(p).toHaveAttribute('data-phase', 'asleep');
  await shot(page, 'wren-0-asleep');
  const t0 = Date.now();
  await p.locator('.wren-wake').click();
  await expect(p).toHaveAttribute('data-phase', /loading|ready/);
  if ((await p.getAttribute('data-phase')) === 'loading') await shot(page, 'wren-1-loading');
  await expect(p).toHaveAttribute('data-phase', 'ready', { timeout: 120_000 });
  log.load_ms = Date.now() - t0;
  const status = await (await page.request.get('/v1/setup/assistant/status')).json();
  log.status = status;
  expect(status.state).toBe('loaded');
  await shot(page, 'wren-2-ready');

  log.q1 = await askWren(page, 'What does the narrator do, in one or two sentences?');
  await shot(page, 'wren-3-answer');
  // the 27B's questions, or the served model's own (PW_MODEL): its default weights, a tool it has on
  const is27 = MODEL === 'qwen3.8-27b';
  const model = await (await page.request.get(`/v1/setup/models/${MODEL}`)).json();
  const rec = (model.profiles as { id: string; recommended: boolean }[]).find((x) => x.recommended)?.id ?? '';
  log.q2 = await askWren(page, `How much memory do the weights of ${is27 ? 'exl3-4.5' : rec} take?`);
  log.sources = await p.locator('.wren-sources').last().innerText().catch(() => null);

  // a tool call: Wren turns a tool off, the page changes, the bar drops
  const [tool, label, ask] = is27 ? ['vision', 'Vision', 'Please turn off the vision tower.'] : ['response_cache', 'Response cache', 'Please turn off the response cache.'];
  const vision = page.locator(`[data-component="${tool}"] input`);
  await expect(vision).toBeChecked();
  const before = Number(await page.locator('.membar').getAttribute('data-total'));
  log.q3 = await askWren(page, ask);
  await expect(vision).not.toBeChecked({ timeout: 30_000 });
  const chip = p.locator('[data-tool-chip]').last();
  await expect(chip).toContainText(`Wren turned off ${label}`);
  log.bar_drop_gib = before - Number(await page.locator('.membar').getAttribute('data-total'));
  await shot(page, 'wren-4-tool-call');
  await chip.locator('button', { hasText: 'Undo' }).click();
  await expect(vision).toBeChecked();
  log.after = await (await page.request.get('/v1/setup/assistant/status')).json();
  note('wren-on', log);
});

test('@wren-off Wren goes to sleep from its panel', async ({ page }) => {
  test.skip(isPhone(page), 'one Wren at a time: the desktop run');
  await page.goto(`./#/setup/${MODEL}/components`);
  const p = await wren(page);
  await expect(p).toHaveAttribute('data-phase', 'ready', { timeout: 30_000 });
  await p.locator('button[aria-label="Put Wren to sleep"]').click();
  await expect(p).toHaveAttribute('data-phase', 'asleep', { timeout: 30_000 });
  await expect.poll(async () => (await (await page.request.get('/v1/setup/assistant/status')).json()).state, { timeout: 30_000 }).toBe('not_loaded');
  await shot(page, 'wren-5-asleep-again');
});

test('@csrf a page on another site cannot post to /v1/setup; the page itself can', async ({ page, context }) => {
  test.skip(isPhone(page), 'the desktop run');
  const target = new URL('/v1/setup/reset', process.env.PW_BASE ?? 'http://127.0.0.1:8000/dashboard/').toString();
  // a dry run of level 3: if a guard ever let it through, nothing would go
  const body = JSON.stringify({ level: 3, confirm: 'delete caches', dry_run: true });
  const out: Record<string, unknown> = { target };
  // 1. the browser: a page on http://evil.test sends the simple POST no preflight stops
  await context.route('http://evil.test/**', (r) => r.fulfill({ contentType: 'text/html', body: '<!doctype html><title>evil</title>' }));
  const evil = await context.newPage();
  await evil.goto('http://evil.test/');
  const ended = new Promise<{ status: number | null; failure: string | null; headers: Record<string, string> }>((done) => {
    evil.on('requestfinished', async (r) => {
      if (r.url() === target) done({ status: (await r.response())?.status() ?? null, failure: null, headers: await r.allHeaders() });
    });
    evil.on('requestfailed', async (r) => {
      if (r.url() === target) done({ status: null, failure: r.failure()?.errorText ?? 'failed', headers: await r.allHeaders() });
    });
  });
  const browserTry = await evil.evaluate(async ([url, b]) => {
    try {
      const r = await fetch(url, { method: 'POST', mode: 'no-cors', credentials: 'include', headers: { 'Content-Type': 'text/plain' }, body: b });
      return { type: r.type, status: r.status };
    } catch (e) {
      return { error: String(e) };
    }
  }, [target, body] as const);
  const end = await ended;
  out.browser = { result: browserTry, ...end };
  // the server refused it (415: not JSON), or the browser never let it leave (a private address
  // asked from a public page); either way nothing ran
  if (end.status !== null) expect(end.status).toBe(415);
  else expect(end.failure).toBeTruthy();
  // 1b. a page on the box itself on another port (PW_EVIL, e.g. a static server on :8899): the browser
  // lets that one through, so the server's own guard must refuse it
  if (process.env.PW_EVIL) {
    const local = await context.newPage();
    await local.goto(process.env.PW_EVIL);
    const answered = new Promise<{ status: number | null; failure: string | null; headers: Record<string, string> }>((done) => {
      local.on('requestfinished', async (r) => {
        if (r.url() === target) done({ status: (await r.response())?.status() ?? null, failure: null, headers: await r.allHeaders() });
      });
      local.on('requestfailed', async (r) => {
        if (r.url() === target) done({ status: null, failure: r.failure()?.errorText ?? 'failed', headers: await r.allHeaders() });
      });
    });
    const tried = await local.evaluate(async ([url, b]) => {
      try {
        const r = await fetch(url, { method: 'POST', mode: 'no-cors', credentials: 'include', headers: { 'Content-Type': 'text/plain' }, body: b });
        return { type: r.type };
      } catch (e) {
        return { error: String(e) };
      }
    }, [target, body] as const);
    const r = await answered;
    out.local_page = { origin: process.env.PW_EVIL, result: tried, ...r };
    if (r.status !== null) expect(r.status).toBe(415);
    else expect(r.failure).toBeTruthy();
    // and a JSON post from it, which needs a preflight the server never answers
    const json = await local.evaluate(async ([url, b]) => {
      try {
        const r = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: b });
        return { status: r.status };
      } catch (e) {
        return { error: String(e) };
      }
    }, [target, body] as const);
    out.local_page_json = json;
    expect(json).toHaveProperty('error');
    await local.close();
  }
  // 2. the same request's answer, and the other two shapes, from outside a browser
  const send = async (headers: Record<string, string>, data: string) => {
    const r = await page.request.post('/v1/setup/reset', { headers, data });
    return { status: r.status(), body: await r.text() };
  };
  out.text_plain = await send({ 'Content-Type': 'text/plain', Origin: 'http://evil.test' }, body);
  out.cross_site = await send({ 'Content-Type': 'application/json', 'Sec-Fetch-Site': 'cross-site' }, body);
  out.cross_origin = await send({ 'Content-Type': 'application/json', Origin: 'http://evil.test' }, body);
  expect((out.text_plain as { status: number }).status).toBe(415);
  expect((out.cross_site as { status: number }).status).toBe(403);
  expect((out.cross_origin as { status: number }).status).toBe(403);
  // 3. the page's own post passes: the same dry run from the dashboard's origin
  await page.goto('./#/system');
  const own = await page.evaluate(async (b) => {
    const r = await fetch('/v1/setup/reset', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: b });
    return { status: r.status, body: await r.json() };
  }, body);
  out.own = { status: own.status, dry_run: own.body.dry_run, items: (own.body.items ?? own.body.removed ?? []).length };
  expect(own.status).toBe(200);
  note(`csrf-${via()}`, out);
  await evil.close();
});

test('@reset the levels with their sizes, level 3 dry, the real level 1, refused while a request runs', async ({ page }) => {
  test.skip(isPhone(page), 'the desktop run');
  const out: Record<string, unknown> = {};
  const levels = await (await page.request.get('/v1/setup/reset')).json();
  out.levels = levels;
  await page.goto('./#/system');
  const card = page.locator('qse-reset .reset-card[data-level="3"]');
  await card.scrollIntoViewIfNeeded();
  await card.locator('.reset-review').click();
  const d = page.locator('dialog.reset-dialog[open]');
  await expect(d).toBeVisible();
  out.level3_items = await d.locator('.reset-items li').allInnerTexts();
  out.level3_total = await d.locator('.reset-total').innerText();
  await expect(d.locator('.reset-go')).toBeDisabled();
  await shot(page, 'reset-0-level3-review');
  await page.keyboard.press('Escape');
  // dry runs from the page's own origin: level 3 and level 1 (the browser's, so the server says 400)
  out.dry3 = await page.evaluate(async () => {
    const r = await fetch('/v1/setup/reset', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ level: 3, confirm: 'delete caches', dry_run: true }) });
    return { status: r.status, body: await r.json() };
  });
  expect((out.dry3 as { status: number }).status).toBe(200);
  out.dry1 = await page.evaluate(async () => {
    const r = await fetch('/v1/setup/reset', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ level: 1, confirm: true, dry_run: true }) });
    return { status: r.status, body: await r.json() };
  });
  // the real level 1 from the setup page: the browser forgets, nothing is posted
  await page.goto(`./#/setup/${MODEL}/weights`);
  await expect(page.locator('.wcard.is-selected')).toBeVisible();
  expect(await page.evaluate(() => localStorage.getItem('tandem.setup.v1'))).not.toBeNull();
  let posted = 0;
  page.on('request', (r) => {
    if (r.url().includes('/v1/setup/reset') && r.method() === 'POST') posted++;
  });
  await page.locator('.setup-foot button', { hasText: 'Reset' }).click();
  const one = page.locator('qse-reset .reset-card[data-level="1"]');
  await one.locator('.reset-review').click();
  const d1 = page.locator('dialog.reset-dialog[open]');
  await shot(page, 'reset-1-level1-confirm');
  await d1.locator('.reset-go').click();
  await expect(d1.locator('[data-reset-result="local"]')).toBeVisible();
  expect(await page.evaluate(() => localStorage.getItem('tandem.setup.v1'))).toBeNull();
  expect(posted).toBe(0);
  await page.keyboard.press('Escape');
  // a request in flight: the server refuses every level and the System view says why
  const started = page.evaluate(() => {
    void fetch('/v1/chat/completions', { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Tandem': 'raw' }, body: JSON.stringify({ model: 'TandemLLM', messages: [{ role: 'user', content: 'Count from 1 to 400, one number a line.' }], max_tokens: 900, temperature: 0, chat_template_kwargs: { enable_thinking: false } }) });
  });
  await started;
  await expect.poll(async () => (await (await page.request.get('/v1/setup/device')).json()).engine.running, { timeout: 30_000 }).toBeGreaterThan(0);
  // a real call (a dry run only lists, busy or not) with a wrong word: the route asks busy before
  // the word, so it is 409 now, and were the box idle it would stop at the word with nothing gone
  out.busy = await page.evaluate(async () => {
    const r = await fetch('/v1/setup/reset', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ level: 3, confirm: 'not the word' }) });
    return { status: r.status, body: await r.json() };
  });
  expect((out.busy as { status: number }).status).toBe(409);
  await page.goto('./#/system');
  await expect(page.locator('[data-reset-busy]')).toBeVisible({ timeout: 20_000 });
  out.busy_text = await page.locator('[data-reset-busy]').innerText();
  await shot(page, 'reset-2-busy');
  note('reset', out);
});

test('@apply PW_TARGET through Finish: every step on the way, then that profile serves', async ({ page }) => {
  test.skip(isPhone(page), 'one install at a time: the desktop run');
  const target = process.env.PW_TARGET;
  test.skip(!target, 'PW_TARGET=<profile>');
  const name = `apply-${target}-${via()}`;
  await fresh(page);
  await page.goto(`./#/setup/${MODEL}/weights`);
  await page.locator(`[data-profile-card="${target}"]`).click();
  await expect(page.locator('.wcard.is-selected')).toHaveAttribute('data-profile-card', target!);
  await page.goto(`./#/setup/${MODEL}/review`);
  await expect(page.locator('.rv-cmds')).toContainText(`ops/use-profile.sh ${target}`);
  const changes = await page.locator('.rv-changes').innerText();
  await shot(page, `${name}-0-review`);
  await expect(page.locator('.setup-finish')).toBeEnabled();
  await page.locator('.setup-finish').click();
  await shot(page, `${name}-1-confirm`);
  await page.locator('.setup-confirm-go').click();
  await expect(page).toHaveURL(new RegExp(`#/setup/${MODEL}/install$`));
  // every step's state as the page shows it, once a second, until the job ends
  const t0 = Date.now();
  const seen: { t: number; steps: Record<string, string>; restarting: boolean; job: string | null }[] = [];
  const shotsTaken = new Set<string>();
  let last = '';
  for (;;) {
    const steps: Record<string, string> = await page.locator('.inst-step').evaluateAll((els) => Object.fromEntries(els.map((e) => [e.getAttribute('data-step') ?? '', (e.className.match(/is-([a-z_]+)/g) ?? []).join(' ')])));
    const restarting = (await page.locator('[data-restarting]').count()) > 0;
    const job = await page.locator('[data-job-state]').first().getAttribute('data-job-state').catch(() => null);
    const key = JSON.stringify([steps, restarting, job]);
    if (key !== last) {
      seen.push({ t: Math.round((Date.now() - t0) / 100) / 10, steps, restarting, job });
      last = key;
      for (const [s, cls] of Object.entries(steps)) {
        const k = `${s}-${cls}`;
        if (/running|waiting/.test(cls) && !shotsTaken.has(k)) {
          shotsTaken.add(k);
          await shot(page, `${name}-2-${s}-${cls.replace(/\s+/g, '_')}`);
        }
      }
      if (restarting && !shotsTaken.has('restarting')) {
        shotsTaken.add('restarting');
        await shot(page, `${name}-3-restarting`);
      }
    }
    if (job && /^(done|failed|rolled_back)$/.test(job)) break;
    if (Date.now() - t0 > 18 * 60_000) break;
    await page.waitForTimeout(1000);
  }
  await shot(page, `${name}-4-end`);
  note(name, { target, changes, seen, ms: Date.now() - t0 });
  await expect(page.locator('[data-job-state="done"]')).toBeVisible();
  const health = await (await page.request.get('/health')).json();
  expect(health.weights.name).toBe(target);
});

test('@setupmode a box in setup mode: /health says so, the page opens on the guide, the other views are off', async ({ page }) => {
  const health = await (await page.request.get('/health')).json();
  expect(health.mode).toBe('setup');
  // the model list is empty (a client that lists models keeps working); a chat is 503 setup_mode
  const models = await (await page.request.get('/v1/models')).json();
  expect(models.data).toEqual([]);
  const chat = await page.request.post('/v1/chat/completions', { data: { model: 'TandemLLM', messages: [{ role: 'user', content: 'hi' }], max_tokens: 4 } });
  expect(chat.status()).toBe(503);
  const chatBody = await chat.json();
  expect(chatBody.error.code).toBe('setup_mode');
  note(`setupmode-${isPhone(page) ? 'phone' : 'desktop'}`, { health, models, chat: { status: chat.status(), body: chatBody } });
  await page.goto('./#/usage');
  await expect(page).toHaveURL(/#\/setup$/);
  await expect(page.locator(`.mtile[data-model="${MODEL}"]`)).toBeVisible();
  await expect(page.locator('.topbar .pill')).toHaveText('setup');
  await expect(page.locator('[data-setup-mode]')).toBeVisible();
  if (!isPhone(page)) await expect(page.locator('.rail a.rail-tab.is-off')).toHaveCount(8); // VIS-161: Chat included
  await shot(page, 'setupmode-0-tiles');
  await page.locator(`.mtile[data-model="${MODEL}"]`).click();
  await expect(page.locator('.setup-step')).toHaveAttribute('data-step', 'overview');
  await page.goto(`./#/setup/${MODEL}/weights`);
  await expect(page.locator('.wcard').first()).toBeVisible();
  await shot(page, 'setupmode-1-weights');
  if (isPhone(page)) await page.locator('.wren-fab').click();
  await expect(page.locator('.wren-panel')).toHaveAttribute('data-phase', /ready|loading|asleep/);
  await shot(page, 'setupmode-2-wren');
});
