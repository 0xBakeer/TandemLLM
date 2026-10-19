// QA VIS-79/81/83: the setup guide on the backend's real behaviour, against the mock shaped like
// server/setup_api.py: the sign-in after a 401, setup mode without error spam, the plan's error codes
// with the step that fixes them, apply's refusals, Wren's page state, the reset items' actions and
// level 2 as a job, and the header and memory bar that nothing shows through. @mock
import { expect, test, type Page } from '@playwright/test';
import { IS_MOCK } from './helpers';
import { open, setupMock } from './setup-helpers';

test.skip(!IS_MOCK, 'the setup guide specs drive the mock (POST /__mock/setup)');

const MODEL = 'qwen3.8-27b';
const isPhone = (page: Page) => (page.viewportSize()?.width ?? 1440) <= 1100;

test.beforeEach(async ({ request }) => {
  await setupMock(request);
});

test('a box with a password: the sign-in replaces the steps, a wrong one says so, the right one opens the guide', async ({ page, request }) => {
  await setupMock(request, { unauthorized: true });
  await open(page, '#/setup');
  const card = page.locator('[data-setup-signin]');
  await expect(card).toBeVisible();
  await expect(card).toContainText('secrets.env');
  await expect(page.locator('.mtile')).toHaveCount(0);
  // the shell's own 401 banner is not shown over the page's sign-in
  await expect(page.locator('[data-refused]')).toHaveCount(0);
  await card.locator('.signin-input').fill('wrong');
  await card.locator('button[type="submit"]').click();
  await expect(card.locator('[data-signin-error]')).toHaveText('That password is not right.');
  await card.locator('.signin-input').fill('mock');
  await card.locator('button[type="submit"]').click();
  await expect(page.locator(`.mtile[data-model="${MODEL}"]`)).toBeVisible();
  await expect(page.locator('[data-setup-signin]')).toHaveCount(0);
  // the cookie opens the setup routes from here on
  await page.goto(`./#/setup/${MODEL}/weights`);
  await expect(page.locator('.wcard.is-selected')).toBeVisible();
});

test('a box with a password and the dashboard login off: the sign-in says how to get in', async ({ page, request }) => {
  await setupMock(request, { unauthorized: true, loginOff: true });
  await open(page, `#/setup/${MODEL}/weights`);
  const card = page.locator('[data-setup-signin]');
  await card.locator('.signin-input').fill('mock');
  await card.locator('button[type="submit"]').click();
  await expect(card.locator('[data-signin-error]')).toContainText('QSE_DASHBOARD_LOGIN=on');
});

test('setup mode: no /v1/dashboard request, the pill says setup, no offline banner', async ({ page, request }) => {
  await setupMock(request, { setupMode: true });
  const dash: string[] = [];
  page.on('request', (r) => {
    const u = new URL(r.url());
    if (u.pathname.startsWith('/v1/dashboard/') || u.pathname.startsWith('/v1/conductor/')) dash.push(u.pathname);
  });
  await page.goto('./#/usage');
  await expect(page).toHaveURL(/#\/setup$/);
  await expect(page.locator(`.mtile[data-model="${MODEL}"]`)).toBeVisible();
  await expect(page.locator('.topbar .pill')).toHaveText('setup');
  await page.waitForTimeout(1500);
  expect(dash).toEqual([]);
  await expect(page.locator('.banner-offline')).toHaveCount(0);
});

test('setup mode ends (the box serves a model): the other views open again', async ({ page, request }) => {
  await setupMock(request, { setupMode: true });
  await page.goto('./#/setup');
  await expect(page.locator('.rail .rail-tab.is-off')).toHaveCount(8);
  await setupMock(request, { setupMode: false });
  await expect(page.locator('.rail .rail-tab.is-off')).toHaveCount(0, { timeout: 15_000 });
  await expect(page.locator('[data-setup-mode]')).toHaveCount(0);
  await expect(page.locator('.topbar .pill')).not.toHaveText('setup');
});

test("Review: the plan's error names its code and links to the step that fixes it; Finish stays off", async ({ page, request }) => {
  await setupMock(request, { secretsExist: true });
  await open(page, `#/setup/${MODEL}/review`);
  await page.locator('[data-security-warning] .sec-opt', { hasText: 'Generate a password on the box' }).click();
  const err = page.locator('.rv-notes li[data-error-code="secrets_exist"]');
  await expect(err).toContainText('secrets.env already exists');
  await expect(err.locator('a')).toHaveAttribute('href', `#/setup/${MODEL}/methods`);
  await expect(page.locator('.setup-finish')).toBeDisabled();
});

test('generate a password with the dashboard login off: the page says the sign-in needs it, one click turns it on', async ({ page }) => {
  await open(page, `#/setup/${MODEL}/review`);
  const card = page.locator('[data-security-warning]');
  await card.locator('.sec-opt', { hasText: 'Generate a password on the box' }).click();
  await expect(card.locator('[data-login-off]')).toContainText('Dashboard login on');
  await card.locator('[data-login-off] button').click();
  await expect(card.locator('[data-login-off]')).toHaveCount(0);
  const saved = await page.evaluate(() => JSON.parse(localStorage.getItem('tandem.setup.v1')!).selection.advanced.dashboard_login);
  expect(saved).toBe('on');
});

test('apply refused while requests run: the count, and nothing starts', async ({ page, request }) => {
  await open(page, `#/setup/${MODEL}/review`);
  await expect(page.locator('.setup-finish')).toBeEnabled();
  await setupMock(request, { busy: 1 });
  await page.locator('.setup-finish').click();
  await page.locator('.setup-confirm-go').click();
  await expect(page.locator('dialog.setup-confirm [role="alert"]')).toHaveText('1 request is running. Install when they finish.');
  await expect(page).toHaveURL(new RegExp(`#/setup/${MODEL}/review$`));
});

test("each Wren turn carries the step's visible fields and the estimate", async ({ page, request }) => {
  await setupMock(request, { assistant: 'loaded' });
  await open(page, `#/setup/${MODEL}/parallelism`);
  if (isPhone(page)) await page.locator('.wren-fab').click();
  const p = page.locator('.wren-panel');
  const sent = page.waitForRequest((r) => r.url().endsWith('/v1/setup/assistant/chat'));
  await p.locator('.wren-input').fill('how many parallel chats do I need?');
  await p.locator('.wren-input').press('Enter');
  const body = (await sent).postDataJSON();
  expect(body.page.step).toBe('parallelism');
  expect(body.page.visible.fields.map((f: { field: string }) => f.field)).toEqual(expect.arrayContaining(['slots', 'window']));
  expect(body.page.estimate).toEqual(expect.objectContaining({ total_gb: expect.any(Number), budget_gb: expect.any(Number), fits: true, fits_worst: expect.any(Boolean), kv_worst_gb: expect.any(Number) }));
});

async function review(page: Page, level: number) {
  const card = page.locator(`qse-reset .reset-card[data-level="${level}"]`);
  await card.scrollIntoViewIfNeeded();
  await card.locator('.reset-review').click();
  const d = page.locator('dialog.reset-dialog[open]');
  await expect(d).toBeVisible();
  return d;
}

test('level 3: what the engine clears itself, what is in memory, and what stays while in use', async ({ page, request }) => {
  await setupMock(request, { shadowOn: true });
  await open(page, '#/system');
  const d = await review(page, 3);
  await expect(d.locator('[data-action="engine"]').first()).toContainText('the engine clears it itself');
  await expect(d.locator('[data-item="response_cache"]')).toContainText('in memory');
  const skip = d.locator('[data-action="skip"]');
  await expect(skip).toHaveClass(/is-missing/);
  await expect(skip).toContainText('stays');
  await d.locator('.reset-word').fill('delete caches');
  await d.locator('.reset-go').click();
  const res = d.locator('[data-reset-result="partial"]');
  await expect(res.locator('.reset-failed li')).toHaveCount(1);
  await expect(res).toContainText('shadow reader runs');
});

test('level 2 with no configuration written yet still runs: it restarts on the recommended weights, as a job', async ({ page, request }) => {
  await setupMock(request, { resetEmpty: true });
  await open(page, '#/system');
  const d = await review(page, 2);
  await expect(d.locator('[data-reset-empty]')).toContainText('still restarts on the recommended weights');
  await d.locator('.reset-go').click();
  await expect(d.locator('.inst-title')).toHaveText(/Going back to the shipped configuration|Back on the shipped configuration/);
  await expect(d.locator('[data-job-state="done"]')).toBeVisible({ timeout: 20_000 });
  await expect(d.locator('.inst-title')).toHaveText('Back on the shipped configuration');
});

test('desktop: nothing shows through the header or the memory bar while the steps scroll', async ({ page }) => {
  test.skip(isPhone(page), 'the docked layout');
  await open(page, `#/setup/${MODEL}/components`);
  const top = page.locator('.topbar');
  const bg = await top.evaluate((el) => {
    const cs = getComputedStyle(el);
    return { bg: cs.backgroundColor, blur: cs.backdropFilter };
  });
  // a solid colour: rgb(...) or rgba(..., 1), and no glass
  expect(bg.bg).toMatch(/^rgb\(|, 1\)$/);
  expect(['none', '']).toContain(bg.blur);
  // scroll "Use the recommended values" under the header: the header and the bar cover it
  const tools = page.locator('.step-tools .btn').first();
  const header = (await top.boundingBox())!;
  const y0 = (await tools.boundingBox())!.y;
  await page.evaluate((dy) => window.scrollBy(0, dy), y0 - header.y - 10);
  const box = (await tools.boundingBox())!;
  expect(box.y).toBeLessThan(header.y + header.height);
  const hit = await page.evaluate(([x, y]) => {
    const el = document.elementFromPoint(x, y);
    return el?.closest('.topbar, .setup-sticky') ? 'covered' : (el?.className ?? 'nothing');
  }, [box.x + box.width / 2, Math.max(header.y + 2, box.y + box.height / 2)]);
  expect(hit).toBe('covered');
});
