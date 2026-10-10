// QA VIS-83 (VIS-82): the reset levels in the System view and on the setup page. @mock
import { expect, test, type Page } from '@playwright/test';
import { IS_MOCK, noHorizontalOverflow } from './helpers';
import { open, setupMock } from './setup-helpers';

test.skip(!IS_MOCK, 'the reset specs drive the mock (POST /__mock/setup)');

const isPhone = (page: Page) => (page.viewportSize()?.width ?? 1440) <= 768;

async function review(page: Page, level: number) {
  const card = page.locator(`qse-reset .reset-card[data-level="${level}"]`);
  await card.scrollIntoViewIfNeeded();
  await expect(card.locator('.reset-review')).toBeEnabled();
  await card.locator('.reset-review').click();
  const dialog = page.locator('dialog.reset-dialog[open]');
  await expect(dialog).toBeVisible();
  return dialog;
}

test.beforeEach(async ({ request }) => {
  await setupMock(request);
});

test('Review lists each cache with its size, the total, the restart note and the word; nothing goes yet', async ({ page }) => {
  let posted = 0;
  page.on('request', (r) => {
    if (r.url().endsWith('/v1/setup/reset') && r.method() === 'POST') posted++;
  });
  await open(page, '#/system');
  const d = await review(page, 3);
  // two disk tiers (the running one the engine clears), the lookup store, two caches in memory
  await expect(d.locator('.reset-items li')).toHaveCount(5);
  await expect(d.locator('.reset-items')).toContainText('Disk cache');
  await expect(d.locator('.reset-items')).toContainText('11.2 GiB');
  await expect(d.locator('.reset-total')).toContainText('Total 14.7 GiB');
  // the lookup store: the running engine keeps its copy until a restart (server/setup/reset.py)
  await expect(d.locator('.reset-total')).toContainText('Restarts the engine');
  await expect(d.locator('.reset-word-show')).toHaveText('delete caches');
  expect(posted).toBe(0);
});

test('the confirmation needs the exact word', async ({ page }) => {
  await open(page, '#/system');
  const d = await review(page, 3);
  const go = d.locator('.reset-go');
  for (const typed of ['', 'delete cache', 'DELETE CACHES', 'delete caches ']) {
    await d.locator('.reset-word').fill(typed);
    await expect(go, JSON.stringify(typed)).toBeDisabled();
  }
  await d.locator('.reset-word').fill('delete caches');
  await expect(go).toBeEnabled();
  const attrs = await d.locator('.reset-word').evaluate((el) => [el.getAttribute('autocapitalize'), el.getAttribute('autocorrect'), el.getAttribute('spellcheck')]);
  expect(attrs).toEqual(['off', 'off', 'false']);
});

test('level 1 is local: tandem.setup.v1 goes, no request, the guide starts at the tiles', async ({ page }) => {
  await open(page, '#/setup/qwen3.8-27b/weights');
  await page.locator('[data-profile-card="exl3-4.0"]').click();
  expect(await page.evaluate(() => localStorage.getItem('tandem.setup.v1'))).not.toBeNull();
  let posted = 0;
  page.on('request', (r) => {
    if (r.url().includes('/v1/setup/reset') && r.method() === 'POST') posted++;
  });
  await page.locator('.setup-foot button', { hasText: 'Reset' }).click();
  const d = await review(page, 1);
  await d.locator('.reset-go').click();
  await expect(d.locator('[data-reset-result="local"]')).toContainText('This browser forgot your progress and the chat');
  expect(await page.evaluate(() => localStorage.getItem('tandem.setup.v1'))).toBeNull();
  expect(posted).toBe(0);
  await expect(page).toHaveURL(/#\/setup$/);
});

test('the result says what was freed, and the System view refreshes', async ({ page }) => {
  await open(page, '#/system');
  const d = await review(page, 3);
  await d.locator('.reset-word').fill('delete caches');
  const refresh = page.waitForRequest((r) => r.url().includes('/v1/dashboard/system'));
  await d.locator('.reset-go').click();
  await expect(d.locator('[data-reset-result="ok"]')).toContainText('Freed 14.7 GiB');
  await refresh;
});

test('a busy engine: every server level is off with the reason', async ({ page, request }) => {
  await setupMock(request, { busy: 2 });
  await open(page, '#/system');
  await expect(page.locator('[data-reset-busy]')).toContainText('2 requests are running. Reset is available when they finish.');
  for (const n of [2, 3, 4, 5]) await expect(page.locator(`qse-reset .reset-card[data-level="${n}"] .reset-review`)).toBeDisabled();
  await expect(page.locator('qse-reset .reset-card[data-level="1"] .reset-review')).toBeEnabled();
});

test('a 409 while the dialog is open says the same', async ({ page, request }) => {
  await open(page, '#/system');
  const d = await review(page, 3);
  await d.locator('.reset-word').fill('delete caches');
  await setupMock(request, { busy: 1 });
  await d.locator('.reset-go').click();
  await expect(d.locator('.reset-error')).toContainText('Reset is available when they finish');
});

test('level 2 shows the install progress of its job', async ({ page }) => {
  await open(page, '#/system');
  const d = await review(page, 2);
  await expect(d.locator('.reset-total')).toContainText('Restarts the engine');
  // level 2 is a plain confirm, no word (server/setup/reset.py WORDS)
  await expect(d.locator('.reset-word')).toHaveCount(0);
  await d.locator('.reset-go').click();
  await expect(d.locator('qse-install-progress')).toBeVisible();
  await expect(d.locator('[data-job-state="done"]')).toBeVisible({ timeout: 20_000 });
});

test('level 5 is the danger level, names the ledger and needs its own word', async ({ page }) => {
  await open(page, '#/system');
  const card = page.locator('qse-reset .reset-card[data-level="5"]');
  await expect(card).toHaveClass(/is-danger/);
  await expect(card.locator('.reset-danger-tag')).toHaveText(/Cannot be undone/);
  const d = await review(page, 5);
  await expect(page.locator('dialog.reset-dialog.is-danger')).toBeVisible();
  await expect(d).toContainText('The usage history');
  await expect(d).toContainText("The conductor's settings and journal");
  await expect(d.locator('[data-item$="/memory"]')).toContainText('at least');
  await expect(d.locator('.reset-word-show')).toHaveText('delete history');
});

test('a partial failure lists what stayed and does not say success', async ({ page, request }) => {
  await setupMock(request, { resetPartial: true });
  await open(page, '#/system');
  const d = await review(page, 3);
  await d.locator('.reset-word').fill('delete caches');
  await d.locator('.reset-go').click();
  const res = d.locator('[data-reset-result="partial"]');
  await expect(res).toContainText('could not delete 2 items');
  await expect(res.locator('.reset-failed li')).toHaveCount(2);
  await expect(res).not.toContainText('Freed');
});

test('nothing to delete: it says so and the button stays off', async ({ page, request }) => {
  await setupMock(request, { resetEmpty: true });
  await open(page, '#/system');
  const d = await review(page, 3);
  await expect(d.locator('[data-reset-empty]')).toHaveText('This level has nothing to delete.');
  await expect(d.locator('.reset-go')).toBeDisabled();
});

test('a double click sends one request', async ({ page }) => {
  await open(page, '#/system');
  const d = await review(page, 3);
  await d.locator('.reset-word').fill('delete caches');
  let posted = 0;
  page.on('request', (r) => {
    if (r.url().endsWith('/v1/setup/reset') && r.method() === 'POST') posted++;
  });
  await d.locator('.reset-go').dblclick();
  await expect(d.locator('[data-reset-result]')).toBeVisible();
  expect(posted).toBe(1);
});

test('a 401: a notice about the admin password, no password prompt', async ({ page, request }) => {
  await setupMock(request, { unauthorized: true });
  await open(page, '#/system');
  await expect(page.locator('qse-reset .reset-load-error')).toContainText('admin password');
  await expect(page.locator('qse-reset input[type="password"]')).toHaveCount(0);
});

test('390 px: the dialog is a full-screen sheet, the field is not zoomed, nothing scrolls sideways', async ({ page }) => {
  test.skip(!isPhone(page), 'phone only');
  await open(page, '#/system');
  const d = await review(page, 3);
  const box = await d.boundingBox();
  const vp = page.viewportSize()!;
  expect(box!.width).toBeGreaterThanOrEqual(vp.width - 1);
  expect(box!.height).toBeGreaterThanOrEqual(vp.height - 1);
  const fs = await d.locator('.reset-word').evaluate((el) => parseFloat(getComputedStyle(el).fontSize));
  expect(fs).toBeGreaterThanOrEqual(16);
  expect(await noHorizontalOverflow(page)).toBe(true);
});
