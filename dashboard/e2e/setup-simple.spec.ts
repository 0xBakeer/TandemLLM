// The simple guide on the mock: a fresh browser opens in simple mode, the three pages
// walk — tiles, quant cards, folded settings — and Finish runs the job from the open
// review fold. The advanced flow is the sibling files' territory. @mock
import { expect, test } from '@playwright/test';
import { IS_MOCK } from './helpers';
import { open, setupMock } from './setup-helpers';

test.skip(!IS_MOCK, 'the setup guide specs drive the mock (POST /__mock/setup)');

const MODEL = 'qwen3.8-27b';

test.beforeEach(async ({ request }) => {
  await setupMock(request);
});

test('a new browser opens simple: three pages, the open review fold, and a Finish that installs', async ({ page }) => {
  await open(page, '#/setup');
  const sw = page.locator('.setup-mode-switch');
  await expect(sw.getByRole('button', { name: 'Simple', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await expect(sw.getByRole('button', { name: 'Advanced', exact: true })).toHaveAttribute('aria-pressed', 'false');

  await page.locator(`.mtile[data-model="${MODEL}"]`).click();
  await expect(page).toHaveURL(new RegExp(`#/setup/${MODEL}/weights$`));
  await expect(page.locator('.wcard.is-selected')).toBeVisible();
  await expect(page.locator('.setup-nav-mid')).toContainText('Step 2 of 3');

  await page.locator('.setup-next').click();
  await expect(page).toHaveURL(new RegExp(`#/setup/${MODEL}/review$`));
  await expect(page.locator('.setup-folds details.setup-fold').first()).toBeVisible();
  // the review fold is the one that must never hide a reason: it opens on its own
  await expect(page.locator('details[data-fold="review"]')).toHaveAttribute('open', '');

  const finish = page.locator('.setup-finish');
  await expect(finish).toBeEnabled();
  await finish.click();
  await page.locator('.setup-confirm-go').click();
  await expect(page.locator('[data-job-state="done"]').first()).toBeVisible({ timeout: 20_000 });
});

test('the switch remembers: advanced stays after a reload, simple returns on a new context', async ({ page, browser }) => {
  await open(page, '#/setup');
  await page.locator('.setup-mode-switch').getByRole('button', { name: 'Advanced', exact: true }).click();
  await page.reload();
  await expect(page.locator('.setup-mode-switch').getByRole('button', { name: 'Advanced', exact: true })).toHaveAttribute('aria-pressed', 'true');
  const fresh = await browser.newContext();
  const p2 = await fresh.newPage();
  await open(p2, '#/setup');
  await expect(p2.locator('.setup-mode-switch').getByRole('button', { name: 'Simple', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await fresh.close();
});
