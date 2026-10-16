// QA VIS-150, VIS-152, VIS-154, VIS-156 (VIS-149, VIS-151, VIS-153, VIS-155): the review's
// downloads and the install page of a finished job against the mock, desktop and 390 px. @mock
import { expect, test, type Page } from '@playwright/test';
import { IS_MOCK, noHorizontalOverflow } from './helpers';
import { open, setupMock, setSlots } from './setup-helpers';

test.skip(!IS_MOCK, 'the setup guide specs drive the mock (POST /__mock/setup)');

const MODEL = 'qwen3.8-27b';

test.beforeEach(async ({ request }) => {
  await setupMock(request);
});

async function install(page: Page) {
  await open(page, `#/setup/${MODEL}/weights`);
  await page.locator('[data-profile-card="exl3-6.0"]').click();
  await open(page, `#/setup/${MODEL}/parallelism`);
  await setSlots(page, 2);
  await open(page, `#/setup/${MODEL}/review`);
  await expect(page.locator('.rv-table tbody tr').first()).toBeVisible();
  await page.locator('.setup-finish').click();
  await page.locator('.setup-confirm-go').click();
  await expect(page.locator('[data-job-state="done"]')).toBeVisible({ timeout: 20_000 });
}

test('Reload after the job finished: ten reloads show the final state and the whole log', async ({ page }) => {
  await install(page);
  const count = await page.locator('[data-log-count]').textContent();
  expect(Number(count)).toBeGreaterThan(5);
  for (let i = 0; i < 10; i++) {
    await page.reload();
    await expect(page.locator('[data-job-state="done"]')).toBeVisible();
    await expect(page.locator('.inst-step.is-running, .inst-step.is-pending, .inst-step.is-waiting')).toHaveCount(0);
    await expect(page.locator('[data-log-count]')).toHaveText(count!);
  }
  // collapsed on success, with its count; it opens to the first and the last line
  const box = page.locator('.inst-logbox');
  await expect(box).not.toHaveAttribute('open', '');
  await box.locator('summary').click();
  await expect(page.locator('.inst-log')).toContainText('[setup] selection valid');
  await expect(page.locator('.inst-log')).toContainText('[smoke] chat request 200');
  await noHorizontalOverflow(page);
});

test('The job page lists the changes under the check step', async ({ page }) => {
  await install(page);
  const list = page.locator('.inst-step[data-step="check"] [data-job-changes]');
  await expect(list).toBeVisible();
  const n = await list.locator('li').count();
  expect(n).toBeGreaterThan(0);
  await expect(list.locator('summary')).toHaveText(`${n} setting${n === 1 ? '' : 's'} change`);
  await list.locator('summary').click();
  await expect(list.locator('li').first()).toContainText('→');
  await noHorizontalOverflow(page);
});

test('Missing component weights that can be fetched: the review lists the download with its size', async ({ page, request }) => {
  await setupMock(request, { shadowMissing: true });
  await open(page, `#/setup/${MODEL}/review`);
  const d = page.locator('[data-review-downloads] [data-download="qwen3.5-0.8b"]');
  await expect(d).toContainText('Qwen3.5-0.8B (the shadow reader)');
  await expect(d).toContainText('Shadow reader');
  await expect(d).toContainText('1.77 GB');
  await noHorizontalOverflow(page);
});

test('A component fails after the restart: installed, with a warning naming the tool and its reason', async ({ page, request }) => {
  await setupMock(request, { shadowMissing: true });
  await install(page);
  for (let i = 0; i < 2; i++) {
    // live, then after a reload
    const w = page.locator('[data-job-warnings] li.is-warn');
    await expect(w).toHaveCount(1);
    await expect(w).toContainText('Shadow reader did not load (FileNotFoundError');
    await expect(page.locator('.inst-head .inst-sub').first()).toHaveText('exl3-6.0 is live on port 8000.');
    if (i === 0) await page.reload();
  }
  await noHorizontalOverflow(page);
});
