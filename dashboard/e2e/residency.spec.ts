// OPS-41 / OPS-42 in the browser: the System view's memory plan (the engine's split, what may load
// beside it, the last refusal) and the limits edited in the Conductor view, on the mock
// (`POST /__mock/conductor {residencyMem}` sets the plan's MemAvailable).
import { expect, test, type Page } from '@playwright/test';
import { IS_MOCK, login, noHorizontalOverflow, setMode } from './helpers';

function consoleErrors(page: Page): string[] {
  const out: string[] = [];
  page.on('console', (m) => m.type() === 'error' && out.push(m.text()));
  page.on('pageerror', (e) => out.push(String(e)));
  return out;
}

test.describe('the memory plan @mock', () => {
  test.skip(!IS_MOCK, 'needs the mock conductor');
  test.beforeEach(async ({ request }) => {
    await setMode(request, 'ok');
    await request.post('/__mock/conductor', { data: { reset: true, requireToken: false, residencyMem: 31 } });
  });
  test.afterAll(async ({ request }) => {
    await request.post('/__mock/conductor', { data: { reset: true, residencyMem: 31 } });
  });

  test('the System view shows the split, the floor, every role with its claim and the last refusal', async ({ page }) => {
    await login(page, '#/system');
    const errors = consoleErrors(page);
    const card = page.locator('#card-residency');
    await expect(card.locator('[data-split] [data-part]')).toHaveCount(9, { timeout: 10_000 });
    await expect(card.locator('[data-part-row="kv_pool"]')).toContainText('16.00 GiB');
    await expect(card.locator('[data-residency]')).toContainText('31.0 GiB');
    await expect(card.locator('[data-residency]')).toContainText('15.0 GiB + 6.0 GiB');
    await expect(card.locator('[data-residency-role="decider"]')).toContainText('8.2 GiB');
    await expect(card.locator('[data-residency-role="decider"]')).toContainText('fits, 22.8 GiB left');
    await expect(card.locator('[data-residency-role="reader"]')).toContainText('refused: it would break the floor');
    await expect(card.locator('[data-last-refusal]')).toContainText('reader');
    await expect(card.locator('[data-load-stages]')).toHaveCount(1);
    expect(errors).toEqual([]);
  });

  test('at 18 GiB nothing fits: the decider is refused at the floor', async ({ page, request }) => {
    await request.post('/__mock/conductor', { data: { residencyMem: 18 } });
    await login(page, '#/system');
    await expect(page.locator('[data-residency-role="decider"]')).toContainText('refused: it would break the floor (9.8 GiB left, 21.0 GiB needed)', { timeout: 10_000 });
  });

  test('the limits are edited in the Conductor view and the plan follows', async ({ page }) => {
    await login(page, '#/conductor?screen=profiles');
    const errors = consoleErrors(page);
    const sec = page.locator('[data-sec="residency"]');
    await sec.locator('summary').click();
    await sec.locator('input[data-res="floor_gib"]').fill('20');
    await expect(sec.locator('[data-claim="decider"] input[data-claim-gib="decider"]')).toHaveValue('8.2');
    await sec.getByLabel('change note').fill('floor 20 for a test');
    await expect(sec.locator('[data-save]')).toBeEnabled({ timeout: 5000 });
    await sec.locator('[data-save]').click();
    await expect(sec.locator('[data-savebar] .status.num')).toContainText('live v1', { timeout: 8000 });
    await page.goto('./#/system');
    // 31 - 8.2 = 22.8 is under 20 + 6
    await expect(page.locator('[data-residency-role="decider"]')).toContainText('refused: it would break the floor (22.8 GiB left, 26.0 GiB needed)', { timeout: 10_000 });
    expect(errors).toEqual([]);
  });

  test('phone: the plan fits 390 px without a sideways scroll', async ({ page }, info) => {
    test.skip(!info.project.name.includes('phone'), 'phone project only');
    await login(page, '#/system');
    await expect(page.locator('#card-residency [data-residency-role="decider"]')).toBeVisible({ timeout: 10_000 });
    await page.locator('#card-residency').scrollIntoViewIfNeeded();
    expect(await noHorizontalOverflow(page)).toBe(true);
    await page.locator('#card-residency').screenshot({ path: 'e2e-artifacts/residency-phone.png' });
  });

  test('desktop screenshot for the ticket', async ({ page }, info) => {
    test.skip(!info.project.name.startsWith('chromium-desktop'), 'one desktop project');
    await login(page, '#/system');
    await expect(page.locator('#card-residency [data-residency-role="decider"]')).toBeVisible({ timeout: 10_000 });
    await page.locator('#card-residency').screenshot({ path: 'e2e-artifacts/residency-desktop.png' });
  });
});
