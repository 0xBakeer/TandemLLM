// VIS-17 — the box at a glance.
import { expect, test } from '@playwright/test';
import { IS_MOCK, login, setMode } from './helpers';

test.describe('system', () => {
  test.beforeEach(async ({ request }) => {
    if (IS_MOCK) await setMode(request, 'ok');
  });

  test('system cards show their values with units and the code hash can be copied', async ({ page, context }) => {
    await context.grantPermissions(['clipboard-read', 'clipboard-write']).catch(() => undefined);
    await login(page, '#/system');
    await expect(page.locator('#card-engine')).toContainText(/up|d |h /);
    await expect(page.locator('#card-memory')).toContainText('GB');
    await expect(page.locator('#card-gpu')).toContainText('°C');
    await expect(page.locator('#card-gpu')).toContainText(' W');
    await expect(page.locator('#card-gpu')).toContainText('MHz');
    await expect(page.locator('#card-queue')).toContainText('of 8');
    await expect(page.locator('#card-inflight')).toContainText('served');
    await expect(page.locator('#card-ledger')).toContainText('rows');
    await expect(page.locator('#card-disk')).toContainText('GB');
    await expect(page.locator('#card-engine')).toContainText('QWEN38_* set');
    const copy = page.locator('#card-engine').getByRole('button', { name: 'Copy code hash' });
    await copy.click();
    await expect(copy).toHaveClass(/is-done/);
  });

  test('GPU source unavailable: the card says not available and nothing else breaks', async ({ page, request }) => {
    test.skip(!IS_MOCK, 'needs the mock switch');
    await setMode(request, 'nogpu');
    await login(page, '#/system');
    await expect(page.locator('#card-gpu')).toContainText('Not available');
    await expect(page.locator('#card-memory')).toContainText('GB');
    await expect(page.locator('#card-engine .pill')).toBeVisible();
    await setMode(request, 'ok');
  });

  test('rolling memory chart accumulates one point per 5 s poll', async ({ page }) => {
    await login(page, '#/system');
    await expect(page.locator('#chart-memory .panel-foot')).toContainText(/1 sample/);
    await expect(page.locator('#chart-memory .panel-foot')).toContainText(/3 samples/, { timeout: 15_000 });
    await expect(page.locator('#chart-memory .legend')).toContainText('GPU allocated');
    await expect(page.locator('#chart-memory .legend')).toContainText('unified available');
    await expect(page.locator('#chart-gpu qse-ribbon')).toHaveCount(2);
  });

  test('warnings: waiting 7 of 8 highlights the queue card as nearly full', async ({ page, request }) => {
    test.skip(!IS_MOCK, 'needs the mock switch');
    await setMode(request, 'ok', { waiting: 7 });
    await login(page, '#/system');
    await expect(page.locator('#card-queue')).toHaveClass(/is-warn/);
    await expect(page.locator('#card-queue')).toContainText('queue nearly full');
    await expect(page.locator('#card-queue .meter-fill')).toHaveClass(/is-warn/);
    await setMode(request, 'ok', { waiting: 0 });
  });

  test('cache budget: 6.1 GB of 8 GB shows 76 % with both numbers', async ({ page }) => {
    await login(page, '#/system');
    const head = page.locator('#card-caches .meter-head');
    await expect(head).toContainText('of 8.00 GB');
    if (IS_MOCK) {
      await expect(head).toContainText('6.10 GB');
      await expect(head).toContainText('76 %');
    }
    const w = await page.locator('#card-caches .meter-fill').evaluate((e) => parseFloat((e as HTMLElement).style.width));
    expect(w).toBeGreaterThan(0);
  });

  test('disk space and ledger drops warn in signal orange', async ({ page, request }) => {
    test.skip(!IS_MOCK, 'needs the mock switch (busy mode = 15 GB free, 3 dropped rows)');
    await setMode(request, 'busy');
    await login(page, '#/system');
    await expect(page.locator('#card-disk')).toHaveClass(/is-warn/);
    await expect(page.locator('#card-disk')).toContainText('Below 20 GB');
    await expect(page.locator('#card-ledger')).toHaveClass(/is-warn/);
    await expect(page.locator('#card-ledger')).toContainText('3 dropped');
    await setMode(request, 'ok');
  });

  test('polling stops while the tab is hidden', async ({ page }) => {
    await login(page, '#/system');
    const calls: number[] = [];
    page.on('request', (r) => r.url().includes('/v1/dashboard/system') && calls.push(Date.now()));
    await page.evaluate(() => {
      Object.defineProperty(document, 'visibilityState', { value: 'hidden', configurable: true });
      document.dispatchEvent(new Event('visibilitychange'));
    });
    const n0 = calls.length;
    await page.waitForTimeout(11_000);
    expect(calls.length - n0).toBeLessThanOrEqual(0);
    await page.evaluate(() => {
      Object.defineProperty(document, 'visibilityState', { value: 'visible', configurable: true });
      document.dispatchEvent(new Event('visibilitychange'));
    });
    await expect.poll(() => calls.length, { timeout: 8000 }).toBeGreaterThan(n0);
  });
});
