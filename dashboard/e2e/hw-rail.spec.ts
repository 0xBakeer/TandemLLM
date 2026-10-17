// The hardware rail: a shell-level strip of what the box is doing right now —
// tiles first (the order a Spark operator reads them: GPU temp, GPU W, RAM, CPU,
// disk, LAN), one line of secondary numbers, then four charts. Fed entirely by
// the System route's own payload; it mounts only once /health has answered and
// the box serves, and never through the setup shell.
import { expect, test } from '@playwright/test';
import { IS_FAKE, IS_MOCK, login, setMode } from './helpers';

test.describe('hardware rail', () => {
  test.beforeEach(async ({ request }) => {
    if (IS_MOCK) await setMode(request, 'ok');
  });

  test('the rail rides the shell: tiles, the secondary line, four charts', async ({ page }) => {
    await login(page, '#/chat');
    // The host element carries no box of its own — the tab and drawer are both fixed,
    // so the assertions target them. Desktop (1280 here) opens the drawer by default.
    await expect(page.locator('.hw-tab')).toBeVisible();
    const rail = page.locator('.hw-rail');
    await expect(rail).toBeVisible();
    await expect(rail.locator('.hw-tile-value').first()).toBeVisible();
    if (!IS_FAKE) await expect(rail.locator('.hw-tile-value').first()).toContainText(/[0-9]/);
    await expect(rail.locator('.hw-chart qse-ribbon')).toHaveCount(4);
    await expect(rail.locator('.hw-tile-label', { hasText: 'GPU Power' })).toHaveCount(1);
    if (!IS_FAKE) {
      // the temperature legend carries the live value — a number with its unit, no hover needed
      await expect(rail.locator('.hw-chart qse-ribbon').first().locator('.legend .num').first())
        .toContainText(/[0-9]+\s+°C/);
    }
  });

  test('a host without a GPU reads empty tiles, not an error', async ({ page, request }) => {
    test.skip(!IS_MOCK, 'nogpu is a mock-tier mode');
    await setMode(request, 'nogpu');
    await login(page, '#/chat');
    const rail = page.locator('.hw-rail');
    await expect(rail).toBeVisible();
    await expect(rail.locator('.hw-tile-value').first()).toContainText(/[–—]/);
    await expect(rail.locator('.hw-rail-empty')).toHaveCount(0); // the host half is still live
  });

  test('the setup shell never grows a rail and the rail polls nothing there', async ({
    page,
    request,
  }) => {
    test.skip(!IS_MOCK, 'the setup shell is a mock-tier state');
    await request.post('/__mock/setup', { data: { setupMode: true } });
    let systemPolls = 0;
    page.on('request', (r) => {
      if (r.url().includes('/v1/dashboard/system')) systemPolls += 1;
    });
    await page.goto('/dashboard/');
    await expect(page.locator('[data-setup-mode]')).toBeVisible();
    await expect(page.locator('qse-hw-rail')).toHaveCount(0);
    await expect(page.locator('.hw-tab')).toHaveCount(0);
    await page.waitForTimeout(5500); // one poll period plus slack
    expect(systemPolls).toBe(0);
    await request.post('/__mock/setup', { data: { setupMode: false } }); // unpoison the worker
  });
});
