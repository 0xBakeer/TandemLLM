// the shell: served, login, wrong token, session expiry, navigation + reload, theme,
// mobile, engine offline, and the login switched off (no token screen, no sign-out). (Static path traversal is the engine's static handler; it is covered in
// the fake-engine tier, not against the mock.)
import { expect, test } from '@playwright/test';
import { IS_MOCK, TOKEN, login, setMode } from './helpers';

test.describe('shell', () => {
  test.beforeEach(async ({ request }) => {
    if (IS_MOCK) await setMode(request, 'ok');
  });

  test('served: /dashboard redirects to /dashboard/ and the shell loads from one origin', async ({ page, baseURL }) => {
    const origin = new URL(baseURL!).origin;
    const foreign: string[] = [];
    page.on('request', (r) => {
      if (!r.url().startsWith(origin)) foreign.push(r.url());
    });
    const res = await page.goto(new URL('/dashboard', baseURL).toString());
    expect(res?.ok()).toBeTruthy();
    expect(page.url()).toMatch(/\/dashboard\/(\?.*)?(#.*)?$/);
    await expect(page.locator('qse-app')).toBeVisible();
    expect(foreign).toEqual([]);
  });

  test('login: no session shows the token screen; a correct token opens the Usage tab', async ({ page, context }) => {
    await context.clearCookies();
    await page.goto('./');
    await expect(page.locator('#token')).toBeVisible();
    await page.locator('#token').fill(TOKEN);
    await page.getByRole('button', { name: 'Open the dashboard' }).click();
    await expect(page.locator('.shell')).toBeVisible();
    await expect(page.locator('.rail-tab.is-on')).toHaveText(/Usage/);
    await expect(page.locator('qse-usage')).toBeVisible();
  });

  test('wrong token: the screen says it was refused and no session cookie is set', async ({ page, context }) => {
    await context.clearCookies();
    await page.goto('./');
    await page.locator('#token').fill('definitely-wrong');
    await page.getByRole('button', { name: 'Open the dashboard' }).click();
    await expect(page.locator('.login-error')).toHaveText(/refused/);
    const cookies = await context.cookies();
    expect(cookies.find((c) => c.name === 'qse_dash')).toBeUndefined();
    await expect(page.locator('#token')).toBeVisible();
  });

  test('session expiry: the next 401 returns to the token screen and keeps the tab in the URL', { tag: '@mock' }, async ({ page, request }) => {
    test.skip(!IS_MOCK, 'needs the mock failure switch');
    await login(page, '#/system');
    await expect(page.locator('qse-system')).toBeVisible();
    await setMode(request, 'expire');
    await expect(page.locator('#token')).toBeVisible({ timeout: 15_000 });
    expect(page.url()).toContain('#/system');
    await page.locator('#token').fill(TOKEN);
    await page.getByRole('button', { name: 'Open the dashboard' }).click();
    await expect(page.locator('qse-system')).toBeVisible();
  });

  test('navigation and reload keep the Dev tab', async ({ page }) => {
    await login(page);
    await page.getByRole('link', { name: 'Dev' }).click();
    await expect(page.locator('qse-dev')).toBeVisible();
    await page.reload();
    await expect(page.locator('qse-dev')).toBeVisible();
    await expect(page.locator('.rail-tab.is-on')).toHaveText(/Dev/);
  });

  test('theme: follows a light OS preference, and a toggle to dark is remembered', async ({ page }) => {
    await page.emulateMedia({ colorScheme: 'light' });
    await page.addInitScript(() => {
      // forget a remembered theme once, before the first load only
      if (!sessionStorage.getItem('pw-theme-cleared')) {
        localStorage.removeItem('qse.theme');
        sessionStorage.setItem('pw-theme-cleared', '1');
      }
    });
    await login(page);
    await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
    await page.getByRole('button', { name: /Switch to dark theme/ }).click();
    await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
    await page.reload();
    await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  });

  test('engine offline: the pill says offline and panels name the endpoint', { tag: '@mock' }, async ({ page, request }) => {
    test.skip(!IS_MOCK, 'needs the mock failure switch');
    await login(page, '#/system');
    await expect(page.locator('.pill[data-status]')).toHaveText(/ok|busy/);
    await setMode(request, 'offline');
    await expect(page.locator('.pill[data-status="offline"]')).toBeVisible({ timeout: 15_000 });
    await expect(page.locator('.state-error').first()).toContainText('/v1/dashboard/system');
    await setMode(request, 'ok');
    await expect(page.locator('.pill[data-status="ok"], .pill[data-status="busy"]')).toBeVisible({ timeout: 15_000 });
  });

  test('status pill shows the engine state, version and uptime', async ({ page }) => {
    await login(page);
    await expect(page.locator('.pill[data-status]')).toBeVisible();
    await expect(page.locator('.topbar-meta')).toContainText(/up \d/);
  });

  test('sign out returns to the token screen', async ({ page }) => {
    await login(page);
    await page.getByRole('button', { name: 'Sign out' }).click();
    await expect(page.locator('#token')).toBeVisible();
  });

  test('login off: the dashboard opens straight into its views, with no token screen and no sign-out', { tag: '@mock' }, async ({ page, context, request }) => {
    test.skip(!IS_MOCK, 'needs the mock login switch');
    await setMode(request, 'ok', { login: false });
    await context.clearCookies();
    await page.goto('./#/performance');
    await expect(page.locator('.shell')).toBeVisible();
    await expect(page.locator('#token')).toHaveCount(0);
    await expect(page.getByRole('button', { name: 'Sign out' })).toHaveCount(0);
    await expect(page.locator('qse-performance')).toBeVisible();
    // the 5-minute strip reads /v1/dashboard/metrics, which needs no session with the login off
    await expect(page.locator('.live-metrics .stat')).toHaveCount(3);
    await expect(page.locator('.live-metrics .state-error')).toHaveCount(0);
    await expect(page.locator('#token')).toHaveCount(0);
    await page.getByRole('link', { name: 'System' }).click();
    await expect(page.locator('qse-system')).toBeVisible();
    const cookies = await context.cookies();
    expect(cookies.find((c) => c.name === 'qse_dash')).toBeUndefined();
  });
});
