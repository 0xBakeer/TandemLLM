// the shell: served, no password prompt anywhere (AU1, 2026-10-06: no sign-in screen, no sign-out,
// no token field), a 401 from a server with a password set (a banner, still no prompt), navigation +
// reload, theme, mobile, engine offline. (Static path traversal is the engine's static handler; it is covered in
// the fake-engine tier, not against the mock.)
import { expect, test } from '@playwright/test';
import { IS_MOCK, login, setMode } from './helpers';

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

  test('AU1: no password set, the dashboard opens straight into Usage: no sign-in screen, no token field, no sign-out', async ({ page, context }) => {
    await context.clearCookies();
    await page.goto('./');
    await expect(page.locator('.shell')).toBeVisible();
    await expect(page.locator('.rail a.rail-tab.is-on')).toHaveText(/Usage/);
    await expect(page.locator('qse-usage')).toBeVisible();
    await expect(page.locator('#token, #cond-token, .login-card')).toHaveCount(0);
    await expect(page.getByRole('button', { name: 'Sign out' })).toHaveCount(0);
    await expect(page.locator('[data-refused]')).toHaveCount(0);
    const s = await (await page.request.get('/v1/dashboard/session')).json();
    expect(s).toMatchObject({ login: false, password_set: false });
  });

  test('AU1: a server with a password set answers 401; the page shows a banner, never a sign-in, and keeps the tab', { tag: '@mock' }, async ({ page, request }) => {
    test.skip(!IS_MOCK, 'needs the mock password switch');
    await login(page, '#/system');
    await expect(page.locator('qse-system')).toBeVisible();
    await setMode(request, 'ok', { password: true });
    await page.reload();
    await expect(page.locator('[data-refused]')).toBeVisible({ timeout: 15_000 });
    await expect(page.locator('[data-refused]')).toContainText('admin password is set');
    expect(page.url()).toContain('#/system');
    await expect(page.locator('.shell')).toBeVisible();
    await expect(page.locator('#token, #cond-token, .login-card, input[type="password"]')).toHaveCount(0);
    await setMode(request, 'ok');
  });

  test('navigation and reload keep the Dev tab', async ({ page }) => {
    await login(page);
    await page.getByRole('link', { name: 'Dev' }).click();
    await expect(page.locator('qse-dev')).toBeVisible();
    await page.reload();
    await expect(page.locator('qse-dev')).toBeVisible();
    // on the phone the Dev link is behind More and the More slot shows Dev: the visible one counts
    await expect(page.locator('.rail-tab.is-on:visible')).toHaveText(/Dev/);
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

  test('phone: the bottom bar is Chat, Usage, Performance and More, every label whole at 390 px', async ({ page }, info) => {
    test.skip(info.project.name !== 'webkit-phone', 'the bottom bar is the phone layout');
    // NP1's night pass (2026-10-04, issue 5): "Backends" showed as "Backen…"; since 2026-10-13 the
    // bar holds four slots and the other views sit behind More
    await login(page);
    await expect(page.locator('.rail-tab').first()).toBeVisible();
    const shown = await page.evaluate(() =>
      [...document.querySelectorAll<HTMLElement>('.rail-tab')]
        .filter((tab) => tab.offsetParent !== null)
        .map((tab) => {
          const el = tab.querySelector('.rail-label') as HTMLElement;
          return { text: el.textContent, over: el.scrollWidth - el.clientWidth };
        }));
    expect(shown.map((x) => x.text)).toEqual(['Chat', 'Usage', 'Performance', 'More']);
    expect(shown.filter((x) => x.over > 0)).toEqual([]);
    // the More sheet lists the rest; a view behind More takes the More slot
    await page.locator('.rail-more').click();
    await expect(page.locator('.more-sheet')).toBeVisible();
    await expect(page.locator('.more-item')).toHaveText(['Dev', 'Playground', 'Conductor', 'Backends', 'System', 'Setup']);
    await page.locator('.more-item', { hasText: 'Dev' }).click();
    await expect(page).toHaveURL(/#\/dev/);
    await expect(page.locator('.more-sheet')).toBeHidden();
    await expect(page.locator('.rail-more')).toHaveText('Dev');
    await expect(page.locator('.rail-more')).toHaveClass(/is-on/); // the button's own state — on purpose, not the view's
  });

  test('AU1: no password set, Performance reads /v1/dashboard/metrics with no session and System opens; no cookie is set', { tag: '@mock' }, async ({ page, context }) => {
    test.skip(!IS_MOCK, 'reads the mock strip');
    await context.clearCookies();
    await page.goto('./#/performance');
    await expect(page.locator('.shell')).toBeVisible();
    await expect(page.locator('qse-performance')).toBeVisible();
    await expect(page.locator('.live-metrics .stat')).toHaveCount(3);
    await expect(page.locator('.live-metrics .state-error')).toHaveCount(0);
    await page.getByRole('link', { name: 'System' }).click();
    await expect(page.locator('qse-system')).toBeVisible();
    const cookies = await context.cookies();
    expect(cookies.find((c) => c.name === 'qse_dash')).toBeUndefined();
  });
});
