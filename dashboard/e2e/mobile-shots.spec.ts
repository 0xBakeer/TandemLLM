// The phone fix of 2026-10-13 (an iPhone pass): the bottom bar of four slots with the rest
// behind More, the composer's two rows with every control in view, and the bar stepping aside
// while typing. iPhone sizes 390x844 and 320x568, dark and light, against the in-browser mock.
// Picked up by the `shots` project into screenshots/mobile-*.png (PW_SHOTS_DIR overrides it).
// Chromium's emulation raises no keyboard: the typing scene shows the focused composer with the
// bar gone and the thread grown by its height (on the phone the keys cover the rest).
import { expect, test, type Page } from '@playwright/test';
import { login, noHorizontalOverflow } from './helpers';

const DIR = process.env.PW_SHOTS_DIR ?? 'screenshots';
const SIZES = { 390: { width: 390, height: 844 }, 320: { width: 320, height: 568 } } as const;
type Size = keyof typeof SIZES;

async function open(page: Page, theme: 'dark' | 'light', hash: string): Promise<void> {
  await page.emulateMedia({ colorScheme: theme });
  await page.addInitScript((t) => {
    localStorage.setItem('qse.theme', t);
    localStorage.removeItem('qse.chat.last');
    sessionStorage.clear();
  }, theme);
  await login(page, hash, '?chat=mock');
  await page.locator('qse-chat, qse-setup, qse-usage').first().waitFor();
}

const shot = (page: Page, name: string, full = false) => page.screenshot({ path: `${DIR}/mobile-${name}.png`, fullPage: full });

for (const size of [390, 320] as Size[]) {
  for (const theme of ['dark', 'light'] as const) {
    test(`phone ${size} ${theme}: the thread while a run is on, the composer with Queue and Stop`, async ({ page }) => {
      await page.setViewportSize(SIZES[size]);
      await open(page, theme, '#/chat/c_sync');
      await page.locator('.ch-now').waitFor();
      await page.waitForTimeout(2200);
      // four slots in the bar, every label whole
      const bar = page.locator('.rail-tab:visible');
      await expect(bar).toHaveText(['Chat', 'Usage', 'Performance', 'More']);
      // every control of the composer is in the document and reachable
      for (const t of ['model', 'thinking effort', 'tools, skills and MCP servers']) await expect(page.locator(`.ch-pill[title="${t}"]`)).toBeVisible();
      await expect(page.locator('.ch-pill', { hasText: 'plan' })).toBeAttached();
      await expect(page.locator('.ch-stop')).toBeVisible();
      await expect(page.locator('.ch-queue-btn')).toBeVisible();
      await shot(page, `thread-${size}-${theme}`);
    });

    test(`phone ${size} ${theme}: the composer idle, then the bar gone while typing`, async ({ page }) => {
      await page.setViewportSize(SIZES[size]);
      await open(page, theme, '#/chat/c_explain');
      await page.locator('.ch-foot').first().waitFor();
      await page.waitForTimeout(300);
      const edge = await page.locator('.ch-pills').getAttribute('data-edge');
      if (size === 390) expect(edge).toBe('none'); // every pill in view, no scrolling
      else {
        // at 320 the strip may scroll (a longer model name does); then the edge with more fades
        expect(['none', 'start']).toContain(edge);
        if (edge === 'start') {
          await page.locator('.ch-pills').evaluate((el) => el.scrollTo({ left: el.scrollWidth }));
          await expect(page.locator('.ch-pills')).toHaveAttribute('data-edge', 'end');
          await page.locator('.ch-pills').evaluate((el) => el.scrollTo({ left: 0 }));
        }
      }
      await shot(page, `composer-idle-${size}-${theme}`);
      await page.locator('#ch-input').click();
      await page.keyboard.type('Make the test deterministic and run it');
      await expect(page.locator('.shell')).toHaveClass(/is-typing/);
      await page.waitForTimeout(400); // the bar's slide
      // the bar is off the screen while a field has focus
      const railTop = await page.locator('.rail').evaluate((el) => el.getBoundingClientRect().top);
      expect(railTop).toBeGreaterThanOrEqual(SIZES[size].height - 1);
      // the thread column runs to the screen's bottom edge: the bar's room is the thread's
      const chBottom = await page.locator('.ch').evaluate((el) => el.getBoundingClientRect().bottom);
      expect(Math.abs(chBottom - SIZES[size].height)).toBeLessThanOrEqual(2);
      await shot(page, `composer-typing-${size}-${theme}`);
    });

    test(`phone ${size} ${theme}: the More sheet`, async ({ page }) => {
      await page.setViewportSize(SIZES[size]);
      await open(page, theme, '#/chat/c_explain');
      await page.locator('.ch-foot').first().waitFor();
      await page.locator('.rail-more').click();
      await expect(page.locator('.more-sheet')).toBeVisible();
      await page.waitForTimeout(350);
      await shot(page, `more-${size}-${theme}`);
      // Escape closes it; the opener has focus again
      await page.keyboard.press('Escape');
      await expect(page.locator('.more-sheet')).toBeHidden();
    });

    test(`phone ${size} ${theme}: the composer's options`, async ({ page }) => {
      await page.setViewportSize(SIZES[size]);
      await open(page, theme, '#/chat/c_explain');
      await page.locator('.ch-foot').first().waitFor();
      await page.locator('.ch-pill[title="tools, skills and MCP servers"]').click();
      await expect(page.locator('.ch-pop')).toBeVisible();
      await page.waitForTimeout(250);
      await shot(page, `options-tools-${size}-${theme}`);
      await page.locator('.ch-pill[title="thinking effort"]').click();
      await expect(page.locator('.ch-pop')).toBeVisible();
      await page.waitForTimeout(250);
      await shot(page, `options-effort-${size}-${theme}`);
      await page.locator('.ch-pill[title="model"]').click();
      await expect(page.locator('.ch-pop')).toBeVisible();
      await page.waitForTimeout(250);
      await shot(page, `options-model-${size}-${theme}`);
    });
  }
}

// the other phone screens, at the narrow size, for the same overflow
test('phone 320 dark: a view behind More takes the More slot; the context sheet, settings, setup, usage', async ({ page }) => {
  await page.setViewportSize(SIZES[320]);
  await open(page, 'dark', '#/chat/c_sync');
  await page.locator('.ch-now').waitFor();
  await page.getByRole('button', { name: 'Context panel', exact: true }).click();
  await page.waitForTimeout(400);
  await shot(page, 'context-320-dark');
  await page.getByRole('button', { name: 'Close the context panel' }).click();
  await page.goto('./?chat=mock#/chat/settings/mcp');
  await page.locator('.ch-settings-main').waitFor();
  await page.waitForTimeout(300);
  await shot(page, 'settings-320-dark');
  await page.goto('./?chat=mock#/setup');
  await page.locator('qse-setup').waitFor();
  await page.waitForTimeout(800);
  await expect(page.locator('.rail-more')).toHaveText('Setup');
  await expect(page.locator('.rail-more')).toHaveClass(/is-on/);
  await shot(page, 'setup-320-dark');
  await page.goto('./?chat=mock#/usage');
  await page.locator('qse-usage').waitFor();
  await page.waitForTimeout(800);
  await shot(page, 'usage-320-dark');
});

test('phone 390 light: the Setup guide and the System view behind More', async ({ page }) => {
  await page.setViewportSize(SIZES[390]);
  await open(page, 'light', '#/setup');
  await page.locator('qse-setup').waitFor();
  await page.waitForTimeout(800);
  await shot(page, 'setup-390-light');
  await page.locator('.rail-more').click();
  await expect(page.locator('.more-sheet')).toBeVisible();
  await page.locator('.more-item', { hasText: 'System' }).click();
  await expect(page).toHaveURL(/#\/system/);
  await expect(page.locator('.more-sheet')).toBeHidden();
  await expect(page.locator('.rail-more')).toHaveText('System');
  await page.waitForTimeout(600);
  await shot(page, 'system-390-light');
});

// SRV-804: the Log panel's export button (SRV-803) on a phone. The save itself on iOS Safari and
// on Android Chrome is a check by hand on a device; here the button is in view and a download starts.
for (const size of [390, 320] as Size[]) {
  test(`phone ${size} dark: The export works on a phone: the button is in view without scrolling the toolbar sideways`, async ({ page }) => {
    await page.setViewportSize(SIZES[size]);
    await page.emulateMedia({ colorScheme: 'dark' });
    await login(page, '#/dev');
    await expect(page.locator('.console[data-state="open"]')).toBeVisible();
    const button = page.locator('[data-log-export]');
    await button.scrollIntoViewIfNeeded();
    await expect(button).toBeVisible();
    const box = await button.boundingBox();
    expect(box).not.toBeNull();
    expect(box!.x).toBeGreaterThanOrEqual(0);
    expect(box!.x + box!.width).toBeLessThanOrEqual(SIZES[size].width);
    // the toolbar wraps; it does not scroll sideways, and neither does the page
    const tools = page.locator('#console .panel-tools');
    expect(await tools.evaluate((el) => el.scrollWidth <= el.clientWidth + 1)).toBe(true);
    expect(await noHorizontalOverflow(page)).toBe(true);
    // every other control of the toolbar is still in view at this width
    for (const title of ['Wrap long lines', 'Clear view']) await expect(page.locator(`#console button[title="${title}"]`)).toBeVisible();
    await expect(page.getByRole('radio', { name: 'warning' })).toBeVisible();
    await expect(page.getByPlaceholder('search (server-side)')).toBeVisible();
    await shot(page, `log-export-${size}-dark`);
    const [download] = await Promise.all([page.waitForEvent('download'), button.click()]);
    expect(download.suggestedFilename()).toMatch(/^tandemllm-[A-Za-z0-9.-]+-\d{8}T\d{6}Z\.log$/);
    await expect(page.locator('[data-log-export-error]')).toHaveCount(0);
  });
}

test('phone 390 dark: a refused export shows its error inside the panel, within the screen', async ({ page }) => {
  await page.setViewportSize(SIZES[390]);
  await page.emulateMedia({ colorScheme: 'dark' });
  await login(page, '#/dev');
  await expect(page.locator('.console[data-state="open"]')).toBeVisible();
  await page.route('**/v1/dashboard/logs/export', (route) => route.fulfill({ status: 401, contentType: 'application/json', body: '{"error":{"type":"unauthorized","message":"no session"}}' }));
  await page.locator('[data-log-export]').click();
  const error = page.locator('#console [data-log-export-error] .state-error');
  await expect(error).toContainText('No file was saved.');
  const box = await error.boundingBox();
  expect(box!.x).toBeGreaterThanOrEqual(0);
  expect(box!.x + box!.width).toBeLessThanOrEqual(SIZES[390].width);
  expect(await noHorizontalOverflow(page)).toBe(true);
  await shot(page, 'log-export-error-390-dark');
});
