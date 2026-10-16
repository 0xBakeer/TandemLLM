// The phone fix of 2026-10-13 (an iPhone pass): the bottom bar of four slots with the rest
// behind More, the composer's two rows with every control in view, and the bar stepping aside
// while typing. iPhone sizes 390x844 and 320x568, dark and light, against the in-browser mock.
// Picked up by the `shots` project into screenshots/mobile-*.png (PW_SHOTS_DIR overrides it).
// Chromium's emulation raises no keyboard: the typing scene shows the focused composer with the
// bar gone and the thread grown by its height (on the phone the keys cover the rest).
import { expect, test, type Page } from '@playwright/test';
import { login } from './helpers';

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
