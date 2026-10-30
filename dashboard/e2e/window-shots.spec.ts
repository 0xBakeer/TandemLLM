// VIS-199: pictures of the window steppers against the mock (Super VL, 1,048,576 tokens), phone
// 390 px and desktop 1440 px, dark and light: the default (256k), the top rung (1M), a typed value
// above the model with Next refused, and how many chats can all hold the window (SRV-707). Into
// screenshots/window-*.png (PW_SHOTS_DIR overrides it); the phone has no sideways scroll.
import { expect, test, type Page } from '@playwright/test';
import { IS_MOCK, login, noHorizontalOverflow } from './helpers';
import { setupMock, setWindow } from './setup-helpers';

test.skip(!IS_MOCK, 'drives the mock (POST /__mock/setup)');

const DIR = process.env.PW_SHOTS_DIR ?? 'screenshots';
const HASH = '#/setup/nemotron-3.5-super-vl-120b-a12b/parallelism';
const SIZES = { desktop: { width: 1440, height: 900 }, 390: { width: 390, height: 844 } } as const;

async function open(page: Page, theme: 'dark' | 'light'): Promise<void> {
  await page.emulateMedia({ colorScheme: theme });
  await page.addInitScript((t) => {
    localStorage.setItem('qse.theme', t);
    localStorage.removeItem('tandem.setup.v1');
  }, theme);
  await login(page, HASH, `?t=${Date.now()}`);
  await page.locator('#pf-win-in').waitFor();
}

async function shot(page: Page, name: string, phone: boolean, at = 'window'): Promise<void> {
  if (phone) {
    await page.locator(`[data-field="parallelism.${at}"]`).evaluate((el) => el.scrollIntoView({ block: 'start' }));
    await page.evaluate(() => window.scrollBy(0, -130));
  } else {
    await page.evaluate(() => window.scrollTo(0, 200));
  }
  await page.waitForTimeout(150);
  await page.screenshot({ path: `${DIR}/window-${name}.png` });
}

for (const size of ['390', 'desktop'] as const) {
  for (const theme of ['dark', 'light'] as const) {
    const tag = `${size}-${theme}`;
    const phone = size !== 'desktop';
    test(`${tag}: default, 1M, above the model, the guarantee`, async ({ page, request }) => {
      await page.setViewportSize(SIZES[size]);
      await setupMock(request);
      await open(page, theme);
      await expect(page.locator('#pf-win-in')).toHaveValue('262,144');
      await shot(page, `1-default-${tag}`, phone);
      const longer = page.getByRole('button', { name: 'Longest conversation: one step longer' });
      await longer.click();
      await longer.click();
      await expect(page.locator('#pf-win-in')).toHaveValue('1,048,576');
      await shot(page, `2-top-1m-${tag}`, phone);
      await setWindow(page, 'window', '2m');
      await page.locator('.setup-next').click();
      await expect(page.locator('[data-next-refused]')).toBeVisible();
      await shot(page, `3-above-model-${tag}`, phone);
      await setWindow(page, 'window', '128k');
      await page.locator('#pf-win-in').blur();
      await expect(page.locator('[data-lanes-full]')).toBeVisible();
      await shot(page, `4-lanes-full-${tag}`, phone, 'slots');
      if (phone) expect(await noHorizontalOverflow(page)).toBe(true);
    });
  }
}
