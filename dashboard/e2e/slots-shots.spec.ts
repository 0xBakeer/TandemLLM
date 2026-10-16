// VIS-199: pictures of the parallel-chats stepper against the mock (Super VL), phone 390 px and
// desktop 1440 px, dark and light: the default (3), above the measured range (5), a count that does
// not fit (16) with Next refused, and a typed value the engine does not take (17). Into
// screenshots/slots-*.png (PW_SHOTS_DIR overrides it); the phone has no sideways scroll.
import { expect, test, type Page } from '@playwright/test';
import { IS_MOCK, login, noHorizontalOverflow } from './helpers';
import { setSlots, setupMock } from './setup-helpers';

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
  await page.locator('#pf-slots-in').waitFor();
}

async function shot(page: Page, name: string, phone: boolean): Promise<void> {
  if (phone) {
    // the field under the sticky memory bar, as a thumb sees it
    await page.locator('[data-field="parallelism.slots"]').evaluate((el) => el.scrollIntoView({ block: 'start' }));
    await page.evaluate(() => window.scrollBy(0, -130));
    await page.waitForTimeout(150);
    await page.screenshot({ path: `${DIR}/slots-${name}.png` });
  } else {
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.waitForTimeout(150);
    await page.screenshot({ path: `${DIR}/slots-${name}.png` });
  }
}

for (const size of ['390', 'desktop'] as const) {
  for (const theme of ['dark', 'light'] as const) {
    const tag = `${size}-${theme}`;
    const phone = size !== 'desktop';
    test(`${tag}: default, untested, over and refused`, async ({ page, request }) => {
      await page.setViewportSize(SIZES[size]);
      await setupMock(request);
      await open(page, theme);
      await expect(page.locator('#pf-slots-in')).toHaveValue('3');
      await shot(page, `1-default-${tag}`, phone);
      await page.getByRole('button', { name: 'One chat more' }).click();
      await page.getByRole('button', { name: 'One chat more' }).click();
      await expect(page.locator('[data-slots-untested]')).toBeVisible();
      await shot(page, `2-untested-${tag}`, phone);
      await setSlots(page, 16);
      await expect(page.locator('[data-slots-error]')).toBeVisible();
      await page.locator('.setup-next').click();
      await expect(page.locator('[data-next-refused]')).toBeVisible();
      await shot(page, `3-over-${tag}`, phone);
      await setSlots(page, 17);
      await expect(page.locator('[data-slots-error]')).toContainText('at most 16');
      await shot(page, `4-engine-max-${tag}`, phone);
      if (phone) expect(await noHorizontalOverflow(page)).toBe(true);
    });
  }
}
