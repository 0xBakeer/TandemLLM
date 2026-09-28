// Screenshots of every view — desktop and phone, dark and light — into dashboard/screenshots/.
import { test } from '@playwright/test';
import { login } from './helpers';

const VIEWS = ['usage', 'performance', 'dev', 'system'] as const;
const SIZES = { desktop: { width: 1440, height: 900 }, phone: { width: 390, height: 844 } } as const;
// Shots from a real server (PW_BASE) go beside the mock's under a prefix of their own, e.g.
// PW_SHOTS_PREFIX=fake-engine- PW_SHOTS_THEMES=light (light shots on the fake engine).
const PREFIX = process.env.PW_SHOTS_PREFIX ?? '';
const THEMES = (process.env.PW_SHOTS_THEMES ?? 'dark,light').split(',') as ('dark' | 'light')[];

for (const theme of THEMES) {
  for (const [size, viewport] of Object.entries(SIZES)) {
    for (const view of VIEWS) {
      test(`${view} ${size} ${theme}`, async ({ page }) => {
        await page.setViewportSize(viewport);
        await page.emulateMedia({ colorScheme: theme });
        await page.addInitScript((t) => localStorage.setItem('qse.theme', t), theme);
        await login(page, `#/${view}`);
        await page.waitForTimeout(view === 'system' ? 7000 : 2500);
        await page.locator('.skeleton').first().waitFor({ state: 'detached', timeout: 10_000 }).catch(() => undefined);
        await page.screenshot({ path: `screenshots/${PREFIX}${view}-${size}-${theme}.png`, fullPage: true });
        // The phone's bottom tab bar is position: fixed, which a full-page capture places mid-page;
        // a viewport-only shot shows it as a phone does.
        if (size === 'phone') await page.screenshot({ path: `screenshots/${PREFIX}${view}-${size}-${theme}-fold.png`, fullPage: false });
      });
    }
    test(`login ${size} ${theme}`, async ({ page, context }) => {
      test.skip(PREFIX !== '', 'the token screen shows no data: the mock shots have it');
      await page.setViewportSize(viewport);
      await page.emulateMedia({ colorScheme: theme });
      await page.addInitScript((t) => localStorage.setItem('qse.theme', t), theme);
      await context.clearCookies();
      await page.goto('./');
      await page.locator('#token').waitFor();
      await page.screenshot({ path: `screenshots/${PREFIX}login-${size}-${theme}.png`, fullPage: true });
    });
  }
}
