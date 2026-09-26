// Screenshots of the Live panel (VIS-23) in the busy scenario — desktop and phone, dark and light.
import { test } from '@playwright/test';
import { IS_MOCK, login } from './helpers';

const SIZES = { desktop: { width: 1440, height: 900 }, phone: { width: 390, height: 844 } } as const;
const PREFIX = process.env.PW_SHOTS_PREFIX ?? '';
const THEMES = (process.env.PW_SHOTS_THEMES ?? 'dark,light').split(',') as ('dark' | 'light')[];

for (const theme of THEMES) {
  for (const [size, viewport] of Object.entries(SIZES)) {
    test(`live ${size} ${theme}`, async ({ page, request }) => {
      await page.setViewportSize(viewport);
      await page.emulateMedia({ colorScheme: theme });
      await page.addInitScript((t) => localStorage.setItem('qse.theme', t), theme);
      if (IS_MOCK) await request.post('/__mock/live', { data: { scenario: 'busy' } });
      await login(page, '#/performance');
      await page.locator('.live-counts').waitFor({ timeout: 10_000 });
      await page.waitForTimeout(4500); // three samples: the 2 s figure and the sparkline's newest points
      await page.locator('.skeleton').first().waitFor({ state: 'detached', timeout: 10_000 }).catch(() => undefined);
      // the panel and the /metrics strip under it
      const panel = page.locator('.live');
      await panel.screenshot({ path: `screenshots/${PREFIX}live-${size}-${theme}.png` });
      if (size === 'phone') await page.screenshot({ path: `screenshots/${PREFIX}live-${size}-${theme}-fold.png`, fullPage: false });
      if (IS_MOCK) await request.post('/__mock/live', { data: { scenario: 'clear' } });
    });
  }
}
