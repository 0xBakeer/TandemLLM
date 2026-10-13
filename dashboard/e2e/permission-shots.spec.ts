// Permission modes (2026-10-13) against the in-browser chat mock: the composer's mode pill and its
// popover, YOLO's orange, the settings page's default, and the badge on a tool card that ran
// without asking. Desktop 1440 px and the phone at 390 and 320 px, dark and light. Pictures into
// screenshots/permission-*.png (PW_SHOTS_DIR overrides it).
import { expect, test, type Page } from '@playwright/test';
import { login } from './helpers';

const DIR = process.env.PW_SHOTS_DIR ?? 'screenshots';
const SIZES = { desktop: { width: 1440, height: 900 }, 390: { width: 390, height: 844 }, 320: { width: 320, height: 568 } } as const;
type Size = keyof typeof SIZES;

async function open(page: Page, theme: 'dark' | 'light', hash: string): Promise<void> {
  await page.emulateMedia({ colorScheme: theme });
  await page.addInitScript((t) => {
    localStorage.setItem('qse.theme', t);
    localStorage.removeItem('qse.chat.last');
    sessionStorage.clear();
  }, theme);
  await login(page, hash, `?chat=mock&t=${Date.now()}`);
  await page.locator('qse-chat').waitFor();
}
const shot = (page: Page, name: string) => page.screenshot({ path: `${DIR}/permission-${name}.png` });

for (const size of ['desktop', 390, 320] as Size[]) {
  for (const theme of ['dark', 'light'] as const) {
    const tag = `${size}-${theme}`;
    test(`${tag}: the mode pill reads auto, its popover lists the three, YOLO turns it orange`, async ({ page }) => {
      await page.setViewportSize(SIZES[size]);
      await open(page, theme, '#/chat/c_explain');
      const pill = page.locator('.ch-pill-mode');
      await expect(pill).toHaveText('auto');
      // on a phone the pill strip scrolls (its edge fades): bring the pill into view first
      await page.locator('.ch-pills').evaluate((el) => el.scrollTo({ left: el.scrollWidth }));
      await pill.click();
      await expect(page.locator('.ch-pop [role=menuitemradio]')).toHaveCount(3);
      await shot(page, `popover-${tag}`);
      await page.locator('.ch-pop-item.is-yolo').click();
      await expect(pill).toHaveText('YOLO');
      await expect(pill).toHaveClass(/is-yolo/);
      await shot(page, `yolo-${tag}`);
      // back to ask, the pill follows
      await page.locator('.ch-pills').evaluate((el) => el.scrollTo({ left: el.scrollWidth }));
      await pill.click();
      await page.locator('.ch-pop [role=menuitemradio]').first().click();
      await expect(pill).toHaveText('ask');
    });
  }
}

for (const size of ['desktop', 390] as Size[]) {
  test(`${size} dark: the settings page's default and the badge on a tool card that ran on its own`, async ({ page }) => {
    await page.setViewportSize(SIZES[size]);
    await open(page, 'dark', '#/chat/settings/tools');
    const list = page.locator('.ch-mode-list');
    await expect(list.locator('.ch-mode.is-on')).toHaveAttribute('data-mode', 'auto');
    await list.scrollIntoViewIfNeeded();
    await shot(page, `settings-${size}-dark`);
    await list.locator('.ch-mode.is-yolo input').check();
    await expect(list.locator('.ch-mode.is-on')).toHaveAttribute('data-mode', 'yolo');
    await shot(page, `settings-yolo-${size}-dark`);
    // a chat with finished bash calls: the badge says auto
    await open(page, 'dark', '#/chat/c_sync');
    await page.locator('.ch-approved').first().waitFor();
    await expect(page.locator('.ch-approved.is-auto').first()).toHaveText('auto');
    await page.locator('.ch-approved.is-auto').first().scrollIntoViewIfNeeded();
    await shot(page, `badge-${size}-dark`);
  });
}
