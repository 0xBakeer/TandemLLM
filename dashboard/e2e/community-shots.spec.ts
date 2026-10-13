// The community weights section of the weights step (CONTRACT.md 2026-10-13) against the mock:
// desktop 1440 px and the phone at 390 and 320 px, dark and light. Pictures into
// screenshots/community-*.png (PW_SHOTS_DIR overrides it) and a few checks on the way: the sync
// on open, the search, the sort, Use as a selection, the empty state and the phone's width.
import { expect, test, type Page } from '@playwright/test';
import { IS_MOCK, login, noHorizontalOverflow } from './helpers';
import { setupMock } from './setup-helpers';

test.skip(!IS_MOCK, 'the community specs drive the mock (POST /__mock/setup)');

const DIR = process.env.PW_SHOTS_DIR ?? 'screenshots';
const HASH = '#/setup/qwen3.8-27b/weights';
const SIZES = { desktop: { width: 1440, height: 900 }, 390: { width: 390, height: 844 }, 320: { width: 320, height: 568 } } as const;
type Size = keyof typeof SIZES;
const isPhone = (s: Size) => s !== 'desktop';

async function open(page: Page, theme: 'dark' | 'light'): Promise<void> {
  await page.emulateMedia({ colorScheme: theme });
  await page.addInitScript((t) => {
    localStorage.setItem('qse.theme', t);
    localStorage.removeItem('tandem.setup.v1');
  }, theme);
  // a fresh document each time (a goto to the same URL with another hash would not reload)
  await login(page, HASH, `?t=${Date.now()}`);
  await page.locator('[data-community]').waitFor();
}

const shot = (page: Page, name: string, el?: string) => (el ? page.locator(el).screenshot({ path: `${DIR}/community-${name}.png` }) : page.screenshot({ path: `${DIR}/community-${name}.png` }));

/** the section alone, cut from the whole page (an element shot would carry the sticky memory bar
 *  over its head); on the phone a viewport shot as well, with the shell around it */
async function sectionShot(page: Page, size: Size, name: string): Promise<void> {
  const box = await page.locator('.cw').evaluate((el) => {
    const r = el.getBoundingClientRect();
    return { x: Math.max(0, r.left - 8), y: r.top + window.scrollY - 8, width: r.width + 16, height: r.height + 16 };
  });
  await page.screenshot({ path: `${DIR}/community-${name}.png`, fullPage: true, clip: box });
  if (isPhone(size)) {
    await page.locator('.cw-h').evaluate((el) => el.scrollIntoView({ block: 'start' }));
    await page.evaluate(() => window.scrollBy(0, -150));
    await page.waitForTimeout(150);
    await shot(page, `${name}-screen`);
  }
}

for (const size of ['desktop', 390, 320] as Size[]) {
  for (const theme of ['dark', 'light'] as const) {
    const tag = `${size}-${theme}`;

    test(`${tag}: the section syncs on open, then shows three packs from GitHub`, async ({ page, request }) => {
      await page.setViewportSize(SIZES[size]);
      await setupMock(request, { communityMs: 2500 });
      await open(page, theme);
      // the sync runs: the head says so and the cards dim, the cached list stays on screen
      await expect(page.locator('.cw-src')).toHaveAttribute('data-source', 'syncing');
      await expect(page.locator('[data-community-card]')).toHaveCount(3);
      await sectionShot(page, size, `syncing-${tag}`);
      await expect(page.locator('.cw-src')).toHaveAttribute('data-source', 'github', { timeout: 6000 });
      await expect(page.locator('.cw-src-text')).toContainText('from GitHub, synced');
      await expect(page.locator('[data-count]')).toHaveText('3 packs');
      // ThinkingCap first (newest), verified, with the measured speeds and the greyed image modality
      const first = page.locator('[data-community-card]').first();
      await expect(first).toHaveAttribute('data-community-card', 'thinkingcap-27b-kearuga-exl3-4.00');
      await expect(first.locator('.ccard-login')).toHaveText('@0xWhiteMage');
      await expect(first.locator('.ccard-login')).toHaveAttribute('rel', 'noopener noreferrer');
      await expect(first.locator('[data-speed] .ccard-sp-n').first()).toHaveText('39.6');
      await expect(first.locator('[data-modality="image"]')).toHaveClass(/is-off/);
      await expect(page.locator('[data-dropped]')).toContainText('1 entry was left out as invalid');
      if (isPhone(size)) expect(await noHorizontalOverflow(page)).toBe(true);
      await sectionShot(page, size, `list-${tag}`);
      if (!isPhone(size)) {
        // the whole step: our radio cards above, the community section under them
        await page.evaluate(() => window.scrollTo(0, 0));
        await page.screenshot({ path: `${DIR}/community-step-${tag}.png`, fullPage: true });
        // the tooltip of a modality the engine cannot run yet
        await first.locator('[data-modality="image"]').hover();
        await page.waitForTimeout(250);
        await shot(page, `tooltip-${tag}`, '.cw-grid');
      }
    });

    test(`${tag}: search narrows the list, sort reorders it`, async ({ page, request }) => {
      await page.setViewportSize(SIZES[size]);
      await setupMock(request, { communityMs: 50 });
      await open(page, theme);
      await expect(page.locator('.cw-src')).toHaveAttribute('data-source', 'github', { timeout: 6000 });
      const search = page.locator('.cw-search input');
      await search.fill('white');
      await expect(page.locator('[data-community-card]')).toHaveCount(1);
      await expect(page.locator('[data-count]')).toHaveText('1 of 3 packs');
      await sectionShot(page, size, `search-${tag}`);
      await search.fill('nothing like this');
      await expect(page.locator('[data-community-nomatch]')).toBeVisible();
      if (size === 'desktop') await sectionShot(page, size, `nomatch-${tag}`);
      await page.locator('[data-community-nomatch] button').click();
      await expect(page.locator('[data-community-card]')).toHaveCount(3);
      await page.locator('.cw-sort select').selectOption('size');
      const order = await page.locator('[data-community-card]').evaluateAll((els) => els.map((e) => e.getAttribute('data-community-card')));
      expect(order).toEqual(['mock-stock-27b-exl3-3.00', 'thinkingcap-27b-kearuga-exl3-4.00', 'mock-sparktuner-27b-exl3-5.00']);
      await sectionShot(page, size, `sort-size-${tag}`);
    });

    test(`${tag}: Use selects the pack like one of our cards, and Review names it`, async ({ page, request }) => {
      await page.setViewportSize(SIZES[size]);
      await setupMock(request, { communityMs: 50 });
      await open(page, theme);
      await expect(page.locator('.cw-src')).toHaveAttribute('data-source', 'github', { timeout: 6000 });
      const card = page.locator('[data-community-card="thinkingcap-27b-kearuga-exl3-4.00"]');
      await card.locator('.ccard-use').click();
      await expect(card).toHaveClass(/is-selected/);
      await expect(card.locator('.ccard-use')).toHaveText(/Selected/);
      await expect(card).toHaveAttribute('data-field', 'weights.profile');
      // our radio cards all off
      await expect(page.locator('.wcard.is-selected')).toHaveCount(0);
      await sectionShot(page, size, `selected-${tag}`);
      // a hash change in the same document (a new document would start with no saved progress; the
      // phone's rail has no step links)
      await page.evaluate(() => (location.hash = '#/setup/qwen3.8-27b/review'));
      await expect(page.locator('.review')).toBeVisible();
      await expect(page.locator('.review')).toContainText('ThinkingCap Qwen3.8-27B Kearuga');
      if (!isPhone(size)) await shot(page, `review-${tag}`);
    });
  }
}

test('desktop dark: the empty state, the offline copy and the failed sync', async ({ page, request }) => {
  await page.setViewportSize(SIZES.desktop);
  await setupMock(request, { communityEmpty: true, communityMs: 50 });
  await open(page, 'dark');
  await expect(page.locator('[data-community-empty]')).toBeVisible();
  await expect(page.locator('[data-community-empty] a.btn-primary')).toHaveAttribute('href', /issues\/new\?template=community-quant/);
  await sectionShot(page, 'desktop', 'empty-desktop-dark');

  await setupMock(request, { communitySource: 'error', communityMs: 50 });
  await open(page, 'dark');
  await expect(page.locator('.cw-src')).toHaveAttribute('data-source', 'cache', { timeout: 6000 });
  await expect(page.locator('.cw-src-err')).toContainText('The sync failed');
  await sectionShot(page, 'desktop', 'offline-desktop-dark');
  // Refresh runs the sync again
  await setupMock(request, { communityMs: 1500 });
  await page.locator('.cw-refresh').click();
  await expect(page.locator('.cw-src')).toHaveAttribute('data-source', 'syncing');
  await expect(page.locator('.cw-src')).toHaveAttribute('data-source', 'github', { timeout: 6000 });
  await page.locator('[data-dropped] button').click();
  await expect(page.locator('[data-dropped] li')).toHaveCount(1);
  await sectionShot(page, 'desktop', 'dropped-desktop-dark');
});

// the 0.8.2 live check: "On this box" ran past the card's edge and was cut at 320; the sync error
// with a long URL was clipped. Both wrap now.
for (const size of ['desktop', 390, 320] as Size[]) {
  test(`${size} dark: a pack on the box and a long sync error wrap inside the card and the head`, async ({ page, request }) => {
    await page.setViewportSize(SIZES[size]);
    await setupMock(request, { communityOnDisk: true, communitySource: 'error', communityMs: 50 });
    await open(page, 'dark');
    await expect(page.locator('.cw-src-err')).toContainText('community-index.json');
    const card = page.locator('[data-community-card="thinkingcap-27b-kearuga-exl3-4.00"]');
    await expect(card.locator('.ccard-readout dd').last()).toHaveText('On this box');
    // nothing reaches past the card's or the page's right edge
    const over = await page.evaluate(() => {
      const bad: string[] = [];
      const page_ = document.documentElement.clientWidth;
      for (const el of Array.from(document.querySelectorAll<HTMLElement>('.cw *'))) {
        const r = el.getBoundingClientRect();
        if (r.width && r.right > page_ + 1) bad.push(el.className);
      }
      for (const card_ of Array.from(document.querySelectorAll<HTMLElement>('.ccard'))) {
        const cr = card_.getBoundingClientRect();
        for (const el of Array.from(card_.querySelectorAll<HTMLElement>('*'))) {
          const r = el.getBoundingClientRect();
          if (r.width && r.right > cr.right + 1 && !el.closest('.tip-host')) bad.push(el.className);
        }
      }
      return bad;
    });
    expect(over).toEqual([]);
    await sectionShot(page, size, `ondisk-${size}-dark`);
  });
}

test('phone 390 dark: the whole weights step, our cards and the community section', async ({ page, request }) => {
  await page.setViewportSize(SIZES[390]);
  await setupMock(request, { communityMs: 50 });
  await open(page, 'dark');
  await expect(page.locator('.cw-src')).toHaveAttribute('data-source', 'github', { timeout: 6000 });
  await page.screenshot({ path: `${DIR}/community-step-390-dark.png`, fullPage: true });
});
