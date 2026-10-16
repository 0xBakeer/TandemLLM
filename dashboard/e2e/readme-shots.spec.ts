// The README's grid and the gallery's pictures (VIS-177) against the mock: the Chat tab with its
// preview pane, the Setup Guide's tiles, Bolt's speed table and turboderp's packs, Performance live
// in the busy scenario and its history, Usage, and the Chat tab on the phone. Desktop 1440x900 at
// scale 2, the phone 390x844 at scale 3, dark. Into screenshots/readme/*.png (PW_SHOTS_DIR
// overrides it); scripts/readme-shots.sh turns them into the WebP files under docs/gallery/. Run
// with pw-readme.config.ts.
import { test, type Page } from '@playwright/test';
import { IS_MOCK, login } from './helpers';
import { setupMock } from './setup-helpers';

test.skip(!IS_MOCK, 'the README pictures are made from the mock');

const DIR = process.env.PW_SHOTS_DIR ?? 'screenshots/readme';
const DESKTOP = { width: 1440, height: 900 };

// the pictures show the product, not the mock: no "mock data" badges, the release's version; the
// top bar solid (as in the setup guide), so no row of a scrolled page shows through its glass
const CLEAN = '.ch-mock-badge { display: none !important; } .topbar { background: var(--bg) !important; backdrop-filter: none !important; }';
const VERSION = '0.11.1';

async function prepare(page: Page): Promise<void> {
  await page.emulateMedia({ colorScheme: 'dark' });
  await page.addInitScript(() => {
    localStorage.setItem('qse.theme', 'dark');
    localStorage.removeItem('qse.chat.last');
    localStorage.removeItem('tandem.setup.v1');
    sessionStorage.clear();
  });
  await page.route('**/v1/dashboard/system**', async (route) => {
    const res = await route.fetch();
    const body = await res.json();
    body.engine.version = VERSION;
    await route.fulfill({ response: res, json: body });
  });
}

async function openChat(page: Page, hash: string): Promise<void> {
  await prepare(page);
  await login(page, hash, '?chat=mock');
  await page.locator('qse-chat').waitFor();
  await page.addStyleTag({ content: CLEAN });
}

async function openView(page: Page, hash: string): Promise<void> {
  await prepare(page);
  await login(page, hash, `?t=${Date.now()}`);
  await page.addStyleTag({ content: CLEAN });
}

const shot = (page: Page, name: string) => page.screenshot({ path: `${DIR}/${name}.png` });

test.describe('desktop', () => {
  test.use({ viewport: DESKTOP, deviceScaleFactor: 2 });

  test('chat: a page written to the sandbox, open in the preview pane', async ({ page }) => {
    await openChat(page, '#/chat/c_shop');
    await page.getByRole('button', { name: 'Hide the chats' }).click();
    await page.locator('.ch-call-head').first().click();
    await page.getByRole('button', { name: /Open in Preview/ }).first().click();
    await page.locator('.ch-preview-frame iframe').waitFor();
    await page.locator('.ch-dock-btn').click();
    await page.waitForTimeout(1500);
    // the thread from the ask down: the prompt, the tool call and the answer
    await page.locator('.ch-threadwrap').evaluate((wrap) => {
      for (const el of [wrap, ...Array.from(wrap.querySelectorAll<HTMLElement>('*'))]) el.scrollTop = 0;
    });
    await page.waitForTimeout(300);
    await shot(page, 'chat-preview');
  });

  test('setup: the models step', async ({ page, request }) => {
    await setupMock(request);
    await openView(page, '#/setup');
    await page.locator('qse-setup').waitFor();
    await page.waitForTimeout(1500);
    await shot(page, 'setup-models');
  });

  test("setup: Bolt's measured speed table", async ({ page, request }) => {
    await setupMock(request);
    await openView(page, '#/setup/nemotron-3.5-bolt-74b-a9b');
    await page.locator('qse-setup').waitFor();
    await page.waitForTimeout(1500);
    // the table under the sticky memory bar, the weights line above it
    await page.evaluate(() => window.scrollBy(0, 360));
    await page.waitForTimeout(300);
    await shot(page, 'setup-speed');
  });

  test('setup: turboderp packs in the weights step', async ({ page, request }) => {
    await setupMock(request, { communityMs: 300 });
    await openView(page, '#/setup/qwen3.8-flash-next/weights');
    await page.locator('[data-community]').waitFor();
    await page.waitForTimeout(1500);
    await page.locator('[data-community]').evaluate((el) => el.scrollIntoView({ block: 'start' }));
    await page.evaluate(() => window.scrollBy(0, -170));
    await page.waitForTimeout(300);
    await shot(page, 'setup-community');
  });

  test('performance: the live panel in the busy scenario', async ({ page, request }) => {
    await request.post('/__mock/live', { data: { scenario: 'busy' } });
    await openView(page, '#/performance');
    await page.locator('.live-counts').waitFor({ timeout: 10_000 });
    await page.waitForTimeout(30_000);
    await shot(page, 'performance');
    await page.locator('.history').evaluate((el) => el.scrollIntoView({ block: 'start' }));
    await page.evaluate(() => window.scrollBy(0, -60));
    await page.waitForTimeout(800);
    await shot(page, 'performance-history');
    await request.post('/__mock/live', { data: { scenario: 'clear' } });
  });

  test('usage', async ({ page }) => {
    await openView(page, '#/usage');
    await page.waitForTimeout(3000);
    await shot(page, 'usage');
  });
});

test.describe('phone', () => {
  test.use({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 3, isMobile: true, hasTouch: true });

  test('chat on the phone: an answer with code', async ({ page }) => {
    await openChat(page, '#/chat/c_explain');
    await page.locator('.ch-foot').first().waitFor();
    await page.waitForTimeout(800);
    await shot(page, 'phone-chat-answer');
  });

  test('chat on the phone: a run on', async ({ page }) => {
    await openChat(page, '#/chat/c_sync');
    await page.locator('.ch-now').waitFor();
    await page.waitForTimeout(5000);
    await shot(page, 'phone-chat-run');
  });
});
