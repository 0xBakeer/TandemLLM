// The Chat tab without the chat API (VIS-179, QA VIS-180): the probe's answers are routed with
// page.route, so every state shows on the mock tier with no engine and no chatd. Chromium and
// WebKit on a desktop and an iPhone 14. Spark only, never on a laptop:
//   npx playwright test e2e/chat-down.spec.ts --project=chromium-desktop --project=webkit-desktop --project=webkit-phone
import { expect, test, type Page, type Route } from '@playwright/test';
import { CAPABILITIES } from '../src/chat/mock-data';
import { login } from './helpers';

// the PWA's worker would see the chat routes before page.route does
test.use({ serviceWorkers: 'block' });

const phone = (page: Page) => (page.viewportSize()?.width ?? 1440) < 768;
const SHOTS = process.env.PW_SHOTS;

type Answer = { status: number; body?: unknown; type?: string } | 'network';
const DOWN: Answer = { status: 502, body: { error: { type: 'chat_unavailable', message: 'the chat service is not running' } } };

/** /v1/chat-app/* answers `probe()` for the capabilities; once that is 200, the rest of a chatd with
 *  no chats (or `rest()` when it says otherwise). Counts the capability probes. */
async function chatApi(page: Page, probe: () => Answer, rest: () => Answer | null = () => null) {
  const seen = { probes: 0 };
  await page.route('**/v1/chat-app/**', async (route: Route) => {
    const url = new URL(route.request().url());
    const path = url.pathname.replace(/^.*\/v1\/chat-app/, '');
    const a: Answer = path === '/capabilities' ? (seen.probes++, probe()) : (rest() ?? (path.startsWith('/conversations') ? { status: 200, body: { conversations: [], folders: [] } } : path === '/events' ? { status: 200, body: '', type: 'text/event-stream' } : { status: 200, body: [] }));
    if (a === 'network') return route.abort('connectionrefused');
    const body = path === '/capabilities' && a.status === 200 && a.body === undefined ? { ...CAPABILITIES, session: { password_set: false, signed_in: true, sandbox: 'not_installed' } } : a.body;
    await route.fulfill({ status: a.status, contentType: a.type ?? 'application/json', body: typeof body === 'string' ? body : JSON.stringify(body ?? {}) });
  });
  return seen;
}

async function open(page: Page, hash = '#/chat', query = ''): Promise<void> {
  await page.addInitScript(() => localStorage.removeItem('qse.chat.last'));
  await login(page, hash, query);
  await page.locator('qse-chat').waitFor();
}

async function noSideScroll(page: Page): Promise<boolean> {
  return page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1);
}

test('chatd is not running: the panel with the command, no composer, no mock', async ({ page }) => {
  await chatApi(page, () => DOWN);
  await open(page);
  const panel = page.locator('.ch-down');
  await expect(panel).toBeVisible({ timeout: 3_000 });
  await expect(panel).toHaveAttribute('role', 'status');
  await expect(panel).toHaveAttribute('aria-live', 'polite');
  await expect(panel.locator('.ch-down-title')).toHaveText('The chat service is not running.');
  await expect(panel).toContainText('the chat service is not running');           // the server's message
  await expect(panel.locator('.ch-down-cmd code')).toHaveText('~/TandemLLM/bin/tandem chat start');
  await expect(page.locator('qse-chat-composer')).toHaveCount(0);
  await expect(page.locator('.ch-mock-badge')).toHaveCount(0);
  await expect(page.locator('[data-backend="mock"]')).toHaveCount(0);
  expect(await noSideScroll(page)).toBe(true);
  if (phone(page)) {
    const box = (await panel.boundingBox())!;
    expect(box.width).toBeGreaterThanOrEqual(page.viewportSize()!.width - 1);
    expect((await page.locator('.ch-down-check').boundingBox())!.height).toBeGreaterThanOrEqual(44);
    expect((await page.locator('.ch-down-cmd .btn').boundingBox())!.height).toBeGreaterThanOrEqual(44);
    const pre = page.locator('.ch-down-cmd pre');
    expect(await pre.evaluate((el) => getComputedStyle(el).overflowX)).toBe('auto');
  }
  if (SHOTS) await page.screenshot({ path: `${SHOTS}/chat-down-${test.info().project.name}.png`, fullPage: true });
});

test('Check again keeps the panel and says when it checked; chatd coming back makes the tab live', async ({ page }) => {
  let up = false;
  const seen = await chatApi(page, () => (up ? { status: 200 } : DOWN));
  await open(page);
  await expect(page.locator('.ch-down')).toBeVisible();
  await expect(page.locator('.ch-down-when')).toContainText('next in');
  await page.locator('.ch-down-check').click();
  await expect(page.locator('.ch-down-when')).toContainText('Last checked');
  await expect(page.locator('.ch-down')).toBeVisible();
  const before = seen.probes;
  expect(before).toBeGreaterThanOrEqual(2);
  up = true;
  await page.locator('.ch-down-check').click();
  await expect(page.locator('.ch-down')).toHaveCount(0);
  await expect(page.locator('qse-chat-composer')).toBeVisible();
  await expect(page.locator('[data-backend="remote"]')).toHaveCount(1);
  await expect(page.locator('.ch-mock-badge')).toHaveCount(0);
});

test('the panel probes again by itself and recovers without a reload', async ({ page }) => {
  let up = false;
  const seen = await chatApi(page, () => (up ? { status: 200 } : DOWN));
  await open(page);
  await expect(page.locator('.ch-down')).toBeVisible();
  await page.waitForTimeout(3_000);
  expect(seen.probes).toBe(1);                                                        // not before 5 s
  up = true;
  await expect(page.locator('qse-chat-composer')).toBeVisible({ timeout: 8_000 });   // the 5 s probe
});

test('chatd refuses the key', async ({ page }) => {
  await chatApi(page, () => ({ status: 403, body: { error: { type: 'forbidden', message: "chatd answers the engine server's forward only" } } }));
  await open(page);
  await expect(page.locator('.ch-down-title')).toHaveText("chatd is running but does not accept this server's key.");
  await expect(page.locator('.ch-down')).toContainText('Restart both so they read the same secrets.env');
});

test('an engine without chat routes', async ({ page }) => {
  await chatApi(page, () => ({ status: 404, body: '', type: 'text/plain' }));
  await open(page);
  await expect(page.locator('.ch-down-title')).toHaveText('This engine build has no Chat tab server (it needs 0.8.0 or newer).');
});

test('a network error: cannot reach the engine server, Retry, not the mock', async ({ page }) => {
  await chatApi(page, () => 'network');
  await open(page);
  await expect(page.locator('.ch-down-title')).toHaveText('Cannot reach the engine server.');
  await expect(page.locator('.ch-down-check')).toHaveText('Retry');
  await expect(page.locator('[data-backend="mock"]')).toHaveCount(0);
});

test('not an engine: a 200 with a page in a production build', async ({ page }) => {
  await chatApi(page, () => ({ status: 200, body: '<!doctype html><title>x</title>', type: 'text/html' }));
  await open(page);
  await expect(page.locator('.ch-down-title')).toHaveText('This address does not serve the chat API.');
});

test('the mock only on request, its badge on every width', async ({ page }) => {
  await open(page, '#/chat', '?chat=mock');
  const badge = page.locator('.ch-mock-head');
  await expect(badge).toBeVisible();
  await expect(badge).toHaveAttribute('title', 'opened with ?chat=mock');
  await expect(page.locator('[data-backend="mock"]')).toHaveCount(1);
  await expect(page.locator('.ch-empty-note')).toHaveCount(0);
  expect(await noSideScroll(page)).toBe(true);
  if (SHOTS) await page.screenshot({ path: `${SHOTS}/chat-mock-badge-${test.info().project.name}.png` });
});

test('chatd stops during a session: one banner, no toast every 10 s, back by itself', async ({ page }) => {
  let down = false;
  await chatApi(page, () => (down ? DOWN : { status: 200 }), () => (down ? DOWN : null));
  await open(page);
  await expect(page.locator('qse-chat-composer')).toBeVisible();
  down = true;
  await expect(page.locator('.ch-down-banner')).toHaveText('The chat service stopped. Your chats are safe. Retrying.', { timeout: 15_000 });   // the 10 s list poll
  await expect(page.locator('.ch-down-banner')).toHaveCount(1);
  await page.waitForTimeout(11_000);
  await expect(page.locator('.ch-toast')).toHaveCount(0);
  await expect(page.locator('.ch-down-banner')).toHaveCount(1);
  if (SHOTS) await page.screenshot({ path: `${SHOTS}/chat-lost-${test.info().project.name}.png` });
  down = false;
  await expect(page.locator('.ch-down-banner')).toHaveCount(0, { timeout: 25_000 });
});
