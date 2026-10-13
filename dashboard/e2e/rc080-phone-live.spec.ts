// Release candidate 0.8.0 on a phone-sized screen, against the deployed engine and chatd over https
// (the Spark's live tier, 2026-10-13: Khaled's iPhone pass and the rc-mobile fix). Spark only:
//   PW_LIVE=1 PW_BASE=https://tandem.khaled.de/dashboard/ PW_SHOTS=<dir> \
//     npx playwright test -c pw-chat-live.config.ts e2e/rc080-phone-live.spec.ts
//
// At 390x844: the Chat tab with the four-slot bar, a message sent from the composer and its answer,
// the composer's controls in view, the Context panel as a sheet with its split, and the More sheet.
import { expect, test, type APIRequestContext } from '@playwright/test';

const SHOTS = process.env.PW_SHOTS ?? 'e2e-live-shots';
const PHONE = { width: 390, height: 844 };

test.skip(!process.env.PW_LIVE, 'the live tier runs on the Spark with PW_LIVE=1');

async function run(request: APIRequestContext, id: string) {
  const r = await request.get(`../v1/chat-app/conversations/${id}`);
  return ((await r.json()) as { run: unknown; messages: { status: string }[] });
}

test('a phone sends a message, reads the answer, and opens the context and More sheets', async ({ browser, request }) => {
  const ctx = await browser.newContext({ viewport: PHONE, isMobile: true, hasTouch: true,
    userAgent: 'Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1' });
  const page = await ctx.newPage();
  await page.goto('./#/chat');
  await page.locator('qse-chat').waitFor();
  await expect(page.locator('.ch-mock-badge')).toHaveCount(0);
  await expect(page.locator('.rail-tab:visible')).toHaveCount(4);
  await page.screenshot({ path: `${SHOTS}/phone-1-chat.png` });
  await page.locator('.ch-new').first().click();
  await page.waitForFunction(() => /#\/chat\/.+/.test(location.hash));
  const id = decodeURIComponent(page.url().split('#/chat/')[1]);
  for (const t of ['model and profile', 'thinking effort', 'tools, skills and MCP servers']) {
    await expect(page.locator(`.ch-pill[title="${t}"]`)).toBeVisible();
  }
  await page.locator('#ch-input').fill('In one sentence: what is a lighthouse for?');
  await page.locator('.ch-send').click();
  await expect.poll(async () => (await run(request, id)).messages.length, { timeout: 20_000 }).toBeGreaterThanOrEqual(2);
  await expect.poll(async () => (await run(request, id)).run, { timeout: 120_000 }).toBeNull();
  const s = await run(request, id);
  expect(s.messages[s.messages.length - 1].status).toBe('done');
  await page.waitForTimeout(600);
  await page.screenshot({ path: `${SHOTS}/phone-2-answer.png` });
  await page.getByRole('button', { name: 'Context panel', exact: true }).click();
  await page.waitForTimeout(600);
  await page.screenshot({ path: `${SHOTS}/phone-3-context.png` });
  await page.getByRole('button', { name: 'Close the context panel' }).click();
  await page.locator('.rail-more').click();
  await expect(page.locator('.more-sheet')).toBeVisible();
  await page.screenshot({ path: `${SHOTS}/phone-4-more.png` });
  await ctx.close();
});
