// Release candidate 0.8.0 on the Spark (the box's live checks of 2026-10-12): the deployed engine
// with chatd behind its /v1/chat-app, over https. Spark only, never on a laptop:
//   PW_LIVE=1 PW_BASE=https://tandem.khaled.de/dashboard/ PW_SHOTS=<dir> \
//     npx playwright test -c pw-chat-live.config.ts e2e/rc080-live.spec.ts
//
// 1. the chats of the :4180 store are in the rail;
// 2. the Context panel after two turns: the split by part from usage.tandem (system prompt and
//    tool schemas counted) and the cache the second turn came from;
// 3. two chats stream at once, and the engine reports two slots decoding together;
// 4. the Setup Guide shows both models, Flash-Next's windows per pack and the 5.05 badge;
// 6. an image sent from the Chat tab as an attachment reaches the model (VIS-97 on Flash-Next);
// 5. the engine reports three slots (the Parallel tab shows them while requests run, test 3), and
//    the system view names the Flash-Next pack.
import { expect, test, type APIRequestContext, type Browser, type Page } from '@playwright/test';

const SHOTS = process.env.PW_SHOTS ?? 'e2e-live-shots';
const DESKTOP = { width: 1440, height: 900 };
const LONG = 'Write a story of about 900 words about a lighthouse keeper who learns to repair clocks. Plain prose in paragraphs: no headings, no lists, no Markdown.';

test.skip(!process.env.PW_LIVE, 'the live tier runs on the Spark with PW_LIVE=1');

interface Snap {
  id: string;
  run: { id: string; phase: string } | null;
  messages: { id: string; role: string; status: string; parts: { type: string; text?: string }[] }[];
  context?: { parts: { kind: string; tokens: number }[]; cache: { warm: number; cold: number; source: string }; split?: Record<string, number> | null } | null;
}

async function snap(request: APIRequestContext, id: string): Promise<Snap> {
  const r = await request.get(`../v1/chat-app/conversations/${id}`);
  expect(r.ok()).toBeTruthy();
  return (await r.json()) as Snap;
}

const textLen = (s: Snap, mid: string) =>
  (s.messages.find((m) => m.id === mid)?.parts ?? []).map((p) => p.text ?? '').join('').length;

async function openChat(browser: Browser, hash: string) {
  const ctx = await browser.newContext({ viewport: DESKTOP });
  const page = await ctx.newPage();
  await page.goto(`./${hash}`);
  await page.locator('qse-chat').waitFor();
  await expect(page.locator('.ch-mock-badge')).toHaveCount(0);
  return { ctx, page };
}

async function start(page: Page, request: APIRequestContext, text: string): Promise<{ id: string; mid: string }> {
  await page.locator('.ch-new').first().click();
  await page.waitForFunction(() => /#\/chat\/.+/.test(location.hash));
  const id = decodeURIComponent(page.url().split('#/chat/')[1]);
  await page.locator('#ch-input').fill(text);
  await page.locator('.ch-send').click();
  await expect.poll(async () => (await snap(request, id)).messages.length, { timeout: 20_000 }).toBeGreaterThanOrEqual(2);
  const s = await snap(request, id);
  return { id, mid: s.messages[s.messages.length - 1].id };
}

test('the chats of the :4180 store are in the rail', async ({ browser, request }) => {
  const r = await request.get('../v1/chat-app/conversations');
  const list = ((await r.json()) as { conversations: { title: string | null }[] }).conversations;
  expect(list.length).toBeGreaterThanOrEqual(13);
  const a = await openChat(browser, '#/chat');
  await expect(a.page.locator('.ch-rail')).toContainText('Lighthouse Log Confirmation');
  await a.page.screenshot({ path: `${SHOTS}/rc080-1-old-chats.png` });
  await a.ctx.close();
});

test('the Context panel: the split by part and the cache the second turn came from', async ({ browser, request }) => {
  const a = await openChat(browser, '#/chat');
  const { id } = await start(a.page, request, 'In two sentences: what does a lighthouse keeper do?');
  await expect.poll(async () => (await snap(request, id)).run, { timeout: 180_000 }).toBeNull();
  await a.page.locator('#ch-input').fill('And in one sentence: why are lighthouses painted in stripes?');
  await a.page.locator('.ch-send').click();
  await expect.poll(async () => (await snap(request, id)).messages.length, { timeout: 20_000 }).toBe(4);
  await expect.poll(async () => (await snap(request, id)).run, { timeout: 180_000 }).toBeNull();
  const ctx = (await snap(request, id)).context!;
  const parts = Object.fromEntries(ctx.parts.map((p) => [p.kind, p.tokens]));
  console.log(`[rc080] split ${JSON.stringify(ctx.split)}, parts ${JSON.stringify(parts)}, cache ${JSON.stringify(ctx.cache)}`);
  expect(ctx.split).toBeTruthy();
  expect(parts.system).toBeGreaterThan(0);
  expect(parts.tools).toBeGreaterThan(0);
  expect(ctx.cache.warm).toBeGreaterThan(0);
  await expect(a.page.locator('.ch-rib-cache')).toContainText('warm');
  await a.page.waitForTimeout(500);
  await a.page.screenshot({ path: `${SHOTS}/rc080-2-context-panel.png` });
  await a.ctx.close();
});

test('two chats stream at once and the engine decodes them together', async ({ browser, request }) => {
  const cctx = await browser.newContext({ viewport: DESKTOP });
  const c = { ctx: cctx, page: await cctx.newPage() };
  await c.page.goto('./#/performance?tab=parallel');
  const a = await openChat(browser, '#/chat');
  const b = await openChat(browser, '#/chat');
  const one = await start(a.page, request, LONG);
  const two = await start(b.page, request, LONG.replace('lighthouse keeper', 'baker'));
  let together = 0;
  await expect
    .poll(
      async () => {
        const p = (await (await request.get('../v1/dashboard/parallel')).json()) as { engine_slots?: { decoding: number } };
        together = Math.max(together, p.engine_slots?.decoding ?? 0);
        return together;
      },
      { timeout: 120_000, intervals: [500] },
    )
    .toBeGreaterThanOrEqual(2);
  const l1 = [textLen(await snap(request, one.id), one.mid), textLen(await snap(request, two.id), two.mid)];
  await a.page.waitForTimeout(3000);
  const l2 = [textLen(await snap(request, one.id), one.mid), textLen(await snap(request, two.id), two.mid)];
  expect(l2[0]).toBeGreaterThan(l1[0]);
  expect(l2[1]).toBeGreaterThan(l1[1]);
  const health = (await (await request.get('../health')).json()) as Record<string, unknown>;
  console.log(`[rc080] decoding together: ${together}; /health parallel ${JSON.stringify(health.parallel)}; +${l2[0] - l1[0]} and +${l2[1] - l1[1]} chars in 3 s`);
  await a.page.screenshot({ path: `${SHOTS}/rc080-3-two-chats-a.png` });
  await b.page.screenshot({ path: `${SHOTS}/rc080-3-two-chats-b.png` });
  // the Parallel tab shows the engine's slots while it has requests in flight
  await expect(c.page.locator('.par-slots')).toHaveAttribute('data-slots', '3', { timeout: 30_000 });
  await c.page.screenshot({ path: `${SHOTS}/rc080-3-parallel-slots.png` });
  await c.ctx.close();
  await a.page.locator('.ch-stop').click();
  await b.page.locator('.ch-stop').click();
  await expect.poll(async () => (await snap(request, one.id)).run, { timeout: 30_000 }).toBeNull();
  await expect.poll(async () => (await snap(request, two.id)).run, { timeout: 30_000 }).toBeNull();
  await a.ctx.close();
  await b.ctx.close();
});

test('the Setup Guide shows both models, the Flash-Next windows and the 5.05 badge', async ({ browser }) => {
  const ctx = await browser.newContext({ viewport: DESKTOP });
  const page = await ctx.newPage();
  await page.goto('./#/setup');
  await expect(page.locator('body')).toContainText('Qwen3.8-Flash-Next', { timeout: 20_000 });
  await expect(page.locator('body')).toContainText('Qwen3.8-27B');
  await page.screenshot({ path: `${SHOTS}/rc080-4-setup-models.png` });
  await page.goto('./#/setup/qwen3.8-flash-next/weights');
  await page.locator('[data-profile-card="flashnext-5.05"]').waitFor({ timeout: 20_000 });
  for (const id of ['2.05', '3.05', '4.05', '5.05', '6.05']) await expect(page.locator(`[data-profile-card="flashnext-${id}"]`)).toHaveCount(1);
  await expect(page.locator('[data-profile-card="flashnext-5.05"]')).toContainText('does not load yet');
  const body = await page.locator('body').innerText();
  console.log(`[rc080] setup weights: 262,144 ${body.includes('262,144')}, 212,992 ${body.includes('212,992')}`);
  await page.screenshot({ path: `${SHOTS}/rc080-4-setup-flashnext-weights.png`, fullPage: true });
  await ctx.close();
});

test('the engine has three slots and the system view names the pack', async ({ browser, request }) => {
  const p = (await (await request.get('../v1/dashboard/parallel')).json()) as { engine_slots?: { n: number } };
  expect(p.engine_slots?.n).toBe(3);
  const ctx = await browser.newContext({ viewport: DESKTOP });
  const page = await ctx.newPage();
  await page.goto('./#/system');
  await expect(page.locator('body')).toContainText('flashnext-3.05', { timeout: 20_000 });
  await page.screenshot({ path: `${SHOTS}/rc080-5-system.png`, fullPage: true });
  await ctx.close();
});

test('an image sent from the Chat tab as an attachment reaches the model', async ({ browser, request }) => {
  const image = process.env.PW_IMAGE;
  test.skip(!image, 'PW_IMAGE: a PNG to attach');
  const a = await openChat(browser, '#/chat');
  await a.page.locator('.ch-new').first().click();
  await a.page.waitForFunction(() => /#\/chat\/.+/.test(location.hash));
  const id = decodeURIComponent(a.page.url().split('#/chat/')[1]);
  await a.page.locator('.ch-attach-btn input[type=file]').setInputFiles(image!);
  await expect(a.page.locator('.ch-attachments-draft .ch-chip')).toHaveCount(1, { timeout: 15_000 });
  await a.page.locator('#ch-input').fill('What colours are the two shapes in the picture? One short sentence.');
  await a.page.locator('.ch-send').click();
  await expect.poll(async () => (await snap(request, id)).messages.length, { timeout: 30_000 }).toBe(2);
  await expect.poll(async () => (await snap(request, id)).run, { timeout: 180_000 }).toBeNull();
  const s = await snap(request, id);
  const answer = (s.messages[1].parts ?? []).filter((p) => p.type === 'text').map((p) => p.text ?? '').join('');
  const stats = (s.messages[1] as unknown as { stats?: { prompt_tokens?: number } }).stats;
  console.log(`[rc080] attachment: prompt ${stats?.prompt_tokens} tokens, status ${s.messages[1].status}, answer ${JSON.stringify(answer.slice(0, 120))}`);
  expect(s.messages[1].status).toBe('done');
  expect(stats?.prompt_tokens ?? 0).toBeGreaterThan(140);
  expect(answer.toLowerCase()).toMatch(/red|blue/);
  await a.page.waitForTimeout(500);
  await a.page.screenshot({ path: `${SHOTS}/rc080-6-image-attachment.png` });
  await a.ctx.close();
});
