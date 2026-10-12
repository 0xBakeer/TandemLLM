// The Chat tab against a real chatd and a real engine (VIS-101, SRV-345; the phase 1 demo of the
// Memo note "Chat 10"). Spark only, never on a laptop:
//   PW_LIVE=1 PW_BASE=http://127.0.0.1:4180/dashboard/ npx playwright test -c pw-chat-live.config.ts
// PW_BASE is a server that answers /v1/chat-app/* (the engine with the forward, or chatd's front).
// PW_SHOTS: where the screenshots go (default e2e-live-shots/).
//
// 1. a long answer started in one browser keeps generating after that browser closes;
// 2. a phone (390 px) attaches mid-run, shows the current token within 2 s, keeps streaming;
// 3. Stop ends it everywhere with its text kept; 4. a queued message with Send now steers it;
// 5. a reload resumes the same answer with no text twice.
import { expect, test, type APIRequestContext, type Browser, type Page } from '@playwright/test';

const SHOTS = process.env.PW_SHOTS ?? 'e2e-live-shots';
const DESKTOP = { width: 1440, height: 900 };
const PHONE = { width: 390, height: 844 };
const LONG = 'Write a story of about 900 words about a lighthouse keeper who learns to repair clocks. Plain prose in paragraphs: no headings, no lists, no Markdown.';

test.skip(!process.env.PW_LIVE, 'the live tier runs on the Spark with PW_LIVE=1');

const alnum = (s: string) => s.replace(/[^A-Za-z0-9]/g, '');

interface Snap {
  id: string;
  seq: string;
  run: { id: string; phase: string } | null;
  messages: { id: string; role: string; status: string; parts: { type: string; text?: string }[] }[];
}

async function snap(request: APIRequestContext, id: string): Promise<Snap> {
  const r = await request.get(`../v1/chat-app/conversations/${id}`);
  expect(r.ok()).toBeTruthy();
  return (await r.json()) as Snap;
}

/** the answer's text as the server has it: thinking and text parts, letters and digits only */
function serverText(s: Snap, mid: string): string {
  const m = s.messages.find((x) => x.id === mid);
  return alnum((m?.parts ?? []).filter((p) => p.type === 'thinking' || p.type === 'text').map((p) => p.text ?? '').join(''));
}

/** the same, as the page shows it */
async function pageText(page: Page, mid: string): Promise<string> {
  return page.evaluate((id) => {
    const a = document.querySelector(`article[data-mid="${id}"]`);
    if (!a) return '';
    const parts = Array.from(a.querySelectorAll('[data-part="thinking"] .ch-think-body, [data-part="text"]'));
    return parts.map((p) => p.textContent ?? '').join('').replace(/[^A-Za-z0-9]/g, '');
  }, mid);
}

async function openChat(browser: Browser, hash: string, viewport = DESKTOP) {
  const ctx = await browser.newContext({ viewport, ...(viewport === PHONE ? { isMobile: true, hasTouch: true } : {}) });
  const page = await ctx.newPage();
  await page.goto(`./${hash}`);
  await page.locator('qse-chat').waitFor();
  await expect(page.locator('.ch-mock-badge')).toHaveCount(0);
  return { ctx, page };
}

/** a new chat with `text` sent from the page; returns the conversation and answer ids */
async function start(page: Page, request: APIRequestContext, text: string): Promise<{ id: string; mid: string }> {
  await page.locator('.ch-new').first().click();
  await page.waitForFunction(() => /#\/chat\/.+/.test(location.hash));
  const id = decodeURIComponent(page.url().split('#/chat/')[1]);
  await page.locator('#ch-input').fill(text);
  await page.locator('.ch-send').click();
  await expect.poll(async () => (await snap(request, id)).messages.length, { timeout: 20_000 }).toBe(2);
  const s = await snap(request, id);
  return { id, mid: s.messages[1].id };
}

async function waitText(request: APIRequestContext, id: string, mid: string, n: number): Promise<void> {
  await expect.poll(async () => serverText(await snap(request, id), mid).length, { timeout: 120_000 }).toBeGreaterThan(n);
}

test('a long answer outlives its browser; a phone attaches mid-run and keeps streaming', async ({ browser, request }) => {
  const a = await openChat(browser, '#/chat');
  const { id, mid } = await start(a.page, request, LONG);
  await waitText(request, id, mid, 300);
  await a.page.screenshot({ path: `${SHOTS}/live-1-desktop-running.png` });
  await a.ctx.close();
  // no browser at all now: the run goes on in chatd
  const t0 = serverText(await snap(request, id), mid).length;
  await new Promise((r) => setTimeout(r, 4000));
  const s1 = await snap(request, id);
  expect(s1.run).not.toBeNull();
  expect(serverText(s1, mid).length).toBeGreaterThan(t0 + 20);

  // the phone opens the chat mid-run
  const b = await openChat(browser, `#/chat/${encodeURIComponent(id)}`, PHONE);
  const atAttach = serverText(await snap(request, id), mid);
  const tail = atAttach.slice(-40);
  const t1 = Date.now();
  await expect.poll(async () => (await pageText(b.page, mid)).includes(tail), { timeout: 2_000, intervals: [100] }).toBe(true);
  const shownIn = Date.now() - t1;
  await expect(b.page.locator('.ch-now')).toBeVisible();
  await b.page.screenshot({ path: `${SHOTS}/live-2-phone-attached.png` });
  const p1 = (await pageText(b.page, mid)).length;
  await b.page.waitForTimeout(3000);
  const p2 = (await pageText(b.page, mid)).length;
  expect(p2).toBeGreaterThan(p1);
  await b.page.screenshot({ path: `${SHOTS}/live-3-phone-streaming.png` });
  console.log(`[live] attached at ${atAttach.length} chars, the page had them within ${shownIn} ms; +${p2 - p1} chars in 3 s`);
  // to the end: the page and the server agree, letter for letter
  await expect.poll(async () => (await snap(request, id)).run, { timeout: 240_000 }).toBeNull();
  await expect(b.page.locator('.ch-now')).toHaveCount(0, { timeout: 10_000 });
  const final = serverText(await snap(request, id), mid);
  await expect.poll(async () => pageText(b.page, mid), { timeout: 5_000 }).toBe(final);
  await b.page.screenshot({ path: `${SHOTS}/live-4-phone-done.png`, fullPage: false });
  await b.ctx.close();
});

test('Stop ends the answer and keeps its text', async ({ browser, request }) => {
  const a = await openChat(browser, '#/chat');
  const { id, mid } = await start(a.page, request, LONG);
  await waitText(request, id, mid, 200);
  await a.page.locator('.ch-stop').click();
  await expect(a.page.locator(`article[data-mid="${mid}"]`)).toHaveAttribute('data-status', 'stopped', { timeout: 10_000 });
  const s = await snap(request, id);
  expect(s.run).toBeNull();
  expect(s.messages[1].status).toBe('stopped');
  expect(serverText(s, mid).length).toBeGreaterThan(200);
  await a.page.waitForTimeout(400);
  await a.page.screenshot({ path: `${SHOTS}/live-5-stopped.png` });
  await a.ctx.close();
});

test('a queued message steers: Send now stops the answer and sends it', async ({ browser, request }) => {
  const a = await openChat(browser, '#/chat');
  const { id, mid } = await start(a.page, request, LONG);
  await waitText(request, id, mid, 150);
  await a.page.locator('#ch-input').fill('Stop the story. In one short sentence: what is the capital of France?');
  await a.page.keyboard.press('Enter');
  await expect(a.page.locator('.ch-queued')).toHaveCount(1, { timeout: 10_000 });
  await a.page.screenshot({ path: `${SHOTS}/live-6-queued.png` });
  await a.page.locator('.ch-queued button', { hasText: 'Send now' }).click();
  await expect(a.page.locator(`article[data-mid="${mid}"]`)).toHaveAttribute('data-status', 'stopped', { timeout: 15_000 });
  await expect.poll(async () => (await snap(request, id)).messages.length, { timeout: 20_000 }).toBe(4);
  const s = await snap(request, id);
  expect(s.messages[2].role).toBe('user');
  expect(s.messages[2].parts[0].text).toContain('capital of France');
  await expect.poll(async () => (await snap(request, id)).run, { timeout: 180_000 }).toBeNull();
  await expect(a.page.locator('.ch-queued')).toHaveCount(0);
  await a.page.waitForTimeout(500);
  await a.page.screenshot({ path: `${SHOTS}/live-7-steered.png` });
  await a.ctx.close();
});

test('a reload resumes the same answer with nothing twice', async ({ browser, request }) => {
  const a = await openChat(browser, '#/chat');
  const { id, mid } = await start(a.page, request, LONG);
  await waitText(request, id, mid, 250);
  const before = await pageText(a.page, mid);
  await a.page.reload();
  await a.page.locator(`article[data-mid="${mid}"]`).waitFor({ timeout: 10_000 });
  await expect.poll(async () => (await pageText(a.page, mid)).startsWith(before.slice(0, 200)), { timeout: 5_000 }).toBe(true);
  const r1 = (await pageText(a.page, mid)).length;
  await a.page.waitForTimeout(2500);
  expect((await pageText(a.page, mid)).length).toBeGreaterThan(r1);
  await a.page.screenshot({ path: `${SHOTS}/live-8-reloaded.png` });
  // stop it to keep the box free, then the page must equal the server exactly
  await a.page.locator('.ch-stop').click();
  await expect.poll(async () => (await snap(request, id)).run, { timeout: 15_000 }).toBeNull();
  const final = serverText(await snap(request, id), mid);
  await expect.poll(async () => pageText(a.page, mid), { timeout: 5_000 }).toBe(final);
  await a.ctx.close();
});
