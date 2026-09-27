// VIS-23 — the Live panel: per-request and all-together speed at one second.
import { expect, test, type APIRequestContext, type Page } from '@playwright/test';
import { IS_FAKE, IS_MOCK, login, noHorizontalOverflow, setMode } from './helpers';

const isPhone = (page: Page) => (page.viewportSize()?.width ?? 1440) <= 768;

async function busy(request: APIRequestContext, scenario: 'busy' | 'clear' = 'busy') {
  const res = await request.post('/__mock/live', { data: { scenario } });
  expect(res.ok()).toBeTruthy();
}

/** Start a streamed chat in the background and return a promise of its final `timings`. */
function streamChat(request: APIRequestContext, content: string, maxTokens = 400): Promise<Record<string, number>> {
  return request
    .post('/v1/chat/completions', {
      data: { messages: [{ role: 'user', content }], stream: true, max_tokens: maxTokens, chat_template_kwargs: { enable_thinking: false } },
      headers: { 'X-Requested-With': 'qse-dashboard' },
      timeout: 120_000,
    })
    .then(async (res) => {
      const text = await res.text();
      const m = [...text.matchAll(/^data: (\{.*\})$/gm)].map((x) => JSON.parse(x[1]) as { timings?: Record<string, number> }).find((c) => c.timings);
      return m?.timings ?? {};
    });
}

test.describe('live panel', () => {
  test.beforeEach(async ({ request }) => {
    if (IS_MOCK) {
      await setMode(request, 'ok');
      await busy(request, 'clear');
    }
  });

  test('sits at the top of Performance, streams at one second, and the /metrics strip stays under it', async ({ page }) => {
    await login(page, '#/performance');
    const live = page.locator('.live');
    await expect(live).toBeVisible();
    await expect(page.locator('.view-performance > qse-live-panel')).toHaveCount(1);
    await expect(live.locator('#live-decode .live-fig-label')).toHaveText('decode, all requests');
    await expect(live.locator('#live-prefill .live-fig-label')).toHaveText('prefill');
    await expect(live.locator('qse-spark')).toHaveCount(2);
    await expect(live.locator('.live-counts .live-count')).toHaveCount(8);
    await expect(live.locator('.live-state')).toContainText(/streaming|connecting/);
    await expect.poll(async () => live.locator('.live-state').textContent(), { timeout: 10_000 }).toMatch(/streaming · 1 s/);
    // the three 5-minute figures from /metrics, in their own strip under the panel
    const strip = page.locator('.live-metrics');
    await expect(strip.locator('.stat')).toHaveCount(3);
    await expect(strip.locator('.stat').nth(0)).toContainText('time to first token');
    await expect(strip.locator('.stat').nth(1)).toContainText('tokens per block');
    await expect(strip.locator('.stat').nth(2)).toContainText('draft acceptance');
    await expect(strip.locator('.live-metrics-sub')).toContainText('/metrics');
  });

  test('idle: the figures say so and the list is empty', { tag: '@mock' }, async ({ page }) => {
    test.skip(!IS_MOCK, 'a real engine may be busy');
    await login(page, '#/performance');
    const live = page.locator('.live');
    await expect(live.locator('.live-state')).toContainText('streaming', { timeout: 10_000 });
    const decode = live.locator('#live-decode');
    // the mock engine starts a request every ~15 s on its own; either state is a valid reading
    await expect(decode.locator('.live-fig-sub')).toContainText(/nothing decoding|decoding/);
    await expect(live.locator('.state-empty, .live-list')).toHaveCount(1);
  });

  test('a decoding request has a live row whose tokens grow, then keeps its final numbers', async ({ page, request }) => {
    await login(page, '#/performance');
    const live = page.locator('.live');
    await expect(live.locator('.live-state')).toContainText('streaming', { timeout: 10_000 });
    // the mock streams ~60 tok/s, the fake engine 10 tok/s on FAKE_SLOW, the real engine ~43 tok/s: a few seconds each
    const done = streamChat(request, IS_MOCK ? 'Explain tokens per block in a paragraph.' : IS_FAKE ? 'FAKE_SLOW explain the engine in detail please' : 'Explain how speculative decoding verifies a block of drafted tokens, in about 300 words.', IS_MOCK ? 120 : IS_FAKE ? 80 : 400);
    const sel = isPhone(page) ? '.live-card.is-decode' : '.live-row.is-decode';
    const row = live.locator(sel).first();
    await expect(row).toBeVisible({ timeout: 15_000 });
    await expect(row.locator('.phase')).toContainText('decoding');
    const tokens = async () => Number((await row.locator(isPhone(page) ? '.live-card-grid dd b' : 'td:nth-child(5) b').first().textContent())?.replace(/,/g, ''));
    const t1 = await tokens();
    await expect.poll(tokens, { timeout: 10_000 }).toBeGreaterThan(t1);
    await expect(row).toContainText(/tok\/s|\d+\.\d \/ \d+\.\d/);
    const timings = await done;
    const finished = live.locator(isPhone(page) ? '.live-card.is-done' : '.live-row.is-done').first();
    await expect(finished).toBeVisible({ timeout: 10_000 });
    await expect(finished.locator('.phase')).toContainText(/stop|length/);
    await expect(finished).toContainText(/s ago/);
    if (timings.predicted_n) {
      // the row's final tokens (the first bold figure) are the response's own predicted_n
      await expect(finished.locator('b').first()).toHaveText(String(timings.predicted_n).replace(/\B(?=(\d{3})+(?!\d))/g, ','));
    }
  });

  test('busy: counts, phases and the all-together sum', { tag: '@mock' }, async ({ page, request }) => {
    test.skip(!IS_MOCK, 'needs the mock scenario');
    await busy(request);
    await login(page, '#/performance');
    const live = page.locator('.live');
    const counts = live.locator('.live-counts');
    await expect(counts).toContainText('in flight', { timeout: 10_000 });
    // the mock engine keeps serving on its own, so the scenario's counts are a floor, not an equality
    const count = async (label: string) => Number(await counts.locator('.live-count', { hasText: label }).locator('dd').textContent());
    expect(await count('in flight')).toBeGreaterThanOrEqual(3);
    expect(await count('queued')).toBeGreaterThanOrEqual(1);
    expect(await count('prefilling')).toBeGreaterThanOrEqual(1);
    expect(await count('decoding')).toBeGreaterThanOrEqual(1);
    expect(await count('done, last minute')).toBeGreaterThanOrEqual(2);
    const items = live.locator(isPhone(page) ? '.live-card' : '.live-row');
    await expect.poll(() => items.count()).toBeGreaterThanOrEqual(5);
    // decoding first, then prefilling, queued, then the finished ones: the order never goes back
    const order = { decoding: 0, prefilling: 1, queued: 2 } as Record<string, number>;
    const words = (await items.locator('.phase').allTextContents()).map((w) => w.replace(/^[^a-zA-Z]+/, '').trim()); // drop the glyph
    const ranks = words.map((w) => order[w] ?? 3);
    for (let i = 1; i < ranks.length; i++) expect(ranks[i]).toBeGreaterThanOrEqual(ranks[i - 1]);
    expect(words).toEqual(expect.arrayContaining(['decoding', 'prefilling', 'queued', 'stop', 'error']));
    await expect(items.filter({ hasText: 'error' }).first().locator('.phase')).toHaveClass(/phase-failed/);
    // the prefill figure says it is prefilling, with the prompt length
    await expect(live.locator('#live-prefill .live-fig-sub')).toContainText(/prefilling 8,192 tokens/);
    // after a few samples the decode figure is a number near the decoding row's 2 s rate (43 tok/s)
    const fig = async () => Number(await live.locator('#live-decode .live-fig-num').textContent());
    await expect.poll(fig, { timeout: 10_000 }).toBeGreaterThan(20);
    expect(await fig()).toBeLessThan(80);
    await busy(request, 'clear');
  });

  test('phone: figures stack, rows are cards, nothing scrolls sideways', async ({ page, request }) => {
    test.skip(!isPhone(page), 'the phone project');
    if (IS_MOCK) await busy(request);
    await login(page, '#/performance');
    await expect(page.locator('.live-counts')).toContainText('in flight', { timeout: 10_000 });
    await expect(page.locator('.live-table')).toBeHidden();
    if (IS_MOCK) await expect(page.locator('.live-card').first()).toBeVisible();
    expect(await noHorizontalOverflow(page)).toBe(true);
    const box = await page.locator('#live-decode').boundingBox();
    const box2 = await page.locator('#live-prefill').boundingBox();
    expect(box2!.y).toBeGreaterThan(box!.y + box!.height - 1);
    if (IS_MOCK) await busy(request, 'clear');
  });

  test('the stream drops and comes back: the numbers stay meanwhile', { tag: '@mock' }, async ({ page, request }) => {
    test.skip(!IS_MOCK, 'needs the mock drop switch');
    await busy(request);
    await setMode(request, 'drop:2');
    await login(page, '#/performance');
    const live = page.locator('.live');
    await expect(live.locator('.live-counts')).toContainText('in flight', { timeout: 10_000 });
    await expect(live.locator('.live-state')).toContainText(/reconnecting|connecting/, { timeout: 8000 });
    await expect(live.locator('.live-counts .live-count').first().locator('dd')).toHaveText('3');
    await expect(live.locator('.live-state')).toContainText('streaming', { timeout: 10_000 });
    await setMode(request, 'ok');
    await busy(request, 'clear');
  });

  test('a lost session sends the app back to the token screen', { tag: '@mock' }, async ({ page, request }) => {
    test.skip(!IS_MOCK, 'needs the mock switch');
    await login(page, '#/performance');
    await expect(page.locator('.live-state')).toContainText('streaming', { timeout: 10_000 });
    await setMode(request, '401');
    await page.reload();
    await expect(page.locator('#token')).toBeVisible({ timeout: 10_000 });
    await setMode(request, 'ok');
  });

  test('reduced motion: nothing pulses', { tag: '@mock' }, async ({ page, request }) => {
    test.skip(!IS_MOCK, 'needs the mock scenario');
    await busy(request);
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await login(page, '#/performance');
    const glyph = page.locator('.phase-decode .phase-glyph:visible').first(); // the table's copy is hidden on a phone
    await expect(glyph).toBeVisible({ timeout: 10_000 });
    expect(await glyph.evaluate((el) => getComputedStyle(el).animationName)).toBe('none');
    await busy(request, 'clear');
  });
});
