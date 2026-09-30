// VIS-40 / VIS-41: the Live panel's prefill and decode figures show the speed NOW with each
// request's warm-up left out (decode from token 50, prefill after its first chunk), "—" when
// nothing runs, the tile equal to the banner, and the finished rows keep their end-to-end numbers.
// The mock plays each case (`POST /__mock/live {scenario: "rates", case}`); the fake engine plays
// a real chunked prefill with a slow first chunk (FAKE_WARM_PREFILL).
import { expect, test, type APIRequestContext, type Page } from '@playwright/test';
import { IS_FAKE, IS_MOCK, login, noHorizontalOverflow, setMode } from './helpers';

const isPhone = (page: Page) => (page.viewportSize()?.width ?? 1440) <= 768;

async function play(request: APIRequestContext, kind: string, at = 0, extra: Record<string, unknown> = {}) {
  expect((await request.post('/__mock/live', { data: { scenario: 'clear', simulator: false, liveRates: true } })).ok()).toBeTruthy();
  expect((await request.post('/__mock/live', { data: { scenario: 'rates', case: kind, at, ...extra } })).ok()).toBeTruthy();
}

/** The tile and the banner, read in one go so both come from the same event. */
async function read(page: Page) {
  return page.evaluate(() => {
    const q = (s: string) => document.querySelector(s)?.textContent?.trim() ?? '';
    return {
      prefill: q('#live-prefill .live-fig-num'),
      prefillSub: q('#live-prefill .live-fig-sub'),
      prefillCap: q('#live-prefill .live-fig-cap'),
      prefillFoot: q('#live-prefill .live-fig-foot'),
      decode: q('#live-decode .live-fig-num'),
      decodeSub: q('#live-decode .live-fig-sub'),
      decodeCap: q('#live-decode .live-fig-cap'),
      banner: q('.now-numbers'),
    };
  });
}

async function open(page: Page) {
  await login(page, '#/performance');
  await expect(page.locator('.live-state')).toContainText('streaming', { timeout: 10_000 });
}

test.describe('live rates without the warm-up (mock)', () => {
  test.skip(!IS_MOCK, 'the mock plays each case');
  test.afterEach(async ({ request }) => {
    await request.post('/__mock/live', { data: { scenario: 'clear', simulator: true, liveRates: true } });
    await setMode(request, 'ok');
  });

  test('1: the running prefill drives the headline, never the last finished request', { tag: '@mock' }, async ({ page, request }) => {
    await play(request, 'prefill-now', 19_000);
    await open(page);
    await expect(page.locator('#live-prefill .live-fig-num')).toHaveText('743', { timeout: 10_000 });
    await expect(page.locator('.now-numbers')).toContainText('743 tok/s');
    await expect(page.locator('#live-prefill .live-fig-cap')).toHaveText('after the first chunk');
    await expect(page.locator('#live-decode .live-fig-cap')).toHaveText('from token 50');
    for (let i = 0; i < 8; i++) {
      expect((await read(page)).prefill).not.toBe('433');
      await page.waitForTimeout(250);
    }
  });

  test('2: the tile equals the banner on 20 consecutive events', { tag: '@mock' }, async ({ page, request }) => {
    await play(request, 'prefill-now', 19_000);
    await open(page);
    await expect(page.locator('#live-prefill .live-fig-num')).toHaveText('743', { timeout: 10_000 });
    for (let i = 0; i < 20; i++) {
      const s = await read(page);
      expect(s.banner).toContain(`${s.prefill} tok/s`);
      await page.waitForTimeout(250);
    }
  });

  test('3: the headline is the now-rate, not an average', { tag: '@mock' }, async ({ page, request }) => {
    await play(request, 'now-rate', 200 + 3000 + 10 * (1024 / 900) * 1000 + 6 * (1024 / 600) * 1000);
    await open(page);
    await expect.poll(async () => Number((await read(page)).prefill.replace(/,/g, '')), { timeout: 10_000 }).toBeGreaterThan(590);
    expect(Number((await read(page)).prefill)).toBeLessThan(610);
  });

  test('4: the warm-up chunk is left out: "—" while it runs, then 731 from chunks 2 and 3', { tag: '@mock' }, async ({ page, request }) => {
    await play(request, 'warm-chunk', 200 + 1000);
    await open(page);
    const s = await read(page);
    expect(s.prefill).toBe('—');
    expect(s.prefillSub).toMatch(/^warming up · /);
    expect(s.banner).toContain('warming up');
    await expect(page.locator('#live-prefill .live-fig-num')).toHaveText('731', { timeout: 12_000 });
    await expect(page.locator('.now-numbers')).toContainText('731 tok/s');
  });

  test('5: a prefill of one chunk or less never shows a live rate', { tag: '@mock' }, async ({ page, request }) => {
    await play(request, 'one-chunk', 0);
    await open(page);
    await expect(page.locator('#live-prefill .live-fig-sub')).toHaveText(/^warming up · prefilling 900 tokens/, { timeout: 4000 });
    let looked = 0;
    for (let i = 0; i < 8; i++) {
      const s = await read(page);
      if (!/prefilling/.test(s.prefillSub)) break; // its prefill is over
      expect(s.prefill).toBe('—');
      looked++;
      await page.waitForTimeout(250);
    }
    expect(looked).toBeGreaterThan(1);
  });

  test('6: decode warms up to token 50, then shows its rate', { tag: '@mock' }, async ({ page, request }) => {
    await play(request, 'decode-warmup', 1200);
    await open(page);
    await expect(page.locator('#live-decode .live-fig-sub')).toHaveText(/^warming up \(\d+ of 50 tokens\)$/, { timeout: 8000 });
    expect((await read(page)).decode).toBe('—');
    await expect.poll(async () => Number((await read(page)).decode), { timeout: 12_000 }).toBeGreaterThan(10);
    expect(Number((await read(page)).decode)).toBeLessThan(30);
    await expect(page.locator('#live-decode .live-fig-sub')).toHaveText('1 decoding · last 2 s');
  });

  test('7 + 8: a short reply never shows a live rate; its finished row keeps timings, with both numbers in its tooltip', { tag: '@mock' }, async ({ page, request }) => {
    await play(request, 'short-reply', 0);
    await open(page);
    for (let i = 0; i < 6; i++) {
      expect((await read(page)).decode).toBe('—');
      await page.waitForTimeout(250);
    }
    const done = page.locator(isPhone(page) ? '.live-card.is-done' : '.live-row.is-done').first();
    await expect(done).toBeVisible({ timeout: 8000 });
    if (!isPhone(page)) {
      await expect(done.locator('.c-decode .clip').first()).toHaveText(/^\d+\.\d$/);
      await expect(done.locator('.c-decode .live-rate-note')).toHaveAttribute('title', /end to end \(timings\)/);
    }
  });

  test('8: a finished request: the end-to-end numbers stay, the windowed ones beside them', { tag: '@mock' }, async ({ page, request }) => {
    await play(request, 'prefill-now', 19_000);
    await open(page);
    const a = page.locator(isPhone(page) ? '.live-card[data-id^="chatcmpl-1f2552d48"]' : '.live-row[data-id^="chatcmpl-1f2552d48"]');
    await expect(a).toBeVisible({ timeout: 8000 });
    if (isPhone(page)) {
      await expect(a.locator('dd').filter({ hasText: 'tok/s' }).first()).toContainText('433 tok/s');
      return;
    }
    await expect(a.locator('.c-prefill .clip').first()).toHaveText('433');
    await expect(a.locator('.c-prefill .live-rate-note')).toHaveText('mostly cached');
    await expect(a.locator('.c-prefill .live-rate-note')).toHaveAttribute('title', /mostly cached \(2,571 new of 195,083\).*after the first chunk: .* end to end \(timings\): 433 tok\/s/);
    await expect(a.locator('.c-decode .live-rate-note')).toHaveAttribute('title', /decode from token 50: .* · decode end to end \(timings\): \d+\.\d tok\/s/);
  });

  test('9 + 10: nothing running: both tiles read "—", with no greyed last value', { tag: '@mock' }, async ({ page, request }) => {
    await play(request, 'prefill-now', 19_000);
    await request.post('/__mock/live', { data: { scenario: 'clear', simulator: false } });
    await open(page);
    await expect(page.locator('#live-prefill .live-fig-sub')).toHaveText('nothing prefilling', { timeout: 8000 });
    await expect(page.locator('#live-decode .live-fig-sub')).toHaveText('nothing decoding');
    const s = await read(page);
    expect(s.prefill).toBe('—');
    expect(s.decode).toBe('—');
  });

  test('11: the prefill chart gets a sample a second, and the peak comes from them', { tag: '@mock' }, async ({ page, request }) => {
    await play(request, 'prefill-now', 6000);
    await open(page);
    await expect(page.locator('#live-prefill .live-fig-foot')).toContainText('peak 743 tok/s', { timeout: 10_000 });
    await expect(page.locator('#live-prefill qse-spark')).toHaveAttribute('mode', 'ribbon');
  });

  test('12: two prefills at once: the sum, and each rate in the subtitle', { tag: '@mock' }, async ({ page, request }) => {
    await play(request, 'two-prefills', 12_000);
    await open(page);
    await expect(page.locator('#live-prefill .live-fig-num')).toHaveText('1,253', { timeout: 10_000 });
    await expect(page.locator('#live-prefill .live-fig-sub')).toHaveText('2 prefilling · 743 + 510');
    if (isPhone(page)) expect(await noHorizontalOverflow(page)).toBe(true);
  });

  test('13: a fully cached prompt: no rate, "fully cached, no prefill"', { tag: '@mock' }, async ({ page, request }) => {
    await play(request, 'fully-cached', 2000);
    await open(page);
    await expect(page.locator('#live-prefill .live-fig-sub')).toHaveText('fully cached, no prefill (195,083 tokens)', { timeout: 8000 });
    expect((await read(page)).prefill).toBe('—');
  });

  test('14: a mostly cached prompt is labelled', { tag: '@mock' }, async ({ page, request }) => {
    await play(request, 'mostly-cached', 100 + 4000 + 1200 + 300);
    await open(page);
    await expect(page.locator('#live-prefill .live-fig-sub')).toHaveText('mostly cached (2,571 new of 195,083)', { timeout: 8000 });
    if (!isPhone(page)) await expect(page.locator('.live-row.is-prefill .c-prefill .live-rate-note')).toHaveText('mostly cached');
    // left out of the samples: nothing of it reaches the chart's peak
    await page.waitForTimeout(2200);
    expect((await read(page)).prefillFoot).not.toMatch(/peak 853/);
  });

  test('15: a single-call prefill: "—" with the tokens and the time', { tag: '@mock' }, async ({ page, request }) => {
    await play(request, 'single-call', 5000);
    await open(page);
    await expect(page.locator('#live-prefill .live-fig-sub')).toHaveText(/^prefilling 4,000 tokens · \d+\.\d s$/, { timeout: 8000 });
    expect((await read(page)).prefill).toBe('—');
  });

  test('16: an older server without the live fields renders the figures it always had', { tag: '@mock' }, async ({ page, request }) => {
    await play(request, 'prefill-now', 19_000, { liveRates: false });
    await request.post('/__mock/live', { data: { liveRates: false } });
    await open(page);
    await expect(page.locator('#live-prefill .live-fig-num')).toHaveText('433.2', { timeout: 10_000 }); // 1.0: the last finished
    await expect(page.locator('#live-prefill .live-fig-cap')).toHaveCount(0);
    await expect(page.locator('#live-prefill qse-spark')).toHaveAttribute('mode', 'dots');
    await expect(page.locator('.live-rate-note')).toHaveCount(0);
  });
});

test.describe('live rates without the warm-up (fake engine)', () => {
  test.skip(!IS_FAKE, 'the fake engine plays a real chunked prefill');

  test('a chunked prefill with a slow first chunk: "—", then the rate, the same in the tile and the banner; decode from token 50', async ({ page, request }) => {
    await open(page);
    const done = request.post('/v1/chat/completions', {
      // a new first word every run: the fake reports a prefix-cache hit for a prompt it has seen
      data: { messages: [{ role: 'user', content: `${Date.now()}${test.info().project.name} FAKE_WARM_PREFILL FAKE_SLOW go` }], stream: true, max_tokens: 90, chat_template_kwargs: { enable_thinking: false } },
      headers: { 'X-Requested-With': 'qse-dashboard' },
      timeout: 120_000,
    });
    await expect(page.locator('#live-prefill .live-fig-sub')).toContainText('warming up', { timeout: 10_000 });
    expect((await read(page)).prefill).toBe('—');
    await expect.poll(async () => (await read(page)).prefill, { timeout: 15_000 }).toMatch(/^[\d,]+$/);
    for (let i = 0; i < 6; i++) {
      const s = await read(page);
      if (!/^[\d,]+$/.test(s.prefill)) break; // the prefill is over
      expect(s.banner).toContain(`${s.prefill} tok/s`);
      await page.waitForTimeout(200);
    }
    await expect(page.locator('#live-decode .live-fig-sub')).toHaveText(/^warming up \(\d+ of 50 tokens\)$/, { timeout: 10_000 });
    await expect.poll(async () => (await read(page)).decode, { timeout: 20_000 }).toMatch(/^\d+\.\d$/);
    const res = await done;
    const timings = [...(await res.text()).matchAll(/^data: (\{.*\})$/gm)].map((m) => JSON.parse(m[1]) as { timings?: Record<string, number> }).find((c) => c.timings)!.timings!;
    if (!isPhone(page)) {
      const row = page.locator('.live-row.is-done').first();
      await expect(row).toBeVisible({ timeout: 8000 });
      await expect(row.locator('.c-decode .clip').first()).toHaveText(timings.predicted_per_second.toFixed(1));
    }
  });
});
