// VIS-16 — the LM Studio-style developer view.
import { expect, test } from '@playwright/test';
import { IS_MOCK, finishRequest, injectLogLines, login, setMode } from './helpers';

test.describe('dev', () => {
  test.beforeEach(async ({ request }) => {
    if (IS_MOCK) await setMode(request, 'ok');
  });

  test('live logs: a finished request shows its [req] line at the bottom within a second', async ({ page, request }) => {
    test.skip(!IS_MOCK, 'the fake-engine tier drives a real request instead');
    await login(page, '#/dev');
    await expect(page.locator('.console[data-state="open"]')).toBeVisible();
    const rid = await finishRequest(request);
    const line = page.locator('.log-line', { hasText: rid });
    await expect(line).toBeVisible({ timeout: 1500 });
    const last = page.locator('.log-line').last();
    await expect(last).toContainText(rid);
  });

  test('pause: 50 new lines do not move the view; a chip says so and jumps to the newest', async ({ page, request }) => {
    test.skip(!IS_MOCK, 'needs the mock log injector');
    await login(page, '#/dev');
    await expect(page.locator('.console[data-state="open"]')).toBeVisible();
    await page.getByRole('button', { name: 'following' }).click();
    await expect(page.getByRole('button', { name: 'paused' })).toBeVisible();
    const scroller = page.locator('.console-scroll');
    const before = await scroller.evaluate((el) => el.scrollTop);
    await injectLogLines(request, 50, 'info', '[server] pause test line');
    await expect(page.locator('.chip-float')).toContainText(/(5\d|6\d|7\d) new lines/);
    expect(await scroller.evaluate((el) => el.scrollTop)).toBe(before);
    await page.locator('.chip-float').click();
    await expect(page.locator('.chip-float')).toHaveCount(0);
    await expect(page.locator('.log-line').last()).toContainText('pause test line');
  });

  test('level filter: warning shows only warning and error lines and re-opens the stream with level=warning', async ({ page }) => {
    const streams: string[] = [];
    page.on('request', (r) => r.url().includes('/v1/dashboard/logs?') && streams.push(r.url()));
    await login(page, '#/dev');
    await expect(page.locator('.console[data-state="open"]')).toBeVisible();
    await page.getByRole('radio', { name: 'warning' }).click();
    await expect.poll(() => streams.some((u) => u.includes('level=warning'))).toBe(true);
    await expect(page.locator('.log-line').first()).toBeVisible();
    const bad = await page.locator('.log-line.level-info, .log-line.level-debug').count();
    expect(bad).toBe(0);
  });

  test('search: only matching lines, the match highlighted', async ({ page }) => {
    await login(page, '#/dev');
    await expect(page.locator('.console[data-state="open"]')).toBeVisible();
    await page.getByPlaceholder('search (server-side)').fill('finish=error');
    await expect(page.locator('mark').first()).toHaveText(/finish=error/i, { timeout: 15_000 });
    const lines = page.locator('.log-line:not(.log-gap)');
    const n = await lines.count();
    for (let i = 0; i < Math.min(n, 20); i++) await expect(lines.nth(i)).toContainText(/finish=error/i);
  });

  test('reconnect: after the stream drops it resumes with Last-Event-ID and shows no duplicate lines', async ({ page, request }) => {
    test.skip(!IS_MOCK, 'needs the mock drop switch');
    await setMode(request, 'drop:2');
    const streams: { url: string; lastId: string | undefined }[] = [];
    page.on('request', (r) => r.url().includes('/v1/dashboard/logs?') && streams.push({ url: r.url(), lastId: r.headers()['last-event-id'] }));
    await login(page, '#/dev');
    await expect(page.locator('.console[data-state="open"]')).toBeVisible();
    await expect.poll(() => streams.length, { timeout: 15_000 }).toBeGreaterThanOrEqual(2);
    expect(streams[1].lastId ?? new URL(streams[1].url).searchParams.get('since')).toBeTruthy();
    await expect(page.locator('.console[data-state="open"]')).toBeVisible({ timeout: 10_000 });
    await injectLogLines(request, 3, 'info', '[server] after reconnect');
    await expect(page.locator('.log-line', { hasText: 'after reconnect' }).first()).toBeVisible();
    const seqs = await page.evaluate(() => Array.from(document.querySelectorAll('.log-line[data-seq]')).map((e) => e.getAttribute('data-seq')));
    expect(seqs.length).toBeGreaterThan(0);
    expect(new Set(seqs).size).toBe(seqs.length);
    await setMode(request, 'ok');
  });

  test('request timing breakdown: an expanded row shows queue, prefill and decode to scale', async ({ page }) => {
    await login(page, '#/dev');
    const row = page.locator('.req-row:not(.is-failed)').first();
    await expect(row).toBeVisible();
    await row.click();
    const bar = page.locator('.req-expand qse-timing-bar');
    await expect(bar).toBeVisible();
    const widths = await bar.locator('.timing-seg').evaluateAll((els) => els.map((e) => parseFloat((e as HTMLElement).style.width)));
    expect(widths.length).toBeGreaterThanOrEqual(2);
    expect(Math.abs(widths.reduce((a, b) => a + b, 0) - 100)).toBeLessThan(0.5);
    await expect(page.locator('.req-expand .timing-legend')).toContainText(/queue .*ms|queue .*s/);
  });

  test('no prompt text anywhere: the row says text is not recorded', async ({ page }) => {
    await login(page, '#/dev');
    await page.locator('.req-row').first().click();
    await expect(page.locator('.req-expand')).toContainText('text is not recorded');
    await expect(page.locator('.req-expand')).toContainText('no prompt or answer text');
  });

  test('config search: DEEP shows QWEN38_DEEP and QWEN38_DEEP_AFTER with their values', async ({ page }) => {
    await login(page, '#/dev');
    await expect(page.locator('#config .model-card')).toBeVisible();
    await page.getByPlaceholder('search flags, e.g. DEEP').fill('DEEP');
    const keys = page.locator('#config .cfg-key');
    await expect(keys).toHaveCount(2);
    await expect(keys.nth(0)).toHaveText('QWEN38_DEEP');
    await expect(keys.nth(1)).toHaveText('QWEN38_DEEP_AFTER');
    await expect(page.locator('#config .cfg-val').nth(0)).toHaveText('32');
    await page.getByPlaceholder('search flags, e.g. DEEP').fill('TOKEN');
    await expect(page.locator('#config .cfg-val.is-secret').first()).toHaveText('<redacted>');
  });

  test('test request: "Say hi" with max_tokens 32 and thinking off streams and shows usage and timings', async ({ page }) => {
    await login(page, '#/dev');
    await page.locator('#testbox textarea').fill('Say hi');
    await page.locator('#testbox input[type="number"]').fill('32');
    const thinking = page.locator('#testbox .switch input');
    if (await thinking.isChecked()) await thinking.uncheck();
    await page.getByRole('button', { name: 'Send' }).click();
    await expect(page.locator('.test-answer')).toContainText(/\w+/, { timeout: 20_000 });
    await expect(page.locator('.test-json')).toContainText('ttft_ms', { timeout: 20_000 });
    await expect(page.locator('.test-json')).toContainText('predicted_per_second');
    await expect(page.locator('.test-json')).toContainText('completion_tokens');
    await expect(page.locator('.test-result .stat').first()).toContainText('time to first token');
    await expect(page.locator('.test-reasoning')).toHaveCount(0);
  });

  test('test request with thinking on shows a reasoning block', async ({ page }) => {
    await login(page, '#/dev');
    await page.locator('#testbox .switch input').check();
    await page.getByRole('button', { name: 'Send' }).click();
    await expect(page.locator('.test-reasoning pre')).toContainText(/\w+/, { timeout: 20_000 });
    await expect(page.locator('.test-answer')).toContainText(/\w+/, { timeout: 20_000 });
  });

  test('busy engine: the 503 message and its Retry-After are shown', async ({ page, request }) => {
    test.skip(!IS_MOCK, 'needs the mock busy switch');
    await setMode(request, 'busy');
    await login(page, '#/dev');
    await page.getByRole('button', { name: 'Send' }).click();
    await expect(page.locator('#testbox .state-error')).toContainText('503');
    await expect(page.locator('#testbox .state-error')).toContainText('retry after 5 s');
    await setMode(request, 'ok');
  });

  test('a [req] line links to its row in the table', async ({ page, request }) => {
    test.skip(!IS_MOCK, 'needs the mock request trigger');
    await login(page, '#/dev');
    await expect(page.locator('.console[data-state="open"]')).toBeVisible();
    const rid = await finishRequest(request);
    await expect(page.locator('.req-row', { hasText: rid.slice(0, 8) }).or(page.locator('.log-line', { hasText: rid }))).toBeVisible({ timeout: 15_000 });
    await page.locator('.log-line a.log-req', { hasText: rid }).first().click();
    await expect(page.locator('.req-row.is-open')).toBeVisible({ timeout: 15_000 });
    await expect(page.locator('.req-expand code')).toHaveText(rid);
  });
});
