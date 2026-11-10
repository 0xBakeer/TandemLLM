// the LM Studio-style developer view.
import { readFileSync } from 'node:fs';
import { expect, test, type APIRequestContext, type Page } from '@playwright/test';
import { IS_MOCK, finishRequest, injectLogLines, login, setMode } from './helpers';

// SRV-804: the Log panel's export (SRV-803)
const EXPORT = '/v1/dashboard/logs/export';
const ENTRY = /^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z (debug|info|warning|error) /;

/** Press the export button and read the file the browser saved. */
async function exportLog(page: Page): Promise<{ name: string; type: string; head: string[]; entries: string[] }> {
  const answer = page.waitForResponse((r) => r.url().includes(EXPORT));
  const [download] = await Promise.all([page.waitForEvent('download'), page.locator('[data-log-export]').click()]);
  const type = (await answer).headers()['content-type'] ?? '';
  const text = readFileSync(await download.path(), 'utf8');
  const cut = text.indexOf('\n\n');
  const entries: string[] = [];
  for (const line of text.slice(cut + 2).split('\n')) {
    if (ENTRY.test(line)) entries.push(line);
    else if (line && entries.length) entries[entries.length - 1] += `\n${line}`;
  }
  return { name: download.suggestedFilename(), type, head: text.slice(0, cut).split('\n'), entries };
}
const field = (head: string[], label: string) => head.find((l) => l.startsWith(`${label}: `))?.slice(label.length + 2) ?? '';
/** How many lines the mock engine's ring holds now. */
const ringLines = async (request: APIRequestContext) => Number((await (await request.get('/__mock/mode')).json()).logLines);

test.describe('dev', () => {
  test.beforeEach(async ({ request }) => {
    if (IS_MOCK) await setMode(request, 'ok');
  });

  test('live logs: a finished request shows its [req] line at the bottom within a second', { tag: '@mock' }, async ({ page, request }) => {
    test.skip(!IS_MOCK, 'the fake-engine tier drives a real request instead');
    await login(page, '#/dev');
    await expect(page.locator('.console[data-state="open"]')).toBeVisible();
    const rid = await finishRequest(request);
    const line = page.locator('.log-line', { hasText: rid });
    await expect(line).toBeVisible({ timeout: 1500 });
    const last = page.locator('.log-line').last();
    await expect(last).toContainText(rid);
  });

  test('pause: 50 new lines do not move the view; a chip says so and jumps to the newest', { tag: '@mock' }, async ({ page, request }) => {
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
    // @mock: the mock's log always has warning lines; a healthy engine's may have none
    if (IS_MOCK) await expect(page.locator('.log-line').first()).toBeVisible();
    else await expect(page.locator('.console[data-state="open"]')).toBeVisible();
    const bad = await page.locator('.log-line.level-info, .log-line.level-debug').count();
    expect(bad).toBe(0);
  });

  test('search: only matching lines, the match highlighted', async ({ page, request }) => {
    await login(page, '#/dev');
    await expect(page.locator('.console[data-state="open"]')).toBeVisible();
    // @mock: an error line is the mock's; on a real engine every [req] line carries finish=
    const term = IS_MOCK ? 'finish=error' : 'finish=';
    // @mock: the mock's own ring holds such a line at some hours only (its rows follow the clock),
    // so the case brings two of its own
    if (IS_MOCK) await injectLogLines(request, 2, 'error', '[server] the search case: finish=error');
    await page.getByPlaceholder('search (server-side)').fill(term);
    await expect(page.locator('mark').first()).toHaveText(term, { timeout: 15_000 });
    // read the lines in one go (a live console re-renders between a count and an nth()), until the
    // re-opened stream has replaced what was on screen before the search
    const lines = page.locator('.log-line:not(.log-gap)');
    await expect.poll(async () => (await lines.allTextContents()).slice(0, 20).every((t) => t.includes(term))).toBe(true);
  });

  test('reconnect: after the stream drops it resumes with Last-Event-ID and shows no duplicate lines', { tag: '@mock' }, async ({ page, request }) => {
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

  test('VIS-60: recent requests leave internal calls out until the switch is on', async ({ page }) => {
    await login(page, '#/dev');
    const sw = page.locator('#requests [data-filter="internal"] input');
    await expect(sw).toBeVisible();
    await expect(sw).not.toBeChecked();
    const asked = page.waitForRequest((r) => r.url().includes('/v1/dashboard/requests?') && r.url().includes('internal=1'));
    await sw.check();
    await asked;
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
    if (IS_MOCK) {
      // @mock: the mock's two flags; the real engine also lists its parsed arguments (four keys)
      await expect(keys).toHaveCount(2);
      await expect(keys.nth(0)).toHaveText('QWEN38_DEEP');
      await expect(keys.nth(1)).toHaveText('QWEN38_DEEP_AFTER');
      await expect(page.locator('#config .cfg-val').nth(0)).toHaveText('32');
    } else {
      for (const k of await keys.allTextContents()) expect(k.toUpperCase()).toContain('DEEP');
    }
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

  test('busy engine: the 503 message and its Retry-After are shown', { tag: '@mock' }, async ({ page, request }) => {
    test.skip(!IS_MOCK, 'needs the mock busy switch');
    await setMode(request, 'busy');
    await login(page, '#/dev');
    await page.getByRole('button', { name: 'Send' }).click();
    await expect(page.locator('#testbox .state-error')).toContainText('503');
    await expect(page.locator('#testbox .state-error')).toContainText('retry after 5 s');
    await setMode(request, 'ok');
  });

  test('a [req] line links to its row in the table', { tag: '@mock' }, async ({ page, request }) => {
    test.skip(!IS_MOCK, 'needs the mock request trigger');
    await login(page, '#/dev');
    await expect(page.locator('.console[data-state="open"]')).toBeVisible();
    const rid = await finishRequest(request);
    await expect(page.locator('.req-row', { hasText: rid.slice(0, 8) }).or(page.locator('.log-line', { hasText: rid }))).toBeVisible({ timeout: 15_000 });
    await page.locator('.log-line a.log-req', { hasText: rid }).first().click();
    await expect(page.locator('.req-row.is-open')).toBeVisible({ timeout: 15_000 });
    await expect(page.locator('.req-expand code')).toHaveText(rid);
  });

  test('export: The button downloads the log as a text file with a header', async ({ page, request }) => {
    const streams: string[] = [];
    page.on('request', (r) => r.url().includes('/v1/dashboard/logs?') && streams.push(r.url()));
    await login(page, '#/dev');
    await expect(page.locator('.console[data-state="open"]')).toBeVisible();
    const button = page.locator('[data-log-export]');
    await expect(button).toBeVisible();
    await expect(button).toHaveAttribute('title', 'Export the whole log as a text file');
    await expect(button).toHaveText('Export');
    // beside "Clear view", in the same toolbar and of its size
    const clear = page.locator('#console .panel-tools button[title="Clear view"]');
    const [a, b] = [await clear.boundingBox(), await button.boundingBox()];
    expect(b?.height).toBe(a?.height);
    expect(await page.locator('#console .panel-tools button').last().getAttribute('title')).toBe('Export the whole log as a text file');
    const opened = streams.length;
    const before = IS_MOCK ? await ringLines(request) : 0;
    const file = await exportLog(page);
    const after = IS_MOCK ? await ringLines(request) : 0;
    expect(file.type).toMatch(/^text\/plain/);
    expect(file.name).toMatch(/^tandemllm-[A-Za-z0-9.-]+-\d{8}T\d{6}Z\.log$/);
    expect(file.head[0]).toBe('TandemLLM engine log');
    expect(file.head.slice(1).map((l) => l.slice(0, l.indexOf(':')))).toEqual(['version', 'commit', 'uptime', 'exported', 'count', 'first line', 'last line', 'ring', 'log content']);
    expect(field(file.head, 'exported')).toMatch(/^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ \(UTC\)$/);
    expect(field(file.head, 'uptime')).toMatch(/\d+ s$/);
    expect(field(file.head, 'count')).toBe(`${file.entries.length} lines`);
    if (IS_MOCK) {
      // @mock: the mock engine's version and commit, and the count against what its ring holds
      expect(file.name).toContain('tandemllm-0.1.0-rc4-');
      expect(field(file.head, 'version')).toBe('0.1.0-rc4');
      expect(field(file.head, 'commit')).toBe('7d0b176');
      expect(file.entries.length).toBeGreaterThanOrEqual(before);
      expect(file.entries.length).toBeLessThanOrEqual(after);
    } else {
      // the file name carries the version the header gives, made safe for a file name
      expect(file.name).toContain(`tandemllm-${field(file.head, 'version').replace(/[^A-Za-z0-9.-]/g, '-')}-`);
    }
    // the export left the page as it was: no error, the same stream, still live
    await expect(page.locator('[data-log-export-error]')).toHaveCount(0);
    await expect(page.locator('.console[data-state="open"]')).toBeVisible();
    await expect(button).toBeEnabled();
    expect(streams.length).toBe(opened);
  });

  test('export: The file holds the whole ring, not the lines on screen', { tag: '@mock' }, async ({ page, request }) => {
    test.skip(!IS_MOCK, 'needs the mock log injector');
    await login(page, '#/dev');
    await expect(page.locator('.console[data-state="open"]')).toBeVisible();
    // the page shows warnings that match one search term
    await page.getByRole('radio', { name: 'warning' }).click();
    await page.getByPlaceholder('search (server-side)').fill('needle-803');
    // the ring fills past its 10,000 lines at every level
    for (const level of ['debug', 'info', 'error']) await injectLogLines(request, 3400, level);
    await injectLogLines(request, 5, 'warning', '[server] needle-803 on screen');
    await expect(page.locator('.log-line:not(.log-gap)')).toHaveCount(5, { timeout: 15_000 });
    expect(await ringLines(request)).toBe(10_000);
    const file = await exportLog(page);
    expect(field(file.head, 'count')).toBe('10000 lines');
    expect(file.entries).toHaveLength(10_000);
    expect(field(file.head, 'ring')).toMatch(/^full, the \d+ lines before the first one are not in this file \(the ring keeps 10000\)$/);
    // every level is in the file, the debug lines the page never asked for included
    for (const level of ['debug', 'info', 'warning', 'error']) expect(file.entries.some((e) => e.split(' ')[1] === level), level).toBe(true);
    expect(file.entries.filter((e) => e.includes('needle-803'))).toHaveLength(5);
    // in the ring's order: the time never runs back, and each injected run counts up
    const times = file.entries.map((e) => e.slice(0, 24));
    expect(times.every((t, i) => i === 0 || t >= times[i - 1])).toBe(true);
    for (const level of ['debug', 'info', 'error']) {
      const ns = file.entries.filter((e) => e.includes(` ${level} [server] injected line `)).map((e) => Number(e.slice(e.lastIndexOf(' ') + 1)));
      expect(ns.length, level).toBeGreaterThan(1000);
      expect(ns.every((n, i) => i === 0 || n === ns[i - 1] + 1), level).toBe(true);
    }
    // the page still shows its own five lines at its own level and search
    await expect(page.getByRole('radio', { name: 'warning' })).toHaveAttribute('aria-checked', 'true');
    await expect(page.getByPlaceholder('search (server-side)')).toHaveValue('needle-803');
    await expect(page.locator('.log-line:not(.log-gap)')).toHaveCount(5);
  });

  test('export: The session is refused: a 401 shows the error and no download starts', async ({ page }) => {
    let downloads = 0;
    page.on('download', () => downloads++);
    await login(page, '#/dev');
    await expect(page.locator('.console[data-state="open"]')).toBeVisible();
    await page.route(`**${EXPORT}`, (route) => route.fulfill({ status: 401, contentType: 'application/json', body: JSON.stringify({ error: { type: 'unauthorized', message: 'a session or an admin token is required' } }) }));
    await page.locator('[data-log-export]').click();
    const error = page.locator('#console [data-log-export-error] .state-error');
    await expect(error).toBeVisible();
    await expect(error).toHaveAttribute('role', 'alert');
    await expect(error).toContainText('Could not export the log from');
    await expect(error.locator('code')).toHaveText(EXPORT);
    await expect(error).toContainText('session expired. No file was saved.');
    await page.waitForTimeout(500);
    expect(downloads).toBe(0);
    // the log itself is still there and live under the error
    await expect(page.locator('.console[data-state="open"]')).toBeVisible();
    await error.getByRole('button', { name: 'Dismiss' }).click();
    await expect(page.locator('[data-log-export-error]')).toHaveCount(0);
    expect(downloads).toBe(0);
  });

  test('export: The engine cannot be reached: the error shows, no file is saved, and "Try again" saves it once the engine answers', async ({ page }) => {
    let downloads = 0;
    page.on('download', () => downloads++);
    await login(page, '#/dev');
    await expect(page.locator('.console[data-state="open"]')).toBeVisible();
    await page.route(`**${EXPORT}`, (route) => route.abort('connectionrefused'));
    await page.locator('[data-log-export]').click();
    const error = page.locator('#console [data-log-export-error] .state-error');
    await expect(error).toContainText('engine unreachable');
    await expect(error).toContainText(/[.!?] No file was saved\./);
    await page.waitForTimeout(500);
    expect(downloads).toBe(0);
    await page.unroute(`**${EXPORT}`);
    const [download] = await Promise.all([page.waitForEvent('download'), error.getByRole('button', { name: 'Try again' }).click()]);
    expect(download.suggestedFilename()).toMatch(/^tandemllm-[A-Za-z0-9.-]+\.log$/);
    await expect(page.locator('[data-log-export-error]')).toHaveCount(0);
  });

  test('export: a browser that cannot save a generated file is told so and offered the file from the engine', async ({ page }) => {
    let asked = 0;
    let downloads = 0;
    page.on('request', (r) => r.url().includes(EXPORT) && asked++);
    page.on('download', () => downloads++);
    // a browser with no way to hand a page-built file to the user
    await page.addInitScript(() => {
      Object.defineProperty(URL, 'createObjectURL', { value: undefined, configurable: true });
    });
    await login(page, '#/dev');
    await expect(page.locator('.console[data-state="open"]')).toBeVisible();
    await page.locator('[data-log-export]').click();
    const error = page.locator('#console [data-log-export-error] .state-error');
    await expect(error).toContainText('This browser cannot save a file that the page hands it.');
    await expect(error).toContainText('No file was saved.');
    const link = error.locator('a[data-log-export-link]');
    await expect(link).toHaveAttribute('href', new RegExp(`${EXPORT}$`));
    expect(asked).toBe(0);
    expect(downloads).toBe(0);
  });

  test('export: Every existing feature of the page stays with the button added', { tag: '@mock' }, async ({ page, request }) => {
    test.skip(!IS_MOCK, 'needs the mock log injector');
    await login(page, '#/dev');
    await expect(page.locator('.console[data-state="open"]')).toBeVisible();
    await exportLog(page);
    // wrap
    const wrap = page.locator('#console button[title="Wrap long lines"]');
    await wrap.click();
    await expect(page.locator('.console')).toHaveClass(/is-wrap/);
    await wrap.click();
    await expect(page.locator('.console')).not.toHaveClass(/is-wrap/);
    // pause, then follow again
    await page.getByRole('button', { name: 'following' }).click();
    await injectLogLines(request, 4, 'info', '[server] after the export');
    await expect(page.locator('.chip-float')).toContainText('new line');
    await page.locator('.chip-float').click();
    await expect(page.getByRole('button', { name: 'following' })).toBeVisible();
    await expect(page.locator('.log-line', { hasText: 'after the export' }).first()).toBeVisible();
    // copy line is still on each line
    await expect(page.locator('.log-line .log-copy button').first()).toBeAttached();
    // clear view empties the view, not the ring: the next export still has the lines
    await page.locator('#console button[title="Clear view"]').click();
    await expect(page.locator('.log-line', { hasText: 'after the export' })).toHaveCount(0);
    const file = await exportLog(page);
    expect(file.entries.filter((e) => e.endsWith(' info [server] after the export'))).toHaveLength(4);
    // and the stream goes on into the cleared view
    await injectLogLines(request, 1, 'info', '[server] still live');
    await expect(page.locator('.log-line', { hasText: 'still live' })).toBeVisible();
  });
});
