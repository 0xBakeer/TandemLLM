// VIS-24 / VIS-25 — the Live panel says what the engine is doing now (contract 1.1): the Now line,
// the activity cell, the client flag, the continues note, the constrained tag, the timeline, the
// last 20 stops, the stream's health, the 1.0 fallback, the phone, reduced motion, and the frame
// budget at four events a second. The mock scenarios (`POST /__mock/live`) drive it; the fake
// engine and the box run the untagged tests.
import { expect, test, type APIRequestContext, type Page } from '@playwright/test';
import { IS_FAKE, IS_MOCK, login, noHorizontalOverflow, setMode } from './helpers';

const isPhone = (page: Page) => (page.viewportSize()?.width ?? 1440) <= 768;

async function scenario(request: APIRequestContext, name: string, extra: Record<string, unknown> = {}) {
  const res = await request.post('/__mock/live', { data: { scenario: name, ...extra } });
  expect(res.ok()).toBeTruthy();
  return res.json() as Promise<{ requests: number; recent: number }>;
}

/** Start a streamed chat in the background (the fake-engine and box tiers). */
function streamChat(request: APIRequestContext, content: string, maxTokens = 400, tools = false): Promise<string> {
  const body: Record<string, unknown> = { messages: [{ role: 'user', content }], stream: true, max_tokens: maxTokens };
  if (tools) {
    body.tools = [{ type: 'function', function: { name: 'get_weather', description: 'weather', parameters: { type: 'object', properties: { city: { type: 'string' } }, required: ['city'] } } }];
  } else body.chat_template_kwargs = { enable_thinking: !IS_MOCK };
  return request.post('/v1/chat/completions', { data: body, headers: { 'X-Requested-With': 'qse-dashboard' }, timeout: 180_000 }).then((r) => r.text());
}

test.describe('live activity', () => {
  test.beforeEach(async ({ request }) => {
    if (IS_MOCK) {
      await setMode(request, 'ok');
      // the mock engine's own requests would outrank a scenario in the Now line: pause them here
      await scenario(request, 'clear', { contract: '1.1', activity: true, simulator: false });
    }
  });
  test.afterAll(async ({ request }) => {
    if (IS_MOCK) await scenario(request, 'clear', { contract: '1.1', activity: true, simulator: true });
  });

  test('the Now line sits above the figures and reads the engine state; the head says the stream is healthy', async ({ page }) => {
    await login(page, '#/performance');
    const live = page.locator('.live');
    await expect(live.locator('#live-now')).toBeVisible({ timeout: 10_000 });
    await expect(live.locator('.now-headline')).not.toBeEmpty();
    // the Now line comes before the two figures in the document
    const nowBox = await live.locator('#live-now').boundingBox();
    const figBox = await live.locator('#live-decode').boundingBox();
    expect(nowBox!.y).toBeLessThan(figBox!.y);
    await expect(live.locator('.live-state')).toContainText(/streaming|connecting/, { timeout: 10_000 });
    await expect(live.locator('.live-state .dot')).toBeVisible();
    // the Last 20 section is there (a fresh engine may have no stops yet)
    await expect(live.locator('.recent')).toBeVisible();
  });

  test('an agent turn: prefilling with a moving bar, thinking, writing, calling tool write_file; the timeline grows in that order', { tag: '@mock' }, async ({ page, request }) => {
    test.skip(!IS_MOCK, 'needs the mock scenario');
    await scenario(request, 'agent-turn', { at: 6000 }); // 6 s in: the prefill is under way
    await login(page, '#/performance');
    const live = page.locator('.live');
    const headline = live.locator('.now-headline');
    await expect(headline).toContainText(/^Prefilling [\d,]+ of 48,210 \(\d+ %\)$/, { timeout: 10_000 });
    await expect(live.locator('.now-bar')).toBeVisible();
    await expect(live.locator('.now-numbers')).toContainText(/tok\/s/);
    await expect(live.locator('.now-numbers')).toContainText('12,288 cached');
    await expect(live.locator('.now-client')).toHaveText('opencode');
    await expect(live.locator('.live-state')).toContainText('4/s', { timeout: 10_000 });
    const pct = async () => Number((await headline.textContent())!.match(/\((\d+) %\)/)![1]);
    const p1 = await pct();
    await expect.poll(pct, { timeout: 6000 }).toBeGreaterThan(p1);
    const width = async () => Number(await live.locator('.now-bar-fill').evaluate((el) => (el as HTMLElement).style.width.replace('%', '')));
    expect(await width()).toBeGreaterThan(0);
    await expect(headline).toHaveText('Thinking', { timeout: 15_000 });
    await expect(live.locator('.now-numbers')).toContainText(/thinking tokens/);
    await expect(live.locator('.now-bar')).toHaveCount(0);
    const row = live.locator(isPhone(page) ? '.live-card' : '.live-row').first();
    await expect(row.locator('.phase.act')).toContainText('thinking');
    await expect(headline).toHaveText('Writing', { timeout: 15_000 });
    await expect(row.locator('.phase.act')).toContainText('writing');
    await expect(headline).toHaveText('Calling tool write_file', { timeout: 10_000 });
    await expect(row.locator('.phase.act')).toContainText('write_file');
    // the arguments figure grows while the call streams
    const kb = async () => {
      const m = (await live.locator('.now-numbers').textContent())!.match(/([\d.]+) (B|KB) of arguments/)!;
      return Number(m[1]) * (m[2] === 'KB' ? 1000 : 1);
    };
    const k1 = await kb();
    await expect.poll(kb, { timeout: 5000 }).toBeGreaterThan(k1);
    // the timeline: one segment per state, in order (the strip on a desktop, the list on a phone)
    const order = isPhone(page) ? await row.locator('.tl-list li').allTextContents() : await row.locator('.sr-only li').allTextContents();
    const words = order.map((x) => x.replace(/^\+[\d.]+ (m \d+ )?s\s*/, '').replace(/\s+(\d+ cached|write_file)$/, '').trim());
    expect(words).toEqual(['queued', 'prefilling', 'thinking', 'writing', 'tool call']);
    if (!isPhone(page)) {
      await expect(row.locator('.tl .tl-seg')).toHaveCount(5);
      const tones = await row.locator('.tl .tl-seg').evaluateAll((els) => els.map((e) => [...e.classList].find((c) => c.startsWith('tone-'))));
      expect(tones).toEqual(['tone-queued', 'tone-prefill', 'tone-think', 'tone-write', 'tone-tool']);
    }
  });

  test('waiting for the client between two requests, then the next one continues the first', { tag: '@mock' }, async ({ page, request }) => {
    test.skip(!IS_MOCK, 'needs the mock scenario');
    await scenario(request, 'agent-turn', { at: 39_500 }); // 0.9 s before the tool_calls finish
    await login(page, '#/performance');
    const live = page.locator('.live');
    await expect(live.locator('.now-headline')).toHaveText('Waiting for client: running tool write_file', { timeout: 10_000 });
    await expect(live.locator('#live-now .now-glyph')).toHaveText('◌');
    const since = async () => parseFloat((await live.locator('.now-since').textContent())!);
    const s1 = await since();
    await expect.poll(since, { timeout: 5000 }).toBeGreaterThan(s1);
    // the finished request stays in the list with its stop as the cell, and Last 20 has the sentence
    await expect(live.locator('.recent-sentence').first()).toHaveText(/^tool call: write_file after [\d.]+ s$/);
    await expect(live.locator('.recent .stop').first()).toContainText('tool call');
    // the next request arrives and says which one it continues
    await expect(live.locator('.now-headline')).toContainText(/^Prefilling/, { timeout: 12_000 });
    const row = live.locator(isPhone(page) ? '.live-card' : '.live-row').first();
    await expect(row.locator('.flag-continues')).toContainText(/continues chatcmpl-77aa1c0e2 after [\d.]+ s \(client ran write_file\)$/);
  });

  test('the client disconnects during a prefill: a red flag on the row, then a red ✗ in Last 20', { tag: '@mock' }, async ({ page, request }) => {
    test.skip(!IS_MOCK, 'needs the mock scenario');
    await scenario(request, 'abandoned', { at: 1000 });
    await login(page, '#/performance');
    const live = page.locator('.live');
    await expect(live.locator('.now-headline')).toContainText(/^Prefilling [\d,]+ of 60,014/, { timeout: 10_000 });
    const row = live.locator(isPhone(page) ? '.live-card' : '.live-row').first();
    await expect(row.locator('.flag-bad')).toContainText(/client disconnected( [\d.]+ s ago)?$/, { timeout: 8000 });
    await expect(row.locator('.flag-bad .flag-glyph')).toHaveText('✗');
    await expect(row.locator('.flag-bad')).toHaveCSS('color', /rgb\(230, 103, 103\)|rgb\(201, 60, 60\)/); // --err, dark / light
    await expect(live.locator('.now-warning')).toContainText('client disconnected');
    await expect(live.locator('.now-headline')).toHaveText('Idle', { timeout: 12_000 });
    const rec = live.locator('.recent-row').first();
    await expect(rec.locator('.stop')).toContainText('abandoned');
    await expect(rec.locator('.stop .phase-glyph')).toHaveText('✗');
    await expect(rec).toHaveClass(/tone-bad/);
    await expect(rec.locator('.recent-sentence')).toHaveText(/^abandoned by the client after [\d.]+ s of silent prefill, 0 tokens sent$/);
    await expect(rec.locator('.tl')).toHaveClass(/is-bad/);
  });

  test('every stop reason reads as a sentence with its glyph', { tag: '@mock' }, async ({ page, request }) => {
    test.skip(!IS_MOCK, 'needs the mock scenario');
    await scenario(request, 'stops');
    await login(page, '#/performance');
    const live = page.locator('.live');
    await expect(live.locator('.now-headline')).toHaveText('Idle', { timeout: 10_000 });
    await expect(live.locator('.state-empty')).toContainText('Idle. Nothing in flight.');
    const rows = live.locator('.recent-row');
    await expect(rows).toHaveCount(10);
    const marks = await rows.locator('.stop').allTextContents();
    expect(marks.map((m) => m.trim())).toEqual(['✓stop', '✓length', '✓tool call', '✗timeout', '✗abandoned', '✗error', '!refused', '!rejected', '✗cancelled', '✓stop']);
    const sentences = await rows.locator('.recent-sentence').allTextContents();
    expect(sentences.map((s) => s.trim())).toEqual([
      expect.stringMatching(/^finished at the end of the answer after 41\.2 s, 812 tokens sent$/),
      'stopped at the length limit (32,000 tokens)',
      'tool call: write_file, bash (2 calls) after 41.2 s',
      'timed out after 41.2 s',
      'abandoned by the client after 73 s of silent prefill, 0 tokens sent',
      'error: RuntimeError after 5 tokens',
      'refused: queue full',
      'rejected: the prompt is longer than the context window',
      'cancelled: the server was shutting down',
      'finished at a stop string after 41.2 s, 3 tokens sent',
    ]);
    const tones = await rows.evaluateAll((els) => els.map((e) => [...e.classList].find((c) => c.startsWith('tone-'))));
    expect(tones).toEqual(['tone-ok', 'tone-ok', 'tone-ok', 'tone-bad', 'tone-bad', 'tone-bad', 'tone-warn', 'tone-warn', 'tone-bad', 'tone-ok']);
    if (!isPhone(page)) await expect(rows.nth(2).locator('.tools')).toHaveText('write_file, bash');
    // a stop opens the Dev tab's request detail
    await rows.nth(0).locator('.recent-open').click();
    await expect(page).toHaveURL(/#\/dev\?request=chatcmpl-/);
    await expect(page.locator('.view-dev, qse-dev').first()).toBeVisible();
  });

  test('constrained: the JSON schema and tool_choice tags, a queued place', { tag: '@mock' }, async ({ page, request }) => {
    test.skip(!IS_MOCK, 'needs the mock scenario');
    await scenario(request, 'constrained', { at: 1500 });
    await login(page, '#/performance');
    const live = page.locator('.live');
    const rows = live.locator(isPhone(page) ? '.live-card' : '.live-row');
    await expect(rows).toHaveCount(2, { timeout: 10_000 });
    await expect(rows.nth(0).locator('.tag')).toHaveText('JSON schema');
    await expect(rows.nth(1).locator('.phase.act')).toContainText(/queued/);
    await expect(rows.nth(1).locator('.act-detail')).toContainText(/about 2nd/);
    await expect(rows.nth(1).locator('.tag')).toHaveText('tool_choice');
    await expect(live.locator('.now-numbers')).toContainText('JSON schema');
    await expect(live.locator('.now-headline')).toHaveText('Calling tool write_file', { timeout: 10_000 });
    await expect(live.locator('.now-numbers')).toContainText('tool_choice');
  });

  test('the decode figure carries tokens per round and ms a round', { tag: '@mock' }, async ({ page, request }) => {
    test.skip(!IS_MOCK, 'needs the mock scenario');
    await scenario(request, 'agent-turn', { at: 20_000 }); // thinking
    await login(page, '#/performance');
    await expect(page.locator('#live-decode .live-fig-rounds')).toContainText(/^[\d.]+ tokens per round · [\d.]+ ms a round \(last second\)$/, { timeout: 10_000 });
    const row = page.locator(isPhone(page) ? '.live-card' : '.live-row').first();
    await expect(row.locator('.live-split')).toContainText(/^\d[\d,]* thinking · \d[\d,]* content$/);
  });

  test('the stream drops: the head says reconnecting since N s, the numbers stay and dim, a snapshot restores everything', { tag: '@mock' }, async ({ page, request }) => {
    test.skip(!IS_MOCK, 'needs the mock drop switch');
    await scenario(request, 'agent-turn', { at: 20_000 });
    await setMode(request, 'drop:3');
    await login(page, '#/performance');
    const live = page.locator('.live');
    await expect(live.locator('.now-headline')).toHaveText('Thinking', { timeout: 10_000 });
    await expect(live.locator('.live-state')).toContainText(/reconnecting( since [\d.]+ s)?/, { timeout: 8000 });
    await expect(live).toHaveClass(/is-stale/);
    await expect(live.locator('.now-headline')).toHaveText('Thinking'); // the last numbers stay
    await expect(live.locator('.live-state')).toContainText('streaming', { timeout: 10_000 });
    await expect(live).not.toHaveClass(/is-stale/);
    await setMode(request, 'ok');
  });

  test('a 1.0 server: VIS-23\'s panel, no Now line, no timeline, no Last 20, no errors', { tag: '@mock' }, async ({ page, request }) => {
    test.skip(!IS_MOCK, 'needs the mock contract switch');
    const errors: string[] = [];
    page.on('pageerror', (e) => errors.push(e.message));
    await scenario(request, 'busy', { contract: '1.0' });
    await login(page, '#/performance');
    const live = page.locator('.live');
    await expect(live.locator('.live-counts')).toContainText('in flight', { timeout: 10_000 });
    await expect(live.locator('#live-now')).toHaveCount(0);
    await expect(live.locator('.recent')).toHaveCount(0);
    await expect(live.locator('.tl')).toHaveCount(0);
    await expect(live.locator('.live-table th').first()).toHaveText('phase');
    await expect(live.locator('.phase').first()).toContainText(/decoding|prefilling|queued/);
    await expect(live.locator('.live-state')).toContainText('streaming · 1 s', { timeout: 10_000 });
    expect(errors).toEqual([]);
    await scenario(request, 'clear', { contract: '1.1' });
  });

  test('the kill switch (--live-activity off): 1.1 with null activity renders the engine label and the phase words', { tag: '@mock' }, async ({ page, request }) => {
    test.skip(!IS_MOCK, 'needs the mock switch');
    await scenario(request, 'busy', { activity: false });
    await login(page, '#/performance');
    const live = page.locator('.live');
    await expect(live.locator('.now-headline')).toHaveText('Decoding', { timeout: 10_000 });
    await expect(live.locator('.phase').first()).toContainText('decoding');
    await expect(live.locator('.recent')).toHaveCount(0);
    await scenario(request, 'clear', { activity: true });
  });

  test('on a phone: the Now line wraps, requests are cards with the activity as title, the timeline is a list, nothing scrolls sideways at 360 px', { tag: '@mock' }, async ({ page, request }) => {
    test.skip(!isPhone(page), 'the phone project');
    test.skip(!IS_MOCK, 'needs the mock scenario');
    await scenario(request, 'agent-turn', { at: 30_000 }); // writing
    await login(page, '#/performance');
    const live = page.locator('.live');
    await expect(live.locator('.now-headline')).toHaveText('Writing', { timeout: 10_000 });
    const card = live.locator('.live-card').first();
    await expect(card.locator('.live-card-head .phase.act')).toContainText('writing');
    await expect(card.locator('.tl-list li')).toHaveCount(4); // queued, prefilling, thinking, writing
    await expect(card.locator('.tl')).toHaveCount(0);
    expect(await noHorizontalOverflow(page)).toBe(true);
    // the last 20 as cards, the sentence first, a 44 px target
    await scenario(request, 'stops');
    await expect(live.locator('.recent-row').first()).toBeVisible({ timeout: 10_000 });
    const btn = await live.locator('.recent-row .recent-open').first().boundingBox();
    expect(btn!.height).toBeGreaterThanOrEqual(44);
    await page.setViewportSize({ width: 360, height: 780 });
    await scenario(request, 'agent-turn', { at: 6000 });
    await expect(live.locator('.now-headline')).toContainText(/^Prefilling/, { timeout: 10_000 });
    expect(await noHorizontalOverflow(page)).toBe(true);
    const box = await live.locator('#live-now').boundingBox();
    expect(box!.width).toBeLessThanOrEqual(360);
  });

  test('reduced motion: nothing pulses, the bar does not slide', { tag: '@mock' }, async ({ page, request }) => {
    test.skip(!IS_MOCK, 'needs the mock scenario');
    await scenario(request, 'agent-turn', { at: 6000 });
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await login(page, '#/performance');
    const live = page.locator('.live');
    await expect(live.locator('#live-now.is-moving')).toBeVisible({ timeout: 10_000 });
    expect(await live.locator('.now-glyph').evaluate((el) => getComputedStyle(el).animationName)).toBe('none');
    expect(await live.locator('.now-bar-fill').evaluate((el) => getComputedStyle(el).transitionProperty)).toMatch(/^(none|all)$/);
    const glyph = live.locator('.act.is-moving .phase-glyph:visible').first();
    await expect(glyph).toBeVisible();
    expect(await glyph.evaluate((el) => getComputedStyle(el).animationName)).toBe('none');
    const running = await page.evaluate(() => document.getAnimations().filter((a) => a.playState === 'running').length);
    expect(running).toBe(0);
  });

  test('four events a second with eight rows and twenty stops: one sparkline point a second and no long task over 50 ms', { tag: '@mock' }, async ({ page, request }) => {
    test.skip(!IS_MOCK, 'needs the mock scenario');
    await scenario(request, 'stops');
    await scenario(request, 'busy');
    await scenario(request, 'constrained', { at: 1500 });
    await scenario(request, 'agent-turn', { at: 6000 });
    await login(page, '#/performance');
    const live = page.locator('.live');
    await expect(live.locator('.live-state')).toContainText('4/s', { timeout: 10_000 });
    expect(await live.locator(isPhone(page) ? '.live-card' : '.live-row').count()).toBeGreaterThanOrEqual(6);
    expect(await live.locator('.recent-row').count()).toBeGreaterThanOrEqual(10);
    const soak = Number(process.env.PW_SOAK_S ?? 12) * 1000;
    const longTasks = await page.evaluate(async (ms) => {
      const tasks: number[] = [];
      let supported = true;
      try {
        const obs = new PerformanceObserver((list) => {
          for (const e of list.getEntries()) tasks.push(e.duration);
        });
        obs.observe({ type: 'longtask', buffered: true });
      } catch {
        supported = false; // WebKit has no longtask entries: fall back to a frame-gap watch below
      }
      const gaps: number[] = [];
      let last = performance.now();
      await new Promise<void>((resolve) => {
        const end = performance.now() + ms;
        const step = () => {
          const t = performance.now();
          gaps.push(t - last);
          last = t;
          if (t < end) requestAnimationFrame(step);
          else resolve();
        };
        requestAnimationFrame(step);
      });
      return { supported, tasks, maxGap: Math.max(...gaps) };
    }, soak);
    if (longTasks.supported) expect(longTasks.tasks.filter((d) => d > 50)).toEqual([]);
    // a frame gap above 50 ms + one 16.7 ms frame means a task blocked the thread past the budget
    expect(longTasks.maxGap).toBeLessThan(120);
    // the ring gained about one point a second, not four (the second sparkline has dots, count the ribbon's)
    const points = await live.locator('#live-decode qse-spark').evaluate((el) => (el as unknown as { values: (number | null)[] }).values.filter((v) => v != null).length);
    expect(points).toBeGreaterThan(0);
    expect(points).toBeLessThanOrEqual(300);
  });

  test('the debug overlay logs event gaps', { tag: '@mock' }, async ({ page, request }) => {
    test.skip(!IS_MOCK, 'needs the mock scenario');
    await scenario(request, 'agent-turn', { at: 6000 });
    const logs: string[] = [];
    page.on('console', (m) => m.text().startsWith('[live] seq') && logs.push(m.text()));
    await login(page, '#/performance?debug=live');
    await expect(page.locator('.live-debug')).toBeVisible({ timeout: 10_000 });
    await expect.poll(() => logs.length, { timeout: 8000 }).toBeGreaterThan(4);
    await expect(page.locator('.live-debug-gaps span').first()).toBeVisible();
  });

  test('a real request on the fake engine or the box: the states the page shows', async ({ page, request }) => {
    test.skip(IS_MOCK, 'the mock has its scenarios; this walks a real engine');
    await login(page, '#/performance');
    const live = page.locator('.live');
    await expect(live.locator('#live-now')).toBeVisible({ timeout: 10_000 });
    const seen = new Set<string>();
    const done = streamChat(request, IS_FAKE ? 'FAKE_SLOW explain the engine in detail please' : 'Think about why speculative decoding helps, then say what the weather tool would need. Then call get_weather for Berlin.', IS_FAKE ? 120 : 600, !IS_FAKE);
    const t0 = Date.now();
    while (Date.now() - t0 < 120_000) {
      const h = (await live.locator('.now-headline').textContent())?.trim() ?? '';
      if (h) seen.add(h.replace(/[\d,]+ of [\d,]+ \(\d+ %\)|[\d,]+ tokens$/, 'N'));
      if (h === 'Idle' && seen.size > 1) break;
      await page.waitForTimeout(100);
    }
    await done;
    const words = [...seen].join(' | ');
    expect(words).toMatch(/Prefilling/);
    expect(words).toMatch(/Thinking|Writing/);
    if (!IS_FAKE) expect(words).toMatch(/Calling tool get_weather|Waiting for client: running tool get_weather/);
    await expect(live.locator('.recent-sentence').first()).toContainText(/finished|tool call|stopped/, { timeout: 10_000 });
  });
});
