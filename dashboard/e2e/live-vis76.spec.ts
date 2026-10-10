// VIS-76: the Perf tab on the 0.6.x feed. A prefill with one mark per 8,192-row window
// (`POST /__mock/live {scenario: "rates", case: "exl3-window"}`): the Now card's bar is filled to
// the newest mark in the prefill colour and a hatched estimate moves on past it, said to be an
// estimate, while the headline keeps the measured figure; the tile counts the card's total.
// The Parallel view on the 09:52 shape (`POST /__mock/parallel {fixture: "stall"}`): a stall is
// said as one, the In flight tile speaks the legend's words, a lane held for memory is waiting.
import { expect, test, type Page } from '@playwright/test';
import { IS_MOCK, login, noHorizontalOverflow } from './helpers';

async function nowCard(page: Page) {
  return page.evaluate(() => {
    const card = document.querySelector('#live-now') as HTMLElement | null;
    const fill = card?.querySelector('.now-bar-fill') as HTMLElement | null;
    const est = card?.querySelector('[data-estimate]') as HTMLElement | null;
    const track = card?.querySelector('.now-bar') as HTMLElement | null;
    return {
      headline: card?.querySelector('.now-headline')?.textContent?.trim() ?? '',
      numbers: card?.querySelector('.now-numbers')?.textContent?.trim() ?? '',
      tone: card ? getComputedStyle(card).getPropertyValue('--tone').trim() : '',
      fillBg: fill ? getComputedStyle(fill).backgroundColor : '',
      trackBg: track ? getComputedStyle(track).backgroundColor : '',
      fillW: fill?.getBoundingClientRect().width ?? 0,
      trackW: track?.getBoundingClientRect().width ?? 0,
      estW: est?.getBoundingClientRect().width ?? 0,
      tile: document.querySelector('#live-prefill .live-fig-sub')?.textContent?.trim() ?? '',
    };
  });
}

test.describe('VIS-76: a prefill between window marks (mock)', () => {
  test.skip(!IS_MOCK, 'the mock plays the windowed prefill');
  test.afterEach(async ({ request }) => {
    await request.post('/__mock/live', { data: { scenario: 'clear', simulator: true, liveRates: true } });
  });

  test('the bar is filled to the mark in the prefill colour, a hatched estimate moves on, the headline stays measured', { tag: '@mock' }, async ({ page, request }) => {
    expect((await request.post('/__mock/live', { data: { scenario: 'clear', simulator: false, liveRates: true } })).ok()).toBeTruthy();
    // 17 s in: the first mark (18,432 of 44,558) came at 12.4 s, the second comes at ~23 s
    expect((await request.post('/__mock/live', { data: { scenario: 'rates', case: 'exl3-window', at: 17_000 } })).ok()).toBeTruthy();
    await login(page, '#/performance');
    await expect(page.locator('.now-headline')).toHaveText('Prefilling 18,432 of 44,558 (41 %)', { timeout: 10_000 });
    const a = await nowCard(page);
    // the Live panel's own tone (parallel.css no longer resets it): a coloured fill on a grey track
    expect(a.tone).not.toBe('');
    expect(a.fillBg).not.toBe('rgba(0, 0, 0, 0)');
    expect(a.fillBg).not.toBe(a.trackBg);
    expect(a.fillW / a.trackW).toBeGreaterThan(0.39);
    expect(a.fillW / a.trackW).toBeLessThan(0.43);
    expect(a.estW).toBeGreaterThan(0);
    expect(a.numbers).toMatch(/^≈ \d+ % est\. · about \d+ s left · 10,240 cached$/);
    expect(a.tile).toContain('prefilling 18,432 of 44,558 · 41 %');
    // two seconds later the estimate has moved and the ETA counted down; the measured part has not
    await page.waitForTimeout(2000);
    const b = await nowCard(page);
    expect(b.headline).toBe(a.headline);
    expect(Math.abs(b.fillW - a.fillW)).toBeLessThan(1);
    expect(b.estW).toBeGreaterThan(a.estW);
    const eta = (s: string) => Number(/about (\d+) s left/.exec(s)?.[1]);
    expect(eta(b.numbers)).toBeLessThan(eta(a.numbers));
    // the second mark: the live rate replaces the words, the solid part jumps to the mark
    await expect(page.locator('.now-headline')).toHaveText('Prefilling 26,624 of 44,558 (60 %)', { timeout: 10_000 });
    await expect(page.locator('.now-numbers')).toContainText('775 tok/s');
    await expect(page.locator('#live-prefill .live-fig-num')).toHaveText('775');
    expect(await noHorizontalOverflow(page)).toBe(true);
  });

  test('before the first mark: the restored share is filled, the tile counts the card\'s total', { tag: '@mock' }, async ({ page, request }) => {
    expect((await request.post('/__mock/live', { data: { scenario: 'clear', simulator: false, liveRates: true } })).ok()).toBeTruthy();
    expect((await request.post('/__mock/live', { data: { scenario: 'rates', case: 'exl3-window', at: 4000 } })).ok()).toBeTruthy();
    await login(page, '#/performance');
    await expect(page.locator('.now-headline')).toHaveText('Prefilling 44,558 tokens', { timeout: 10_000 });
    const s = await nowCard(page);
    expect(s.fillW / s.trackW).toBeGreaterThan(0.2);
    expect(s.fillW / s.trackW).toBeLessThan(0.25);
    expect(s.estW).toBe(0);
    expect(s.tile).toMatch(/^warming up · prefilling 44,558 tokens · \d/);
  });
});

test.describe('VIS-76: the Parallel view on the 0.6.x stall shape (mock)', () => {
  test.skip(!IS_MOCK, 'the mock serves the fixture');
  test.afterEach(async ({ request }) => {
    await request.post('/__mock/parallel', { data: { fixture: null, reset: true, streams: 14, engine_slots: 1 } });
  });

  test('an aggregate 0 with three decoding says the stall; the In flight tile matches the legend', { tag: '@mock' }, async ({ page, request }) => {
    expect((await request.post('/__mock/parallel', { data: { fixture: 'stall' } })).ok()).toBeTruthy();
    await login(page, '#/performance?tab=parallel');
    await expect(page.locator('[data-stall]')).toHaveText('3 decoding, stalled behind a prefill (41 % of 44,558)', { timeout: 10_000 });
    await expect(page.locator('[data-total-tps] .num')).toHaveText('0.0');
    await expect(page.locator('[data-kpi="in-flight"] .stat-sub')).toHaveText('1 waiting · 1 prefilling · 3 decoding');
    await expect(page.locator('.par-mix-item[data-tone="wait"] b')).toHaveText('1');
    await expect(page.locator('.par-mix-item[data-tone="prefill"] b')).toHaveText('1');
    await expect(page.locator('.par-mix-item[data-tone="write"] b')).toHaveText('1');
    // the lane held for memory: waiting, in the waiting tone, with its own words
    const mem = page.locator('.par-row[data-state="waiting_memory"]');
    await expect(mem).toHaveClass(/tone-wait/);
    await expect(mem.locator('.par-state-word')).toHaveText('waiting for memory');
    expect(await noHorizontalOverflow(page)).toBe(true);
  });
});
