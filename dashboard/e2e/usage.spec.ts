// VIS-14 — a year of tokens at a glance.
import { expect, test } from '@playwright/test';
import { IS_MOCK, login, noHorizontalOverflow, setMode } from './helpers';

test.describe('usage', () => {
  test.beforeEach(async ({ request }) => {
    if (IS_MOCK) await setMode(request, 'ok');
  });

  test('heatmap of the year: 365 cells in ≤ 53 Monday-first columns, today outlined', async ({ page }) => {
    await login(page);
    const cells = page.locator('.heat-cell');
    await expect(cells).toHaveCount(365);
    const cols = await page.evaluate(() => new Set(Array.from(document.querySelectorAll('.heat-cell')).map((c) => c.getAttribute('x'))).size);
    expect(cols).toBeGreaterThanOrEqual(52);
    expect(cols).toBeLessThanOrEqual(53);
    await expect(page.locator('.heat-cell.is-today')).toHaveCount(1);
    const today = new Date().toLocaleDateString('sv-SE', { timeZone: 'Europe/Berlin' });
    await expect(page.locator('.heat-cell.is-today')).toHaveAttribute('data-date', today);
    // Monday first: the weekday labels start with Mon
    await expect(page.locator('.heat-svg text').filter({ hasText: 'Mon' })).toHaveCount(1);
  });

  test('colour scale: five quantile steps, zero days empty', async ({ page }) => {
    await login(page);
    await expect(page.locator('.heat-cell')).toHaveCount(365);
    const levels = await page.evaluate(() => {
      const counts: Record<string, number> = {};
      for (const c of Array.from(document.querySelectorAll('.heat-cell'))) {
        const l = c.getAttribute('data-level') ?? '0';
        counts[l] = (counts[l] ?? 0) + 1;
      }
      return counts;
    });
    for (const l of ['1', '2', '3', '4', '5']) expect(levels[l], `level ${l}`).toBeGreaterThan(0);
    if (IS_MOCK) expect(levels['0']).toBeGreaterThan(0);
    await expect(page.locator('.heat-legend .heat-swatch')).toHaveCount(5);
  });

  test('day details: focusing a cell shows its tokens, requests and p50 speed', async ({ page }) => {
    await login(page);
    const cells = page.locator('.heat-cell[data-level="5"]');
    await expect(cells.first()).toBeVisible();
    const cell = cells.last();
    const date = await cell.getAttribute('data-date');
    await cell.focus();
    const tip = page.locator('.heat-tip');
    await expect(tip).toContainText('tokens');
    await expect(tip).toContainText('requests');
    await expect(tip).toContainText('cached');
    await expect(tip).toContainText('reasoning');
    await expect(tip).toContainText('p50');
    // keyboard: the arrow moves to the next day
    await page.keyboard.press('ArrowLeft');
    const focused = await page.evaluate(() => document.activeElement?.getAttribute('data-date'));
    expect(focused).not.toBe(date);
    expect(focused).toBeTruthy();
  });

  test('totals: four windows whose split bars sum to the total', async ({ page }) => {
    await login(page);
    const cards = page.locator('.stat-window');
    await expect(cards).toHaveCount(4);
    await expect(cards.nth(0)).toContainText('Today');
    await expect(cards.nth(3)).toContainText('Year');
    for (let i = 0; i < 4; i++) {
      const c = cards.nth(i);
      await expect(c.locator('.stat-value .num')).not.toHaveText('—');
      const widths = await c.locator('.split-seg').evaluateAll((els) => els.map((e) => parseFloat((e as HTMLElement).style.width)));
      const sum = widths.reduce((a, b) => a + b, 0);
      if (widths.length) expect(Math.abs(sum - 100)).toBeLessThan(0.5);
    }
    await expect(page.locator('.hero-number')).not.toHaveText('—');
  });

  test('metric switch recolours by requests', async ({ page }) => {
    await login(page);
    await expect(page.locator('.heat-cell')).toHaveCount(365);
    const before = await page.locator('.heat-cell').evaluateAll((els) => els.map((e) => e.getAttribute('data-level')).join(''));
    await page.getByLabel('Metric').selectOption('requests');
    await expect(page.locator('.panel-heat .panel-title')).toHaveText('Requests per day');
    const after = await page.locator('.heat-cell').evaluateAll((els) => els.map((e) => e.getAttribute('data-level')).join(''));
    expect(after).not.toBe(before);
  });

  test('filter by client: cards, heatmap and charts follow, a chip shows, the hash keeps it', async ({ page }) => {
    await login(page);
    const yearBefore = await page.locator('.hero-number').textContent();
    await page.getByLabel('Client').selectOption({ index: 1 });
    await expect(page.locator('.chip')).toContainText('client');
    expect(page.url()).toContain('client=');
    await expect(page.locator('.hero-caption')).toContainText('filtered');
    await expect(page.locator('.hero-number')).not.toHaveText(yearBefore!);
    await expect(page.locator('.heat-cell')).toHaveCount(365);
    await page.locator('.chip').click();
    await expect(page.locator('.chip')).toHaveCount(0);
    expect(page.url()).not.toContain('client=');
  });

  test('custom range of 45 days: the bars show 45 days, the heatmap still the year', async ({ page }) => {
    await login(page);
    await page.getByRole('radio', { name: 'custom' }).click();
    const today = new Date().toLocaleDateString('sv-SE', { timeZone: 'Europe/Berlin' });
    const from = new Date(Date.now() - 44 * 86400000).toLocaleDateString('sv-SE', { timeZone: 'Europe/Berlin' });
    await page.locator('.toolbar input[type="date"]').nth(0).fill(from);
    await page.locator('.toolbar input[type="date"]').nth(1).fill(today);
    await expect(page.locator('#chart-tokens .panel-sub')).toContainText('–');
    // 45 columns: count distinct x of bars in the tokens chart
    await expect
      .poll(async () => page.evaluate(() => new Set(Array.from(document.querySelectorAll('#chart-tokens rect[width]:not(.hover-band)')).map((r) => r.getAttribute('x'))).size))
      .toBe(45);
    await expect(page.locator('.heat-cell')).toHaveCount(365);
  });

  test('range buttons: 30 d shows 30 bars', async ({ page }) => {
    await login(page);
    await page.getByRole('radio', { name: '30 d' }).click();
    await expect
      .poll(async () => page.evaluate(() => new Set(Array.from(document.querySelectorAll('#chart-tokens rect[width]:not(.hover-band)')).map((r) => r.getAttribute('x'))).size))
      .toBe(30);
    await expect(page.locator('#chart-tokens .panel-sub')).toHaveText('last 30 days');
  });

  test('new ledger: at most three coloured days and a note saying when history starts', async ({ page, request }) => {
    test.skip(!IS_MOCK, 'needs the mock failure switch');
    await setMode(request, 'empty');
    await login(page);
    await expect(page.locator('.note-info')).toContainText('History starts on');
    await expect(page.locator('.heat-cell')).toHaveCount(365);
    const coloured = await page.locator('.heat-cell:not([data-level="0"])').count();
    expect(coloured).toBeLessThanOrEqual(3);
    await setMode(request, 'ok');
  });

  test('top days lists ten days with tokens, requests and a client', async ({ page }) => {
    await login(page);
    await expect(page.locator('#top-days tbody tr')).toHaveCount(10);
    await expect(page.locator('#top-days tbody tr').first()).toContainText(/\d/);
  });

  test('charts have tooltips reachable by keyboard', async ({ page }) => {
    await login(page);
    const chart = page.locator('#chart-requests .chart');
    await chart.focus();
    await page.keyboard.press('ArrowLeft');
    await expect(page.locator('#chart-requests .tip')).toBeVisible();
    await expect(page.locator('#chart-requests .tip')).toContainText('requests');
  });

  test('phone: one-column cards, bottom tab bar, heatmap scrolls with the current week visible', async ({ page }, testInfo) => {
    test.skip(!testInfo.project.name.includes('phone'), 'phone project only');
    await login(page);
    await expect(page.locator('.rail')).toBeVisible();
    const railBox = await page.locator('.rail').boundingBox();
    expect(railBox!.y).toBeGreaterThan(600); // bottom bar
    const cards = page.locator('.stat-window');
    const a = await cards.nth(0).boundingBox();
    const b = await cards.nth(1).boundingBox();
    expect(b!.y).toBeGreaterThan(a!.y + a!.height - 1); // stacked
    expect(await noHorizontalOverflow(page)).toBe(true);
    await expect(page.locator('.heat-cell')).toHaveCount(365);
    const todayVisible = await page.locator('.heat-cell.is-today').evaluate((el) => {
      const s = el.closest('.heat-scroll')!.getBoundingClientRect();
      const r = el.getBoundingClientRect();
      return r.left >= s.left && r.right <= s.right;
    });
    expect(todayVisible).toBe(true);
  });
});
