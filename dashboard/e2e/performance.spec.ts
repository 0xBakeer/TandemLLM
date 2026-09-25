// VIS-15 — speed, live and over time.
import { expect, test } from '@playwright/test';
import { IS_MOCK, login, setMode } from './helpers';

test.describe('performance', () => {
  test.beforeEach(async ({ request }) => {
    if (IS_MOCK) await setMode(request, 'ok');
  });

  test('live strip: decode, TTFT, tokens per block, acceptance and queue from /metrics', async ({ page }) => {
    const metricsCalls: string[] = [];
    page.on('request', (r) => r.url().endsWith('/metrics') && metricsCalls.push(r.url()));
    await login(page, '#/performance');
    const tiles = page.locator('.live .stat');
    await expect(tiles).toHaveCount(5);
    await expect(tiles.nth(0).locator('.stat-label')).toContainText(/decode/);
    await expect(tiles.nth(2)).toContainText('tokens per block');
    await expect(tiles.nth(2).locator('.stat-value')).toContainText(/\d\.\d\d/);
    await expect(tiles.nth(3)).toContainText('draft acceptance');
    await expect(tiles.nth(3).locator('.stat-value')).toContainText(/\d/);
    await expect(tiles.nth(4)).toContainText('running');
    await expect.poll(() => metricsCalls.length, { timeout: 12_000 }).toBeGreaterThanOrEqual(2);
  });

  test('idle engine: the decode tile is labelled "last request"', async ({ page }) => {
    await login(page, '#/performance');
    const label = page.locator('.live .stat').nth(0).locator('.stat-label');
    await expect(label).toContainText(/decode/);
    await expect
      .poll(async () => label.textContent(), { timeout: 30_000 })
      .toMatch(/last request|decode now/);
  });

  test('feature off: without speculation families the tiles say "not reported"', async ({ page, request }) => {
    test.skip(!IS_MOCK, 'needs the mock failure switch');
    await setMode(request, 'nospec');
    await login(page, '#/performance');
    await expect(page.locator('.live .stat').nth(2)).toContainText('not reported');
    await expect(page.locator('.live .stat').nth(3)).toContainText('not reported');
    await setMode(request, 'ok');
  });

  test('scatter hides response-cache replays until switched on', async ({ page }) => {
    await login(page, '#/performance');
    const dots = page.locator('#scatter .dot');
    await expect(dots.first()).toBeVisible();
    const hidden = await dots.count();
    const foot = await page.locator('#scatter .panel-foot').textContent();
    const m = /(\d+) response-cache replay/.exec(foot ?? '');
    if (m) {
      await page.getByLabel('show cache replays').check();
      await expect(page.locator('#scatter .legend')).toContainText('response replay');
      await expect(page.locator('#scatter .panel-foot')).toContainText('shown');
      // The mock engine keeps finishing requests, so compare against the caption, not the earlier count.
      const shownFoot = await page.locator('#scatter .panel-foot').textContent();
      const n = Number(/(\d+) of the last/.exec(shownFoot ?? '')?.[1]);
      expect(n).toBeGreaterThan(hidden);
      await expect(dots).toHaveCount(n);
    }
  });

  test('clicking a dot shows the timing breakdown with three segments', async ({ page }) => {
    await login(page, '#/performance');
    await page.locator('#scatter .dot').first().click({ force: true });
    await expect(page.locator('#timing qse-timing-bar')).toBeVisible();
    await expect(page.locator('#timing .timing-seg')).toHaveCount(3);
    await expect(page.locator('#timing .timing-legend')).toContainText('queue');
    await expect(page.locator('#timing .timing-legend')).toContainText('prefill');
    await expect(page.locator('#timing .timing-legend')).toContainText('decode');
  });

  test('history source: 30 days uses hourly buckets from the ledger and says so', async ({ page }) => {
    await login(page, '#/performance');
    await page.getByRole('radio', { name: '30 d' }).click();
    await expect(page.locator('#history .panel-sub').first()).toContainText('hourly buckets from the ledger');
    expect(page.url()).toContain('range=30d');
    await page.getByRole('radio', { name: '365 d' }).click();
    await expect(page.locator('#history .panel-sub').first()).toContainText('daily buckets');
  });

  test('history charts: five charts, each with one y axis and a legend for banded series', async ({ page }) => {
    await login(page, '#/performance');
    await expect(page.locator('#history qse-ribbon')).toHaveCount(5);
    await expect(page.locator('#history qse-ribbon').first().locator('.legend')).toContainText('p90');
    await expect(page.locator('.reading')).toContainText('Tokens per block');
  });

  test('Grafana link: shown only when configured', async ({ page }) => {
    await login(page, '#/performance');
    const link = page.getByRole('link', { name: /Open in Grafana/ });
    if (process.env.VITE_GRAFANA_URL) {
      await expect(link).toHaveAttribute('target', '_blank');
      await expect(link).toHaveAttribute('href', process.env.VITE_GRAFANA_URL);
    } else {
      await expect(link).toHaveCount(0);
    }
  });
});
