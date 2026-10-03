// speed, live and over time.
import { expect, test } from '@playwright/test';
import { IS_FAKE, IS_MOCK, login, setMode } from './helpers';

test.describe('performance', () => {
  test.beforeEach(async ({ request }) => {
    if (IS_MOCK) await setMode(request, 'ok');
  });

  test('the 5-minute strip under the Live panel: TTFT, tokens per block, acceptance from /metrics', async ({ page }) => {
    const metricsCalls: string[] = [];
    page.on('request', (r) => r.url().endsWith('/metrics') && metricsCalls.push(r.url()));
    await login(page, '#/performance');
    // the live panel is first (e2e/live.spec.ts); the /metrics figures follow it
    await expect(page.locator('.view-performance > qse-live-panel .live')).toBeVisible();
    const tiles = page.locator('.live-metrics .stat');
    await expect(tiles).toHaveCount(3);
    await expect(tiles.nth(0).locator('.stat-label')).toContainText(/time to first token/);
    await expect(tiles.nth(1)).toContainText('tokens per block');
    await expect(tiles.nth(2)).toContainText('draft acceptance');
    if (!IS_FAKE) {
      // the fake engine has no speculation families: these two say "not reported" there
      await expect(tiles.nth(1).locator('.stat-value')).toContainText(/\d\.\d\d/);
      await expect(tiles.nth(2).locator('.stat-value')).toContainText(/\d/);
    }
    await expect.poll(() => metricsCalls.length, { timeout: 12_000 }).toBeGreaterThanOrEqual(2);
  });

  test('feature off: without speculation families the tiles say "not reported"', { tag: '@mock' }, async ({ page, request }) => {
    test.skip(!IS_MOCK, 'needs the mock failure switch');
    await setMode(request, 'nospec');
    await login(page, '#/performance');
    await expect(page.locator('.live-metrics .stat').nth(1)).toContainText('not reported');
    await expect(page.locator('.live-metrics .stat').nth(2)).toContainText('not reported');
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

  test('the timing breakdown says where the prompt came from, what it reused and forwarded', async ({ page }) => {
    await login(page, '#/performance');
    await page.locator('#scatter .dot').first().click({ force: true });
    const kv = page.locator('#timing .kv');
    await expect(kv).toContainText('cache');
    await expect(kv.locator('.cache-tag')).toHaveText(/RAM end anchor|RAM resident|RAM state store|NVMe disk|cold|response cache/);
    await expect(kv).toContainText('reused');
    await expect(kv).toContainText('forwarded');
  });

  test('cache sources, last hour: one row per tier with requests, reused and forwarded tokens', async ({ page }) => {
    await login(page, '#/performance');
    const box = page.locator('#cache-sources');
    await expect(box.locator('.panel-title')).toHaveText('Cache sources, last hour');
    const rows = box.locator('.cache-sum tbody tr');
    await expect(rows.first()).toBeVisible();
    const n = await rows.count();
    expect(n).toBeGreaterThan(0);
    for (let i = 0; i < n; i++) {
      await expect(rows.nth(i).locator('.cache-tag')).toHaveText(/RAM end anchor|RAM resident|RAM state store|NVMe disk|cold|response cache/);
      await expect(rows.nth(i).locator('td.num').first()).toHaveText(/^[\d,]+$/);
    }
    await expect(box.locator('.cache-sum-bar span')).toHaveCount(n);
    await expect(box.locator('.panel-foot')).toContainText('of their prompt tokens came out of a cache');
    // no horizontal scroll at 390 px: the table fits its panel
    const over = await box.evaluate((el) => el.scrollWidth - el.clientWidth);
    expect(over).toBeLessThanOrEqual(1);
  });

  test('the Last 20 table: a cache cell per row, the tier and reused / new tokens', { tag: '@mock' }, async ({ page, request }) => {
    test.skip(!IS_MOCK, 'needs the mock scenario');
    const res = await request.post('/__mock/live', { data: { scenario: 'stops' } });
    expect(res.ok()).toBeTruthy();
    await login(page, '#/performance');
    const rows = page.locator('.live .recent-row');
    await expect(rows).toHaveCount(10);
    // the ones that ran: the source and the split; refused and rejected never ran, the abandoned
    // one left during its prefill: a dash
    const cells = rows.locator('td.rc-cache');
    const ran = rows.filter({ has: page.locator('td.rc-cache .cache-tag') });
    await expect(ran).toHaveCount(7);
    await expect(ran.first().locator('.cache-tag')).toHaveText('cold');
    await expect(ran.first().locator('.cache-split')).toHaveText('0 reused · 8.2K new');
    await expect(ran.first().locator('.cache-tag')).toHaveAttribute('title', /cold: 0 of 8,192 prompt tokens reused, 8,192 forwarded/);
    await expect(cells.nth(4)).toHaveText('—');
    await expect(cells.nth(6)).toHaveText('—');
    await expect(cells.nth(7)).toHaveText('—');
    await expect(ran.first().locator('td.rc-cache')).toBeVisible();
    await request.post('/__mock/live', { data: { scenario: 'clear' } });
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
