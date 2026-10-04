// Performance → Parallel: every request in flight on every backend at once. On the mock's
// simulator (mock/parallel.ts, 14 streams over six backends): the aggregate, the per-backend cards,
// the stacked chart, the lanes; rows keep their place between polls; "not reported" where a
// backend gives no live count; idle; no horizontal scroll at 390 px; the hero does not move.
import { expect, test } from '@playwright/test';
import { IS_MOCK, login, noHorizontalOverflow } from './helpers';

test.describe('parallel view', () => {
  test.beforeEach(async ({ request }) => {
    test.skip(!IS_MOCK, 'the mock simulates the parallel load');
    await request.post('/__mock/parallel', { data: { reset: true, streams: 14 } });
  });
  test.afterEach(async ({ request }) => {
    if (IS_MOCK) await request.post('/__mock/parallel', { data: { reset: true, streams: 14 } });
  });

  test('the tab, the aggregate, the backends, the chart and the lanes', async ({ page }) => {
    await login(page, '#/performance');
    await page.getByRole('tab', { name: 'Parallel' }).click();
    await expect(page).toHaveURL(/tab=parallel/);
    const par = page.locator('.par');
    await expect(par).toHaveAttribute('data-in-flight', '14');
    await expect(page.locator('[data-total-tps] .num')).toHaveText(/^\d[\d,.]*$/, { timeout: 10_000 });
    for (const id of ['vllm1', 'sglang1', 'ds', 'oa', 'fake']) {
      await expect(page.locator(`.par-bk[data-backend="${id}"]`)).toBeVisible({ timeout: 10_000 });
    }
    await expect(page.locator('.par-bk[data-backend="ds"] .par-kind')).toHaveText('DeepSeek');
    // the chart draws one layer per backend once two samples exist
    await expect(page.locator('qse-stack path.stack-area').first()).toBeVisible({ timeout: 8000 });
    // the lanes: every request in flight has a row
    expect(await page.locator('.par-row[data-lane]:not(.is-done)').count()).toBe(14);
    // a backend whose frames were never checked has no live count: it says so
    await expect(page.locator('.par-row[data-backend="oa"] .par-nr').first()).toBeVisible({ timeout: 20_000 });
    // the KPIs
    await expect(page.locator('[data-kpi="in-flight"] .stat-value')).toHaveText('14');
    await expect(page.locator('[data-kpi="ttft"] .stat-value')).toContainText(/ms|s/, { timeout: 15_000 });
    expect(await noHorizontalOverflow(page)).toBe(true);
    // back to the engine's page
    await page.getByRole('tab', { name: 'Engine' }).click();
    await expect(page.locator('qse-performance')).toBeVisible();
  });

  test('rows keep their place between polls', async ({ page }) => {
    await login(page, '#/performance?tab=parallel');
    await expect(page.locator('.par')).toHaveAttribute('data-in-flight', '14');
    const rows = async () => page.locator('.par-table > .par-row:not(.par-row-head)').evaluateAll((els) => els.map((e) => (e as HTMLElement).dataset.lane ?? null));
    const a = await rows();
    await page.waitForTimeout(2500);
    const b = await rows();
    let same = 0;
    a.forEach((id, i) => {
      if (id && b.includes(id)) {
        expect(b.indexOf(id), `lane ${id} moved`).toBe(i);
        same++;
      }
    });
    expect(same).toBeGreaterThan(5);
  });

  test('the hero does not move while the numbers change', async ({ page }) => {
    await login(page, '#/performance?tab=parallel');
    await expect(page.locator('qse-stack path.stack-area').first()).toBeVisible({ timeout: 8000 });
    const box1 = await page.locator('.par-hero').boundingBox();
    const k1 = await page.locator('.par-kpis').boundingBox();
    await page.waitForTimeout(3200);
    const box2 = await page.locator('.par-hero').boundingBox();
    const k2 = await page.locator('.par-kpis').boundingBox();
    expect(Math.abs((box2?.height ?? 0) - (box1?.height ?? 0))).toBeLessThanOrEqual(1);
    expect(Math.abs((k2?.y ?? 0) - (k1?.y ?? 0))).toBeLessThanOrEqual(1);
  });

  test('idle: nothing in flight', async ({ page, request }) => {
    await request.post('/__mock/parallel', { data: { reset: true, streams: 0 } });
    await login(page, '#/performance?tab=parallel');
    await expect(page.locator('.par-empty')).toContainText('Nothing in flight');
    await expect(page.locator('[data-total-tps] .num')).toHaveText('0');
    expect(await noHorizontalOverflow(page)).toBe(true);
  });
});
