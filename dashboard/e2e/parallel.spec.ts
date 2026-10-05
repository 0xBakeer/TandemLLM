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
    if (IS_MOCK) await request.post('/__mock/parallel', { data: { reset: true, streams: 14, engine_slots: 1 } });
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

  test("the engine's sequence slots: a slot tag per engine lane, the slots in the engine card", async ({ page, request }) => {
    // server --parallel 3: the engine serves up to three requests at once (server/slots.py)
    await request.post('/__mock/parallel', { data: { reset: true, streams: 14, engine_slots: 3 } });
    try {
      await login(page, '#/performance?tab=parallel');
      const card = page.locator('.par-bk[data-backend="engine"] .par-slots');
      await expect(card).toBeVisible({ timeout: 20_000 });
      await expect(card.locator('.par-slot')).toHaveCount(3);
      await expect(card.locator('.par-slot[data-slot="0"] .par-slot-w')).toHaveText('256k');
      await expect(card.locator('.par-slot[data-slot="1"] .par-slot-w')).toHaveText('64k');
      await expect(page.locator('.par-row[data-backend="engine"] .par-slot-tag').first()).toBeVisible({ timeout: 20_000 });
      await expect(card.locator('.par-slots-foot')).toContainText(/\d-way \d+/, { timeout: 20_000 });
      expect(await noHorizontalOverflow(page)).toBe(true);
      await page.screenshot({ path: test.info().outputPath('parallel-engine-slots.png'), fullPage: true });
    } finally {
      await request.post('/__mock/parallel', { data: { reset: true, streams: 14, engine_slots: 1 } });
    }
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

  test('a backend card keeps its height when it has many streams', async ({ page, request }) => {
    // T1 on the box, N = 32: 24 streams on one backend wrapped its dots to a second line and every
    // lane below the cards moved (layout shift 0.14 in 20 s)
    await request.post('/__mock/parallel', { data: { reset: true, streams: 48 } });
    await login(page, '#/performance?tab=parallel');
    await expect(page.locator('.par')).toHaveAttribute('data-in-flight', '48', { timeout: 10_000 });
    const rows = page.locator('.par-bk-dots');
    await expect(rows.first()).toBeVisible();
    for (const h of await rows.evaluateAll((els) => els.map((e) => e.getBoundingClientRect().height))) {
      expect(h).toBeLessThan(24);
    }
    // the count is never cut by the dots
    const cut = await page.locator('.par-bk-count').evaluateAll((els) =>
      els.filter((e) => e.scrollWidth > e.clientWidth + 1 || e.getBoundingClientRect().right > e.closest('.par-bk')!.getBoundingClientRect().right).length);
    expect(cut).toBe(0);
  });
});
