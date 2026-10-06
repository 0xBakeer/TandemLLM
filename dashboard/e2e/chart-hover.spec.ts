// Hovering a chart never changes the height of the chart, its panel or the panel's grid row
// (VIS-62). Two causes were found on 2026-10-07:
// - `.grid-2-wide .panel { min-height: 100% }`: a percentage min-height on a grid item resolves
//   against its own row, a cycle. Chromium treats it as auto; WebKit (Safari) resolves it against
//   the last layout's row height, so every re-render of the row (each mousemove over a ribbon
//   redraws its crosshair and tip) grew the row by 16 px and it never shrank back. Both
//   #/conductor rows and the #/performance scatter row had it.
// - the heatmap's day readout sits in the flow under the grid: on a phone the full readout wraps
//   to three lines where the hint took one, so the Usage panel grew while a day was read.
// Every chart type is hovered at several points (heatmap cells and scatter dots by their own
// targets), with the heights of the chart, its panel and the panel's grid row read at each step
// and after the pointer left. Mock tier; the charts' data does not change while the test hovers.
import { expect, test, type Page } from '@playwright/test';
import { IS_MOCK, login } from './helpers';

type Frame = Record<string, number>;

/** heights of the chart host, its panel and, when the panel sits in a grid, the grid row */
async function frame(page: Page, sel: string, idx: number): Promise<Frame> {
  return page.evaluate(
    ([sel, idx]) => {
      const host = document.querySelectorAll(sel as string)[idx as number] as HTMLElement | undefined;
      if (!host) return {};
      const out: Record<string, number> = {};
      const h = (el: Element | null | undefined) => (el ? Math.round(el.getBoundingClientRect().height * 10) / 10 : -1);
      out.chart = h(host);
      const panel = host.closest('.panel, .live-fig');
      out.panel = h(panel);
      // the row of panels side by side (not the page's own column, which polls may lengthen)
      const row = panel?.parentElement;
      if (row && /\bgrid-[23]\b/.test(row.className)) {
        out.row = h(row);
        [...row.children].forEach((c, i) => (out[`row.${i}`] = h(c)));
      }
      return out;
    },
    [sel, idx] as const,
  );
}

/** points to hover: fractions of the chart box, or the chart's own targets */
async function targets(page: Page, sel: string, idx: number): Promise<{ x: number; y: number }[]> {
  const host = page.locator(sel).nth(idx);
  const box = (await host.boundingBox())!;
  const own = sel === 'qse-heatmap' ? host.locator('rect.heat-cell[data-level]:not([data-level="0"])') : sel === 'qse-scatter' ? host.locator('.dot') : null;
  const pts: { x: number; y: number }[] = [];
  if (own) {
    const n = await own.count();
    for (const k of [0, Math.floor(n / 2), n - 1]) {
      const b = n ? await own.nth(k).boundingBox() : null;
      if (b) pts.push({ x: b.x + b.width / 2, y: b.y + b.height / 2 });
    }
  }
  for (const [fx, fy] of [
    [0.1, 0.4],
    [0.35, 0.3],
    [0.5, 0.5],
    [0.7, 0.2],
    [0.95, 0.6],
    [0.5, 0.9],
  ])
    pts.push({ x: box.x + box.width * fx, y: box.y + Math.min(box.height * fy, box.height - 4) });
  return pts;
}

const PAGES: { hash: string; charts: string[] }[] = [
  { hash: '#/conductor?screen=overview', charts: ['qse-ribbon'] },
  { hash: '#/usage', charts: ['qse-heatmap', 'qse-bars', 'qse-ribbon'] },
  { hash: '#/performance', charts: ['qse-scatter', 'qse-spark'] },
  { hash: '#/performance?tab=parallel', charts: ['qse-stack'] },
];

test.describe('hovering a chart keeps its height', () => {
  test.skip(!IS_MOCK, 'the mock data holds still while the test hovers');

  for (const { hash, charts } of PAGES) {
    test(`${hash}: ${charts.join(', ')}`, { tag: '@mock' }, async ({ page }) => {
      test.setTimeout(90_000);
      await login(page, hash);
      for (const sel of charts) await expect(page.locator(sel).first()).toBeVisible({ timeout: 10_000 });
      await page.waitForTimeout(1500); // first polls in, the load-time renders done
      const grew: string[] = [];
      for (const sel of charts) {
        const count = await page.locator(sel).count();
        for (let i = 0; i < count; i++) {
          const host = page.locator(sel).nth(i);
          if (!(await host.isVisible())) continue;
          await host.scrollIntoViewIfNeeded();
          await page.mouse.move(1, 1);
          await page.waitForTimeout(150);
          const before = await frame(page, sel, i);
          const check = (when: string, now: Frame) => {
            for (const [k, v] of Object.entries(before)) if (Math.abs((now[k] ?? -1) - v) > 1) grew.push(`${sel}[${i}] ${k} ${when}: ${v} -> ${now[k]}`);
          };
          for (const p of await targets(page, sel, i)) {
            await page.mouse.move(p.x, p.y, { steps: 3 });
            await page.waitForTimeout(120);
            check(`hover ${Math.round(p.x)},${Math.round(p.y)}`, await frame(page, sel, i));
          }
          await page.mouse.move(1, 1, { steps: 2 });
          await page.waitForTimeout(200);
          check('after', await frame(page, sel, i));
        }
      }
      expect(grew, grew.join('\n')).toEqual([]);
    });
  }

  test('a tap on a heatmap day keeps the Usage panel height (phone)', { tag: '@mock' }, async ({ page, isMobile }) => {
    test.skip(!isMobile, 'touch, phone width');
    await login(page, '#/usage');
    const heat = page.locator('qse-heatmap');
    await expect(heat).toBeVisible({ timeout: 10_000 });
    await page.waitForTimeout(800);
    const before = await frame(page, 'qse-heatmap', 0);
    const cells = heat.locator('rect.heat-cell[data-level]:not([data-level="0"])');
    await cells.last().scrollIntoViewIfNeeded();
    await cells.last().tap();
    await expect(heat.locator('.heat-tip-date').first()).toBeVisible();
    const after = await frame(page, 'qse-heatmap', 0);
    expect(after.chart).toBeCloseTo(before.chart, 0);
    expect(after.panel).toBeCloseTo(before.panel, 0);
  });
});
