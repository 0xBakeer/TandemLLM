// the Performance page holds still under the 4 Hz live stream. The mock `loop` scenario
// (an opencode tool loop: seven finished turns with long "continues" notes and one turn walking a
// single-call prefill, thinking, writing and a tool call) changes the Now line, both figures, the
// counts and every row's cells on every event. The test records the boxes of the panel's parts,
// the table columns and every row for ~8 s and asserts none of them moves or resizes; on Chromium
// it also sums the browser's own layout-shift entries (CLS), which must stay ~0.
// PERFUI_SHOTS=<dir> also writes a screenshot early (prefilling) and late (tool call).
import { expect, test, type Page } from '@playwright/test';
import { IS_MOCK, login, setMode } from './helpers';

// below 1100 px the request list is cards
const isCards = (page: Page) => (page.viewportSize()?.width ?? 1440) <= 1100;
type Box = { x: number; y: number; w: number; h: number };

async function boxes(page: Page, cards: boolean): Promise<Record<string, Box>> {
  return page.evaluate((cards) => {
    const out: Record<string, { x: number; y: number; w: number; h: number }> = {};
    const put = (key: string, el: Element | null) => {
      if (!el) return;
      const r = el.getBoundingClientRect();
      out[key] = { x: Math.round(r.x * 10) / 10, y: Math.round((r.y + window.scrollY) * 10) / 10, w: Math.round(r.width * 10) / 10, h: Math.round(r.height * 10) / 10 };
    };
    const live = document.querySelector('.live')!;
    put('head', live.querySelector('.live-head'));
    put('state', live.querySelector('.live-state'));
    put('now', live.querySelector('#live-now'));
    put('decode', live.querySelector('#live-decode'));
    put('decode-spark', live.querySelector('#live-decode qse-spark'));
    put('prefill', live.querySelector('#live-prefill'));
    put('prefill-spark', live.querySelector('#live-prefill qse-spark'));
    live.querySelectorAll('.live-count').forEach((el, i) => put(`count${i}`, el));
    put('list', live.querySelector('.live-list'));
    if (cards) {
      live.querySelectorAll('.live-card').forEach((el, i) => put(`card${i}`, el));
    } else {
      live.querySelectorAll('.live-table th').forEach((el, i) => put(`th${i}`, el));
      live.querySelectorAll('.live-row').forEach((el, i) => put(`row${i}`, el));
      // every cell of the first row (the one in flight) and of the second (a finished one)
      live.querySelectorAll('.live-row:nth-child(-n+2) td').forEach((el, i) => put(`td${i}`, el));
      live.querySelectorAll('.recent-table th').forEach((el, i) => put(`rth${i}`, el));
    }
    put('recent', live.querySelector('.recent'));
    put('metrics', document.querySelector('.live-metrics'));
    put('scatter', document.querySelector('#scatter'));
    return out;
  }, cards);
}

test.describe('the Performance page holds still', () => {
  test.skip(!IS_MOCK, 'needs the mock loop scenario');
  test.afterAll(async ({ request }) => {
    await request.post('/__mock/live', { data: { scenario: 'clear', contract: '1.1', activity: true, simulator: true } });
  });

  const widths: (number | null)[] = [null, 1200, 1024];
  for (const vw of widths) {
  test(`nothing on the page moves while the live stream changes every cell${vw ? ` (${vw} px)` : ''}`, { tag: '@mock' }, async ({ page, request, browserName }) => {
    test.skip(vw != null && browserName !== 'chromium', 'the extra widths run once, on Chromium');
    test.setTimeout(60_000);
    if (vw) await page.setViewportSize({ width: vw, height: 900 });
    const phone = isCards(page);
    await setMode(request, 'ok');
    expect((await request.post('/__mock/live', { data: { scenario: 'clear', contract: '1.1', activity: true, simulator: false } })).ok()).toBeTruthy();
    await login(page, '#/performance');
    await expect(page.locator('.live-state')).toContainText('streaming', { timeout: 10_000 });
    // a request the mock engine started before the simulator was paused would finish mid-watch and
    // add a row to the Last 20 (a real change, not a jump): let it end first (the mock caps one at 20 s)
    await expect(page.locator('.live-count', { hasText: 'in flight' }).locator('dd')).toHaveText('0', { timeout: 25_000 });
    // the browser's own layout-shift entries (Chromium only; WebKit has no Layout Instability API)
    const clsSupported = await page.evaluate(() => {
      const w = window as unknown as { __shifts: { value: number; t: number; sources: string[] }[] };
      w.__shifts = [];
      try {
        new PerformanceObserver((list) => {
          for (const e of list.getEntries() as unknown as { value: number; startTime: number; sources?: { node?: Node }[] }[])
            w.__shifts.push({ value: e.value, t: e.startTime, sources: (e.sources ?? []).map((s) => (s.node instanceof Element ? s.node.className || s.node.tagName : String(s.node?.nodeName))) });
        }).observe({ type: 'layout-shift', buffered: false });
        return PerformanceObserver.supportedEntryTypes.includes('layout-shift');
      } catch {
        return false;
      }
    });
    expect((await request.post('/__mock/live', { data: { scenario: 'loop' } })).ok()).toBeTruthy();
    const items = page.locator(phone ? '.live-card' : '.live-row');
    await expect(items).toHaveCount(8, { timeout: 10_000 });
    await expect(page.locator('.live-state')).toContainText('4/s', { timeout: 10_000 });
    await expect(page.locator('.now-headline')).toContainText(/Prefilling/, { timeout: 5000 });
    await page.waitForTimeout(600); // the first events after the scenario lands; /metrics tiles filled
    await expect(page.locator('.live-metrics .stats')).toBeVisible();
    const t0 = await page.evaluate(() => performance.now());

    const shots = process.env.PERFUI_SHOTS;
    const tag = `${test.info().project.name}${vw ? `-${vw}` : ''}`;
    const first = await boxes(page, phone);
    if (shots) await page.screenshot({ path: `${shots}/${tag}-1-prefilling.png`, fullPage: true });
    const headlines = new Set<string>();
    const moved: string[] = [];
    for (let i = 0; i < 32; i++) {
      await page.waitForTimeout(250);
      headlines.add((await page.locator('.now-headline').textContent())?.trim().split(' ')[0] ?? '');
      const b = await boxes(page, phone);
      for (const [k, v] of Object.entries(first)) {
        const n = b[k];
        if (!n) {
          moved.push(`${k} gone at tick ${i}`);
          continue;
        }
        const d = Math.max(Math.abs(n.x - v.x), Math.abs(n.y - v.y), Math.abs(n.w - v.w), Math.abs(n.h - v.h));
        if (d > 1) moved.push(`${k} tick ${i}: ${JSON.stringify(v)} -> ${JSON.stringify(n)}`);
      }
    }
    if (shots) await page.screenshot({ path: `${shots}/${tag}-2-toolcall.png`, fullPage: true });
    // the scenario really did walk the states while we watched
    expect([...headlines].join(' ')).toMatch(/Prefilling/);
    expect([...headlines].join(' ')).toMatch(/Thinking|Writing|Calling/);
    let cls: number | null = null;
    let shifts: { value: number; t: number; sources: string[] }[] = [];
    if (clsSupported && browserName === 'chromium') {
      shifts = await page.evaluate((t0) => (window as unknown as { __shifts: { value: number; t: number; sources: string[] }[] }).__shifts.filter((s) => s.t >= t0), t0);
      cls = shifts.reduce((a, s) => a + s.value, 0);
    }
    console.log(`${tag}: ${moved.length} box moves over 32 ticks; CLS ${cls == null? 'n/a': cls.toFixed(5)}`);
    expect(moved.slice(0, 20), `${moved.length} boxes moved`).toEqual([]);
    // what is left is text moving inside its own fixed box (a number that gains a digit pushes its
    // unit; a timeline segment grows): far under the 0.1 a page counts as stable, and ~0 in practice
    if (cls != null) expect(cls, JSON.stringify(shifts.slice(0, 5))).toBeLessThan(0.01);
    // and the page still fits its width
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1)).toBe(true);
  });
  }
});
