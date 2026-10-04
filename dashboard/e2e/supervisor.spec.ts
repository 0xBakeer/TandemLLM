// VIS-44 / VIS-45 in the browser: the supervisor (the session monitor) where the owner already looks.
// One test per Gherkin scenario of VIS-44, on the mock (`POST /__mock/live {scenario:
// "supervisor"}` plays a running turn whose reasoning loop D1 closes 9 s in, a finished turn with
// two actions and one with a shadow decision; `POST /__mock/conductor {fixture: "night140"}` adds
// a 140-turn session). Every test collects the console: an error fails it.
import { expect, test, type APIRequestContext, type Page } from '@playwright/test';
import { IS_MOCK, login, noHorizontalOverflow, setMode } from './helpers';

const RUN = 'chatcmpl-9f3d1e0c8b7a6f5e4d3c';
const TWO = 'chatcmpl-2b7e44c1d0a9f8e7d6c5';
const SHADOW = 'chatcmpl-5ad0c2e1b3f4a5968778';

async function mockLive(request: APIRequestContext, data: Record<string, unknown>): Promise<{ now: number }> {
  const res = await request.post('/__mock/live', { data });
  expect(res.ok()).toBeTruthy();
  return res.json();
}

/** Console errors and page errors while the test runs. */
function consoleErrors(page: Page): string[] {
  const out: string[] = [];
  page.on('console', (m) => {
    if (m.type() === 'error') out.push(m.text());
  });
  page.on('pageerror', (e) => out.push(String(e)));
  return out;
}

/** The row or the card of a request, whichever this viewport shows. */
const rowOf = (page: Page, rid: string) => page.locator(`.live [data-id="${rid}"]:visible`);

test.describe('the supervisor on the Live view, the request rows and the sessions @mock', () => {
  test.skip(!IS_MOCK, 'needs the mock scenarios');
  test.beforeEach(async ({ request }) => {
    await setMode(request, 'ok');
    await request.post('/__mock/conductor', { data: { reset: true, requireToken: false } });
    await mockLive(request, { scenario: 'clear', contract: '1.1', activity: true, simulator: false, monitor: 'on' });
  });
  test.afterAll(async ({ request }) => {
    await mockLive(request, { scenario: 'clear', simulator: true, monitor: null });
    await request.post('/__mock/conductor', { data: { reset: true } });
  });

  test('1: the banner shows a live action within 1 s, and the state chip reads supervising during the hidden forward', async ({ page, request }) => {
    await login(page, '#/performance');
    const errors = consoleErrors(page); // after the sign-in: its session check answers 401 by design
    await expect(page.locator('[data-monitor-mode="on"]')).toHaveText('monitor on');
    const line = page.locator('[data-sup-line]');
    await expect(line).toBeVisible({ timeout: 10_000 });
    // the moment the line first names the action, stamped in the page
    await page.evaluate(() => {
      const w = window as unknown as { __supSeen?: number };
      const obs = new MutationObserver(() => {
        const el = document.querySelector('[data-sup-line]');
        if (!w.__supSeen && el?.textContent?.includes('closed the reasoning')) w.__supSeen = Date.now();
      });
      obs.observe(document.body, { subtree: true, childList: true, characterData: true });
    });
    // the action fires 9 s after the request arrived: start the scenario 7.5 s in
    const { now } = await mockLive(request, { scenario: 'supervisor', at: 7500 });
    const actedAt = now + 1500;
    await expect(line).toContainText('Supervisor: closed the reasoning at 2,880 tokens (D1, score 0.93)', { timeout: 6000 });
    const seen = await page.evaluate(() => (window as unknown as { __supSeen?: number }).__supSeen ?? 0);
    expect(seen - actedAt, 'the banner within a second of the action').toBeLessThan(1000);
    expect(seen).toBeGreaterThanOrEqual(actedAt - 50);
    // the hidden forward (0.8 s, held a second after): the chip and the headline say supervising
    await expect(rowOf(page, RUN).locator('.phase.act')).toContainText('supervising', { timeout: 1500 });
    await expect(page.locator('.now-headline')).toContainText('Supervising: D1 closing the reasoning');
    // and after it the chip goes back to the state, the line stays
    await expect(rowOf(page, RUN).locator('.phase.act')).not.toContainText('supervising', { timeout: 5000 });
    await expect(line).toContainText('closed the reasoning at 2,880 tokens');
    expect(errors).toEqual([]);
  });

  test('2: a finished request with two actions has the badge 2, and its drawer lists both with task, tier, mode, action and outcome', async ({ page, request }) => {
    await mockLive(request, { scenario: 'supervisor', at: 20_000 });
    await login(page, '#/performance');
    const errors = consoleErrors(page); // after the sign-in: its session check answers 401 by design
    const b = rowOf(page, TWO).locator('[data-sup-badge="act"]');
    await expect(b).toHaveText(/2$/, { timeout: 10_000 });
    await b.click();
    const drawer = page.locator(`[data-drawer="${TWO}"]:visible`);
    const evs = drawer.locator('[data-sup-event]');
    await expect(evs).toHaveCount(2);
    await expect(evs.nth(0)).toContainText('D1');
    await expect(evs.nth(0)).toContainText('tier 0');
    await expect(evs.nth(0)).toContainText('acted');
    await expect(evs.nth(0)).toContainText('closed the reasoning at 2,304 tokens');
    await expect(evs.nth(0)).toContainText('score 0.91');
    await expect(evs.nth(0)).toContainText('22 tok');
    await expect(evs.nth(0)).toContainText('tool call · tool_calls');
    await expect(evs.nth(1)).toContainText('E1');
    await expect(evs.nth(1)).toContainText('continued an empty turn');
    // the finished request in the Dev tab's ledger rows: the same badge and the same list
    await page.goto('./#/dev');
    const dev = page.locator('.table-req tr.req-row').filter({ has: page.locator('[data-sup-badge="act"]') }).first();
    await expect(dev.locator('[data-sup-badge="act"]')).toHaveText(/2$/, { timeout: 15_000 });
    await dev.locator('[data-sup-badge="act"]').click();
    await expect(page.locator(`.req-sup[data-drawer="${TWO}"] [data-sup-event]`)).toHaveCount(2);
    expect(errors).toEqual([]);
  });

  test('3: a shadow decision with no action has an outlined badge and its drawer marks it shadow', async ({ page, request }) => {
    await mockLive(request, { scenario: 'supervisor', at: 20_000 });
    await login(page, '#/performance');
    const errors = consoleErrors(page); // after the sign-in: its session check answers 401 by design
    const b = rowOf(page, SHADOW).locator('[data-sup-badge]');
    await expect(b).toHaveAttribute('data-sup-badge', 'shadow', { timeout: 10_000 });
    await expect(b).toHaveText(/1$/);
    const style = await b.evaluate((el) => {
      const cs = getComputedStyle(el);
      return { bg: cs.backgroundColor, border: cs.borderTopStyle };
    });
    expect(style.border).toBe('dashed');
    expect(style.bg).toBe('rgba(0, 0, 0, 0)');
    const filled = await rowOf(page, TWO).locator('[data-sup-badge="act"]').evaluate((el) => getComputedStyle(el).backgroundColor);
    expect(filled).not.toBe('rgba(0, 0, 0, 0)');
    await b.click();
    const ev = page.locator(`[data-drawer="${SHADOW}"]:visible [data-sup-event]`);
    await expect(ev).toHaveCount(1);
    await expect(ev).toHaveAttribute('data-sup-event', 'shadow');
    await expect(ev).toContainText('would have had the tool calls written again');
    expect(errors).toEqual([]);
  });

  test('4 and 5: the 140-turn night on one time axis, and the Markdown export lists every hidden injection', async ({ page, request }) => {
    await request.post('/__mock/conductor', { data: { fixture: 'night140' } });
    await login(page, '#/conductor?screen=sessions&session=c%3Ases_night_140');
    const errors = consoleErrors(page); // after the sign-in: its session check answers 401 by design
    const axis = page.locator('[data-axis]');
    await expect(axis.locator('[data-axis-turn]')).toHaveCount(140, { timeout: 10_000 });
    await expect(axis.locator('[data-axis-act="act"]')).toHaveCount(3);
    await expect(axis.locator('[data-axis-compaction]')).toHaveCount(1);
    await expect(axis.locator('[data-axis-checker]')).toHaveCount(9);
    // every mark inside the axis
    const box = (await axis.locator('.axis-turns').boundingBox())!;
    for (const sel of ['[data-axis-act]', '[data-axis-compaction]', '[data-axis-checker]']) {
      for (const m of await axis.locator(sel).all()) {
        const r = (await m.boundingBox())!;
        expect(r.x + r.width / 2).toBeGreaterThanOrEqual(box.x - 1);
        expect(r.x + r.width / 2).toBeLessThanOrEqual(box.x + box.width + 1);
      }
    }
    await expect(axis.locator('[data-axis-checker]').first()).toHaveAttribute('title', /pagodacheck FAIL: \(a\) [\d.]+ > 1/);
    // 5: the export button downloads the Markdown with every hidden text
    const [dl] = await Promise.all([page.waitForEvent('download'), page.locator('[data-export="md"]').click()]);
    const md = await (await import('node:fs/promises')).readFile((await dl.path())!, 'utf8');
    expect(md).toContain('3 interventions in 140 turns');
    for (const t of ['D1: the hidden text of act 1', 'S2: the hidden text of act 2', 'E1: the hidden text of act 3']) expect(md).toContain(t);
    expect(errors).toEqual([]);
  });

  test('6: an older server (no monitor mode, no conductor fields) renders nothing supervisor-related and logs no error', async ({ page, request }) => {
    await mockLive(request, { scenario: 'agent-turn', at: 20_000, monitor: null });
    await login(page, '#/performance');
    const errors = consoleErrors(page); // after the sign-in: its session check answers 401 by design
    await expect(page.locator('.live-row:visible, .live-card:visible').first()).toBeVisible({ timeout: 10_000 });
    await page.waitForTimeout(1500);
    for (const sel of ['[data-sup-line]', '[data-sup-badge]', '[data-monitor-mode]', '[data-drawer]']) await expect(page.locator(sel)).toHaveCount(0);
    await page.goto('./#/dev');
    await expect(page.locator('.table-req tbody tr').first()).toBeVisible({ timeout: 10_000 });
    await expect(page.locator('[data-sup-badge]')).toHaveCount(0);
    expect(errors).toEqual([]);
  });

  test('7: at 390 px the banner, the badges, the drawer and the timeline fit without a sideways scroll', async ({ page, request }, info) => {
    test.skip(!info.project.name.includes('phone'), 'phone project only');
    await mockLive(request, { scenario: 'supervisor', at: 20_000 });
    await login(page, '#/performance');
    const errors = consoleErrors(page); // after the sign-in: its session check answers 401 by design
    await expect(page.locator('[data-sup-line]')).toContainText('Supervisor:', { timeout: 10_000 });
    const sup = (await page.locator('[data-sup-line]').boundingBox())!;
    expect(sup.x + sup.width).toBeLessThanOrEqual(391);
    await rowOf(page, TWO).locator('[data-sup-badge]').click();
    await expect(page.locator(`[data-drawer="${TWO}"]:visible [data-sup-event]`)).toHaveCount(2);
    expect(await noHorizontalOverflow(page), 'the Live view scrolls sideways').toBe(true);
    await page.screenshot({ path: 'e2e-artifacts/supervisor-live-phone.png', fullPage: false });
    await request.post('/__mock/conductor', { data: { fixture: 'night140' } });
    await page.goto('./#/conductor?screen=sessions&session=c%3Ases_night_140');
    await expect(page.locator('[data-axis] [data-axis-turn]')).toHaveCount(140, { timeout: 10_000 });
    expect(await noHorizontalOverflow(page), 'the session timeline scrolls sideways').toBe(true);
    await page.locator('[data-axis]').screenshot({ path: 'e2e-artifacts/supervisor-axis-phone.png' });
    expect(errors).toEqual([]);
  });

  test('desktop screenshots for the ticket', async ({ page, request }, info) => {
    test.skip(!info.project.name.startsWith('chromium-desktop'), 'one desktop project');
    await mockLive(request, { scenario: 'supervisor', at: 8600 });
    await login(page, '#/performance');
    await expect(page.locator('[data-sup-line]')).toContainText('Supervisor:', { timeout: 10_000 });
    await rowOf(page, TWO).locator('[data-sup-badge]').click();
    await expect(page.locator(`[data-drawer="${TWO}"]:visible [data-sup-event]`)).toHaveCount(2);
    await page.locator('.live').screenshot({ path: 'e2e-artifacts/supervisor-live-desktop.png' });
    await request.post('/__mock/conductor', { data: { fixture: 'night140' } });
    await page.goto('./#/conductor?screen=sessions&session=c%3Ases_night_140');
    await expect(page.locator('[data-axis] [data-axis-turn]')).toHaveCount(140, { timeout: 10_000 });
    await page.locator('[data-timeline]').screenshot({ path: 'e2e-artifacts/supervisor-session-desktop.png' });
  });
});
