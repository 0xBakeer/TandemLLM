// OPS-41 / OPS-42 in the browser: the System view's memory plan (the engine's split, what may load
// beside it, the last refusal) and the limits edited in the Conductor view, on the mock
// (`POST /__mock/conductor {residencyMem}` sets the plan's MemAvailable). OPS-47 / OPS-48: the
// engine's memory limits in the same panel -- the ceiling and peak lines, every limit with what it
// holds now, edits that apply at the next request or say they need a restart, MAX_LEN locked, and
// the planner counting the engine's growth to its peak (the mock: 37.9 GiB free, 70 of 77 GiB).
import { expect, test, type Page } from '@playwright/test';
import { IS_MOCK, login, noHorizontalOverflow, setMode } from './helpers';

function consoleErrors(page: Page): string[] {
  const out: string[] = [];
  page.on('console', (m) => m.type() === 'error' && out.push(m.text()));
  page.on('pageerror', (e) => out.push(String(e)));
  return out;
}

test.describe('the memory plan @mock', () => {
  test.skip(!IS_MOCK, 'needs the mock conductor');
  test.beforeEach(async ({ request }) => {
    await setMode(request, 'ok');
    await request.post('/__mock/conductor', { data: { reset: true, requireToken: false, residencyMem: 37.9 } });
  });
  test.afterAll(async ({ request }) => {
    await request.post('/__mock/conductor', { data: { reset: true, residencyMem: 37.9 } });
  });

  test('the System view shows the split, the floor, every role with its claim and the last refusal', async ({ page }) => {
    await login(page, '#/system');
    const errors = consoleErrors(page);
    const card = page.locator('#card-residency');
    await expect(card.locator('[data-split] [data-part]')).toHaveCount(9, { timeout: 10_000 });
    await expect(card.locator('[data-part-row="kv_pool"]')).toContainText('16.00 GiB');
    await expect(card.locator('[data-residency]')).toContainText('37.9 GiB');
    await expect(card.locator('[data-engine-growth]')).toContainText('7.0 GiB');
    await expect(card.locator('[data-residency]')).toContainText('15.0 GiB + 6.0 GiB');
    await expect(card.locator('[data-residency-role="decider"]')).toContainText('8.2 GiB');
    // OPS-47: 37.9 - 7 (the engine's growth to its peak) - 8.2
    await expect(card.locator('[data-residency-role="decider"]')).toContainText('fits, 22.7 GiB left');
    await expect(card.locator('[data-residency-role="reader"]')).toContainText('refused: it would break the floor');
    await expect(card.locator('[data-last-refusal]')).toContainText('reader');
    await expect(card.locator('[data-load-stages]')).toHaveCount(1);
    expect(errors).toEqual([]);
  });

  test('at 18 GiB nothing fits: the decider is refused at the floor', async ({ page, request }) => {
    await request.post('/__mock/conductor', { data: { residencyMem: 18 } });
    await login(page, '#/system');
    await expect(page.locator('[data-residency-role="decider"]')).toContainText('refused: it would break the floor (2.8 GiB left, 21.0 GiB needed)', { timeout: 10_000 });
  });

  test('the limits are edited in the Conductor view and the plan follows', async ({ page }) => {
    await login(page, '#/conductor?screen=profiles');
    const errors = consoleErrors(page);
    const sec = page.locator('[data-sec="residency"]');
    await sec.locator('summary').click();
    await sec.locator('input[data-res="floor_gib"]').fill('20');
    await expect(sec.locator('[data-claim="decider"] input[data-claim-gib="decider"]')).toHaveValue('8.2');
    await sec.getByLabel('change note').fill('floor 20 for a test');
    await expect(sec.locator('[data-save]')).toBeEnabled({ timeout: 5000 });
    await sec.locator('[data-save]').click();
    await expect(sec.locator('[data-savebar] .status.num')).toContainText('live v1', { timeout: 8000 });
    await page.goto('./#/system');
    // 37.9 - 7 - 8.2 = 22.7 is under 20 + 6
    await expect(page.locator('[data-residency-role="decider"]')).toContainText('refused: it would break the floor (22.7 GiB left, 26.0 GiB needed)', { timeout: 10_000 });
    expect(errors).toEqual([]);
  });

  test('OPS-47: every limit with what it holds now, the two lines, the switch and MAX_LEN locked', async ({ page }) => {
    await login(page, '#/system');
    const errors = consoleErrors(page);
    const card = page.locator('#card-residency');
    await expect(card.locator('[data-limit]')).toHaveCount(15, { timeout: 10_000 });
    await expect(card.locator('[data-mark="ceiling"]')).toContainText('ceiling 70 GiB');
    await expect(card.locator('[data-mark="peak"]')).toContainText('peak 77 GiB');
    await expect(card.locator('[data-limit="RESIDENT_PIN_GB"] [data-limit-use]')).toHaveText('3.85 of 4.00 GiB');
    await expect(card.locator('[data-limit="RESIDENT_PARK_GB"] [data-applies]')).toHaveText('live');
    await expect(card.locator('[data-limit="PREFIX_CHUNK"] [data-applies]')).toHaveText('restart');
    await expect(card.locator('[data-limit="MAX_LEN"]')).toContainText('262,144 tokens');
    await expect(card.locator('[data-limit="MAX_LEN"] [data-applies]')).toHaveText('locked');
    await expect(card.locator('[data-switch]')).toContainText('81,199 tokens re-read in 125.2 s');
    await expect(card.locator('[data-parked]')).toContainText('22,528 rows past its head');
    await expect(card.locator('[data-budget]')).toContainText('262,144-token prefill 7.34 GiB');
    expect(errors).toEqual([]);
  });

  test('OPS-47: a live limit applies at the next request; a restart key says so; MAX_LEN has no input', async ({ page }) => {
    await login(page, '#/system');
    const errors = consoleErrors(page);
    const card = page.locator('#card-residency');
    await card.locator('[data-edit-limits]').click();
    await expect(card.locator('[data-limit-input="RESIDENT_PARK_GB"]')).toHaveValue('4', { timeout: 10_000 });
    await expect(card.locator('[data-limit="MAX_LEN"] [data-limit-input]')).toHaveCount(0);
    await card.locator('[data-limit-input="RESIDENT_PARK_GB"]').fill('2');
    await card.locator('[data-limit-input="RESIDENT_PARK_GB"]').press('Tab');
    await card.locator('[data-limit-input="PREFIX_CHUNK"]').fill('512');
    await card.locator('[data-limit-input="PREFIX_CHUNK"]').press('Tab');
    await card.getByLabel('change note').fill('park 2 GiB, a 512-row chunk, for a test');
    await expect(card.locator('[data-save]')).toBeEnabled({ timeout: 5000 });
    await card.locator('[data-save]').click();
    await expect(card.locator('[data-savebar] .status.num')).toContainText('live v1', { timeout: 8000 });
    await expect(card.locator('[data-limit="RESIDENT_PARK_GB"] [data-source]')).toHaveText('dashboard', { timeout: 10_000 });
    // the parked 2.25 GiB were over the new cap: given back
    await expect(card.locator('[data-limit="RESIDENT_PARK_GB"] [data-limit-use]')).toHaveText('0.00 of 2.00 GiB');
    await expect(card.locator('[data-limit="PREFIX_CHUNK"] [data-pending]')).toHaveText('set to 512 tokens; the engine runs 1,024 tokens until it restarts');
    // a peak under the ceiling: the panel says so at once, and the server's rule refuses the pair
    await card.locator('[data-limit-input="MEM_PEAK_GIB"]').fill('60');
    await card.locator('[data-limit-input="MEM_PEAK_GIB"]').press('Tab');
    await expect(card.locator('[data-limits-warn]')).toContainText('peak (60 GiB) is under the ceiling (70 GiB)');
    await card.locator('[data-limit-input="MEM_CEILING_GIB"]').fill('72');
    await card.locator('[data-limit-input="MEM_CEILING_GIB"]').press('Tab');
    await expect(card.locator('[data-errors]')).toContainText('MEM_PEAK_GIB is under MEM_CEILING_GIB', { timeout: 5000 });
    await expect(card.locator('[data-save]')).toBeDisabled();
    expect(errors).toEqual([]);
  });

  test('phone: the plan fits 390 px without a sideways scroll', async ({ page }, info) => {
    test.skip(!info.project.name.includes('phone'), 'phone project only');
    await login(page, '#/system');
    await expect(page.locator('#card-residency [data-residency-role="decider"]')).toBeVisible({ timeout: 10_000 });
    await page.locator('#card-residency').scrollIntoViewIfNeeded();
    expect(await noHorizontalOverflow(page)).toBe(true);
    await page.locator('#card-residency').screenshot({ path: 'e2e-artifacts/residency-phone.png' });
  });

  test('desktop screenshot for the ticket', async ({ page }, info) => {
    test.skip(!info.project.name.startsWith('chromium-desktop'), 'one desktop project');
    await login(page, '#/system');
    await expect(page.locator('#card-residency [data-residency-role="decider"]')).toBeVisible({ timeout: 10_000 });
    await page.locator('#card-residency').screenshot({ path: 'e2e-artifacts/residency-desktop.png' });
  });
});
