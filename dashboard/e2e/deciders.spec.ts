// VIS-46 / VIS-47 in the browser: the Conductor view's Deciders screen on the mock
// (`POST /__mock/conductor {deciderMem, decidersOff}`). The test matrix is VIS-46's Gherkin.
import { expect, test, type Page } from '@playwright/test';
import { IS_MOCK, login, noHorizontalOverflow, setMode } from './helpers';

function consoleErrors(page: Page): string[] {
  const out: string[] = [];
  page.on('console', (m) => m.type() === 'error' && out.push(m.text()));
  page.on('pageerror', (e) => out.push(String(e)));
  return out;
}

const row = (page: Page, name: string) => page.locator(`[data-decider-row="${name}"]`);

test.describe('the Deciders screen @mock', () => {
  test.skip(!IS_MOCK, 'needs the mock conductor');
  test.beforeEach(async ({ request }) => {
    await setMode(request, 'ok');
    await request.post('/__mock/conductor', { data: { reset: true, requireToken: false } });
  });
  test.afterAll(async ({ request }) => {
    await request.post('/__mock/conductor', { data: { reset: true } });
  });

  test('1 what runs is visible: states, limits, precision and the memory row', async ({ page }) => {
    await login(page, '#/conductor?screen=deciders');
    const errors = consoleErrors(page);
    await expect(row(page, 'laya-english').locator('[data-decider-state]')).toHaveText('loaded', { timeout: 10_000 });
    await expect(row(page, 'laya-multilingual').locator('[data-decider-state]')).toHaveText('not loaded');
    await expect(row(page, 'laya-english')).toContainText('512 / 192 / 20');
    await expect(row(page, 'laya-english')).toContainText('fp32 parameters under autocast(bfloat16)');
    await expect(row(page, 'laya-multilingual')).toContainText('1024 / 256 / 20');
    const mem = page.locator('[data-decider-memory]');
    await expect(mem).toContainText('37.9 GiB');
    await expect(mem).toContainText('7.0 GiB'); // OPS-47: the engine's growth to its declared peak
    await expect(mem).toContainText('8.2 GiB');
    await expect(mem).toContainText('15.0 GiB + 6.0 GiB');
    await expect(mem).toContainText('fits, 22.7 GiB left after the claim');
    expect(errors).toEqual([]);
  });

  test('2 Load goes through loading to loaded', async ({ page }) => {
    await login(page, '#/conductor?screen=deciders');
    await row(page, 'laya-multilingual').locator('[data-load]').click();
    await expect(row(page, 'laya-multilingual').locator('[data-decider-state]')).toHaveText(/loading the weights|warming up/, { timeout: 5_000 });
    await expect(row(page, 'laya-multilingual').locator('[data-decider-state]')).toHaveText('loaded', { timeout: 10_000 });
    await expect(row(page, 'laya-multilingual').locator('[data-unload]')).toBeVisible();
  });

  test('3 a refused load says why with the numbers and nothing loads', async ({ page, request }) => {
    await request.post('/__mock/conductor', { data: { deciderMem: 28 } });
    await login(page, '#/conductor?screen=deciders');
    await expect(page.locator('[data-decider-memory]')).toContainText('refused: it would break the floor', { timeout: 10_000 });
    await row(page, 'laya-typed-decisions').locator('[data-load]').click();
    const note = page.locator('[data-decider-note="laya-typed-decisions"]');
    await expect(note.locator('[data-refusal]')).toContainText('refused: it would break the floor (MemAvailable 28.0 GiB, engine growth 7.0 GiB, claim 8.2 GiB, 12.8 GiB left, 21.0 GiB needed)', { timeout: 8_000 });
    await expect(note).toContainText('refused by the residency planner');
    await expect(row(page, 'laya-typed-decisions').locator('[data-decider-state]')).toHaveText('not loaded');
  });

  test('4 Unload asks first, then the model is not loaded', async ({ page }) => {
    await login(page, '#/conductor?screen=deciders');
    await row(page, 'laya-english').locator('[data-unload]').click();
    await page.locator('qse-cond-dialog [data-confirm]').click();
    await expect(row(page, 'laya-english').locator('[data-decider-state]')).toHaveText('not loaded', { timeout: 8_000 });
    await expect(row(page, 'laya-english').locator('[data-load]')).toBeVisible();
  });

  test('5 the model per question is saved as a new version with a note', async ({ page }) => {
    await login(page, '#/conductor?screen=deciders');
    const errors = consoleErrors(page);
    await page.locator('[data-question-model="classify"]').selectOption('laya-english', { timeout: 10_000 });
    await page.getByLabel('change note').fill('classify by the English checkpoint');
    await expect(page.locator('[data-save]')).toBeEnabled({ timeout: 5_000 });
    await page.locator('[data-save]').click();
    await expect(page.locator('[data-savebar] .status.num').first()).toContainText('live v1', { timeout: 8_000 });
    await page.locator('[data-versions-btn]').click();
    await expect(page.locator('[data-version="1"]')).toContainText('classify by the English checkpoint');
    await expect(page.locator('[data-question-model="classify"]')).toHaveValue('laya-english');
    expect(errors).toEqual([]);
  });

  test('5b L1: the Laya route for scripts is switched off and its rate changed as a new version', async ({ page }) => {
    await login(page, '#/conductor?screen=deciders');
    const errors = consoleErrors(page);
    const api = page.locator('[data-decider-api]');
    await expect(api.locator('[data-api-stats]')).toContainText('12 calls', { timeout: 10_000 });
    await api.getByRole('radio', { name: 'off' }).click();
    await api.locator('[data-api-field="rate_per_min"]').fill('60');
    await api.locator('[data-api-field="rate_per_min"]').blur();
    await page.getByLabel('change note').fill('Laya route off for the night');
    await expect(page.locator('[data-save]')).toBeEnabled({ timeout: 5_000 });
    await page.locator('[data-save]').click();
    await expect(page.locator('[data-savebar] .status.num').first()).toContainText('live v1', { timeout: 8_000 });
    await expect(api.getByRole('radio', { name: 'off' })).toHaveAttribute('aria-checked', 'true');
    await expect(api.locator('[data-api-field="rate_per_min"]')).toHaveValue('60');
    // X1: the playtest sessions' rate and the live ones
    await expect(api.locator('[data-api-field="drive_rate_per_s"]')).toHaveValue('10');
    await expect(api.locator('[data-api-stats]')).toContainText('1 playtest sessions live');
    expect(errors).toEqual([]);
  });

  test('6 the test box answers with probabilities, the routing and the latency', async ({ page }) => {
    await login(page, '#/conductor?screen=deciders');
    await page.locator('[data-test-state]').fill('please debug the parser crash', { timeout: 10_000 });
    await page.locator('[data-test-options]').fill('plan, debug, chat');
    await page.locator('[data-test-ask]').click();
    const out = page.locator('[data-test-out]');
    await expect(out.locator('[data-test-choice]')).toHaveText('debug', { timeout: 8_000 });
    await expect(out.locator('[data-prob]')).toHaveCount(3);
    await expect(out).toContainText('laya-english');
    await expect(out).toContainText('ms');
    await expect(out.locator('[data-test-routing]')).toContainText('English Latin text');
    await page.locator('[data-test-options]').fill('only');
    await page.locator('[data-test-ask]').click();
    await expect(page.locator('[data-test-error]')).toHaveText('Give at least two options.');
  });

  test('7 the runtime off: the screen says how to turn it on and offers no Load', async ({ page, request }) => {
    await request.post('/__mock/conductor', { data: { decidersOff: true } });
    await login(page, '#/conductor?screen=deciders');
    await expect(page.locator('[data-deciders-off]')).toContainText('This engine serves no decision model', { timeout: 10_000 });
    await expect(page.locator('[data-deciders-off]')).toContainText('QWEN38_DECIDER_MODELS');
    await expect(page.locator('[data-load]')).toHaveCount(0);
  });

  test('8 phone: no sideways page scroll at 390 px', async ({ page }, info) => {
    test.skip(!info.project.name.includes('phone'), 'phone project only');
    await login(page, '#/conductor?screen=deciders');
    await expect(row(page, 'laya-english')).toBeVisible({ timeout: 10_000 });
    expect(await noHorizontalOverflow(page)).toBe(true);
    await page.locator('[data-test-ask]').scrollIntoViewIfNeeded();
    expect(await noHorizontalOverflow(page)).toBe(true);
    await page.screenshot({ path: 'e2e-artifacts/deciders-phone.png', fullPage: true });
  });

  test('desktop screenshot for the ticket', async ({ page }, info) => {
    test.skip(!info.project.name.startsWith('chromium-desktop'), 'one desktop project');
    await login(page, '#/conductor?screen=deciders');
    await expect(row(page, 'laya-english')).toBeVisible({ timeout: 10_000 });
    await page.screenshot({ path: 'e2e-artifacts/deciders-desktop.png', fullPage: true });
  });
});
