// VIS-199 (QA VIS-200): parallel chats as a stepper on the memory step, against the mock (Super VL:
// the card measured 1 to 3, three by default, the pool off, so one more chat is 3.1 GiB at a
// 262,144-token window, its KV and the slot's measured fixed part; 8 fit the mock box, 9 do not). Minus, plus, typing and the arrow keys
// move the memory bar at once; above 3 is a quiet note; a count that does not fit, or one the
// engine does not take, says why next to the field and beside Next, which refuses it.
import { expect, test, type Page } from '@playwright/test';
import { IS_MOCK } from './helpers';
import { open, setSlots, setupMock } from './setup-helpers';

test.skip(!IS_MOCK, 'drives the mock (POST /__mock/setup)');

const VL = 'nemotron-3.5-super-vl-120b-a12b';
const field = (page: Page) => page.locator('[data-field="parallelism.slots"]');
const input = (page: Page) => page.locator('#pf-slots-in');
const total = async (page: Page) => Number(await page.locator('.membar').getAttribute('data-total'));

test.beforeEach(async ({ page, request }) => {
  await setupMock(request);
  await page.addInitScript(() => localStorage.removeItem('tandem.setup.v1'));
  await open(page, `#/setup/${VL}/parallelism`);
  await expect(input(page)).toHaveValue('3');
});

test('minus, a number field and plus, labelled; the bar and the cost line follow every step', async ({ page }) => {
  const f = field(page);
  await expect(f.getByRole('button', { name: 'One chat fewer' })).toBeVisible();
  await expect(f.getByRole('button', { name: 'One chat more' })).toBeVisible();
  await expect(input(page)).toHaveAttribute('role', 'spinbutton');
  await expect(input(page)).toHaveAttribute('aria-valuenow', '3');
  await expect(input(page)).toHaveAttribute('aria-valuemax', '16');
  await expect(page.locator('[data-slots-cost]')).toHaveText('Each extra chat takes about 3.1 GiB at a 262,144-token window.');
  await expect(page.locator('[data-slots-untested]')).toHaveCount(0);
  const t3 = await total(page);
  await f.getByRole('button', { name: 'One chat more' }).click();
  await expect(input(page)).toHaveValue('4');
  await expect.poll(() => total(page)).toBeCloseTo(t3 + 3.1, 1);
  await expect(page.locator('[data-slots-untested]')).toHaveText('Measured up to 3 on this model. More is untested.');
  await expect(page.locator('[data-field="parallelism.side_window"] .pfield-what')).toContainText('chats 2 to 4');
  await f.getByRole('button', { name: 'One chat fewer' }).click();
  await f.getByRole('button', { name: 'One chat fewer' }).click();
  await expect(input(page)).toHaveValue('2');
  await expect.poll(() => total(page)).toBeCloseTo(t3 - 3.1, 1);
  await expect(page.locator('[data-field="parallelism.side_window"] .pfield-what')).toContainText('chat 2.');
  await f.getByRole('button', { name: 'One chat fewer' }).click();
  await expect(input(page)).toHaveValue('1');
  await expect(f.getByRole('button', { name: 'One chat fewer' })).toBeDisabled();
  // one chat: no extra chats' window to set
  await expect(page.locator('[data-field="parallelism.side_window"]')).toHaveCount(0);
});

test('typing a count and the arrow keys change it at once; the readout keeps two rows', async ({ page }) => {
  const t3 = await total(page);
  await setSlots(page, 7);
  await expect.poll(() => total(page)).toBeCloseTo(t3 + 4 * 3.1, 1);
  await expect(page.locator('.membar')).toHaveAttribute('data-fits', 'true');
  await expect(page.locator('.spar-slot').nth(1).locator('.spar-slot-n')).toHaveText('2–7');
  await expect(page.locator('.spar-slot')).toHaveCount(2);
  await input(page).press('ArrowUp');
  await expect(input(page)).toHaveValue('8');
  await input(page).press('ArrowDown');
  await input(page).press('ArrowDown');
  await expect(input(page)).toHaveValue('6');
  const saved = await page.evaluate(() => JSON.parse(localStorage.getItem('tandem.setup.v1') ?? '{}'));
  expect(saved.selection.slots).toBe(6);
});

test('a count that does not fit: said next to the field and in the bar, Next refuses it until it fits', async ({ page }) => {
  await setSlots(page, 9);
  await expect(page.locator('.membar')).toHaveAttribute('data-fits', 'false');
  await expect(page.locator('.membar')).toHaveClass(/is-over/);
  const err = page.locator('[data-slots-error]');
  await expect(err).toHaveText(/^9 chats do not fit: \d+\.\d GiB over\. At most 8 fit with these settings\.$/);
  await expect(input(page)).toHaveAttribute('aria-invalid', 'true');
  await page.locator('.setup-next').click();
  await expect(page.locator('.setup-step')).toHaveAttribute('data-step', 'parallelism');
  await expect(page.locator('[data-next-refused]')).toContainText('Next stays on this step. 9 chats do not fit');
  await expect(input(page)).toBeFocused();
  // never clamped: the count stays what was asked for
  await expect(input(page)).toHaveValue('9');
  await field(page).getByRole('button', { name: 'One chat fewer' }).click();
  await expect(input(page)).toHaveValue('8');
  await expect(err).toHaveCount(0);
  await expect(page.locator('[data-next-refused]')).toHaveCount(0);
  await page.locator('.setup-next').click();
  await expect(page.locator('.setup-step')).toHaveAttribute('data-step', 'methods');
});

test('a typed value the engine does not take stays as typed with the reason; Next refuses it', async ({ page }) => {
  const before = await total(page);
  await setSlots(page, 17);
  await expect(input(page)).toHaveValue('17');
  await expect(page.locator('[data-slots-error]')).toHaveText('The engine runs at most 16 chats at once. Its shared check pass holds 32 tokens, and each chat needs 2.');
  // the selection keeps its last good count, so the bar does not move
  expect(await total(page)).toBe(before);
  await page.locator('.setup-next').click();
  await expect(page.locator('.setup-step')).toHaveAttribute('data-step', 'parallelism');
  await expect(page.locator('[data-next-refused]')).toContainText('at most 16 chats');
  for (const [raw, why] of [
    ['', 'Type how many chats, 1 or more.'],
    ['0', 'At least 1 chat.'],
    ['2.5', 'Type a whole number, like 2 or 3.'],
  ]) {
    await setSlots(page, raw);
    await expect(page.locator('[data-slots-error]')).toHaveText(why);
  }
  await setSlots(page, 6);
  await expect(page.locator('[data-slots-error]')).toHaveCount(0);
  await expect(input(page)).toHaveValue('6');
});

test('a count above the card goes through review and apply: QSE_PARALLEL 5', async ({ page }) => {
  await setSlots(page, 5);
  await open(page, `#/setup/${VL}/review`);
  const row = page.locator('.rv-table tbody tr', { hasText: 'QSE_PARALLEL' });
  await expect(row).toContainText('5');
  await expect(page.locator('.setup-finish')).toBeEnabled();
});
