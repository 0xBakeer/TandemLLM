// VIS-199 (QA VIS-200): the window steppers of the memory step against the mock (Super VL: its card
// holds 1,048,576 tokens on a 4,096 step; three chats of 262,144 by default, the pool off). − and +
// walk 8k ... 1M, the field takes any number of tokens, and a value the model does not take stays as
// typed with the model's real limit while Next refuses it. The bar follows every change.
// The guide opens in SIMPLE mode for a fresh browser; this file drives the ADVANCED
// wizard, so every context asks for the advanced flow by name before it starts.

import { expect, test, type Page } from '@playwright/test';
import { IS_MOCK } from './helpers';
import { open, setupMock, setWindow } from './setup-helpers';

test.skip(!IS_MOCK, 'drives the mock (POST /__mock/setup)');

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => localStorage.setItem('tandem.setup.simple', '0'));
});

const VL = 'nemotron-3.5-super-vl-120b-a12b';
const win = (page: Page) => page.locator('#pf-win-in');
const side = (page: Page) => page.locator('#pf-side-in');
const field = (page: Page, key: string) => page.locator(`[data-field="parallelism.${key}"]`);
const total = async (page: Page) => Number(await page.locator('.membar').getAttribute('data-total'));

test.beforeEach(async ({ page, request }) => {
  await setupMock(request);
  await page.addInitScript(() => localStorage.removeItem('tandem.setup.v1'));
  await open(page, `#/setup/${VL}/parallelism`);
  await expect(win(page)).toHaveValue('262,144');
});

test('− and + walk the ladder up to the model\'s 1,048,576; the bar and the short form follow', async ({ page }) => {
  const f = field(page, 'window');
  await expect(win(page)).toHaveAttribute('role', 'spinbutton');
  await expect(win(page)).toHaveAttribute('aria-valuemax', '1048576');
  await expect(f.locator('[data-win-short]')).toHaveText('256k');
  await expect(page.locator('.seg-btn', { hasText: '256k' })).toHaveCount(0);
  const t0 = await total(page);
  await f.getByRole('button', { name: 'Longest conversation: one step longer' }).click();
  await expect(win(page)).toHaveValue('524,288');
  await expect(f.locator('[data-win-short]')).toHaveText('512k');
  await expect.poll(() => total(page)).toBeGreaterThan(t0);
  await f.getByRole('button', { name: 'Longest conversation: one step longer' }).click();
  await expect(win(page)).toHaveValue('1,048,576');
  await expect(f.locator('[data-win-short]')).toHaveText('1M');
  await expect(f.getByRole('button', { name: 'Longest conversation: one step longer' })).toBeDisabled();
  await f.getByRole('button', { name: 'Longest conversation: one step shorter' }).click();
  await expect(win(page)).toHaveValue('524,288');
});

test('the arrow keys move one rung, PageDown two, Home to the shortest', async ({ page }) => {
  await win(page).focus();
  await win(page).press('ArrowUp');
  await expect(win(page)).toHaveValue('524,288');
  await win(page).press('ArrowDown');
  await win(page).press('ArrowDown');
  await expect(win(page)).toHaveValue('131,072');
  await win(page).press('PageDown');
  await expect(win(page)).toHaveValue('32,768');
  await win(page).press('Home');
  await expect(win(page)).toHaveValue('8,192');
  await expect(field(page, 'window').getByRole('button', { name: 'Longest conversation: one step shorter' })).toBeDisabled();
});

test('any number can be typed: 100k is 102,400, shown with separators once the field lets go', async ({ page }) => {
  const t0 = await total(page);
  await setWindow(page, 'window', '100k');
  await expect.poll(() => total(page)).not.toBe(t0);
  await win(page).blur();
  await expect(win(page)).toHaveValue('102,400');
  await expect(field(page, 'window').locator('[data-win-short]')).toHaveText('100k');
  // from a value between two rungs, + goes to the next rung up
  await field(page, 'window').getByRole('button', { name: 'Longest conversation: one step longer' }).click();
  await expect(win(page)).toHaveValue('131,072');
  await setWindow(page, 'window', '1m');
  await win(page).blur();
  await expect(win(page)).toHaveValue('1,048,576');
});

test('above the model, below it, off its step: kept as typed with the reason; Next refuses it', async ({ page }) => {
  const before = await total(page);
  await setWindow(page, 'window', '2m');
  const err = field(page, 'window').locator('[data-win-error]');
  await expect(err).toHaveText('This model holds at most 1,048,576 tokens.');
  await expect(win(page)).toHaveValue('2m');
  await expect(win(page)).toHaveAttribute('aria-invalid', 'true');
  expect(await total(page)).toBe(before);
  await page.locator('.setup-next').click();
  await expect(page.locator('.setup-step')).toHaveAttribute('data-step', 'parallelism');
  await expect(page.locator('[data-next-refused]')).toContainText('Longest conversation: This model holds at most 1,048,576 tokens.');
  await expect(win(page)).toBeFocused();
  for (const [raw, why] of [
    ['4096', 'At least 8,192 tokens.'],
    ['100000', 'Use a multiple of 4,096 tokens, like 98,304 or 102,400.'],
    ['lots', 'Type a number of tokens, like 32k or 131,072.'],
  ]) {
    await setWindow(page, 'window', raw);
    await expect(err).toHaveText(why);
  }
  await setWindow(page, 'window', '512k');
  await expect(err).toHaveCount(0);
  await page.locator('.setup-next').click();
  await expect(page.locator('.setup-step')).toHaveAttribute('data-step', 'methods');
});

test('the extra chats hold at most the first chat\'s window, and the cost line follows theirs', async ({ page }) => {
  await setWindow(page, 'window', '128k');
  await win(page).blur();
  await expect(side(page)).toHaveValue('131,072');
  const f = field(page, 'side_window');
  await expect(f.getByRole('button', { name: 'Longest conversation in the extra chats: one step longer' })).toBeDisabled();
  await expect(f.locator('[data-win-max]')).toHaveText("The extra chats hold at most the first chat's 131,072 tokens.");
  await setWindow(page, 'side_window', '262144');
  await expect(f.locator('[data-win-error]')).toHaveText("The extra chats hold at most the first chat's 131,072 tokens.");
  await setWindow(page, 'side_window', '64k');
  await expect(f.locator('[data-win-error]')).toHaveCount(0);
  await expect(page.locator('[data-slots-cost]')).toHaveText('Each extra chat takes about 0.5 GiB at a 65,536-token window.');
});
