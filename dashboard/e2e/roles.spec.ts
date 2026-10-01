// SRV-134's Playwright rows for the role contract (SRV-133) on the Profiles screen: every role shows
// its contract as form fields with the enum values, a kind's forbidden window and writes are off
// with the reason, the forms of sharing follow the tokenizer hash, an edit saves as a new version,
// and the section reads at 390 px without sideways scrolling.
import { expect, test, type Page } from '@playwright/test';
import { IS_MOCK, login, noHorizontalOverflow, setMode } from './helpers';

test.skip(!IS_MOCK, 'the role contract spec runs on the mock tier (@mock)');

async function roles(page: Page): Promise<void> {
  await setMode(page.request, 'ok');
  await page.request.post('/__mock/conductor', { data: { reset: true, requireToken: false, guard: false, roleDown: null } });
  await login(page, '#/conductor?screen=profiles');
  await page.locator('summary', { hasText: 'Roles' }).click();
  await expect(page.locator('[data-role-row="decider"]')).toBeVisible();
}

test('every role shows its contract with the enum values @mock', async ({ page }) => {
  await roles(page);
  const dec = page.locator('[data-role-contract="decider"]');
  await expect(dec.locator('[data-role-field="kind"]')).toHaveValue('decider');
  await expect(dec.locator('[data-role-field="window"]')).toHaveValue('idle');
  await expect(dec.locator('[data-role-field="fallback"]')).toHaveValue('PASS');
  await expect(dec.locator('[data-role-field="writes"]')).toHaveValue('nothing');
  await expect(dec.locator('[data-role-field="tokenizer_sha256"]')).toHaveValue('none');
  await expect(dec.locator('[data-role-field="kind"] option')).toHaveText(['reasoner', 'decider', 'probe', 'clerk', 'librarian', 'tool']);
  await expect(page.locator('[data-role-row="decider"] [data-role-field="memory_claim_gib"]')).toHaveValue('0');
  await expect(page.locator('[data-role-row="classifier"] [data-role-field="preemptible"]')).not.toBeChecked();
  await expect(page.locator('[data-engine-sha]')).toContainText('5f0c5f0c5f0c');
});

test('a kind’s forbidden window, writes and sharing are off with the reason @mock', async ({ page }) => {
  await roles(page);
  const reader = page.locator('[data-role-row="reader"]');
  await expect(reader.locator('[data-role-field="kind"]')).toHaveValue('clerk');
  await expect(reader.locator('[data-role-field="window"] option[value="in_decode"]')).toHaveAttribute('disabled', '');
  await expect(reader.locator('[data-role-field="writes"] option[value="client_visible"]')).toHaveAttribute('disabled', '');
  await expect(reader.locator('[data-share="tokens"]')).toBeDisabled();
  await expect(reader.locator('[data-share-why="tokens"]')).toContainText('tokenizer hash');
  await expect(reader.locator('[data-share="states"]')).toBeDisabled();
  // the engine's own tokenizer hash opens token sharing
  await reader.locator('[data-role-field="tokenizer_sha256"]').fill('5f0c'.repeat(16));
  await expect(reader.locator('[data-share="tokens"]')).toBeEnabled();
  await reader.locator('[data-share="tokens"]').click();
  await expect(reader.locator('[data-share="tokens"]')).toHaveAttribute('aria-pressed', 'true');
});

test('an edit of the contract saves as a new version @mock', async ({ page }) => {
  await roles(page);
  const sec = page.locator('[data-sec="roles"]');
  const dec = page.locator('[data-role-row="decider"]');
  await dec.locator('[data-role-field="latency_budget_ms"]').fill('150');
  await dec.locator('[data-role-field="fallback"]').selectOption('skip');
  await sec.getByLabel('change note').fill('decider budget 150 ms');
  await sec.locator('[data-save]').click();
  await expect(sec.locator('[data-saved]')).toContainText('version');
  const got = await (await page.request.get('/v1/conductor/roles')).json();
  expect(got.doc.decider.latency_budget_ms).toBe(150);
  expect(got.doc.decider.fallback).toBe('skip');
  expect(got.version).toBe(1);
});

test('the roles read at 390 px without sideways scrolling @mock', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await roles(page);
  await page.locator('[data-role-row="reader"]').scrollIntoViewIfNeeded();
  expect(await noHorizontalOverflow(page)).toBe(true);
  const box = await page.locator('[data-role-row="decider"] [data-role-field="tokenizer_sha256"]').boundingBox();
  expect(box!.x + box!.width).toBeLessThanOrEqual(390);
});
