// VIS-49's Playwright rows for the Memory screen (VIS-48) on the mock tier: the sessions list,
// a transcript with its hidden notes highlighted and a jump to their journal row, the compaction
// folded, the keep toggle, the settings saved as a new version, the token asked for, and the phone
// layout at 390 px (cards, the viewer full width, nothing sideways). C1: the Disk cache tab -- its
// tiles and entries (a preview for opencode heads and packages only), its settings saved as a
// version with the 50 GB floor held, the clear button, a server without the tier, the phone.
import { expect, test, type Page } from '@playwright/test';
import { IS_MOCK, login, noHorizontalOverflow, setMode } from './helpers';

test.skip(!IS_MOCK, 'the Memory spec runs on the mock tier (@mock)');

async function memory(page: Page, extra = ''): Promise<void> {
  await setMode(page.request, 'ok');
  await page.request.post('/__mock/conductor', { data: { reset: true, requireToken: false, guard: false, roleDown: null } });
  await login(page, `#/conductor?screen=memory${extra}`);
  await expect(page.locator('qse-cond-memory')).toBeVisible();
}

test('the sessions list shows date, client, turns, acts, checker and keep (scenario 1) @mock', async ({ page }) => {
  await memory(page);
  const rows = page.locator('[data-memory-sessions] tbody tr');
  await expect(rows).toHaveCount(5);
  const pagoda = page.locator('[data-memory-row="c-ses_pagoda"]');
  await expect(pagoda).toContainText('opencode');
  await expect(pagoda).toContainText('overnight');
  await expect(pagoda).toContainText('pagodacheck summer PASS');
  await expect(pagoda.locator('td').nth(5)).toHaveText('6');
  await expect(pagoda.locator('td').nth(6)).toHaveText('1');
  await expect(page.locator('[data-keep="c-ses_kept"]').first()).toBeChecked();
  await expect(page.locator('[data-writer]')).toContainText('1 secret redacted');
});

test('a transcript highlights its hidden notes, folds its compaction and opens the journal row (scenario 2) @mock', async ({ page }) => {
  await memory(page);
  // the table on a desk, the cards on a phone: whichever shows
  await page.locator('[data-memory-row="c-ses_pagoda"], [data-memory-card="c-ses_pagoda"]').filter({ visible: true }).click();
  const t = page.locator('[data-transcript="c:ses_pagoda"]');
  await expect(t.locator('.mem-block')).toHaveCount(6);
  await expect(t.locator('.mem-rt-hidden')).toHaveCount(2);
  await expect(t.locator('[data-compaction]')).not.toHaveAttribute('open', '');
  await expect(t.locator('[data-compaction] [data-sec-kind="summary"]')).toBeHidden();
  await t.locator('[data-compaction] > summary').click();
  await expect(t.locator('[data-compaction] [data-sec-kind="summary"]')).toBeVisible();
  await expect(t.locator('[data-block="5"]')).toContainText('[redacted:bearer]');
  await t.locator('[data-journal]').first().click();
  await expect(page.locator('.drawer')).toBeVisible();
  await page.locator('.drawer').getByRole('button', { name: 'Close' }).click();
  await t.locator('[data-journal]').nth(1).click();
  await expect(page.locator('.drawer')).toBeVisible();
});

test('keep is switched on and stays on after a reload (scenario 3) @mock', async ({ page }) => {
  await memory(page);
  const box = page.locator('[data-keep="c-ses_minicalc"]').filter({ visible: true });
  await expect(box).not.toBeChecked();
  await box.check();
  await expect(box).toBeChecked();
  const r = await (await page.request.get('/v1/conductor/memory/sessions')).json();
  expect(r.sessions.find((s: { session: string }) => s.session === 'c:ses_minicalc').keep).toBe(true);
  await page.reload();
  await expect(page.locator('[data-keep="c-ses_minicalc"]').filter({ visible: true })).toBeChecked();
});

test('a long transcript pages at 200 blocks, newest first @mock', async ({ page }) => {
  await memory(page);
  await page.request.post('/__mock/conductor', { data: { memoryLong: 450 } });
  await login(page, '#/conductor?screen=memory&memory=c%3Ases_long');
  const t = page.locator('[data-transcript="c:ses_long"]');
  await expect(t.locator('[data-pager]').first()).toContainText('page 3 of 3');
  await expect(t.locator('.mem-block')).toHaveCount(50);
  await t.locator('[data-pager]').first().getByRole('button', { name: 'Older' }).click();
  await expect(t.locator('[data-pager]').first()).toContainText('page 2 of 3');
  await expect(t.locator('.mem-block')).toHaveCount(200);
});

test('the transcript settings save as a new settings version @mock', async ({ page }) => {
  await memory(page);
  const sec = page.locator('[data-sec="transcripts"]');
  await sec.locator('summary').click();
  await sec.getByRole('radiogroup', { name: 'transcript scope' }).getByRole('radio', { name: 'all' }).click();
  await sec.locator('[data-mem-setting="clip_chars"]').fill('6000');
  await sec.getByLabel('change note').fill('transcripts: every request, clip at 6,000');
  await sec.locator('[data-save]').click();
  await expect(sec.locator('[data-saved]')).toContainText('version');
  const s = await (await page.request.get('/v1/conductor/settings')).json();
  expect(s.doc.memory.transcripts).toEqual({ scope: 'all', retention_days: 30, clip_chars: 6000, loop_chars: 200 });
});

test('the transcripts ask for the admin token when the server wants it @mock', async ({ page }) => {
  await memory(page);
  await page.request.post('/__mock/conductor', { data: { requireToken: true } });
  await page.goto('./#/conductor?screen=overview');
  await page.goto('./#/conductor?screen=memory');
  const dialog = page.getByRole('dialog', { name: 'Unlock editing' });
  await expect(dialog).toBeVisible();
  await dialog.locator('#cond-token').fill('mock');
  await dialog.getByRole('button', { name: 'Unlock' }).click();
  await expect(page.locator('[data-memory-sessions] tbody tr')).toHaveCount(5);
});

test('phone width: the list is cards, the viewer full width, nothing sideways (scenario 4) @mock', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await memory(page);
  await expect(page.locator('[data-memory-sessions]')).toBeHidden();
  await expect(page.locator('[data-memory-card]')).toHaveCount(5);
  expect(await noHorizontalOverflow(page)).toBe(true);
  await page.locator('[data-memory-card="c-ses_pagoda"]').click();
  const t = page.locator('[data-transcript="c:ses_pagoda"]');
  await expect(t.locator('.mem-block').first()).toBeVisible();
  await t.locator('[data-block="3"] details').first().evaluate((d) => ((d as HTMLDetailsElement).open = true));
  expect(await noHorizontalOverflow(page)).toBe(true);
  const box = await t.boundingBox();
  expect(box!.width).toBeGreaterThan(340);
  expect(box!.x + box!.width).toBeLessThanOrEqual(390);
});

// ------------------------------------------------------------------ the disk cache tab (C1)
async function disk(page: Page): Promise<void> {
  await memory(page, '&mtab=disk');
  await expect(page.locator('qse-cond-disk')).toBeVisible();
}

test('the disk tab shows what is on disk, the free space and the most-hit prefixes @mock', async ({ page }) => {
  await disk(page);
  const tiles = page.locator('[data-disk-tiles] .stat');
  await expect(tiles).toHaveCount(4);
  await expect(tiles.nth(1)).toContainText('276 GB');
  await expect(tiles.nth(2)).toContainText('65 %');
  const rows = page.locator('[data-disk-entries] tbody tr');
  await expect(rows).toHaveCount(6);
  await expect(page.locator('[data-disk-row="opencode-head-27648"]')).toContainText('You are opencode');
  await expect(page.locator('[data-disk-row="package-31744"]')).toContainText('voxel-seasons');
  await expect(page.locator('[data-disk-row="prefix-61440"] [data-disk-preview]')).toHaveCount(0);
  await expect(page.locator('[data-disk-free]')).toContainText('276 GB free');
  await page.locator('[data-mem-tab="sessions"]').click();
  await expect(page.locator('[data-memory-sessions] tbody tr')).toHaveCount(5);
});

test('the disk settings save as a new settings version, and the floor holds @mock', async ({ page }) => {
  await disk(page);
  const sec = page.locator('[data-sec="disk"]');
  await sec.locator('[data-disk-setting="budget_gb"]').fill('80');
  await sec.locator('[data-disk-setting="ttl_days"]').fill('14');
  await sec.getByLabel('change note').fill('disk cache: 80 GB, two weeks');
  await sec.locator('[data-save]').click();
  await expect(sec.locator('[data-saved]')).toContainText('version');
  const s = await (await page.request.get('/v1/conductor/settings')).json();
  expect(s.doc.memory.disk_cache).toEqual({ mode: 'on', budget_gb: 80, ttl_days: 14, admit_after: 2, min_free_gb: 50 });
  const d = await (await page.request.get('/v1/conductor/memory/disk')).json();
  expect(d.settings.budget_gb).toBe(80);
  await sec.locator('[data-disk-setting="min_free_gb"]').fill('20');
  await expect(sec.locator('[data-disk-errors]')).toContainText('at least 50 GB');
  // the server's check refuses it too, and the save waits for a valid document
  await expect(sec.locator('[data-errors]')).toContainText('min_free_gb');
  await expect(sec.locator('[data-save]')).toBeDisabled();
  const again = await (await page.request.get('/v1/conductor/settings')).json();
  expect(again.doc.memory.disk_cache.min_free_gb).toBe(50);
  await sec.locator('[data-disk-setting="min_free_gb"]').fill('50');
  await sec.locator('[data-disk-setting="budget_gb"]').fill('400');
  await expect(sec.locator('[data-disk-short]')).toContainText('fit above the 50 GB');
});

test('clear empties the disk cache after a confirmation @mock', async ({ page }) => {
  await disk(page);
  await page.locator('[data-disk-clear]').click();
  const dialog = page.getByRole('dialog', { name: 'Clear the disk cache?' });
  await expect(dialog).toBeVisible();
  await dialog.getByRole('button', { name: 'Clear' }).click();
  await expect(page.locator('[data-disk-cleared]')).toContainText('cleared 6 entries');
  await expect(page.locator('[data-disk-entries]')).toHaveCount(0);
  await expect(page.locator('#cond-disk')).toContainText('Nothing on disk yet');
  await expect(page.locator('[data-disk-clear]')).toBeDisabled();
});

test('a server without the tier says so and keeps its settings editable @mock', async ({ page }) => {
  await disk(page);
  await page.request.post('/__mock/conductor', { data: { diskRunning: false } });
  await page.reload();
  await expect(page.locator('#cond-disk')).toContainText('does not run on this server');
  await expect(page.locator('[data-sec="disk"] [data-disk-setting="budget_gb"]')).toBeVisible();
});

test('phone width: the disk tab stacks, its entries are cards, nothing sideways @mock', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await disk(page);
  await expect(page.locator('[data-disk-entries]')).toBeHidden();
  await expect(page.locator('[data-disk-card]')).toHaveCount(6);
  expect(await noHorizontalOverflow(page)).toBe(true);
  await page.locator('[data-disk-events] > summary').click();
  expect(await noHorizontalOverflow(page)).toBe(true);
});
