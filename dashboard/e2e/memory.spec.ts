// VIS-49's Playwright rows for the Memory screen (VIS-48) on the mock tier: the sessions list,
// a transcript with its hidden notes highlighted and a jump to their journal row, the compaction
// folded, the keep toggle, the settings saved as a new version, the token asked for, and the phone
// layout at 390 px (cards, the viewer full width, nothing sideways).
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
