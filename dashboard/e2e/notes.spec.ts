// The Memory screen's Notes tab (2026-10-05) on the mock tier: the tree with its counts, a note as a
// file, an edit that becomes a ratified revision with its diff in the history, ratify and pin, the
// runtime's search with its scores, the requests a note went into, archive and delete, the export,
// the per-user delete behind its typed confirmation, and the phone at 390 px (the tree a picker, the
// note full width, nothing sideways). Synthetic notes only.
import { expect, test, type Page } from '@playwright/test';
import { IS_MOCK, login, noHorizontalOverflow, setMode } from './helpers';

test.skip(!IS_MOCK, 'the Notes spec runs on the mock tier (@mock)');

async function notes(page: Page, extra = ''): Promise<void> {
  await setMode(page.request, 'ok');
  await page.request.post('/__mock/conductor', { data: { reset: true, requireToken: false, guard: false, roleDown: null } });
  await login(page, `#/conductor?screen=memory&mtab=notes${extra}`);
  await expect(page.locator('qse-cond-notes')).toBeVisible();
}

const open = (page: Page, id: string) => page.locator(`[data-note-row="${id}"], [data-note-card="${id}"]`).filter({ visible: true }).click();

test('the tree, the list and the tiles show the memory as files @mock', async ({ page }) => {
  await notes(page);
  await expect(page.locator('[data-notes-tiles]')).toContainText('live notes');
  await expect(page.locator('[data-notes-tiles]')).toContainText('6');
  const rows = page.locator('[data-note-row], [data-note-card]').filter({ visible: true });
  await expect(rows).toHaveCount(6);
  // the scope picks the node (the tree on a desk, the picker on a phone)
  const tree = page.locator('[data-notes-tree]');
  if (await tree.isVisible()) await tree.locator('[data-node="user"]').click();
  else await page.locator('[data-notes-picker]').selectOption('user');
  await expect(rows).toHaveCount(2);
  await expect(page.locator('[data-notes-root]')).toContainText('1 archived');
});

test('a note opens as a file; an edit is a ratified revision with its diff @mock', async ({ page }) => {
  await notes(page);
  await open(page, '6700a1b20002');
  const n = page.locator('[data-note="6700a1b20002"]');
  await expect(n).toContainText('Personal notes are written in German');
  await expect(n.locator('[data-note-meta]')).toContainText('provisional');
  await n.locator('[data-note-act="edit"]').click();
  await n.locator('[data-edit="body"]').fill('Diary and personal notes in German. Code and commits in English.');
  await n.locator('[data-edit-save]').click();
  await expect(page.locator('[data-notes-flash]')).toContainText('ratified');
  await expect(n.locator('[data-note-meta]')).toContainText('ratified');
  await expect(n.locator('[data-note-meta]')).toContainText('you');
  await n.locator('[data-note-history] [data-rev="0"]').click();
  await expect(n.locator('[data-note-diff] .diff-add')).toContainText('Code and commits in English.');
  await expect(n.locator('[data-note-diff] .diff-del')).toContainText('Diary and personal notes go in German');
  // a ratified rule can be pinned
  await n.locator('[data-note-act="pin"]').click();
  await expect(n.locator('[data-note-meta]')).toContainText('pinned');
});

test('ratify, the runtime search with scores, and the injected requests @mock', async ({ page }) => {
  await notes(page);
  await open(page, '6700a1b20003');
  const n = page.locator('[data-note="6700a1b20003"]');
  await expect(n.locator('[data-note-uses]')).toContainText('2 injections');
  await n.locator('[data-note-act="ratify"]').click();
  await expect(n.locator('[data-note-meta]')).toContainText('ratified');
  await page.locator('[data-notes-query]').fill('summer scene');
  await page.locator('[data-notes-search]').click();
  await expect(page.locator('[data-notes-hits] li').first()).toContainText('offerings recolour');
  await expect(page.locator('[data-notes-hits] [data-score]').first()).not.toHaveText('0.00');
  await page.locator('[data-notes-injected] > summary').click();
  await expect(page.locator('[data-notes-injected]')).toContainText('c:ses_pagoda');
  await page.locator('[data-notes-writes] > summary').click();
  await expect(page.locator('[data-notes-writes]')).toContainText('refused (cap)');
});

test('archive, delete, export and the typed delete of a user @mock', async ({ page }) => {
  await notes(page);
  await open(page, '6700a1b20005');
  page.once('dialog', (d) => d.accept('the task is done'));
  await page.locator('[data-note="6700a1b20005"] [data-note-act="archive"]').click();
  await expect(page.locator('[data-note="6700a1b20005"] [data-note-meta]')).toContainText('archived');
  await expect(page.locator('[data-notes-root]')).toContainText('2 archived');
  await page.locator('[data-note="6700a1b20005"] [data-note-act="delete"]').click();
  await page.getByRole('button', { name: 'Delete', exact: true }).click();
  await expect(page.locator('[data-note="6700a1b20005"]')).toHaveCount(0);
  const dl = page.waitForEvent('download');
  await page.locator('[data-notes-export]').click();
  expect((await dl).suggestedFilename()).toBe('memory-owner.zip');
  await page.locator('[data-notes-danger] > summary').click();
  const del = page.locator('[data-notes-delete-user]');
  await expect(del).toBeDisabled();
  await page.locator('[data-notes-confirm]').fill('owner');
  await expect(del).toBeEnabled();
  await del.click();
  await expect(page.locator('[data-notes-flash]')).toContainText("owner's memory deleted");
});

test('the phone: the tree is a picker, the note takes the width, nothing goes sideways @mock', async ({ page }) => {
  await notes(page);
  await open(page, '6700a1b20003');
  await expect(page.locator('[data-note="6700a1b20003"]')).toBeVisible();
  expect(await noHorizontalOverflow(page)).toBe(true);
  const vw = page.viewportSize()!.width;
  if (vw <= 768) {
    await expect(page.locator('[data-notes-tree]')).toBeHidden();
    await expect(page.locator('[data-notes-picker]')).toBeVisible();
    const box = await page.locator('[data-note="6700a1b20003"]').boundingBox();
    expect(box!.width).toBeGreaterThan(vw - 60);
  } else {
    await expect(page.locator('[data-notes-tree]')).toBeVisible();
  }
});
