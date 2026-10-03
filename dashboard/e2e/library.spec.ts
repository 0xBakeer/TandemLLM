// R1 (2026-10-03), the package library in the dashboard, on the mock engine: the Packages screen
// lists the library as a kind of its one list (G2), searches it through GET /v1/library/search
// (the server's ranking, its scores), says when nothing matches, opens a skill; a profile's
// library switch saves as a new version; and the screen fits a 390 px phone.
import { expect, test, type Page } from '@playwright/test';
import { IS_MOCK, login, noHorizontalOverflow, setMode } from './helpers';

test.skip(!IS_MOCK, 'the library spec runs on the mock tier (@mock)');

async function open(page: Page, screen: string, extra = ''): Promise<void> {
  await setMode(page.request, 'ok');
  await page.request.post('/__mock/conductor', { data: { reset: true, requireToken: false, guard: false, roleDown: null, library: true } });
  await login(page, `#/conductor?screen=${screen}${extra}`);
  await expect(page.locator('qse-conductor')).toBeVisible();
}

test('the Packages screen lists and searches the library @mock', async ({ page }) => {
  await open(page, 'packages');
  // G2: one list; the library is a kind of it, searched by the server's own index
  await page.locator('[data-pk-filter="kind"]').selectOption('library');
  await expect(page.locator('[data-pk-row]')).toHaveCount(5);
  await expect(page.locator('[data-pk-row="threejs-scene"]')).toContainText('three_d');
  await page.locator('[data-pk-filter="kind"]').selectOption('all');
  await expect(page.locator('[data-pk-row="ui-design"]')).toHaveCount(1);
  await page.locator('[data-pk-search]').fill('canvas animating game');
  await expect(page.locator('[data-pk-row]').first()).toHaveAttribute('data-pk-row', 'canvas-2d');
  await expect(page.locator('[data-score]').first()).toHaveText(/^\d+\.\d\d$/);
  const names = await page.locator('[data-pk-row]').evaluateAll((rows) => rows.map((r) => r.getAttribute('data-pk-row')));
  expect(names).toEqual(['canvas-2d', 'browser-game']);
  await page.locator('[data-pk-search]').fill('zzzz');
  await expect(page.locator('[data-pk-empty]')).toContainText('zzzz');
  await page.locator('[data-pk-search]').fill('');
  await page.locator('[data-pk-row="debug-method"]').click();
  await expect(page.locator('[data-pk-detail="debug-method"]')).toBeVisible();
});

test('a profile turns the library to shadow @mock', async ({ page }) => {
  await open(page, 'profiles', '&edit=coding');
  const sec = page.locator('[data-sec="library"]');
  await sec.locator('summary').click();
  const seg = sec.getByRole('radiogroup', { name: 'package library' });
  await expect(seg.getByRole('radio', { name: 'off' })).toHaveAttribute('aria-checked', 'true');
  await seg.getByRole('radio', { name: 'shadow' }).click();
  const editor = page.locator('[data-editor="coding"]');
  await editor.getByLabel('change note').fill('library shadow');
  await editor.locator('[data-save]').click();
  await expect(editor.locator('[data-saved]')).toContainText('version');
  const prof = await (await page.request.get('/v1/conductor/profiles/coding', { headers: { Authorization: 'Bearer mock' } })).json();
  expect(prof.doc.library).toBe('shadow');
  // the Routing screen's package picks do not list the library's skills
  await login(page, '#/conductor?screen=routing');
  await expect(page.locator('qse-conductor')).toBeVisible();
  const ce = page.locator('[data-class-row="code_edit"]');
  await ce.locator('summary').click();
  await expect(ce.locator('.pkg-picks .pick', { hasText: 'ui-design' })).toBeVisible();
  await expect(ce.locator('.pkg-picks .pick', { hasText: 'browser-game' })).toHaveCount(0);
});

test('phone width: the list scrolls inside the page, nothing sideways @mock', async ({ page }, info) => {
  test.skip(!info.project.name.includes('phone'), 'phone project only');
  await open(page, 'packages');
  await page.locator('[data-pk-search]').fill('game');
  await expect(page.locator('[data-pk-row]').first()).toBeVisible();
  expect(await noHorizontalOverflow(page)).toBe(true);
  await page.locator('[data-pk-list]').screenshot({ path: 'e2e-artifacts/library-phone.png' });
});
