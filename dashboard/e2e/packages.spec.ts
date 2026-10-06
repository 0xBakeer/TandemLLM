// G2 (2026-10-03), the Packages and Skills screen on the mock engine, desktop 1440 and phone 390:
// the one list of every kind with its tokens, limit and use; the search (the library through the
// server's own ranking) and the filters; a package's detail (rendered and raw text, frontmatter,
// reference, usage, a library skill's neighbours); an inline edit with the token counter and the
// limit warning, checked and diffed before it saves as a new version (and the admin token asked
// for when the page may not write); a new inline package; attach and detach through the router
// document, general packages first; a version restore; the copy of a file to an inline package;
// and the library's retrieval per profile with the picks of its shadow.
import { expect, test, type Page } from '@playwright/test';
import { IS_MOCK, login, noHorizontalOverflow, setMode } from './helpers';

test.skip(!IS_MOCK, 'the packages spec runs on the mock tier (@mock)');

const SHOTS = process.env.PW_SHOTS ?? 'e2e-artifacts';

async function open(page: Page, extra = '', opts: Record<string, unknown> = {}): Promise<void> {
  await setMode(page.request, 'ok');
  await page.request.post('/__mock/conductor', { data: { reset: true, requireToken: false, guard: false, roleDown: null, library: true, ...opts } });
  await login(page, `#/conductor?screen=packages${extra}`);
  await expect(page.locator('qse-conductor')).toBeVisible();
  await expect(page.locator('[data-pk-row], [data-lib-profile]').first()).toBeAttached();
}

const rowNames = (page: Page) => page.locator('[data-pk-row]').evaluateAll((rows) => rows.map((r) => r.getAttribute('data-pk-row')));
const isPhone = (name: string) => name.includes('phone');

test('one list of every kind, with tokens, limits and use @mock', async ({ page }, info) => {
  await open(page);
  const kinds = await page.locator('[data-pk-row]').evaluateAll((rows) => [...new Set(rows.map((r) => r.getAttribute('data-category')))]);
  expect(kinds).toEqual(['package', 'task_skill', 'test_skill', 'card', 'library']);
  const card = page.locator('[data-pk-row="tandem-library"]');
  await expect(card).toContainText('card');
  await expect(card.locator('[data-pk-tokens]')).toContainText('/ 200');
  await expect(card.locator('[data-pk-tokens]')).toContainText('over');
  const coding = page.locator('[data-pk-row="coding"]');
  await expect(coding).toContainText('general');
  await expect(coding.locator('[data-class="code_edit"]')).toBeVisible();
  await expect(page.locator('[data-pk-row="ui-design"] [data-pk-usage]')).toContainText('shadow');
  await expect(page.locator('[data-pk-row="browser-game"] [data-pk-usage]')).toContainText('picks');
  await expect(page.locator('[data-pk-over-count]')).toBeVisible();
  await page.screenshot({ path: `${SHOTS}/g2-list-${isPhone(info.project.name) ? 'phone' : 'desktop'}.png`, fullPage: false });
  if (isPhone(info.project.name)) expect(await noHorizontalOverflow(page)).toBe(true);
});

test('filters: kind, class, switch and over the limit @mock', async ({ page }) => {
  await open(page);
  await page.locator('[data-pk-filter="kind"]').selectOption('library');
  expect(await rowNames(page)).toEqual(['browser-game', 'canvas-2d', 'debug-method', 'pandas-cleaning', 'threejs-scene']);
  await page.locator('[data-pk-filter="kind"]').selectOption('all');
  await page.locator('[data-pk-filter="class"]').selectOption('code_edit');
  const ce = await rowNames(page);
  expect(ce.slice(0, 3)).toEqual(['coding', 'testing', 'test-web']);
  expect(ce).toContain('browser-game');
  expect(ce).not.toContain('ui-design');
  await page.locator('[data-pk-filter="class"]').selectOption('');
  await page.locator('[data-pk-filter="over"]').check();
  const over = await rowNames(page);
  expect(over).toContain('tandem-library');
  expect(over).not.toContain('coding');
  await page.locator('[data-pk-filter="over"]').uncheck();
  await page.locator('[data-pk-filter="enabled"]').selectOption('off');
  await expect(page.locator('[data-pk-empty]')).toBeVisible();
});

test('search: the library by its own ranking, the rest by text @mock', async ({ page }, info) => {
  await open(page);
  const search = page.locator('[data-pk-search]');
  await search.fill('canvas animating game');
  await expect(page.locator('[data-pk-row="canvas-2d"] [data-score]')).toHaveText(/^\d+\.\d\d$/);
  expect(await rowNames(page)).toEqual(['canvas-2d', 'browser-game']);
  // the text here finds coding's description; the library's own search finds "smallest"
  await search.fill('small steps');
  expect(await rowNames(page)).toEqual(['coding', 'debug-method']);
  await expect(page.locator('[data-pk-row="coding"] [data-score]')).toHaveCount(0);
  await expect(page.locator('[data-pk-row="debug-method"] [data-score]')).toBeVisible();
  await search.fill('zzzz');
  await expect(page.locator('[data-pk-empty]')).toContainText('zzzz');
  await search.fill('');
  await expect(page.locator('[data-pk-row]')).toHaveCount(17);
  if (isPhone(info.project.name)) expect(await noHorizontalOverflow(page)).toBe(true);
});

test('detail: the text rendered and raw, frontmatter, reference, usage and neighbours @mock', async ({ page }, info) => {
  await open(page);
  await page.locator('[data-pk-row="browser-game"]').click();
  const d = page.locator('[data-pk-detail="browser-game"]');
  await expect(d).toBeVisible();
  await expect(d.locator('[data-pk-category]')).toHaveText('library skill');
  await expect(d.locator('[data-pk-front]')).toContainText('description');
  await expect(d.locator('[data-pk-md] li').first()).toContainText('Rule 1');
  await expect(d.locator('[data-pk-path]')).toContainText('library/browser-game/SKILL.md');
  await d.locator('[data-pk-ref] summary').click();
  await expect(d.locator('[data-pk-ref] [data-pk-md]')).toContainText('function step');
  await d.getByRole('radio', { name: 'raw' }).click();
  await expect(d.locator('[data-pk-raw]').first()).toContainText('# browser-game');
  await expect(d.locator('[data-pk-raw]').first()).toContainText('- Rule 1');
  await page.screenshot({ path: `${SHOTS}/g2-detail-${isPhone(info.project.name) ? 'phone' : 'desktop'}.png` });
  if (isPhone(info.project.name)) {
    await expect(page.locator('[data-pk-list]')).toBeHidden();
    expect(await noHorizontalOverflow(page)).toBe(true);
  }
  await d.locator('[data-pk-pane="usage"]').click();
  await expect(d.locator('[data-pk-usage-tiles]')).toContainText('library picks');
  await expect(d.locator('[data-pk-recent] [data-pk-use]').first()).toBeVisible();
  const nb = d.locator('[data-pk-neighbour]').first();
  await expect(nb).toBeVisible();
  const next = await nb.getAttribute('data-pk-neighbour');
  await nb.click();
  await expect(page.locator(`[data-pk-detail="${next}"]`)).toBeVisible();
  await page.locator('[data-pk-close]').click();
  await expect(page.locator('[data-pk-detail]')).toHaveCount(0);
  await expect(page.locator('[data-pk-list]')).toBeVisible();
});

test('edit an inline package: counter, limit warning, check, diff, save; no token asked (AU1) @mock', async ({ page }, info) => {
  await open(page);
  await page.locator('[data-pk-row="testing"]').click();
  const d = page.locator('[data-pk-detail="testing"]');
  await d.locator('[data-pk-pane="edit"]').click();
  const text = '# Testing\n\nTest through input only.\n\n- Click, type, wait real frames.\n- Read the console.';
  await d.locator('[data-pk-body]').fill(text);
  // the estimate at once, then the engine's count (the mock's tokenizer is denser than 3.5)
  await expect(d.locator('[data-pk-editor-count]')).not.toContainText('≈', { timeout: 5000 });
  await expect(d.locator('[data-pk-editor-count]')).toContainText(`${Math.round(text.length / 3.2)} of 3,000`);
  await d.locator('[data-pk-max]').fill('20');
  await expect(d.locator('[data-pk-editor-warn]')).toContainText('max_tokens');
  await expect(d.locator('[data-pk-limit-warn]')).toBeVisible();
  await d.locator('[data-pk-max]').fill('3000');
  await expect(d.locator('[data-pk-editor-warn]')).toHaveCount(0);
  await d.getByRole('radio', { name: 'preview' }).click();
  await expect(d.locator('[data-pk-preview] li')).toHaveCount(2);
  await page.screenshot({ path: `${SHOTS}/g2-edit-${isPhone(info.project.name) ? 'phone' : 'desktop'}.png` });
  await page.locator('[data-check-save="packages"]').click();
  const review = page.locator('[data-review="packages"]');
  await expect(review).toContainText('The server accepts');
  await expect(review.locator('[data-diff]')).toContainText('testing');
  await expect(review.locator('[data-review-save="packages"]')).toBeDisabled();
  await page.locator('[data-note="packages"]').fill('testing: the input rule');
  await review.locator('[data-review-save="packages"]').click();
  // AU1: no password set, the save goes straight through: no dialog, no token field
  await expect(page.locator('[data-saved="packages"]')).toContainText('version');
  await expect(page.locator('qse-cond-dialog, #cond-token')).toHaveCount(0);
  await expect(page.locator('[data-review="packages"]')).toHaveCount(0);
  const doc = await (await page.request.get('/v1/conductor/packages')).json();
  expect(doc.doc.packages.testing.body).toContain('Read the console.');
  expect(doc.meta.note).toBe('testing: the input rule');
});

test('a new inline package is created, checked and saved @mock', async ({ page }) => {
  await open(page);
  await page.locator('[data-pk-new]').click();
  const d = page.locator('[data-pk-detail="package-1"]');
  await expect(d).toBeVisible();
  await expect(page.locator('[data-pk-row="package-1"]')).toContainText('not saved');
  await d.locator('[data-pk-title]').fill('Release notes');
  await d.locator('[data-pk-description]').fill('How to write the release notes of this project.');
  await d.locator('[data-pk-body]').fill('# Release notes\n\nOne line a change, newest first.');
  await page.locator('[data-check-save="packages"]').click();
  await expect(page.locator('[data-review="packages"] [data-diff]')).toContainText('package-1');
  await page.locator('[data-note="packages"]').fill('a release-notes package');
  await page.locator('[data-review-save="packages"]').click();
  await expect(page.locator('[data-saved="packages"]')).toContainText('version');
  await expect(page.locator('[data-pk-row="package-1"]')).not.toContainText('not saved');
  const doc = await (await page.request.get('/v1/conductor/packages')).json();
  expect(doc.doc.packages['package-1']).toMatchObject({ title: 'Release notes', source: 'inline' });
});

test('attach and detach through the router document, general first @mock', async ({ page }, info) => {
  await open(page, '&pkg=ui-design&pview=classes');
  const d = page.locator('[data-pk-detail="ui-design"]');
  await expect(d.locator('[data-pk-classes]')).toBeVisible();
  await d.locator('[data-pk-attach="code_edit"]').check();
  await expect(d.locator('[data-pk-order="code_edit"]')).toHaveText(/1\. coding.*2\. testing.*3\. test-web.*4\. ui-design/s);
  await expect(page.locator('[data-savebar="router"]')).toBeVisible();
  await page.screenshot({ path: `${SHOTS}/g2-attach-${isPhone(info.project.name) ? 'phone' : 'desktop'}.png` });
  await page.locator('[data-check-save="router"]').click();
  await expect(page.locator('[data-review="router"] [data-diff]')).toContainText('ui-design');
  await page.locator('[data-note="router"]').fill('ui-design on code_edit');
  await page.locator('[data-review-save="router"]').click();
  await expect(page.locator('[data-saved="router"]')).toContainText('version');
  let router = await (await page.request.get('/v1/conductor/router', { headers: { Authorization: 'Bearer mock' } })).json();
  expect(router.doc.classes.find((c: { id: string }) => c.id === 'code_edit').policy.packages).toEqual(['coding', 'testing', 'test-web', 'ui-design']);
  await expect(page.locator('[data-pk-row="ui-design"] [data-class="code_edit"]')).toHaveCount(1);
  await d.locator('[data-pk-attach="code_edit"]').uncheck();
  await page.locator('[data-check-save="router"]').click();
  await page.locator('[data-note="router"]').fill('ui-design off code_edit');
  await page.locator('[data-review-save="router"]').click();
  await expect(page.locator('[data-saved="router"]')).toContainText('version');
  router = await (await page.request.get('/v1/conductor/router', { headers: { Authorization: 'Bearer mock' } })).json();
  expect(router.doc.classes.find((c: { id: string }) => c.id === 'code_edit').policy.packages).toEqual(['coding', 'testing', 'test-web']);
  // a library skill is searched, not listed
  await login(page, '#/conductor?screen=packages&pkg=debug-method&pview=classes');
  await expect(page.locator('[data-pk-library-classes]')).toBeVisible();
  await expect(page.locator('[data-pk-attach]')).toHaveCount(0);
});

test('switch, check, save, then restore the version before @mock', async ({ page }) => {
  await open(page, '&pkg=pencil-sketch');
  const d = page.locator('[data-pk-detail="pencil-sketch"]');
  await d.locator('[data-pk-enabled]').uncheck();
  await page.locator('[data-check-save="packages"]').click();
  await expect(page.locator('[data-review="packages"] [data-diff]')).toContainText('enabled');
  await page.locator('[data-note="packages"]').fill('pencil-sketch off');
  await page.locator('[data-review-save="packages"]').click();
  await expect(page.locator('[data-saved="packages"]')).toContainText('version');
  await expect(page.locator('[data-pk-row="pencil-sketch"]')).toContainText('off');
  await page.locator('[data-versions-btn="packages"]').click();
  const versions = page.locator('[data-versions]');
  await expect(versions.locator('[data-version]').first()).toContainText('live');
  const before = versions.locator('[data-version]').nth(1);
  const v = await before.getAttribute('data-version');
  await before.getByRole('button', { name: 'Restore' }).click();
  await page.locator('[data-confirm]').click();
  await expect(page.locator('[data-saved="packages"]')).toContainText(`restore of v${v}`);
  const doc = await (await page.request.get('/v1/conductor/packages')).json();
  expect(doc.doc.packages['pencil-sketch'].enabled).toBe(true);
  expect(doc.meta.restored_from).toBe(Number(v));
});

test('a file on the box is read-only, or copied to an inline package @mock', async ({ page }) => {
  await open(page, '&pkg=pencil-sketch&pview=edit');
  const d = page.locator('[data-pk-detail="pencil-sketch"]');
  await expect(d.locator('[data-pk-readonly]')).toContainText('SKILL.md');
  await expect(d.locator('[data-pk-body]')).toHaveCount(0);
  await d.locator('[data-pk-copy-inline-edit]').click();
  await expect(d.locator('[data-pk-body]')).toHaveValue(/# pencil-sketch/);
  await expect(page.locator('[data-check-save="packages"]')).toBeEnabled();
  await page.locator('[data-check-save="packages"]').click();
  await expect(page.locator('[data-review="packages"] [data-diff]')).toContainText('inline');
});

test('the library: each profile switch beside the picks of its shadow @mock', async ({ page }, info) => {
  await open(page, '&ptab=library');
  const profiles = page.locator('[data-lib-profiles]');
  await expect(profiles.locator('[data-lib-profile="coding"]')).toBeVisible();
  await expect(profiles.locator('[data-lib-profile="raw"]').getByRole('radio', { name: 'on' })).toBeDisabled();
  const rows = page.locator('[data-lib-row]');
  await expect(rows.first()).toBeVisible();
  await expect(rows.first()).toContainText('shadow');
  await expect(page.locator('[data-lib-tiles]')).toContainText('messages searched');
  await expect(page.locator('.lib-cand.is-chosen').first()).toBeVisible();
  await page.screenshot({ path: `${SHOTS}/g2-library-${isPhone(info.project.name) ? 'phone' : 'desktop'}.png`, fullPage: false });
  if (isPhone(info.project.name)) expect(await noHorizontalOverflow(page)).toBe(true);
  await page.locator('[data-lib-filter]').selectOption('chat');
  await expect(page.locator('[data-lib-rows]')).toHaveCount(0);
  await page.locator('[data-lib-filter]').selectOption('');
  // switch coding to shadow: checked, the diff in the confirmation, a new version
  await profiles.locator('[data-lib-profile="coding"]').getByRole('radio', { name: 'shadow' }).click();
  const dialog = page.locator('qse-cond-dialog');
  await expect(dialog).toContainText('library');
  await expect(dialog.locator('[data-diff]')).toContainText('shadow');
  await dialog.locator('[data-confirm]').click();
  await expect(page.locator('[data-lib-saved]')).toContainText('coding: library shadow');
  await expect(profiles.locator('[data-lib-profile="coding"]').getByRole('radio', { name: 'shadow' })).toHaveAttribute('aria-checked', 'true');
  const prof = await (await page.request.get('/v1/conductor/profiles/coding', { headers: { Authorization: 'Bearer mock' } })).json();
  expect(prof.doc.library).toBe('shadow');
  expect(prof.meta.note).toBe('library to shadow');
  // a pick opens its skill
  const pick = page.locator('[data-lib-open]').first();
  const name = await pick.getAttribute('data-lib-open');
  await pick.click();
  await expect(page.locator(`[data-pk-detail="${name}"]`)).toBeVisible();
});
