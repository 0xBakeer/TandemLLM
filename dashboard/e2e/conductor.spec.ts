// The Conductor view (VIS-42, QA VIS-43; SRV-129's UI rows, QA SRV-130), on the mock engine:
// the mode switch asks before "on" and shows the reload; promotion is gated by the rule; the raw
// and bench profiles are read-only; a pair from two builds is "not comparable"; the guard line
// shows when no speed guard ran for the serving build; a class added on the Routing screen saves
// as a new version; the test box shows the exact note; the drawer shows the exact injected text
// and takes a label; editing asks for the admin token when the server wants it; and every screen
// fits a 390 px phone. SRV-113/115 (QA SRV-114/116): the Monitor screen renders every tool check's
// rules from the policy's schema and saves an edit as a new monitor version; its journal opens a
// decision with the exact hidden text and takes a label; the Overview lists the monitor's last
// actions; a session's strip marks the monitor's acts and its export holds both halves. All
// mock-only (the seeded day and the switches are the mock's).
import { expect, test, type Page } from '@playwright/test';
import { IS_MOCK, login, noHorizontalOverflow, setMode } from './helpers';

test.skip(!IS_MOCK, 'the Conductor spec runs on the mock tier (@mock)');

const SCREENS = ['overview', 'routing', 'monitor', 'profiles', 'packages', 'sessions', 'compare'];

async function open(page: Page, screen = 'overview', extra = ''): Promise<void> {
  // another spec may leave the mock with the login off, where a write rightly asks for the token
  await setMode(page.request, 'ok');
  await page.request.post('/__mock/conductor', { data: { reset: true, requireToken: false, guard: false, roleDown: null } });
  await login(page, `#/conductor?screen=${screen}${extra}`);
  await expect(page.locator('qse-conductor')).toBeVisible();
}

test('mode switch with confirmation: shadow, then on after a confirmation, with the reload time @mock', async ({ page }) => {
  await open(page);
  await page.getByLabel('active profile').selectOption('coding');
  const sw = page.getByRole('radiogroup', { name: 'mode of coding' });
  await sw.getByRole('radio', { name: 'shadow' }).click();
  await expect(page.locator('[data-reloaded]')).toContainText('reloaded');
  await expect(sw.getByRole('radio', { name: 'shadow' })).toHaveAttribute('aria-checked', 'true');
  await sw.getByRole('radio', { name: 'on' }).click();
  const dialog = page.getByRole('dialog', { name: 'Turn the coding profile on?' });
  await expect(dialog).toBeVisible();
  await dialog.getByRole('button', { name: 'Turn on' }).click();
  await expect(sw.getByRole('radio', { name: 'on' })).toHaveAttribute('aria-checked', 'true');
  await expect(page.locator('[data-reloaded]')).toContainText(/v\d+ · reloaded/);
  const versions = await (await page.request.get('/v1/conductor/profiles/coding/versions', { headers: { Authorization: 'Bearer mock' } })).json();
  expect(versions.versions[0].note).toBe('mode to on');
});

test('cancelling the confirmation changes nothing @mock', async ({ page }) => {
  await open(page);
  await page.getByLabel('active profile').selectOption('overnight');
  const sw = page.getByRole('radiogroup', { name: 'mode of overnight' });
  await sw.getByRole('radio', { name: 'shadow' }).click();
  await expect(sw.getByRole('radio', { name: 'shadow' })).toHaveAttribute('aria-checked', 'true');
  await sw.getByRole('radio', { name: 'on' }).click();
  await page.getByRole('dialog').getByRole('button', { name: 'Cancel' }).click();
  await expect(sw.getByRole('radio', { name: 'shadow' })).toHaveAttribute('aria-checked', 'true');
});

test('promotion is gated: D1 at 0.93 over 41 may act, E2 with 12 may not @mock', async ({ page }) => {
  await open(page, 'compare');
  const d1 = page.locator('[data-precision-row="task/D1"]');
  await expect(d1).toContainText('precision 0.93 over 41');
  await expect(d1.getByRole('button', { name: 'Promote to act' })).toBeEnabled();
  const e2 = page.locator('[data-precision-row="task/E2"]');
  await expect(e2).toContainText('12 of 30 labelled decisions');
  await expect(e2.getByRole('button', { name: 'Promote to act' })).toBeDisabled();
  await expect(page.locator('#cond-precision')).toContainText('on is allowed at precision ≥ 0.9 over ≥ 30 labelled decisions');
  // a routing decision that earned it: promote, confirm, a new profile version
  const pk = page.locator('[data-precision-row="coding/packages"]');
  await pk.getByRole('button', { name: 'Promote to on' }).click();
  await page.getByRole('dialog').getByRole('button', { name: 'Promote' }).click();
  await expect(page.locator('.note-info')).toContainText('packages promoted');
  const doc = await (await page.request.get('/v1/conductor/profiles/coding', { headers: { Authorization: 'Bearer mock' } })).json();
  expect(doc.doc.decisions.packages).toBe('on');
  expect(doc.meta.note).toMatch(/^packages to on, precision 0\.94 \/ 35$/);
});

test('raw and bench are locked in the UI @mock', async ({ page }) => {
  await open(page, 'profiles', '&edit=bench');
  await expect(page.locator('[data-lock-reason]')).toContainText('raw passthrough, byte-identical; benchmarks and the paper');
  const editor = page.locator('[data-editor="bench"]');
  for (const input of await editor.locator('[data-sec="mapping"] input, [data-sec="decisions"] button').all()) await expect(input).toBeDisabled();
  await expect(editor.locator('[data-savebar]')).toContainText('locked');
  await expect(editor.getByRole('button', { name: 'Save as new version' })).toHaveCount(0);
  await page.getByLabel('active profile').selectOption('raw');
  for (const r of await page.getByRole('radiogroup', { name: 'mode of raw' }).getByRole('radio').all()) await expect(r).toBeDisabled();
});

test('a pair from two builds is not comparable @mock', async ({ page }) => {
  await open(page, 'compare');
  const pagoda = page.locator('[data-suite="pagoda seasons"]').first();
  await expect(pagoda.locator('[data-not-comparable]')).toBeVisible();
  await expect(pagoda).toContainText('builds 0.3.0 and 0.3.1-test');
  await expect(page.locator('[data-suite="spiral portrait"]').first().locator('[data-not-comparable]')).toHaveCount(0);
});

test('the guard line shows when no speed guard ran for the serving build @mock', async ({ page }) => {
  await open(page);
  await expect(page.locator('[data-guard]')).toContainText('speed guard not run for build 0.3.1-test');
  await page.request.post('/__mock/conductor', { data: { guard: true } });
  await page.reload();
  await expect(page.locator('qse-conductor')).toBeVisible();
  await expect(page.locator('.cond-title')).toBeVisible();
  await page.waitForTimeout(500);
  await expect(page.locator('[data-guard]')).toHaveCount(0);
});

test('a class added on the Routing screen saves as a new router version @mock', async ({ page }) => {
  await open(page, 'routing');
  await page.locator('[data-add-class]').click();
  const row = page.locator('[data-class-row="class_1"]');
  await row.locator('[data-class-id]').fill('data_analysis');
  const added = page.locator('[data-class-row="data_analysis"]');
  await added.locator('[data-class-desc]').fill('Analyse a table or a CSV file and report the numbers.');
  await added.getByRole('radiogroup', { name: 'effort of data_analysis' }).getByRole('radio', { name: 'high', exact: true }).click();
  await page.getByLabel('change note').fill('add data_analysis');
  await expect(page.locator('[data-save]')).toBeEnabled();
  await page.locator('[data-save]').click();
  await expect(page.locator('[data-saved]')).toContainText(/version 1, reloaded/);
  const doc = await (await page.request.get('/v1/conductor/router', { headers: { Authorization: 'Bearer mock' } })).json();
  expect(doc.doc.classes.map((c: { id: string }) => c.id)).toContain('data_analysis');
  expect(doc.meta.note).toBe('add data_analysis');
  await page.locator('[data-versions-btn]').click();
  await expect(page.locator('[data-versions]')).toContainText('add data_analysis');
});

test('an invalid edit shows the server\'s reason and cannot be saved @mock', async ({ page }) => {
  await open(page, 'routing');
  await page.locator('summary', { hasText: 'The note' }).click();
  await page.locator('[data-wrapper]').fill('no placeholder');
  await expect(page.locator('[data-errors]')).toContainText('{text}');
  await page.getByLabel('change note').fill('broken');
  await expect(page.locator('[data-save]')).toBeDisabled();
});

test('the test box runs the cascade and shows the exact note @mock', async ({ page }) => {
  await open(page, 'routing');
  await page.locator('[data-test-text]').fill('Make the settings page calmer, a design_ui job.');
  await page.locator('[data-run-test]').click();
  const res = page.locator('[data-test-result]');
  await expect(res).toContainText('design_ui');
  await expect(res.locator('.stage').first()).toContainText('decider');
  await expect(page.locator('[data-test-note]')).toContainText('<runtime-note>');
  await expect(page.locator('[data-test-note]')).toContainText('# ui-design');
});

test('the drawer shows the exact injected text and takes a label @mock', async ({ page }) => {
  await open(page, 'sessions', '&session=c%3Ases_night_pagoda');
  const acted = page.locator('[data-timeline] .feed-row').filter({ hasText: 'note (' }).first();
  await acted.click();
  const drawer = page.locator('[data-drawer]');
  await expect(drawer.locator('[data-injected] .note-text')).toContainText('<runtime-note>');
  await expect(drawer).toContainText('The client never saw this text');
  await drawer.locator('[data-label="right"]').click();
  await expect(drawer).toContainText('labelled right');
  await drawer.getByRole('button', { name: 'Close' }).click();
  await expect(drawer).toHaveCount(0);
  await expect(page.locator('[data-export="md"]')).toHaveAttribute('href', /export\?format=md/);
});

test('editing asks for the admin token when the server wants it @mock', async ({ page }) => {
  await open(page, 'profiles', '&edit=chat');
  await page.request.post('/__mock/conductor', { data: { requireToken: true } });
  await page.locator('[data-editor="chat"] [data-decision-mode="rules"]').getByRole('radio', { name: 'on' }).click();
  await page.locator('[data-editor="chat"]').getByLabel('change note').fill('rules on');
  await page.locator('[data-editor="chat"] [data-save]').click();
  const dialog = page.getByRole('dialog', { name: 'Unlock editing' });
  await expect(dialog).toBeVisible();
  await dialog.locator('#cond-token').fill('mock');
  await dialog.getByRole('button', { name: 'Unlock' }).click();
  await expect(page.locator('[data-editor="chat"] [data-saved]')).toContainText('version');
});

test('a role test answers from the Profiles screen @mock', async ({ page }) => {
  await open(page, 'profiles', '&edit=coding');
  await page.locator('summary', { hasText: 'Roles' }).click();
  await page.locator('[data-role-test="decider"]').click();
  await expect(page.locator('[data-role-row="decider"]')).toContainText('ok in');
});

test('try in Playground sends the profile header and the composite model @mock', async ({ page }) => {
  await open(page, 'profiles', '&edit=coding');
  await page.locator('[data-try]').click();
  await expect(page.locator('[data-pg-profile]')).toContainText('TandemLLM · coding');
  const reqP = page.waitForRequest((r) => r.url().includes('/v1/chat/completions') && r.method() === 'POST');
  await page.locator('#pg-composer').fill('hello');
  await page.locator('.pg-send').click();
  const req = await reqP;
  expect(req.headers()['x-tandem-profile']).toBe('coding');
  expect(req.postDataJSON().model).toBe('TandemLLM');
});

test('the Monitor screen renders the tool checks from the schema and saves an edit as a new version @mock', async ({ page }) => {
  await open(page, 'monitor');
  const c4 = page.locator('[data-monitor-task="C4"]');
  await expect(c4).toContainText('malformed_call');
  await expect(c4.locator('[data-param="cut"]')).toHaveValue('give_up');
  const c1 = page.locator('[data-monitor-task="C1"]');
  await expect(c1.locator('[data-item="rm_rf_outside_root"]')).toBeVisible();
  await expect(c1.locator('[data-item="git_push"] [data-param="pattern"]')).toHaveValue(/git/);
  await expect(page.locator('[data-monitor-task="C2"] [data-param="max_lines"]')).toHaveValue('400');
  await c4.locator('input[data-param="regenerations"]').fill('1');
  await page.getByLabel('change note').fill('C4: one regeneration');
  await expect(page.locator('[data-save]')).toBeEnabled();
  await page.locator('[data-save]').click();
  await expect(page.locator('[data-saved]')).toContainText(/version 1, reloaded/);
  const doc = await (await page.request.get('/v1/conductor/monitor', { headers: { Authorization: 'Bearer mock' } })).json();
  expect(doc.doc.tasks.C4.regenerations).toBe(1);
  expect(doc.meta.note).toBe('C4: one regeneration');
  // a broken value is the server's 422 reason, not a save
  await c4.locator('[data-param="cut"]').selectOption('content');
  await page.locator('[data-monitor-task="C2"] input[data-param="max_lines"]').fill('0');
  await expect(page.locator('[data-errors]')).toBeVisible();
  await expect(page.locator('[data-save]')).toBeDisabled();
});

test('the monitor journal opens a decision with its exact hidden text and takes a label @mock', async ({ page }) => {
  await open(page, 'monitor');
  const journal = page.locator('[data-journal]');
  await expect(journal.locator('.feed-row').first()).toBeVisible();
  await page.getByLabel('kind').selectOption('acted');
  await page.getByLabel('task').selectOption('C4');
  const row = journal.locator('.feed-row').filter({ hasText: 'had the tool calls written again' }).first();
  await row.click();
  const drawer = page.locator('[data-event-drawer]');
  await expect(drawer.locator('.note-text').first()).toContainText('Your tool call was not run');
  await expect(drawer).toContainText(/Hidden text at token [\d,]+ · 48 tokens/);
  await drawer.locator('[data-label="wrong"]').click();
  await expect(drawer).toContainText('labelled wrong');
});

test('the Overview lists the monitor\'s last actions and opens one @mock', async ({ page }) => {
  await open(page);
  await expect(page.locator('[data-tiles]')).toContainText('monitor, 24 h');
  await expect(page.locator('[data-gave-up]')).toHaveText('1');
  const first = page.locator('[data-last-actions] .feed-row').first();
  await expect(first).toBeVisible();
  await first.click();
  await expect(page.locator('[data-event-drawer]')).toBeVisible();
});

test('a session marks the monitor\'s acts on its strip and lists its journal @mock', async ({ page }) => {
  await open(page, 'sessions');
  await expect(page.locator('[data-sessions] thead')).toContainText('monitor acts');
  await open(page, 'sessions', '&session=c%3Ases_night_pagoda');
  await expect(page.locator('[data-timeline] [data-monitor-mark]').first()).toBeVisible();
  await expect(page.locator('[data-timeline] [data-journal] .feed-row').first()).toBeVisible();
  const exp = await (await page.request.get('/v1/conductor/sessions/c%3Ases_night_pagoda/export', { headers: { Authorization: 'Bearer mock' } })).json();
  expect(exp.summary).toMatch(/^\d+ interventions? in \d+ turns?$/);
  expect(new Set(exp.injections.map((i: { source: string }) => i.source))).toEqual(new Set(['router', 'monitor']));
});

test('phone width: every screen in one column, no sideways scroll, the switch tappable @mock', async ({ page }, info) => {
  test.skip(!info.project.name.includes('phone'), 'phone project only');
  for (const s of SCREENS) {
    await open(page, s, s === 'profiles' ? '&edit=coding' : s === 'sessions' ? '&session=c%3Ases_night_pagoda' : '');
    await page.waitForTimeout(600);
    expect(await noHorizontalOverflow(page), `${s} scrolls sideways`).toBe(true);
  }
  const on = page.getByRole('radiogroup', { name: /mode of/ }).getByRole('radio', { name: 'on' });
  const box = await on.boundingBox();
  expect(box!.height).toBeGreaterThanOrEqual(44);
  await page.screenshot({ path: 'e2e-artifacts/conductor-phone.png' });
});
