// QA VIS-79 (VIS-78): the setup wizard at #/setup against the mock, desktop and 390 px. @mock
import { expect, test, type Page } from '@playwright/test';
import { IS_MOCK, noHorizontalOverflow } from './helpers';
import { open, setupMock, setSlots } from './setup-helpers';

test.skip(!IS_MOCK, 'the setup guide specs drive the mock (POST /__mock/setup)');

const isPhone = (page: Page) => (page.viewportSize()?.width ?? 1440) <= 768;
const MODEL = 'qwen3.8-27b';

test.beforeEach(async ({ request }) => {
  await setupMock(request);
});

test('a tile shows the model and opens its overview', async ({ page }) => {
  await open(page, '#/setup');
  const tile = page.locator(`.mtile[data-model="${MODEL}"]`);
  await expect(tile).toContainText('Qwen3.8-27B');
  await expect(tile).toContainText('27B parameters');
  await expect(tile).toContainText('Recommended');
  await expect(tile.locator('.mtile-speed-n')).toHaveText('84.4');
  await tile.click();
  await expect(page).toHaveURL(new RegExp(`#/setup/${MODEL}/overview$`));
  await expect(page.locator('.setup-step')).toHaveAttribute('data-step', 'overview');
});

test('the recommended setup is preselected and fits', async ({ page }) => {
  await open(page, `#/setup/${MODEL}/weights`);
  await expect(page.locator('.wcard.is-selected')).toHaveAttribute('data-profile-card', 'exl3-4.5');
  await expect(page.locator('.membar')).toHaveAttribute('data-fits', 'true');
  await page.locator('.setup-next').click();
  for (const id of ['drafters', 'lookup', 'conductor', 'narrator', 'laya', 'shadow', 'vision', 'prompt_cache', 'state_store', 'disk_tier', 'nvme_spill']) {
    await expect(page.locator(`[data-component="${id}"] input`), id).toBeChecked();
  }
  for (const id of ['response_cache', 'proxy']) await expect(page.locator(`[data-component="${id}"] input`), id).not.toBeChecked();
});

test('Next and Previous: the URL, the rail and the page agree and nothing is lost', async ({ page }) => {
  await open(page, `#/setup/${MODEL}/weights`);
  await page.locator('[data-profile-card="exl3-4.0"]').click();
  for (let i = 0; i < 3; i++) await page.locator('.setup-next').click();
  await page.locator('.setup-prev').click();
  await expect(page).toHaveURL(new RegExp(`#/setup/${MODEL}/parallelism$`));
  await expect(page.locator('.setup-step')).toHaveAttribute('data-step', 'parallelism');
  if (!isPhone(page)) await expect(page.locator('[data-rail="parallelism"] a')).toHaveAttribute('aria-current', 'step');
  else await expect(page.locator('.rail-phone')).toContainText('Step 5 of 8');
  await page.goBack();
  await expect(page.locator('.setup-step')).toHaveAttribute('data-step', 'methods');
  await page.goto(`./#/setup/${MODEL}/weights`);
  await expect(page.locator('.wcard.is-selected')).toHaveAttribute('data-profile-card', 'exl3-4.0');
});

test('a reload restores the step and the selection from tandem.setup.v1', async ({ page }) => {
  await open(page, `#/setup/${MODEL}/weights`);
  await page.locator('[data-profile-card="exl3-4.0"]').click();
  await page.locator('.setup-next').click();
  await page.locator('[data-component="vision"]').click();
  await page.locator('.setup-next').click();
  await expect(page.locator('.setup-step')).toHaveAttribute('data-step', 'parallelism');
  const saved = await page.evaluate(() => JSON.parse(localStorage.getItem('tandem.setup.v1') ?? 'null'));
  expect(saved).toMatchObject({ v: 1, model: MODEL, step: 'parallelism', selection: { profile: 'exl3-4.0', components: { vision: false } } });
  await page.reload();
  await expect(page.locator('.setup-step')).toHaveAttribute('data-step', 'parallelism');
  await page.goto(`./#/setup/${MODEL}/components`);
  await expect(page.locator('[data-component="vision"] input')).not.toBeChecked();
});

test('a deep link to Review opens it with the default selection', async ({ page }) => {
  await open(page, `#/setup/${MODEL}/review`);
  await expect(page.locator('.setup-step')).toHaveAttribute('data-step', 'review');
  await expect(page.locator('[data-review-memory]')).toHaveClass(/is-ok/);
  // the mock box already runs these settings: the plan says nothing in serve.env changes, and the
  // weights switch from exl3-6.0
  await expect(page.locator('.rv-changes')).toContainText('Nothing in serve.env changes.');
  await expect(page.locator('.rv-cmds')).toContainText('ops/use-profile.sh exl3-4.5');
});

test('over budget blocks Finish with How to fix, and a shorter window lets it go', async ({ page }) => {
  await open(page, `#/setup/${MODEL}/weights`);
  await page.locator('[data-profile-card="exl3-6.0"]').click();
  await page.goto(`./#/setup/${MODEL}/components`);
  for (const id of ['response_cache', 'proxy']) await page.locator(`[data-component="${id}"]`).click();
  await page.goto(`./#/setup/${MODEL}/parallelism`);
  await page.locator('[data-field="parallelism.kv_pool"] input').click();
  await page.goto(`./#/setup/${MODEL}/review`);
  await expect(page.locator('.membar')).toHaveAttribute('data-fits', 'false');
  await expect(page.locator('[data-membar-say]')).toContainText(/^Over by \d+\.\d GiB\./);
  await expect(page.locator('.setup-finish')).toBeDisabled();
  await expect(page.locator('[data-finish-blocked]')).toContainText('does not fit');
  const fix = page.locator('.rv-fixes li', { hasText: 'Lower the longest conversation' });
  await expect(fix).toBeVisible();
  await fix.locator('a').click();
  await expect(page.locator('.setup-step')).toHaveAttribute('data-step', 'parallelism');
  await page.locator('[data-field="parallelism.window"] .seg-btn', { hasText: '64k' }).click();
  await expect(page.locator('.membar')).toHaveAttribute('data-fits', 'true');
  await page.goto(`./#/setup/${MODEL}/review`);
  await expect(page.locator('.setup-finish')).toBeEnabled();
});

test('a dependency is enforced with its reason', async ({ page }) => {
  await open(page, `#/setup/${MODEL}/components`);
  await page.locator('[data-component="conductor"]').click();
  await expect(page.locator('[data-cascade]')).toContainText('need the conductor, so they are off too');
  for (const id of ['narrator', 'laya', 'shadow']) {
    await expect(page.locator(`[data-component="${id}"] input`)).not.toBeChecked();
    await expect(page.locator(`[data-component="${id}"] input`)).toBeDisabled();
    await expect(page.locator(`[data-component="${id}"] [data-reason]`)).toHaveText('Needs the conductor');
  }
  await page.locator('[data-component="conductor"]').click();
  await expect(page.locator('[data-component="shadow"] input')).toBeEnabled();
});

test('the measured table sorts, switches mode and group, marks the choice', async ({ page }) => {
  await open(page, `#/setup/${MODEL}/overview`);
  const rows = page.locator('.mt-table tbody tr.mt-row:not(.is-apart)');
  await expect(rows.first()).toHaveAttribute('data-profile', 'exl3-2.5'); // greedy code, fastest first
  await page.locator('.mt-sort[data-col="code"]').click();
  await expect(rows.first()).toHaveAttribute('data-profile', 'exl3-6.0');
  await page.locator('.seg-btn[data-val="sampled"]').click();
  await expect(page.locator('.mt-table')).toHaveAttribute('data-mode', 'sampled_512');
  await page.locator('.seg-btn[data-val="long"]').click();
  await expect(page.locator('.mt-table')).toHaveAttribute('data-mode', 'long_sampled');
  await expect(page.locator('.mt-sort[data-col="long32k"]')).toBeVisible();
  await expect(page.locator('.mt-row.is-selected')).toHaveAttribute('data-profile', 'exl3-4.5');
  await expect(page.locator('.mt-apart-head')).toHaveText('older build, not comparable');
  await expect(page.locator('.mt-row.is-apart')).toHaveAttribute('data-profile', 'nvfp4');
});

test('the estimate route down: the bar keeps working from the page and says so', async ({ page, request }) => {
  await setupMock(request, { estimateDown: true });
  await open(page, `#/setup/${MODEL}/parallelism`);
  const before = await page.locator('.membar').getAttribute('data-total');
  await setSlots(page, 2);
  await expect(page.locator('[data-local-estimate]')).toHaveText('Using a local estimate: the box did not answer.');
  await expect(page.locator('.membar')).not.toHaveAttribute('data-total', before!);
});

async function finish(page: Page) {
  await open(page, `#/setup/${MODEL}/review`);
  await expect(page.locator('.setup-finish')).toBeEnabled();
  await page.locator('.setup-finish').click();
  await page.locator('.setup-confirm-go').click();
}

test('Finish: Wren is asked to unload, the steps run, the way out appears', async ({ page, request }) => {
  await setupMock(request, { assistant: 'loaded' });
  const unload = page.waitForRequest((r) => r.url().endsWith('/v1/setup/assistant/unload') && r.method() === 'POST');
  await finish(page);
  await unload;
  await expect(page).toHaveURL(new RegExp(`#/setup/${MODEL}/install$`));
  await expect(page.locator('[data-job-state="done"]')).toBeVisible({ timeout: 20_000 });
  await expect(page.locator('.inst-step[data-step="smoke"]')).toHaveClass(/is-done/);
  await expect(page.locator('.inst-actions a', { hasText: 'See it run' })).toHaveAttribute('href', '#/performance');
});

test('requests still running: the job waits for them before anything stops, and says so', async ({ page, request }) => {
  await setupMock(request, { waitIdle: true });
  await finish(page);
  const step = page.locator('.inst-step[data-step="write_config"]');
  await expect(step).toHaveClass(/is-waiting/, { timeout: 10_000 });
  await expect(step).toContainText('waiting for 2 requests to finish');
  await expect(page.locator('[data-job-state="done"]')).toBeVisible({ timeout: 20_000 });
  await expect(step).toHaveClass(/is-done/);
});

test('a failed smoke rolls back and offers the way back to Review', async ({ page, request }) => {
  await setupMock(request, { scenario: 'rollback' });
  await finish(page);
  await expect(page.locator('[data-job-state="rolled_back"]')).toBeVisible({ timeout: 20_000 });
  await expect(page.locator('.inst-step[data-step="smoke"]')).toHaveClass(/is-failed/);
  await expect(page.locator('[data-rolled-back]')).toContainText('went back to exl3-4.5');
  await page.locator('.inst-actions button', { hasText: 'Back to Review' }).click();
  await expect(page.locator('.setup-step')).toHaveAttribute('data-step', 'review');
});

test('the engine restarts during the install: the page waits and resumes the job', async ({ page, request }) => {
  await setupMock(request, { scenario: 'restart', gap_s: 5 });
  await finish(page);
  await expect(page.locator('[data-restarting]')).toBeVisible({ timeout: 15_000 });
  await expect(page.locator('[data-restarting]')).toContainText('The engine is restarting');
  await expect(page.locator('[data-job-state="done"]')).toBeVisible({ timeout: 40_000 });
  await expect(page.locator('[data-restarting]')).toHaveCount(0);
});

test('corrupt, old or stale saved progress starts over with a note', async ({ page }) => {
  await page.addInitScript(() => {
    if (!sessionStorage.getItem('seeded')) {
      localStorage.setItem('tandem.setup.v1', '{not json');
      sessionStorage.setItem('seeded', '1');
    }
  });
  await open(page, '#/setup');
  await expect(page.locator('[data-setup-notice]')).toContainText('could not read the saved progress');
  await page.goto(`./#/setup/${MODEL}/weights`);
  await expect(page.locator('.wcard').first()).toBeVisible();
  // a saved profile the registry no longer has (written after the page's own save)
  await page.evaluate(() => localStorage.setItem('tandem.setup.v1', JSON.stringify({ v: 1, model: 'qwen3.8-27b', step: 'weights', visited: [], updatedAt: 1, chat: [], selection: { model: 'qwen3.8-27b', profile: 'exl3-9.9' } })));
  await page.reload();
  await expect(page.locator('[data-setup-notice]')).toContainText('exl3-9.9');
  await expect(page.locator('.wcard.is-selected')).toHaveAttribute('data-profile-card', 'exl3-4.5');
});

test('storage that throws: the wizard works and says progress is not saved', async ({ page }) => {
  await page.addInitScript(() => {
    const boom = () => {
      throw new DOMException('denied', 'SecurityError');
    };
    Object.defineProperty(window, 'localStorage', { get: boom, configurable: true });
  });
  await open(page, `#/setup/${MODEL}/weights`);
  await expect(page.locator('[data-setup-notice]')).toContainText('progress will not be kept');
  await page.locator('[data-profile-card="exl3-4.0"]').click();
  await expect(page.locator('.wcard.is-selected')).toHaveAttribute('data-profile-card', 'exl3-4.0');
});

test('the guide starts from what the box runs now; Review warns when a client-facing setting changes', async ({ page, request }) => {
  await setupMock(request, { current: { profile: 'exl3-6.0', slots: 2, components: { vision: false } } });
  await open(page, `#/setup/${MODEL}/weights`);
  await expect(page.locator('.wcard.is-selected')).toHaveAttribute('data-profile-card', 'exl3-6.0');
  await expect(page.locator('[data-setup-notice]')).toContainText('starts from what this box runs now');
  await page.goto(`./#/setup/${MODEL}/components`);
  await expect(page.locator('[data-component="vision"] input')).not.toBeChecked();
  await page.goto(`./#/setup/${MODEL}/review`);
  await expect(page.locator('.rv-changes')).toContainText('Nothing in serve.env changes.');
  await page.goto(`./#/setup/${MODEL}/methods`);
  await page.locator('details.adv summary').click();
  const port = page.locator('[data-field="methods.port"] input');
  await port.fill('8124');
  await port.press('Tab');
  await page.goto(`./#/setup/${MODEL}/review`);
  await expect(page.locator('.rv-notes')).toContainText('PORT changes from 8000 to 8124');
});

test('setup mode: #/ goes to #/setup and the other views are off with a reason', async ({ page, request }) => {
  await setupMock(request, { setupMode: true });
  await page.goto('./#/usage');
  await expect(page).toHaveURL(/#\/setup$/);
  await expect(page.locator('[data-setup-mode]')).toBeVisible();
  // VIS-161: every view but Setup is off with the reason, the Chat tab included
  const on = page.locator('.rail a.rail-tab:not(.is-off)');
  await expect(on).toHaveCount(1);
  await expect(on).toHaveAttribute('href', '#/setup');
  const off = page.locator('.rail a.rail-tab.is-off');
  await expect(off).toHaveCount(8);
  for (const tab of await off.all()) await expect(tab).toHaveAttribute('title', 'Not available while the box is in setup mode');
  await expect(off.filter({ hasText: 'Chat' })).toHaveAttribute('aria-disabled', 'true');
  await page.evaluate(() => (location.hash = '#/system'));
  await expect(page).toHaveURL(/#\/setup$/);
});

test('private weights without a token: the notice, and Finish stays off', async ({ page, request }) => {
  await setupMock(request, { hfToken: false, privateProfile: 'exl3-3.0' });
  await open(page, `#/setup/${MODEL}/weights`);
  await page.locator('[data-profile-card="exl3-3.0"]').click();
  await expect(page.locator('[data-token-blocked]')).toContainText('no Hugging Face token');
  await page.goto(`./#/setup/${MODEL}/review`);
  await expect(page.locator('.setup-finish')).toBeDisabled();
  await expect(page.locator('[data-finish-blocked]')).toContainText('Hugging Face token');
});

test('the open-box warning on Review offers to generate a password', async ({ page }) => {
  await open(page, `#/setup/${MODEL}/review`);
  const card = page.locator('[data-security-warning]');
  await expect(card).toContainText('Anyone who can reach this port can change the setup');
  await card.locator('.sec-opt', { hasText: 'Generate a password on the box' }).click();
  await expect(card.locator('.sec-opt.is-on')).toContainText('Generate a password on the box');
  const saved = await page.evaluate(() => JSON.parse(localStorage.getItem('tandem.setup.v1')!).selection.advanced.admin_password);
  expect(saved).toBe('generate');
});

test('390 px: no sideways scroll, a compact bar, Step n of 8, 44 px targets', async ({ page }) => {
  test.skip(!isPhone(page), 'phone only');
  for (const step of ['overview', 'weights', 'components', 'parallelism', 'methods', 'review']) {
    await open(page, `#/setup/${MODEL}/${step}`);
    await expect(page.locator('.setup-step')).toHaveAttribute('data-step', step);
    expect(await noHorizontalOverflow(page), step).toBe(true);
  }
  await expect(page.locator('.rail-phone')).toContainText('Step 7 of 8');
  await expect(page.locator('.membar-more')).toBeHidden();
  await page.locator('.membar-toggle').click();
  await expect(page.locator('.membar-legend')).toBeVisible();
  for (const sel of ['.setup-prev', '.setup-finish', '.wren-fab']) {
    const box = await page.locator(sel).boundingBox();
    expect(box!.height, sel).toBeGreaterThanOrEqual(44);
  }
});

test('reduced motion: Wren is still and the steps render', async ({ browser }) => {
  const ctx = await browser.newContext({ reducedMotion: 'reduce', viewport: { width: 1440, height: 900 } });
  const page = await ctx.newPage();
  await page.goto(`./#/setup/${MODEL}/components`);
  await expect(page.locator('.comp').first()).toBeVisible();
  const running = await page.evaluate(() => document.getAnimations().filter((a) => a.playState === 'running' && (a as CSSAnimation).animationName?.startsWith('w-')).length);
  expect(running).toBe(0);
  await ctx.close();
});
