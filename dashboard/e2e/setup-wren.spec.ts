// QA VIS-81 (VIS-80): Wren, the setup guide's chat, against the mock's scripted model. @mock
// QA VIS-92 (VIS-91): the live loop of 0.7.1 (refused calls asked again four times, a notice per
// call, no answer, the phone's sheet dropping per call), with the mock playing the 2B's calls.
import { expect, test, type Page } from '@playwright/test';
import { IS_MOCK } from './helpers';
import { open, setupMock } from './setup-helpers';

test.skip(!IS_MOCK, 'the Wren specs drive the mock (POST /__mock/setup)');

const MODEL = 'qwen3.8-27b';
const isPhone = (page: Page) => (page.viewportSize()?.width ?? 1440) <= 1100;

/** the panel: docked on a desktop, the bottom sheet on a phone (opened from the floating wren) */
async function panel(page: Page) {
  if (isPhone(page) && !(await page.locator('.wren-panel.is-open').count())) await page.locator('.wren-fab').click();
  const p = page.locator('.wren-panel');
  await expect(p).toBeVisible();
  return p;
}

async function ask(page: Page, text: string) {
  const p = await panel(page);
  await p.locator('.wren-input').fill(text);
  await p.locator('.wren-input').press('Enter');
}

const figure = (page: Page) => page.locator('qse-wren-figure.wren-main');

test.beforeEach(async ({ request }) => {
  await setupMock(request, { assistant: 'loaded' });
});

test('Wake Wren: the loading ring, then ready', async ({ page, request }) => {
  await setupMock(request, { assistant: 'not_loaded', loadMs: 2500 });
  await open(page, `#/setup/${MODEL}/overview`);
  const p = await panel(page);
  await expect(p).toHaveAttribute('data-phase', 'asleep');
  await p.locator('.wren-wake').click();
  await expect(p).toHaveAttribute('data-phase', 'loading');
  await expect(p.locator('qse-wren-figure.wren-hero')).toHaveAttribute('state', 'loading');
  await expect(p.locator('qse-wren-figure.wren-hero .w-ring-fill')).toBeVisible();
  await expect(p).toHaveAttribute('data-phase', 'ready', { timeout: 10_000 });
  await expect(p.locator('[data-wren-status]')).toHaveText('Ready');
});

test('a question: listening, thinking, talking, idle, with the sources', async ({ page }) => {
  await open(page, `#/setup/${MODEL}/components`);
  const p = await panel(page);
  await p.locator('.wren-input').focus();
  await p.locator('.wren-input').pressSequentially('what is the narrator?', { delay: 10 });
  await expect(figure(page)).toHaveAttribute('state', 'listening');
  await p.locator('.wren-input').press('Enter');
  await expect(figure(page)).toHaveAttribute('state', 'thinking');
  await expect(figure(page)).toHaveAttribute('state', 'talking', { timeout: 5000 });
  await expect(figure(page)).toHaveAttribute('state', 'idle', { timeout: 10_000 });
  await expect(p.locator('.wren-msg.is-assistant').last()).toContainText('The narrator is a small model');
  await expect(p.locator('.wren-sources')).toContainText('The server: roles');
});

test('Wren ticks a box: the field pulses, the bar drops, Undo puts it back', async ({ page }) => {
  await open(page, `#/setup/${MODEL}/components`);
  const before = Number(await page.locator('.membar').getAttribute('data-total'));
  await ask(page, 'turn off the vision tower');
  await expect(page.locator('[data-component="vision"] input')).not.toBeChecked({ timeout: 10_000 });
  await expect(page.locator('[data-field="components.vision"][data-wren-acting]')).toBeVisible();
  const after = Number(await page.locator('.membar').getAttribute('data-total'));
  expect(before - after).toBeCloseTo(0.86, 1);
  const p = await panel(page);
  const chip = p.locator('[data-tool-chip]').first();
  await expect(chip).toContainText('Wren turned off Vision');
  await chip.locator('button', { hasText: 'Undo' }).click();
  await expect(page.locator('[data-component="vision"] input')).toBeChecked();
  await expect(chip).toHaveClass(/is-undone/);
});

test('Wren fills a form field: slots 2', async ({ page }) => {
  await open(page, `#/setup/${MODEL}/parallelism`);
  await ask(page, 'use 2 slots please');
  await expect(page.locator('[data-field="parallelism.slots"] .seg-btn.is-on')).toHaveText(/^2/, { timeout: 10_000 });
});

test('an unknown component is refused: nothing changes, one notice says so', async ({ page }) => {
  await open(page, `#/setup/${MODEL}/components`);
  const before = await page.locator('.membar').getAttribute('data-total');
  await ask(page, 'turn on flux');
  const p = await panel(page);
  await expect(p.locator('[data-tool-rejected]')).toHaveText('Wren could not make that change, so nothing on the page changed.', { timeout: 10_000 });
  await expect(figure(page)).toHaveAttribute('state', 'idle', { timeout: 10_000 });
  await expect(p.locator('[data-tool-rejected]')).toHaveCount(1);
  await expect(page.locator('.membar')).toHaveAttribute('data-total', before!);
  // the error went back to the model as the tool message
  const chat = await page.evaluate(() => JSON.parse(localStorage.getItem('tandem.setup.v1')!).chat);
  const tool = chat.find((m: { role: string }) => m.role === 'tool');
  expect(JSON.parse(tool.content)).toMatchObject({ ok: false, error: expect.stringContaining('This setup has no tool called flux') });
});

test('Wren cannot finish: the call is refused and no install starts', async ({ page }) => {
  await open(page, `#/setup/${MODEL}/review`);
  let applied = false;
  page.on('request', (r) => {
    if (r.url().endsWith('/v1/setup/apply')) applied = true;
  });
  await ask(page, 'finish the setup for me');
  const p = await panel(page);
  await expect(p.locator('[data-tool-rejected]')).toBeVisible({ timeout: 10_000 });
  await expect(page.locator('.setup-step')).toHaveAttribute('data-step', 'review');
  expect(applied).toBe(false);
});

test('a model that keeps calling tools: the fourth round has the tools off and ends in words', async ({ page }) => {
  await open(page, `#/setup/${MODEL}/overview`);
  const bodies: { tools?: boolean }[] = [];
  page.on('request', (r) => {
    if (r.url().endsWith('/v1/setup/assistant/chat')) bodies.push(JSON.parse(r.postData() ?? '{}'));
  });
  await ask(page, 'loop');
  const p = await panel(page);
  await expect(p.locator('.wren-msg.is-assistant').last()).toContainText('You are on the Tools step now.', { timeout: 20_000 });
  await expect(p.locator('.wren-note')).toHaveCount(0);
  const chat = await page.evaluate(() => JSON.parse(localStorage.getItem('tandem.setup.v1')!).chat);
  expect(chat.filter((m: { role: string }) => m.role === 'tool')).toHaveLength(3);
  expect(bodies.map((b) => b.tools !== false)).toEqual([true, true, true, false]);
});

/** counts every time the sheet dropped to a peek or closed, however short */
async function watchSheet(page: Page) {
  await page.evaluate(() => {
    const el = document.querySelector('.wren-panel')!;
    const w = window as unknown as { __sheet: { peek: number; closed: number } };
    w.__sheet = { peek: 0, closed: 0 };
    new MutationObserver(() => {
      if (el.classList.contains('is-peek')) w.__sheet.peek++;
      if (!el.classList.contains('is-open')) w.__sheet.closed++;
    }).observe(el, { attributes: true, attributeFilter: ['class'] });
  });
}
const sheetMoves = (page: Page) => page.evaluate(() => (window as unknown as { __sheet: { peek: number; closed: number } }).__sheet);

test('VIS-91: a question the 2B answers with calls the page refuses gets its answer, no notice and no round limit', async ({ page }) => {
  await open(page, `#/setup/${MODEL}/overview`);
  const bodies: { tools?: boolean }[] = [];
  page.on('request', (r) => {
    if (r.url().endsWith('/v1/setup/assistant/chat')) bodies.push(JSON.parse(r.postData() ?? '{}'));
  });
  const p = await panel(page);
  if (isPhone(page)) await watchSheet(page);
  await ask(page, 'Which one is for coding tasks?');
  await expect(p.locator('.wren-msg.is-assistant').last()).toContainText('124.17 tokens a second', { timeout: 20_000 });
  await expect(figure(page)).toHaveAttribute('state', 'idle', { timeout: 10_000 });
  await expect(p.locator('[data-tool-rejected]')).toHaveCount(0);
  await expect(p.locator('.wren-note')).toHaveCount(0);
  expect(bodies.length).toBeLessThanOrEqual(3);
  expect(bodies[bodies.length - 1].tools).toBe(false);
  if (isPhone(page)) {
    expect(await sheetMoves(page)).toEqual({ peek: 0, closed: 0 });
    await expect(page.locator('.wren-panel.is-open')).toHaveCount(1);
  }
});

test('VIS-91: 390 px, a change keeps the sheet open with a toast; Show me closes it at the field', async ({ page }) => {
  test.skip(!isPhone(page), 'phone only');
  await open(page, `#/setup/${MODEL}/components`);
  const p = await panel(page);
  await watchSheet(page);
  await ask(page, 'turn off vision');
  const toast = p.locator('[data-wren-toast]');
  await expect(toast).toContainText('Wren turned off Vision', { timeout: 10_000 });
  await expect(figure(page)).toHaveAttribute('state', 'idle', { timeout: 10_000 });
  expect(await sheetMoves(page)).toEqual({ peek: 0, closed: 0 });
  await expect(page.locator('.wren-panel.is-open')).toHaveCount(1);
  await toast.locator('button', { hasText: 'Show me' }).click();
  await expect(page.locator('.wren-panel.is-open')).toHaveCount(0);
  const field = page.locator('[data-field="components.vision"]');
  await expect(field).toHaveAttribute('data-wren-acting', '');
  await expect(field).toBeInViewport();
  await expect(page.locator('[data-component="vision"] input')).not.toBeChecked();
});

test('VIS-91: desktop, a change has no toast and the field pulses beside the docked panel', async ({ page }) => {
  test.skip(isPhone(page), 'desktop only');
  await open(page, `#/setup/${MODEL}/components`);
  await ask(page, 'turn off vision');
  await expect(page.locator('[data-field="components.vision"][data-wren-acting]')).toBeVisible({ timeout: 10_000 });
  const p = await panel(page);
  await expect(p.locator('[data-tool-chip]').first()).toContainText('Wren turned off Vision');
  await expect(p.locator('[data-wren-toast]')).toHaveCount(0);
});

test('a load refused for memory: the reason, and the wizard still works', async ({ page, request }) => {
  await setupMock(request, { assistant: 'not_loaded', refuse: true });
  await open(page, `#/setup/${MODEL}/weights`);
  const p = await panel(page);
  await p.locator('.wren-wake').click();
  await expect(p.locator('[data-wren-refused]')).toContainText('Not enough memory beside the engine');
  await expect(p.locator('qse-wren-figure.wren-hero')).toHaveAttribute('state', 'error');
  if (isPhone(page)) await p.locator('.wren-close').click();
  await page.locator('[data-profile-card="exl3-4.0"]').click();
  await expect(page.locator('.wcard.is-selected')).toHaveAttribute('data-profile-card', 'exl3-4.0');
});

test('Stop keeps the partial answer, marked', async ({ page }) => {
  await open(page, `#/setup/${MODEL}/overview`);
  await ask(page, 'tell me everything, the long version');
  const p = await panel(page);
  await expect(p.locator('.wren-msg.is-streaming')).toBeVisible({ timeout: 5000 });
  await p.locator('.wren-stop').click();
  await expect(p.locator('.wren-msg.is-assistant').last()).toContainText('(stopped)');
  await expect(figure(page)).toHaveAttribute('state', 'idle');
});

test('a reload keeps the chat', async ({ page }) => {
  await open(page, `#/setup/${MODEL}/components`);
  await ask(page, 'what is the narrator?');
  const p = await panel(page);
  await expect(figure(page)).toHaveAttribute('state', 'idle', { timeout: 10_000 });
  await expect(p.locator('.wren-msg.is-assistant:not(.is-streaming)')).toHaveCount(1);
  await page.reload();
  const q = await panel(page);
  await expect(q.locator('.wren-msg.is-user')).toHaveText('what is the narrator?');
  await expect(q.locator('.wren-msg.is-assistant')).toContainText('The narrator is a small model');
});

test('Finish puts Wren to sleep and the composer explains why', async ({ page }) => {
  await open(page, `#/setup/${MODEL}/review`);
  await page.locator('.setup-finish').click();
  await page.locator('.setup-confirm-go').click();
  const p = await panel(page);
  await expect(p.locator('[data-wren-finished]')).toContainText('Wren is asleep while the setup runs.');
  await expect(p.locator('.wren-input')).toBeDisabled();
  await expect(p.locator('qse-wren-figure.wren-hero')).toHaveAttribute('state', 'sleeping');
});

test('a stream error: the error pose, plain words and Retry', async ({ page }) => {
  await open(page, `#/setup/${MODEL}/overview`);
  await ask(page, 'cause an error');
  const p = await panel(page);
  await expect(p.locator('[data-wren-error]')).toContainText('out of memory while decoding', { timeout: 10_000 });
  await expect(figure(page)).toHaveAttribute('state', 'error');
  const resend = page.waitForRequest((r) => r.url().endsWith('/v1/setup/assistant/chat'));
  await p.locator('[data-wren-error] button', { hasText: 'Try again' }).click();
  await resend;
});

test('another tab answering: a busy note, no error pose', async ({ page }) => {
  await open(page, `#/setup/${MODEL}/overview`);
  await ask(page, 'other tab');
  const p = await panel(page);
  await expect(p.locator('.wren-note', { hasText: 'answering in another tab' })).toBeVisible({ timeout: 5000 });
  await expect(figure(page)).toHaveAttribute('state', 'idle');
});

test('390 px: the sheet opens from the figure, keeps focus, Esc closes, the page does not scroll', async ({ page }) => {
  test.skip(!isPhone(page), 'phone only');
  await open(page, `#/setup/${MODEL}/components`);
  await page.locator('.wren-fab').click();
  await expect(page.locator('.wren-panel.is-open')).toBeVisible();
  expect(await page.evaluate(() => getComputedStyle(document.documentElement).overflow)).toBe('hidden');
  for (let i = 0; i < 8; i++) await page.keyboard.press('Tab');
  expect(await page.evaluate(() => !!document.activeElement?.closest('.wren-panel'))).toBe(true);
  await page.keyboard.press('Escape');
  await expect(page.locator('.wren-panel.is-open')).toHaveCount(0);
  expect(await page.evaluate(() => getComputedStyle(document.documentElement).overflow)).not.toBe('hidden');
  // the sheet stays up while Wren changes a field (VIS-91: it dropped to a peek per call)
  await ask(page, 'turn off vision');
  await expect(page.locator('[data-wren-toast]')).toBeVisible({ timeout: 10_000 });
  await expect(page.locator('.wren-panel.is-peek')).toHaveCount(0);
});
