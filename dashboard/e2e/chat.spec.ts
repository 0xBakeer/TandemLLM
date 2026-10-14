// The Chat tab's phases 2 to 4 against the in-browser mock (VIS-106, VIS-108, VIS-110, SRV-364):
// Chromium and WebKit on a desktop and an iPhone 14, the same projects as the rest of the
// dashboard. Spark only, never on a laptop:
//   npx playwright test e2e/chat.spec.ts --project=chromium-desktop --project=webkit-desktop --project=webkit-phone
// The live tier (chat-live3.spec.ts) runs the same flows on a real chatd, sandbox and engine.
import { expect, test, type Page } from '@playwright/test';
import { login } from './helpers';

async function open(page: Page, hash: string): Promise<void> {
  await page.addInitScript(() => {
    localStorage.removeItem('qse.chat.last');
    sessionStorage.clear();
  });
  await login(page, hash, '?chat=mock');
  await page.locator('qse-chat').waitFor();
}

const phone = (page: Page) => (page.viewportSize()?.width ?? 1440) < 768;

test('the compaction divider sits in the thread and opens to the summary', async ({ page }) => {
  await open(page, '#/chat/c_exl3');
  const d = page.locator('.ch-divider').first();
  await expect(d).toContainText('Earlier messages summarized: 98.1K to 31.4K tokens');
  await d.getByRole('button').click();
  await expect(d.locator('.ch-cp-summary')).toContainText('EXL3 4.5 and 6.0');
  await expect(d.getByRole('button')).toHaveAttribute('aria-expanded', 'true');
});

test('plan mode: the pill, Approve under the answer, the todo list in the plan pane', async ({ page }) => {
  await open(page, '#/chat/c_explain');
  await page.locator('.ch-foot').first().waitFor();
  await expect(page.locator('.ch-planbar')).toHaveCount(0);
  const pill = page.locator('.ch-pill', { hasText: 'plan' });
  await pill.click();
  await expect(pill).toHaveAttribute('aria-pressed', 'true');
  await expect(page.locator('.ch-planbar')).toBeVisible();
  await page.locator('.ch-planbar').getByRole('button', { name: 'Edit plan' }).click();
  await page.locator('#ch-plan-edit').fill('Read the list\nPick dinners\nShop once');
  await page.locator('.ch-planbar').getByRole('button', { name: 'Approve' }).click();
  await expect(pill).toHaveAttribute('aria-pressed', 'false');
  await expect(page.locator('.ch-user .ch-text').last()).toHaveText('Carry out the approved plan.');
  if (phone(page) || !(await page.locator('.ch-panes').count())) await page.getByRole('button', { name: 'Panes', exact: true }).click();
  await page.locator('qse-chat-panes').getByRole('button', { name: 'Open a pane' }).click();
  await page.locator('.ch-pane-menu [role="menuitem"]', { hasText: 'Plan' }).click();
  await expect(page.locator('.ch-todos .ch-todo')).toHaveText(['Read the list, pending', 'Pick dinners, pending', 'Shop once, pending']);
});

test('rewind to an earlier message: the thread ends before it and the composer holds it', async ({ page }) => {
  await open(page, '#/chat/c_exl3');
  await expect(page.locator('.ch-user')).toHaveCount(2);
  page.once('dialog', (d) => void d.accept());
  await page.locator('.ch-user').nth(1).getByRole('button', { name: 'Rewind to here' }).click();
  await expect(page.locator('.ch-user')).toHaveCount(1);
  await expect(page.locator('#ch-input')).toHaveValue('Which one for the weekly code check?');
});

test('settings: an Open WebUI tools file imports, a secret valve never shows its value', async ({ page }) => {
  await open(page, '#/chat/settings/tools');
  await page.locator('.ch-file-btn', { hasText: 'Import a tools file' }).locator('input[type=file]').setInputFiles({ name: 'weather.py', mimeType: 'text/x-python', buffer: Buffer.from('class Tools:\n    pass\n') });
  const kit = page.locator('.ch-kit').first();
  await expect(kit).toContainText('weather.py');
  await expect(page.locator('.ch-tool', { hasText: 'owui__weather__run' })).toContainText('Open WebUI tool');
  const key = kit.locator('input[type=password]');
  await expect(key).toHaveValue('');
  await key.fill('k-secret-123');
  await kit.getByRole('button', { name: 'Save valves' }).click();
  await expect(key).toHaveValue('');
  await expect(key).toHaveAttribute('placeholder', 'stored; type to replace');
  expect(await page.content()).not.toContain('k-secret-123');
});

test('settings: a skill zip uploads and can be removed again', async ({ page }) => {
  await open(page, '#/chat/settings/packages');
  await page.locator('.ch-file-btn', { hasText: 'Upload a skill' }).locator('input[type=file]').setInputFiles({ name: 'tea-timer.zip', mimeType: 'application/zip', buffer: Buffer.from('PK') });
  const row = page.locator('.ch-pkg', { hasText: 'tea-timer' });
  await expect(row.locator('.tag')).toHaveText('yours');
  page.once('dialog', (d) => void d.accept());
  await row.getByRole('button', { name: 'Remove tea-timer' }).click();
  await expect(row).toHaveCount(0);
});

test('the terminal pane is xterm.js; a phone gets the key row', async ({ page }) => {
  await open(page, '#/chat/c_sync');
  await page.locator('.ch-now').waitFor();
  if (phone(page)) await page.getByRole('button', { name: 'Panes', exact: true }).click();
  await expect(page.locator('qse-chat-panes .xterm-rows')).toContainText('sandbox', { timeout: 15_000 });
  await expect(page.locator('.ch-term-keys')).toHaveCount(phone(page) ? 1 : 0);
  if (phone(page)) await expect(page.locator('.ch-term-keys button')).toHaveCount(9);
});

test('a pinned skill shows in the Context panel with its tokens', async ({ page }) => {
  await open(page, '#/chat/c_explain');
  await page.locator('.ch-pill[title="tools, skills and MCP servers"]').click();
  const pin = page.locator('.ch-pop-check', { hasText: 'web design answer' }).locator('input');
  await pin.check();
  await page.keyboard.press('Escape');
  if (phone(page) || !(await page.locator('qse-chat-context').isVisible())) await page.getByRole('button', { name: 'Context panel', exact: true }).click();
  await expect(page.locator('[data-sec="skills"] [data-pin="web-design-answer"]')).toContainText('web-design-answer');
});

test('This device: the notification switches', async ({ page }) => {
  await open(page, '#/chat/settings/device');
  await expect(page.locator('.ch-settings-h')).toHaveText('This device');
  await expect(page.locator('.ch-fieldset .ch-check')).toHaveCount(3);
  await expect(page.locator('[data-notify]')).toBeVisible();
});

// ---- web search (VIS-157, QA VIS-158) ----------------------------------------------------------

test('web search settings: SearXNG fields, the checks, Test, Save, the stored value masked', async ({ page }) => {
  await open(page, '#/chat/settings/search');
  const form = page.locator('[data-ws-form]');
  await expect(form.locator('[data-ws-searxng]')).toHaveCount(0);
  await form.locator('[data-ws="provider"]').selectOption('searxng');
  await expect(form.locator('[data-ws-searxng]')).toBeVisible();
  await expect(form.locator('.ch-ws-err')).toContainText("SearXNG needs the instance's URL");
  await expect(form.locator('[data-ws-save]')).toBeDisabled();
  await form.locator('[data-ws="searxng_url"]').fill('https://search.example.org/');
  await form.locator('[data-ws="searxng_auth_value"]').fill('Bearer k1');
  await expect(form.locator('.ch-ws-err')).toContainText('name the header');
  await form.locator('[data-ws="searxng_auth_header"]').fill('Authorization');
  await expect(form.locator('.ch-ws-err')).toHaveCount(0);
  await form.locator('[data-ws-test]').click();
  await expect(form.locator('[data-ws-result]')).toContainText('SearXNG answered 3 results');
  await expect(form.locator('.ch-ws-sample li')).toHaveCount(3);
  await form.locator('[data-ws-save]').click();
  await expect(form.locator('[data-ws="searxng_auth_value"]')).toHaveAttribute('placeholder', 'stored; type to replace');
  await expect(form.locator('[data-ws="searxng_auth_value"]')).toHaveValue('');
  await expect(form.locator('[data-ws="searxng_url"]')).toHaveValue('https://search.example.org');
  await form.locator('[data-ws="searxng_url"]').fill('http://json-off.example:8888');
  await form.locator('[data-ws-test]').click();
  await expect(form.locator('[data-ws-result]')).toContainText('search.formats');
  if (phone(page)) {
    const wide = await page.evaluate(() => document.scrollingElement!.scrollWidth - window.innerWidth);
    expect(wide).toBeLessThanOrEqual(0);
    const cols = await form.locator('.ch-ws-grid').first().evaluate((el) => getComputedStyle(el).gridTemplateColumns.split(' ').length);
    expect(cols).toBe(1);
  }
  await page.screenshot({ path: test.info().outputPath('web-search-settings.png'), fullPage: true });
});

test('a web search card lists its results as links; the tools menu names the provider', async ({ page }) => {
  await open(page, '#/chat');
  await page.locator('#ch-input').fill('Search the latest on speculative decoding');
  await page.locator('.ch-composer .ch-send').click();
  const card = page.locator('.ch-call[data-tool="web_search"]');
  await expect(card.locator('.ch-call-word')).toHaveText('3 results', { timeout: 30_000 });
  await expect(card.locator('.ch-call-args')).toHaveText('speculative decoding tokens per round');
  const links = card.locator('.ch-search a.ch-search-title');
  await expect(links).toHaveCount(3);
  await expect(links.first()).toHaveAttribute('target', '_blank');
  await expect(links.first()).toHaveAttribute('rel', 'noopener noreferrer');
  await expect(card.locator('.ch-search-host').first()).toHaveText('example.org');
  await page.screenshot({ path: test.info().outputPath('web-search-card.png') });
  await page.locator('.ch-pill', { hasText: /tools?/ }).first().click();
  await expect(page.locator('[data-pop-tool="web_search"] [data-search-provider]')).toContainText('with DuckDuckGo');
  if (phone(page)) expect(await page.evaluate(() => document.scrollingElement!.scrollWidth - window.innerWidth)).toBeLessThanOrEqual(0);
});
