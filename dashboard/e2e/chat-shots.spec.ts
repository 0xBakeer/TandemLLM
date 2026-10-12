// Screenshots of the Chat tab (VIS-99) against the in-browser mock: the empty state, the run that
// was already on when the tab opened (the live-resume feel), the permission prompt, the panes
// (terminal, diff, preview), the Context panel's ribbon, the settings pages, and the phone's
// drawer and sheets. Desktop 1440x900 and phone 390x844, dark and light. Picked up by the `shots`
// project into screenshots/chat-*.png (PW_SHOTS_DIR overrides the folder).
import { test, type Page } from '@playwright/test';
import { login } from './helpers';

const DIR = process.env.PW_SHOTS_DIR ?? 'screenshots';
const SIZES = { desktop: { width: 1440, height: 900 }, phone: { width: 390, height: 844 } } as const;

async function open(page: Page, theme: 'dark' | 'light', hash: string): Promise<void> {
  await page.emulateMedia({ colorScheme: theme });
  await page.addInitScript((t) => {
    localStorage.setItem('qse.theme', t);
    localStorage.removeItem('qse.chat.last');
    sessionStorage.clear();
  }, theme);
  await login(page, hash, '?chat=mock');
  await page.locator('qse-chat').waitFor();
}

const shot = (page: Page, name: string, full = false) => page.screenshot({ path: `${DIR}/chat-${name}.png`, fullPage: full });

for (const theme of ['dark', 'light'] as const) {
  test(`chat empty desktop ${theme}`, async ({ page }) => {
    await page.setViewportSize(SIZES.desktop);
    await open(page, theme, '#/chat');
    await page.locator('.ch-empty-title').waitFor();
    await page.waitForTimeout(600);
    await shot(page, `empty-desktop-${theme}`);
  });

  test(`chat running desktop ${theme}`, async ({ page }) => {
    await page.setViewportSize(SIZES.desktop);
    await open(page, theme, '#/chat/c_sync');
    await page.locator('.ch-now').waitFor();
    await page.waitForTimeout(2500);
    await shot(page, `running-desktop-${theme}`);
  });
}

test('chat running with the terminal pane, desktop dark', async ({ page }) => {
  await page.setViewportSize(SIZES.desktop);
  await open(page, 'dark', '#/chat/c_sync');
  await page.locator('.ch-now').waitFor();
  // the seeded layout has the terminal open; wait for the pane and some output
  await page.locator('qse-chat-panes .xterm-rows').first().waitFor({ timeout: 10_000 });
  await page.waitForTimeout(3000);
  await shot(page, 'running-terminal-desktop-dark');
});

test('chat permission prompt, desktop dark', async ({ page }) => {
  await page.setViewportSize(SIZES.desktop);
  await open(page, 'dark', '#/chat/c_rename');
  await page.locator('.ch-perm').waitFor();
  await page.waitForTimeout(500);
  await shot(page, 'permission-desktop-dark');
});

test('chat finished turn with code and figures, desktop dark', async ({ page }) => {
  await page.setViewportSize(SIZES.desktop);
  await open(page, 'dark', '#/chat/c_explain');
  await page.locator('.ch-foot').first().waitFor();
  await page.waitForTimeout(500);
  await shot(page, 'answer-desktop-dark');
});

test('chat preview pane and diff pane, desktop dark', async ({ page }) => {
  await page.setViewportSize(SIZES.desktop);
  await open(page, 'dark', '#/chat/c_shop');
  await page.locator('.ch-call-head').first().click();
  await page.getByRole('button', { name: /Open in Preview/ }).first().click();
  await page.locator('.ch-preview-frame iframe').waitFor();
  await page.waitForTimeout(1200);
  await shot(page, 'preview-desktop-dark');
  await page.locator('.ch-dock-btn').click();
  await page.waitForTimeout(400);
  await shot(page, 'preview-dock-right-desktop-dark');
  await page.locator('.ch-pane-plus button').first().click();
  await page.getByRole('menuitem', { name: 'Diff' }).click();
  await page.locator('.ch-diff-table').waitFor();
  await page.waitForTimeout(400);
  await shot(page, 'diff-desktop-dark');
});

test('chat cold cache and compaction in the context panel, desktop dark', async ({ page }) => {
  await page.setViewportSize(SIZES.desktop);
  await open(page, 'dark', '#/chat/c_exl3');
  await page.locator('.ch-rib').waitFor();
  await page.getByRole('button', { name: 'show summary' }).click();
  await page.waitForTimeout(400);
  await shot(page, 'context-cold-desktop-dark');
});

test('chat interrupted answer with Continue, desktop dark', async ({ page }) => {
  await page.setViewportSize(SIZES.desktop);
  await open(page, 'dark', '#/chat/c_sql');
  await page.locator('.ch-interrupted').waitFor();
  await page.waitForTimeout(300);
  await shot(page, 'interrupted-desktop-dark');
});

test('chat composer popovers and slash commands, desktop dark', async ({ page }) => {
  await page.setViewportSize(SIZES.desktop);
  await open(page, 'dark', '#/chat/c_explain');
  await page.locator('.ch-pill').nth(2).click();
  await page.locator('.ch-pop-wide').waitFor();
  await shot(page, 'composer-tools-desktop-dark');
  await page.keyboard.press('Escape');
  await page.locator('body').click({ position: { x: 700, y: 200 } });
  await page.locator('#ch-input').fill('/');
  await page.locator('.ch-slash').waitFor();
  await shot(page, 'composer-slash-desktop-dark');
});

for (const section of ['tools', 'mcp', 'prompts', 'sandbox'] as const) {
  test(`chat settings ${section}, desktop dark`, async ({ page }) => {
    await page.setViewportSize(SIZES.desktop);
    await open(page, 'dark', '#/chat/c_sync');
    await page.locator('.ch-now').waitFor();
    await page.goto(`./?chat=mock#/chat/settings/${section}`);
    await page.locator('.ch-settings-main').waitFor();
    if (section === 'mcp') {
      await page.getByRole('button', { name: 'Add server' }).first().click();
      await page.locator('.ch-add-mcp').waitFor();
    }
    await page.waitForTimeout(500);
    await shot(page, `settings-${section}-desktop-dark`, true);
  });
}

test('chat reconnect scenario, desktop dark', async ({ page }) => {
  await page.setViewportSize(SIZES.desktop);
  await page.emulateMedia({ colorScheme: 'dark' });
  await page.addInitScript(() => localStorage.setItem('qse.theme', 'dark'));
  await login(page, '#/chat/c_sync', '?chat=mock&scenario=reconnect');
  await page.locator('.ch-now').waitFor();
  await page.locator('.ch-attach[data-attach="reconnecting"]').waitFor({ timeout: 15_000 });
  await page.waitForTimeout(1500);
  await shot(page, 'reconnecting-desktop-dark');
});

// ---- the phone --------------------------------------------------------------------------------

for (const theme of ['dark', 'light'] as const) {
  test(`chat running phone ${theme}`, async ({ page }) => {
    await page.setViewportSize(SIZES.phone);
    await open(page, theme, '#/chat/c_sync');
    await page.locator('.ch-now').waitFor();
    await page.waitForTimeout(2500);
    await shot(page, `running-phone-${theme}`);
  });
}

test('chat phone: empty, drawer, context sheet, panes sheet, permission, settings', async ({ page }) => {
  await page.setViewportSize(SIZES.phone);
  await open(page, 'dark', '#/chat');
  await page.locator('.ch-empty-title').waitFor();
  await page.waitForTimeout(400);
  await shot(page, 'empty-phone-dark');
  await page.locator('.ch-head-rail').click();
  await page.waitForTimeout(400);
  await shot(page, 'drawer-phone-dark');
  await page.locator('.ch-scrim').click({ position: { x: 370, y: 400 } });
  await page.goto('./?chat=mock#/chat/c_sync');
  await page.locator('.ch-now').waitFor();
  await page.getByRole('button', { name: 'Context panel', exact: true }).click();
  await page.waitForTimeout(400);
  await shot(page, 'context-phone-dark');
  // the context sheet is the full width on a phone: its close button is the way out
  await page.getByRole('button', { name: 'Close the context panel' }).click();
  await page.getByRole('button', { name: 'Panes', exact: true }).click();
  await page.locator('qse-chat-panes .xterm-rows').first().waitFor({ timeout: 10_000 });
  await page.waitForTimeout(800);
  await shot(page, 'panes-phone-dark');
  await page.locator('qse-chat-panes').getByRole('button', { name: 'Close panes' }).click();
  await page.goto('./?chat=mock#/chat/c_rename');
  await page.locator('.ch-perm').waitFor();
  await page.waitForTimeout(300);
  await shot(page, 'permission-phone-dark');
  await page.goto('./?chat=mock#/chat/settings/mcp');
  await page.locator('.ch-settings-main').waitFor();
  await page.waitForTimeout(300);
  await shot(page, 'settings-mcp-phone-dark', true);
  await page.goto('./?chat=mock#/chat/c_explain');
  await page.locator('.ch-foot').first().waitFor();
  await page.locator('#ch-input').click();
  await page.waitForTimeout(300);
  await shot(page, 'answer-phone-dark');
});
