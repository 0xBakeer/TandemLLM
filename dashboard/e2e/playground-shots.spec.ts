// Screenshots of the Playground (VIS-22) — desktop and phone, dark and light, a finished turn;
// plus a tool call with its result box open, and the phone's setup sheet. Picked up by the
// `shots` project (testMatch /shots\.spec\.ts/) into dashboard/screenshots/playground-*.png.
import { test, type Page } from '@playwright/test';
import { login } from './helpers';

const SIZES = { desktop: { width: 1440, height: 900 }, phone: { width: 390, height: 844 } } as const;

async function bench(page: Page, theme: 'dark' | 'light'): Promise<void> {
  await page.emulateMedia({ colorScheme: theme });
  await page.addInitScript((t) => {
    localStorage.setItem('qse.theme', t);
    localStorage.removeItem('qse.playground.draft');
  }, theme);
  await login(page, '#/playground');
  await page.locator('qse-playground').waitFor();
}

async function turn(page: Page, text: string): Promise<void> {
  await page.locator('#pg-composer').fill(text);
  await page.locator('.pg-send').click();
  await page.locator('.pg-msg.role-assistant').last().locator('.pg-msg-foot', { hasText: /tok\/s|aborted|error/ }).waitFor({ timeout: 30_000 });
  await page.waitForTimeout(300);
}

for (const theme of ['dark', 'light'] as const) {
  for (const [size, viewport] of Object.entries(SIZES)) {
    test(`playground ${size} ${theme}`, async ({ page }) => {
      await page.setViewportSize(viewport);
      await bench(page, theme);
      await page.locator('#pg-system').fill('You are the engine’s own assistant. Be exact and brief.');
      await page.locator('#pg-param-temperature').fill('0.6');
      await turn(page, 'How does tokens per block relate to decode speed? One code example.');
      // the bench is viewport-sized (the transcript scrolls inside), so the viewport is the shot;
      // the phone keeps a full-page capture too, like the other views
      await page.screenshot({ path: `screenshots/playground-${size}-${theme}.png`, fullPage: size === 'phone' });
      if (size === 'phone') await page.screenshot({ path: `screenshots/playground-${size}-${theme}-fold.png`, fullPage: false });
    });
  }
}

for (const [size, viewport] of Object.entries(SIZES)) {
  test(`playground tools ${size} dark`, async ({ page }) => {
    await page.setViewportSize(viewport);
    await bench(page, 'dark');
    if (size === 'phone') await page.locator('.pg-setup-toggle').click();
    await page.locator('.pg-section[data-section="tools"] .pg-section-head').click();
    await page.getByRole('button', { name: 'get_weather(city, unit?)' }).click();
    if (size === 'phone') await page.locator('.pg-setup-close').click();
    await turn(page, 'What is the weather in Berlin right now?');
    await page.screenshot({ path: `screenshots/playground-tools-${size}-dark.png`, fullPage: false });
  });
}

test('playground setup sheet phone dark', async ({ page }) => {
  await page.setViewportSize(SIZES.phone);
  await bench(page, 'dark');
  await page.locator('#pg-param-temperature').fill('0.6');
  await page.locator('.pg-setup-toggle').click();
  await page.waitForTimeout(400);
  await page.screenshot({ path: 'screenshots/playground-setup-phone-dark.png', fullPage: false });
});
