// The setup guide's e2e helpers (VIS-79, VIS-81, VIS-83): the mock's switch and opening a page.
import { expect, type APIRequestContext, type Page } from '@playwright/test';

/** POST /__mock/setup: the mock back to the start, then `body`'s switches (mock/setup.ts). */
export async function setupMock(request: APIRequestContext, body: Record<string, unknown> = {}): Promise<void> {
  const r = await request.post('/__mock/setup', { data: { reset: true, ...body } });
  expect(r.ok()).toBeTruthy();
}

export async function open(page: Page, hash: string): Promise<void> {
  await page.goto(`./${hash}`);
  await expect(page.locator('qse-setup, qse-system').first()).toBeVisible();
}
