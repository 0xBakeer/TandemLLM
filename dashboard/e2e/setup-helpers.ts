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

/** VIS-199: the parallel-chats stepper's field: type a count (the input event sets it at once). */
export async function setSlots(page: Page, n: number | string): Promise<void> {
  await page.locator('#pf-slots-in').fill(String(n));
}

/** VIS-199: the window stepper's field (SRV-707: one window for every chat): type tokens, as a person would. */
export async function setWindow(page: Page, _key: 'window', v: number | string): Promise<void> {
  await page.locator('#pf-win-in').fill(String(v));
}
