import { expect, type Page, type APIRequestContext } from '@playwright/test';

export const TOKEN = process.env.PW_TOKEN ?? 'mock';
export const IS_MOCK = !process.env.PW_BASE || process.env.PW_MOCK === '1';

/** Sign in through the token screen and land on `hash` (default #/usage). */
export async function login(page: Page, hash = '#/usage', query = ''): Promise<void> {
  await page.goto(`./${query}${hash}`);
  const token = page.locator('#token');
  if (await token.isVisible({ timeout: 8000 }).catch(() => false)) {
    await token.fill(TOKEN);
    await page.getByRole('button', { name: 'Open the dashboard' }).click();
  }
  await expect(page.locator('.shell')).toBeVisible();
}

/** Flip the mock engine's failure switch (mock tier only). */
export async function setMode(request: APIRequestContext, mode: string, extra: Record<string, unknown> = {}): Promise<void> {
  const res = await request.post('/__mock/mode', { data: { mode, ...extra } });
  expect(res.ok()).toBeTruthy();
}

/** Let the mock engine finish one request now: a [req] line and a ledger row. */
export async function finishRequest(request: APIRequestContext): Promise<string> {
  const res = await request.post('/__mock/request');
  const body = await res.json();
  return body.request_id as string;
}

export async function injectLogLines(request: APIRequestContext, count: number, level = 'info', msg?: string): Promise<void> {
  await request.post('/__mock/log', { data: { count, level, msg } });
}

/** True when nothing but the heatmap scroller scrolls horizontally. */
export async function noHorizontalOverflow(page: Page): Promise<boolean> {
  return page.evaluate(() => {
    const doc = document.documentElement;
    if (doc.scrollWidth > doc.clientWidth + 1) return false;
    const offenders: string[] = [];
    for (const el of Array.from(document.querySelectorAll<HTMLElement>('.content *'))) {
      if (el.closest('.heat-scroll') || el.closest('.table-scroll') || el.closest('.console-scroll')) continue;
      const r = el.getBoundingClientRect();
      if (r.right > doc.clientWidth + 1 && r.width > 0) offenders.push(el.className || el.tagName);
    }
    return offenders.length === 0;
  });
}
