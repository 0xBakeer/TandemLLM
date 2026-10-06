import { expect, type Page, type APIRequestContext } from '@playwright/test';

/** The mock's admin token, for the tests of a server with a password set (`{password: true}`). */
export const TOKEN = process.env.PW_TOKEN ?? 'mock';
export const IS_MOCK = !process.env.PW_BASE || process.env.PW_MOCK === '1';
/** The fake-engine tier (`server/app.py --fake-engine`, PW_FAKE=1): the real server with no GPU, no
 *  QWEN38_* flags and no speculation, so those readings are absent there by construction. */
export const IS_FAKE = process.env.PW_FAKE === '1';

// @mock marks what only the mock's data can satisfy. Whole scenarios that need a mock
// switch carry the tag and skip themselves on a real engine; an assertion on the mock's numbers --
// a year of history, its flags, its cache budget, an error line in its log -- sits under
// `if (IS_MOCK)` with an @mock comment, and the real tiers check the data-relative form instead.

/** Open the dashboard on `hash` (default #/usage). The name stays from the sign-in days: since AU1
 *  (2026-10-06) the dashboard asks for no password, so this only waits for the shell. */
export async function login(page: Page, hash = '#/usage', query = ''): Promise<void> {
  await page.goto(`./${query}${hash}`);
  await expect(page.locator('.shell')).toBeVisible({ timeout: 8000 });
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
      if (el.closest('.heat-scroll') || el.closest('.table-scroll') || el.closest('.console-scroll') || el.closest('.cond-tabs') || el.closest('.strip')) continue;
      const r = el.getBoundingClientRect();
      if (r.right > doc.clientWidth + 1 && r.width > 0) offenders.push(el.className || el.tagName);
    }
    return offenders.length === 0;
  });
}
