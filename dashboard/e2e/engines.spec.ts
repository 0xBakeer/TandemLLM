// P2, P3: the vLLM and SGLang backends' engines. Performance shows one card per backend from the
// qse_backend_* families of /metrics (it needs two scrapes, 5 s apart, for the window figures);
// System shows the same cards from the backend rows; a backend's own page in Backends shows its
// engine. No horizontal scroll at 390 px.
import { expect, test } from '@playwright/test';
import { IS_MOCK, login, setMode } from './helpers';

test.describe('backend engines', () => {
  test.beforeEach(async ({ request }) => {
    test.skip(!IS_MOCK, 'the mock serves a vLLM and an SGLang backend');
    await setMode(request, 'ok');
    await request.post('/__mock/backends', { data: { reset: true } });
    await request.post('/__mock/live', { data: { simulator: false } });
  });
  // no requests of the mock's own engine while these pages stay open (the window figures need two
  // scrapes): a later spec that counts the recent rows starts from its own scenario
  test.afterEach(async ({ request }) => {
    if (IS_MOCK) await request.post('/__mock/live', { data: { scenario: 'clear', simulator: true } });
  });

  test('Performance: a vLLM and an SGLang card with their figures', async ({ page }) => {
    await login(page, '#/performance');
    const v = page.locator('#backend-engines [data-engine="vllm1"]');
    const s = page.locator('#backend-engines [data-engine="sglang1"]');
    await expect(v).toContainText('vLLM 0.30.0');
    await expect(s).toContainText('SGLang 0.5.21');
    await expect(v.locator('[data-engine-kv]')).toContainText('%');
    // the window figures arrive with the second scrape
    await expect(v).toContainText('prefix-cache hits', { timeout: 15_000 });
    await expect(v.locator('[data-engine-steps]')).not.toContainText('none', { timeout: 15_000 });
    await expect(s).toContainText('not reported by this backend');
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    expect(overflow).toBeLessThanOrEqual(0);
  });

  test('System and the backend page show the engine block', async ({ page }) => {
    await login(page, '#/system');
    await expect(page.locator('#backend-engines [data-engine="vllm1"]')).toContainText('prefix-cache hits');
    await expect(page.locator('#backend-engines [data-engine="vllm1"] [data-engine-steps]')).toContainText('3.20');
    await expect(page.locator('#backend-engines [data-engine="sglang1"]')).toContainText('SGLang 0.5.21');
    await login(page, '#/backends?backend=vllm1');
    await expect(page.locator('#backend-engine')).toContainText('--enable-prompt-tokens-details');
    await login(page, '#/backends');
    await expect(page.locator('[data-backend="sglang1"] [data-engine-line]')).toContainText('KV 40 %');
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    expect(overflow).toBeLessThanOrEqual(0);
  });
});
