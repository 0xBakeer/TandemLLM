// OPS-177: the Setup Guide as a phone user sees it on a scratch install (tools/release_gate_setup.sh,
// GATE_URL, GATE_SELECTION, GATE_SHOTS). The install is started through the API and followed by the
// page as the page that started it would (the job id in the guide's saved state): while the engine
// restarts the restart card shows (VIS-209); when it answers again the finished screen with Start
// chatting shows within 10 s (VIS-211). Screenshots of both go to GATE_SHOTS.
import * as fs from 'node:fs';
import { expect, test } from '@playwright/test';

const URL = process.env.GATE_URL ?? 'http://127.0.0.1:8000';
const SHOTS = process.env.GATE_SHOTS ?? 'gate-shots';

test('an install: the restart card while the engine restarts, then the finished screen', async ({ page, request }) => {
  fs.mkdirSync(SHOTS, { recursive: true });
  const sel = JSON.parse(process.env.GATE_SELECTION ?? '{}');
  await page.goto(`${URL}/dashboard/#/setup`);
  await expect(page.locator('qse-setup')).toBeVisible();
  // the guide's own seed for the model: a full selection object for the saved state
  const cur = await (await request.get(`${URL}/v1/setup/models/${sel.model}/current`)).json();
  const r = await request.post(`${URL}/v1/setup/apply`, { data: { selection: sel } });
  expect(r.status(), await r.text()).toBe(202);
  const job = (await r.json()).job;
  fs.writeFileSync(`${SHOTS}/job-id.txt`, job.id);
  await page.evaluate(
    ([model, selection, id]) =>
      localStorage.setItem('tandem.setup.v1', JSON.stringify({ v: 1, model, step: 'install', selection, visited: ['models', 'install'], chat: [], updatedAt: Date.now(), job: id })),
    [sel.model, { ...cur.selection, ...sel }, job.id],
  );
  await page.goto(`${URL}/dashboard/#/setup/${sel.model}/install`);
  // the restart: one card, no raw errors
  const card = page.locator('[data-restart-card]');
  await expect(card).toBeVisible({ timeout: 120_000 });
  await expect(card).toContainText('Restarting the engine with');
  await page.waitForTimeout(3000);
  await page.screenshot({ path: `${SHOTS}/restart-card-390.png`, fullPage: true });
  await expect(page.locator('text=Could not load')).toHaveCount(0);
  await expect(page.locator('[data-update]')).toHaveCount(0);
  // the engine answers again: the finished screen within 10 s
  let up = 0;
  for (let i = 0; i < 300 && !up; i++) {
    try {
      const h = await request.get(`${URL}/health`, { timeout: 3000 });
      const b = h.ok() ? await h.json() : null;
      if (b && b.mode !== 'setup' && b.weights?.name) up = Date.now();
    } catch {
      /* still restarting */
    }
    if (!up) await page.waitForTimeout(1000);
  }
  expect(up, 'the engine never answered').toBeGreaterThan(0);
  await expect(page.locator('[data-start-chatting]')).toBeVisible({ timeout: 10_000 });
  const took = Date.now() - up;
  fs.writeFileSync(`${SHOTS}/finished-after-ms.txt`, String(took));
  await expect(page.locator('[data-job-finished]')).toContainText('is ready');
  await expect(page.locator('.inst-spin')).toHaveCount(0);
  await page.screenshot({ path: `${SHOTS}/finished-390.png`, fullPage: true });
});
