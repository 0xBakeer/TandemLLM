// OPS-177: the Setup Guide as a phone user sees it on a scratch install (tools/release_gate_setup.sh,
// GATE_URL, GATE_SELECTION, GATE_SHOTS). The install is started through the API and followed by the
// page as the page that started it would (the job id in the guide's saved state). The gate has broken
// run/ops/stop.sh, so the first try stops at the preflight: the failed screen says so in plain words,
// Copy diagnostics hands out the report (or shows it to select), and once the gate has put the file
// back (GATE_SHOTS/fixed.txt) Retry runs the install again without a download. While the engine
// restarts the restart card shows (VIS-209); when it answers again the finished screen with Start
// chatting shows within 10 s (VIS-211). Then the weights step removes another profile's weights
// through its Remove button and confirmation (VIS-215). Screenshots go to GATE_SHOTS.
import * as fs from 'node:fs';
import { expect, test } from '@playwright/test';

const URL = process.env.GATE_URL ?? 'http://127.0.0.1:8000';
const SHOTS = process.env.GATE_SHOTS ?? 'gate-shots';

test('an install: a failed try, Retry, the restart card, the finished screen, then Remove', async ({ page, request }) => {
  fs.mkdirSync(SHOTS, { recursive: true });
  const sel = JSON.parse(process.env.GATE_SELECTION ?? '{}');
  await page.goto(`${URL}/dashboard/#/setup`);
  await expect(page.locator('qse-setup')).toBeVisible();
  // the guide's own seed for the model: a full selection object for the saved state
  const cur = await (await request.get(`${URL}/v1/setup/models/${sel.model}/current`)).json();
  const r = await request.post(`${URL}/v1/setup/apply`, { data: { selection: sel } });
  expect(r.status(), await r.text()).toBe(202);
  const job = (await r.json()).job;
  fs.writeFileSync(`${SHOTS}/first-job-id.txt`, job.id);
  await page.evaluate(
    ([model, selection, id]) =>
      localStorage.setItem('tandem.setup.v1', JSON.stringify({ v: 1, model, step: 'install', selection, visited: ['models', 'install'], chat: [], updatedAt: Date.now(), job: id })),
    [sel.model, { ...cur.selection, ...sel }, job.id],
  );
  // a new document reads the saved state (the open one keeps its own and would write it back on
  // the hash change)
  await page.goto('about:blank');
  await page.goto(`${URL}/dashboard/#/setup/${sel.model}/install`);

  // ---- the failed try (VIS-213): plain words, nothing stopped, the way on
  const failedEnd = page.locator('[data-job-failed-end]');
  await expect(failedEnd).toBeVisible({ timeout: 60_000 });
  await expect(page.locator('[data-failed-step="preflight"]')).toBeVisible();
  await expect(page.locator('[data-step="preflight"] .inst-detail')).toContainText('ops/stop.sh has an error');
  await expect(page.locator('[data-step="preflight"] .inst-detail')).toContainText('Nothing was stopped or changed.');
  await expect(page.locator('[data-step="switch"]')).toHaveClass(/is-skipped/);
  await page.locator('[data-copy-diagnostics]').click();
  // the clipboard where the browser lets the page copy, else the text to select
  await expect(page.locator('[data-copy-diagnostics]', { hasText: 'Copied' }).or(page.locator('[data-diagnostics-text]'))).toBeVisible();
  const diag = await (await request.get(`${URL}/v1/setup/jobs/${job.id}/diagnostics`)).text();
  expect(diag).toContain('TandemLLM Setup Guide diagnostics');
  expect(diag).toContain('preflight');
  fs.writeFileSync(`${SHOTS}/diagnostics.txt`, diag);
  await page.screenshot({ path: `${SHOTS}/failed-390.png`, fullPage: true });
  // the gate puts stop.sh back once it sees the failure
  await expect.poll(() => fs.existsSync(`${SHOTS}/fixed.txt`), { timeout: 60_000 }).toBe(true);
  await page.locator('[data-job-retry]').click();

  // ---- the retry: a new job, the download not repeated
  await expect.poll(() => page.evaluate(() => JSON.parse(localStorage.getItem('tandem.setup.v1') ?? '{}').job), { timeout: 15_000 }).not.toBe(job.id);
  const retryId: string = await page.evaluate(() => JSON.parse(localStorage.getItem('tandem.setup.v1') ?? '{}').job);
  fs.writeFileSync(`${SHOTS}/job-id.txt`, retryId);
  const retry = await (await request.get(`${URL}/v1/setup/jobs/${retryId}?format=json`)).json();
  expect(retry.retry_of).toBe(job.id);

  // the page knows where the status responder is (OPS-175: /health's status_port, kept for the restart)
  await expect.poll(() => page.evaluate(() => localStorage.getItem('tandem.statusPort')), { timeout: 15_000 }).not.toBeNull();
  // the restart: one card, no raw errors
  const card = page.locator('[data-restart-card]');
  await expect(card).toBeVisible({ timeout: 120_000 });
  await expect(card).toContainText('Restarting the engine with');
  await page.waitForTimeout(3000);
  await page.screenshot({ path: `${SHOTS}/restart-card-390.png`, fullPage: true });
  // fed by the status responder: the load's phases, not the time alone
  await expect(card).toHaveAttribute('data-blind', 'no', { timeout: 20_000 });
  await expect(page.locator('.rcard-phase').first()).toBeVisible();
  await page.screenshot({ path: `${SHOTS}/restart-card-phases-390.png`, fullPage: true });
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
  const done = await (await request.get(`${URL}/v1/setup/jobs/${retryId}?format=json`)).json();
  expect(done.steps.find((s: { id: string }) => s.id === 'fetch').state).toBe('skipped');

  // ---- Remove (VIS-215): another profile's weights, with the confirmation
  await page.goto(`${URL}/dashboard/#/setup/${sel.model}/weights`);
  const row = page.locator('[data-installed="exl3-3.0"]');
  await expect(row).toBeVisible({ timeout: 20_000 });
  await expect(page.locator('[data-installed="exl3-2.5"] [data-remove]')).toHaveCount(0); // the running one
  await row.locator('[data-remove]').click();
  const ask = row.locator('[data-remove-confirm]');
  await expect(ask).toContainText('bash ops/use-profile.sh exl3-3.0 --fetch-only');
  await page.screenshot({ path: `${SHOTS}/remove-confirm-390.png`, fullPage: true });
  await ask.locator('[data-remove-yes]').click();
  await expect(page.locator('[data-removed]')).toContainText('run/weights-removed.txt');
  await expect(row).toHaveCount(0);
  await page.screenshot({ path: `${SHOTS}/removed-390.png`, fullPage: true });
});
