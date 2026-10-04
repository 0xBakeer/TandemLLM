// The proxy's Backends screen and Setup flow (P1, contract 1.3) on the mock engine: Setup on an
// empty registry, add a backend after a test connection, the write-only key and its hint, a field
// the environment locks, delete asks first, the routing rules with the env rule locked, a user's
// key shown once, a model's price and switches, the backend filter on Usage, a relayed request's
// drawer measured at the proxy, and the screen at 390 px.
import { expect, test, type Page } from '@playwright/test';
import { IS_MOCK, login, noHorizontalOverflow, setMode } from './helpers';

test.skip(!IS_MOCK, 'the backends spec runs on the mock tier (@mock)');

async function open(page: Page, hash = '#/backends', mock: Record<string, unknown> = { reset: true }): Promise<void> {
  await setMode(page.request, 'ok');
  await page.request.post('/__mock/backends', { data: mock });
  await login(page, hash);
  if (hash.startsWith('#/backends')) await expect(page.locator('qse-backends')).toBeVisible();
}

test('an empty registry opens on Setup, and a backend is tested and saved @mock', async ({ page }) => {
  await open(page, '#/backends', { reset: true, empty: true });
  await expect(page.locator('#setup')).toBeVisible();
  await page.locator('.bk-setup-card[data-kind="deepseek"]').click();
  const form = page.locator('#backend-form');
  await expect(form.locator('[data-field="url"] input')).toHaveValue('https://api.deepseek.com/v1');
  await form.locator('[data-field="key"] input').fill('sk-fixture-ABCDEFGH1234');
  await form.getByRole('button', { name: 'Test connection' }).click();
  await expect(form.locator('[data-test="ok"]')).toContainText('2 models');
  await form.getByRole('button', { name: 'Save backend' }).click();
  await expect(page.locator('#backend-detail')).toBeVisible();
  await expect(page.locator('#backend-detail')).toContainText('…1234');
  await expect(page.locator('body')).not.toContainText('sk-fixture-ABCDEFGH1234');
});

test('a failing test says why, before anything is saved @mock', async ({ page }) => {
  await open(page, '#/backends?screen=setup');
  await page.locator('.bk-setup-card[data-kind="generic"]').click();
  const form = page.locator('#backend-form');
  await form.locator('[data-field="url"] input').fill('http://down.local:1/v1');
  await form.getByRole('button', { name: 'Test connection' }).click();
  await expect(form.locator('[data-test="fail"]')).toContainText('refused');
});

test('a field the environment sets is locked and names its variable @mock', async ({ page }) => {
  await open(page);
  const card = page.locator('.bk-card[data-backend="deepseek"]');
  await expect(card).toContainText('up');
  await card.getByRole('button', { name: 'Edit' }).click();
  const url = page.locator('[data-field="url"]');
  await expect(url).toContainText('set by TANDEM_BACKEND_DEEPSEEK_URL');
  await expect(url.locator('input')).toBeDisabled();
  await expect(page.locator('[data-field="key"]')).toContainText('ends in');
  await page.getByRole('button', { name: 'Cancel' }).click();
  // delete asks first, and the env-declared backend refuses
  await page.locator('.bk-card[data-backend="vllm1"]').getByRole('button', { name: 'Edit' }).click();
  let asked = false;
  page.once('dialog', (d) => {
    asked = true;
    void d.accept();
  });
  await page.getByRole('button', { name: 'Delete' }).click();
  await expect(page.locator('.bk-card[data-backend="vllm1"]')).toHaveCount(0);
  expect(asked).toBe(true);
});

test('routing rules keep the env rule; a user key is shown once @mock', async ({ page }) => {
  await open(page, '#/backends?screen=routing');
  await expect(page.locator('.bk-rule').first()).toContainText('set by TANDEM_ROUTE_1');
  await page.getByRole('button', { name: 'Add rule' }).click();
  await page.locator('.bk-rule').nth(1).locator('input.mono').fill('Qwen/*');
  await page.getByRole('button', { name: 'Save rules' }).click();
  await expect(page.locator('.view-backends [role="status"]')).toContainText('rules saved');
  await page.getByRole('tab', { name: 'Users' }).click();
  await page.locator('#bk-user-label').fill('laptop');
  await page.getByRole('button', { name: 'Make a key' }).click();
  const shown = page.locator('.bk-newkey code');
  await expect(shown).toHaveText(/^tdm-/);
  await page.getByRole('button', { name: 'I saved it' }).click();
  await expect(page.locator('.bk-newkey')).toHaveCount(0);
  await expect(page.locator('tr[data-user="laptop"]')).toBeVisible();
});

test('a model gets a price from the list and its switches @mock', async ({ page }) => {
  await open(page, '#/backends?backend=deepseek');
  const row = page.locator('tr[data-model="deepseek-reasoner"]');
  await row.locator('input[aria-label="deepseek-reasoner input price"]').fill('0.55');
  await row.locator('input[aria-label="deepseek-reasoner output price"]').fill('2.19');
  await row.getByRole('button', { name: 'Save' }).click();
  await expect(page.locator('.view-backends [role="status"]')).toContainText('deepseek-reasoner saved');
  await page.locator('tr[data-model="deepseek-chat"] .bk-row-tools button').first().click();
  await expect(page.locator('.bk-features')).toContainText('packages');
});

test('usage filters by backend; a relayed request is measured at the proxy @mock', async ({ page }) => {
  await open(page, '#/usage');
  await expect(page.locator('[data-filter="backend"] select')).toBeVisible();
  await page.locator('[data-filter="backend"] select').selectOption('deepseek');
  await expect(page.locator('.chip', { hasText: 'backend DeepSeek' })).toBeVisible();
  await expect(page.locator('[data-fact="cost"]')).toContainText('$0.04');
  await page.goto('./#/dev');
  const tag = page.locator('.proxy-tag[data-backend="deepseek"]').first();
  await expect(tag).toBeVisible();
  await tag.click();
  await expect(page.locator('[data-measured="proxy"]').first()).toContainText('Measured at the proxy');
});

test('the Backends screen fits a phone @mock', async ({ page }, info) => {
  test.skip(!info.project.name.includes('phone'), 'phone layout');
  await open(page);
  await expect(page.locator('.bk-card').first()).toBeVisible();
  expect(await noHorizontalOverflow(page)).toBe(true);
  await page.locator('.bk-card[data-backend="vllm1"]').getByRole('button', { name: 'Edit' }).click();
  expect(await noHorizontalOverflow(page)).toBe(true);
  await page.goto('./#/backends?screen=setup');
  expect(await noHorizontalOverflow(page)).toBe(true);
});
