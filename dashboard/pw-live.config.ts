// The setup guide's live tier (e2e/setup-live.spec.ts): the real box, Chromium only (WebKit does not
// launch on the Spark), desktop and 390 px. Run it on the box, never on a laptop:
//   PW_LIVE=1 PW_BASE=http://127.0.0.1:8000/dashboard/ npx playwright test -c pw-live.config.ts -g @read
// PW_BASE=https://<the reverse proxy>/dashboard/ runs the same steps through it. PW_SHOTS: where the
// screenshots go (default e2e-live-shots/).
import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  testMatch: /setup-live\.spec\.ts$/,
  timeout: 20 * 60_000,
  expect: { timeout: 15_000 },
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [['list'], ['json', { outputFile: 'e2e-live-results.json' }]],
  outputDir: 'e2e-live-artifacts',
  use: {
    baseURL: process.env.PW_BASE ?? 'http://127.0.0.1:8000/dashboard/',
    ignoreHTTPSErrors: true,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    video: 'off',
    colorScheme: 'dark',
    locale: 'en-GB',
    timezoneId: 'Europe/Berlin',
  },
  projects: [
    { name: 'desktop', use: { ...devices['Desktop Chrome'], viewport: { width: 1440, height: 900 } } },
    { name: 'phone', use: { ...devices['Pixel 7'], viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true } },
  ],
});
