// The preview pane on a real sandbox (e2e/chat-preview-live.spec.ts, SRV-512): WebKit as a 390 px
// iPhone and desktop Chromium, one worker. Against the throwaway stack or a live box's https name:
//   PW_LIVE=1 PW_BASE=https://tandem.khaled.de/dashboard/ npx playwright test -c pw-chat-preview.config.ts
// WebKit needs its system libraries; the Playwright image (mcr.microsoft.com/playwright) has them.
import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  testMatch: /chat-preview-live\.spec\.ts$/,
  timeout: 6 * 60_000,
  expect: { timeout: 15_000 },
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [['list'], ['json', { outputFile: 'e2e-chat-preview-results.json' }]],
  outputDir: 'e2e-chat-preview-artifacts',
  use: {
    baseURL: process.env.PW_BASE ?? 'http://127.0.0.1:4188/dashboard/',
    // a box's own CA signs its https name
    ignoreHTTPSErrors: true,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    video: 'off',
    colorScheme: 'dark',
    locale: 'en-GB',
    timezoneId: 'Europe/Berlin',
  },
  projects: [
    { name: 'webkit-390', use: { ...devices['iPhone 13'] } },
    { name: 'chromium', use: { ...devices['Desktop Chrome'], viewport: { width: 1440, height: 900 } } },
  ],
});
