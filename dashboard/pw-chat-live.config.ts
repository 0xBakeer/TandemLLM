// The Chat tab's live tier (e2e/chat-live.spec.ts): a real chatd and a real engine, Chromium only
// (WebKit does not launch on the Spark), one worker. Run it on the box, never on a laptop:
//   PW_LIVE=1 PW_BASE=http://127.0.0.1:4180/dashboard/ npx playwright test -c pw-chat-live.config.ts
import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  testMatch: /(chat-live[23]?|rc080-live|rc080-phone-live)\.spec\.ts$/,
  timeout: 10 * 60_000,
  expect: { timeout: 15_000 },
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [['list'], ['json', { outputFile: 'e2e-chat-live-results.json' }]],
  outputDir: 'e2e-chat-live-artifacts',
  use: {
    ...devices['Desktop Chrome'],
    // the full Chromium in its new headless mode: the headless shell denies notifications (VIS-110)
    channel: 'chromium',
    baseURL: process.env.PW_BASE ?? 'http://127.0.0.1:4180/dashboard/',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    video: 'off',
    colorScheme: 'dark',
    locale: 'en-GB',
    timezoneId: 'Europe/Berlin',
  },
});
