// The panes against a throwaway Chat tab (e2e/chat-panes-live.spec.ts), WebKit and Chromium, one
// worker. Run it on a box with the sandbox image, never on a laptop:
//   python3 tests/chat_e2e_stack.py --front 127.0.0.1:4188 --dashboard dashboard/dist &
//   PW_LIVE=1 PW_BASE=http://127.0.0.1:4188/dashboard/ npx playwright test -c pw-chat-panes.config.ts
// WebKit needs its system libraries; the Playwright image (mcr.microsoft.com/playwright) has them.
import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  testMatch: /chat-panes-live\.spec\.ts$/,
  timeout: 3 * 60_000,
  expect: { timeout: 15_000 },
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [['list'], ['json', { outputFile: 'e2e-chat-panes-results.json' }]],
  outputDir: 'e2e-chat-panes-artifacts',
  use: {
    baseURL: process.env.PW_BASE ?? 'http://127.0.0.1:4188/dashboard/',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    video: 'off',
    colorScheme: 'dark',
    locale: 'en-GB',
    timezoneId: 'Europe/Berlin',
  },
  projects: [
    { name: 'webkit', use: { ...devices['Desktop Safari'] } },
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
  ],
});
