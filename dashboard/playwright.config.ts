// Playwright — the mock tier of the built app on `vite preview --mode mock` (the mock
// middleware answers every endpoint), Chromium + WebKit, desktop 1440×900 and phone 390×844.
// PW_BASE=http://localhost:5173/dashboard/ points the suite at the dev server instead.
// The real-engine tiers (fake-engine on CPU, the box on :8011) use the same specs with
// PW_BASE set and PW_TOKEN=<admin token>; mock-only scenarios are tagged @mock and skipped there.

import { defineConfig, devices } from '@playwright/test';

const base = process.env.PW_BASE ?? 'http://localhost:4173/dashboard/';
const usePreview = !process.env.PW_BASE;

export default defineConfig({
  testDir: './e2e',
  timeout: 45_000,
  expect: { timeout: 8_000 },
  fullyParallel: false,
  workers: 1,
  retries: process.env.CI ? 1 : 0,
  reporter: [['list'], ['html', { open: 'never', outputFolder: 'e2e-report' }], ['json', { outputFile: 'e2e-results.json' }]],
  outputDir: 'e2e-artifacts',
  use: {
    baseURL: base,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    video: 'off',
    colorScheme: 'dark',
    locale: 'en-GB',
    timezoneId: 'Europe/Berlin',
  },
  projects: [
    { name: 'chromium-desktop', testIgnore: /shots\.spec\.ts/, use: { ...devices['Desktop Chrome'], viewport: { width: 1440, height: 900 } } },
    { name: 'webkit-desktop', testIgnore: /shots\.spec\.ts/, use: { ...devices['Desktop Safari'], viewport: { width: 1440, height: 900 } } },
    { name: 'webkit-phone', testIgnore: /shots\.spec\.ts/, use: { ...devices['iPhone 14'], viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true } },
    { name: 'shots', testMatch: /shots\.spec\.ts/, use: { ...devices['Desktop Chrome'], viewport: { width: 1440, height: 900 } } },
  ],
  webServer: usePreview
    ? {
        command: 'npm run build && npm run preview:mock',
        url: 'http://localhost:4173/dashboard/',
        reuseExistingServer: true,
        timeout: 60_000,
      }
    : undefined,
});
