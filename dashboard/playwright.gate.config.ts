// The Setup Guide release gate's browser part (OPS-177, tools/release_gate_setup.sh): one spec against
// a scratch install the gate started (GATE_URL), WebKit at 390 px as on a phone, no web server of its
// own. It runs in the mcr.microsoft.com/playwright container on the box.
import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: '.',
  testMatch: /setup-gate\.spec\.ts/,
  timeout: 600_000,
  expect: { timeout: 10_000 },
  workers: 1,
  reporter: [['list']],
  outputDir: process.env.GATE_SHOTS ? `${process.env.GATE_SHOTS}/artifacts` : 'gate-artifacts',
  use: {
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    colorScheme: 'dark',
    locale: 'en-GB',
    timezoneId: 'Europe/Berlin',
  },
  projects: [{ name: 'webkit-phone', use: { ...devices['iPhone 14'], viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true } }],
});
