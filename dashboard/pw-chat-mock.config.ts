// The Chat tab's mock tier on the Spark (e2e/chat.spec.ts): Chromium at 1440 px and at an iPhone
// 14's 390 px. WebKit needs system libraries the Spark lacks (libva, libnice, libx264: sudo), so
// the phone runs in Chromium's iPhone emulation here; playwright.config.ts keeps the WebKit
// projects for a box that has them.
import { defineConfig, devices } from '@playwright/test';
import base from './playwright.config';

export default defineConfig({
  ...base,
  testMatch: /chat\.spec\.ts$/,
  projects: [
    { name: 'chromium-desktop', use: { ...devices['Desktop Chrome'], viewport: { width: 1440, height: 900 } } },
    { name: 'chromium-phone', use: { ...devices['Desktop Chrome'], viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true, deviceScaleFactor: 3 } },
  ],
});
