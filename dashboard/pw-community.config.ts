// The community weights and permission mode pictures on the Spark (e2e/community-shots.spec.ts,
// e2e/permission-shots.spec.ts): Chromium alone (WebKit
// needs libraries the Spark lacks), the sizes set in the spec, against a preview server started by
// hand on a free port (PW_BASE=http://localhost:4199/dashboard/).
import { defineConfig, devices } from '@playwright/test';
import base from './playwright.config';

export default defineConfig({
  ...base,
  testMatch: /(community|permission)-shots\.spec\.ts$/,
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'], viewport: { width: 1440, height: 900 } } }],
});
