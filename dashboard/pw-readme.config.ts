// The README's pictures (e2e/readme-shots.spec.ts): Chromium alone, the sizes set in the spec,
// against a preview server started by hand on a free port:
//   npm run build && npx vite preview --mode mock --port 4197 --strictPort &
//   PW_BASE=http://localhost:4197/dashboard/ PW_MOCK=1 npx playwright test -c pw-readme.config.ts
import { defineConfig, devices } from '@playwright/test';
import base from './playwright.config';

export default defineConfig({
  ...base,
  testMatch: /readme-shots\.spec\.ts$/,
  // UTC: the pictures name no local time zone
  use: { ...base.use, timezoneId: 'UTC' },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
});
