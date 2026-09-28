// Screenshots of the live activity: the agent turn mid tool call and the stops list,
// desktop and phone, dark and light. `PW_SHOTS_PREFIX=real-` on the box tier; `PW_SHOTS_DIR`
// writes copies somewhere else as well (a review folder).
import { copyFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { test } from '@playwright/test';
import { IS_MOCK, login } from './helpers';

const SIZES = { desktop: { width: 1440, height: 900 }, phone: { width: 390, height: 844 } } as const;
const PREFIX = process.env.PW_SHOTS_PREFIX ?? '';
const THEMES = (process.env.PW_SHOTS_THEMES ?? 'dark,light').split(',') as ('dark' | 'light')[];
const EXTRA = process.env.PW_SHOTS_DIR;

const SCENES: { name: string; scenario: string; at?: number; wait: number }[] = [
  { name: 'activity', scenario: 'agent-turn', at: 35_500, wait: 1500 }, // calling tool write_file, 18 KB of arguments on the way
  { name: 'activity-prefill', scenario: 'agent-turn', at: 9000, wait: 1200 }, // prefilling with the bar
  { name: 'activity-waiting', scenario: 'agent-turn', at: 41_500, wait: 1200 }, // waiting for the client
  { name: 'activity-abandoned', scenario: 'abandoned', at: 4200, wait: 1200 }, // the client left mid-prefill
  { name: 'activity-stops', scenario: 'stops', wait: 1200 },
];

function save(path: string) {
  if (!EXTRA) return;
  mkdirSync(EXTRA, { recursive: true });
  copyFileSync(path, join(EXTRA, path.split('/').pop()!));
}

for (const theme of THEMES) {
  for (const [size, viewport] of Object.entries(SIZES)) {
    for (const scene of SCENES) {
      test(`${scene.name} ${size} ${theme}`, async ({ page, request }) => {
        await page.setViewportSize(viewport);
        await page.emulateMedia({ colorScheme: theme });
        await page.addInitScript((t) => localStorage.setItem('qse.theme', t), theme);
        if (IS_MOCK) {
          await request.post('/__mock/live', { data: { scenario: 'clear', simulator: false } }); // the mock engine's own requests would take the Now line
          await request.post('/__mock/live', { data: { scenario: 'stops' } });
          if (scene.scenario !== 'stops') await request.post('/__mock/live', { data: { scenario: scene.scenario, at: scene.at } });
        }
        await login(page, '#/performance');
        await page.locator('#live-now').waitFor({ timeout: 10_000 });
        await page.waitForTimeout(scene.wait);
        await page.locator('.skeleton').first().waitFor({ state: 'detached', timeout: 10_000 }).catch(() => undefined);
        const path = `screenshots/${PREFIX}${scene.name}-${size}-${theme}.png`;
        // a full-page render clipped to the panel: an element shot scrolls it under the sticky topbar
        await page.mouse.move(0, 0); // off the sparklines: no hover crosshair in the picture
        await page.evaluate(() => document.querySelectorAll('.spark').forEach((el) => el.dispatchEvent(new Event('mouseleave')))); // a tap leaves one on a phone
        await page.evaluate(() => window.scrollTo(0, 0));
        const box = (await page.locator('.live').boundingBox())!;
        await page.screenshot({ path, fullPage: true, clip: { x: box.x, y: box.y, width: box.width, height: box.height } });
        save(path);
        if (size === 'phone') {
          const fold = `screenshots/${PREFIX}${scene.name}-${size}-${theme}-fold.png`;
          await page.evaluate(() => document.querySelectorAll('.spark').forEach((el) => el.dispatchEvent(new Event('mouseleave')))); // the full-page capture re-sets one
          await page.evaluate(() => window.scrollTo(0, 0));
          await page.screenshot({ path: fold, fullPage: false });
          save(fold);
        }
        if (IS_MOCK) await request.post('/__mock/live', { data: { scenario: 'clear', simulator: true } });
      });
    }
  }
}
