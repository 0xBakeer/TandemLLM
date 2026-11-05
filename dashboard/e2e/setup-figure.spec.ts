// QA VIS-90 (VIS-89): Wren's figure, the gold DGX Spark box. Every state's pose on the svg, the
// host keeps its size, the eyes stay 5 px across at the 50 px button, reduced motion is still and
// tells the states apart, nothing names an animal. Screenshots go to $PW_SHOTS_DIR (default
// screenshots/figure): the states sheet in dark, light and reduced motion, the desktop panel, the
// 390 px sheet and the mini box landing on a changed field. @mock
// The guide opens in SIMPLE mode for a fresh browser; this file drives the ADVANCED
// wizard, so every context asks for the advanced flow by name before it starts.

import { expect, test, type Page } from '@playwright/test';
import { IS_MOCK } from './helpers';
import { open, setupMock } from './setup-helpers';

test.skip(!IS_MOCK, 'the figure specs drive the mock (POST /__mock/setup)');

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => localStorage.setItem('tandem.setup.simple', '0'));
});
// pictures at 2x, so the cut edge and the dots can be judged
test.use({ deviceScaleFactor: 2 });

const MODEL = 'qwen3.8-27b';
const SHOTS = process.env.PW_SHOTS_DIR ?? 'screenshots/figure';
const STATES = ['idle', 'listening', 'thinking', 'loading', 'talking', 'acting', 'error', 'sleeping'] as const;
const SIZES = [148, 76, 50];
const isPhone = (page: Page) => (page.viewportSize()?.width ?? 1440) <= 1100;

// the ticket's table (VIS-89): eyes, mouth, extra, LED, motion
const POSE: Record<(typeof STATES)[number], string[]> = {
  idle: ['dot', 'smile', 'none', 'breathe', 'breathe'],
  listening: ['wide', 'o', 'none', 'steady', 'lean'],
  thinking: ['up', 'flat', 'question', 'amber', 'wiggle'],
  loading: ['lid', 'flat', 'none', 'steady', 'slow'],
  talking: ['dot', 'open', 'none', 'steady', 'bounce'],
  acting: ['side', 'wide', 'spark', 'flash', 'hop'],
  error: ['small', 'wavy', 'alarm', 'blink', 'shake'],
  sleeping: ['shut', 'tiny', 'z', 'off', 'sleep'],
};

/** A sheet over the page: one column per state, the figure at 148, 76 and 50 px (the last inside a
 *  64 px round button like the phone's). Loading shows progress 0.4. */
async function sheet(page: Page): Promise<void> {
  await page.waitForFunction(() => !!customElements.get('qse-wren-figure'));
  await page.evaluate(
    ({ states, sizes }) => {
      const s = document.createElement('div');
      s.id = 'figsheet';
      s.style.cssText = 'position:fixed;inset:0;z-index:9999;background:var(--bg);display:grid;grid-template-columns:repeat(8,1fr);align-content:center;gap:10px;padding:24px;font:13px var(--font-ui);color:var(--ink-2)';
      for (const st of states) {
        const col = document.createElement('div');
        col.style.cssText = 'display:grid;justify-items:center;gap:22px';
        for (const size of sizes) {
          const f = document.createElement('qse-wren-figure') as HTMLElement & { state: string; size: number; progress: number | null };
          f.state = st;
          f.size = size;
          f.progress = st === 'loading' ? 0.4 : null;
          f.dataset.sheet = `${st}-${size}`;
          if (size === 50) {
            const b = document.createElement('div');
            b.style.cssText = 'width:64px;height:64px;border-radius:50%;display:grid;place-items:center;background:var(--bg-2);border:1px solid var(--rule-strong)';
            b.className = 'figsheet-fab';
            b.append(f);
            col.append(b);
          } else col.append(f);
        }
        const label = document.createElement('div');
        label.textContent = st;
        col.append(label);
        s.append(col);
      }
      document.body.append(s);
    },
    { states: [...STATES], sizes: SIZES },
  );
  await page.locator('#figsheet qse-wren-figure svg').first().waitFor();
}

/** Every animation paused at the same moment, and the talking mouths caught mid-open. */
async function freeze(page: Page): Promise<void> {
  await page.evaluate(async () => {
    await new Promise((r) => setTimeout(r, 300));
    for (const a of document.getAnimations()) {
      a.pause();
      a.currentTime = 700;
    }
    for (const f of document.querySelectorAll<HTMLElement & { blip(): void }>('#figsheet qse-wren-figure[state="talking"]')) f.blip();
    for (const a of document.getAnimations()) if (a.playState === 'running') {
      a.pause();
      a.currentTime = 72;
    }
  });
}

test.beforeEach(async ({ request }) => {
  await setupMock(request, { assistant: 'loaded' });
});

test('each state has its pose, the host keeps its size, the eyes are 5 px at 50 px, no animal is named', async ({ page }) => {
  await open(page, `#/setup/${MODEL}/overview`);
  await sheet(page);
  for (const st of STATES) {
    for (const size of SIZES) {
      const host = page.locator(`[data-sheet="${st}-${size}"]`);
      const svg = host.locator('svg.goldbox');
      const attrs = await svg.evaluate((el) => ['eyes', 'mouth', 'extra', 'led', 'motion'].map((k) => el.getAttribute(`data-${k}`)));
      expect(attrs, `${st} ${size}`).toEqual(POSE[st]);
      await expect(svg).toHaveAttribute('aria-hidden', 'true');
      const box = await host.boundingBox();
      expect(Math.round(box!.width), `${st} ${size} width`).toBe(size);
      expect(Math.round(box!.height), `${st} ${size} height`).toBe(size);
    }
    const eye = await page.locator(`[data-sheet="${st}-50"] .gb-eye-l ellipse`).boundingBox();
    if (st !== 'sleeping') expect(eye!.width, `${st} eye at 50 px`).toBeGreaterThanOrEqual(5);
  }
  await expect(page.locator('[data-sheet="loading-148"] .w-ring-fill')).toBeVisible();
  const markup = await page.locator('#figsheet').innerHTML();
  expect(markup).not.toMatch(/\b(bird|beak|feather|wing|perch|chirp|preen|ruffle)/i);
  expect(await page.locator('#figsheet [aria-label], #figsheet title').count()).toBe(0);
});

test('progress null makes the ring chase; outside 0..1 is clamped', async ({ page }) => {
  await open(page, `#/setup/${MODEL}/overview`);
  await sheet(page);
  const f = page.locator('[data-sheet="loading-148"]');
  await f.evaluate((el: HTMLElement & { progress: number | null }) => (el.progress = null));
  await expect(f.locator('.w-ring-spin')).toBeVisible();
  await f.evaluate((el: HTMLElement & { progress: number | null }) => (el.progress = 7));
  await expect(f.locator('.w-ring-fill')).toHaveAttribute('stroke-dasharray', '100.0 100');
});

test('reduced motion: nothing runs, hop and blip return at once, every state is a different still picture', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await open(page, `#/setup/${MODEL}/overview`);
  await sheet(page);
  await page.waitForTimeout(400);
  const running = await page.evaluate(() => document.getAnimations().filter((a) => a.playState === 'running' && (a.effect as KeyframeEffect | null)?.target?.closest?.('#figsheet')).length);
  expect(running).toBe(0);
  const ms = await page.locator('[data-sheet="acting-148"]').evaluate(async (el: HTMLElement & { hop(): Promise<void>; blip(): void }) => {
    const t = performance.now();
    await el.hop();
    el.blip();
    return { ms: performance.now() - t, anims: el.getAnimations({ subtree: true }).length };
  });
  expect(ms.ms).toBeLessThan(50);
  expect(ms.anims).toBe(0);
  // the still picture: which parts show, and where the eyes and the box are
  const pictures = await page.evaluate((states) => {
    const vis = (el: Element) => getComputedStyle(el).opacity !== '0' && getComputedStyle(el).display !== 'none';
    return states.map((st) => {
      const svg = document.querySelector(`[data-sheet="${st}-148"] svg`)!;
      const parts = [...svg.querySelectorAll('.gb-m, .gb-shut, .gb-lid, .gb-question, .gb-alarm, .gb-z, .gb-spark, .gb-eye, .w-ring')].filter(vis).map((e) => e.getAttribute('class'));
      const eyes = getComputedStyle(svg.querySelector('.gb-eyes')!).transform + getComputedStyle(svg.querySelector('.gb-eye-l')!).transform;
      return JSON.stringify([parts, eyes, getComputedStyle(svg.querySelector('.gb-all')!).transform]);
    });
  }, [...STATES]);
  expect(new Set(pictures).size).toBe(STATES.length);
  // the marks stay visible
  for (const [st, sel] of [['thinking', '.gb-question text'], ['error', '.gb-alarm'], ['error', '.gb-sweat'], ['sleeping', '.gb-z text']] as const) {
    expect(await page.locator(`[data-sheet="${st}-148"] ${sel}`).first().evaluate((e) => Number(getComputedStyle(e).opacity)), `${st} ${sel}`).toBeGreaterThan(0.5);
  }
  if (!isPhone(page)) await page.screenshot({ path: `${SHOTS}/states-reduced-motion-dark.png` });
});

for (const theme of ['dark', 'light'] as const) {
  test(`the states sheet, ${theme}`, async ({ page }) => {
    test.skip(isPhone(page), 'the sheet is a desktop picture');
    await page.emulateMedia({ colorScheme: theme });
    await page.addInitScript((t) => localStorage.setItem('qse.theme', t), theme);
    await open(page, `#/setup/${MODEL}/overview`);
    await sheet(page);
    await freeze(page);
    await page.screenshot({ path: `${SHOTS}/states-${theme}.png` });
    await page.locator('#figsheet > div').first().screenshot({ path: `${SHOTS}/idle-close-${theme}.png` });
  });
}

test('the panel and the mini box landing on a changed field', async ({ page }) => {
  for (const theme of ['dark', 'light'] as const) {
    await page.emulateMedia({ colorScheme: theme });
    await page.evaluate((t) => localStorage.setItem('qse.theme', t), theme).catch(() => undefined);
    await page.addInitScript((t) => localStorage.setItem('qse.theme', t), theme);
    await open(page, `#/setup/${MODEL}/components`);
    await page.reload();
    await expect(page.locator('qse-setup')).toBeVisible();
    const where = isPhone(page) ? 'phone' : 'desktop';
    if (isPhone(page)) {
      await expect(page.locator('.wren-fab qse-wren-figure')).toBeVisible();
      await page.screenshot({ path: `${SHOTS}/${where}-fab-${theme}.png` });
      await page.locator('.wren-fab').click();
      await expect(page.locator('.wren-panel.is-open')).toBeVisible();
      await page.waitForTimeout(500);
      await page.screenshot({ path: `${SHOTS}/${where}-sheet-${theme}.png` });
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    } else {
      await page.waitForTimeout(500);
      await page.screenshot({ path: `${SHOTS}/${where}-panel-${theme}.png` });
    }
    const p = page.locator('.wren-panel');
    await p.locator('.wren-input').fill('turn off the vision tower');
    await p.locator('.wren-input').press('Enter');
    const mini = page.locator('[data-field="components.vision"] > .wren-mini');
    await expect(mini).toBeAttached({ timeout: 10_000 });
    await expect(mini.locator('svg.wren-mini-box')).toBeVisible();
    await page.waitForTimeout(760);
    await page.screenshot({ path: `${SHOTS}/${where}-acting-${theme}.png` });
    const field = await page.locator('[data-field="components.vision"]').boundingBox();
    if (field) await page.screenshot({ path: `${SHOTS}/${where}-acting-field-${theme}.png`, clip: { x: Math.max(0, field.x - 20), y: Math.max(0, field.y - 50), width: Math.min(field.width + 40, page.viewportSize()!.width), height: field.height + 70 } });
    await expect(mini).toHaveCount(0, { timeout: 5000 });
    await expect(page.locator('[data-component="vision"] input')).not.toBeChecked();
    await setupMock(page.request, { assistant: 'loaded' });
    await page.evaluate(() => localStorage.removeItem('tandem.setup.v1'));
  }
});
