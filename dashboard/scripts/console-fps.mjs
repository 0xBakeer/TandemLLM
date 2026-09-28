// the 10,000-line log console's scroll frame rate, against the mock (`npm run preview:mock`).
//   node scripts/console-fps.mjs http://localhost:4173/dashboard/ /tmp/trace.json
// Injects 10,000 lines into the live stream, then wheel-scrolls the console down and back up for 5 s
// in Chromium, timing every animation frame; the Chrome trace (devtools.timeline) goes to the path given.
import { chromium } from '@playwright/test';

const base = process.argv[2], out = process.argv[3];
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
await page.goto(base + '#/dev');
await page.locator('#token').fill('mock');
await page.getByRole('button', { name: 'Open the dashboard' }).click();
await page.locator('.console[data-state="open"]').waitFor();
await page.request.post(base.replace(/\/dashboard\/$/, '') + '/__mock/log', { data: { count: 10000, msg: '[server] fps probe line with some ordinary text in it, as long as a real req line is' } });
await page.waitForFunction(() => document.querySelectorAll('.log-line').length > 0 && (document.querySelector('.console-spacer')?.getBoundingClientRect().height ?? 0) >= 10000 * 18 * 0.9, null, { timeout: 30000 });
const info = await page.evaluate(() => ({ spacer: document.querySelector('.console-spacer').getBoundingClientRect().height, rendered: document.querySelectorAll('.log-line').length, status: document.querySelector('.console-status, .console-head')?.textContent?.trim().slice(0, 80) }));
await page.locator('.console-scroll').hover();
await browser.startTracing(page, { path: out, screenshots: false, categories: ['devtools.timeline', 'disabled-by-default-devtools.timeline.frame', 'blink', 'cc', 'gpu'] });
await page.evaluate(() => { window.__maxTop = 0; document.querySelector('.console-scroll').addEventListener('scroll', (e) => { window.__maxTop = Math.max(window.__maxTop, e.target.scrollTop); }); window.__frames = []; let last = performance.now(); const f = (t) => { window.__frames.push(t - last); last = t; if (!window.__stop) requestAnimationFrame(f); }; requestAnimationFrame(f); });
// 5 s of wheel scrolling: down through the whole list, then back up
const t0 = Date.now();
while (Date.now() - t0 < 5000) { await page.mouse.wheel(0, (Date.now() - t0) < 2500 ? 1200 : -1200); await page.waitForTimeout(16); }
const frames = await page.evaluate(() => { window.__stop = true; return window.__frames.slice(2); });
const maxTop = await page.evaluate(() => window.__maxTop);
await browser.stopTracing();
frames.sort((a, b) => a - b);
const q = (p) => frames[Math.min(frames.length - 1, Math.floor(p * frames.length))];
const long = frames.filter((d) => d > 1000 / 60 * 1.5).length;
console.log(JSON.stringify({ ...info, max_scrollTop: maxTop, frames: frames.length, fps: +(frames.length / (frames.reduce((a, b) => a + b, 0) / 1000)).toFixed(1), p50_ms: +q(0.5).toFixed(2), p95_ms: +q(0.95).toFixed(2), p99_ms: +q(0.99).toFixed(2), max_ms: +frames[frames.length - 1].toFixed(2), over_25ms: long }));
await browser.close();
