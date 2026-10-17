// dist/ hygiene: no mock code, relative asset paths, size budget. Runs against the committed build.
import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { gzipSync } from 'node:zlib';
import { describe, expect, it } from 'vitest';

const DIST = join(__dirname, '..', 'dist');

describe('dist/', () => {
  const present = existsSync(join(DIST, 'index.html'));
  it.skipIf(!present)('contains no generator or mock server code', () => {
    const files = readdirSync(join(DIST, 'assets')).filter((f) => f.endsWith('.js'));
    expect(files.length).toBeGreaterThan(0);
    for (const f of files) {
      const src = readFileSync(join(DIST, 'assets', f), 'utf8');
      for (const marker of ['createMockMiddleware', 'generateYear', 'renderMetrics', 'qse_mock', 'mulberry', 'LogRing', '__mock/mode']) {
        expect(src.includes(marker), `${f} contains ${marker}`).toBe(false);
      }
    }
  });
  it.skipIf(!present)('0.8.0: the Chat tab carries a Beta pill in the rail and in the chat header', () => {
    const files = readdirSync(join(DIST, 'assets'));
    const js = files.filter((f) => f.endsWith('.js')).map((f) => readFileSync(join(DIST, 'assets', f), 'utf8'));
    const css = files.filter((f) => f.endsWith('.css')).map((f) => readFileSync(join(DIST, 'assets', f), 'utf8')).join('');
    const pills = js.filter((src) => src.includes('beta-pill') && src.includes('The Chat tab is in beta'));
    expect(pills.length).toBeGreaterThanOrEqual(2);
    expect(css).toMatch(/\.beta-pill\{/);
    expect(css).toMatch(/\.rail-tab \.beta-pill\{/);
  });
  it.skipIf(!present)('index.html uses relative asset paths and no external origin', () => {
    const html = readFileSync(join(DIST, 'index.html'), 'utf8');
    expect(html).toMatch(/src="\.\/assets\//);
    expect(html).toMatch(/href="\.\/assets\//);
    expect(html).not.toMatch(/https?:\/\//);
  });
  // 445 KB since the Setup Guide's restart card, finished screen, Retry and diagnostics and its weights list (VIS-209, VIS-211, VIS-213, VIS-215, 2026-10-17); 435 KB since the verify phase's check card and Checked block (SRV-532, 2026-10-17); 434 KB since the Setup Guide's window steppers (VIS-199, 2026-10-16); 433 KB since its parallel-chats stepper (VIS-199, 2026-10-16); 431 KB since the Chat tab's design pass (VIS-185, 2026-10-16); 430 KB since the preview's target, waiting state, page report and new-server watch (SRV-511, 2026-10-16); 428 KB since a live run's reasons on the Now figures (VIS-181) and the Chat tab's panel for a box whose chatd is down (VIS-179, 2026-10-16); 425 KB since the sandbox's reason on a call and the attachment chip's tokens (VIS-167, VIS-169, 2026-10-15); 424 KB since the Setup Guide's job log and change list merged with the media players (2026-10-14); 423 KB since the media players and the Context panel's memory notes (2026-10-14); 420 since 0.9.0; 415 since the community weights section (2026-10-13); 410 since 0.8.0's phone bar, composer and Beta pill; 405 before (scripts/check-size.mjs)
  it.skipIf(!present)('total gzip size is under 445 KB', () => {
    let total = 0;
    const walk = (d: string) => {
      for (const f of readdirSync(d)) {
        const p = join(d, f);
        if (statSync(p).isDirectory()) walk(p);
        else total += gzipSync(readFileSync(p)).length;
      }
    };
    walk(DIST);
    expect(total).toBeLessThan(445 * 1024);
  });
});
