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
  it.skipIf(!present)('index.html uses relative asset paths and no external origin', () => {
    const html = readFileSync(join(DIST, 'index.html'), 'utf8');
    expect(html).toMatch(/src="\.\/assets\//);
    expect(html).toMatch(/href="\.\/assets\//);
    expect(html).not.toMatch(/https?:\/\//);
  });
  it.skipIf(!present)('total gzip size is under 300 KB', () => {
    let total = 0;
    const walk = (d: string) => {
      for (const f of readdirSync(d)) {
        const p = join(d, f);
        if (statSync(p).isDirectory()) walk(p);
        else total += gzipSync(readFileSync(p)).length;
      }
    };
    walk(DIST);
    expect(total).toBeLessThan(300 * 1024);
  });
});
