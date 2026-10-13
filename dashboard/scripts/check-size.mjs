// Prints the gzip size of every file in dist/ and fails above the budget: 300 KB before the Chat
// tab, 400 KB with it (VIS-99: its own chunk of about 40 KB gzip plus the mock it loads on
// demand), 405 KB since the terminal pane runs xterm.js (VIS-105, 2026-10-12: an 81 KB chunk that
// loads only when a terminal opens) and the PWA's worker and notifications (VIS-109), 410 KB since
// 0.8.0's phone bar and composer (rc-mobile, 405.0 KB) and the Chat tab's Beta pill took it to the limit
// (2026-10-13, 405.1 KB): no feature cut, as decided for the release. 415 KB since the community
// weights section of the setup guide (2026-10-13, 412.3 KB: the cards, their search and sort, the
// GitHub sync state and six inline glyphs for the networks). Room looked for first: terser is
// larger than oxc here, top-level mangling is already on, no CSS rule is dead, and one combined
// CSS file would save 2.5 KB but load every view's CSS up front; the module-preload polyfill went.
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { gzipSync } from 'node:zlib';

const dist = new URL('../dist/', import.meta.url).pathname;
let total = 0;
const walk = (d, rel = '') => {
  for (const f of readdirSync(d)) {
    const p = join(d, f);
    if (statSync(p).isDirectory()) walk(p, rel + f + '/');
    else {
      const gz = gzipSync(readFileSync(p)).length;
      total += gz;
      console.log(`${(gz / 1024).toFixed(1).padStart(7)} KB  ${rel}${f}`);
    }
  }
};
walk(dist);
console.log(`${(total / 1024).toFixed(1).padStart(7)} KB  total gzip (budget 415 KB)`);
if (total > 415 * 1024) {
  console.error('over budget');
  process.exit(1);
}
