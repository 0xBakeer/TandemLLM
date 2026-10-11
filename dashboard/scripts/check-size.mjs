// Prints the gzip size of every file in dist/ and fails above the 400 KB budget (300 KB before the
// Chat tab, VIS-99, which is its own chunk of about 40 KB gzip plus the mock it loads on demand).
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
console.log(`${(total / 1024).toFixed(1).padStart(7)} KB  total gzip (budget 400 KB)`);
if (total > 400 * 1024) {
  console.error('over budget');
  process.exit(1);
}
