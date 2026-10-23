// The bundle's gzip size, the same on every Node, OS and CPU (#130). What Node's own zlib writes
// depends on the zlib Node is built with: main's dist/ read 470,034 bytes on a Node built with stock
// zlib (Homebrew's, macOS) and 471,208 on the official Linux builds of Node 22, 24 and 26, so a
// budget between the two passed on one and failed on the other. pako is zlib in JavaScript, pinned
// in package.json, and writes the same bytes everywhere; at level 6 its total equals stock zlib's,
// the measure the budget was set with.
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';
import pako from 'pako';

/** The budget for the total gzip size of dist/, in bytes. */
export const BUDGET = 460 * 1024;

/** The gzip size of one file's bytes, pako at level 6. */
export const gzipSize = (bytes) => pako.gzip(bytes, { level: 6 }).length;

/** Every file under `dir` with its gzip size, and the total. */
export const distSizes = (dir) => {
  const files = [];
  const walk = (d, rel = '') => {
    for (const f of readdirSync(d).sort()) {
      const p = join(d, f);
      if (statSync(p).isDirectory()) walk(p, rel + f + '/');
      else files.push({ path: rel + f, gzip: gzipSize(readFileSync(p)) });
    }
  };
  walk(dir);
  return { files, total: files.reduce((n, f) => n + f.gzip, 0) };
};
