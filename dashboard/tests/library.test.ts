// R1, the package library on the Packages screen (src/lib/conductor.ts): a class lists only the
// packages that are not the library's, the library's skills are listed by name, and a search
// answer keeps the server's order and drops a name the document no longer holds.
import { describe, expect, it } from 'vitest';
import type { PackageDoc, PackagesDoc } from '../src/api/conductor';
import { classPackages, libraryNames, libraryRows } from '../src/lib/conductor';

const pkg = (over: Partial<PackageDoc> = {}): PackageDoc => ({ title: 't', kind: 'skill', source: 'inline', description: 'a package', body: 'x', max_tokens: 500, enabled: true, ...over });
const doc: PackagesDoc = {
  skill_dirs: ['~/.config/opencode/skills', '~/.qwen38-spark-engine/conductor/skills/library'],
  packages: { 'ui-design': pkg(), testing: pkg({ general: true }), 'web-layout': pkg({ library: true, source: 'skill:web-layout' }), 'browser-game': pkg({ library: true, source: 'skill:browser-game' }), 'tandem-library': pkg({ kind: 'package' }) },
};

describe('the package library', () => {
  it('keeps the library out of the packages a class may list', () => {
    expect(classPackages(doc)).toEqual(['ui-design', 'testing', 'tandem-library']);
    expect(classPackages(null)).toEqual([]);
  });
  it('lists the library by name', () => {
    expect(libraryNames(doc)).toEqual(['browser-game', 'web-layout']);
    expect(libraryRows(doc, null)).toEqual([{ name: 'browser-game', score: null }, { name: 'web-layout', score: null }]);
  });
  it('keeps the search order and drops names the document does not hold', () => {
    const hits = [{ name: 'web-layout', description: '', score: 4.2 }, { name: 'gone', description: '', score: 3 }, { name: 'browser-game', description: '', score: 1.5 }];
    expect(libraryRows(doc, hits)).toEqual([{ name: 'web-layout', score: 4.2 }, { name: 'browser-game', score: 1.5 }]);
    expect(libraryRows(doc, [])).toEqual([]);
  });
});
