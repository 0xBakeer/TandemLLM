// G2, the Packages and Skills screen's arithmetic (src/lib/packages.ts): the one list from the
// draft and the catalog, the search (text here, the library's ranking from the server) and the
// filters, the per-kind limit, a class's packages in K1's order with the limit of 12, the
// reference and frontmatter splits, a new inline package and the copy of a file to an inline one.
import { describe, expect, it } from 'vitest';
import type { CatalogEntry, PackageCatalog, PackageDoc, PackagesDoc, RouterDoc } from '../src/api/conductor';
import { attach, CLASS_PACKAGES_MAX, categoryOf, copyToInline, detach, estimateTokens, filterRows, generalFirst, isOver, limitOf, libraryModeWords, newInline, newPackageName, NO_FILTERS, packageRows, retrievalsOf, spanText, splitFrontmatter, splitReference, textMatch } from '../src/lib/packages';

const pkg = (over: Partial<PackageDoc> = {}): PackageDoc => ({ title: 't', kind: 'package', source: 'inline', description: 'a package', body: 'x', max_tokens: 3000, enabled: true, ...over });
const entry = (over: Partial<CatalogEntry> = {}): CatalogEntry => ({ category: 'package', source: 'inline', path: null, available: true, error: null, tokens: 10, body_tokens: 10, reference_tokens: 0, chars: 35, cut: false, limit: 3000, limit_source: 'max_tokens', over: false, classes: [], profiles: {}, fits: null, ...over });
const doc: PackagesDoc = {
  skill_dirs: [],
  packages: {
    coding: pkg({ general: true, description: 'How to work on code: small steps.' }),
    testing: pkg({ general: true }),
    'ui-design': pkg({ kind: 'skill', description: 'Calm interfaces, one accent colour.' }),
    'test-web': pkg({ kind: 'skill', description: 'Test a page through input.' }),
    'tandem-library': pkg({ description: 'The library card.' }),
    'browser-game': pkg({ kind: 'skill', source: 'skill:browser-game', library: true, max_tokens: 500, description: 'A 2D browser game loop.' }),
    'canvas-2d': pkg({ kind: 'skill', source: 'skill:canvas-2d', library: true, max_tokens: 500, enabled: false, description: 'Drawing on a canvas.' }),
  },
};
const catalog: PackageCatalog = {
  version: 4,
  router_version: 10,
  tokenizer: 'engine',
  limits: { library: 350, library_card: 200 },
  library: { card: 'tandem-library', profiles: { coding: 'shadow', chat: 'off' } },
  packages: {
    coding: entry({ classes: ['code_edit'] }),
    testing: entry(),
    'ui-design': entry({ category: 'task_skill' }),
    'test-web': entry({ category: 'test_skill' }),
    'tandem-library': entry({ category: 'card', limit: 200, limit_source: 'house', over: true, body_tokens: 240 }),
    'browser-game': entry({ category: 'library', limit: 350, limit_source: 'house', fits: ['code_edit', 'drawing'] }),
    'canvas-2d': entry({ category: 'library', limit: 350, limit_source: 'house', fits: ['drawing'], over: true }),
  },
};
const router = (lists: Record<string, string[]>): RouterDoc => ({ classes: Object.entries(lists).map(([id, packages]) => ({ id, label: id, description: '', policy: { effort: 'medium', packages, rules: [] } })) }) as unknown as RouterDoc;

describe('the one list', () => {
  const rows = packageRows(doc, catalog, { coding: { injected: { day: 1, week: 3 }, shadow: { day: 0, week: 0 }, pick: { day: 0, week: 0 } } }, router({ code_edit: ['coding', 'testing'], design_ui: ['ui-design'] }));
  it('orders by kind, then by name, with the catalog, the draft router and the usage', () => {
    expect(rows.map((r) => r.name)).toEqual(['coding', 'testing', 'ui-design', 'test-web', 'tandem-library', 'browser-game', 'canvas-2d']);
    expect(rows[0].classes).toEqual(['code_edit']);
    expect(rows[0].usage?.injected.week).toBe(3);
    expect(rows[2].classes).toEqual(['design_ui']);
  });
  it('guesses the kind of a package only the draft holds', () => {
    expect(categoryOf(pkg({ library: true }), null)).toBe('library');
    expect(categoryOf(pkg({ kind: 'skill' }), null)).toBe('task_skill');
    expect(categoryOf(pkg(), null)).toBe('package');
    expect(categoryOf(pkg(), entry({ category: 'card' }))).toBe('card');
    const draft = packageRows({ packages: { ...doc.packages, 'package-1': newInline(1) } }, catalog, null, null);
    expect(draft.find((r) => r.name === 'package-1')?.entry).toBeNull();
  });
  it('filters by kind, class (a library skill by its fit), switch and the limit', () => {
    const f = (o: Partial<typeof NO_FILTERS>) => filterRows(rows, { ...NO_FILTERS, ...o }, null, catalog).map((r) => r.name);
    expect(f({ category: 'library' })).toEqual(['browser-game', 'canvas-2d']);
    expect(f({ cls: 'drawing' })).toEqual(['browser-game', 'canvas-2d']);
    expect(f({ cls: 'code_edit' })).toEqual(['coding', 'testing', 'browser-game']);
    expect(f({ enabled: 'off' })).toEqual(['canvas-2d']);
    expect(f({ over: true })).toEqual(['tandem-library', 'canvas-2d']);
    expect(f({ category: 'card', over: true })).toEqual(['tandem-library']);
  });
  it('searches the text here and ranks the library as the server does', () => {
    expect(filterRows(rows, { ...NO_FILTERS, q: 'small steps' }, [], catalog).map((r) => r.name)).toEqual(['coding']);
    const hits = [{ name: 'canvas-2d', description: '', score: 3.2 }, { name: 'browser-game', description: '', score: 1.1 }];
    const got = filterRows(rows, { ...NO_FILTERS, q: 'game loop' }, hits, catalog);
    expect(got.map((r) => [r.name, r.score])).toEqual([['canvas-2d', 3.2], ['browser-game', 1.1]]);
    // a word in the name finds a package the library search did not answer
    expect(filterRows(rows, { ...NO_FILTERS, q: 'TEST-web' }, [], catalog).map((r) => r.name)).toEqual(['test-web']);
    expect(textMatch(rows[0], 'coding small')).toBe(true);
    expect(filterRows(rows, { ...NO_FILTERS, q: 'zzz' }, [], catalog)).toEqual([]);
  });
});

describe('the limit', () => {
  it('is the house rule for a library skill and the library card, else max_tokens', () => {
    expect(limitOf('browser-game', doc.packages['browser-game'], 'library', catalog)).toEqual({ limit: 350, source: 'house' });
    expect(limitOf('tandem-library', doc.packages['tandem-library'], 'card', catalog)).toEqual({ limit: 200, source: 'house' });
    expect(limitOf('coding', doc.packages.coding, 'package', catalog)).toEqual({ limit: 3000, source: 'max_tokens' });
    expect(limitOf('x', pkg({ library: true, max_tokens: 100 }), 'library', catalog)).toEqual({ limit: 100, source: 'max_tokens' });
    expect(limitOf('tandem-library', pkg(), 'card', null)).toEqual({ limit: 200, source: 'house' });
  });
  it('is over by the server for a saved package and by the draft text for a new one', () => {
    const rows = packageRows({ packages: { big: pkg({ body: 'x'.repeat(3600), max_tokens: 1000 }), small: pkg() } }, null, null, null);
    expect(isOver(rows[0], null)).toBe(true);
    expect(isOver(rows[1], null)).toBe(false);
    expect(estimateTokens('x'.repeat(35))).toBe(10);
    // a library draft counts only the part before its reference
    const lib = packageRows({ packages: { l: pkg({ library: true, max_tokens: 500, body: `${'a'.repeat(700)}\n\n## Reference\n\n${'b'.repeat(5000)}` }) } }, null, null, null);
    expect(isOver(lib[0], null)).toBe(false);
  });
});

describe("a class's packages", () => {
  it('attaches general first, keeps the order, refuses the library and the thirteenth', () => {
    const r = router({ code_edit: ['testing', 'test-web'] });
    const a = attach(r, 'code_edit', 'coding', doc);
    expect(a.error).toBeNull();
    expect(a.router.classes[0].policy.packages).toEqual(['testing', 'coding', 'test-web']);
    expect(attach(a.router, 'code_edit', 'ui-design', doc).router.classes[0].policy.packages).toEqual(['testing', 'coding', 'test-web', 'ui-design']);
    expect(attach(r, 'code_edit', 'test-web', doc).router).toBe(r);
    expect(attach(r, 'code_edit', 'browser-game', doc).error).toMatch(/library skill/);
    expect(attach(r, 'nope', 'coding', doc).error).toMatch(/no class/);
    const full = router({ code_edit: Array.from({ length: CLASS_PACKAGES_MAX }, (_, i) => `p${i}`) });
    expect(attach(full, 'code_edit', 'coding', doc).error).toMatch(/12 packages/);
    expect(r.classes[0].policy.packages).toEqual(['testing', 'test-web']);
  });
  it('detaches without touching the other classes', () => {
    const r = router({ code_edit: ['coding', 'testing'], debug: ['coding'] });
    const d = detach(r, 'code_edit', 'coding');
    expect(d.classes.map((c) => c.policy.packages)).toEqual([['testing'], ['coding']]);
  });
  it('orders general packages first', () => {
    expect(generalFirst(['ui-design', 'coding', 'test-web', 'testing'], doc)).toEqual(['coding', 'testing', 'ui-design', 'test-web']);
    expect(generalFirst(['a', 'b'], null)).toEqual(['a', 'b']);
  });
});

describe('texts', () => {
  it('splits a library text at its reference and leaves any other whole', () => {
    expect(splitReference('# A\n\nrules\n\n## Reference\n\ncode', true)).toEqual(['# A\n\nrules', '## Reference\n\ncode']);
    expect(splitReference('# A\n\n## Reference\n\ncode', false)).toEqual(['# A\n\n## Reference\n\ncode', '']);
    expect(splitReference('# A\n\n## References and more', true)[1]).toBe('');
  });
  it('reads a frontmatter', () => {
    const got = splitFrontmatter('---\nname: x\ndescription: "a: b"\n  folded: no\n---\n# Body\n');
    expect(got.front).toEqual({ name: 'x', description: 'a: b' });
    expect(got.body).toBe('# Body\n');
    expect(splitFrontmatter('# none').front).toEqual({});
  });
  it('makes a new inline package and copies a file to one', () => {
    expect(newPackageName({ packages: { 'package-1': pkg() } })).toBe('package-2');
    expect(newInline(2)).toMatchObject({ title: 'Package 2', source: 'inline', kind: 'package', enabled: true });
    const lib = doc.packages['browser-game'];
    expect(copyToInline(lib, '# B', '## Reference\n\nx')).toEqual({ ...lib, source: 'inline', body: '# B\n\n## Reference\n\nx' });
    expect(copyToInline(doc.packages['ui-design'], '# U', '').body).toBe('# U');
  });
});

describe('usage and the library', () => {
  it('says the day and the week', () => {
    expect(spanText({ injected: { day: 2, week: 9 }, shadow: { day: 0, week: 1 }, pick: { day: 0, week: 0 } }, 'injected')).toBe('2 · 9');
    expect(spanText(null, 'pick')).toBe('0 · 0');
  });
  it("keeps one profile's retrieval rows", () => {
    const rows = [{ profile: 'coding' }, { profile: 'chat' }, { profile: 'coding' }] as Parameters<typeof retrievalsOf>[0];
    expect(retrievalsOf(rows, 'coding')).toHaveLength(2);
    expect(retrievalsOf(rows, '')).toHaveLength(3);
  });
  it('says what each switch does', () => {
    expect(libraryModeWords('on')).toMatch(/into the prompt/);
    expect(libraryModeWords('shadow')).toMatch(/nothing goes into the prompt/);
    expect(libraryModeWords('off')).toMatch(/does not search/);
  });
});
