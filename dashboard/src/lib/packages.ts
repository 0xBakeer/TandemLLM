// The Packages and Skills screen's arithmetic (G2, 2026-10-03): one list of every package from the
// draft document and the server's catalog, its search and filters, the per-kind token limit, a
// class's package list in K1's order (general packages first), the reference split, a new inline
// package and the copy of a file-backed skill to an inline one. No DOM; unit-tested.

import type { CatalogEntry, LibraryHit, Mode3, PackageCatalog, PackageCategory, PackageDoc, PackagesDoc, Retrieval, RouterDoc, UsageCounts } from '../api/conductor';

export const CATEGORY_ORDER: PackageCategory[] = ['package', 'task_skill', 'test_skill', 'card', 'library'];
export const CATEGORY_WORDS: Record<PackageCategory, string> = {
  package: 'package',
  task_skill: 'task skill',
  test_skill: 'test skill',
  card: 'card',
  library: 'library skill',
};
/** the router document's limit for a class's packages (server/conductor/schema.py POLICY, L1) */
export const CLASS_PACKAGES_MAX = 12;
/** the server's estimate and its cut: 3.5 characters a token (server/conductor/packages.py) */
export const CHARS_PER_TOKEN = 3.5;
/** the house rule tools/library_install.py checks, when the catalog has not said */
export const HOUSE_LIMITS = { library: 350, library_card: 200 };

export interface PackageRow {
  name: string;
  pkg: PackageDoc;
  /** the server's view of the saved version; null for a package only the draft holds */
  entry: CatalogEntry | null;
  category: PackageCategory;
  /** the router classes (of the draft router) that list it */
  classes: string[];
  usage: UsageCounts | null;
  /** the library search's score, when a query is on */
  score: number | null;
}

export type CategoryFilter = 'all' | PackageCategory;
export interface Filters {
  q: string;
  category: CategoryFilter;
  cls: string;
  enabled: 'all' | 'on' | 'off';
  over: boolean;
}
export const NO_FILTERS: Filters = { q: '', category: 'all', cls: '', enabled: 'all', over: false };

export function estimateTokens(text: string): number {
  return Math.floor(text.length / CHARS_PER_TOKEN);
}

/** The category the server gave, else the draft's own guess (a package not saved yet). */
export function categoryOf(p: PackageDoc, entry: CatalogEntry | null): PackageCategory {
  if (entry) return entry.category;
  return p.library ? 'library' : p.kind === 'skill' ? 'task_skill' : 'package';
}

/** A package's limit as the server sets it: a library skill 350 and the library card 200 (the
 *  house rule), every other package its max_tokens, the lower of the two when both apply. */
export function limitOf(name: string, p: PackageDoc, category: PackageCategory, cat: Pick<PackageCatalog, 'limits' | 'library'> | null): { limit: number; source: 'house' | 'max_tokens' } {
  const limits = cat?.limits ?? HOUSE_LIMITS;
  const house = category === 'library' ? limits.library : name === (cat?.library.card ?? 'tandem-library') ? limits.library_card : null;
  return house != null && house <= p.max_tokens ? { limit: house, source: 'house' } : { limit: p.max_tokens, source: 'max_tokens' };
}

export function classesOf(router: Pick<RouterDoc, 'classes'> | null, name: string): string[] {
  return (router?.classes ?? []).filter((c) => (c.policy.packages ?? []).includes(name)).map((c) => c.id);
}

/** Every package of the draft, in category order then by name, with the catalog's entry, the
 *  draft router's classes and the week's usage. */
export function packageRows(doc: Pick<PackagesDoc, 'packages'> | null, cat: PackageCatalog | null, usage: Record<string, UsageCounts> | null, router: Pick<RouterDoc, 'classes'> | null): PackageRow[] {
  return Object.entries(doc?.packages ?? {})
    .map(([name, pkg]) => {
      const entry = cat?.packages[name] ?? null;
      return { name, pkg, entry, category: categoryOf(pkg, entry), classes: classesOf(router, name), usage: usage?.[name] ?? null, score: null };
    })
    .sort((a, b) => CATEGORY_ORDER.indexOf(a.category) - CATEGORY_ORDER.indexOf(b.category) || a.name.localeCompare(b.name));
}

function words(q: string): string[] {
  return q.toLowerCase().split(/\s+/).filter(Boolean);
}

export function textMatch(row: PackageRow, q: string): boolean {
  const hay = `${row.name} ${row.pkg.title} ${row.pkg.description} ${row.pkg.source}`.toLowerCase();
  return words(q).every((w) => hay.includes(w));
}

/** Over its limit: the server's word for the saved version; a package only the draft holds is
 *  estimated from its inline text. */
export function isOver(row: PackageRow, cat: Pick<PackageCatalog, 'limits' | 'library'> | null): boolean {
  if (row.entry) return row.entry.over;
  return row.pkg.source === 'inline' && estimateTokens(splitReference(row.pkg.body, !!row.pkg.library)[0]) > limitOf(row.name, row.pkg, row.category, cat).limit;
}

/** The filters, and the search: a client-side match on name, title, description and source for
 *  every row, and for the library also the server's own search (`hits`, its BM25 order and
 *  scores; a library row it ranks comes first in its category, best first). */
export function filterRows(rows: PackageRow[], f: Filters, hits: LibraryHit[] | null, cat: Pick<PackageCatalog, 'limits' | 'library'> | null = null): PackageRow[] {
  const score = new Map((hits ?? []).map((h) => [h.name, h.score]));
  const q = f.q.trim();
  const out = rows
    .filter((r) => f.category === 'all' || r.category === f.category)
    .filter((r) => !f.cls || r.classes.includes(f.cls) || (r.category === 'library' && (r.entry?.fits ?? []).includes(f.cls)))
    .filter((r) => f.enabled === 'all' || (f.enabled === 'on') === r.pkg.enabled)
    .filter((r) => !f.over || isOver(r, cat))
    .filter((r) => !q || textMatch(r, q) || (r.category === 'library' && score.has(r.name)))
    .map((r) => ({ ...r, score: q && r.category === 'library' ? score.get(r.name) ?? null : null }));
  if (!q) return out;
  return out.sort((a, b) => {
    const c = CATEGORY_ORDER.indexOf(a.category) - CATEGORY_ORDER.indexOf(b.category);
    if (c) return c;
    if (a.score != null || b.score != null) return (b.score ?? -1) - (a.score ?? -1) || a.name.localeCompare(b.name);
    return a.name.localeCompare(b.name);
  });
}

/** K1's order of a class's packages: the general ones first, each group in its own order. */
export function generalFirst(list: string[], doc: Pick<PackagesDoc, 'packages'> | null): string[] {
  const gen = (n: string) => !!doc?.packages[n]?.general;
  return [...list.filter(gen), ...list.filter((n) => !gen(n))];
}

/** The router with `name` added to class `cls` (general first), or why it cannot be. */
export function attach(router: RouterDoc, cls: string, name: string, doc: Pick<PackagesDoc, 'packages'> | null): { router: RouterDoc; error: string | null } {
  const i = router.classes.findIndex((c) => c.id === cls);
  if (i < 0) return { router, error: `no class ${cls}` };
  if (doc?.packages[name]?.library) return { router, error: `${name} is a library skill: the router searches the library, a class does not list it` };
  const cur = router.classes[i].policy.packages ?? [];
  if (cur.includes(name)) return { router, error: null };
  if (cur.length >= CLASS_PACKAGES_MAX) return { router, error: `${cls} already lists ${CLASS_PACKAGES_MAX} packages, the most a class may` };
  const classes = router.classes.map((c, k) => (k === i ? { ...c, policy: { ...c.policy, packages: generalFirst([...cur, name], doc) } } : c));
  return { router: { ...router, classes }, error: null };
}

export function detach(router: RouterDoc, cls: string, name: string): RouterDoc {
  return { ...router, classes: router.classes.map((c) => (c.id === cls ? { ...c, policy: { ...c.policy, packages: (c.policy.packages ?? []).filter((n) => n !== name) } } : c)) };
}

const REFERENCE = /^## Reference\b/m;
const FRONT = /^---\s*\n([\s\S]*?)\n---\s*\n/;

/** (the part that goes in, the reference) of a library text; a text that is not a library
 *  skill's goes in whole. */
export function splitReference(text: string, library: boolean): [string, string] {
  const m = library ? REFERENCE.exec(text) : null;
  return m ? [text.slice(0, m.index).trim(), text.slice(m.index).trim()] : [text.trim(), ''];
}

export function splitFrontmatter(text: string): { front: Record<string, string>; body: string } {
  const m = FRONT.exec(text);
  if (!m) return { front: {}, body: text };
  const front: Record<string, string> = {};
  for (const line of m[1].split('\n')) {
    const i = line.indexOf(':');
    if (i > 0 && !/^\s/.test(line)) front[line.slice(0, i).trim()] = line.slice(i + 1).trim().replace(/^["']|["']$/g, '');
  }
  return { front, body: text.slice(m[0].length) };
}

export function newPackageName(doc: Pick<PackagesDoc, 'packages'>): string {
  let n = 1;
  while (doc.packages[`package-${n}`]) n++;
  return `package-${n}`;
}

export function newInline(n: number): PackageDoc {
  return { title: `Package ${n}`, kind: 'package', source: 'inline', description: 'What this package is for, in one sentence.', body: '# Instructions\n\n', max_tokens: 3000, enabled: true };
}

/** A file-backed package as an inline one with the file's text (its frontmatter off; a library
 *  skill keeps its reference, which the server splits off again). */
export function copyToInline(p: PackageDoc, body: string, reference: string): PackageDoc {
  return { ...p, source: 'inline', body: reference ? `${body}\n\n${reference}` : body };
}

export function spanText(u: UsageCounts | null, how: keyof UsageCounts): string {
  const s = u?.[how];
  return s ? `${s.day} · ${s.week}` : '0 · 0';
}

/** The library rows of one profile (all: every row). */
export function retrievalsOf(rows: Retrieval[], profile: string): Retrieval[] {
  return profile ? rows.filter((r) => r.profile === profile) : rows;
}

/** What turning a profile's library switch to `mode` does, for the confirmation. */
export function libraryModeWords(mode: Mode3): string {
  return mode === 'on'
    ? 'the router puts the skill it picks into the prompt from the next request, and the library card on the coding classes with tools'
    : mode === 'shadow'
      ? 'the router searches and picks, and the journal shows the pick; nothing goes into the prompt'
      : 'the router does not search the library';
}
