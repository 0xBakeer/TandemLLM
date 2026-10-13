// OPS-42's vitest rows for the System view's memory plan (src/lib/residency.ts): the engine's split
// as one bar in load order with the allocator's own share last, and the planner's words for each
// refusal (2, 3, 4, 5 of OPS-41's Gherkin as the dashboard shows them).
import { describe, expect, it } from 'vitest';
import type { MemoryBudget, MemoryLimitRow, MemorySplit, ResidencyDoc, ResidencyRow, ResidentFile } from '../src/api/types';
import { appliesWords, eventWords, gib, inUseWords, limitValue, roleRows, roleStateWords, pendingWords, rowWords, scaledBar, splitSegments, switchWords, tableWords, usageShare, usageWords, windowWords, withEngineLimit } from '../src/lib/residency';

const G = 1024 ** 3;
const split: MemorySplit = { allocated: 70 * G, reserved: 72 * G, parts: { kv_pool: 16 * G, weights: 16 * G, drafter_kv: 10 * G, vision: 0 }, other: 28 * G, reserve: 2 * G, mem_available: 31 * G, max_len: 262144, drafter_caches: 2 };
const row = (over: Partial<ResidencyRow>): ResidencyRow => ({ role: 'decider', allowed: true, reason: null, mem_available_gib: 31, claim_gib: 8.2, basis: 'measured', floor_gib: 15, margin_gib: 6, need_gib: 21, after_gib: 22.8, ...over });

// VIS-95: Flash-Next 3.05 as the live :8000 reported it on 2026-10-12 (/v1/dashboard/system memory.split
// and caches.resident_files), and a cold 4.05's table as the engine reports a cold policy
const fnSplit: MemorySplit = { allocated: 53834397184, reserved: 55643734016, parts: { weights: 47550375616, mtp_weights: 962024788, kv_pool: 1163950080, drafter_kv: 96995840, gdn_state: 115662864, state_store: 281009184, resident_prefix: 238346256 }, other: 3426032556, reserve: 1809336832, mem_available: 66243969024, max_len: 40960, drafter_caches: 0, pinned_cache: 10975956992 };
const warm: ResidentFile = { name: 'ngram_table', path: '/p/3.05/ngram_embedding.safetensors', policy: 'warm', file_bytes: 32640156672, resident_bytes: 9185603584 };
const cold: ResidentFile = { name: 'ngram_table', path: '/p/4.05/ngram_embedding.safetensors', policy: 'cold', file_bytes: 39040193720, resident_bytes: 0, cache_bytes: 2 * G, hits: 960, misses: 40 };

describe('Flash-Next on the system view (VIS-95)', () => {
  it('names the draft layer in the split, after the weights', () => {
    const s = splitSegments(fnSplit);
    expect(s.map((x) => x.key).slice(0, 3)).toEqual(['weights', 'mtp_weights', 'kv_pool']);
    expect(s[1].label).toBe('draft layer (MTP)');
    expect(s.some((x) => x.key === 'pinned_cache')).toBe(false); // page cache, not an allocator part
  });
  it('says what the n-gram table holds under either policy', () => {
    expect(tableWords([warm])).toBe('n-gram table (warm): 8.55 of 30.40 GiB in the page cache, counted as used by every floor check');
    expect(tableWords([cold])).toBe('n-gram table (cold): read past the page cache through a 2.00 GiB row cache, 96 % hits');
    expect(tableWords(undefined)).toBe('');
    expect(tableWords([])).toBe('');
  });
  it("puts max_len against the profile's largest window", () => {
    expect(windowWords(40960, 262144)).toBe("40,960 of 262,144 (the profile's largest)");
    expect(windowWords(8192, null)).toBe('8,192 (the profile fits no window on this device)');
    expect(windowWords(262144, undefined)).toBe('262,144'); // the 27B's profiles name no window
    expect(windowWords(null, 262144)).toBe('—');
  });
});

describe('the split bar', () => {
  it('orders the parts as they load, drops the empty ones, ends with other and reserve', () => {
    const s = splitSegments(split);
    expect(s.map((x) => x.key)).toEqual(['weights', 'kv_pool', 'drafter_kv', 'other', 'reserve']);
    expect(s.reduce((a, x) => a + x.share, 0)).toBeCloseTo(1, 9);
    expect(s[0].share).toBeCloseTo(16 / 72, 9);
  });
  it('is empty without a split (an older server)', () => {
    expect(splitSegments(null)).toEqual([]);
    expect(gib(null)).toBe('—');
    expect(gib(16.35 * G)).toBe('16.35 GiB');
  });
});

describe('the planner in words', () => {
  it('2: a role that fits says what is left', () => {
    expect(rowWords(row({}))).toBe('fits, 22.8 GiB left');
  });
  it('3: the floor refusal names the numbers', () => {
    expect(rowWords(row({ role: 'reader', allowed: false, reason: 'refused_memory', mem_available_gib: 18, claim_gib: 12, after_gib: 6 }))).toBe('refused: it would break the floor (6.0 GiB left, 21.0 GiB needed)');
  });
  it('4 and 5: a running load and a hold', () => {
    expect(rowWords(row({ allowed: false, reason: 'refused_load_in_progress', after_gib: null }))).toBe('refused: another load is running');
    expect(rowWords(row({ allowed: false, reason: 'refused_hold', after_gib: null }))).toBe('refused: a hold owns the board');
  });
});

// OPS-48's vitest rows for OPS-47: the limits as the Memory plan panel shows them.
const lim = (over: Partial<MemoryLimitRow>): MemoryLimitRow => ({ key: 'RESIDENT_PARK_GB', unit: 'GiB', what: 'the rows of one other session', value: 4, flag: 4, source: 'serve.env', applies: 'next request', locked: null, running: 4, pending_restart: false, used: 2.25 * G, cap: 4 * G, ...over });
const budget = { ceiling: 70 * G, peak: 77 * G, slack: 2 * G } as MemoryBudget;

describe('the limits', () => {
  it('draws the ceiling and the peak on one scale with the split', () => {
    const b = scaledBar(split, budget);
    const total = b.segs.reduce((a, x) => a + x.share, 0);
    expect(total + b.rest).toBeCloseTo(1, 9);
    expect(b.ceiling!).toBeCloseTo(70 / (77 * 1.03), 9);
    expect(b.peak!).toBeGreaterThan(b.ceiling!);
    expect(b.peak!).toBeGreaterThan(total); // 72 GiB reserved sits left of the 77 GiB line
    expect(scaledBar(split, null).ceiling).toBeNull();
  });
  it('says each value in its unit, and what it holds against its cap', () => {
    expect(limitValue(lim({}), 70)).toBe('70 GiB');
    expect(limitValue(lim({ unit: 'tokens' }), 262144)).toBe('262,144 tokens');
    expect(limitValue(lim({ unit: 'on/off' }), 0)).toBe('off');
    expect(usageWords(lim({}))).toBe('2.25 of 4.00 GiB');
    expect(usageShare(lim({}))).toBeCloseTo(0.5625, 9);
    expect(usageShare(lim({ used: null }))).toBeNull();
  });
  it('says when a change applies, and when the engine still runs the old value', () => {
    expect(appliesWords(lim({}))).toBe('applies at the next request');
    expect(appliesWords(lim({ applies: 'restart' }))).toBe('needs a restart');
    expect(appliesWords(lim({ applies: 'locked', locked: "the context window is 262,144 tokens (a fixed project limit)" }))).toContain('262,144');
    const chunk = lim({ key: 'PREFIX_CHUNK', unit: 'tokens', applies: 'restart', value: 512, running: 1024, pending_restart: true });
    expect(pendingWords(chunk)).toBe('set to 512 tokens; the engine runs 1,024 tokens until it restarts');
    expect(pendingWords(lim({}))).toBeNull();
  });
  it('quotes the re-read cost of a session switch and the shed log', () => {
    expect(switchWords({ ts: 0, from: 'a', to: 'b', tokens: 97583, reused: 16384, reread: 81199, via: 'resident', ms: 125200 })).toBe('81,199 tokens re-read in 125.2 s, 16,384 reused (a → b, from the shared head)');
    expect(switchWords(null)).toBeNull();
    expect(eventWords({ kind: 'shed', ts: 0, level: 'park', why: 'peak', freed: 3 * G })).toBe('gave back the parked session: 3.00 GiB (for a prefill under the peak)');
    expect(eventWords({ kind: 'refused', ts: 0, n: 262144, need: 80 * G, peak: 77 * G })).toContain('503');
  });
  it('edits one engine limit in the residency document, and removes it back to serve.env', () => {
    const doc: ResidencyDoc = { floor_gib: 15, margin_gib: 6, settle_s: 5, order: [], claims: {} };
    const a = withEngineLimit(doc, 'RESIDENT_PARK_GB', 2);
    expect(a.engine).toEqual({ RESIDENT_PARK_GB: 2 });
    expect(doc.engine).toBeUndefined();
    expect(withEngineLimit(a, 'RESIDENT_PARK_GB', null).engine).toBeUndefined();
  });
});

// 0.6.1: the shared KV pool's pages are committed bytes outside the allocator, and the bar adds up
// to MemTotal - MemAvailable (the 2026-10-10 04:35 reading: 16 GiB "KV pool" and 10 GiB "drafter
// caches" were the reserved address ranges of slot 0 while 12.75 GiB were committed in all).
describe('the split with the shared KV pool', () => {
  const pooled: MemorySplit = { allocated: 46 * G, reserved: 51 * G, parts: { weights: 16 * G, kv_pool: 9 * G, drafter_kv: 3.75 * G, narrator: 3.6 * G }, vmm_parts: ['kv_pool', 'drafter_kv'], other: 26.4 * G, reserve: 5 * G, kv_committed: 12.75 * G, kv_reserved_va: 104 * G, mem_total: 121.6 * G, mem_available: 45.5 * G, in_use: 76.1 * G, outside: 12.35 * G, max_len: 262144, drafter_caches: 2 };
  it('puts the allocator first, then the committed KV pages and what is outside, summing to in use', () => {
    const s = splitSegments(pooled);
    expect(s.map((x) => x.key)).toEqual(['weights', 'narrator', 'other', 'reserve', 'kv_pool', 'drafter_kv', 'outside']);
    const sum = s.reduce((a, x) => a + x.bytes, 0);
    expect(sum).toBeCloseTo(16 * G + 3.6 * G + 26.4 * G + 5 * G + 12.75 * G + 12.35 * G, 0);
    expect(s.find((x) => x.key === 'kv_pool')!.label).toBe('KV pages, committed');
  });
  it('says in use as allocator + KV pages + outside, and the reserved range as not memory', () => {
    expect(inUseWords(pooled)).toContain('KV pages committed 12.8');
    expect(inUseWords(pooled)).toContain('KV addresses reserved 104.0 GiB (not memory)');
    expect(inUseWords(split)).toBe('');
  });
});

describe('the small models under memory pressure', () => {
  const rep = { dynamic: true, order: ['shadow', 'narrator'], policy: { floor_gib: 15, margin_gib: 4, hold_s: 60, thrash_s: 600, max_hold_s: 3600 }, headroom_gib: 30,
    roles: { shadow: { models: ['qwen3.5-0.8b'], state: 'unloaded_for_memory', out_s: 200, freed_gib: 1.9, reload_at_gib: 21.5, hold_s: 60, unloads: 1, reloads: 0 }, narrator: { models: ['qwen3.5-2b'], state: 'reloading', reload_at_gib: 24.5 } } };
  it('names each state in words, in unload order, with the reload line', () => {
    const rows = roleRows(rep);
    expect(rows.map((r) => r.name)).toEqual(['shadow', 'narrator']);
    expect(rows[0].words).toBe('unloaded for memory (3 min, 1.9 GiB back)');
    expect(rows[0].line).toBe('reloads at 21.5 GiB headroom held 60 s');
    expect(rows[1].words).toBe('loading again');
    expect(roleStateWords({ models: [], state: 'loaded' })).toBe('loaded');
    expect(roleRows(null)).toEqual([]);
  });
});
