// OPS-42's vitest rows for the System view's memory plan (src/lib/residency.ts): the engine's split
// as one bar in load order with the allocator's own share last, and the planner's words for each
// refusal (2, 3, 4, 5 of OPS-41's Gherkin as the dashboard shows them).
import { describe, expect, it } from 'vitest';
import type { MemorySplit, ResidencyRow } from '../src/api/types';
import { gib, rowWords, splitSegments } from '../src/lib/residency';

const G = 1024 ** 3;
const split: MemorySplit = { allocated: 70 * G, reserved: 72 * G, parts: { kv_pool: 16 * G, weights: 16 * G, drafter_kv: 10 * G, vision: 0 }, other: 28 * G, reserve: 2 * G, mem_available: 31 * G, max_len: 262144, drafter_caches: 2 };
const row = (over: Partial<ResidencyRow>): ResidencyRow => ({ role: 'decider', allowed: true, reason: null, mem_available_gib: 31, claim_gib: 8.2, basis: 'measured', floor_gib: 15, margin_gib: 6, need_gib: 21, after_gib: 22.8, ...over });

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
