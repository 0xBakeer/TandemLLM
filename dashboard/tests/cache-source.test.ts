import { describe, expect, it } from 'vitest';
import { cacheSummary, cacheTier, reuse, TIER_LABEL } from '../src/lib/cache-source';

describe('cache sources (0.3.21-test)', () => {
  it('the engine names read as the tier they sit in', () => {
    expect(cacheTier('resident')).toBe('resident');
    expect(cacheTier('end')).toBe('end');
    expect(cacheTier('disk')).toBe('disk');
    expect(cacheTier('spill')).toBe('disk');
    expect(cacheTier('prefix')).toBe('store');
    expect(cacheTier('session')).toBe('store');
    expect(cacheTier('response')).toBe('response');
    expect(cacheTier('none')).toBe('cold');
    expect(cacheTier(null)).toBe('cold');
    expect(cacheTier(undefined)).toBe('cold');
    expect(TIER_LABEL[cacheTier('end')]).toBe('RAM end anchor');
    expect(TIER_LABEL[cacheTier('disk')]).toBe('NVMe disk');
    expect(TIER_LABEL[cacheTier('resident')]).toBe('RAM resident');
    expect(TIER_LABEL[cacheTier('none')]).toBe('cold');
  });

  it('reused and forwarded: the server\'s forwarded count when it sends one', () => {
    expect(reuse(33956, 28672)).toEqual({ reused: 28672, forwarded: 5284 });
    expect(reuse(100, null)).toEqual({ reused: 0, forwarded: 100 });
    // the end anchor's alignment serves the rows' tokens: the client's count and the forwarded one differ
    expect(reuse(73374, 73300, 80)).toEqual({ reused: 73300, forwarded: 80 });
    expect(reuse(null, 5)).toEqual({ reused: null, forwarded: null });
    expect(reuse(10, 50)).toEqual({ reused: 10, forwarded: 0 });
  });

  it('the last hour, per tier, in a fixed order; older and prompt-less rows left out', () => {
    const now = Date.parse('2026-10-03T12:00:00Z');
    const at = (minAgo: number) => new Date(now - minAgo * 60_000).toISOString();
    const rows = [
      { ts: at(1), prompt_tokens: 33956, cached_tokens: 28672, ttft_ms: 8900, cache_source: 'disk' },
      { ts: at(2), prompt_tokens: 73374, cached_tokens: 68608, ttft_ms: 4100, cache_source: 'resident' },
      { ts: at(3), prompt_tokens: 74000, cached_tokens: 73700, ttft_ms: 700, cache_source: 'end' },
      { ts: at(4), prompt_tokens: 75000, cached_tokens: 74900, ttft_ms: 500, cache_source: 'end' },
      { ts: at(5), prompt_tokens: 24062, cached_tokens: 0, ttft_ms: 33400, cache_source: 'none' },
      { ts: at(30), prompt_tokens: 200, cached_tokens: 200, ttft_ms: 3, cache_source: 'response' },
      { ts: at(61), prompt_tokens: 9999, cached_tokens: 0, ttft_ms: 1, cache_source: 'none' },
      { ts: at(10), prompt_tokens: null, cached_tokens: null, ttft_ms: null, cache_source: null },
    ];
    const s = cacheSummary(rows, now);
    expect(s.rows.map((r) => r.tier)).toEqual(['end', 'resident', 'disk', 'cold', 'response']);
    const end = s.rows[0];
    expect(end).toMatchObject({ requests: 2, reused: 148600, forwarded: 400, ttftP50: 600, label: 'RAM end anchor' });
    expect(s.rows[2]).toMatchObject({ tier: 'disk', requests: 1, reused: 28672, forwarded: 5284 });
    expect(s.rows[3]).toMatchObject({ tier: 'cold', reused: 0, forwarded: 24062 });
    expect(s.requests).toBe(6);
    expect(s.reused + s.forwarded).toBe(33956 + 73374 + 74000 + 75000 + 24062 + 200);
    expect(s.reusedShare).toBeCloseTo(s.reused / (s.reused + s.forwarded), 6);
    expect(cacheSummary([], now)).toMatchObject({ rows: [], requests: 0, reusedShare: null });
  });
});
