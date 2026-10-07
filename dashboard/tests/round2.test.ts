// FX2 (the UT2 self-test round 2): VIS-58, the Live row's cached share never reads
// above the client's prompt and TandemLLM's own notes show as their own figure; VIS-60, the Usage
// and Requests calls leave TandemLLM's own calls out unless the "show internal" switch asks.
import { afterEach, describe, expect, it, vi } from 'vitest';
import { api } from '../src/api/client';
import { cachedShare } from '../src/lib/live';

afterEach(() => vi.unstubAllGlobals());

function stubFetch(body: unknown): string[] {
  const urls: string[] = [];
  vi.stubGlobal('fetch', async (url: string) => {
    urls.push(url);
    return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } });
  });
  return urls;
}

describe('VIS-58: the live cached share', () => {
  it('names the hidden notes beside the cached tokens', () => {
    expect(cachedShare({ cached_tokens: 15957, hidden_tokens: 4625 })).toBe('15,957 cached · +4,625 hidden');
  });
  it('a request with no notes reads as before, and an older server without the field too', () => {
    expect(cachedShare({ cached_tokens: 13253, hidden_tokens: 0 })).toBe('13,253 cached');
    expect(cachedShare({ cached_tokens: 13253 })).toBe('13,253 cached');
    expect(cachedShare({ cached_tokens: 0, hidden_tokens: null })).toBeNull();
  });
});

describe('VIS-60: internal calls are left out unless asked', () => {
  it('usage and requests send internal=1 only with the switch on', async () => {
    const urls = stubFetch({ requests: [], next_before: null, buckets: [], totals: {}, top_days: [], dimensions: { models: [], clients: [] } });
    await api.usage({ bucket: 'day', tz: 'UTC' });
    await api.usage({ bucket: 'day', tz: 'UTC', internal: true });
    await api.requests({ limit: 25 });
    await api.requests({ limit: 25, internal: true });
    expect(urls[0]).not.toContain('internal');
    expect(urls[1]).toContain('internal=1');
    expect(urls[2]).not.toContain('internal');
    expect(urls[3]).toContain('internal=1');
  });
});
