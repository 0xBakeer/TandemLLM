// Dashboard races: the stale-response guards (a second filter change must not let the older
// response land) and the dev console's grep-debounce timer cleared on detach, so a late
// openStream cannot start a ResumableStream nothing will ever close again.
// Node env, no DOM: the elements are instantiated without connecting (lit only needs
// customElements to register the tags, stubbed here).

import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest';

// must run before the view modules load: their decorators register the tags
vi.hoisted(() => {
  const g = globalThis as Record<string, unknown>;
  if (!g.customElements) g.customElements = { define: () => {}, get: () => null };
  if (g.__GRAFANA_URL__ === undefined) g.__GRAFANA_URL__ = '';
});

import { api } from '../src/api/client';
import type { Requests, Usage } from '../src/api/types';
import { QsePerformance } from '../src/views/performance';
import { QseUsage } from '../src/views/usage';
import { QseDev } from '../src/views/dev';

const R = (rows: unknown[]): Requests => ({ requests: rows as Requests['requests'], next_before: null }) as Requests;
const U = (n: number): Usage =>
  ({
    bucket: 'day',
    tz: 'UTC',
    from: '2026-01-01',
    to: '2026-12-31',
    totals: { requests: n, errors: 0, prompt_tokens: 0, cached_tokens: 0, completion_tokens: 0, reasoning_tokens: 0, total_tokens: n },
    buckets: [],
    top_days: [],
    dimensions: { models: [], clients: [] },
  }) as unknown as Usage;

describe('Performance: stale responses are dropped (A2)', () => {
  let el: QsePerformance;

  beforeEach(() => {
    vi.useFakeTimers();
    el = new QsePerformance();
    (el as { params: URLSearchParams }).params = new URLSearchParams();
  });
  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it('two recent loads resolved out of order: the older one must not win', async () => {
    const calls: ((v: Requests) => void)[] = [];
    vi.spyOn(api, 'requests').mockImplementation(() => new Promise((res) => calls.push(res)));
    const p1 = (el as unknown as { loadRecent(): Promise<void> }).loadRecent(); // older filter
    const p2 = (el as unknown as { loadRecent(): Promise<void> }).loadRecent(); // newer filter
    calls[0](R([{ id: 111 }]));
    await p1;
    expect((el as unknown as { recent: unknown }).recent).toEqual({ state: 'loading' }); // stale: dropped
    calls[1](R([{ id: 222 }]));
    await p2;
    expect((el as unknown as { recent: { state: string; data?: unknown[] } }).recent).toEqual({ state: 'ready', data: [{ id: 222 }] });
  });

  it('an error from a stale recent load must not clobber a newer ready state', async () => {
    const calls: { res: (v: Requests) => void; rej: (e: unknown) => void }[] = [];
    vi.spyOn(api, 'requests').mockImplementation(() => new Promise((res, rej) => calls.push({ res, rej })));
    const view = el as unknown as { loadRecent(): Promise<void>; recent: { state: string; data?: unknown[] } };
    const p1 = view.loadRecent();
    const p2 = view.loadRecent();
    calls[1].res(R([{ id: 222 }])); // the newer filter's response lands first
    await p2;
    expect(view.recent).toEqual({ state: 'ready', data: [{ id: 222 }] });
    calls[0].rej(new Error('boom')); // the older one fails afterwards
    await p1.catch(() => {});
    expect(view.recent).toEqual({ state: 'ready', data: [{ id: 222 }] }); // the stale error is dropped
  });

  it('two history loads resolved out of order: the older one must not win', async () => {
    const calls: ((v: Usage) => void)[] = [];
    vi.spyOn(api, 'usage').mockImplementation(() => new Promise((res) => calls.push(res)));
    const view = el as unknown as { loadHistory(): Promise<void>; history: unknown };
    const p1 = view.loadHistory();
    const p2 = view.loadHistory();
    calls[0](U(1));
    await p1;
    expect(view.history).toEqual({ state: 'loading' }); // stale: dropped
    calls[1](U(2));
    await p2;
    expect(view.history).toEqual({ state: 'ready', data: U(2) });
  });
});

describe('Usage: stale responses are dropped (A3)', () => {
  let el: QseUsage;

  beforeEach(() => {
    vi.useFakeTimers();
    el = new QseUsage();
    (el as unknown as { params: URLSearchParams }).params = new URLSearchParams();
  });
  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it('two loads resolved out of order: the older one must not win', async () => {
    const calls: ((v: Usage) => void)[] = [];
    vi.spyOn(api, 'usage').mockImplementation(() => new Promise((res) => calls.push(res)));
    const view = el as unknown as { load(): Promise<void>; usage: unknown };
    const p1 = view.load();
    const p2 = view.load();
    calls[0](U(1));
    await p1;
    expect(view.usage).toEqual({ state: 'loading' }); // stale: dropped
    calls[1](U(2));
    await p2;
    expect(view.usage).toEqual({ state: 'ready', data: U(2) });
  });

  it('a custom-range load is skipped entirely once a newer load started', async () => {
    const calls: ((v: Usage) => void)[] = [];
    vi.spyOn(api, 'usage').mockImplementation(() => new Promise((res) => calls.push(res)));
    (el as unknown as { params: URLSearchParams }).params = new URLSearchParams('range=custom&from=2026-01-01&to=2026-01-31');
    const view = el as unknown as { load(): Promise<void>; usage: unknown; custom: unknown };
    const p1 = view.load(); // custom range, in flight
    (el as unknown as { params: URLSearchParams }).params = new URLSearchParams('range=90d'); // the newer load is not custom
    const p2 = view.load();
    calls[0](U(1)); // let the stale load continue past its first await
    await p1;
    expect(calls.length).toBe(2); // it returned without firing its second (custom) request
    calls[1](U(2));
    await p2;
    expect(view.usage).toEqual({ state: 'ready', data: U(2) });
    expect(view.custom).toBe(null);
  });
});

describe('Dev: the grep debounce timer is cleared on detach (A1)', () => {
  let fetchSpy: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    vi.useFakeTimers();
    fetchSpy = vi.fn(() => Promise.resolve({ ok: true, body: null }));
    (globalThis as { fetch: unknown }).fetch = fetchSpy;
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it('setGrep schedules the debounce', () => {
    const el = new QseDev();
    (el as unknown as { setGrep(v: string): void }).setGrep('foo');
    expect((el as unknown as { grepTimer: unknown }).grepTimer).not.toBeNull();
  });

  it('after disconnectedCallback the pending debounce is cleared and never opens a stream', () => {
    const el = new QseDev();
    (el as unknown as { setGrep(v: string): void }).setGrep('foo');
    el.disconnectedCallback();
    expect((el as unknown as { grepTimer: unknown }).grepTimer).toBeNull();
    vi.advanceTimersByTime(1000);
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it('openStream refuses to run on a detached element', () => {
    const el = new QseDev();
    (el as unknown as { openStream(): void }).openStream();
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it('a connected element still opens its stream', () => {
    const el = new QseDev();
    Object.defineProperty(el, 'isConnected', { value: true, configurable: true });
    (el as unknown as { openStream(): void }).openStream();
    expect(fetchSpy).toHaveBeenCalled();
    el.disconnectedCallback();
  });
});
