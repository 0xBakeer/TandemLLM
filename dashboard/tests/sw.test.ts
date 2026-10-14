// OPS-131: the service worker (pwa/sw.js) serves the cached shell for a navigation the reverse proxy
// answers 502, 503 or 504, and the page reloads once the engine answers (src/lib/away.ts). The
// worker's source runs here against fake caches and a fake fetch; the Gherkin of OPS-131 that is
// the worker's and the page's is the matrix (the proxy's and the status page's half is not this
// file's).
import { readFileSync } from 'node:fs';
import { describe, expect, it, vi } from 'vitest';
import { restartText, swAway, whenEngineUp } from '../src/lib/away';

type Handler = (e: unknown) => void;

function worker(network: (url: string) => Promise<Response>, shell: string | null = '<!doctype html><title>TandemLLM</title>') {
  const src = readFileSync(new URL('../pwa/sw.js', import.meta.url), 'utf8').replace("'__BUILD__'", '"test"').replace('__SHELL__', '["./index.html"]');
  const on: Record<string, Handler> = {};
  const self = {
    addEventListener: (k: string, f: Handler) => (on[k] = f),
    location: { origin: 'https://tandem.example' },
    registration: { scope: 'https://tandem.example/dashboard/' },
  };
  const caches = {
    match: async () => (shell == null ? undefined : new Response(shell, { headers: { 'Content-Type': 'text/html' } })),
    open: async () => ({ match: async () => undefined, put: async () => undefined, addAll: async () => undefined }),
    keys: async () => [],
    delete: async () => true,
  };
  new Function('self', 'caches', 'fetch', src)(self, caches, (req: { url: string }) => network(req.url));
  /** the worker's answer to a GET of `path` (navigate: a page load), or null when it lets it pass */
  return async (path: string, mode: 'navigate' | 'cors' = 'navigate'): Promise<Response | null> => {
    let out: Promise<Response> | null = null;
    on.fetch({ request: { method: 'GET', url: 'https://tandem.example' + path, mode }, respondWith: (p: Promise<Response>) => (out = p) });
    return out ? await out : null;
  };
}

const status = (s: number) => async () => new Response(`nginx ${s}`, { status: s, headers: { 'Content-Type': 'text/html' } });

describe('OPS-131: the dashboard stays usable while the engine restarts', () => {
  it('Reload the install page during the restart: a 502, 503 or 504 page load gets the cached shell', async () => {
    for (const s of [502, 503, 504]) {
      const get = worker(status(s));
      const page = await get('/dashboard/');
      expect(page?.status).toBe(200);
      expect(await page!.text()).toContain('<title>TandemLLM</title>');
      // the page asks why, once
      const why = await (await get('/dashboard/sw-away', 'cors'))!.json();
      expect(why).toMatchObject({ away: true, status: s });
      expect(await (await get('/dashboard/sw-away', 'cors'))!.json()).toEqual({ away: false });
    }
  });

  it('a network error gets the cached shell too (status 0); a page the network serves is the network\'s', async () => {
    let get = worker(async () => {
      throw new TypeError('Failed to fetch');
    });
    expect(await (await get('/dashboard/#/setup'))!.text()).toContain('TandemLLM');
    expect(await (await get('/dashboard/sw-away', 'cors'))!.json()).toMatchObject({ away: true, status: 0 });
    get = worker(async () => new Response('fresh', { status: 200 }));
    expect(await (await get('/dashboard/'))!.text()).toBe('fresh');
    expect(await (await get('/dashboard/sw-away', 'cors'))!.json()).toEqual({ away: false });
    // other errors are the page's own: a 404 stays a 404
    get = worker(status(404));
    expect((await get('/dashboard/'))!.status).toBe(404);
  });

  it('First visit during the restart: with nothing cached, the proxy\'s answer stands', async () => {
    const get = worker(status(503), null);
    const page = await get('/dashboard/');
    expect(page?.status).toBe(503);
  });

  it('API calls during the restart are never the worker\'s', async () => {
    const get = worker(status(502));
    expect(await get('/v1/setup/jobs/1e1ecbede39315c9', 'cors')).toBeNull();
    expect(await get('/dashboard/v1/health', 'cors')).toBeNull();
  });

  it('the page reads the worker\'s answer and reloads once /health answers', async () => {
    const res = (b: unknown, type = 'application/json') => async () => new Response(JSON.stringify(b), { headers: { 'Content-Type': type } });
    expect(await swAway(res({ away: true, status: 502, at: 5 }) as typeof fetch)).toEqual({ status: 502, at: 5 });
    expect(await swAway(res({ away: false }) as typeof fetch)).toBeNull();
    expect(await swAway(res('<html>', 'text/html') as typeof fetch)).toBeNull(); // no worker: the engine's 404 page
    expect(await swAway((async () => Promise.reject(new TypeError('x'))) as typeof fetch)).toBeNull();
    expect(restartText(502)).toBe('The engine is restarting (the proxy answered 502). This page reloads when it answers.');
    expect(restartText(0)).toContain('cannot be reached');
    vi.useFakeTimers();
    try {
      const answers = [false, false, true];
      const health = vi.fn(async () => answers.shift() ?? true);
      const up = vi.fn();
      whenEngineUp(health, up, 3000);
      await vi.advanceTimersByTimeAsync(0);
      expect(health).toHaveBeenCalledTimes(1);
      expect(up).not.toHaveBeenCalled();
      await vi.advanceTimersByTimeAsync(6000);
      expect(health).toHaveBeenCalledTimes(3);
      expect(up).toHaveBeenCalledTimes(1);
      await vi.advanceTimersByTimeAsync(30_000);
      expect(health).toHaveBeenCalledTimes(3);
      // stopped before the engine came back: no reload
      const up2 = vi.fn();
      const stop = whenEngineUp(async () => false, up2, 3000);
      stop();
      await vi.advanceTimersByTimeAsync(30_000);
      expect(up2).not.toHaveBeenCalled();
    } finally {
      vi.useRealTimers();
    }
  });
});
