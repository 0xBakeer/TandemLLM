// VIS-179 (QA VIS-180): the Chat tab's probe never falls back to the mock outside ?chat=mock and
// the dev server; every other answer is a named state, chosen by the status and `error.type` and
// never by the message; the panel's words per state, and the re-probe's schedule.
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { setApiBase } from '../src/api/client';
import { classifyProbe, isChatDown, pickBackend, type ChatBackend, type ChatDownCause } from '../src/chat/backend';
import { downText, probeDelay, Reprobe, type PageVisibility } from '../src/chat/chat-down';

const MOCK = { kind: 'mock' } as unknown as ChatBackend;
const REMOTE = { kind: 'remote' } as unknown as ChatBackend;

function answer(status: number, body: unknown, type = 'application/json'): Response {
  return new Response(typeof body === 'string' ? body : JSON.stringify(body), { status, headers: { 'Content-Type': type } });
}

function loaders() {
  return { mock: vi.fn(async () => MOCK), remote: vi.fn(async () => REMOTE) };
}

describe('pickBackend', () => {
  beforeEach(() => setApiBase(''));
  afterEach(() => vi.unstubAllGlobals());

  const rows: [string, () => Response | Promise<never>, ChatDownCause | 'live', number | null, string][] = [
    ['200 JSON', () => answer(200, { session: { password_set: false, signed_in: true } }), 'live', 200, ''],
    ['502 chat_unavailable', () => answer(502, { error: { type: 'chat_unavailable', message: 'the chat service is not running' } }), 'chatd_down', 502, 'the chat service is not running'],
    ['502 no forward key', () => answer(502, { error: { type: 'chat_unavailable', message: 'chatd has not started on this box yet (no forward key)' } }), 'chatd_down', 502, 'chatd has not started on this box yet (no forward key)'],
    ['403 forbidden from chatd', () => answer(403, { error: { type: 'forbidden', message: "chatd answers the engine server's forward only" } }), 'chatd_refused', 403, "chatd answers the engine server's forward only"],
    ['404', () => answer(404, '', 'text/plain'), 'no_chat_routes', 404, ''],
    ['503 setup_mode', () => answer(503, { error: { type: 'setup_mode', message: 'the box is in setup mode' } }), 'setup_mode', 503, 'the box is in setup mode'],
    ['200 HTML', () => answer(200, '<!doctype html><title>x</title>', 'text/html'), 'not_an_engine', 200, ''],
    ['500', () => answer(500, { error: { type: 'server_error', message: 'boom' } }), 'server_error', 500, 'boom'],
    ['502 from a proxy, not JSON', () => answer(502, '<html>Bad Gateway</html>', 'text/html'), 'server_error', 502, ''],
    ['network error', () => Promise.reject(new TypeError('Failed to fetch')), 'offline', null, 'Failed to fetch'],
  ];

  for (const [name, res, cause, status, message] of rows) {
    it(`${name} -> ${cause} in a production build`, async () => {
      const fetch = vi.fn(async () => res());
      vi.stubGlobal('fetch', fetch);
      const load = loaders();
      const got = await pickBackend(new URLSearchParams(), load, false);
      expect(fetch).toHaveBeenCalledWith('/v1/chat-app/capabilities', expect.objectContaining({ credentials: 'same-origin' }));
      expect(load.mock).not.toHaveBeenCalled();
      if (cause === 'live') {
        expect(got).toEqual({ backend: REMOTE, reason: 'probed' });
        return;
      }
      expect(load.remote).not.toHaveBeenCalled();
      expect(got).toEqual({ backend: null, reason: 'unavailable', down: { cause, status, message } });
    });
  }

  it('?chat=mock loads the mock and never probes', async () => {
    const fetch = vi.fn();
    vi.stubGlobal('fetch', fetch);
    const load = loaders();
    expect(await pickBackend(new URLSearchParams('chat=mock'), load, false)).toEqual({ backend: MOCK, reason: 'forced' });
    expect(fetch).not.toHaveBeenCalled();
  });

  it('the dev server, which has no chat routes, gets the mock with its own reason', async () => {
    for (const res of [() => answer(404, ''), () => answer(200, '<!doctype html>', 'text/html')]) {
      vi.stubGlobal('fetch', vi.fn(async () => res()));
      const load = loaders();
      expect(await pickBackend(new URLSearchParams(), load, true)).toEqual({ backend: MOCK, reason: 'dev' });
    }
    // a dev server in front of an engine whose chatd is down shows the state, not the mock
    vi.stubGlobal('fetch', vi.fn(async () => answer(502, { error: { type: 'chat_unavailable', message: 'x' } })));
    const load = loaders();
    expect((await pickBackend(new URLSearchParams(), load, true)).reason).toBe('unavailable');
    expect(load.mock).not.toHaveBeenCalled();
  });

  it('the state comes from the status and error.type, never from the message', () => {
    const msg = 'chat_unavailable forbidden setup_mode the chat service is not running';
    expect(classifyProbe(502, true, { error: { type: 'other', message: msg } })?.cause).toBe('server_error');
    expect(classifyProbe(403, true, { error: { type: 'unauthorized', message: msg } })?.cause).toBe('server_error');
    expect(classifyProbe(503, true, { error: { type: 'busy', message: msg } })?.cause).toBe('server_error');
    expect(classifyProbe(502, true, { error: { type: 'chat_unavailable', message: 'anything at all' } })?.cause).toBe('chatd_down');
    expect(classifyProbe(502, true, null)?.cause).toBe('server_error');
  });

  it('a failed live call says chatd went away only for a 502 chat_unavailable', () => {
    expect(isChatDown({ status: 502, body: { error: { type: 'chat_unavailable', message: 'x' } } })).toBe(true);
    expect(isChatDown({ status: 502, body: '<html>' })).toBe(false);
    expect(isChatDown({ status: 500, body: { error: { type: 'chat_unavailable' } } })).toBe(false);
    expect(isChatDown(new Error('x'))).toBe(false);
    expect(isChatDown(null)).toBe(false);
  });
});

describe('the panel', () => {
  const at = (cause: ChatDownCause, message = 'whatever the server said') => ({ cause, status: 502, message });

  it('names the state and the way out for each cause', () => {
    const down = downText(at('chatd_down'));
    expect(down.title).toBe('The chat service is not running.');
    expect(down.command).toBe('~/TandemLLM/bin/tandem chat start');
    expect(down.alt).toContain('bash ops/chatd.sh start');
    expect(down.button).toBe('Check again');
    expect(downText(at('chatd_refused')).title).toBe("chatd is running but does not accept this server's key.");
    expect(downText(at('chatd_refused')).command).toContain('tandem chat restart');
    expect(downText(at('chatd_refused')).lead).toContain('Restart both');
    expect(downText(at('no_chat_routes')).title).toBe('This engine build has no Chat tab server (it needs 0.8.0 or newer).');
    expect(downText(at('setup_mode')).link?.href).toBe('#/setup');
    expect(downText(at('not_an_engine')).title).toBe('This address does not serve the chat API.');
    expect(downText(at('offline')).title).toBe('Cannot reach the engine server.');
    expect(downText(at('offline')).button).toBe('Retry');
    expect(downText({ cause: 'server_error', status: 500, message: '' }).title).toBe('The chat server answered HTTP 500.');
  });

  it('never words the panel after the message', () => {
    for (const cause of ['chatd_down', 'chatd_refused', 'no_chat_routes', 'setup_mode', 'not_an_engine', 'server_error', 'offline'] as ChatDownCause[]) {
      expect(downText(at(cause, 'one'))).toEqual(downText(at(cause, 'chatd is fine, use the mock')));
    }
  });
});

describe('the re-probe', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  function page(hidden = false) {
    const fs = new Set<() => void>();
    const p: PageVisibility & { set(h: boolean): void; listeners: number } = {
      hidden: () => hidden,
      on: (f) => fs.add(f),
      off: (f) => fs.delete(f),
      set(h: boolean) {
        hidden = h;
        fs.forEach((f) => f());
      },
      get listeners() {
        return fs.size;
      },
    };
    return p;
  }

  it('waits 5 s, then 10 s, 20 s and 30 s from there on', () => {
    expect([0, 1, 2, 3, 4, 9].map(probeDelay)).toEqual([5_000, 10_000, 20_000, 30_000, 30_000, 30_000]);
  });

  it('probes on that schedule until the chat API is back, then stops', async () => {
    const calls: number[] = [];
    let back = false;
    const r = new Reprobe(async () => (calls.push(Date.now()), back), undefined, page());
    const t0 = Date.now();
    r.start();
    await vi.advanceTimersByTimeAsync(5_000);
    await vi.advanceTimersByTimeAsync(10_000);
    await vi.advanceTimersByTimeAsync(20_000);
    expect(calls.map((t) => t - t0)).toEqual([5_000, 15_000, 35_000]);
    expect(r.nextAt! - Date.now()).toBe(30_000);
    back = true;
    await vi.advanceTimersByTimeAsync(30_000);
    expect(calls.length).toBe(4);
    await vi.advanceTimersByTimeAsync(120_000);
    expect(calls.length).toBe(4);
    expect(r.nextAt).toBeNull();
  });

  it('"Check again" probes at once and records the time; nothing runs while the page is hidden', async () => {
    let n = 0;
    const pv = page();
    const r = new Reprobe(async () => (n++, false), undefined, pv);
    r.start();
    await r.now();
    expect(n).toBe(1);
    expect(r.lastAt).toBe(Date.now());
    pv.set(true);
    await vi.advanceTimersByTimeAsync(300_000);
    expect(n).toBe(1);
    pv.set(false);                      // visible again: a probe at once, then the schedule
    await vi.advanceTimersByTimeAsync(0);
    expect(n).toBe(2);
    r.stop();
    expect(pv.listeners).toBe(0);
    await vi.advanceTimersByTimeAsync(300_000);
    expect(n).toBe(2);
  });
});
