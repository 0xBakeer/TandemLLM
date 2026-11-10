// SRV-804: the Log panel's export (SRV-803). The file name's rule, the save step against a
// stand-in browser, the mock engine's file in the contract's form, the route fetched by the page's
// own client from the mock engine over HTTP, the fetch's time limit, and the panel's source: the
// button sits beside "Clear view" and every control the panel had is still there. The engine's own
// file is tested in tests/test_logbuf.py, the panel in a browser in e2e/dev.spec.ts.
import { readFileSync } from 'node:fs';
import { createServer, type Server } from 'node:http';
import type { AddressInfo } from 'node:net';
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { api, ApiError, onApi, setApiBase } from '../src/api/client';
import { CannotSave, canSave, EXPORT_LIMIT_MS, EXPORT_NAME_MAX, exportName, LOGS_EXPORT, nameFrom, releaseSaved, saveFile, sentence, type SaveEnv } from '../src/lib/log-export';
import { EXPORT_TITLE, exportText, uptimeText } from '../mock/log-export.ts';
import { LogRing } from '../mock/logs.ts';
import { createMockMiddleware } from '../mock/server.ts';

const EXAMPLE = readFileSync(new URL('../../docs/contract/dashboard-v1/logs-export.example.txt', import.meta.url), 'utf8');
const DEV = readFileSync(new URL('../src/views/dev.ts', import.meta.url), 'utf8');
const CSS = readFileSync(new URL('../src/styles/app.css', import.meta.url), 'utf8');
const LOGBUF = readFileSync(new URL('../../server/logbuf.py', import.meta.url), 'utf8');
const ENGINE = readFileSync(new URL('../../server/app.py', import.meta.url), 'utf8');
const NAME = /^[A-Za-z0-9.-]+$/;
const ENTRY = /^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z (debug|info|warning|error) /;

/** the header (the lines before the first blank one) and the entries of an export */
function parts(text: string): { head: string[]; entries: string[] } {
  const cut = text.indexOf('\n\n');
  const head = text.slice(0, cut).split('\n');
  const entries: string[] = [];
  for (const line of text.slice(cut + 2).split('\n')) {
    if (ENTRY.test(line)) entries.push(line);
    else if (line && entries.length) entries[entries.length - 1] += `\n${line}`;
  }
  return { head, entries };
}
const labels = (head: string[]) => head.slice(1).map((l) => l.slice(0, l.indexOf(':')));
const field = (head: string[], label: string) => head.find((l) => l.startsWith(`${label}: `))?.slice(label.length + 2);

describe('the file name', () => {
  it('contains the version and the time of export, in letters, digits, dot and dash only', () => {
    const name = exportName('0.15.3+flashnext-3.05', new Date('2026-11-09T00:39:59.120Z'));
    expect(name).toBe('tandemllm-0.15.3-flashnext-3.05-20261109T003959Z.log');
    for (const v of ['0.15.3+flashnext-3.05', 'v1 beta/2', '../../etc', 'a"b;c\nd', 'ü']) expect(exportName(v)).toMatch(NAME);
  });

  it('a missing version reads unknown', () => {
    for (const v of [null, undefined, '', '   ']) expect(exportName(v, new Date('2026-11-09T00:39:59Z'))).toBe('tandemllm-unknown-20261109T003959Z.log');
  });

  it('a long version is cut: the name before ".log" is at most 120 characters and still ends with the time', () => {
    const at = new Date('2026-11-09T00:39:59Z');
    const room = EXPORT_NAME_MAX - 'tandemllm--20261109T003959Z'.length;
    expect(EXPORT_NAME_MAX).toBe(120);
    expect(room).toBe(93);
    for (const n of [1, room - 1, room]) expect(exportName('v'.repeat(n), at)).toBe(`tandemllm-${'v'.repeat(n)}-20261109T003959Z.log`);
    for (const n of [room + 1, 400, 5000]) {
      const name = exportName('v'.repeat(n), at);
      expect(name, String(n)).toBe(`tandemllm-${'v'.repeat(room)}-20261109T003959Z.log`);
      expect(name.length).toBe(EXPORT_NAME_MAX + '.log'.length);
    }
    // the engine's limit is the same number
    expect(LOGBUF).toMatch(/^EXPORT_NAME_MAX = 120\b/m);
  });

  it("the engine's name is taken from Content-Disposition, and no other kind of name is", () => {
    expect(nameFrom('attachment; filename="tandemllm-0.15.3-20261109T003959Z.log"')).toBe('tandemllm-0.15.3-20261109T003959Z.log');
    expect(nameFrom('attachment; filename=tandemllm-0.15.3-20261109T003959Z.log')).toBe('tandemllm-0.15.3-20261109T003959Z.log');
    for (const d of [null, undefined, '', 'attachment', 'attachment; filename="../x.log"', 'attachment; filename="a b.log"', 'attachment; filename=".profile"', 'attachment; filename="a/b.log"']) {
      expect(nameFrom(d), String(d)).toBeNull();
    }
  });
});

/** a stand-in for what `saveFile` needs of a browser */
function browser(opts: { download?: boolean; url?: boolean; events?: boolean } = {}) {
  const did: string[] = [];
  const heard: { type: string; fn: () => void }[] = [];
  let made = 0;
  const env = {
    document: {
      createElement: (tag: string) => {
        const a: Record<string, unknown> = { tag };
        if (opts.download !== false) a.download = '';
        a.click = () => did.push(`click ${String(a.download)} ${String(a.href)}`);
        a.remove = () => did.push('remove');
        return a;
      },
      body: { append: () => did.push('append') },
    },
    URL: opts.url === false ? {} : { createObjectURL: () => (did.push('url'), `blob:stand-in-${++made}`), revokeObjectURL: (u: string) => did.push(`revoke ${u}`) },
    ...(opts.events === false ? {} : { addEventListener: (type: string, fn: () => void) => heard.push({ type, fn }) }),
  };
  return { env: env as unknown as SaveEnv, did, heard, freed: () => did.filter((d) => d.startsWith('revoke ')) };
}

describe('the save step', () => {
  afterEach(() => {
    releaseSaved(); // the held object URL is the module's own state
    vi.useRealTimers();
  });

  it('hands the bytes to the browser as a download under the given name', () => {
    const b = browser();
    saveFile(new Blob(['x']), 'tandemllm-0.15.3-20261109T003959Z.log', b.env);
    expect(b.did).toEqual(['url', 'append', 'click tandemllm-0.15.3-20261109T003959Z.log blob:stand-in-1', 'remove']);
  });

  it('the object URL is not freed after a fixed time: a browser that reads it late still finds it', () => {
    vi.useFakeTimers();
    const b = browser();
    saveFile(new Blob(['x']), 'a.log', b.env);
    expect(vi.getTimerCount()).toBe(0);
    vi.advanceTimersByTime(3_600_000);
    expect(b.freed()).toEqual([]);
  });

  it('the next export frees the one before it, and only that one', () => {
    const b = browser();
    saveFile(new Blob(['x']), 'a.log', b.env);
    saveFile(new Blob(['y']), 'b.log', b.env);
    expect(b.freed()).toEqual(['revoke blob:stand-in-1']);
    // freed before the next one is made, so the page holds one file at most
    expect(b.did.indexOf('revoke blob:stand-in-1')).toBeLessThan(b.did.lastIndexOf('url'));
    saveFile(new Blob(['z']), 'c.log', b.env);
    expect(b.freed()).toEqual(['revoke blob:stand-in-1', 'revoke blob:stand-in-2']);
  });

  it('the page frees the last one when it goes: one pagehide listener, however many exports', () => {
    const b = browser();
    saveFile(new Blob(['x']), 'a.log', b.env);
    saveFile(new Blob(['y']), 'b.log', b.env);
    expect(b.heard.map((h) => h.type)).toEqual(['pagehide']);
    b.heard[0].fn();
    expect(b.freed()).toEqual(['revoke blob:stand-in-1', 'revoke blob:stand-in-2']);
    b.heard[0].fn(); // nothing is held any more
    expect(b.freed()).toHaveLength(2);
    // and an export after the page came back holds its own again
    saveFile(new Blob(['z']), 'c.log', b.env);
    expect(b.freed()).toHaveLength(2);
    releaseSaved();
    expect(b.freed().at(-1)).toBe('revoke blob:stand-in-3');
  });

  it('a browser with no page events still saves, and the next export frees the URL', () => {
    const b = browser({ events: false });
    saveFile(new Blob(['x']), 'a.log', b.env);
    saveFile(new Blob(['y']), 'b.log', b.env);
    expect(b.did.filter((d) => d.startsWith('click '))).toHaveLength(2);
    expect(b.freed()).toEqual(['revoke blob:stand-in-1']);
  });

  it('a browser that cannot save a generated file is told apart, and nothing is clicked', () => {
    for (const opts of [{ download: false }, { url: false }]) {
      const b = browser(opts);
      expect(canSave(b.env)).toBe(false);
      expect(() => saveFile(new Blob(['x']), 'a.log', b.env)).toThrow(CannotSave);
      expect(b.did).toEqual([]);
    }
    expect(canSave(browser().env)).toBe(true);
    expect(canSave(undefined)).toBe(false); // vitest's node has no document
    expect(canSave({})).toBe(false);
  });
});

// These cases prove the mock engine's file builder, the one the e2e mock tier downloads from. The
// engine's own file has its cases in tests/test_logbuf.py.
describe("the mock engine's own file builder", () => {
  const NOW = new Date('2026-11-09T00:39:59.500Z');
  const build = { version: '0.15.3+flashnext-3.05', gitSha: '952d072d1a2b', uptimeS: 7 * 3600 + 131, now: NOW, dropped: 0, capacity: 10_000, logContent: false };

  it("the mock's header has the fields of the contract example, in its order", () => {
    const ring = new LogRing();
    ring.push('info', 'server', '[server] queue depth 0/8');
    const mine = parts(exportText(ring.all(), build));
    const theirs = parts(EXAMPLE);
    expect(mine.head[0]).toBe(EXPORT_TITLE);
    expect(theirs.head[0]).toBe(EXPORT_TITLE);
    expect(labels(mine.head)).toEqual(labels(theirs.head));
    expect(labels(mine.head)).toEqual(['version', 'commit', 'uptime', 'exported', 'count', 'first line', 'last line', 'ring', 'log content']);
    expect(field(mine.head, 'version')).toBe('0.15.3+flashnext-3.05');
    expect(field(mine.head, 'commit')).toBe('952d072d1a2b');
    expect(field(mine.head, 'uptime')).toBe(field(theirs.head, 'uptime'));
    expect(field(mine.head, 'exported')).toBe('2026-11-09T00:39:59Z (UTC)');
    expect(field(mine.head, 'count')).toBe('1 lines');
    expect(field(mine.head, 'ring')).toBe(field(theirs.head, 'ring'));
    expect(field(mine.head, 'log content')).toBe('off');
    expect(mine.entries).toHaveLength(1);
    expect(parts(EXAMPLE).entries).toHaveLength(6);
  });

  it('the mock writes unknown for a missing value and on for content logging', () => {
    const head = parts(exportText([], { now: NOW })).head;
    for (const label of ['version', 'commit', 'uptime', 'log content']) expect(field(head, label), label).toBe('unknown');
    expect(field(parts(exportText([], { ...build, logContent: true })).head, 'log content')).toBe('on (--log-content)');
    expect(uptimeText(59)).toBe('59 s');
    expect(uptimeText(86_400 + 61)).toBe('1 d 0 h 1 min 1 s');
    expect(uptimeText(-1)).toBe('unknown');
  });

  it("the mock's file for an empty ring is the header and 0 lines", () => {
    const text = exportText(new LogRing().all(), build);
    const { head, entries } = parts(text);
    expect(field(head, 'count')).toBe('0 lines');
    expect(field(head, 'first line')).toBe('none');
    expect(field(head, 'last line')).toBe('none');
    expect(entries).toEqual([]);
    expect(text.endsWith('\n\n')).toBe(true);
  });

  it("the mock's file for a full ring has its 10,000 entries at all levels in order and says the ring was full", () => {
    const ring = new LogRing(10_000);
    const levels = ['debug', 'info', 'warning', 'error'] as const;
    for (let i = 1; i <= 10_050; i++) ring.push(levels[i % 4], 'server', `[server] line ${i}`, null, Date.parse('2026-11-09T00:00:00Z') + i);
    expect(ring.size).toBe(10_000);
    expect(ring.dropped).toBe(50);
    const all = ring.all();
    expect(all.map((l) => l.seq)).toEqual(Array.from({ length: 10_000 }, (_, i) => i + 51));
    const { head, entries } = parts(exportText(all, { ...build, dropped: ring.dropped, capacity: ring.capacity }));
    expect(field(head, 'count')).toBe('10000 lines');
    expect(field(head, 'ring')).toBe('full, the 50 lines before the first one are not in this file (the ring keeps 10000)');
    expect(entries).toHaveLength(10_000);
    expect(entries.map((e) => Number(e.slice(e.lastIndexOf(' ') + 1)))).toEqual(all.map((l) => l.seq));
    for (const level of levels) expect(entries.filter((e) => e.split(' ')[1] === level)).toHaveLength(2500);
    expect(field(head, 'first line')).toBe(`${all[0].ts} (UTC)`);
    expect(field(head, 'last line')).toBe(`${all[all.length - 1].ts} (UTC)`);
  });

  it('the mock keeps a header value on one line, as the engine does', () => {
    const text = exportText([], { ...build, version: '1.0\nlog content: on (--log-content)', gitSha: 'ab\r\ncd\u0000\u2028ef' });
    const head = parts(text).head;
    expect(head).toHaveLength(10);
    expect(field(head, 'version')).toBe('1.0 log content: on (--log-content)');
    expect(field(head, 'commit')).toBe('ab cd ef');
    expect(head.filter((l) => l.startsWith('log content: '))).toEqual(['log content: off']);
  });

  it('the mock keeps a traceback as one entry with its lines in order', () => {
    const ring = new LogRing();
    ring.push('error', 'traceback', 'Traceback (most recent call last):\n  File "server/app.py", line 1440, in _complete\nRuntimeError: decode step failed');
    ring.push('info', 'server', '[server] after');
    const { head, entries } = parts(exportText(ring.all(), build));
    expect(field(head, 'count')).toBe('2 lines');
    expect(entries).toHaveLength(2);
    expect(entries[0].split('\n')).toEqual([expect.stringMatching(/ error Traceback \(most recent call last\):$/), '  File "server/app.py", line 1440, in _complete', 'RuntimeError: decode step failed']);
  });
});

describe('the page fetches the route from the mock engine', () => {
  let server: Server;
  let origin = '';
  const post = (path: string, body: unknown) => fetch(origin + path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const mock = async () => (await (await post('/__mock/mode', {})).json()) as { mode: string; logLines: number; lastSeq: number };
  const setMode = (mode: string) => post('/__mock/mode', { mode });

  beforeAll(async () => {
    const mw = createMockMiddleware({ live: false });
    server = createServer((req, res) => mw(req, res, () => ((res.statusCode = 404), res.end())));
    await new Promise<void>((done) => server.listen(0, '127.0.0.1', done));
    origin = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
    setApiBase(origin);
  });
  afterEach(async () => {
    vi.unstubAllGlobals();
    await setMode('ok');
  });
  afterAll(async () => {
    setApiBase('');
    server.closeAllConnections();
    await new Promise((done) => server.close(done));
  });

  it('The button downloads the log as a text file with a header', async () => {
    const before = await mock();
    const { name, blob } = await api.logsExport();
    expect(name).toMatch(/^tandemllm-0\.1\.0-rc4-\d{8}T\d{6}Z\.log$/);
    expect(name).toMatch(NAME);
    expect(blob.type).toMatch(/^text\/plain/);
    const { head, entries } = parts(await blob.text());
    expect(head[0]).toBe(EXPORT_TITLE);
    expect(labels(head)).toEqual(labels(parts(EXAMPLE).head));
    expect(field(head, 'version')).toBe('0.1.0-rc4');
    expect(field(head, 'commit')).toBe('7d0b176');
    expect(field(head, 'uptime')).toMatch(/\d+ s$/);
    expect(field(head, 'exported')).toMatch(/^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ \(UTC\)$/);
    expect(field(head, 'count')).toBe(`${before.logLines} lines`);
    expect(entries).toHaveLength(before.logLines);
    expect(api.logsExportUrl()).toBe(origin + LOGS_EXPORT);
  });

  it('the answer is an attachment that no cache keeps', async () => {
    const res = await fetch(origin + LOGS_EXPORT);
    expect(res.status).toBe(200);
    expect(res.headers.get('Content-Type')).toBe('text/plain; charset=utf-8');
    expect(res.headers.get('Content-Disposition')).toMatch(/^attachment; filename="tandemllm-[A-Za-z0-9.-]+\.log"$/);
    expect(res.headers.get('Cache-Control')).toBe('no-store');
    expect(res.headers.get('X-Content-Type-Options')).toBe('nosniff');
  });

  it('The file holds the whole ring, not the lines on screen', async () => {
    await post('/__mock/log', { count: 40, level: 'debug', msg: '[server] below the level on screen' });
    await post('/__mock/log', { count: 3, level: 'warning', msg: '[server] needle' });
    const now = await mock();
    // what a page set to level warning and one search term shows
    const page = (await (await fetch(`${origin}/v1/dashboard/logs?follow=0&level=warning&grep=needle&backlog=1000`)).json()) as { lines: unknown[] };
    expect(page.lines).toHaveLength(3);
    // the page's call carries no level, no search and no position
    const seen: string[] = [];
    const real = globalThis.fetch;
    vi.stubGlobal('fetch', (url: string, init?: RequestInit) => (seen.push(String(url)), real(url, init)));
    const { entries, head } = parts(await (await api.logsExport()).blob.text());
    expect(seen).toEqual([origin + LOGS_EXPORT]);
    expect(entries).toHaveLength(now.logLines);
    expect(field(head, 'count')).toBe(`${now.logLines} lines`);
    expect(entries.filter((e) => e.endsWith(' debug [server] below the level on screen'))).toHaveLength(40);
    expect(entries.slice(-3).every((e) => e.endsWith(' warning [server] needle'))).toBe(true);
    // and the route gives the same file whatever a query says
    const other = parts(await (await real(`${origin}${LOGS_EXPORT}?level=error&grep=needle&backlog=1&since=999999`)).text());
    expect(other.entries).toEqual(entries);
    // the export moved nothing: the ring holds what it held
    expect((await mock()).lastSeq).toBe(now.lastSeq);
    expect((await mock()).logLines).toBe(now.logLines);
  });

  it('The session is refused: a 401 is an error the shell hears, and there is no file', async () => {
    const heard: string[] = [];
    const off = onApi((ev) => heard.push(ev));
    await setMode('401');
    const e = await api.logsExport().catch((x) => x);
    off();
    expect(e).toBeInstanceOf(ApiError);
    expect(e.status).toBe(401);
    expect(e.type).toBe('unauthorized');
    expect(e.endpoint).toBe(LOGS_EXPORT);
    expect(heard).toContain('unauthorized');
  });

  it('The engine cannot be reached: the error names the route, and there is no file', async () => {
    await setMode('offline');
    const e = await api.logsExport().catch((x) => x);
    expect(e).toBeInstanceOf(ApiError);
    expect(e.status).toBe(0);
    expect(e.type).toBe('offline');
    expect(e.endpoint).toBe(LOGS_EXPORT);
    expect(e.message).toContain('engine unreachable');
  });

  it('a failed answer or one that is not a text file is an error, never a file', async () => {
    vi.stubGlobal('fetch', async () => new Response(JSON.stringify({ error: { type: 'internal_error', message: 'the log could not be read' } }), { status: 500, headers: { 'Content-Type': 'application/json' } }));
    const failed = await api.logsExport().catch((x) => x);
    expect(failed).toBeInstanceOf(ApiError);
    expect(failed.status).toBe(500);
    expect(failed.message).toBe('the log could not be read');
    vi.stubGlobal('fetch', async () => new Response('<html>sign in</html>', { status: 200, headers: { 'Content-Type': 'text/html' } }));
    const html = await api.logsExport().catch((x) => x);
    expect(html).toBeInstanceOf(ApiError);
    expect(html.message).toBe('the answer is not a log file');
  });
});

describe('the fetch has a time limit', () => {
  const LATE = 'the engine did not send the log within 60 s';
  /** a fetch that takes the request and never answers: it ends only when the caller gives it up */
  const hang = (seen: { asked: number; given_up: number }) => (_url: string, init?: RequestInit) => {
    seen.asked++;
    return new Promise<Response>((_, reject) => init?.signal?.addEventListener('abort', () => (seen.given_up++, reject(new DOMException('The operation was aborted.', 'AbortError')))));
  };

  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it("the limit is the engine's own handler timeout, 60 s", () => {
    expect(EXPORT_LIMIT_MS).toBe(60_000);
    expect(ENGINE).toMatch(/^HANDLER_TIMEOUT_S = float\(os\.environ\.get\("QSE_HANDLER_TIMEOUT_S", 60\.0\)\)$/m);
  });

  it('An engine that takes the request and sends nothing: the fetch is given up at 60 s, with an error that names the route', async () => {
    const seen = { asked: 0, given_up: 0 };
    vi.stubGlobal('fetch', hang(seen));
    let settled = false;
    const result = api.logsExport().catch((x) => x).finally(() => (settled = true));
    await vi.advanceTimersByTimeAsync(59_999);
    expect(seen).toEqual({ asked: 1, given_up: 0 });
    expect(settled).toBe(false);
    await vi.advanceTimersByTimeAsync(1);
    const e = await result;
    expect(seen).toEqual({ asked: 1, given_up: 1 });
    expect(e).toBeInstanceOf(ApiError);
    expect(e.endpoint).toBe(LOGS_EXPORT);
    expect(e.status).toBe(0);
    expect(e.type).toBe('timeout');
    expect(e.message).toBe(LATE);
    expect(`${sentence(e.message)} No file was saved.`).toBe(`${LATE}. No file was saved.`);
    expect(vi.getTimerCount()).toBe(0);
  });

  it('a body that stalls after the headers is given up at the same limit, counted from the start', async () => {
    vi.stubGlobal('fetch', async (_url: string, init?: RequestInit) => {
      await new Promise((r) => setTimeout(r, 20_000)); // the headers take 20 s
      return { ok: true, status: 200, headers: new Headers({ 'Content-Type': 'text/plain; charset=utf-8' }), blob: () => new Promise((_, reject) => init?.signal?.addEventListener('abort', () => reject(new Error('aborted')))) };
    });
    let settled = false;
    const result = api.logsExport().catch((x) => x).finally(() => (settled = true));
    await vi.advanceTimersByTimeAsync(59_999);
    expect(settled).toBe(false);
    await vi.advanceTimersByTimeAsync(1);
    const e = await result;
    expect(e).toBeInstanceOf(ApiError);
    expect(e.type).toBe('timeout');
    expect(e.message).toBe(LATE);
  });

  it('a good answer leaves no timer behind, and a failure before the limit keeps its own message', async () => {
    vi.stubGlobal('fetch', async () => new Response('TandemLLM engine log\n', { status: 200, headers: { 'Content-Type': 'text/plain; charset=utf-8', 'Content-Disposition': 'attachment; filename="tandemllm-1-20261109T003959Z.log"' } }));
    const got = await api.logsExport();
    expect(got.name).toBe('tandemllm-1-20261109T003959Z.log');
    expect(await got.blob.text()).toBe('TandemLLM engine log\n');
    expect(vi.getTimerCount()).toBe(0);
    vi.stubGlobal('fetch', async () => {
      throw new TypeError('fetch failed');
    });
    const e = await api.logsExport().catch((x) => x);
    expect(e.type).toBe('offline');
    expect(e.message).toBe('engine unreachable (fetch failed)');
    expect(vi.getTimerCount()).toBe(0);
  });

  it('the call can be made again after it was given up', async () => {
    const seen = { asked: 0, given_up: 0 };
    vi.stubGlobal('fetch', hang(seen));
    for (const round of [1, 2]) {
      const result = api.logsExport(5_000).catch((x) => x);
      await vi.advanceTimersByTimeAsync(5_000);
      expect((await result).message).toBe('the engine did not send the log within 5 s');
      expect(seen).toEqual({ asked: round, given_up: round });
    }
  });
});

describe('the Log panel', () => {
  const tools = DEV.slice(DEV.indexOf('const tools = html`'), DEV.indexOf('const body = html`'));
  const run = DEV.match(/private async exportLog\(\): Promise<void> \{\n(?:.*\n)*?  \}\n/)?.[0] ?? '';

  it('the export button sits beside "Clear view" and is styled as it is', () => {
    const clear = tools.indexOf('title="Clear view"');
    const button = tools.indexOf('data-log-export');
    expect(clear).toBeGreaterThan(0);
    expect(button).toBeGreaterThan(clear);
    // nothing but the end of the clear button stands between the two
    expect(tools.slice(clear, button).match(/<button/g)).toHaveLength(1);
    const mine = tools.slice(tools.lastIndexOf('<button', button), tools.indexOf('</button>', button));
    expect(mine).toContain('class="btn btn-ghost btn-sm"');
    expect(mine).toContain('title="Export the whole log as a text file"');
    // labelled as the dashboard's other export buttons are
    expect(tools.slice(button, tools.indexOf('</button>', button))).toContain("${icon('down')} Export");
    expect(mine).toContain('?disabled=${this.exporting}');
    expect(mine).toContain('@click=${() => this.exportLog()}');
  });

  it('Every existing feature of the page stays: each control the toolbar and the lines had is still there', () => {
    for (const kept of [
      '<div class="seg" role="radiogroup" aria-label="level">',
      '@click=${() => this.setLevel(l)}',
      'placeholder="search (server-side)"',
      '@input=${(e: Event) => this.setGrep((e.target as HTMLInputElement).value)}',
      "title=${this.follow ? 'Pause' : 'Follow'}",
      '@click=${() => (this.follow ? (this.follow = false) : this.resume())}',
      'title="Wrap long lines" aria-pressed=${this.wrap} @click=${() => (this.wrap = !this.wrap)}',
      'title="Clear view" @click=${() => ((this.lines = []), (this.pending = 0))}',
    ]) expect(tools, kept).toContain(kept);
    for (const kept of [
      "copyButton(`${l.ts} ${l.level} ${l.msg}`, 'Copy line')",
      'this.jumpToRequest(l.request_id as string);',
      'resuming after <code>${this.stream?.lastId',
      "cls: 'panel-console', id: 'console'",
    ]) expect(DEV, kept).toContain(kept);
    // the export touches neither the lines on screen nor the stream
    for (const untouched of ['this.lines', 'this.pending', 'this.stream', 'this.level', 'this.grep', 'this.follow', 'openStream']) expect(run, untouched).not.toContain(untouched);
  });

  it('an error message with no full stop gets one, so "No file was saved." reads as its own sentence', () => {
    expect(sentence('session expired')).toBe('session expired.');
    expect(sentence(' the engine did not answer ')).toBe('the engine did not answer.');
    expect(sentence('Open the file from the engine instead.')).toBe('Open the file from the engine instead.');
    expect(sentence('Is the engine up?')).toBe('Is the engine up?');
    expect(sentence('')).toBe('');
  });

  it('the page checks that it can save before it asks, and saves only what the engine answered', () => {
    expect(run.indexOf('if (!canSave()) throw new CannotSave();')).toBeGreaterThan(0);
    expect(run.indexOf('if (!canSave()) throw new CannotSave();')).toBeLessThan(run.indexOf('await api.logsExport()'));
    expect(run.indexOf('await api.logsExport()')).toBeLessThan(run.indexOf('saveFile(blob,'));
    expect(run.match(/saveFile\(/g)).toHaveLength(1);
  });

  it('an error shows as the other panels show theirs, says that no file was saved, and can be retried', () => {
    const shown = DEV.match(/private renderExportError\(\)[^]*?\n  \}\n/)?.[0] ?? '';
    expect(shown).toContain('<div class="state state-error" role="alert">');
    expect(shown).toContain('Could not export the log from <code>${x.endpoint}</code>');
    expect(shown).toContain('${sentence(x.message)} No file was saved.');
    expect(shown).toContain('Try again');
    expect(shown).toContain('Dismiss');
    // a browser that cannot save gets the engine's own attachment instead
    expect(shown).toContain("? this.exportLink('btn')");
    const link = DEV.match(/private exportLink\([^]*?\n  \}\n/)?.[0] ?? '';
    expect(link).toContain('href=${api.logsExportUrl()} target="_blank" rel="noopener"');
    expect(link).toContain('Open the file</a>');
    expect(DEV).toContain('This browser cannot save a file that the page hands it.');
    expect(DEV.indexOf('${this.renderExportError()}')).toBeLessThan(DEV.indexOf('<div class="console ${this.wrap'));
  });

  it('a good export leaves one line with the name and the link, and the next press clears it', () => {
    // set only after the file was handed over, under the name it was handed over with
    expect(run).toContain('saveFile(blob, file);\n      this.exportSaved = file;');
    // cleared at the press, before the fetch, with the old error
    const cleared = run.indexOf('this.exportSaved = null;');
    expect(cleared).toBeGreaterThan(run.indexOf('this.exportError = null;'));
    expect(cleared).toBeLessThan(run.indexOf('await api.logsExport()'));
    expect(run.slice(run.indexOf('} catch (e) {'))).not.toContain('exportSaved');
    const shown = DEV.match(/private renderExportSaved\(\)[^]*?\n  \}\n/)?.[0] ?? '';
    expect(shown).toContain('if (!this.exportSaved) return nothing;');
    expect(shown).toContain('<div class="console-export-saved" role="status">');
    expect(shown).toContain('<span>Saved as <code>${this.exportSaved}</code></span>');
    expect(shown).toContain('${this.exportLink()}');
    expect(DEV).toContain('${this.renderExportSaved()}${this.renderExportError()}');
    // the line wraps on a phone, and a long name breaks inside it
    expect(CSS).toMatch(/\.console-export-saved \{\n  display: flex;\n  flex-wrap: wrap;/);
    expect(CSS).toMatch(/\.console-export-saved code \{\n  overflow-wrap: anywhere;\n\}/);
  });

  it('the error block is inset like the panel at both widths', () => {
    expect(CSS).toMatch(/\.console-export \{\n  padding: 0 18px 12px;\n\}/);
    expect(CSS).toMatch(/\.console-export \{\n    padding: 0 14px 12px;\n  \}/);
  });
});
