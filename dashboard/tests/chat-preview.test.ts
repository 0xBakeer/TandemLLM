// The preview pane's target (SRV-511): the newest running server, the servers that start listening
// while the panes are open (and only those), a previewed page's report, and the pane's own wiring.
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { freshServers, newestServer, portPane, readReport, reportWords } from '../src/chat/preview';

const vite = { port: 5173, cwd: '/workspace/pagoda-garden', started: 200, command: 'node vite', http: true };
const preview = { port: 4173, cwd: '/workspace/pagoda-garden', started: 100, command: 'node vite preview', http: true };
const db = { port: 5432, cwd: '/', started: 900, http: false };

describe('the server a preview shows', () => {
  it('is the newest one that answers HTTP', () => {
    expect(newestServer([preview, vite, db])).toBe(vite);
    expect(newestServer([db])).toBeNull();
    expect(newestServer([])).toBeNull();
    expect(portPane(vite)).toEqual({ kind: 'preview', ref: 'port:5173', title: 'port 5173' });
  });
});

describe('a server that starts listening', () => {
  it('opens once; what listened at the first look, a database, and a port seen before do not', () => {
    let s = freshServers(null, [preview]);
    expect(s.fresh).toEqual([]);
    s = freshServers(s.seen, [preview, vite, db]);
    expect(s.fresh).toEqual([vite]);
    s = freshServers(s.seen, [preview, vite]);
    expect(s.fresh).toEqual([]);
  });
  it('opens again when its port went away and came back (a restarted dev server)', () => {
    let s = freshServers(null, [vite]);
    s = freshServers(s.seen, []);
    s = freshServers(s.seen, [vite]);
    expect(s.fresh).toEqual([vite]);
  });
});

describe("a previewed page's report", () => {
  const msg = { tandemPreview: 1, url: 'https://tandem.example/v1/chat-app/conversations/C/sandbox/ports/5173/', title: 'Pagoda', errors: ['TypeError: x'], failed: ['script https://x/src/main.js'], elements: 4, text: 0, canvas: 1 };
  it('is read only from the shim, with its lists capped', () => {
    expect(readReport({ type: 'other' })).toBeNull();
    expect(readReport('tandemPreview')).toBeNull();
    expect(readReport(null)).toBeNull();
    const r = readReport({ ...msg, errors: Array.from({ length: 30 }, (_, i) => `e${i}`), elements: 'x' });
    expect(r?.errors).toHaveLength(20);
    expect(r?.elements).toBe(0);
    expect(readReport(msg)).toEqual({ url: msg.url, title: 'Pagoda', errors: ['TypeError: x'], failed: ['script https://x/src/main.js'], elements: 4, text: 0, canvas: 1 });
  });
  it('says in the bar what went wrong, and nothing when it loaded clean', () => {
    expect(reportWords(readReport(msg))).toBe('1 file failed to load, 1 error');
    expect(reportWords(readReport({ ...msg, errors: [], failed: [] }))).toBe('');
    expect(reportWords(readReport({ ...msg, errors: [], failed: [], elements: 0, canvas: 0 }))).toBe('the page is empty');
    expect(reportWords(null)).toBe('');
  });
});

describe("the preview pane's wiring", () => {
  const src = readFileSync(new URL('../src/chat/panes.ts', import.meta.url), 'utf8');
  it('takes reports only from its own iframe and passes them to chatd', () => {
    expect(src).toMatch(/e\.source !== frame\.contentWindow/);
    expect(src).toMatch(/previewReport\(this\.convId, \{ ref: p\.ref, \.\.\.r \}\)/);
  });
  it('never gives the iframe allow-same-origin (CONTRACT 5)', () => {
    expect(src).toMatch(/sandbox="allow-scripts allow-forms allow-popups allow-modals"/);
    expect(src).not.toMatch(/sandbox="[^"]*allow-same-origin/);
  });
  it('offers the plus menu the newest running server first', () => {
    const pick = src.slice(src.indexOf('private async pick('), src.indexOf('private async newestPage('));
    expect(pick.indexOf('sandboxPorts')).toBeGreaterThan(0);
    expect(pick.indexOf('sandboxPorts')).toBeLessThan(pick.indexOf('latestPage'));
  });
});
