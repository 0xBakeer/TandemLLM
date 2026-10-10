// AU1 (2026-10-06, the owner: "remove the password prompt from the ui from everywhere"): the
// dashboard's API layer sends no token and never asks for one. The conductor's `write` (every edit,
// and every memory read) carries no Authorization header; a 401 -- only from a server with an admin
// password set -- is a plain `unauthorized` error with words, not a request for a token; a token an
// older build kept in this browser is forgotten.
import { afterEach, describe, expect, it, vi } from 'vitest';
import { ApiError } from '../src/api/client';
import { conductor, forgetStoredToken } from '../src/api/conductor';

type Call = { url: string; init: RequestInit };

function stubFetch(status: number, body: unknown): Call[] {
  const calls: Call[] = [];
  vi.stubGlobal('fetch', async (url: string, init: RequestInit = {}) => {
    calls.push({ url, init });
    return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
  });
  return calls;
}

function headersOf(c: Call): Record<string, string> {
  return Object.fromEntries(Object.entries((c.init.headers ?? {}) as Record<string, string>).map(([k, v]) => [k.toLowerCase(), v]));
}

afterEach(() => vi.unstubAllGlobals());

describe('AU1: no token anywhere', () => {
  it('a memory read and a note edit go out without an Authorization header', async () => {
    const calls = stubFetch(200, { notes: [], counts: {} });
    await conductor.notes();
    await conductor.noteEdit('n1', { body: 'x' });
    await conductor.memorySessions();
    expect(calls).toHaveLength(3);
    for (const c of calls) expect(headersOf(c).authorization).toBeUndefined();
    expect(calls[1].init.method).toBe('PUT');
  });

  it('the exports (raw fetches) send no Authorization header either', async () => {
    const calls = stubFetch(200, {});
    await conductor.notesExport();
    await conductor.memoryExport('k', 'md');
    for (const c of calls) expect(headersOf(c).authorization).toBeUndefined();
  });

  it('a 401 (a server with a password set) is a plain unauthorized error, not a token request', async () => {
    stubFetch(401, { error: { type: 'unauthorized', message: 'a bearer token or the dashboard session is needed' } });
    const e = await conductor.notes().catch((x) => x);
    expect(e).toBeInstanceOf(ApiError);
    expect(e.status).toBe(401);
    expect(e.type).toBe('unauthorized');
    expect(e.type).not.toBe('token_needed');
    expect(e.message).toContain('admin password set');
  });

  it('a token an older build stored is removed from both storages', () => {
    const mk = () => {
      const m = new Map<string, string>([['qse-conductor-token', 'old'], ['qse.theme', 'dark']]);
      return { getItem: (k: string) => m.get(k) ?? null, setItem: (k: string, v: string) => void m.set(k, v), removeItem: (k: string) => void m.delete(k), m };
    };
    const local = mk();
    const session = mk();
    vi.stubGlobal('localStorage', local);
    vi.stubGlobal('sessionStorage', session);
    forgetStoredToken();
    expect(local.m.has('qse-conductor-token')).toBe(false);
    expect(session.m.has('qse-conductor-token')).toBe(false);
    expect(local.m.get('qse.theme')).toBe('dark');
  });

  it('storage that throws is no error', () => {
    vi.stubGlobal('localStorage', { removeItem: () => { throw new Error('blocked'); } });
    expect(() => forgetStoredToken()).not.toThrow();
  });
});

describe('AU1: no password prompt in the source', () => {
  it('no view renders a token field, an unlock step or a sign-in screen', async () => {
    const { readdirSync, readFileSync, statSync } = await import('node:fs');
    const { join } = await import('node:path');
    const { fileURLToPath } = await import('node:url');
    const root = fileURLToPath(new URL('../src', import.meta.url));
    const files: string[] = [];
    const walk = (d: string) => {
      for (const f of readdirSync(d)) {
        const p = join(d, f);
        if (statSync(p).isDirectory()) walk(p);
        else if (p.endsWith('.ts')) files.push(p);
      }
    };
    walk(root);
    const bad: string[] = [];
    for (const f of files) {
      const text = readFileSync(f, 'utf8');
      const rel = f.slice(root.length + 1);
      // the Backends form's upstream API key (a provider's sk-..., not the dashboard's password) is
      // the one password-type input, plus the setup guide's Review step where the user may CHOOSE a
      // new admin password for an open box (D10, VIS-78): it sets one, it never asks for one
      if (/type="password"/.test(text) && rel !== join('views', 'backends.ts') && rel !== join('setup', 'view.ts')) bad.push(`${rel}: a password input`);
      for (const [re, what] of [
        [/Unlock editing|>Unlock<|'Unlock'/, 'an unlock step'],
        [/Admin token needed|Enter admin token|Enter the admin token/, 'a token prompt'],
        [/id="(cond-)?token"/, 'a token field'],
        [/Sign out|Open the dashboard/, 'a sign-in screen'],
        [/\bwithToken\b|\baskToken\b|\btokenState\b|token_needed/, 'the token plumbing'],
      ] as const) {
        if (re.test(text)) bad.push(`${rel}: ${what}`);
      }
    }
    expect(bad).toEqual([]);
  });
});
