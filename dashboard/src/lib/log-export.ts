// The Log panel's export (SRV-803). The engine builds the file (GET /v1/dashboard/logs/export: a
// build header, then every line its ring holds); the page fetches it, so a 401 or an engine that
// does not answer is an error in the panel and never a saved file, and hands the bytes to the
// browser to save under the name the engine chose.

export const LOGS_EXPORT = '/v1/dashboard/logs/export';

/** The engine's naming rule, for an answer that came without a name:
 *  `tandemllm-<version>-<time of export, UTC>.log`, in letters, digits, dot and dash only. */
export function exportName(version: string | null | undefined, now: Date = new Date()): string {
  const v = String(version ?? '').trim().replace(/[^A-Za-z0-9.-]/g, '-') || 'unknown';
  const t = now.toISOString().replace(/[-:]/g, '').replace(/\.\d+Z$/, 'Z');
  return `tandemllm-${v}-${t}.log`;
}

/** The file name of a `Content-Disposition` header, when it is one the naming rule allows. */
export function nameFrom(disposition: string | null | undefined): string | null {
  const m = /filename="?([^";]+)"?/i.exec(disposition ?? '');
  const name = m ? m[1].trim() : '';
  return /^[A-Za-z0-9.-]+$/.test(name) && !name.startsWith('.') ? name : null;
}

/** What `saveFile` needs of the browser; a test passes its own. */
export interface SaveEnv {
  document: Pick<Document, 'createElement'> & { body: { append(node: unknown): void } };
  URL: { createObjectURL(blob: Blob): string; revokeObjectURL(url: string): void };
  later?: (fn: () => void, ms: number) => unknown;
}

/** Whether this browser can save a file the page holds in memory (an `<a download>` on an object URL). */
export function canSave(env: Partial<SaveEnv> | undefined = globalThis as unknown as Partial<SaveEnv>): boolean {
  try {
    if (typeof env?.URL?.createObjectURL !== 'function' || typeof env.document?.createElement !== 'function') return false;
    return 'download' in env.document.createElement('a');
  } catch {
    return false;
  }
}

export class CannotSave extends Error {
  constructor() {
    super('this browser cannot save a file the page built');
  }
}

/** Hand `blob` to the browser as a download named `name`. Throws `CannotSave` where that is not possible. */
export function saveFile(blob: Blob, name: string, env: SaveEnv = globalThis as unknown as SaveEnv): void {
  if (!canSave(env)) throw new CannotSave();
  const url = env.URL.createObjectURL(blob);
  const a = env.document.createElement('a') as HTMLAnchorElement;
  a.href = url;
  a.download = name;
  a.rel = 'noopener';
  a.hidden = true;
  env.document.body.append(a);
  try {
    a.click();
  } finally {
    a.remove();
    // the download reads the object URL after the click returns; free it once it has had time to start
    (env.later ?? setTimeout)(() => env.URL.revokeObjectURL(url), 60_000);
  }
}
