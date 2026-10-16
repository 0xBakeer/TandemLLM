// The preview pane's target (SRV-511): which server it shows, when a new one opens on its own, and
// the report a previewed page posts (chatd/preview.py's shim). The rules, kept simple on purpose:
//  1. A preview that cannot show what it was asked for shows the server that runs: a port that does
//     not listen goes to the one that does, a bundler project's index.html to its dev server
//     (chatd resolves both and says why in `note`). With nothing to show, the pane says why, offers
//     to start the project's dev server, and opens the server as soon as it listens.
//  2. A server that starts listening while the chat's panes are open opens in a Preview tab, once
//     per port (closing the tab keeps it closed until that port goes away and comes back). A port
//     that already listened when the panes opened does not pop up; the plus menu's Preview opens it.
//  3. The plus menu's Preview opens the newest running server, else the last page the chat made.

import type { ListeningPort, PreviewReport } from './backend';

/** how often the panes ask what listens while they are open, and a waiting pane its target */
export const PORT_POLL_MS = 4_000;
export const WAIT_POLL_MS = 2_500;

/** the newest server that answers HTTP, null when none */
export function newestServer(ports: ListeningPort[]): ListeningPort | null {
  let best: ListeningPort | null = null;
  for (const p of ports) {
    if (p.http === false) continue;
    if (!best || (p.started ?? 0) > (best.started ?? 0) || ((p.started ?? 0) === (best.started ?? 0) && p.port > best.port)) best = p;
  }
  return best;
}

/** The servers that started listening since the last look. `seen` null: the first look, which only
 *  learns what already listens. A port that stopped listening is forgotten, so it opens again when
 *  it comes back. */
export function freshServers(seen: Set<number> | null, ports: ListeningPort[]): { seen: Set<number>; fresh: ListeningPort[] } {
  const now = new Set(ports.filter((p) => p.http !== false).map((p) => p.port));
  if (!seen) return { seen: now, fresh: [] };
  return { seen: now, fresh: ports.filter((p) => p.http !== false && !seen.has(p.port)) };
}

/** the pane of a port in the sandbox */
export function portPane(p: { port: number }): { kind: 'preview'; ref: string; title: string } {
  return { kind: 'preview', ref: `port:${p.port}`, title: `port ${p.port}` };
}

/** a previewed page's report (window message), checked; null for any other message */
export function readReport(data: unknown): Omit<PreviewReport, 'ref'> | null {
  if (!data || typeof data !== 'object' || (data as { tandemPreview?: unknown }).tandemPreview !== 1) return null;
  const d = data as Record<string, unknown>;
  const strs = (v: unknown) => (Array.isArray(v) ? v.slice(0, 20).map((x) => String(x).slice(0, 300)) : []);
  const num = (v: unknown) => (typeof v === 'number' && Number.isFinite(v) ? v : 0);
  return { url: String(d.url ?? '').slice(0, 500), title: String(d.title ?? '').slice(0, 200), errors: strs(d.errors), failed: strs(d.failed), elements: num(d.elements), text: num(d.text), canvas: num(d.canvas) };
}

/** the pane bar's one-line state of a report: "2 errors" and the like, '' when it loaded clean */
export function reportWords(r: Omit<PreviewReport, 'ref'> | null): string {
  if (!r) return '';
  const parts: string[] = [];
  if (r.failed.length) parts.push(`${r.failed.length} file${r.failed.length === 1 ? '' : 's'} failed to load`);
  if (r.errors.length) parts.push(`${r.errors.length} error${r.errors.length === 1 ? '' : 's'}`);
  if (!parts.length && !r.elements && !r.text && !r.canvas) parts.push('the page is empty');
  return parts.join(', ');
}
