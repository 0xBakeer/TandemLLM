// Pure helpers for the panes: the terminal's colours from the dashboard's tokens and the phone's
// Ctrl key (xterm-pane.ts uses them), and the Diff pane's parser.

import type { ITheme } from '@xterm/xterm';

const ANSI: (keyof ITheme)[] = ['black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white', 'brightBlack', 'brightRed', 'brightGreen', 'brightYellow', 'brightBlue', 'brightMagenta', 'brightCyan', 'brightWhite'];
// the tokens behind each basic colour; bright and plain share one, as before
const TOKENS = ['--ink-3', '--err', '--ok', '--warn-ink', '--cobalt-hi', '--series-violet', '--series-aqua', '--ink', '--ink-3', '--err', '--ok', '--warn-ink', '--cobalt-hi', '--series-violet', '--series-aqua', '--ink'];

export function themeFromTokens(read: (name: string) => string): ITheme {
  const t: ITheme = {
    background: read('--bg-inset'),
    foreground: read('--log-info') || read('--ink'),
    cursor: read('--orange-hi'),
    cursorAccent: read('--bg-inset'),
    selectionBackground: read('--cobalt-wash') || read('--cobalt-dim'),
  };
  ANSI.forEach((k, i) => ((t as Record<string, string>)[k] = read(TOKENS[i])));
  return t;
}

export function ctrlOf(data: string): string {
  if (data.length !== 1) return data;
  const c = data.toUpperCase().charCodeAt(0);
  return c >= 64 && c <= 95 ? String.fromCharCode(c - 64) : data;
}

/** a unified diff → rows with a kind, for the Diff pane */
export type DiffRow = { kind: 'file' | 'hunk' | 'add' | 'del' | 'ctx' | 'meta'; text: string; old?: number | null; new?: number | null };

export function parseDiff(text: string): DiffRow[] {
  const rows: DiffRow[] = [];
  let o = 0;
  let n = 0;
  for (const line of text.replace(/\r\n?/g, '\n').split('\n')) {
    if (line.startsWith('+++ ') || line.startsWith('--- ')) rows.push({ kind: 'file', text: line });
    else if (line.startsWith('@@')) {
      const m = /@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@/.exec(line);
      o = m ? Number(m[1]) : 0;
      n = m ? Number(m[2]) : 0;
      rows.push({ kind: 'hunk', text: line });
    } else if (line.startsWith('+')) rows.push({ kind: 'add', text: line.slice(1), old: null, new: n++ });
    else if (line.startsWith('-')) rows.push({ kind: 'del', text: line.slice(1), old: o++, new: null });
    else if (line.startsWith('diff ') || line.startsWith('index ')) rows.push({ kind: 'meta', text: line });
    else if (line === '' && rows.length && rows[rows.length - 1].kind !== 'ctx') continue;
    else rows.push({ kind: 'ctx', text: line.replace(/^ /, ''), old: o++, new: n++ });
  }
  return rows;
}

export function diffCounts(rows: DiffRow[]): { add: number; del: number; files: number } {
  return { add: rows.filter((r) => r.kind === 'add').length, del: rows.filter((r) => r.kind === 'del').length, files: rows.filter((r) => r.kind === 'file' && r.text.startsWith('+++')).length };
}
