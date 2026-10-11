// A small terminal screen in the xterm style: lines of styled runs, a block cursor, the 16 basic
// SGR colours, bold and dim, CR/LF, backspace, clear. Pure: feed() takes bytes as text, lines()
// gives what to draw. Enough for a tool's output and a shell; a real PTY would use xterm.js.

export interface Run {
  text: string;
  fg: number | null; // 0..15
  bold: boolean;
  dim: boolean;
}

export interface Line {
  runs: Run[];
}

export class TerminalScreen {
  readonly lines: Line[] = [{ runs: [] }];
  private col = 0;
  private fg: number | null = null;
  private bold = false;
  private dim = false;
  private pending = '';
  readonly maxLines: number;

  constructor(maxLines = 2000) {
    this.maxLines = maxLines;
  }

  feed(data: string): void {
    let s = this.pending + data;
    this.pending = '';
    let i = 0;
    while (i < s.length) {
      const ch = s[i];
      if (ch === '\x1b') {
        const m = /^\x1b\[([0-9;?]*)([A-Za-z])/.exec(s.slice(i));
        if (!m) {
          if (s.length - i < 12) {
            this.pending = s.slice(i);
            return;
          }
          i++;
          continue;
        }
        this.csi(m[1], m[2]);
        i += m[0].length;
        continue;
      }
      if (ch === '\r') {
        this.col = 0;
        i++;
        continue;
      }
      if (ch === '\n') {
        this.newline();
        i++;
        continue;
      }
      if (ch === '\b' || ch === '\x7f') {
        this.backspace();
        i++;
        continue;
      }
      if (ch === '\t') {
        this.put(' '.repeat(8 - (this.col % 8)));
        i++;
        continue;
      }
      if (ch < ' ') {
        i++;
        continue;
      }
      // a run of plain text
      let j = i + 1;
      while (j < s.length && s[j] >= ' ' && s[j] !== '\x1b') j++;
      this.put(s.slice(i, j));
      i = j;
    }
    s = '';
  }

  private csi(params: string, cmd: string): void {
    if (cmd === 'm') {
      const ps = params === '' ? [0] : params.split(';').map((x) => Number(x) || 0);
      for (const p of ps) {
        if (p === 0) {
          this.fg = null;
          this.bold = false;
          this.dim = false;
        } else if (p === 1) this.bold = true;
        else if (p === 2) this.dim = true;
        else if (p === 22) {
          this.bold = false;
          this.dim = false;
        } else if (p === 39) this.fg = null;
        else if (p >= 30 && p <= 37) this.fg = p - 30;
        else if (p >= 90 && p <= 97) this.fg = p - 90 + 8;
      }
    } else if (cmd === 'J') {
      this.lines.splice(0, this.lines.length, { runs: [] });
      this.col = 0;
    } else if (cmd === 'K') {
      const line = this.cur();
      line.runs = cut(line.runs, this.col);
    } else if (cmd === 'H') {
      this.col = 0;
    }
  }

  private cur(): Line {
    return this.lines[this.lines.length - 1];
  }

  private newline(): void {
    this.lines.push({ runs: [] });
    this.col = 0;
    if (this.lines.length > this.maxLines) this.lines.splice(0, this.lines.length - this.maxLines);
  }

  private backspace(): void {
    if (this.col > 0) this.col--;
  }

  private put(text: string): void {
    const line = this.cur();
    const width = lineWidth(line);
    if (this.col < width) {
      // overwrite from col (a CR then text, or a backspace then a space)
      line.runs = cut(line.runs, this.col);
    }
    const last = line.runs[line.runs.length - 1];
    if (last && last.fg === this.fg && last.bold === this.bold && last.dim === this.dim) last.text += text;
    else line.runs.push({ text, fg: this.fg, bold: this.bold, dim: this.dim });
    this.col += text.length;
  }

  /** the cursor's line index and column */
  get cursor(): { line: number; col: number } {
    return { line: this.lines.length - 1, col: this.col };
  }

  get text(): string {
    return this.lines.map((l) => l.runs.map((r) => r.text).join('')).join('\n');
  }
}

function lineWidth(l: Line): number {
  return l.runs.reduce((a, r) => a + r.text.length, 0);
}

function cut(runs: Run[], col: number): Run[] {
  const out: Run[] = [];
  let x = 0;
  for (const r of runs) {
    if (x + r.text.length <= col) {
      out.push(r);
      x += r.text.length;
    } else if (x < col) {
      out.push({ ...r, text: r.text.slice(0, col - x) });
      x = col;
    }
  }
  return out;
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
