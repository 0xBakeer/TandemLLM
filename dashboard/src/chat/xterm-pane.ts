// The terminal pane's screen: xterm.js (plan decision D13), loaded on demand by panes.ts so the
// dashboard's first load does not carry it. One Terminal per pane, opened once into its own
// element; the pane moves that element in and out of the page as tabs switch, so the scrollback
// and the cursor survive. Colours come from the dashboard's tokens (the 16 basic colours as the
// earlier screen had them), read again when the theme changes.

import { Terminal } from '@xterm/xterm';
import { FitAddon } from '@xterm/addon-fit';
import '@xterm/xterm/css/xterm.css';
import { ctrlOf, themeFromTokens } from './terminal';

export interface XtermOptions {
  /** a tool call's output: no input, no cursor */
  readOnly: boolean;
  onInput(data: string): void;
  onResize(cols: number, rows: number): void;
}

export interface XtermHandle {
  /** the element xterm drew into; the pane appends it where the screen goes */
  readonly el: HTMLElement;
  write(data: string | Uint8Array): void;
  /** fit to the element's box and tell the PTY; call after the element moved or resized */
  fit(): void;
  focus(): void;
  /** the phone's key row: the next key typed is sent with Ctrl */
  armCtrl(on: boolean): void;
  /** the screen's text, for tests and the copy button */
  text(): string;
  setTheme(): void;
  dispose(): void;
}

export function mountXterm(opts: XtermOptions): XtermHandle {
  const el = document.createElement('div');
  el.className = 'ch-xterm';
  const read = (name: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const term = new Terminal({
    cursorBlink: !matchMedia('(prefers-reduced-motion: reduce)').matches,
    cursorStyle: 'block',
    disableStdin: opts.readOnly,
    fontFamily: read('--font-mono') || 'ui-monospace, monospace',
    fontSize: 12.5,
    lineHeight: 1.2,
    scrollback: 5000,
    convertEol: opts.readOnly,
    theme: themeFromTokens(read),
    allowProposedApi: false,
  });
  const fitter = new FitAddon();
  term.loadAddon(fitter);
  let ctrl = false;
  term.onData((d) => {
    if (opts.readOnly) return;
    if (ctrl) {
      ctrl = false;
      d = ctrlOf(d);
      el.dispatchEvent(new CustomEvent('ch-ctrl-used', { bubbles: true }));
    }
    opts.onInput(d);
  });
  term.onResize(({ cols, rows }) => opts.onResize(cols, rows));
  let opened = false;
  return {
    el,
    write: (data) => term.write(data),
    fit() {
      if (!el.isConnected || !el.clientWidth) return;
      if (!opened) {
        term.open(el);
        opened = true;
      }
      try {
        fitter.fit();
      } catch {
        /* not laid out yet */
      }
    },
    focus: () => term.focus(),
    armCtrl: (on) => (ctrl = on),
    text() {
      const b = term.buffer.active;
      const out: string[] = [];
      for (let i = 0; i < b.length; i++) out.push(b.getLine(i)?.translateToString(true) ?? '');
      return out.join('\n').replace(/\n+$/, '');
    },
    setTheme: () => (term.options.theme = themeFromTokens(read)),
    dispose: () => term.dispose(),
  };
}
