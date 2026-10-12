// the terminal pane's pure parts: Ctrl from the phone's key row, the colours from the tokens
import { describe, expect, it } from 'vitest';
import { ctrlOf, themeFromTokens } from '../src/chat/terminal';

describe('the terminal pane', () => {
  it('sends the control code for the next key after Ctrl', () => {
    expect(ctrlOf('c')).toBe('\x03');
    expect(ctrlOf('C')).toBe('\x03');
    expect(ctrlOf('d')).toBe('\x04');
    expect(ctrlOf('[')).toBe('\x1b');
    expect(ctrlOf('1')).toBe('1');
    expect(ctrlOf('ab')).toBe('ab');
  });
  it('takes its colours from the dashboard tokens', () => {
    const t = themeFromTokens((n) => `var(${n})`);
    expect(t.background).toBe('var(--bg-inset)');
    expect(t.cursor).toBe('var(--orange-hi)');
    expect(t.red).toBe('var(--err)');
    expect(t.brightBlue).toBe('var(--cobalt-hi)');
    expect(t.brightWhite).toBe('var(--ink)');
  });
});
