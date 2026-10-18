// Conversation switch (deep audit 0.11.3): nothing in the context strip carries over.
// Pane ids are per pane KIND, so chat B's terminal pane matched chat A's still-open
// session -- whose sandbox is A's -- and B's keystrokes would run in A.
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { foreignScreens } from '../src/chat/panes';

describe('foreignScreens', () => {
  const mk = (conv: string) => ({ conv });
  it('keeps only the current conversation\'s screens', () => {
    const screens = new Map([['terminal:shell', mk('A')], ['terminal:server-3000', mk('A')]]);
    expect(foreignScreens(screens, 'B')).toEqual(['terminal:shell', 'terminal:server-3000']);
    expect(foreignScreens(screens, 'A')).toEqual([]);
  });
  it('an empty room has nothing to drop', () => {
    expect(foreignScreens(new Map(), 'A')).toEqual([]);
  });
});

describe('the strip drops the old conversation on switch', () => {
  const src = readFileSync(new URL('../src/chat/panes.ts', import.meta.url), 'utf8');

  it('updated() notices the switch first thing', () => {
    expect(src).toMatch(/updated\(changed[\s\S]{0,220}this\.convId !== this\.lastConv[\s\S]{0,80}dropConversation\(\)/);
  });

  it('a dropped terminal closes its session and disposes its xterm', () => {
    const drop = src.slice(src.indexOf('private dropConversation'), src.indexOf('private async loadArtifact'));
    expect(drop).toContain('e.session?.close();');
    expect(drop).toContain('e.term?.dispose();');
    // and the maps that hold the other per-conversation views
    for (const m of ['this.files = new Map();', 'this.artifacts = new Map();', 'this.reports = new Map();']) {
      expect(drop).toContain(m);
    }
  });

  it('a read whose conversation moved on mid-fetch is not kept', () => {
    for (const fn of ['loadArtifact', 'loadDir', 'openFile', 'startServer']) {
      const body = src.slice(src.indexOf(`private async ${fn}(`), src.indexOf(`private async ${fn}(`) + 1200);
      expect(body).toContain('if (conv !== this.convId) return;');
    }
  });
});
