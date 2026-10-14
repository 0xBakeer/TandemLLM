// The panes' pure parts (VIS-133, VIS-135): the pane a file opens in, the card of a tool call that
// wrote a page, the page the plus menu's Preview opens, and where the plus menu sits in the markup
// and the styles.
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { callPane, filePane, latestPage, sandboxPath } from '../src/chat/file-panes';
import type { Message, ToolCallPart } from '../src/chat/types';

const call = (name: string, input: unknown, extra: Partial<ToolCallPart> = {}): ToolCallPart => ({
  type: 'tool_call', id: `c-${name}-${Math.random()}`, name, input: JSON.stringify(input),
  output: { content: 'ok', is_error: false, bytes: 2, truncated: false, exit_code: null, images: [] } as unknown as ToolCallPart['output'],
  status: 'done', started_at: null, ended_at: null, ...extra,
});
const msg = (parts: ToolCallPart[]): Message => ({ id: 'm', role: 'assistant', parts, status: 'done' } as unknown as Message);

describe('the pane a file opens in', () => {
  it('puts relative paths under /workspace', () => {
    expect(sandboxPath('index.html')).toBe('/workspace/index.html');
    expect(sandboxPath('./site/a.html')).toBe('/workspace/site/a.html');
    expect(sandboxPath('/workspace/x.svg')).toBe('/workspace/x.svg');
  });
  it('previews pages and SVGs, renders Markdown, and has none for the rest', () => {
    expect(filePane('/workspace/index.html')).toEqual({ kind: 'preview', ref: 'file:/workspace/index.html', title: 'index.html' });
    expect(filePane('site/Logo.SVG')?.kind).toBe('preview');
    expect(filePane('notes.md')).toEqual({ kind: 'markdown', ref: 'file:/workspace/notes.md', title: 'notes.md' });
    expect(filePane('main.py')).toBeNull();
  });
});

describe("a tool call's card", () => {
  it('opens the page a finished write or edit left', () => {
    expect(callPane(call('write_file', { path: '/workspace/index.html', content: '<h1>x</h1>' }))?.ref).toBe('file:/workspace/index.html');
    expect(callPane(call('edit_file', { path: 'a.html', old: 'x', new: 'y' }))?.ref).toBe('file:/workspace/a.html');
  });
  it('keeps a pane the call brought itself', () => {
    const own = { kind: 'image' as const, ref: 'file:/workspace/.tandem/shot.png', title: 'screenshot' };
    expect(callPane(call('render_page', { path: 'a.html' }, { pane: own }))).toBe(own);
  });
  it('has none for a failed, running or unreadable call, or another file', () => {
    expect(callPane(call('write_file', { path: 'a.html', content: '' }, { status: 'failed' }))).toBeNull();
    expect(callPane(call('write_file', { path: 'a.html', content: '' }, { status: 'running' }))).toBeNull();
    expect(callPane(call('write_file', { path: 'a.html' }, { output: { content: 'no', is_error: true } as ToolCallPart['output'] }))).toBeNull();
    expect(callPane({ ...call('write_file', {}), input: '{"path": "a.ht' })).toBeNull();
    expect(callPane(call('write_file', { path: 'a.py', content: '' }))).toBeNull();
    expect(callPane(call('bash', { command: 'cat > a.html' }))).toBeNull();
  });
});

describe("the plus menu's Preview", () => {
  it('opens the last page the thread wrote, edited or presented', () => {
    const m1 = msg([call('write_file', { path: 'old.html', content: '' }), call('write_file', { path: 'notes.md', content: '' })]);
    const m2 = msg([call('edit_file', { path: 'site/new.html', old: 'a', new: 'b' }), call('bash', { command: 'ls' })]);
    expect(latestPage([m1, m2])).toBe('/workspace/site/new.html');
    expect(latestPage([m1])).toBe('/workspace/old.html');
    const shown = call('present', { kind: 'preview', path: 'p.html' }, { pane: { kind: 'preview', ref: 'file:/workspace/p.html', title: null } as never });
    expect(latestPage([m2, msg([shown])])).toBe('/workspace/p.html');
    expect(latestPage([msg([call('bash', { command: 'ls' })])])).toBeNull();
  });
});

describe("the plus menu's place", () => {
  const src = readFileSync(new URL('../src/chat/panes.ts', import.meta.url), 'utf8');
  const css = readFileSync(new URL('../src/styles/chat.css', import.meta.url), 'utf8');
  it('is drawn outside the tab strip, which scrolls sideways and clips what drops below it', () => {
    const strip = src.indexOf('<div class="ch-pane-tabs"');
    const body = src.indexOf('<div class="ch-pane-body"');
    const menu = src.indexOf('class="ch-pop ch-pane-menu"');
    // the strip's closing tag: the last one before the menu's own `${this.plus ? ... }` block
    const menuBlock = src.indexOf('${this.plus\n', strip);
    const stripEnd = src.lastIndexOf('</div>', menuBlock);
    expect(menuBlock).toBeGreaterThan(strip);
    expect(src.slice(strip, stripEnd)).not.toContain('ch-pane-menu');
    expect(strip).toBeGreaterThan(0);
    expect(menu).toBeGreaterThan(stripEnd);
    expect(menu).toBeLessThan(body);
    expect(css).toMatch(/\.ch-pane-tabs \{[^}]*overflow-x: auto/);
  });
  it('outranks the phone rule that pins every popover above the composer', () => {
    const rule = css.match(/\.ch-panes > \.ch-pane-menu\.ch-pop \{([^}]*)\}/);
    expect(rule).not.toBeNull();
    expect(rule![1]).toMatch(/bottom: auto/);
    expect(rule![1]).toMatch(/right: auto/);
    expect(css).toMatch(/\.ch-panes \{\s*position: relative;/);
  });
});
