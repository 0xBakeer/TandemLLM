// the phone layout's decisions (src/lib/phone.ts): the bar's split, the strip's edge state, and
// what counts as typing
import { describe, expect, it } from 'vitest';
import { isTypingTarget, splitTabs, stripEdge, type TabDef } from '../src/lib/phone';

const TABS: TabDef[] = [
  { view: 'chat', label: 'Chat' },
  { view: 'usage', label: 'Usage' },
  { view: 'performance', label: 'Performance' },
  { view: 'dev', label: 'Dev' },
  { view: 'playground', label: 'Playground' },
  { view: 'conductor', label: 'Conductor' },
  { view: 'backends', label: 'Backends' },
  { view: 'system', label: 'System' },
  { view: 'setup', label: 'Setup' },
];
const PRIMARY = ['chat', 'usage', 'performance'] as const;

describe('splitTabs', () => {
  it('keeps the primary views in the bar, in the shell order, and the rest behind More', () => {
    const s = splitTabs(TABS, [...PRIMARY], 'chat');
    expect(s.bar.map((t) => t.view)).toEqual(['chat', 'usage', 'performance']);
    expect(s.more.map((t) => t.view)).toEqual(['dev', 'playground', 'conductor', 'backends', 'system', 'setup']);
    expect(s.moreShows).toBeNull();
  });
  it('the More slot shows the active view when that view is behind More', () => {
    expect(splitTabs(TABS, [...PRIMARY], 'setup').moreShows).toEqual({ view: 'setup', label: 'Setup' });
    expect(splitTabs(TABS, [...PRIMARY], 'dev').moreShows?.label).toBe('Dev');
  });
  it('a primary order that differs from the shell order follows the shell', () => {
    const s = splitTabs(TABS, ['performance', 'chat'], 'usage');
    expect(s.bar.map((t) => t.view)).toEqual(['chat', 'performance']);
    expect(s.moreShows?.view).toBe('usage');
  });
});

describe('stripEdge', () => {
  it('nothing to scroll: no edge', () => {
    expect(stripEdge(0, 300, 300)).toBe('none');
    expect(stripEdge(0, 300, 320)).toBe('none');
  });
  it('at the start, in the middle, at the end', () => {
    expect(stripEdge(0, 480, 300)).toBe('start');
    expect(stripEdge(1, 480, 300)).toBe('start');
    expect(stripEdge(90, 480, 300)).toBe('mid');
    expect(stripEdge(180, 480, 300)).toBe('end');
    expect(stripEdge(179.5, 480, 300)).toBe('end');
  });
});

describe('isTypingTarget', () => {
  // no DOM in this suite: the shapes the function reads, as the browser would hand them over
  const el = (tagName: string, extra: Record<string, unknown> = {}) => ({ tagName, readOnly: false, isContentEditable: false, ...extra }) as unknown as Element;
  it('a textarea and text-like inputs type', () => {
    expect(isTypingTarget(el('TEXTAREA'))).toBe(true);
    expect(isTypingTarget(el('INPUT', { type: 'text' }))).toBe(true);
    expect(isTypingTarget(el('INPUT', { type: '' }))).toBe(true);
    expect(isTypingTarget(el('INPUT', { type: 'search' }))).toBe(true);
    expect(isTypingTarget(el('INPUT', { type: 'number' }))).toBe(true);
    expect(isTypingTarget(el('DIV', { isContentEditable: true }))).toBe(true);
  });
  it('checkboxes, files, buttons and nothing do not', () => {
    expect(isTypingTarget(el('INPUT', { type: 'checkbox' }))).toBe(false);
    expect(isTypingTarget(el('INPUT', { type: 'file' }))).toBe(false);
    expect(isTypingTarget(el('INPUT', { type: 'range' }))).toBe(false);
    expect(isTypingTarget(el('BUTTON'))).toBe(false);
    expect(isTypingTarget(el('DIV'))).toBe(false);
    expect(isTypingTarget(null)).toBe(false);
  });
  it('a read-only field brings no keyboard', () => {
    expect(isTypingTarget(el('TEXTAREA', { readOnly: true }))).toBe(false);
    expect(isTypingTarget(el('INPUT', { type: 'text', readOnly: true }))).toBe(false);
  });
});
