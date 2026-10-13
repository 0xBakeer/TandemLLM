// The phone layout's decisions, kept pure so they can be tested: which views sit in the bottom
// bar and which go behind More, what counts as typing (the bar steps aside then), and where a
// scrolling strip of pills stands so its edges can fade. The shell (app.ts) and the composer
// read these.

import type { View } from './router';

export interface TabDef {
  view: View;
  label: string;
}

export interface TabSplit {
  /** the slots in the bar, in order */
  bar: TabDef[];
  /** the views behind More, in the shell's order */
  more: TabDef[];
  /** the More slot shows this view's icon and label when the active view is behind More, so the
   *  page you are on is never invisible (the setup guide in setup mode lands here too) */
  moreShows: TabDef | null;
}

/** The phone's bottom bar: the primary views in the shell's order, the rest behind More. */
export function splitTabs(tabs: TabDef[], primary: View[], active: View): TabSplit {
  const bar = tabs.filter((t) => primary.includes(t.view));
  const more = tabs.filter((t) => !primary.includes(t.view));
  return { bar, more, moreShows: more.find((t) => t.view === active) ?? null };
}

export type Edge = 'none' | 'start' | 'mid' | 'end';

/** Where a horizontal scroller stands: nothing to scroll, at the start, in the middle, at the end.
 *  The strip fades the edge that has more behind it. */
export function stripEdge(scrollLeft: number, scrollWidth: number, clientWidth: number): Edge {
  if (scrollWidth <= clientWidth + 1) return 'none';
  if (scrollLeft <= 1) return 'start';
  if (scrollLeft + clientWidth >= scrollWidth - 1) return 'end';
  return 'mid';
}

const TYPING_INPUTS = new Set(['text', 'search', 'url', 'email', 'tel', 'password', 'number']);

/** True for an element that brings up a keyboard: a textarea, a text-like input, or editable text. */
export function isTypingTarget(el: Element | null): boolean {
  if (!el) return false;
  const tag = el.tagName;
  if (tag === 'TEXTAREA') return !(el as HTMLTextAreaElement).readOnly;
  if (tag === 'INPUT') {
    const inp = el as HTMLInputElement;
    return TYPING_INPUTS.has((inp.type || 'text').toLowerCase()) && !inp.readOnly;
  }
  return (el as HTMLElement).isContentEditable === true;
}
