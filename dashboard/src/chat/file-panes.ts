// Which pane shows a file in the sandbox, and which page the plus menu's Preview opens. A tool call
// that wrote or edited a page carries no pane of its own (only `present` and `render_page` do), so
// its card, the Files pane and the plus menu derive one from the path here.

import type { Message, PaneRef, ToolCallPart } from './types';

const PREVIEW_EXT = /\.(html?|svg)$/i;
const MARKDOWN_EXT = /\.(md|markdown)$/i;
const WRITERS = new Set(['write_file', 'edit_file']);

/** a sandbox path as chatd shows it: relative paths sit under /workspace */
export function sandboxPath(p: string): string {
  const s = p.trim();
  if (s.startsWith('/')) return s;
  return `/workspace/${s.replace(/^\.\//, '')}`;
}

/** the pane for a file: a page or an SVG in the preview, Markdown rendered; null for the rest */
export function filePane(path: string): PaneRef | null {
  const p = sandboxPath(path);
  const title = p.split('/').pop() || p;
  if (PREVIEW_EXT.test(p)) return { kind: 'preview', ref: `file:${p}`, title };
  if (MARKDOWN_EXT.test(p)) return { kind: 'markdown', ref: `file:${p}`, title };
  return null;
}

function inputPath(t: ToolCallPart): string | null {
  try {
    const a = JSON.parse(t.input || '{}') as { path?: unknown };
    return typeof a.path === 'string' && a.path.trim() ? a.path : null;
  } catch {
    return null;
  }
}

/** a call's pane: its own, else the page or Markdown file a finished write or edit left */
export function callPane(t: ToolCallPart): PaneRef | null {
  if (t.pane) return t.pane;
  if (!WRITERS.has(t.name) || t.status !== 'done' || t.output?.is_error) return null;
  const path = inputPath(t);
  return path ? filePane(path) : null;
}

/** the last page (HTML or SVG) a finished tool call of the thread wrote, edited or presented */
export function latestPage(messages: Message[]): string | null {
  for (let i = messages.length - 1; i >= 0; i--) {
    const parts = messages[i].parts;
    for (let j = parts.length - 1; j >= 0; j--) {
      const t = parts[j];
      if (t.type !== 'tool_call') continue;
      const pane = callPane(t);
      if (pane?.kind === 'preview' && pane.ref.startsWith('file:')) return pane.ref.slice(5);
    }
  }
  return null;
}
