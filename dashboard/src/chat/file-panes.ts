// Which pane shows a file in the sandbox, and which page the plus menu's Preview opens. A tool call
// that wrote or edited a page carries no pane of its own (only `present` and `render_page` do), so
// its card, the Files pane and the plus menu derive one from the path here.

import { apiBase } from '../api/client';
import type { Message, PaneKind, PaneRef, ToolCallPart } from './types';

const PREVIEW_EXT = /\.(html?|svg)$/i;
const MARKDOWN_EXT = /\.(md|markdown)$/i;
// VIS-159: a player or the browser's PDF viewer, from the view route (chatd/panes.py `kind_of`)
const VIDEO_EXT = /\.(mp4|m4v|webm|mov|ogv)$/i;
const AUDIO_EXT = /\.(mp3|m4a|wav|ogg|oga|opus|flac|aac)$/i;
const PDF_EXT = /\.pdf$/i;
/** the pane kinds <qse-media-view> draws */
export const MEDIA_KINDS: ReadonlySet<PaneKind> = new Set<PaneKind>(['video', 'audio', 'pdf', 'binary']);
const WRITERS = new Set(['write_file', 'edit_file']);

/** a sandbox path as chatd shows it: relative paths sit under /workspace */
export function sandboxPath(p: string): string {
  const s = p.trim();
  if (s.startsWith('/')) return s;
  return `/workspace/${s.replace(/^\.\//, '')}`;
}

/** the player a file opens in by its name: video, audio, pdf; null for the rest */
export function mediaKindOf(path: string): 'video' | 'audio' | 'pdf' | null {
  if (VIDEO_EXT.test(path)) return 'video';
  if (AUDIO_EXT.test(path)) return 'audio';
  if (PDF_EXT.test(path)) return 'pdf';
  return null;
}

/** the pane for a file: a page or an SVG in the preview, Markdown rendered, a video, an audio file
 *  or a PDF in its player; null for the rest */
export function filePane(path: string): PaneRef | null {
  const p = sandboxPath(path);
  const title = p.split('/').pop() || p;
  if (PREVIEW_EXT.test(p)) return { kind: 'preview', ref: `file:${p}`, title };
  if (MARKDOWN_EXT.test(p)) return { kind: 'markdown', ref: `file:${p}`, title };
  const media = mediaKindOf(p);
  if (media) return { kind: media, ref: `file:${p}`, title };
  return null;
}

/** a sandbox file's address on the view route (streamed, with byte ranges) */
export function viewUrl(convId: string, path: string): string {
  const rel = sandboxPath(path).replace(/^\/workspace\/?/, '');
  return `${apiBase()}/v1/chat-app/conversations/${encodeURIComponent(convId)}/sandbox/view/${rel.split('/').map(encodeURIComponent).join('/')}`;
}

/** what a tool card plays inline: a video or an audio file a call presented */
export function callMedia(t: ToolCallPart, convId: string): { kind: 'video' | 'audio'; src: string; name: string } | null {
  const pane = callPane(t);
  if (!pane || (pane.kind !== 'video' && pane.kind !== 'audio') || !pane.ref.startsWith('file:')) return null;
  const path = pane.ref.slice(5);
  return { kind: pane.kind, src: viewUrl(convId, path), name: path.split('/').pop() || path };
}

let mediaLoad: Promise<unknown> | null = null;
/** <qse-media-view>'s chunk, loaded once, the first time something shows a media file */
export function loadMediaView(): Promise<unknown> {
  mediaLoad ??= import('./media-view');
  return mediaLoad;
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
