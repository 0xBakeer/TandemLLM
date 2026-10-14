// VIS-159 (QA VIS-160): media files in the Chat tab. The pane a video, an audio file, a PDF or a
// binary opens in, the view route's address, the present card's player, what <qse-media-view>
// draws per kind, the codec note, and the player's own chunk.
import { readFileSync, readdirSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { MEDIA_KINDS, callMedia, filePane, mediaKindOf, viewUrl } from '../src/chat/file-panes';
import { codecsIn, failedWords, isIOS, mediaBody, type MediaView } from '../src/chat/media-body';
import type { ToolCallPart } from '../src/chat/types';

// a template's markup without a DOM: its static parts and its values
function flat(t: unknown): string {
  if (t == null || typeof t === 'symbol' || typeof t === 'boolean' || typeof t === 'function') return '';
  if (Array.isArray(t)) return t.map(flat).join('');
  if (typeof t === 'object' && 'strings' in (t as object)) {
    const r = t as { strings: readonly string[]; values: unknown[] };
    return r.strings.map((x, i) => x + (i < r.values.length ? flat(r.values[i]) : '')).join('');
  }
  return String(t);
}
const view = (v: Partial<MediaView>) => flat(mediaBody({ kind: 'video', src: '/v/a.mp4', name: 'a.mp4', size: 1_048_576, ios: false, ...v }));
const present = (pane: unknown, extra: Partial<ToolCallPart> = {}): ToolCallPart => ({
  type: 'tool_call', id: 'c1', name: 'present', input: '{"kind":"preview","path":"lfm/x.mp4"}',
  output: { content: 'opened a video pane for the user', is_error: false, bytes: 30 } as ToolCallPart['output'],
  status: 'done', started_at: null, ended_at: null, pane: pane as ToolCallPart['pane'], ...extra,
});

describe('the pane a media file opens in', () => {
  it('plays videos and audio and shows PDFs', () => {
    expect(filePane('/workspace/lfm25-explainer/lfm25-24b-a2b-explainer.mp4')).toEqual({ kind: 'video', ref: 'file:/workspace/lfm25-explainer/lfm25-24b-a2b-explainer.mp4', title: 'lfm25-24b-a2b-explainer.mp4' });
    for (const p of ['a.webm', 'b.MOV', 'c.m4v']) expect(mediaKindOf(p)).toBe('video');
    for (const p of ['a.wav', 'b.mp3', 'c.m4a', 'd.ogg', 'e.flac']) expect(filePane(p)?.kind).toBe('audio');
    expect(filePane('report.pdf')?.kind).toBe('pdf');
    expect(filePane('blob.bin')).toBeNull();
    expect(filePane('app.ts')).toBeNull();
    expect([...MEDIA_KINDS].sort()).toEqual(['audio', 'binary', 'pdf', 'video']);
  });
  it('addresses the view route, one encoded segment per directory', () => {
    expect(viewUrl('01ABC', '/workspace/clips/a b#1.mp4')).toBe('/v1/chat-app/conversations/01ABC/sandbox/view/clips/a%20b%231.mp4');
    expect(viewUrl('01ABC', 'x.wav')).toBe('/v1/chat-app/conversations/01ABC/sandbox/view/x.wav');
  });
});

describe('the present card', () => {
  it('plays the video a call presented', () => {
    const m = callMedia(present({ kind: 'video', ref: 'file:/workspace/lfm/x.mp4', title: null }), 'cid');
    expect(m).toEqual({ kind: 'video', src: '/v1/chat-app/conversations/cid/sandbox/view/lfm/x.mp4', name: 'x.mp4' });
    expect(callMedia(present({ kind: 'audio', ref: 'file:/workspace/s.wav', title: null }), 'cid')?.kind).toBe('audio');
  });
  it('has no player for a page, a PDF or a running call without a pane', () => {
    expect(callMedia(present({ kind: 'preview', ref: 'file:/workspace/i.html', title: null }), 'cid')).toBeNull();
    expect(callMedia(present({ kind: 'pdf', ref: 'file:/workspace/d.pdf', title: null }), 'cid')).toBeNull();
    expect(callMedia(present(null, { status: 'running' }), 'cid')).toBeNull();
  });
  it('is drawn in the card, beside the pane button', () => {
    const src = readFileSync(new URL('../src/chat/thread.ts', import.meta.url), 'utf8');
    const card = src.slice(src.indexOf('private renderCall('), src.indexOf('private renderCall(') + 4000);
    expect(card).toContain('callMedia(t, c.id)');
    expect(card).toMatch(/<qse-media-view kind=\$\{media\.kind\} src=\$\{media\.src\}/);
  });
});

describe('what the media view draws', () => {
  it('a video: an inline player that loads only the metadata first', () => {
    const v = view({});
    expect(v).toMatch(/<video class="ch-media-video" controls playsinline preload="metadata" src=\/v\/a\.mp4/);
    expect(v).toContain('Open');
    expect(v).toMatch(/download=a\.mp4>.*Download/s);
    expect(v).toContain('1.0 MB');
  });
  it('audio: an audio player', () => {
    expect(view({ kind: 'audio', src: '/v/s.wav', name: 's.wav' })).toMatch(/<audio class="ch-media-audio" controls preload="metadata" src=\/v\/s\.wav/);
  });
  it('a PDF: the browser viewer with a link inside, a link alone on iOS', () => {
    const pdf = view({ kind: 'pdf', src: '/v/d.pdf', name: 'd.pdf' });
    expect(pdf).toMatch(/<object class="ch-pdf-frame" type="application\/pdf" data=\/v\/d\.pdf/);
    expect(pdf).toMatch(/<object[^]*<a href=\/v\/d\.pdf[^]*<\/object>/);
    const ios = view({ kind: 'pdf', src: '/v/d.pdf', name: 'd.pdf', ios: true });
    expect(ios).not.toContain('<object');
    expect(ios).toContain('Open d.pdf');
  });
  it('a binary: its size and Download, no player', () => {
    const b = view({ kind: 'binary', src: '/v/x.bin', name: 'x.bin', size: 2_500_000 });
    expect(b).not.toMatch(/<video|<audio|<object/);
    expect(b).toContain('2.5 MB, not a file this pane can show');
    expect(b).toContain('Download');
  });
  it('a file the browser cannot play: the note with the codec and Download', () => {
    const head = new TextEncoder().encode('....ftypqt  ....moov....trak....stsd....hvc1....mp4a');
    expect(codecsIn(head)).toEqual(['HEVC (H.265)', 'AAC']);
    expect(codecsIn(new Uint8Array(64))).toEqual([]);
    const f = view({ failed: failedWords(codecsIn(head)) });
    expect(f).not.toContain('<video');
    expect(f).toContain('cannot play this file (HEVC (H.265), AAC)');
    expect(f).toContain('Download');
  });
  it('a tool card: compact, without Open', () => {
    const c = view({ compact: true });
    expect(c).toContain('is-compact');
    expect(c).not.toContain('Open');
  });
  it('knows iOS, the iPad that says it is a Mac included', () => {
    expect(isIOS({ userAgent: 'Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X)' })).toBe(true);
    expect(isIOS({ userAgent: 'Mozilla/5.0 (Macintosh)', platform: 'MacIntel', maxTouchPoints: 5 })).toBe(true);
    expect(isIOS({ userAgent: 'Mozilla/5.0 (Macintosh)', platform: 'MacIntel', maxTouchPoints: 0 })).toBe(false);
  });
});

describe('the panes', () => {
  const src = readFileSync(new URL('../src/chat/panes.ts', import.meta.url), 'utf8');
  it('draw a video, an audio file, a binary and a sandbox PDF in the media view', () => {
    expect(src).toMatch(/case 'video':\s*case 'audio':\s*case 'binary':\s*return this\.renderMedia\(p\)/);
    expect(src).toContain("return p.ref.startsWith('file:') ? this.renderMedia(p) : this.renderPdf(p);");
    expect(src).toMatch(/renderFileView[^]*<qse-media-view kind=/);
  });
  it('load the media view as its own chunk, never statically', () => {
    const dir = new URL('../src/', import.meta.url).pathname;
    const files: string[] = [];
    const walk = (d: string) => {
      for (const e of readdirSync(d, { withFileTypes: true })) {
        if (e.isDirectory()) walk(d + e.name + '/');
        else if (e.name.endsWith('.ts')) files.push(d + e.name);
      }
    };
    walk(dir);
    const statics = files.filter((f) => /^import [^;]*['"]\.\/media-(view|body)['"]/m.test(readFileSync(f, 'utf8')) && !f.endsWith('media-view.ts'));
    expect(statics).toEqual([]);
    expect(readFileSync(dir + 'chat/file-panes.ts', 'utf8')).toContain("import('./media-view')");
  });
});
