// What <qse-media-view> draws for a file from the sandbox's view route (VIS-159): a video or audio
// player, a PDF in the browser's viewer (a link on iOS, whose Safari shows only the first page of
// an embedded PDF), an image, or the size and Download for any other binary. The bar holds the
// name, the size, Open (a new tab) and Download. No DOM here, so the tests read the template.

import { html, nothing, type TemplateResult } from 'lit';
import { bytes } from '../lib/format';
import { icon } from '../ui/bits';

export type MediaKind = 'video' | 'audio' | 'pdf' | 'image' | 'binary';

export interface MediaView {
  kind: MediaKind;
  src: string;
  name: string;
  /** the file's size; null when the caller does not know it */
  size: number | null;
  /** iPhone or iPad Safari: a PDF opens in a tab */
  ios: boolean;
  /** the browser failed to play it: why, with the codec when the file's header names one */
  failed?: string | null;
  /** a tool card's player: shorter, no Open */
  compact?: boolean;
}

export function isIOS(nav: { userAgent: string; platform?: string; maxTouchPoints?: number }): boolean {
  return /iP(hone|od|ad)/.test(nav.userAgent) || (nav.platform === 'MacIntel' && (nav.maxTouchPoints ?? 0) > 1);
}

/** the sample entries an mp4/mov/webm header names, as the codec a reader knows */
const CODECS: [string, string][] = [
  ['avc1', 'H.264'],
  ['avc3', 'H.264'],
  ['hvc1', 'HEVC (H.265)'],
  ['hev1', 'HEVC (H.265)'],
  ['av01', 'AV1'],
  ['vp09', 'VP9'],
  ['vp08', 'VP8'],
  ['mp4v', 'MPEG-4 Part 2'],
  ['ProRes', 'ProRes'],
  ['apcn', 'ProRes'],
  ['mp4a', 'AAC'],
  ['Opus', 'Opus'],
  ['fLaC', 'FLAC'],
  ['ac-3', 'AC-3'],
];

/** the codecs named in the first bytes of a file (its header); [] when none is found */
export function codecsIn(head: Uint8Array): string[] {
  const text = new TextDecoder('latin1').decode(head);
  const out: string[] = [];
  for (const [tag, name] of CODECS) if (text.includes(tag) && !out.includes(name)) out.push(name);
  return out;
}

export function failedWords(codecs: string[]): string {
  return codecs.length ? `This browser cannot play this file (${codecs.join(', ')}). Download it to play it in another player.` : 'This browser cannot play this file. Download it to play it in another player.';
}

function bar(v: MediaView): TemplateResult {
  return html`<div class="ch-pane-bar ch-media-bar"><span class="small ch-media-name">${v.name}</span>${v.size != null ? html`<span class="muted small num">${bytes(v.size, 1)}</span>` : nothing}<span class="grow"></span>${v.compact ? nothing : html`<a class="btn btn-ghost btn-sm" href=${v.src} target="_blank" rel="noopener">${icon('external')} Open</a>`}<a class="btn btn-ghost btn-sm" href=${v.src} download=${v.name}>${icon('down')} Download</a></div>`;
}

export function mediaBody(v: MediaView, onError: (e: Event) => void = () => {}): TemplateResult {
  let body: TemplateResult;
  if (v.failed) body = html`<p class="ch-media-note muted small" role="status">${v.failed}</p>`;
  else if (v.kind === 'video') body = html`<video class="ch-media-video" controls playsinline preload="metadata" src=${v.src} @error=${onError}></video>`;
  else if (v.kind === 'audio') body = html`<audio class="ch-media-audio" controls preload="metadata" src=${v.src} @error=${onError}></audio>`;
  else if (v.kind === 'image') body = html`<div class="ch-img-frame"><img src=${v.src} alt=${v.name} /></div>`;
  else if (v.kind === 'pdf' && v.ios) body = html`<p class="ch-media-note small">This browser shows a PDF in its own tab. <a href=${v.src} target="_blank" rel="noopener">Open ${v.name}</a></p>`;
  else if (v.kind === 'pdf') body = html`<object class="ch-pdf-frame" type="application/pdf" data=${v.src} aria-label=${v.name}><p class="ch-media-note small">No PDF viewer here. <a href=${v.src} target="_blank" rel="noopener">Open ${v.name}</a></p></object>`;
  else body = html`<p class="ch-media-note muted small">${v.size != null ? `${bytes(v.size, 1)}, ` : ''}not a file this pane can show. Download it to open it.</p>`;
  return html`<div class="ch-media is-${v.kind} ${v.compact ? 'is-compact' : ''}" data-media=${v.kind}>${bar(v)}${body}</div>`;
}
