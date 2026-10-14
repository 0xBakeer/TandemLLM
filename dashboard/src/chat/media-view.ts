// <qse-media-view kind src name size compact>: a sandbox file in a player or viewer (VIS-159). Its
// own chunk: panes.ts and thread.ts load it the first time a pane or a tool card shows a video,
// an audio file, a PDF or a binary (file-panes.ts `loadMediaView`). When the browser cannot play
// the file, the first 64 KiB of it name the codec in the note.

import { customElement, property, state } from 'lit/decorators.js';
import { LightElement } from '../ui/base';
import { codecsIn, failedWords, isIOS, mediaBody, type MediaKind } from './media-body';

@customElement('qse-media-view')
export class QseMediaView extends LightElement {
  @property({ type: String }) kind: MediaKind = 'binary';
  @property({ type: String }) src = '';
  @property({ type: String }) name = '';
  @property({ type: Number }) size: number | null = null;
  @property({ type: Boolean }) compact = false;
  @state() private failed: string | null = null;

  updated(changed: Map<string, unknown>): void {
    if (changed.has('src') && changed.get('src') !== undefined) this.failed = null;
  }

  private onError = async (): Promise<void> => {
    let codecs: string[] = [];
    try {
      const r = await fetch(this.src, { headers: { Range: 'bytes=0-65535' }, credentials: 'same-origin' });
      codecs = codecsIn(new Uint8Array(await r.arrayBuffer()));
    } catch {
      // the note goes up without the codec
    }
    this.failed = failedWords(codecs);
  };

  render() {
    return mediaBody({ kind: this.kind, src: this.src, name: this.name, size: Number.isFinite(this.size) ? this.size : null, ios: isIOS(navigator), failed: this.failed, compact: this.compact }, this.onError);
  }
}
