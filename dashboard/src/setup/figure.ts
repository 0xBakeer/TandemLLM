// <qse-wren-figure>: Wren, a small bird drawn in SVG (D7, VIS-80). The pose follows `state`: idle
// breathes and blinks, listening tilts its head, thinking looks around under three dots, loading
// preens inside a progress ring, talking moves its beak, acting hops, error ruffles and droops,
// sleeping tucks its head under a drifting "z". CSS animates the loops (setup.css, `.wren`); WAAPI
// plays the moments that answer an event: `chirp()` on a streamed delta, `hop()` per tool call.
// With prefers-reduced-motion every state is a still pose and nothing moves. The figure is
// decorative (aria-hidden): the panel announces the state in words.

import { html, svg, nothing } from 'lit';
import { customElement, property } from 'lit/decorators.js';
import { LightElement } from '../ui/base';
import type { FigureState } from './figure-state';

export function reducedMotion(): boolean {
  return typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;
}

/** The bird alone, facing left (toward the wizard). viewBox 0 0 100 100. */
const BIRD = svg`
  <g class="w-perch" aria-hidden="true">
    <path d="M6 87.5 C 30 85.8, 62 86.6, 94 88.4" />
    <path d="M70 87 C 74 83, 78 82, 82 82.6" />
  </g>
  <g class="w-all">
    <g class="w-legs">
      <path d="M45 73.5 L43.6 86.2 M43.6 86.2 l-3.6 .6 M43.6 86.2 l2.6 .9" />
      <path d="M54 74 L55 86.4 M55 86.4 l-3.4 .5 M55 86.4 l3 .8" />
    </g>
    <g class="w-tail">
      <path class="w-tail-fill" d="M68.5 55.5 L79.5 23.5 C 81 19.4, 87.6 19.8, 88.3 24.4 L80.6 58.5 Z" />
      <path class="w-bar" d="M79.6 31.6 l6.4 1.6 M77.6 38.4 l6.8 1.6 M75.6 45.2 l7 1.5" />
    </g>
    <g class="w-breathe">
      <path class="w-body" d="M29.5 57 C 29.5 43.5, 43 37.5, 57.5 39.6 C 72 41.8, 80.5 51.5, 78.6 62 C 76.6 72.6, 64.4 78.4, 50.2 77.4 C 37.6 76.5, 29.5 68.4, 29.5 57 Z" />
      <path class="w-belly" d="M31.2 56.4 C 31.6 66.4, 39.6 74.6, 50.4 75.6 C 57.2 76.2, 62.6 73.6, 64.4 70 C 58 66.4, 52 63.6, 46.4 58.6 C 42.4 55, 36.4 53.4, 31.2 56.4 Z" />
      <g class="w-wing">
        <path class="w-wing-fill" d="M47.4 51.2 C 56.8 45.4, 70.6 48.2, 75.4 58.4 C 70.6 66, 58.2 67.6, 49.6 61.4 C 46.6 59, 45.6 54, 47.4 51.2 Z" />
        <path class="w-bar" d="M54.2 51.4 l2.4 9.2 M60.4 50.6 l2.2 10 M66.6 52 l1.8 9.4" />
      </g>
      <g class="w-ruffle">
        <path d="M33 47 l-5.2 -3.4 M40 41 l-2 -5.6 M58 39.6 l1.4 -5.8 M70 44 l4 -4.2 M78.4 55 l5.6 -1.6 M74 70 l5 3" />
      </g>
    </g>
    <g class="w-head">
      <circle class="w-head-fill" cx="34" cy="41.5" r="13" />
      <path class="w-brow" d="M24.4 36.4 C 29.4 32.8, 38.2 33.4, 44.6 38.2" />
      <circle class="w-cheek" cx="31.8" cy="47.2" r="2.6" />
      <g class="w-eye-g">
        <circle class="w-eye" cx="29.6" cy="40.6" r="2.7" />
        <circle class="w-glint" cx="28.7" cy="39.6" r=".85" />
        <path class="w-lid" d="M26.4 40.6 a3.3 3.3 0 0 1 6.6 0 Z" />
        <path class="w-shut" d="M26.6 41 C 28 42.6, 31.2 42.6, 32.6 41" />
      </g>
      <g class="w-beak">
        <path class="w-beak-up" d="M22.4 42.2 C 18 42.4, 13.6 43.4, 10.4 44.8 C 14 45.2, 18.4 45.1, 22.6 45 Z" />
        <path class="w-beak-lo" d="M22.6 45 C 18.4 45.1, 14.4 45.4, 11.6 45.6 C 15 46.8, 19 47.2, 22.8 46.8 Z" />
      </g>
    </g>
  </g>
  <g class="w-dots">
    <circle cx="12" cy="22" r="2.4" /><circle cx="20" cy="17" r="2.9" /><circle cx="29.5" cy="13.5" r="3.4" />
  </g>
  <g class="w-z">
    <text x="15" y="30">z</text><text x="9" y="20">z</text><text x="2" y="10">z</text>
  </g>
  <g class="w-spark">
    <path d="M10 26 l0 -7 M6.5 22.5 l7 0" /><path d="M20 14 l0 -5 M17.5 11.5 l5 0" />
  </g>
`;

@customElement('qse-wren-figure')
export class QseWrenFigure extends LightElement {
  @property({ reflect: true }) state: FigureState = 'idle';
  /** CSS pixels */
  @property({ type: Number }) size = 72;
  /** 0..1 while loading; null hides the ring */
  @property({ attribute: false }) progress: number | null = null;
  private lastChirp = 0;

  /** The beak opens once (a streamed delta). Throttled so a fast stream stays a chatter. */
  chirp(): void {
    const now = performance.now();
    if (now - this.lastChirp < 140 || reducedMotion()) return;
    this.lastChirp = now;
    this.querySelector('.w-beak-lo')?.animate([{ transform: 'rotate(0deg)' }, { transform: 'rotate(17deg)' }, { transform: 'rotate(0deg)' }], { duration: 170, easing: 'ease-out' });
  }

  /** Two hops toward the wizard (left), a squash on each landing. */
  hop(): Promise<void> {
    const all = this.querySelector<SVGGElement>('.w-all');
    if (!all || reducedMotion()) return Promise.resolve();
    const a = all.animate(
      [
        { transform: 'translate(0,0) scale(1,1)', offset: 0 },
        { transform: 'translate(0,0) scale(1.06,.9)', offset: 0.1 },
        { transform: 'translate(-5px,-15px) scale(.95,1.06)', offset: 0.28 },
        { transform: 'translate(-9px,0) scale(1.08,.88)', offset: 0.44 },
        { transform: 'translate(-12px,-10px) scale(.96,1.05)', offset: 0.62 },
        { transform: 'translate(-14px,0) scale(1.05,.92)', offset: 0.78 },
        { transform: 'translate(0,0) scale(1,1)', offset: 1 },
      ],
      { duration: 1100, easing: 'cubic-bezier(.3,.7,.4,1)' },
    );
    this.querySelector('.w-spark')?.animate([{ opacity: 0, transform: 'scale(.4)' }, { opacity: 1, transform: 'scale(1.1)', offset: 0.3 }, { opacity: 0, transform: 'scale(1.3)' }], { duration: 900, delay: 300, easing: 'ease-out' });
    return a.finished.then(() => undefined).catch(() => undefined);
  }

  render() {
    const p = this.progress;
    const showRing = this.state === 'loading';
    return html`<svg class="wren" data-state=${this.state} viewBox="-4 -4 108 108" width=${this.size} height=${this.size} aria-hidden="true" focusable="false">
      ${showRing
        ? svg`<g class="w-ring"><circle class="w-ring-track" cx="50" cy="52" r="49" pathLength="100" />${p != null ? svg`<circle class="w-ring-fill" cx="50" cy="52" r="49" pathLength="100" stroke-dasharray=${`${Math.max(2, Math.min(100, p * 100))} 100`} />` : svg`<circle class="w-ring-spin" cx="50" cy="52" r="49" pathLength="100" stroke-dasharray="18 82" />`}</g>`
        : nothing}
      ${BIRD}
    </svg>`;
  }
}

/** The tiny wren that perches on a field Wren changed (the field's highlight is the pointer). */
export const PERCH_WREN = svg`<svg class="wren-perch-bird" viewBox="0 0 24 20" width="30" height="25" aria-hidden="true" focusable="false">
  <path class="pb-tail" d="M15.6 9.2 L18.4 2 C 18.8 1, 20.4 1.1, 20.5 2.2 L18.6 10.2 Z" />
  <path class="pb-body" d="M5 10.4 C 5 6.6, 9 5, 12.8 5.6 C 16.8 6.2, 19 8.8, 18.4 11.6 C 17.8 14.4, 14.6 15.8, 11 15.6 C 7.6 15.4, 5 13.4, 5 10.4 Z" />
  <circle class="pb-head" cx="7" cy="7.6" r="3.6" />
  <path class="pb-beak" d="M4 7.8 L1 8.6 L4 9.1 Z" />
  <circle class="pb-eye" cx="5.9" cy="7.2" r=".75" />
  <path class="pb-legs" d="M10.4 15.4 l-.4 3.2 M13.2 15.4 l.3 3.2" />
</svg>`;
