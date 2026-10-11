// <qse-wren-figure>: Wren's figure, the gold DGX Spark box (D7, VIS-80, VIS-89), drawn as the
// paper-cut character from Khaled's video "7 weeks with one DGX Spark": a warm gold box, wider than
// tall, with a lighter top, the metal foam as three columns of dots on each side, two dark eyes with
// a highlight, pink cheeks, a mouth and tiny feet, cut out with an edge that reads on either theme.
// A tiny status LED on the front's top-right edge is the one nod to the real box.
//
// The pose comes from poseFor(state) (figure-state.ts) as data attributes on the svg: data-eyes,
// data-mouth, data-extra, data-led and data-motion. setup.css (`.goldbox`) draws every value and runs
// the loops; WAAPI plays the moments that answer an event: `blip()` opens the mouth once for a
// streamed delta, `hop()` jumps toward the wizard for a tool call. With prefers-reduced-motion every
// state is its still pose and nothing moves. Plain shapes and fills only, no url(#id) references:
// three copies live on one page and one of them is always display:none. The figure is decorative
// (aria-hidden); the panel says the state in words.

import { html, svg, nothing } from 'lit';
import { customElement, property } from 'lit/decorators.js';
import { LightElement } from '../ui/base';
import { blipPeak, poseFor, ringFill, type FigureState } from './figure-state';

export function reducedMotion(): boolean {
  return typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;
}

// The silhouette: drawn first with a wide stroke (the cut edge) and offset as the paper shadow.
// Slightly uneven on purpose, like a shape cut with scissors.
const OUTLINE = 'M16.2 29.4L83.6 29Q85.2 29 86 30.4L89.6 38.6Q90.4 40 90.3 43.4L89.8 79.2Q89.7 82.2 86.8 82.2L13.4 82.6Q10.4 82.6 10.4 79.6L10.2 43Q10.1 40.2 10.9 38.8L14.4 30.6Q15 29.4 16.2 29.4Z';
const FRONT = 'M12.5 40L87.5 40.3Q90.3 40.4 90.3 43.4L89.8 79.2Q89.7 82.2 86.8 82.2L13.4 82.6Q10.4 82.6 10.4 79.6L10.2 43Q10.2 40 12.5 40Z';

/** the foam: three columns of four dots on each side of the face, 24 in all */
const DOT_X = [16.6, 21.6, 26.6, 73.4, 78.4, 83.4];
const DOT_Y = [47, 55, 63, 71];
const DOTS = DOT_X.flatMap((x) => DOT_Y.map((y) => `M${x - 1.3} ${y}a1.3 1.3 0 1 0 2.6 0a1.3 1.3 0 1 0-2.6 0`)).join('');
/** while loading the dots light in a wave, column by column */
const LIT = svg`<g class="gb-lit">${DOT_X.flatMap((x, c) => DOT_Y.map((y) => svg`<circle cx=${x} cy=${y} r="1.5" style=${`animation-delay:${(c * 0.12).toFixed(2)}s`} />`))}</g>`;

const eye = (side: 'l' | 'r', x: number) => svg`<g class=${`gb-eye gb-eye-${side}`}>
  <ellipse cx=${x} cy="56.4" rx="4.3" ry="5.5" />
  <circle class="gb-glint" cx=${x + 1.5} cy="54.3" r="1.25" />
  <rect class="gb-lid" x=${x - 4.8} y="50.4" width="9.6" height="5.8" />
</g>`;

/** the box; `lit` is the loading wave over the foam dots (inside the body, so it moves with it) */
const box = (lit: unknown) => svg`
  <ellipse class="gb-ground" cx="50" cy="88.6" rx="36" ry="2.6" />
  <g class="gb-all">
    <g class="gb-feet"><rect x="21" y="79.6" width="10.5" height="8" rx="2.2" /><rect x="68.5" y="79.6" width="10.5" height="8" rx="2.2" /></g>
    <g class="gb-body">
      <path class="gb-cut" d=${OUTLINE} transform="translate(1.4 1.8)" />
      <path class="gb-edge" d=${OUTLINE} />
      <path class="gb-front" d=${FRONT} />
      <path class="gb-chin" d="M10.4 77.6L89.9 77.2 89.8 79.2Q89.7 82.2 86.8 82.2L13.4 82.6Q10.4 82.6 10.4 79.6Z" />
      <path class="gb-top" d="M16.2 29.4L83.6 29Q85.2 29 86 30.4L89.8 39.4Q89.6 40.3 87.5 40.3L12.5 40Q10.6 40 10.9 39L14.4 30.6Q15 29.4 16.2 29.4Z" />
      <path class="gb-seam" d="M11.4 40L89.4 40.3" />
      <path class="gb-dots" d=${DOTS} />
      ${lit}
      <circle class="gb-led-halo" cx="86.4" cy="43.4" r="2.9" />
      <circle class="gb-led" cx="86.4" cy="43.4" r="1.25" />
      <g class="gb-face">
        <ellipse class="gb-blush" cx="28.4" cy="66" rx="4.6" ry="2.8" />
        <ellipse class="gb-blush" cx="71.6" cy="66" rx="4.6" ry="2.8" />
        <g class="gb-eyes">${eye('l', 38.6)}${eye('r', 61.4)}</g>
        <path class="gb-shut" d="M34.4 56.6Q38.6 60.8 42.8 56.6M57.2 56.6Q61.4 60.8 65.6 56.6" />
        <path class="gb-m gb-m-smile" d="M45.8 67.8L47.7 70.4H52.3L54.2 67.8" />
        <ellipse class="gb-m gb-m-o gb-fill" cx="50" cy="69.8" rx="1.8" ry="2.2" />
        <path class="gb-m gb-m-flat" d="M47 69.6H53" />
        <ellipse class="gb-m gb-m-open gb-fill" cx="50" cy="70.4" rx="3.3" ry="3.4" />
        <g class="gb-m gb-m-wide"><path class="gb-fill" d="M44.6 67.2H55.4Q55.4 73 50 73Q44.6 73 44.6 67.2Z" /><ellipse class="gb-tongue" cx="50" cy="71.4" rx="2.6" ry="1.3" /></g>
        <path class="gb-m gb-m-wavy" d="M45.4 70.4L47.7 68.4 50 70.4 52.3 68.4 54.6 70.4" />
        <path class="gb-m gb-m-tiny" d="M48.4 69.8H51.6" />
      </g>
    </g>
  </g>
  <g class="gb-question"><text x="68" y="23">?</text><text x="77" y="15">?</text><text x="86" y="7">?</text></g>
  <g class="gb-alarm"><text x="70" y="22">?!</text><path class="gb-sweat" d="M87 32.4Q84 36.6 84 38.6A3 3 0 0 0 90 38.6Q90 36.6 87 32.4Z" /></g>
  <g class="gb-z"><text x="70" y="22">z</text><text x="78" y="14">z</text><text x="86" y="6">z</text></g>
  <g class="gb-spark"><path d="M10 22l0-7M6.5 18.5l7 0" /><path d="M20 12l0-5M17.5 9.5l5 0" /></g>
`;

/** below this size the svg draws only the box (a tighter frame), so the eyes stay 5 px across */
const SMALL = 56;
let lastPeak = 0;

@customElement('qse-wren-figure')
export class QseWrenFigure extends LightElement {
  @property({ reflect: true }) state: FigureState = 'idle';
  /** CSS pixels */
  @property({ type: Number }) size = 72;
  /** 0..1 while loading; null makes the ring chase */
  @property({ attribute: false }) progress: number | null = null;
  private lastBlip = 0;

  /** The mouth opens once for a streamed delta. Throttled so a fast stream stays a chatter. */
  blip(): void {
    const now = performance.now();
    if (now - this.lastBlip < 140 || reducedMotion()) return;
    this.lastBlip = now;
    lastPeak = blipPeak(Math.random(), lastPeak);
    this.querySelector('.gb-m-open')?.animate([{ transform: 'scaleY(.5)' }, { transform: `scaleY(${lastPeak})`, offset: 0.4 }, { transform: 'scaleY(.5)' }], { duration: 180, easing: 'ease-out' });
  }

  /** Two hops toward the wizard (left), stretched in the air and squashed on each landing. */
  hop(): Promise<void> {
    const all = this.querySelector<SVGGElement>('.gb-all');
    if (!all || reducedMotion()) return Promise.resolve();
    const t = { duration: 1100, easing: 'cubic-bezier(.3,.7,.4,1)' };
    const a = all.animate(
      [
        { transform: 'translate(0,0) scale(1,1)' },
        { transform: 'translate(0,0) scale(1.1,.86)', offset: 0.1 },
        { transform: 'translate(-5px,-16px) scale(.92,1.1)', offset: 0.28 },
        { transform: 'translate(-9px,0) scale(1.12,.84)', offset: 0.44 },
        { transform: 'translate(-12px,-10px) scale(.94,1.07)', offset: 0.62 },
        { transform: 'translate(-14px,0) scale(1.08,.9)', offset: 0.78 },
        { transform: 'translate(0,0) scale(1,1)' },
      ],
      t,
    );
    this.querySelector('.gb-ground')?.animate(
      [
        { transform: 'translateX(0) scaleX(1)' },
        { transform: 'translateX(-5px) scaleX(.6)', offset: 0.28 },
        { transform: 'translateX(-9px) scaleX(1.06)', offset: 0.44 },
        { transform: 'translateX(-12px) scaleX(.75)', offset: 0.62 },
        { transform: 'translateX(-14px) scaleX(1)', offset: 0.78 },
        { transform: 'translateX(0) scaleX(1)' },
      ],
      t,
    );
    this.querySelector('.gb-spark')?.animate([{ opacity: 0, transform: 'scale(.4)' }, { opacity: 1, transform: 'scale(1.1)', offset: 0.3 }, { opacity: 0, transform: 'scale(1.3)' }], { duration: 900, delay: 300, easing: 'ease-out' });
    return a.finished.then(() => undefined).catch(() => undefined);
  }

  render() {
    const pose = poseFor(this.state);
    const loading = this.state === 'loading';
    const fill = loading ? ringFill(this.progress) : null;
    // small: a 84-unit frame around the box instead of 108, drawn larger than the host and pulled
    // back by a negative margin, so the host keeps its size x size box and nothing shifts
    const px = this.size <= SMALL ? Math.round((this.size * 108) / 84) : this.size;
    const m = (px - this.size) / 2;
    return html`<svg
      class="goldbox"
      data-state=${this.state}
      data-eyes=${pose.eyes}
      data-mouth=${pose.mouth}
      data-extra=${pose.extra}
      data-led=${pose.led}
      data-motion=${pose.motion}
      ?data-small=${m > 0}
      viewBox="-4 -4 108 108"
      width=${px}
      height=${px}
      style=${m ? `margin:${-m}px` : nothing}
      aria-hidden="true"
      focusable="false"
    >
      ${loading
        ? svg`<g class="w-ring"><circle class="w-ring-track" cx="50" cy="58" r="48" pathLength="100" />${
            fill != null
              ? svg`<circle class="w-ring-fill" cx="50" cy="58" r="48" pathLength="100" stroke-dasharray=${`${(fill * 100).toFixed(1)} 100`} />`
              : svg`<circle class="w-ring-spin" cx="50" cy="58" r="48" pathLength="100" stroke-dasharray="18 82" />`
          }</g>`
        : nothing}
      ${box(loading ? LIT : nothing)}
    </svg>`;
  }
}

/** The mini gold box that hops onto a field Wren changed (the field's highlight is the pointer):
 *  an "o" in the air, a smile on landing, eyes on the field. */
export const MINI_WREN = svg`<svg class="wren-mini-box" viewBox="0 0 30 22" width="32" height="24" aria-hidden="true" focusable="false">
  <rect class="mw-foot" x="6.4" y="16.4" width="3.4" height="4" rx="1" /><rect class="mw-foot" x="20.2" y="16.4" width="3.4" height="4" rx="1" />
  <path class="mw-edge" d="M6.2 2.6L23.8 2.4 27 6.6Q27.6 7.2 27.6 8.2V17.2Q27.6 18.4 26.4 18.4L3.6 18.6Q2.4 18.6 2.4 17.4V8.2Q2.4 7.2 3 6.6Z" />
  <path class="mw-front" d="M3.6 7L26.4 7.1Q27.6 7.1 27.6 8.3V17.2Q27.6 18.4 26.4 18.4L3.6 18.6Q2.4 18.6 2.4 17.4V8.2Q2.4 7 3.6 7Z" />
  <path class="mw-top" d="M6.2 2.6L23.8 2.4 27 7.1 3 7Z" />
  <ellipse class="mw-blush" cx="8.8" cy="14.4" rx="1.7" ry="1" /><ellipse class="mw-blush" cx="21.2" cy="14.4" rx="1.7" ry="1" />
  <g class="mw-eyes"><ellipse cx="12" cy="11.4" rx="1.25" ry="1.6" /><ellipse cx="18" cy="11.4" rx="1.25" ry="1.6" /></g>
  <ellipse class="mw-o" cx="15" cy="15" rx=".75" ry=".9" />
  <path class="mw-smile" d="M13.6 14.4L14.3 15.3H15.7L16.4 14.4" />
</svg>`;
