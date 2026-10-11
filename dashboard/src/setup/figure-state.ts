// Wren's figure, the gold box, takes a pose from what Wren is doing (D7, VIS-80). `nextFigureState`
// folds the panel's events into the figure's state; it is pure so the order is unit-tested. Error and
// unload win from any state; a sleeping box wakes only by loading. `poseFor` gives each state's
// still pose (eyes, mouth, extra, status LED, motion). With prefers-reduced-motion the still pose is
// all that shows, so no two states share an eyes + mouth + extra triple (unit-tested).

export type FigureState = 'idle' | 'listening' | 'thinking' | 'loading' | 'talking' | 'acting' | 'error' | 'sleeping';

export type FigureEvent =
  | 'focus' // the composer got focus
  | 'input' // the user typed
  | 'blur' // the composer lost focus, empty
  | 'send' // a message went out (waiting for the first token)
  | 'state-thinking' // the stream said thinking
  | 'first-token'
  | 'delta'
  | 'tool-call' // a tool call is being executed
  | 'acted' // the tool call's hop is over and the turn goes on
  | 'done'
  | 'stop' // the user pressed Stop
  | 'error'
  | 'load-start'
  | 'load-progress'
  | 'loaded'
  | 'unload'
  | 'reset'; // the error was dismissed, a retry

export const FIGURE_STATES: FigureState[] = ['idle', 'listening', 'thinking', 'loading', 'talking', 'acting', 'error', 'sleeping'];

export function nextFigureState(cur: FigureState, ev: FigureEvent): FigureState {
  if (ev === 'error') return 'error';
  if (ev === 'unload') return 'sleeping';
  if (ev === 'load-start' || ev === 'load-progress') return 'loading';
  if (cur === 'sleeping') return 'sleeping';
  if (cur === 'loading') return ev === 'loaded' ? 'idle' : 'loading';
  switch (ev) {
    case 'loaded':
    case 'done':
    case 'stop':
    case 'reset':
      return 'idle';
    case 'focus':
    case 'input':
      return cur === 'idle' || cur === 'listening' || cur === 'error' ? 'listening' : cur;
    case 'blur':
      return cur === 'listening' ? 'idle' : cur;
    case 'send':
    case 'state-thinking':
    case 'acted':
      return 'thinking';
    case 'first-token':
    case 'delta':
      return cur === 'acting' ? 'acting' : 'talking';
    case 'tool-call':
      return 'acting';
  }
  return cur;
}

/** Fold a sequence, keeping each change once (what the tests read). */
export function foldFigure(events: FigureEvent[], start: FigureState = 'idle'): FigureState[] {
  const out: FigureState[] = [];
  let s = start;
  for (const e of events) {
    s = nextFigureState(s, e);
    if (out[out.length - 1] !== s) out.push(s);
  }
  return out;
}

/** A still pose. The figure writes these as data-eyes, data-mouth, data-extra, data-led and
 *  data-motion on its svg; setup.css draws each value. */
export interface FigurePose {
  /** dot: open, blinking; wide: bigger, toward the composer; up: up and to the sides; lid: half
   *  closed; side: toward the wizard; small: smaller and apart; shut: closed curves */
  eyes: 'dot' | 'wide' | 'up' | 'lid' | 'side' | 'small' | 'shut';
  mouth: 'smile' | 'o' | 'flat' | 'open' | 'wide' | 'wavy' | 'tiny';
  /** what shows above the box: "?" marks, a spark, a sweat drop with "?!", z */
  extra: 'none' | 'question' | 'spark' | 'alarm' | 'z';
  /** the status LED on the front's top-right edge */
  led: 'breathe' | 'steady' | 'amber' | 'flash' | 'blink' | 'off';
  /** what the whole box does (the hop itself is WAAPI) */
  motion: 'breathe' | 'lean' | 'wiggle' | 'slow' | 'bounce' | 'hop' | 'shake' | 'sleep';
}

const POSES: Record<FigureState, FigurePose> = {
  idle: { eyes: 'dot', mouth: 'smile', extra: 'none', led: 'breathe', motion: 'breathe' },
  listening: { eyes: 'wide', mouth: 'o', extra: 'none', led: 'steady', motion: 'lean' },
  thinking: { eyes: 'up', mouth: 'flat', extra: 'question', led: 'amber', motion: 'wiggle' },
  loading: { eyes: 'lid', mouth: 'flat', extra: 'none', led: 'steady', motion: 'slow' },
  talking: { eyes: 'dot', mouth: 'open', extra: 'none', led: 'steady', motion: 'bounce' },
  acting: { eyes: 'side', mouth: 'wide', extra: 'spark', led: 'flash', motion: 'hop' },
  error: { eyes: 'small', mouth: 'wavy', extra: 'alarm', led: 'blink', motion: 'shake' },
  sleeping: { eyes: 'shut', mouth: 'tiny', extra: 'z', led: 'off', motion: 'sleep' },
};

export function poseFor(state: FigureState): FigurePose {
  return POSES[state] ?? POSES.idle;
}

/** The ring's filled share while loading, clamped to 0.02..1; null (no figure from the server, or
 *  not a number) makes the ring chase instead. */
export function ringFill(progress: number | null): number | null {
  if (progress == null || !Number.isFinite(progress)) return null;
  return Math.max(0.02, Math.min(1, progress));
}

/** How far the mouth opens for one streamed delta (scaleY of the open mouth), from a 0..1 draw:
 *  never shut, and never the same height twice in a row. */
export function blipPeak(r: number, last = 0): number {
  const v = 0.55 + 0.45 * Math.max(0, Math.min(1, Number.isFinite(r) ? r : 0));
  return Math.abs(v - last) < 0.12 ? (v > 0.78 ? v - 0.25 : v + 0.2) : v;
}
