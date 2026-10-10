// The wren figure's pose follows what Wren is doing (D7, VIS-80). `nextFigureState` folds the panel's
// events into the figure's state; it is pure so the order is unit-tested. Error and unload win from
// any state; a sleeping wren wakes only by loading.

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
