// Every component renders into the light DOM: one stylesheet (tokens.css + app.css) styles the
// whole app, CSS custom properties reach every chart, and e2e selectors see plain markup.

import { LitElement } from 'lit';

export class LightElement extends LitElement {
  protected createRenderRoot(): HTMLElement {
    return this;
  }
}

/** A loading / error / empty / ready state for a panel. */
export type Loadable<T> = { state: 'loading' } | { state: 'error'; endpoint: string; message: string } | { state: 'empty'; note?: string } | { state: 'ready'; data: T };

export function loading<T>(): Loadable<T> {
  return { state: 'loading' };
}
