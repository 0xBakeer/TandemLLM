// Charts measure their own width (ResizeObserver) and redraw at real pixel size — crisp text,
// no preserveAspectRatio stretching.

import { property, state } from 'lit/decorators.js';
import { LightElement } from '../ui/base';

export class ChartElement extends LightElement {
  @property({ type: Number }) height = 220;
  @state() protected width = 0;
  private ro: ResizeObserver | null = null;

  connectedCallback(): void {
    super.connectedCallback();
    this.width = this.getBoundingClientRect().width || 600;
    this.ro = new ResizeObserver((entries) => {
      const w = entries[0]?.contentRect.width ?? 0;
      if (w && Math.abs(w - this.width) > 1) this.width = w;
    });
    this.ro.observe(this);
  }

  disconnectedCallback(): void {
    this.ro?.disconnect();
    this.ro = null;
    super.disconnectedCallback();
  }
}

/** Series colour names → CSS variables (the validated palette in tokens.css). */
export type SeriesColor = 'cobalt' | 'orange' | 'aqua' | 'violet' | 'yellow' | 'red' | 'muted';
export function colorVar(c: SeriesColor): string {
  return `var(--series-${c})`;
}

let uid = 0;
export function nextId(prefix = 'c'): string {
  return `${prefix}${++uid}`;
}
