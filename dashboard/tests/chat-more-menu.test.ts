// VIS-183 (QA VIS-184): the chat header's More menu opened upward, under the top bar (2026-10-16):
// it took `.ch-pop`, which rises because it was made for the composer's pills.
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

describe("the header's More menu", () => {
  const css = readFileSync(new URL('../src/styles/chat.css', import.meta.url), 'utf8');
  const rule = (sel: string) => {
    const i = css.indexOf(`${sel} {`);
    return i < 0 ? '' : css.slice(i, css.indexOf('}', i));
  };
  it('opens down from its button, right-aligned, within the screen', () => {
    const r = rule('.ch-more > .ch-more-pop');
    expect(r).toContain('top: calc(100% + 6px)');
    expect(r).toContain('bottom: auto');
    expect(r).toContain('right: 0');
    expect(r).toContain('max-width: min(320px, calc(100vw - 16px))');
    expect(r).toMatch(/max-height: calc\(100dvh/);
  });
  it('outranks the phone rule that makes a .ch-pop rise and span its anchor', () => {
    // two classes beat `.ch-pop` (one) in the phone media query, which comes later in the file
    expect(css.indexOf('.ch-more > .ch-more-pop {')).toBeGreaterThan(-1);
    expect(css).not.toMatch(/\n\.ch-more-pop \{/);
  });
});
