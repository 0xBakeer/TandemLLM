// Issue #11: the 350 ms grep debounce outlived the dev view — a switch inside that window fired
// openStream() on a detached element, starting a follow=1 log stream nothing could close.
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

const src = readFileSync(new URL('../src/views/dev.ts', import.meta.url), 'utf8');
const gone = src.match(/disconnectedCallback\(\): void \{\n(?:.*\n)*?\s*\}/)?.[0] ?? '';

describe('a detached dev view starts no log stream', () => {
  it('the grep debounce is cleared with the other owned resources', () => {
    expect(gone).toContain('clearTimeout(this.grepTimer);');
    expect(gone.indexOf('clearTimeout(this.grepTimer);'))
      .toBeLessThan(gone.indexOf('super.disconnectedCallback();'));
  });

  it('the grep box still debounces into openStream', () => {
    expect(src).toContain('this.grepTimer = setTimeout(() => this.openStream(), 350);');
  });
});
