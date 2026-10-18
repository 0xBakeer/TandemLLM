// Detach (deep audit 0.11.3): the 3 s health probe's readJob(true) lands after the page
// unmounted, and its reopen line started the stream disconnectedCallback had just closed.
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

const src = readFileSync(new URL('../src/setup/install.ts', import.meta.url), 'utf8');

describe('a detached install page revives nothing', () => {
  it('the disconnect flags it first thing', () => {
    const gone = src.slice(src.indexOf('disconnectedCallback(): void {'), src.indexOf('protected willUpdate'));
    expect(gone.indexOf('this.detached = true;')).toBeLessThan(gone.indexOf('this.stream?.close();'));
  });

  it('a read refuses to even start once detached', () => {
    const read = src.slice(src.indexOf('private async readJob'), src.indexOf('private async readJob') + 900);
    expect(read).toContain('if (this.detached || !this.jobId || this.reading) return;');
  });

  it('and the reopen only serves a live page', () => {
    const read = src.slice(src.indexOf('private async readJob'), src.indexOf('private async readJob') + 900);
    expect(read).toContain('if (reopen && !this.detached && this.stream');
  });
});
