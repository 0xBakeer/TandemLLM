// Detach (#34 review): a Finish answer that lands after the wizard was left (browser Back, a hash
// change) must not open the install page, but the job it started runs on the box all the same.
// The shell follows it only through rememberJob() (knownJob(): the restart pill and the way to the
// setup page while the engine is away), so the abandoned branch records it and owns nothing else.
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

const src = readFileSync(new URL('../src/setup/view.ts', import.meta.url), 'utf8');
const finish = src.slice(src.indexOf('private async finish()'), src.indexOf('private applyErrorText('));
const gate = "if (!this.isConnected || this.path[0] !== selection.model || this.wiz.step !== 'review')";
const left = finish.slice(finish.indexOf(gate), finish.indexOf('this.jobInitial = res.job;'));

describe('a Finish answer for an abandoned wizard', () => {
  it('re-checks the mount and the wizard after the apply await', () => {
    expect(finish.indexOf('await setupApi.apply(')).toBeGreaterThan(-1);
    expect(finish.indexOf(gate)).toBeGreaterThan(finish.indexOf('await setupApi.apply('));
  });

  it('records the started job for the shell before it returns', () => {
    expect(src).toMatch(/import \{[^}]*\brememberJob\b[^}]*\} from '\.\.\/lib\/restart';/);
    expect(left).toContain('rememberJob(res.job);');
    expect(left.indexOf('rememberJob(res.job);')).toBeLessThan(left.indexOf('return;'));
  });

  it('and opens nothing from the detached view', () => {
    expect(left).not.toMatch(/this\.dispatch\(|this\.jobInitial|navigate\(|location\.hash/);
  });
});
