// RunToken (deep audit 0.11.3): a boot whose mount went away mid-await must not own its
// timers/polls/streams -- the disconnect's cleanup already ran on nulls. The view wires it
// at every await boundary before a resource-owning call (startReprobe/downClock,
// afterSignIn's listPoll and global stream).
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { RunToken } from '../src/lib/run-token';

describe('RunToken', () => {
  it('starts a fresh generation per mount', () => {
    const t = new RunToken();
    const a = t.start();
    const b = t.start();
    expect(a).not.toBe(b);
    expect(t.live(a)).toBe(false);
    expect(t.live(b)).toBe(true);
  });
  it('a detach stales the mount that was live', () => {
    const t = new RunToken();
    const a = t.start();
    expect(t.live(a)).toBe(true);
    t.stop();
    expect(t.live(a)).toBe(false);
  });
  it('a re-attach stales the old mount without reviving its work', () => {
    const t = new RunToken();
    const a = t.start();
    t.stop();
    const b = t.start();
    expect(t.live(a)).toBe(false);
    expect(t.current()).toBe(b);
  });
});

describe('the chat view wires the token at every owning await', () => {
  const src = readFileSync(new URL('../src/chat/view.ts', import.meta.url), 'utf8');

  it('the mount starts it and the detach stops it first', () => {
    expect(src).toMatch(/connectedCallback\(\)[\s\S]*this\.boots\.start\(\);/);
    const gone = src.match(/disconnectedCallback\(\): void \{\n\s*(?:.*\n)*?\s*\}/)?.[0] ?? '';
    expect(gone).toContain('this.boots.stop();');
  });

  it('boot re-checks it before every resource it would own', () => {
    const owns = ['this.startReprobe(', 'await this.afterSignIn()'];
    const boot = src.slice(src.indexOf('private async boot'), src.indexOf('private async loadCaps'));
    for (const call of owns) {
      const at = boot.indexOf(call);
      expect(at).toBeGreaterThan(-1);
      expect(boot.slice(0, at)).toContain('this.boots.live(run)');
    }
    expect([...boot.matchAll(/if \(!this\.boots\.live\(run\)\) return/g)].length).toBeGreaterThanOrEqual(2);
  });

  it('a chat that loads after the view is gone opens no stream and navigates nowhere', () => {
    const open = src.slice(src.indexOf('private async open('), src.indexOf('// ---- the stream'));
    const load = open.indexOf('await this.backend!.getConversation(id)');
    expect(load).toBeGreaterThan(-1);
    expect(open.slice(0, load)).toContain('const run = this.boots.current();');
    const after = open.slice(load);
    const check = after.search(/if \(!this\.boots\.live\(run\)[^\n]*\) return/);
    expect(check).toBeGreaterThan(-1);
    for (const call of ['navigate(', 'this.attachTo(']) expect(after.indexOf(call)).toBeGreaterThan(check);
    // afterSignIn hands open() a token taken after its own await, so it re-checks first
    const signIn = src.slice(src.indexOf('private async afterSignIn'), src.indexOf('private async loadList'));
    expect(signIn).toContain('const run = this.boots.current();');
    const list = signIn.indexOf('await this.loadList();');
    const guard = signIn.indexOf('if (!this.boots.live(run)) return;');
    expect(guard).toBeGreaterThan(list);
    expect(signIn.indexOf('await this.openFromRoute()')).toBeGreaterThan(guard);
  });

  it('the down-reprobe dies with its mount too', () => {
    const closure = src.slice(src.indexOf('private startReprobe'), src.indexOf('private stopReprobe'));
    expect(closure).not.toContain('live(run)');   // the check below is in boot's closure, not here
    const check = src.slice(src.indexOf('this.startReprobe(async () => {'), src.indexOf('}, true);'));
    expect([...check.matchAll(/if \(!this\.boots\.live\(run\)\) return true;/g)].length)
      .toBeGreaterThanOrEqual(2);
  });
});
