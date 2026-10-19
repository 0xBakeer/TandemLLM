// RunToken (deep audit 0.11.3): a boot whose mount went away mid-await must not own its
// timers/polls/streams -- the disconnect's cleanup already ran on nulls. The view wires it
// at every await boundary before a resource-owning call (startReprobe/downClock,
// afterSignIn's listPoll and global stream), and hands the token down rather than re-reading it.
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
    const owns = ['this.startReprobe(', 'await this.afterSignIn(run)'];
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
    const list = signIn.indexOf('await this.loadList();');
    const guard = signIn.indexOf('if (!this.boots.live(run)) return;', list);
    expect(list).toBeGreaterThan(-1);
    expect(guard).toBeGreaterThan(list);
    expect(signIn.indexOf('await this.openFromRoute()')).toBeGreaterThan(guard);
  });

  // VIS-235: the sign-in handler awaits too; a token read late (after the detach's stop) passes
  it('a sign-in that finishes after the view is gone starts nothing', () => {
    const on = src.slice(src.indexOf('private onSignedIn = async'));
    const handler = on.slice(0, on.indexOf('\n  };') + 5);
    const take = handler.indexOf('const run = this.boots.current();');
    expect(take).toBeGreaterThan(-1);
    const waits = [...handler.matchAll(/await this\.(?:backend\.signIn|loadCaps)\(/g)].map((m) => m.index!);
    expect(waits).toHaveLength(2);
    expect(take).toBeLessThan(waits[0]);
    // after each await, a check before the next step
    for (const at of waits) {
      const rest = handler.slice(at);
      const nl = rest.indexOf('\n');
      expect(rest.slice(nl + 1).trimStart()).toMatch(/^if \(!this\.boots\.live\(run\)\) return;/);
    }
    expect(handler).toContain('await this.loadCaps(run);');
    expect(handler).toContain('await this.afterSignIn(run);');
  });

  it('afterSignIn and loadCaps take the caller\'s token and never read current() themselves', () => {
    const signIn = src.slice(src.indexOf('private async afterSignIn'), src.indexOf('private async loadList'));
    expect(signIn).toMatch(/^private async afterSignIn\(run: number\)/);
    expect(signIn).not.toContain('this.boots.current()');
    expect(signIn.indexOf('if (!this.boots.live(run)) return;')).toBeLessThan(signIn.indexOf('poll('));
    const caps = src.slice(src.indexOf('private async loadCaps'), src.indexOf('private async afterSignIn'));
    expect(caps).toMatch(/^private async loadCaps\(run: number\)/);
    expect(caps).not.toContain('this.boots.current()');
    const check = caps.indexOf('if (!this.boots.live(run)) return;');
    expect(check).toBeGreaterThan(caps.indexOf('catch'));
    expect(caps.indexOf('this.onLost()')).toBeGreaterThan(check);
    expect([...src.matchAll(/this\.afterSignIn\(\)/g)]).toHaveLength(0);
    expect([...src.matchAll(/this\.loadCaps\(\)/g)]).toHaveLength(0);
  });

  it('the down-reprobe dies with its mount too', () => {
    const closure = src.slice(src.indexOf('private startReprobe'), src.indexOf('private stopReprobe'));
    expect(closure).not.toContain('live(run)');   // the check below is in boot's closure, not here
    const check = src.slice(src.indexOf('this.startReprobe(async () => {'), src.indexOf('}, true);'));
    expect([...check.matchAll(/if \(!this\.boots\.live\(run\)\) return true;/g)].length)
      .toBeGreaterThanOrEqual(2);
  });
});
