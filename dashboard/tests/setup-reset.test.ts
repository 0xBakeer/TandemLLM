// VIS-82 (QA VIS-83): the reset confirmation rules.
import { describe, expect, it } from 'vitest';
import type { SetupResetItem } from '../src/api/types';
import { bytesText, confirmMatches, confirmState, confirmValue, once } from '../src/setup/reset-logic';

const item = (more: Partial<SetupResetItem> = {}): SetupResetItem => ({ path: '~/.qwen38-spark-engine/disk', label: 'Disk cache', bytes: 12e9, exists: true, files: 40, at_least: false, kind: 'path', action: 'remove', note: null, restart: false, refused: null, ...more });
const level3 = { confirm_word: 'delete caches', bytes: 12e9, items: [item()] };

describe('confirmMatches', () => {
  it('needs the exact word', () => {
    for (const typed of ['', 'delete cache', 'DELETE CACHES', 'delete caches ', ' delete caches', 'Delete caches']) expect(confirmMatches('delete caches', typed), JSON.stringify(typed)).toBe(false);
    expect(confirmMatches('delete caches', 'delete caches')).toBe(true);
  });
  it('a level without a word is a plain confirm', () => {
    expect(confirmMatches(null, '')).toBe(true);
  });
});

describe('confirmState', () => {
  it('enables only on the exact word', () => {
    expect(confirmState(level3, 'delete cache', { busy: false, sending: false })).toEqual({ enabled: false, reason: 'word' });
    expect(confirmState(level3, 'delete caches', { busy: false, sending: false })).toEqual({ enabled: true, reason: null });
  });
  it('a busy engine refuses every server level', () => {
    expect(confirmState(level3, 'delete caches', { busy: true, sending: false })).toEqual({ enabled: false, reason: 'busy' });
  });
  it('an empty plan cannot be confirmed', () => {
    expect(confirmState({ ...level3, items: [item({ exists: false, bytes: 0 })] }, 'delete caches', { busy: false, sending: false })).toEqual({ enabled: false, reason: 'empty' });
    // an item the server will not remove does not count
    expect(confirmState({ ...level3, items: [item({ refused: 'the active profile' })] }, 'delete caches', { busy: false, sending: false })).toEqual({ enabled: false, reason: 'empty' });
  });
  it('sending disables it (no second request)', () => {
    expect(confirmState(level3, 'delete caches', { busy: false, sending: true }).enabled).toBe(false);
  });
  it('level 1 (local) needs no word', () => {
    expect(confirmState(null, '', { busy: true, sending: false })).toEqual({ enabled: true, reason: null });
  });
  it('the body carries the word, or true for a plain confirm', () => {
    expect(confirmValue({ confirm_word: 'reset' }, 'reset')).toBe('reset');
    expect(confirmValue({ confirm_word: null }, '')).toBe(true);
  });
});

describe('once', () => {
  it('a double click sends one request', async () => {
    let calls = 0;
    const go = once(async () => {
      calls++;
      await new Promise((r) => setTimeout(r, 10));
      return calls;
    });
    const [a, b] = await Promise.all([go(), go()]);
    expect(calls).toBe(1);
    expect(a).toBe(b);
  });
});

describe('bytesText', () => {
  it('reads sizes in binary units', () => {
    expect(bytesText(12 * 2 ** 30)).toBe('12.0 GiB');
    expect(bytesText(300 * 2 ** 20)).toBe('300 MiB');
    expect(bytesText(512)).toBe('512 B');
  });
});
