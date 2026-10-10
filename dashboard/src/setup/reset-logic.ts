// The reset levels' rules (D9, VIS-82), without the DOM: the typed word must match exactly (no
// trimming, no case folding), an empty plan cannot be confirmed, a busy engine refuses every server
// level, and level 1 is the browser's own.

import type { SetupResetLevel } from '../api/types';

export type ResetLevelNo = 1 | 2 | 3 | 4 | 5;

/** The confirmation is valid: the exact word for a typed level, nothing for a plain confirm. */
export function confirmMatches(word: string | null, typed: string): boolean {
  if (word == null) return true;
  return typed === word;
}

export interface ConfirmState {
  enabled: boolean;
  reason: 'busy' | 'empty' | 'word' | 'sending' | null;
}

export function confirmState(level: Pick<SetupResetLevel, 'confirm_word' | 'items' | 'bytes'> | null, typed: string, opts: { busy: boolean; sending: boolean }): ConfirmState {
  if (opts.sending) return { enabled: false, reason: 'sending' };
  if (level == null) return { enabled: true, reason: null }; // level 1, local
  if (opts.busy) return { enabled: false, reason: 'busy' };
  if (!level.items.some((i) => i.exists)) return { enabled: false, reason: 'empty' };
  if (!confirmMatches(level.confirm_word, typed)) return { enabled: false, reason: 'word' };
  return { enabled: true, reason: null };
}

/** The request body: the typed word, or `true` for a level without one. */
export function confirmValue(level: Pick<SetupResetLevel, 'confirm_word'>, typed: string): string | boolean {
  return level.confirm_word == null ? true : typed;
}

/** A one-shot guard: the first call runs, the rest wait for it (a double click sends one request). */
export function once<T>(fn: () => Promise<T>): () => Promise<T> {
  let p: Promise<T> | null = null;
  return () => (p ??= fn().finally(() => (p = null)));
}

export function bytesText(b: number): string {
  if (b >= 2 ** 30) return `${(b / 2 ** 30).toFixed(1)} GiB`;
  if (b >= 2 ** 20) return `${(b / 2 ** 20).toFixed(0)} MiB`;
  if (b >= 1024) return `${(b / 1024).toFixed(0)} KiB`;
  return `${b} B`;
}
