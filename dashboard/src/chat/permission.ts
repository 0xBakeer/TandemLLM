// Permission modes (Khaled, 2026-10-13: "too many allow prompts for cd and ls; I want a yolo
// mode"). A chat runs in one of three modes, the default for new chats sits in the settings:
//   ask   every tool asks, unless its policy says always (the old behaviour)
//   auto  commands and reads in the sandbox run without asking; writes and the internet still ask
//   yolo  nothing asks: the model runs every tool it calls (a never policy still refuses)
// The server decides for real; this copy drives the mock and the words on screen.

import type { ChatSettings, PermissionMode, ToolApproval, ToolPolicy, ToolRisk } from './types';

export const PERMISSION_MODES: PermissionMode[] = ['ask', 'auto', 'yolo'];

export function modeOf(s: Pick<ChatSettings, 'permission_mode'> | null | undefined): PermissionMode {
  return s?.permission_mode ?? 'auto';
}

/** the pill's word */
export function modeWord(m: PermissionMode): string {
  return m === 'yolo' ? 'YOLO' : m;
}
/** one line each, for the popover and the settings */
export function modeDesc(m: PermissionMode): string {
  if (m === 'ask') return 'Every tool asks first, unless its policy says always.';
  if (m === 'auto') return 'Commands and reads in the sandbox run; writes and the internet ask.';
  return 'Nothing asks. The model runs every tool it calls.';
}

export interface CallDecision {
  run: 'ask' | 'allow' | 'deny';
  /** who allowed it when it runs without a prompt; null when it asks or is refused */
  approved: ToolApproval | null;
}

/** what happens before a tool call runs */
export function decideCall(mode: PermissionMode, policy: ToolPolicy, risk: ToolRisk | undefined, autoAcceptEdits = false): CallDecision {
  if (policy === 'never') return { run: 'deny', approved: null };
  if (policy === 'always') return { run: 'allow', approved: 'rule' };
  if (mode === 'yolo') return { run: 'allow', approved: 'yolo' };
  if (risk === 'write' && autoAcceptEdits) return { run: 'allow', approved: 'rule' };
  if (mode === 'auto' && (risk === 'exec' || risk === 'read')) return { run: 'allow', approved: 'auto' };
  return { run: 'ask', approved: null };
}

/** the badge on a tool card that ran without a prompt */
export function approvedWord(a: ToolApproval): string {
  return a === 'rule' ? 'always' : a === 'user' ? '' : a;
}
export function approvedTitle(a: ToolApproval): string {
  if (a === 'auto') return 'ran without asking: auto mode';
  if (a === 'yolo') return 'ran without asking: YOLO mode';
  if (a === 'rule') return 'ran without asking: the tool is always allowed';
  return 'you allowed it';
}
