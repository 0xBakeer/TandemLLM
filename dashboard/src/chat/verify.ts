// SRV-532: the verify phase in the thread. A run that built or changed a page or app checks it in
// a browser before it ends: the check is a tool card of the harness's own (`harness: 'verify'`),
// the Now line says "Verifying the app ...", and the answer ends with a "Checked / Not checked"
// block (a `verify` part) that chatd builds from the run's evidence.

import type { RunPhase, ToolCallPart, VerifyPart } from './types';

export interface VerifyBlock {
  lead: string;
  checked: string[];
  unchecked: string[];
}

/** the part as its lead line and its lists; a part without the lists (none yet) reads its text's
 *  "Checked:" and "Not checked:" lines, whose items are split on "; " */
export function verifyBlock(p: Pick<VerifyPart, 'text' | 'checked' | 'not_checked'>): VerifyBlock {
  const out: VerifyBlock = { lead: '', checked: [], unchecked: [] };
  for (const line of p.text.split('\n')) {
    const m = /^(Checked|Not checked): (.*?)\.?$/.exec(line);
    if (!m) out.lead = out.lead ? `${out.lead} ${line}` : line;
    else (m[1] === 'Checked' ? out.checked : out.unchecked).push(...m[2].split('; ').filter(Boolean));
  }
  if (p.checked) out.checked = p.checked;
  if (p.not_checked) out.unchecked = p.not_checked;
  return out;
}

/** the block's tone: ok for a passed check, warn for problems or changes after the last check,
 *  err when no check could run */
export function verifyTone(state: VerifyPart['state']): 'ok' | 'warn' | 'err' {
  return state === 'passed' ? 'ok' : state === 'not_run' ? 'err' : 'warn';
}

export function isHarnessCall(t: Pick<ToolCallPart, 'harness'>): boolean {
  return t.harness === 'verify';
}

/** the Now line's figure acts while a tool or the verify check runs */
export function actingPhase(phase: RunPhase): boolean {
  return phase === 'tool' || phase === 'verifying';
}
