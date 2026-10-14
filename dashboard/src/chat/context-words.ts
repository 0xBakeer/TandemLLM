// The Context panel's figures in words: a value, or why there is none. A bare "—" told the reader
// nothing (2026-10-14: "KV for this chat —", "resident prefix —", "running, 0 B", "— tokens").

import { bytes, exact } from '../lib/format';
import type { ContextReport, HarnessMemory, SandboxStatus } from './types';

export const NOT_TRACKED = 'not tracked for this model';
export const AFTER_TURN = 'measured after the first turn';
export const NEXT_TURN = 'counted on its next turn';
/** no usage frame from the engine yet (a new chat, another server) */
export const HARNESS_PENDING = 'its picks arrive with the next turn';

type Mem = ContextReport['memory'];
type Backend = ContextReport['model']['backend'];

/** a figure of the last turn: the value, else why it has none */
export function turnFigure(v: number | null | undefined, fmt: (v: number) => string, hadTurn: boolean): string {
  if (v != null && Number.isFinite(v)) return fmt(v);
  return hadTurn ? NOT_TRACKED : AFTER_TURN;
}

/** where the engine keeps this chat's rows */
export function kvWords(mem: Mem, backend: Backend): string {
  if (backend && backend !== 'engine') return NOT_TRACKED;
  if (mem.kv_tokens != null && mem.kv_where === 'slot') return `${exact(mem.kv_tokens)} tokens, in the engine's slot`;
  if (mem.kv_tokens != null && mem.kv_where === 'park') return `${exact(mem.kv_tokens)} tokens, parked in RAM`;
  switch (mem.kv_where) {
    case 'nvme_spill':
      return 'on the NVMe, read back next turn';
    case 'disk_tier':
      return 'in the disk tier, read back next turn';
    case 'ram_snapshot':
      return 'a snapshot in RAM';
    case 'other':
      return 'another chat holds the slot';
    case 'none':
      return 'none held';
    default:
      return 'the engine does not report its caches';
  }
}

/** the resident prefix's snapshots of this chat */
export function residentWords(mem: Mem, backend: Backend): string {
  if (backend && backend !== 'engine') return NOT_TRACKED;
  if (mem.resident_owner === 'this') return bytes(mem.resident_bytes ?? 0, 1);
  if (mem.resident_owner === 'other') return 'held by another chat';
  if (mem.resident_owner === 'none') return 'none held';
  return mem.resident_bytes != null ? bytes(mem.resident_bytes, 1) : 'the engine does not report its caches';
}

/** the sandbox's state, with its memory in use while it runs */
export function sandboxWords(sb: SandboxStatus): string {
  if (sb.state !== 'running') return sb.state.replace('_', ' ');
  return sb.memory_bytes != null ? `running, ${bytes(sb.memory_bytes, 0)}` : 'running, memory not reported by the runtime';
}

/** a skill's size in the prompt */
export function skillTokens(tokens: number | null | undefined): string {
  return tokens != null ? `${exact(tokens)} tokens` : NEXT_TURN;
}

/** a part of the context bar: the engine counts the system prompt and the tool schemas in its
 *  usage frame, so before the first frame (or on a server without one) they have no count yet */
export function partWords(kind: string, tokens: number, split: ContextReport['split'], backend: Backend): string {
  if ((kind === 'system' || kind === 'tools') && !split && !tokens) return backend && backend !== 'engine' ? 'not tracked' : 'next turn';
  return exact(tokens);
}

const MEMORY_OFF: Record<string, string> = { profile: 'off in this profile', model_fit: 'off for this model', backend: 'off for this backend', header: 'off for this request' };

/** the memory notes in the last turn (SRV-471): how many the index named and what they cost, the
 *  notes sent in full with the message, or why the memory is off */
export function memoryWords(harness: ContextReport['harness'], backend: Backend): string {
  if (backend && backend !== 'engine') return NOT_TRACKED;
  if (!harness) return HARNESS_PENDING;
  const m: HarnessMemory | null | undefined = harness.memory;
  if (!m) return MEMORY_OFF.profile;
  if (m.mode === 'off') return MEMORY_OFF[m.why ?? ''] ?? 'off';
  if (m.mode === 'shadow') return 'shadow: the model may write notes, none is read back';
  const idx = m.index_notes ? `${exact(m.index_notes)} note${m.index_notes === 1 ? '' : 's'} in the index, ${exact(m.index_tokens)} tokens` : 'on, no notes yet';
  return m.bodies ? `${idx}; ${exact(m.bodies)} in full for this message, ${exact(m.body_tokens)} tokens` : idx;
}
