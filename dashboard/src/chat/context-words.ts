// The Context panel's figures in words: a value, or why there is none. A bare "—" told the reader
// nothing (2026-10-14: "KV for this chat —", "resident prefix —", "running, 0 B", "— tokens").

import { bytes, compact, exact } from '../lib/format';
import type { Attachment, ContextReport, HarnessMemory, Run, SandboxCallData, SandboxStatus } from './types';

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

/** a Now figure while a run is live: the value, else why there is none at this moment. Live
 *  (2026-10-16, an agent turn on Nemotron 3.5 Super VL) every gap read "not tracked for this
 *  model": between two model calls, in a call's first 50 tokens, before a call had ended. */
export function liveFigure(v: number | null | undefined, fmt: (v: number) => string, r: Run, kind: 'tps' | 'round' | 'accepted', backend: Backend, drafter: string | null): string {
  if (v != null && Number.isFinite(v)) return fmt(v);
  if (backend && backend !== 'engine') return NOT_TRACKED;
  if (kind !== 'tps' && !drafter) return 'no drafter';
  if (r.phase === 'queued') return 'queued';
  if (r.phase === 'prefill') return 'prefilling';
  if (r.phase === 'tool' || r.phase === 'permission') return kind === 'tps' ? 'between model calls' : 'measured when a model call ends';
  return kind === 'tps' ? 'warming up, the first 50 tokens' : 'measured when a model call ends';
}

/** the slot row: the engine's lane tag while a request runs, else who holds the slot. It said
 *  "none held" under "KV for this chat: 32,890 tokens, in the engine's slot" (2026-10-16). */
export function slotWords(slot: string | null, mem: Mem): string {
  if (slot) return slot;
  if (mem.kv_where === 'slot') return 'held by this chat';
  if (mem.kv_where === 'park' || mem.kv_where === 'other') return 'held by another chat';
  return 'none held';
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

/** the call card's line when the sandbox could not run the call (SRV-481, SRV-485) */
export function sandboxCallWords(d: SandboxCallData): string {
  if (d.reason === 'oom') return `Out of memory (${d.limit_gib} GiB sandbox limit): the kernel killed a process of this call.`;
  const gave = d.reason === 'no_memory' && d.engine_released_gib != null ? `, the engine gave back ${d.engine_released_gib} GiB` : '';
  return `The sandbox did not start: ${d.why ?? 'no room'} (waited ${d.waited_s ?? 0} s${gave}). The command did not run.`;
}

/** an attachment chip's figures, and for a file too long to include, what happens to it
 *  (SRV-487, VIS-169) */
export function attachmentWords(a: Attachment): { size: string; warn: string | null } {
  const size = [bytes(a.bytes, 0), a.tokens != null ? `${compact(a.tokens)} tok` : ''].filter(Boolean).join(' · ');
  const warn = a.inline === false ? `${a.name} is too long to include (about ${exact(a.tokens)} tokens): the model gets one line with its name, size and path${a.path ? ` (${a.path})` : " in the chat's workspace"}.` : null;
  return { size, warn };
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

const MEMORY_OFF: Record<string, string> = { profile: 'off in this profile', unset: 'off in this profile', model_fit: 'off for this model', backend: 'off for this backend', header: 'off for this request' };

/** the memory notes in the last turn (SRV-471): how many the index named and what they cost, the
 *  notes sent in full with the message, or why the memory is off */
export function memoryWords(harness: ContextReport['harness'], backend: Backend): string {
  if (backend && backend !== 'engine') return NOT_TRACKED;
  if (!harness) return HARNESS_PENDING;
  const m: HarnessMemory | null | undefined = harness.memory;
  if (!m) return 'not reported by this engine';
  if (m.mode === 'off') return MEMORY_OFF[m.why ?? ''] ?? 'off';
  if (m.mode === 'shadow') return 'shadow: the model may write notes, none is read back';
  const idx = m.index_notes ? `${exact(m.index_notes)} note${m.index_notes === 1 ? '' : 's'} in the index, ${exact(m.index_tokens)} tokens` : 'on, no notes yet';
  return m.bodies ? `${idx}; ${exact(m.bodies)} in full for this message, ${exact(m.body_tokens)} tokens` : idx;
}
