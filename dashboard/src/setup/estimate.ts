// The memory estimate, the TypeScript copy of server/setup/estimate.py (the single source; D5). The
// page computes it on every change so the memory bar never waits, then confirms it with
// POST /v1/setup/estimate. tests/setup-estimate.test.ts holds this copy to the server's fixture.
//
//   parts = the weights + the drafters of that profile + every component that is on + the side
//           slots' fixed part + the engine's own parts (model.yaml memory.parts) + the OS + KV
//   KV typical = sum over slots of min(window, 32,768) x kv bytes/token; KV worst = full windows;
//           without the pool the worst case is held from the start
//   fits = total <= device - floor; the context that fits = (device - floor - fixed) / kv bytes
// Every figure is GiB (2^30 bytes).

import type { SetupEstimate, SetupEstimatePart, SetupModel, SetupProfile, SetupSelection } from '../api/types';

export const GIB = 2 ** 30;
/** what a slot typically holds: the KV of a 32k-token conversation */
export const KV_TYPICAL_TOKENS = 32_768;

const r2 = (x: number) => Math.round(x * 100) / 100;

export function deviceOf(m: SetupModel, s: Pick<SetupSelection, 'device'>) {
  return m.devices.find((d) => d.id === s.device) ?? m.devices[0];
}

export function profileOf(m: SetupModel, id: string): SetupProfile {
  return m.profiles.find((p) => p.id === id) ?? m.profiles.find((p) => p.recommended) ?? m.profiles[0];
}

/** Which bar category a part belongs to (the memory bar's colours and legend). */
export type PartCategory = 'model' | 'helpers' | 'caches' | 'engine' | 'system' | 'chats';

export function categoryOf(m: SetupModel, id: string): PartCategory {
  if (id === 'weights' || id === 'drafters') return 'model';
  if (id === 'kv') return 'chats';
  if (id === 'os') return 'system';
  const c = m.components.find((x) => x.id === id);
  if (c) return c.group === 'roles' ? 'helpers' : c.group === 'speed' ? 'model' : 'caches';
  return 'engine';
}

export function estimate(m: SetupModel, s: SetupSelection, availableGb: number | null = null): SetupEstimate {
  const dev = deviceOf(m, s);
  const prof = profileOf(m, s.profile);
  const par = m.parallelism;
  const parts: SetupEstimatePart[] = [];

  parts.push({ id: 'weights', label: `Weights (${prof.label})`, gb: prof.weights_gib, kind: 'fixed' });
  for (const c of m.components) {
    const on = c.required || (s.components[c.id] ?? c.default);
    if (!on) continue;
    const gb = c.id === 'drafters' && prof.drafters_gib != null ? prof.drafters_gib : c.memory_gib;
    if (gb <= 0 && c.memory_max_gib == null) continue;
    const part: SetupEstimatePart = { id: c.id, label: c.label, gb, kind: c.memory_max_gib != null ? 'reserve' : 'fixed' };
    if (c.memory_max_gib != null) part.worst_gb = c.memory_max_gib;
    parts.push(part);
  }
  const slots = Math.max(1, Math.round(s.slots));
  if (slots > 1) parts.push({ id: 'side_slots', label: `Extra chat slots (${slots - 1})`, gb: (slots - 1) * par.side_slot_const_gib, kind: 'fixed' });
  for (const p of m.memory.parts) parts.push({ id: p.id, label: p.label, gb: p.gib, kind: p.kind });
  parts.push({ id: 'os', label: 'The system and other programs', gb: dev.reserved_gib, kind: 'fixed' });

  const w0 = s.window;
  const side = s.side_window ?? s.window;
  const worstTokens = w0 + (slots - 1) * side;
  const typTokens = Math.min(w0, KV_TYPICAL_TOKENS) + (slots - 1) * Math.min(side, KV_TYPICAL_TOKENS);
  const kvb = par.kv_bytes_per_token;
  const kvWorst = (worstTokens * kvb) / GIB;
  const kvTyp = (typTokens * kvb) / GIB;
  parts.push({ id: 'kv', label: 'Conversations (KV cache)', gb: s.kv_pool ? kvTyp : kvWorst, kind: 'kv', worst_gb: kvWorst });

  const fixed = parts.filter((p) => p.kind !== 'kv').reduce((a, p) => a + p.gb, 0);
  const kv = s.kv_pool ? kvTyp : kvWorst;
  const total = fixed + kv;
  const budget = dev.total_gib - s.floor_gb;
  const room = budget - fixed;
  const notes: string[] = [];
  if (!s.kv_pool) notes.push('Without the shared pool every slot holds its full window from the start.');
  if (fixed + kvWorst > budget) notes.push('Every chat at its full window does not fit; the engine pauses or parks a chat before that.');

  return {
    contract_version: '1.0',
    total_gb: r2(total),
    available_gb: availableGb,
    device_gb: r2(dev.total_gib),
    floor_gb: s.floor_gb,
    parts: parts.map((p) => ({ ...p, gb: r2(p.gb), ...(p.worst_gb != null ? { worst_gb: r2(p.worst_gb) } : {}) })),
    fits: total <= budget + 1e-9,
    context_tokens_fit: Math.max(0, Math.floor((room * GIB) / kvb)),
    kv_bytes_per_token: kvb,
    kv_worst_gb: r2(kvWorst),
    room_gb: r2(room),
    notes,
    fits_worst: fixed + kvWorst <= budget + 1e-9,
    budget_gb: r2(budget),
  };
}

/** How far over the budget (0 when it fits). */
export function overBy(e: SetupEstimate): number {
  const budget = e.budget_gb ?? e.device_gb - e.floor_gb;
  return Math.max(0, r2(e.total_gb - budget));
}
