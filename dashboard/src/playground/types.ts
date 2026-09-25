// The Playground's data: messages in the OpenAI chat shape, the setup (system prompt,
// parameters, tools), and what the engine returns per turn (SRV-27 usage / timings / metrics).

import type { ChatMetrics, ChatTimings, ChatUsage } from '../api/types';

export type Role = 'system' | 'user' | 'assistant' | 'tool';
export const ROLES: Role[] = ['system', 'user', 'assistant', 'tool'];

export interface ToolCall {
  id: string;
  type: 'function';
  function: { name: string; arguments: string };
}

export interface Message {
  /** a client-side id for keyed rendering and edits */
  uid: string;
  role: Role;
  content: string;
  /** an assistant turn that called tools */
  tool_calls?: ToolCall[];
  /** a tool result: which call it answers */
  tool_call_id?: string;
  name?: string;
  /** reasoning shown with an assistant turn (never sent back to the engine) */
  reasoning?: string;
  /** the engine's figures for an assistant turn */
  stats?: TurnStats;
}

export type FinishReason = 'stop' | 'length' | 'tool_calls' | 'timeout' | 'aborted' | 'error' | string;

export interface TurnStats {
  finish: FinishReason | null;
  usage?: ChatUsage;
  timings?: ChatTimings;
  metrics?: ChatMetrics;
  error?: { message: string; type?: string } | null;
  /** wall-clock as seen by the browser, for a stream that ended without timings */
  elapsedMs?: number;
  chunks?: number;
  bytes?: number;
}

/** A parameter that has not been touched is `undefined` and is not sent. */
export interface ParamValues {
  temperature?: number;
  top_p?: number;
  top_k?: number;
  max_tokens?: number;
  stop?: string[];
  presence_penalty?: number;
  frequency_penalty?: number;
  repetition_penalty?: number;
  no_repeat_ngram_size?: number;
  seed?: number;
  /** undefined = the template's default (on); false / true are sent explicitly */
  thinking?: boolean;
  reasoning_effort?: 'low' | 'medium' | 'high' | 'xhigh';
  max_reasoning_tokens?: number;
  draft_temperature?: number;
  reasoning_format?: 'tags' | 'reasoning_content' | 'both';
  /** undefined = on */
  include_usage?: boolean;
}

export type ToolChoice = 'auto' | 'none' | 'required' | { type: 'function'; function: { name: string } };

export interface ToolDef {
  type: 'function';
  function: { name: string; description?: string; parameters?: Record<string, unknown> };
}

export interface Setup {
  system: string;
  params: ParamValues;
  /** the tools editor's text; parsed and validated by tools.ts */
  toolsText: string;
  tool_choice: ToolChoice;
}

export interface Preset extends Setup {
  version: 1;
  name: string;
  savedAt: string;
}

export function emptySetup(): Setup {
  return { system: '', params: {}, toolsText: '', tool_choice: 'auto' };
}

let uidCounter = 0;
export function uid(): string {
  uidCounter++;
  return `m${Date.now().toString(36)}${uidCounter.toString(36)}`;
}
