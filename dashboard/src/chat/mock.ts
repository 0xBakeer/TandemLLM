// The mock chat backend, in the browser: seeded conversations with a branch tree, scripted runs
// (thinking, text, tool calls with permission prompts, artifacts for the panes), one run that is
// already on when the tab opens ("started on another device"), a reconnect scenario, the settings
// stores, a sandbox with files, and a terminal. Every event goes through the same reducer the
// real stream uses, and the event log serves a catch-up from any Last-Event-ID. No network.

import type { StreamState } from '../lib/sse';
import { VersionConflict, type ChatBackend, type SendOptions, type StreamHandler, type Subscription, type TerminalSession } from './backend';
import { ARTIFACTS, CAPABILITIES, COMPACTION, DEFAULT_SETTINGS, FILES, MCP, MODEL, PACKAGES, PRESETS, PROMPTS, TOOLS, contextFor, iso, sandboxFor } from './mock-data';
import { applyEvent } from './store';
import type { Attachment, Capabilities, ChatEvent, ChatSettings, Compaction, Conversation, ConversationStatus, ConversationSummary, ConversationUi, ConversationUsage, Folder, McpServer, McpServerDraft, Message, PackageInfo, PaneRef, Part, Preset, Prompt, Run, SandboxFile, SandboxStatus, StampedEvent, ToolCallPart, ToolInfo, ToolPolicy, TurnStats, UserToolkit } from './types';
import { localId } from './types';

// ---- the branch tree --------------------------------------------------------------------------

interface Node {
  msg: Message;
  parent: string | null;
  children: string[];
  /** which child is on the active path */
  active: number;
}

interface ConvState {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  pinned: boolean;
  archived: boolean;
  version: number;
  ui: ConversationUi;
  folder_id: string | null;
  settings: ChatSettings;
  nodes: Map<string, Node>;
  roots: string[];
  rootActive: number;
  run: Run | null;
  runner: Runner | null;
  queue: Conversation['queue'];
  compactions: Compaction[];
  sandbox: SandboxStatus;
  events: StampedEvent[];
  seq: number;
  subs: Set<StreamHandler>;
  /** this chat's prompt was seen by the engine before (its cache is warm) */
  seen: boolean;
  status: ConversationStatus;
  now: string | null;
  /** client_msg_id → message id: a retried send makes no second message */
  clientIds: Map<string, string>;
  /** the cold cache the seed plays (c_exl3: 150,000 to re-read) */
  cold?: number;
  /** tokens of history beyond the seeded messages (a long chat) */
  historyTokens?: number;
}

const EMPTY_UI: ConversationUi = { panes: [], active: null, split: 0.6 };

// ---- scripts ----------------------------------------------------------------------------------

type Step =
  | { kind: 'prefill'; total: number }
  | { kind: 'thinking'; text: string }
  | { kind: 'text'; text: string }
  | { kind: 'tool'; name: string; input: Record<string, unknown>; output: string[]; ms: number; pane?: PaneRef; error?: boolean; denied?: string }
  | { kind: 'done'; finish?: TurnStats['finish'] };

const WORD_MS = 34;

function script(text: string, longChat: boolean): Step[] {
  const t = text.toLowerCase();
  const pre: Step[] = longChat ? [{ kind: 'prefill', total: 12_000 + Math.round(text.length * 1.3) }] : [];
  if (/\b(fix|flaky|test|bug|fail)/.test(t)) return [...pre, ...CODING_SCRIPT];
  if (/\b(page|site|html|landing|design|shop)/.test(t)) return [...pre, ...WEB_SCRIPT];
  if (/\b(plan|meal|week|list|schedule)/.test(t)) return [...pre, ...PLAN_SCRIPT];
  if (/\b(diff|rename|refactor|edit|change)/.test(t)) return [...pre, ...EDIT_SCRIPT];
  if (/\b(hi|hello|hey)\b/.test(t)) return [...pre, { kind: 'text', text: 'Hi. The engine is up, the drafter is warm, and this chat has a sandbox of its own. Ask away.' }, { kind: 'done' }];
  return [...pre, ...EXPLAIN_SCRIPT(text)];
}

const EXPLAIN_SCRIPT = (q: string): Step[] => {
  const topic = q.trim().replace(/[?.!]+$/, '').slice(0, 72) || 'the question';
  return [
    { kind: 'thinking', text: `The user asks about "${topic}". Give the mechanism first, then one worked example in code, then the two numbers that decide it. Keep it under two hundred words and do not pad.` },
    {
      kind: 'text',
      text: `Here is the short version. The engine verifies a **block** of drafted tokens in one forward pass; the number it keeps per block is what the Context panel calls *tokens per round*. More kept per round means fewer passes per answer, so the decode rate rises with it.\n\n\`\`\`python\ndef tokens_per_round(accepted: list[int]) -> float:\n    """Mean accepted tokens per verify pass, plus the target's own token."""\n    return sum(accepted) / len(accepted) + 1\n\nprint(tokens_per_round([3, 4, 2, 5]))  # 4.5\n\`\`\`\n\n- **Acceptance** is the share of drafted tokens the target agrees with.\n- **Tokens per round** is the mean run length that survives, plus one.\n- The drafter loses first on code with unusual identifiers; prose stays high.\n\nThat is the whole mechanism. The rest is tuning the block size against the acceptance curve you measure on your own text.`,
    },
    { kind: 'done' },
  ];
};

const CODING_SCRIPT: Step[] = [
  { kind: 'thinking', text: 'A flaky test is usually a timing assumption. Read the test and the worker before running anything, then run the one test many times to see how often it fails and with which value.' },
  { kind: 'tool', name: 'read_file', input: { path: 'tests/test_worker.py' }, output: [FILES['/work/sync/tests/test_worker.py']], ms: 600, pane: { kind: 'files', ref: '/work/sync/tests/test_worker.py', title: 'test_worker.py' } },
  { kind: 'text', text: 'The second test asserts that `drain(deadline_s=0.1)` finishes between 5 and 12 jobs with a 10 ms tick. The loop checks the deadline **before** each job, so a job that starts at 99 ms still runs and the count lands on 10 or 11; under load the sleep stretches and it lands on 4. Let me measure how often.' },
  {
    kind: 'tool',
    name: 'bash',
    input: { command: 'uv run pytest -q --count=200 tests/test_worker.py::test_drain_stops_at_deadline', cwd: '/work/sync' },
    output: ['\x1b[1m============================= test session starts ==============================\x1b[0m', 'collected 200 items', '', '........................................\x1b[31mF\x1b[0m..................................... [ 40%]', '..........................\x1b[31mF\x1b[0m............................................. [ 80%]', '.......................................\x1b[31mF\x1b[0m [100%]', '', '\x1b[31m\x1b[1m================================== FAILURES ===================================\x1b[0m', '\x1b[31mE   assert 5 <= 4 <= 12\x1b[0m', '\x1b[31m\x1b[1m3 failed\x1b[0m, \x1b[32m197 passed\x1b[0m in 21.4s'],
    ms: 9_000,
    error: true,
    pane: { kind: 'terminal', ref: 'call', title: 'pytest' },
    denied: 'Without running the test I can still read the loop: the deadline check sits before the job, so the count depends on how long the last `sleep` takes. The fix below stands either way.',
  },
  { kind: 'text', text: '3 of 200 runs fail, each with a count of 4: the last job started inside the window and `sleep` overran. Two changes: check the deadline after the job as well, and tighten the assertion to the range the tick allows.' },
  {
    kind: 'tool',
    name: 'edit_file',
    input: { path: 'src/sync/worker.py', find: 'while time.monotonic() - start < deadline_s:\n            if not self.run_once():\n                break', replace: 'while True:\n            if time.monotonic() - start >= deadline_s:\n                break\n            if not self.run_once():\n                break' },
    output: ['edited src/sync/worker.py: 1 block, +3 -2 lines'],
    ms: 400,
    pane: { kind: 'diff', ref: 'diff:worker', title: 'worker.py' },
    denied: 'Not editing then. The change to make by hand: move the deadline check after `run_once` returns, and set the assertion to `8 <= n <= 11`.',
  },
  { kind: 'tool', name: 'bash', input: { command: 'uv run pytest -q --count=200 tests/test_worker.py', cwd: '/work/sync' }, output: ['collected 400 items', '', '........................................................................ [ 18%]', '........................................................................ [ 36%]', '........................................................................ [ 54%]', '........................................................................ [ 72%]', '........................................................................ [ 90%]', '........................................ [100%]', '\x1b[32m\x1b[1m400 passed\x1b[0m in 38.2s'], ms: 7_000, pane: { kind: 'terminal', ref: 'call', title: 'pytest' }, denied: 'Run the suite yourself when you are ready: `uv run pytest -q --count=200 tests/test_worker.py`.' },
  { kind: 'text', text: '400 of 400 pass. The diff is in the Diff pane; the test now asserts `8 <= n <= 11`, the range a 10 ms tick allows in 100 ms.\n\nOne thing to decide: `drain` now returns one job fewer in the common case, because the check runs after each job. If a caller depends on the old count, keep the old loop and only fix the assertion.' },
  { kind: 'done' },
];

const WEB_SCRIPT: Step[] = [
  { kind: 'thinking', text: 'A small shop page: one html file, no framework, a clear price list, hours, and one call to action. Warm paper and moss, serif type, so it does not read like a template.' },
  { kind: 'text', text: 'One file, no build step. The page has a header, a hero with the one promise, three prices, and the hours. Writing it to the sandbox so the Preview pane can show it.' },
  { kind: 'tool', name: 'write_file', input: { path: 'site/index.html', bytes: ARTIFACTS['preview:shop-page'].content.length }, output: ['wrote site/index.html (2,418 bytes)'], ms: 900, pane: { kind: 'preview', ref: 'preview:shop-page', title: 'index.html' }, denied: 'Then here it is inline instead; copy it into `index.html`.\n\n```html\n' + ARTIFACTS['preview:shop-page'].content.slice(0, 600) + '\n…\n```' },
  { kind: 'text', text: 'The page is in the Preview pane. Three things to change before it goes live: the three prices, the hours in the footer, and the shop name in the title and the header. The palette is three colours (`--ink`, `--paper`, `--moss`) plus one accent for the button, all in the `:root` block at the top.' },
  { kind: 'done' },
];

const PLAN_SCRIPT: Step[] = [
  { kind: 'thinking', text: 'A week of simple meals, one shopping trip, leftovers used twice. Keep the table short and the shopping list in one paragraph.' },
  { kind: 'text', text: ARTIFACTS['md:meal-plan'].content + '\nOpen it in the Markdown pane if you want it on a second screen while you shop.' },
  { kind: 'done' },
];

const EDIT_SCRIPT: Step[] = [
  { kind: 'thinking', text: 'A rename touches the module, its import in the worker, and the test. Show the diff before writing anything.' },
  { kind: 'tool', name: 'read_file', input: { path: 'src/sync/worker.py' }, output: [FILES['/work/sync/src/sync/worker.py']], ms: 500, pane: { kind: 'files', ref: '/work/sync/src/sync/worker.py', title: 'worker.py' } },
  { kind: 'tool', name: 'edit_file', input: { path: 'src/sync/worker.py', find: 'from .queue import Queue', replace: 'from .jobs import Queue' }, output: ['edited src/sync/worker.py: 1 block, +1 -1 lines'], ms: 400, pane: { kind: 'diff', ref: 'diff:worker', title: 'worker.py' }, denied: 'Not writing then. The three places to change by hand: the import in `worker.py`, the file name `queue.py` to `jobs.py`, and the import in the test.' },
  { kind: 'text', text: 'Done in three places: the import in `worker.py`, the module file, and the test. The diff is in the Diff pane. Nothing else imports the module; `grep -rn "sync.queue"` finds no other hit.' },
  { kind: 'done' },
];

// ---- the mock ---------------------------------------------------------------------------------

export interface MockOptions {
  scenario?: string | null;
  now?: () => number;
}

export class MockChatBackend implements ChatBackend {
  readonly kind = 'mock' as const;
  private convs = new Map<string, ConvState>();
  private folders: Folder[] = [
    { id: 'f_engine', name: 'Engine' },
    { id: 'f_everyday', name: 'Everyday' },
  ];
  private toolList: ToolInfo[] = TOOLS.map((t) => ({ ...t }));
  private pkgs: PackageInfo[] = PACKAGES.map((p) => ({ ...p }));
  private kits: UserToolkit[] = [];
  private mcp: McpServer[] = MCP.map((m) => ({ ...m }));
  private promptList: Prompt[] = PROMPTS.map((p) => ({ ...p }));
  private presetList: Preset[] = PRESETS.map((p) => ({ ...p }));
  private scenario: string | null;
  private signedIn = true;
  /** the conversation whose run was on before the tab opened */
  readonly attachedId = 'c_sync';
  /** the chat whose last answer was interrupted by a restart (Continue resumes it) */
  readonly interruptedId = 'c_sql';

  constructor(opts: MockOptions = {}) {
    this.scenario = opts.scenario ?? null;
    this.seed();
  }

  // ---- seeds ----------------------------------------------------------------------------------

  private seed(): void {
    const now = Date.now();
    const mk = (id: string, title: string, ago: number, opts: Partial<ConvState> & { turns: [string, Part[], TurnStats?][] }): ConvState => {
      const st: ConvState = {
        id,
        title,
        created_at: iso(-ago - 600_000, now),
        updated_at: iso(-ago, now),
        pinned: false,
        archived: false,
        version: 1,
        ui: EMPTY_UI,
        folder_id: null,
        settings: { ...DEFAULT_SETTINGS, policies: {} },
        nodes: new Map(),
        roots: [],
        rootActive: 0,
        run: null,
        runner: null,
        queue: [],
        compactions: [],
        sandbox: sandboxFor(false),
        events: [],
        seq: 0,
        subs: new Set(),
        seen: true,
        status: 'idle',
        now: null,
        clientIds: new Map(),
        ...opts,
      };
      let parent: string | null = null;
      let t = now - ago - 600_000;
      for (const [u, parts, stats] of opts.turns) {
        const um = this.node(st, { id: localId('m'), role: 'user', created_at: new Date(t).toISOString(), parent_id: parent, run_id: null, parts: [{ type: 'text', text: u }], status: 'done' }, parent);
        t += 40_000;
        const am = this.node(st, { id: localId('m'), role: 'assistant', created_at: new Date(t).toISOString(), parent_id: um, run_id: localId('run'), parts, status: 'done', stats: stats ?? statsFor(parts) }, um);
        t += 60_000;
        parent = am;
      }
      this.convs.set(id, st);
      return st;
    };

    mk('c_explain', 'How tokens per round sets the decode speed', 2 * 3_600_000, { turns: [['How does tokens per round relate to decode speed? One code example.', EXPLAIN_SCRIPT('How does tokens per round relate to decode speed').filter((s) => s.kind !== 'done').map(stepToPart)]] });
    mk('c_shop', 'Landing page for a bike repair shop', 26 * 3_600_000, {
      pinned: true,
      turns: [['Make a one-page site for a bike repair shop: prices, hours, one button to book.', WEB_SCRIPT.filter((s) => s.kind !== 'done').map(stepToPart)]],
    });
    mk('c_rename', 'Rename the queue module to jobs', 50 * 60_000, {
      turns: [['Rename the queue module to jobs and fix the imports.', EDIT_SCRIPT.slice(0, 2).map(stepToPart)]],
    });
    mk('c_meal', 'Weekly meal plan, one shopping trip', 3 * 86_400_000, { folder_id: 'f_everyday', sandbox: { ...sandboxFor(false, 'memory_guard'), error: null }, turns: [['Plan a week of simple dinners with one shopping trip.', PLAN_SCRIPT.filter((s) => s.kind !== 'done').map(stepToPart)]] });
    mk('c_exl3', 'EXL3 4.5 against 6.0 on this box', 5 * 86_400_000, {
      folder_id: 'f_engine',
      pinned: true,
      seen: false,
      cold: 150_000,
      historyTokens: 160_000,
      compactions: [COMPACTION],
      turns: [
        ['Compare EXL3 4.5 and 6.0 for daily use on this box.', [{ type: 'text', text: '4.5 keeps 3.9 GB more room for the KV pool and decodes 7 % faster on this box. 6.0 is closer to BF16 on code: top-1 agreement 0.986 against 0.971. For daily chat and code, 4.5; for a long code review where every token counts, switch to 6.0 for that chat.' }]],
        ['Which one for the weekly code check?', [{ type: 'text', text: '6.0, once a week, on the same 40 prompts. Keep the 4.5 answers beside them and count the lines that differ. If the count stays under 5 % for a month, drop the check.' }]],
      ],
    });
    mk('c_regex', 'A regex for ISO dates with an optional time', 9 * 86_400_000, { turns: [['Regex for ISO dates with an optional time part.', [{ type: 'text', text: '```\n^\\d{4}-\\d{2}-\\d{2}(?:[T ]\\d{2}:\\d{2}(?::\\d{2})?)?$\n```\n\nThe date is required, the time optional, seconds optional inside the time. It accepts a space or a `T` between them.' }]]] });
    const sql = mk(this.interruptedId, 'Slow query on the orders table', 12 * 86_400_000, { folder_id: 'f_engine', turns: [['Why is this query slow? 14 s on 2 million rows.', [{ type: 'text', text: 'The filter on `created_at` has no index and the `ORDER BY` sorts the whole table. One index on `(customer_id, created_at)` turns the plan into a range scan; expect under 50 ms.' }]]] });
    // the interrupted answer: the engine restarted mid-sentence; Continue resumes it
    {
      const leaf = this.pathIds(sql).pop()!;
      const um = this.node(sql, { id: localId('m'), role: 'user', created_at: iso(-11 * 86_400_000, now), parent_id: leaf, run_id: null, parts: [{ type: 'text', text: 'Write the migration for that index.' }], status: 'done' }, leaf);
      this.node(sql, { id: localId('m'), role: 'assistant', created_at: iso(-11 * 86_400_000 + 20_000, now), parent_id: um, run_id: localId('run'), parts: [{ type: 'text', text: 'A concurrent build keeps the table open while the index fills:\n\n```sql\nCREATE INDEX CONCURRENTLY orders_customer_created\n  ON orders (customer_id, created_at);\n```\n\nTwo notes before you run it: the build cannot run inside a' }], status: 'interrupted' }, um);
      sql.status = 'error';
    }

    // the pending permission chat: the run paused on an edit_file that asks
    const ren = this.convs.get('c_rename')!;
    this.pausedRun(ren);

    // the run that is on: started three minutes ago from another browser
    const sync = mk(this.attachedId, 'Fix the flaky test in the sync worker', 3 * 60_000, { turns: [], sandbox: sandboxFor(true) });
    const uid = this.node(sync, { id: localId('m'), role: 'user', created_at: iso(-200_000, now), parent_id: null, run_id: null, parts: [{ type: 'text', text: 'tests/test_worker.py fails about once in fifty runs. Find out why and fix it; run it 200 times before and after.' }], status: 'done' }, null);
    sync.queue = [{ id: localId('q'), text: 'Also check that ruff passes afterwards.', created_at: iso(-60_000, now), client_msg_id: 'cm_seed1' }];
    sync.ui = { panes: [{ id: 'pane_term', kind: 'terminal', ref: 'shell', title: 'Terminal' }], active: 'pane_term', split: 0.6 };
    const runner = new Runner(this, sync, uid, CODING_SCRIPT, { startedBy: 'another browser', startedAgoMs: 180_000, skip: { steps: 3, outputLines: 3 } });
    runner.start();
  }

  /** c_rename: an edit_file that waits for permission, with the run on. */
  private pausedRun(st: ConvState): void {
    const leaf = this.pathIds(st).pop()!;
    const am = st.nodes.get(leaf)!.msg;
    const call: ToolCallPart = { type: 'tool_call', id: localId('call'), name: 'edit_file', input: JSON.stringify(EDIT_SCRIPT[2].kind === 'tool' ? EDIT_SCRIPT[2].input : {}, null, 2), output: null, status: 'asking', started_at: null, ended_at: null, permission: { summary: 'edit_file wants to change src/sync/worker.py: 1 block' }, pane: null };
    am.parts = [...am.parts, call];
    am.status = 'streaming';
    am.stats = undefined;
    st.run = { id: localId('run'), conversation_id: st.id, message_id: am.id, started_at: iso(-50 * 60_000), started_by: 'another browser', phase: 'permission', now: 'Waiting for permission: edit_file', step: 3, max_steps: 50, queue_position: null, tool: { call_id: call.id, name: 'edit_file' }, prefill: null, decode_tps: null, tokens_per_round: 4.1, acceptance: 0.71, thinking: { used: 318, budget: 8_192 }, tokens: 412, tps_history: [] };
    st.status = 'waiting';
    st.now = st.run.now;
    st.runner = new Runner(this, st, st.nodes.get(leaf)!.parent!, EDIT_SCRIPT, { resumeAt: { step: 2, message: am, call }, startedBy: 'another browser', startedAgoMs: 50 * 60_000 });
  }

  // ---- tree helpers ---------------------------------------------------------------------------

  node(st: ConvState, msg: Message, parent: string | null): string {
    st.nodes.set(msg.id, { msg, parent, children: [], active: 0 });
    if (parent) {
      const p = st.nodes.get(parent)!;
      p.children.push(msg.id);
      p.active = p.children.length - 1;
    } else {
      st.roots.push(msg.id);
      st.rootActive = st.roots.length - 1;
    }
    return msg.id;
  }

  pathIds(st: ConvState): string[] {
    const out: string[] = [];
    let id: string | undefined = st.roots[st.rootActive];
    while (id) {
      out.push(id);
      const n: Node = st.nodes.get(id)!;
      id = n.children[n.active];
    }
    return out;
  }

  private materialize(st: ConvState): Message[] {
    return this.pathIds(st).map((id) => {
      const n = st.nodes.get(id)!;
      const sib = n.parent ? st.nodes.get(n.parent)!.children : st.roots;
      const idx = sib.indexOf(id);
      return { ...n.msg, siblings: sib.length > 1 ? { index: idx, count: sib.length } : undefined };
    });
  }

  private summary(st: ConvState): ConversationSummary {
    const msgs = this.materialize(st);
    const lastUser = [...msgs].reverse().find((m) => m.role === 'user');
    const text = lastUser?.parts.find((p) => p.type === 'text') as { text: string } | undefined;
    return {
      id: st.id,
      title: st.title,
      created_at: st.created_at,
      updated_at: st.updated_at,
      pinned: st.pinned,
      archived: st.archived,
      version: st.version,
      unread: false,
      folder_id: st.folder_id,
      status: st.status,
      preview: (text?.text ?? '').split('\n')[0].slice(0, 120),
      model: st.settings.model,
      profile: st.settings.profile,
      turns: msgs.filter((m) => m.role === 'assistant').length,
      now: st.now,
    };
  }

  private full(st: ConvState): Conversation {
    const messages = this.materialize(st);
    return { ...this.summary(st), messages, head_id: messages.length ? messages[messages.length - 1].id : null, settings: st.settings, run: st.run, context: this.context(st, messages), sandbox: st.sandbox, compactions: st.compactions, usage: usageOf(messages), queue: st.queue, ui: st.ui, seq: String(st.seq) };
  }

  context(st: ConvState, messages = this.materialize(st)) {
    const tokens = Math.round(messages.reduce((a, m) => a + JSON.stringify(m.parts).length, 0) / 4) + (st.historyTokens ?? 0);
    const thinking = st.run ? st.run.thinking.used : 0;
    const files = messages.some((m) => m.parts.some((p) => p.type === 'tool_call' && p.name === 'read_file')) ? 1_460 : 0;
    const compacted = st.compactions.length ? Math.max(0, tokens - 4_000) : tokens;
    return contextFor(compacted, { tools: st.settings.tools.length > 0, files, thinking, warmAll: st.seen, profile: st.settings.profile, effort: st.settings.effort, thinkingOn: st.settings.thinking, slot: st.run ? 'slot 1' : null, cold: st.cold, packages: st.settings.packages });
  }

  // ---- events ---------------------------------------------------------------------------------

  emit(st: ConvState, ev: ChatEvent): void {
    st.seq++;
    const stamped: StampedEvent = { id: String(st.seq), event: ev };
    st.events.push(stamped);
    if (st.events.length > 5_000) st.events.splice(0, st.events.length - 5_000);
    // keep the server's own view current through the same reducer the client uses
    const before = this.full(st);
    const after = applyEvent(before, ev);
    for (const m of after.messages) {
      const n = st.nodes.get(m.id);
      if (n) {
        const { siblings: _s, ...rest } = m;
        n.msg = rest;
      }
    }
    st.status = after.status;
    st.now = after.now ?? null;
    st.run = after.run;
    st.queue = after.queue;
    st.updated_at = new Date().toISOString();
    for (const h of st.subs) h.onEvent(stamped, false);
  }

  private get(id: string): ConvState {
    const st = this.convs.get(id);
    if (!st) throw new Error(`no conversation ${id}`);
    return st;
  }

  // ---- ChatBackend: capabilities, conversations ----------------------------------------------

  async capabilities(): Promise<Capabilities> {
    const pw = this.scenario === 'signin';
    return { ...CAPABILITIES, session: { ...CAPABILITIES.session, password_set: pw, signed_in: !pw || this.signedIn } };
  }

  /** the setup guide's sign-in card did the POST; the mock only remembers the result */
  async signIn(_password: string): Promise<void> {
    this.signedIn = true;
  }
  async signOut(): Promise<void> {
    this.signedIn = false;
  }
  async uploadAttachment(file: File): Promise<Attachment> {
    return { id: localId('att'), name: file.name, kind: file.type.startsWith('image/') ? 'image' : file.type === 'application/pdf' ? 'pdf' : 'file', bytes: file.size, mime: file.type || 'application/octet-stream', url: `/v1/chat-app/attachments/${localId('att')}` };
  }

  async listConversations() {
    const conversations = [...this.convs.values()].map((s) => this.summary(s)).sort((a, b) => b.updated_at.localeCompare(a.updated_at));
    return { conversations, folders: this.folders.slice() };
  }

  async getConversation(id: string): Promise<Conversation> {
    return this.full(this.get(id));
  }

  async createConversation(settings?: Partial<ChatSettings>): Promise<Conversation> {
    const id = localId('c');
    const now = new Date().toISOString();
    const st: ConvState = { id, title: 'New chat', created_at: now, updated_at: now, pinned: false, archived: false, version: 1, ui: EMPTY_UI, folder_id: null, settings: { ...DEFAULT_SETTINGS, ...settings, policies: { ...(settings?.policies ?? {}) } }, nodes: new Map(), roots: [], rootActive: 0, run: null, runner: null, queue: [], compactions: [], sandbox: sandboxFor(false), events: [], seq: 0, subs: new Set(), seen: false, status: 'idle', now: null, clientIds: new Map() };
    this.convs.set(id, st);
    return this.full(st);
  }

  async updateConversation(id: string, patch: { version?: number; title?: string; pinned?: boolean; archived?: boolean; folder_id?: string | null; ui?: ConversationUi }) {
    const st = this.get(id);
    if (patch.version != null && patch.version !== st.version) throw new VersionConflict(this.summary(st));
    if (patch.title !== undefined) st.title = patch.title.trim() || st.title;
    if (patch.pinned !== undefined) st.pinned = patch.pinned;
    if (patch.archived !== undefined) st.archived = patch.archived;
    if (patch.folder_id !== undefined) st.folder_id = patch.folder_id;
    st.version++;
    if (patch.ui) {
      st.ui = { ...patch.ui, panes: patch.ui.panes.map((p) => ({ ...p, id: p.id ?? localId('pane') })) };
      this.emit(st, { type: 'ui.changed', ui: st.ui });
    }
    this.emit(st, { type: 'conversation.updated', patch: { title: st.title, pinned: st.pinned, archived: st.archived, folder_id: st.folder_id, version: st.version } });
    return this.summary(st);
  }

  async deleteConversation(id: string): Promise<void> {
    const st = this.get(id);
    st.runner?.cancel();
    this.convs.delete(id);
  }

  async createFolder(name: string): Promise<Folder> {
    const f = { id: localId('f'), name: name.trim() || 'Folder' };
    this.folders.push(f);
    return f;
  }

  subscribe(id: string, lastEventId: string | null, h: StreamHandler): Subscription {
    const st = this.get(id);
    let closed = false;
    const timers: ReturnType<typeof setTimeout>[] = [];
    const later = (ms: number, f: () => void) => timers.push(setTimeout(() => !closed && f(), ms));
    const open = (resume: string | null, first: boolean) => {
      h.onState(first ? 'connecting' : 'reconnecting');
      later(first ? 120 : 700, () => {
        h.onState('open');
        const after = resume ? Number(resume) : null;
        const replay = after != null ? st.events.filter((e) => Number(e.id) > after) : [];
        // the catch-up: fast, but not instant, so the state reads as a replay
        replay.forEach((e, i) => later(8 * i, () => h.onEvent(e, true)));
        later(8 * replay.length + 20, () => {
          h.onCurrent();
          st.subs.add(h);
        });
      });
    };
    open(lastEventId, true);
    if (this.scenario === 'reconnect' && id === this.attachedId) {
      // the stream drops after six seconds and comes back four seconds later with a catch-up
      let lastSeen = lastEventId;
      const wrapped: StreamHandler = { onEvent: (e, r) => ((lastSeen = e.id), h.onEvent(e, r)), onCurrent: () => h.onCurrent(), onState: (s, d) => h.onState(s, d) };
      later(6_000, () => {
        st.subs.delete(h);
        st.subs.delete(wrapped);
        h.onState('reconnecting');
        later(4_000, () => open(lastSeen, false));
      });
      st.subs.add(wrapped);
      // the wrapped handler records ids; the plain one is what open() adds
      return { close: () => ((closed = true), timers.forEach(clearTimeout), st.subs.delete(h), st.subs.delete(wrapped)) };
    }
    return { close: () => ((closed = true), timers.forEach(clearTimeout), st.subs.delete(h)) };
  }

  // ---- the turn -------------------------------------------------------------------------------

  async send(id: string, msg: SendOptions): Promise<void> {
    const st = this.get(id);
    if (st.clientIds.has(msg.client_msg_id)) return;
    if (st.run) return this.queue(id, msg.text, msg.client_msg_id);
    const path = this.pathIds(st);
    const parent = path.length ? path[path.length - 1] : null;
    const um: Message = { id: localId('m'), role: 'user', created_at: new Date().toISOString(), parent_id: parent, run_id: null, parts: [{ type: 'text', text: msg.text }], status: 'done', attachments: msg.attachments?.length ? msg.attachments : undefined };
    st.clientIds.set(msg.client_msg_id, um.id);
    this.node(st, um, parent);
    if (st.title === 'New chat') {
      st.title = msg.text.split('\n')[0].replace(/[#*`]/g, '').trim().slice(0, 60) || 'New chat';
      this.emit(st, { type: 'conversation.updated', patch: { title: st.title } });
    }
    this.emit(st, { type: 'message.created', message: um });
    const longChat = this.context(st).used > 20_000 || !st.seen;
    const runner = new Runner(this, st, um.id, script(msg.text, longChat), {});
    runner.start();
  }

  async queue(id: string, text: string, clientMsgId: string): Promise<void> {
    const st = this.get(id);
    if (st.queue.some((q) => q.client_msg_id === clientMsgId)) return;
    st.queue = [...st.queue, { id: localId('q'), text, created_at: new Date().toISOString(), client_msg_id: clientMsgId }];
    this.emit(st, { type: 'queue.changed', queue: st.queue });
  }

  /** Send now: the current generation stops (its text stays), this message goes in at once */
  async sendNow(id: string, queuedId: string): Promise<void> {
    const st = this.get(id);
    const q = st.queue.find((x) => x.id === queuedId);
    if (!q) return;
    st.queue = [q, ...st.queue.filter((x) => x.id !== queuedId)];
    this.emit(st, { type: 'queue.changed', queue: st.queue });
    if (st.runner) st.runner.stop();
    else this.drainQueue(st);
  }

  /** Continue: resume an interrupted answer as a fresh run on the same user message */
  async continueRun(id: string): Promise<void> {
    const st = this.get(id);
    const path = this.pathIds(st);
    const leaf = path.length ? st.nodes.get(path[path.length - 1])! : null;
    if (!leaf || leaf.msg.status !== 'interrupted' || !leaf.parent) return;
    const prev = textOf(leaf.msg);
    st.status = 'idle';
    new Runner(this, st, leaf.parent, [{ kind: 'text', text: prev + ' transaction, so run it on its own; and it takes two passes over the table, so plan for about twice the time of a plain build on 2 million rows (around 40 s here).\n\n```sql\n-- check it afterwards\nSELECT indexrelname, idx_scan FROM pg_stat_user_indexes WHERE indexrelname = \'orders_customer_created\';\n```' }, { kind: 'done' }], { replaceId: leaf.msg.id }).start();
  }

  async unqueue(id: string, queuedId: string): Promise<void> {
    const st = this.get(id);
    st.queue = st.queue.filter((q) => q.id !== queuedId);
    this.emit(st, { type: 'queue.changed', queue: st.queue });
  }

  /** after a run: the first queued message goes in */
  drainQueue(st: ConvState): void {
    const next = st.queue[0];
    if (!next) return;
    st.queue = st.queue.slice(1);
    this.emit(st, { type: 'queue.changed', queue: st.queue });
    setTimeout(() => void this.send(st.id, { text: next.text, client_msg_id: next.client_msg_id }), 400);
  }

  async stop(id: string): Promise<void> {
    this.get(id).runner?.stop();
  }

  async regenerate(id: string, messageId: string): Promise<void> {
    const st = this.get(id);
    const n = st.nodes.get(messageId);
    if (!n || !n.parent) return;
    const userText = textOf(st.nodes.get(n.parent)!.msg);
    new Runner(this, st, n.parent, script(userText, true), {}).start();
  }

  async edit(id: string, messageId: string, text: string, clientMsgId: string): Promise<void> {
    const st = this.get(id);
    const n = st.nodes.get(messageId);
    if (!n || st.clientIds.has(clientMsgId)) return;
    const um: Message = { id: localId('m'), role: 'user', created_at: new Date().toISOString(), parent_id: n.parent, run_id: null, parts: [{ type: 'text', text }], status: 'done' };
    st.clientIds.set(clientMsgId, um.id);
    this.node(st, um, n.parent);
    this.emit(st, { type: 'branch.changed', head_id: um.id });
    this.emit(st, { type: 'message.created', message: um });
    new Runner(this, st, um.id, script(text, true), {}).start();
  }

  async switchBranch(id: string, messageId: string, index: number): Promise<Conversation> {
    const st = this.get(id);
    const n = st.nodes.get(messageId);
    if (!n) return this.full(st);
    if (n.parent) {
      const p = st.nodes.get(n.parent)!;
      p.active = Math.max(0, Math.min(p.children.length - 1, index));
    } else st.rootActive = Math.max(0, Math.min(st.roots.length - 1, index));
    const full = this.full(st);
    this.emit(st, { type: 'branch.changed', head_id: full.head_id ?? '' });
    return full;
  }

  async permission(id: string, callId: string, decision: 'once' | 'always' | 'deny', _scope?: 'chat' | 'all'): Promise<void> {
    const st = this.get(id);
    if (decision === 'always' && st.runner?.pending) {
      st.settings = { ...st.settings, policies: { ...st.settings.policies, [st.runner.pending.call.name]: 'always' } };
      this.emit(st, { type: 'settings.changed', settings: st.settings });
    }
    st.runner?.decide(callId, decision === 'deny' ? 'deny' : 'allow');
  }

  async updateSettings(id: string, patch: Partial<ChatSettings>): Promise<ChatSettings> {
    const st = this.get(id);
    st.settings = { ...st.settings, ...patch, policies: { ...st.settings.policies, ...(patch.policies ?? {}) } };
    this.emit(st, { type: 'settings.changed', settings: st.settings });
    this.emit(st, { type: 'context.changed', context: this.context(st) });
    return st.settings;
  }

  async compact(id: string): Promise<void> {
    const st = this.get(id);
    const ctx = this.context(st);
    const path = this.materialize(st);
    const users = path.filter((m) => m.role === 'user');
    const cp: Compaction = { id: localId('cp'), at: new Date().toISOString(), before_tokens: ctx.used, after_tokens: Math.max(2_000, Math.round(ctx.used * 0.32)), duration_ms: 1_200, boundary_id: users.length >= 2 ? users[users.length - 2].id : path[0]?.id, summary: `Folded ${path.length} messages into a summary: the question, what was tried, the current state of the sandbox, and the open items.` };
    st.compactions = [...st.compactions, cp];
    this.emit(st, { type: 'compaction', compaction: cp });
    this.emit(st, { type: 'context.changed', context: this.context(st) });
  }

  // ---- settings stores ------------------------------------------------------------------------

  async tools() {
    return this.toolList.map((t) => ({ ...t }));
  }
  async setTool(toolId: string, patch: { enabled?: boolean; policy?: ToolPolicy }) {
    const t = this.toolList.find((x) => x.id === toolId);
    if (!t) throw new Error(`no tool ${toolId}`);
    Object.assign(t, patch);
    return { ...t };
  }
  async packages() {
    return this.pkgs.map((p) => ({ ...p }));
  }
  async setPackage(pkgId: string, enabled: boolean) {
    const p = this.pkgs.find((x) => x.id === pkgId);
    if (!p) throw new Error(`no package ${pkgId}`);
    p.enabled = enabled;
    return { ...p };
  }
  // the mock reads no file: an upload or an import gives a fixed answer named after the file
  async uploadSkill(file: File) {
    const name = file.name.replace(/\.zip$/i, '').toLowerCase().replace(/[^a-z0-9_.-]+/g, '-');
    const p: PackageInfo = { id: name, name, description: `A skill uploaded from ${file.name}.`, kind: 'skill', version: '', enabled: true, tools: [], user: true };
    this.pkgs = [...this.pkgs.filter((x) => x.id !== name), p];
    return { ...p };
  }
  async removeSkill(pkgId: string) {
    this.pkgs = this.pkgs.filter((x) => x.id !== pkgId);
  }
  async importOwui(file: File) {
    const slug = file.name.replace(/\.(py|json)$/i, '').toLowerCase().replace(/[^a-z0-9]+/g, '_');
    const kit: UserToolkit = { id: localId('kit'), slug, title: file.name, description: 'Imported from Open WebUI.', version: '', requirements: [], enabled: true, tools: [`owui__${slug}__run`], valves: [{ name: 'API_KEY', type: 'string', default: '', description: 'the service key', secret: true, value: '', set: false }], created_at: new Date().toISOString() };
    this.kits = [...this.kits, kit];
    const tool: ToolInfo = { id: kit.tools[0], name: kit.tools[0], description: `${file.name}, an Open WebUI tool`, source: { kind: 'user', id: kit.id }, enabled: true, policy: 'ask', risk: 'exec' };
    this.toolList = [...this.toolList, tool];
    return { toolkits: [kit], tools: [tool] };
  }
  async userToolkits() {
    return this.kits.map((k) => ({ ...k }));
  }
  async patchUserToolkit(id: string, patch: { enabled?: boolean; valves?: Record<string, unknown> }) {
    const k = this.kits.find((x) => x.id === id);
    if (!k) throw new Error(`no toolkit ${id}`);
    if (patch.enabled != null) k.enabled = patch.enabled;
    for (const v of k.valves) if (patch.valves && v.name in patch.valves && patch.valves[v.name] !== '') (v.set = true), (v.value = v.secret ? '' : patch.valves[v.name]);
    return { ...k };
  }
  async removeUserToolkit(id: string) {
    const k = this.kits.find((x) => x.id === id);
    this.kits = this.kits.filter((x) => x.id !== id);
    this.toolList = this.toolList.filter((t) => !k?.tools.includes(t.id));
  }
  async mcpServers() {
    return this.mcp.map((m) => ({ ...m }));
  }
  async addMcpServer(draft: McpServerDraft): Promise<McpServer> {
    const s: McpServer = { ...draft, id: localId('mcp'), status: 'starting', error: null, tools: [], updated_at: new Date().toISOString() };
    this.mcp.push(s);
    setTimeout(() => {
      const t = await0(this.testMcpServer(draft));
      void t.then((r) => {
        s.status = r.ok ? 'connected' : 'failed';
        s.error = r.ok ? null : r.message;
        s.tools = r.tools;
        s.updated_at = new Date().toISOString();
      });
    }, 900);
    return { ...s };
  }
  async removeMcpServer(serverId: string): Promise<void> {
    this.mcp = this.mcp.filter((m) => m.id !== serverId);
  }
  async testMcpServer(draft: McpServerDraft) {
    await tick(700);
    if (draft.transport === 'stdio' && !(draft.command ?? '').trim()) return { ok: false, message: 'no command given', tools: [] };
    if (draft.transport === 'http' && !/^https?:\/\//.test(draft.url ?? '')) return { ok: false, message: 'the URL must start with http:// or https://', tools: [] };
    if (/uvx|npx/.test(draft.command ?? '') && !(draft.args ?? []).length) return { ok: false, message: `${draft.command} needs the package name as its first argument`, tools: [] };
    return { ok: true, message: 'handshake ok: protocol 2025-06-18, 2 tools', tools: [{ name: `${draft.name || 'server'}_list`, description: 'list items' }, { name: `${draft.name || 'server'}_get`, description: 'get one item' }] };
  }
  async restartMcpServer(serverId: string): Promise<McpServer> {
    const s = this.mcp.find((m) => m.id === serverId);
    if (!s) throw new Error(`no server ${serverId}`);
    s.status = 'starting';
    setTimeout(() => {
      s.status = s.id === 'calendar' ? 'failed' : 'connected';
      s.updated_at = new Date().toISOString();
    }, 1_200);
    return { ...s };
  }
  async prompts() {
    return this.promptList.map((p) => ({ ...p }));
  }
  async savePrompt(p: Omit<Prompt, 'id'> & { id?: string }): Promise<Prompt> {
    const existing = p.id ? this.promptList.find((x) => x.id === p.id) : undefined;
    if (existing) {
      Object.assign(existing, p);
      return { ...existing };
    }
    const np: Prompt = { ...p, id: localId('p') };
    this.promptList.push(np);
    return { ...np };
  }
  async deletePrompt(promptId: string): Promise<void> {
    this.promptList = this.promptList.filter((p) => p.id !== promptId);
  }
  async presets() {
    return this.presetList.map((p) => ({ ...p }));
  }
  async savePreset(p: Omit<Preset, 'id'> & { id?: string }): Promise<Preset> {
    const existing = p.id ? this.presetList.find((x) => x.id === p.id) : undefined;
    if (existing) {
      Object.assign(existing, p);
      return { ...existing };
    }
    const np: Preset = { ...p, id: localId('ps') };
    this.presetList.push(np);
    return { ...np };
  }
  async deletePreset(presetId: string): Promise<void> {
    this.presetList = this.presetList.filter((p) => p.id !== presetId);
  }

  // ---- the sandbox ----------------------------------------------------------------------------

  async sandbox(id: string): Promise<SandboxStatus> {
    return this.get(id).sandbox;
  }
  async resetSandbox(id: string): Promise<SandboxStatus> {
    const st = this.get(id);
    st.sandbox = { ...sandboxFor(false), state: 'creating' };
    this.emit(st, { type: 'sandbox.changed', sandbox: st.sandbox });
    setTimeout(() => {
      st.sandbox = { ...sandboxFor(true), uptime_s: 2, disk_bytes: 12e6 };
      this.emit(st, { type: 'sandbox.changed', sandbox: st.sandbox });
    }, 1_600);
    return st.sandbox;
  }
  async sandboxFiles(_id: string, path: string): Promise<SandboxFile[]> {
    const dir = path.replace(/\/$/, '') || '/work/sync';
    const seen = new Map<string, SandboxFile>();
    for (const [p, content] of Object.entries(FILES)) {
      if (!p.startsWith(dir + '/')) continue;
      const rest = p.slice(dir.length + 1);
      const head = rest.split('/')[0];
      const full = `${dir}/${head}`;
      if (rest.includes('/')) seen.set(full, { path: full, kind: 'dir', bytes: 0 });
      else seen.set(full, { path: full, kind: 'file', bytes: content.length });
    }
    return [...seen.values()].sort((a, b) => (a.kind === b.kind ? a.path.localeCompare(b.path) : a.kind === 'dir' ? -1 : 1));
  }
  async readFile(_id: string, path: string) {
    const content = FILES[path];
    if (content == null) throw new Error(`no file ${path}`);
    return { content, mime: path.endsWith('.py') ? 'text/x-python' : path.endsWith('.md') ? 'text/markdown' : 'text/plain', bytes: content.length, etag: String(content.length) };
  }
  async writeFile(_id: string, path: string, content: string, etag?: string): Promise<void> {
    const cur = FILES[path];
    if (cur != null && etag != null && etag !== String(cur.length)) throw new Error('the file changed on the box; reload it first');
    FILES[path] = content;
  }
  async artifact(_id: string, ref: string) {
    // the server's default refs, as the mock's seeded artifacts
    const alias: Record<string, string> = { 'diff:': 'diff:worker', 'file:/workspace/index.html': 'preview:shop-page' };
    const a = ARTIFACTS[alias[ref] ?? ref];
    if (!a) throw new Error(`no artifact ${ref}`);
    return a;
  }

  terminal(id: string, session: string, onData: (data: string) => void): TerminalSession {
    const st = this.get(id);
    let closed = false;
    const timers: ReturnType<typeof setTimeout>[] = [];
    const later = (ms: number, f: () => void) => timers.push(setTimeout(() => !closed && f(), ms));
    const prompt = () => onData('\r\n\x1b[32msandbox\x1b[0m:\x1b[34m/work/sync\x1b[0m$ ');
    if (session !== 'shell') {
      // a tool call's own output
      const call = this.findCall(st, session);
      if (!call) onData(`no output for ${session}\r\n`);
      else {
        let i = 0;
        const input = parseInput(call.input);
        onData(`\x1b[2m$ ${String(input.command ?? call.name)}\x1b[0m\r\n`);
        const lines = (call.output?.content ?? '').split('\n');
        const push = () => {
          if (i >= lines.length) return;
          onData(lines[i++] + '\r\n');
          later(60, push);
        };
        push();
      }
      return { write: () => undefined, resize: () => undefined, close: () => ((closed = true), timers.forEach(clearTimeout)) };
    }
    onData(`\x1b[2mtandem-sandbox 1.3, ${st.sandbox.cwd}, network ${st.sandbox.network}\x1b[0m`);
    prompt();
    let line = '';
    const run = (cmd: string) => {
      const c = cmd.trim();
      const out: string[] = [];
      if (c === 'ls' || c === 'ls -la') out.push('README.md  pyproject.toml  site/  src/  tests/');
      else if (c === 'pwd') out.push('/work/sync');
      else if (/^cat\s+/.test(c)) {
        const p = c.replace(/^cat\s+/, '');
        const content = FILES[p.startsWith('/') ? p : `/work/sync/${p}`];
        out.push(...(content != null ? content.split('\n') : [`cat: ${p}: No such file or directory`]));
      } else if (/pytest/.test(c)) out.push('collected 2 items', '', '..                                                                       [100%]', '\x1b[32m2 passed\x1b[0m in 0.21s');
      else if (c === 'clear') out.push('\x1b[2J\x1b[H');
      else if (c) out.push(`bash: ${c.split(' ')[0]}: command not found`);
      let i = 0;
      const push = () => {
        if (i < out.length) {
          onData(out[i++] + '\r\n');
          later(40, push);
        } else prompt();
      };
      if (out.length) onData('\r\n');
      push();
    };
    return {
      write: (data) => {
        for (const ch of data) {
          if (ch === '\r' || ch === '\n') {
            run(line);
            line = '';
          } else if (ch === '\x7f' || ch === '\b') {
            if (line) {
              line = line.slice(0, -1);
              onData('\b \b');
            }
          } else if (ch >= ' ') {
            line += ch;
            onData(ch);
          }
        }
      },
      resize: () => undefined,
      close: () => ((closed = true), timers.forEach(clearTimeout)),
    };
  }

  private findCall(st: ConvState, callId: string): ToolCallPart | null {
    for (const n of st.nodes.values()) for (const p of n.msg.parts) if (p.type === 'tool_call' && p.id === callId) return p;
    return null;
  }

  policyFor(st: ConvState, tool: string): ToolPolicy {
    return st.settings.policies[tool] ?? this.toolList.find((t) => t.id === tool)?.policy ?? 'ask';
  }
}

// ---- the runner: one scripted turn ------------------------------------------------------------

interface RunnerOptions {
  startedBy?: string;
  startedAgoMs?: number;
  /** the attached run: begin this many steps in, with this many output lines already out */
  skip?: { steps: number; outputLines: number };
  /** c_rename: a run paused at a permission prompt */
  resumeAt?: { step: number; message: Message; call: ToolCallPart };
  /** Continue: the new run writes into this interrupted message instead of a new one */
  replaceId?: string;
}

class Runner {
  private timers: ReturnType<typeof setTimeout>[] = [];
  private cancelled = false;
  private stopped = false;
  private msgId = '';
  private run!: Run;
  private tokens = 0;
  private thinkingUsed = 0;
  private metricsTimer: ReturnType<typeof setInterval> | null = null;
  private phaseTps: number | null = null;
  private toolStart = 0;
  pending: { call: ToolCallPart; resolve: (d: 'allow' | 'deny') => void } | null = null;

  constructor(
    private mock: MockChatBackend,
    private st: ConvState,
    private parentId: string,
    private steps: Step[],
    private opts: RunnerOptions,
  ) {}

  /** the assistant message as the tree holds it now (emit writes every event back) */
  private get msg(): Message {
    return this.st.nodes.get(this.msgId)!.msg;
  }

  private later(ms: number, f: () => void): Promise<void> {
    return new Promise((resolve) => {
      this.timers.push(
        setTimeout(() => {
          if (!this.cancelled) f();
          resolve();
        }, ms),
      );
    });
  }

  start(): void {
    this.st.runner = this;
    const now = Date.now();
    if (this.opts.resumeAt) {
      this.msgId = this.opts.resumeAt.message.id;
      this.run = this.st.run!;
      // wait at the prompt; the decision continues the script from the next step
      this.pending = { call: this.opts.resumeAt.call, resolve: (d) => void this.continueFrom(this.opts.resumeAt!.step, this.opts.resumeAt!.call, d) };
      return;
    }
    const runId = localId('run');
    let msg: Message;
    if (this.opts.replaceId) {
      const n = this.st.nodes.get(this.opts.replaceId)!;
      n.msg = { ...n.msg, status: 'streaming', run_id: runId };
      msg = n.msg;
      this.msgId = msg.id;
    } else {
      msg = { id: localId('m'), role: 'assistant', created_at: new Date(now - (this.opts.startedAgoMs ?? 0)).toISOString(), parent_id: this.parentId, run_id: runId, parts: [], status: 'streaming' };
      this.msgId = msg.id;
      this.mock.node(this.st, msg, this.parentId);
    }
    this.run = { id: runId, conversation_id: this.st.id, message_id: msg.id, started_at: msg.created_at, started_by: this.opts.startedBy ?? null, phase: 'queued', now: 'Queued', step: 0, max_steps: this.st.settings.max_steps, queue_position: null, tool: null, prefill: null, decode_tps: null, tokens_per_round: null, acceptance: null, thinking: { used: 0, budget: this.st.settings.thinking ? 8_192 : 0 }, tokens: 0, tps_history: [] };
    this.st.sandbox = { ...this.st.sandbox, state: 'running', uptime_s: Math.max(this.st.sandbox.uptime_s, 1), memory_bytes: Math.max(this.st.sandbox.memory_bytes, 412e6) };
    this.mock.emit(this.st, { type: 'run.started', run: this.run });
    if (!this.opts.replaceId) this.mock.emit(this.st, { type: 'message.created', message: msg });
    this.mock.emit(this.st, { type: 'sandbox.changed', sandbox: this.st.sandbox });
    this.metricsTimer = setInterval(() => this.metrics(), 1_000);
    void this.play(0);
  }

  private metrics(): void {
    if (this.cancelled || !this.st.run) return;
    const tps = this.phaseTps == null ? null : Math.round((this.phaseTps + (Math.random() - 0.5) * 9) * 10) / 10;
    this.mock.emit(this.st, { type: 'run.metrics', run_id: this.run.id, decode_tps: tps, tokens_per_round: tps == null ? this.st.run.tokens_per_round : Math.round((3.6 + Math.random() * 1.4) * 100) / 100, acceptance: tps == null ? this.st.run.acceptance : Math.round((0.62 + Math.random() * 0.16) * 1000) / 1000, tokens: this.tokens, thinking_used: this.thinkingUsed });
  }

  private step = 0;
  private phase(phase: Run['phase'], now: string, extra: { tool?: Run['tool']; prefill?: Run['prefill'] } = {}): void {
    if (phase === 'thinking' || phase === 'writing') this.step++;
    this.mock.emit(this.st, { type: 'run.phase', run_id: this.run.id, phase, now, step: this.step, ...extra });
  }

  private addPart(p: Part): number {
    this.mock.emit(this.st, { type: 'part.added', message_id: this.msgId, part: p });
    return this.msg.parts.length - 1;
  }

  private async stream(index: number, text: string, field: 'text' | 'input', thinking: boolean): Promise<void> {
    const words = text.split(/(?<=\s)/);
    let i = 0;
    while (i < words.length && !this.cancelled && !this.stopped) {
      const n = 1 + Math.floor(Math.random() * 2);
      const delta = words.slice(i, i + n).join('');
      i += n;
      this.tokens += Math.max(1, Math.round(delta.length / 4));
      if (thinking) this.thinkingUsed += Math.max(1, Math.round(delta.length / 4));
      this.mock.emit(this.st, { type: 'part.delta', message_id: this.msgId, index, delta, field });
      await this.later(WORD_MS * n, () => undefined);
    }
  }

  private async play(from: number): Promise<void> {
    const skip = this.opts.skip;
    for (let i = from; i < this.steps.length; i++) {
      if (this.cancelled || this.stopped) break;
      const s = this.steps[i];
      const skipping = skip ? i < skip.steps : false;
      if (s.kind === 'prefill') {
        if (skipping) continue;
        this.phaseTps = null;
        let done = 0;
        this.phase('prefill', `Prefilling 0 of ${s.total.toLocaleString('en-US')}`, { prefill: { done: 0, total: s.total } });
        while (done < s.total && !this.cancelled) {
          done = Math.min(s.total, done + Math.round(s.total / 9));
          await this.later(170, () => this.phase('prefill', `Prefilling ${done.toLocaleString('en-US')} of ${s.total.toLocaleString('en-US')} (${Math.round((done / s.total) * 100)} %)`, { prefill: { done, total: s.total } }));
        }
      } else if (s.kind === 'thinking') {
        if (skipping) {
          this.thinkingUsed += Math.round(s.text.length / 4);
          this.addPart({ type: 'thinking', text: s.text, tokens: Math.round(s.text.length / 4), done: true });
          continue;
        }
        this.phaseTps = 58;
        this.phase('thinking', 'Thinking');
        const idx = this.addPart({ type: 'thinking', text: '', tokens: null, done: false });
        await this.stream(idx, s.text, 'text', true);
        this.mock.emit(this.st, { type: 'part.done', message_id: this.msgId, index: idx, tokens: Math.round(s.text.length / 4) });
      } else if (s.kind === 'text') {
        if (skipping) {
          this.tokens += Math.round(s.text.length / 4);
          this.addPart({ type: 'text', text: s.text });
          continue;
        }
        this.phaseTps = 54;
        this.phase('writing', 'Writing');
        const idx = this.addPart({ type: 'text', text: '' });
        await this.stream(idx, s.text, 'text', false);
      } else if (s.kind === 'tool') {
        const callId = localId('call');
        const input = JSON.stringify(s.input, null, 2);
        const pane: PaneRef | null = s.pane ? { ...s.pane, ref: s.pane.ref === 'call' ? callId : s.pane.ref } : null;
        if (skipping) {
          this.addPart({ type: 'tool_call', id: callId, name: s.name, input, output: { content: s.output.join('\n'), is_error: !!s.error, bytes: s.output.join('\n').length }, status: s.error ? 'failed' : 'done', started_at: iso(-90_000), ended_at: iso(-80_000), permission: null, pane });
          continue;
        }
        const call: ToolCallPart = { type: 'tool_call', id: callId, name: s.name, input: '', output: null, status: 'running', started_at: null, ended_at: null, permission: null, pane };
        const idx = this.addPart(call);
        this.phaseTps = 52;
        this.phase('writing', `Calling ${s.name}`);
        const partial = skip && i === skip.steps;
        if (partial) {
          // the attached run: the call is already written and running, some output is out
          this.mock.emit(this.st, { type: 'part.delta', message_id: this.msgId, index: idx, delta: input, field: 'input' });
        } else await this.stream(idx, input, 'input', false);
        if (this.cancelled || this.stopped) break;
        const policy = this.mock.policyFor(this.st, s.name);
        let allowed: 'allow' | 'deny' = 'allow';
        if (policy === 'never') allowed = 'deny';
        else if (policy === 'ask' && !partial) {
          this.phaseTps = null;
          this.phase('permission', `Waiting for permission: ${s.name}`, { tool: { call_id: callId, name: s.name } });
          this.mock.emit(this.st, { type: 'tool.asking', message_id: this.msgId, call_id: callId, summary: permissionSummary(s.name, s.input) });
          allowed = await new Promise<'allow' | 'deny'>((resolve) => (this.pending = { call, resolve }));
          this.pending = null;
          if (this.cancelled || this.stopped) break;
        }
        if (allowed === 'deny') {
          this.mock.emit(this.st, { type: 'tool.done', message_id: this.msgId, call_id: callId, status: 'denied', is_error: false });
          if (s.denied) {
            this.phaseTps = 54;
            this.phase('writing', 'Writing');
            const t = this.addPart({ type: 'text', text: '' });
            await this.stream(t, s.denied, 'text', false);
          }
          continue;
        }
        this.phaseTps = null;
        this.toolStart = Date.now();
        this.phase('tool', `Running ${s.name}${s.name === 'bash' ? `: ${String(s.input.command ?? '')}` : ''}`, { tool: { call_id: callId, name: s.name } });
        this.mock.emit(this.st, { type: 'tool.started', message_id: this.msgId, call_id: callId, pane });
        const lines = s.output.join('\n').split('\n');
        const start = partial ? Math.min(lines.length, skip!.outputLines) : 0;
        if (start) this.mock.emit(this.st, { type: 'tool.output', message_id: this.msgId, call_id: callId, delta: lines.slice(0, start).join('\n') + '\n' });
        const per = Math.max(40, s.ms / Math.max(1, lines.length));
        for (let k = start; k < lines.length && !this.cancelled && !this.stopped; k++) {
          await this.later(per, () => this.mock.emit(this.st, { type: 'tool.output', message_id: this.msgId, call_id: callId, delta: lines[k] + (k < lines.length - 1 ? '\n' : '') }));
        }
        if (this.cancelled || this.stopped) break;
        this.mock.emit(this.st, { type: 'tool.done', message_id: this.msgId, call_id: callId, status: s.error ? 'failed' : 'done', is_error: !!s.error });
        this.mock.emit(this.st, { type: 'context.changed', context: this.mock.context(this.st) });
      } else if (s.kind === 'done') {
        this.finish(s.finish ?? 'stop');
        return;
      }
    }
    if (this.stopped && !this.cancelled) this.finish('stopped');
  }

  private continueFrom(step: number, call: ToolCallPart, d: 'allow' | 'deny'): void {
    this.pending = null;
    const s = this.steps[step];
    if (s.kind !== 'tool') return;
    this.metricsTimer = setInterval(() => this.metrics(), 1_000);
    void (async () => {
      if (d === 'deny') {
        this.mock.emit(this.st, { type: 'tool.done', message_id: this.msgId, call_id: call.id, status: 'denied', is_error: false });
        if (s.denied) {
          this.phaseTps = 54;
          this.phase('writing', 'Writing');
          const t = this.addPart({ type: 'text', text: '' });
          await this.stream(t, s.denied, 'text', false);
        }
      } else {
        this.phaseTps = null;
        this.phase('tool', `Running ${s.name}`, { tool: { call_id: call.id, name: s.name } });
        this.mock.emit(this.st, { type: 'tool.started', message_id: this.msgId, call_id: call.id, pane: s.pane ?? null });
        for (const line of s.output) await this.later(300, () => this.mock.emit(this.st, { type: 'tool.output', message_id: this.msgId, call_id: call.id, delta: line }));
        this.mock.emit(this.st, { type: 'tool.done', message_id: this.msgId, call_id: call.id, status: 'done', is_error: false });
      }
      await this.play(step + 1);
    })();
  }

  private finish(finish: TurnStats['finish']): void {
    if (this.metricsTimer) clearInterval(this.metricsTimer);
    this.metricsTimer = null;
    const stats = statsFor(this.msg.parts, finish);
    this.st.seen = true;
    this.mock.emit(this.st, { type: 'run.done', run_id: this.run.id, message_id: this.msgId, stats, status: finish === 'stopped' ? 'stopped' : 'done' });
    this.mock.emit(this.st, { type: 'context.changed', context: this.mock.context(this.st) });
    this.mock.emit(this.st, { type: 'usage.changed', usage: usageOf(this.mock['materialize'](this.st)) });
    this.st.runner = null;
    this.mock.drainQueue(this.st);
  }

  decide(callId: string, d: 'allow' | 'deny'): void {
    if (this.pending && this.pending.call.id === callId) this.pending.resolve(d);
  }

  stop(): void {
    this.stopped = true;
    if (this.pending) this.pending.resolve('deny');
    // the loop ends at its next await; a tool in flight ends here
    for (const p of this.msg.parts) if (p.type === 'tool_call' && (p.status === 'running' || p.status === 'asking')) this.mock.emit(this.st, { type: 'tool.done', message_id: this.msgId, call_id: p.id, status: 'cancelled', is_error: false });
    if (this.timers.length === 0 || this.pending) this.finish('stopped');
  }

  cancel(): void {
    this.cancelled = true;
    this.timers.forEach(clearTimeout);
    if (this.metricsTimer) clearInterval(this.metricsTimer);
  }
}

// ---- helpers ----------------------------------------------------------------------------------

function stepToPart(s: Step): Part {
  if (s.kind === 'thinking') return { type: 'thinking', text: s.text, tokens: Math.round(s.text.length / 4), done: true };
  if (s.kind === 'text') return { type: 'text', text: s.text };
  if (s.kind === 'tool') {
    const id = localId('call');
    return { type: 'tool_call', id, name: s.name, input: JSON.stringify(s.input, null, 2), output: { content: s.output.join('\n'), is_error: !!s.error, bytes: s.output.join('\n').length }, status: s.error ? 'failed' : 'done', started_at: iso(-3_600_000), ended_at: iso(-3_600_000 + s.ms), permission: null, pane: s.pane ? { ...s.pane, ref: s.pane.ref === 'call' ? id : s.pane.ref } : null };
  }
  return { type: 'text', text: '' };
}

function statsFor(parts: Part[], finish: TurnStats['finish'] = 'stop'): TurnStats {
  const text = parts.filter((p) => p.type === 'text').reduce((a, p) => a + (p as { text: string }).text.length, 0);
  const think = parts.filter((p) => p.type === 'thinking').reduce((a, p) => a + (p as { text: string }).text.length, 0);
  const calls = parts.filter((p) => p.type === 'tool_call').length;
  const completion = Math.round((text + think) / 4) + calls * 40;
  const tps = 51 + Math.round(Math.random() * 90) / 10;
  return { finish, ttft_ms: 180 + Math.round(Math.random() * 400), decode_tps: tps, prompt_tokens: 6_200 + Math.round(text / 2), completion_tokens: completion, reasoning_tokens: Math.round(think / 4), cached_tokens: 5_900, tokens_per_round: Math.round((3.4 + Math.random() * 1.6) * 100) / 100, acceptance: Math.round((0.6 + Math.random() * 0.2) * 1000) / 1000, total_ms: Math.round((completion / tps) * 1000) };
}

function usageOf(messages: Message[]): ConversationUsage {
  const a = messages.filter((m) => m.role === 'assistant' && m.stats);
  const sum = (f: (s: TurnStats) => number | null) => a.reduce((x, m) => x + (f(m.stats!) ?? 0), 0);
  return { turns: a.length, prompt_tokens: sum((s) => s.prompt_tokens), completion_tokens: sum((s) => s.completion_tokens), reasoning_tokens: sum((s) => s.reasoning_tokens), cached_tokens: sum((s) => s.cached_tokens), tool_calls: messages.reduce((x, m) => x + m.parts.filter((p) => p.type === 'tool_call').length, 0), wall_s: Math.round(sum((s) => s.total_ms) / 1000) };
}

function textOf(m: Message): string {
  return m.parts
    .filter((p) => p.type === 'text')
    .map((p) => (p as { text: string }).text)
    .join('\n');
}

function parseInput(s: string): Record<string, unknown> {
  try {
    return JSON.parse(s) as Record<string, unknown>;
  } catch {
    return {};
  }
}

export function permissionSummary(name: string, input: Record<string, unknown>): string {
  if (name === 'bash') return `bash wants to run: ${String(input.command ?? '')}${input.cwd ? ` in ${String(input.cwd)}` : ''}`;
  if (name === 'edit_file') return `edit_file wants to change ${String(input.path ?? 'a file')}: 1 block`;
  if (name === 'write_file') return `write_file wants to write ${String(input.path ?? 'a file')}${input.bytes ? ` (${Number(input.bytes).toLocaleString('en-US')} bytes)` : ''}`;
  if (name === 'notes_write') return `notes_write wants to create or update a note`;
  return `${name} wants to run with ${Object.keys(input).length} argument${Object.keys(input).length === 1 ? '' : 's'}`;
}

function tick(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

/** a promise, unchanged: a name for the `setTimeout` body above */
function await0<T>(p: Promise<T>): Promise<T> {
  return p;
}

export type { StreamState };
export type { Attachment };
