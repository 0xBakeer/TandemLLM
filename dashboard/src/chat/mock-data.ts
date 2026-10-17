// The mock's content: generic conversations, the sandbox's files, the artifacts, the settings
// stores. Nothing personal: a made-up project (a sync worker), a made-up shop page, a meal plan.

import type { Capabilities, ChatSettings, Compaction, ContextReport, McpServer, Message, PackageInfo, Part, Preset, Prompt, RunPhase, SandboxStatus, Task, TaskMetrics, TaskReport, ToolCallPart, ToolInfo, WebSearchResult, WebSearchSettings, WorkerContext } from './types';

export const MODEL = 'TandemLLM';

export const DEFAULT_SETTINGS: ChatSettings = {
  model: MODEL,
  profile: null,
  effort: 'auto',
  thinking: true,
  tools: ['bash', 'read_file', 'write_file', 'edit_file', 'web_search', 'web_fetch'],
  packages: ['core-how-it-works', 'coding-answer'],
  packages_blocked: [],
  packages_mode: 'auto',
  mcp: ['notes', 'web-search'],
  prompt_id: null,
  policies: {},
  auto_accept_edits: false,
  verify: true,
  max_steps: 50,
  sandbox_network: 'off',
  permission_mode: 'auto',
  agents: false,
  task_max_steps: 40,
  task_max_per_run: 8,
};

export const CAPABILITIES: Capabilities = {
  models: [
    {
      id: MODEL,
      name: 'TandemLLM',
      weights: 'EXL3 4.5 bpw',
      max_len: 262_144,
      profiles: [],
    },
    { id: 'deepseek-flash', name: 'DeepSeek V4 Flash (cloud)', weights: '', max_len: 1_048_576, profiles: [], backend: 'cloud' },
    { id: 'deepseek-v4-pro', name: 'DeepSeek V4 Pro (cloud)', weights: '', max_len: 1_048_576, profiles: [], backend: 'cloud' },
  ],
  efforts: ['auto', 'low', 'medium', 'high', 'xhigh'],
  commands: [
    { command: 'clear', description: 'start a new chat' },
    { command: 'compact', description: 'fold the older turns into a summary' },
    { command: 'model', description: 'pick the model' },
    { command: 'effort', description: 'set the thinking effort' },
    { command: 'pane', description: 'open a pane: terminal, preview, files, diff' },
    { command: 'mcp', description: 'the MCP servers and their tools' },
    { command: 'prompt', description: 'insert a saved prompt' },
    { command: 'review', description: 'saved prompt: review the code I paste', prompt_id: 'p_review' },
    { command: 'plain', description: 'saved prompt: say it in plain English', prompt_id: 'p_plain' },
    { command: 'commit', description: 'saved prompt: write the commit message', prompt_id: 'p_commit' },
  ],
  defaults: DEFAULT_SETTINGS,
  session: { password_set: false, signed_in: true, sandbox: 'ready' },
  web_search: { provider: 'duckduckgo', providers: ['off', 'duckduckgo', 'searxng'] },
};

export const TOOLS: ToolInfo[] = [
  { id: 'bash', name: 'bash', description: 'run a command in the chat\'s sandbox', source: { kind: 'builtin' }, enabled: true, policy: 'ask', risk: 'exec' },
  { id: 'read_file', name: 'read_file', description: 'read a file from the sandbox', source: { kind: 'builtin' }, enabled: true, policy: 'always', risk: 'read' },
  { id: 'write_file', name: 'write_file', description: 'write a whole file in the sandbox', source: { kind: 'builtin' }, enabled: true, policy: 'ask', risk: 'write' },
  { id: 'edit_file', name: 'edit_file', description: 'replace one block in a file', source: { kind: 'builtin' }, enabled: true, policy: 'ask', risk: 'write' },
  { id: 'web_search', name: 'web_search', description: 'search the web: titles, URLs and snippets to cite', source: { kind: 'builtin' }, enabled: true, policy: 'always', risk: 'network' },
  { id: 'web_fetch', name: 'web_fetch', description: 'fetch a page and return its text', source: { kind: 'builtin' }, enabled: true, policy: 'always', risk: 'network' },
  { id: 'browser', name: 'browser', description: 'drive a headless browser in the sandbox', source: { kind: 'builtin' }, enabled: false, policy: 'ask', risk: 'exec' },
  { id: 'notes_search', name: 'notes_search', description: 'search the notes workspace', source: { kind: 'mcp', id: 'notes' }, enabled: true, policy: 'always' },
  { id: 'notes_read', name: 'notes_read', description: 'read one note by id', source: { kind: 'mcp', id: 'notes' }, enabled: true, policy: 'always' },
  { id: 'notes_write', name: 'notes_write', description: 'create or update a note', source: { kind: 'mcp', id: 'notes' }, enabled: true, policy: 'ask', risk: 'write' },
  { id: 'search', name: 'search', description: 'web search, ten results', source: { kind: 'mcp', id: 'web-search' }, enabled: true, policy: 'always' },
  { id: 'fetch_page', name: 'fetch_page', description: 'fetch a result as readable text', source: { kind: 'mcp', id: 'web-search' }, enabled: true, policy: 'always' },
  { id: 'image_gen', name: 'image_gen', description: 'draw an image from a prompt (the image model on this box)', source: { kind: 'package', id: 'image-answer' }, enabled: false, policy: 'ask' },
];

export const PACKAGES: PackageInfo[] = [
  { id: 'core-how-it-works', name: 'core: how it works', description: 'the engine explains its own mechanism from the docs', kind: 'skill', version: '1.4', enabled: true, tools: [] },
  { id: 'coding-answer', name: 'coding answer', description: 'code first, one worked example, the two numbers that decide it', kind: 'package', version: '2.1', enabled: true, tools: ['bash', 'read_file', 'edit_file'] },
  { id: 'web-design-answer', name: 'web design answer', description: 'a page as one html file, previewed in a pane', kind: 'package', version: '1.2', enabled: true, tools: ['write_file'] },
  { id: 'research-brief', name: 'research brief', description: 'search, read, cite; a brief with the sources', kind: 'package', version: '1.0', enabled: false, tools: ['search', 'fetch_page'] },
  { id: 'math-steps', name: 'math steps', description: 'each step on its own line, the check at the end', kind: 'skill', version: '1.1', enabled: true, tools: [] },
  { id: 'everyday-planner', name: 'everyday planner', description: 'plans, lists and schedules in plain tables', kind: 'skill', version: '1.0', enabled: true, tools: [] },
  { id: 'image-answer', name: 'image answer', description: 'the image model on this box draws what you describe', kind: 'package', version: '0.9', enabled: false, tools: ['image_gen'] },
  { id: 'commit-writer', name: 'commit writer', description: 'a commit message from a diff, one line and a body', kind: 'skill', version: '1.3', enabled: true, tools: [] },
];

export const MCP: McpServer[] = [
  { id: 'notes', name: 'notes', transport: 'stdio', command: 'notes-mcp', args: ['--workspace', 'default'], env: {}, status: 'connected', error: null, tools: [{ name: 'notes_search', description: 'search the notes workspace' }, { name: 'notes_read', description: 'read one note by id' }, { name: 'notes_write', description: 'create or update a note' }], updated_at: iso(-3_600_000) },
  { id: 'web-search', name: 'web-search', transport: 'http', url: 'http://127.0.0.1:8787/mcp', headers: {}, status: 'connected', error: null, tools: [{ name: 'search', description: 'web search, ten results' }, { name: 'fetch_page', description: 'fetch a result as readable text' }], updated_at: iso(-7_200_000) },
  { id: 'calendar', name: 'calendar', transport: 'stdio', command: 'uvx', args: ['calendar-mcp'], env: {}, status: 'failed', error: 'spawn uvx: not found on PATH', tools: [], updated_at: iso(-600_000) },
];

export const PROMPTS: Prompt[] = [
  { id: 'p_review', title: 'Code reviewer', command: 'review', body: 'Review the code I paste. Name the bug first, then the fix, then what to test. No praise.' },
  { id: 'p_plain', title: 'Plain English', command: 'plain', body: 'Say it in plain English for a reader who reads English as a second language. Short sentences. Every claim with a number.' },
  { id: 'p_commit', title: 'Commit message', command: 'commit', body: 'Write the commit message for this diff: one line under 72 characters, a blank line, then the why in two or three sentences.' },
];

export const PRESETS: Preset[] = [
  { id: 'ps_coding', name: 'Coding, high effort', model: MODEL, profile: null, effort: 'high', thinking: true, tools: ['bash', 'read_file', 'write_file', 'edit_file'], prompt_id: null },
  { id: 'ps_quick', name: 'Quick chat', model: MODEL, profile: null, effort: 'low', thinking: false, tools: [], prompt_id: null },
  { id: 'ps_research', name: 'Research', model: MODEL, profile: null, effort: 'high', thinking: true, tools: ['search', 'fetch_page', 'notes_search'], prompt_id: null },
];

export function sandboxFor(running: boolean, stopped_reason: SandboxStatus['stopped_reason'] = running ? null : 'idle'): SandboxStatus {
  return {
    state: running ? 'running' : 'stopped',
    stopped_reason,
    ports: running ? [{ port: 8080, label: 'site' }] : [],
    image: 'tandem-sandbox:1.3 (debian 13, python 3.13, node 24)',
    cwd: '/work/sync',
    cpus: 4,
    memory_bytes: running ? 412e6 : 0,
    memory_limit_bytes: 1024 ** 3,
    disk_bytes: 186e6,
    disk_limit_bytes: 20e9,
    network: 'off',
    allowlist: [],
    uptime_s: running ? 1_840 : 0,
    error: null,
  };
}

export function contextFor(messageTokens: number, opts: { tools: boolean; files: number; thinking: number; warmAll: boolean; profile: string | null; effort: ChatSettings['effort']; thinkingOn: boolean; slot: string | null; cold?: number; packages: string[] }): ContextReport {
  const parts = [
    { kind: 'system' as const, label: 'system prompt', tokens: 1_840 },
    { kind: 'tools' as const, label: 'tool schemas', tokens: opts.tools ? 3_120 : 0 },
    { kind: 'skills' as const, label: 'skills and packages', tokens: 920 },
    { kind: 'files' as const, label: 'files', tokens: opts.files },
    { kind: 'images' as const, label: 'images', tokens: 0 },
    { kind: 'messages' as const, label: 'conversation', tokens: messageTokens },
    { kind: 'thinking' as const, label: 'thinking, this turn', tokens: opts.thinking },
    { kind: 'reserved' as const, label: 'kept for the answer', tokens: 8_192 },
  ];
  const used = parts.filter((p) => p.kind !== 'reserved').reduce((a, p) => a + p.tokens, 0);
  const cold = opts.cold != null ? Math.min(used, opts.cold) : opts.warmAll ? 0 : Math.min(used, 2_304 + Math.round(messageTokens * 0.04));
  return {
    window: 262_144,
    used,
    parts,
    compact_at: Math.round(262_144 * 0.8),
    cache: { warm: used - cold, cold, cold_eta_ms: cold ? Math.round(cold * 1.33) : null, source: cold === used ? 'none' : 'resident', slot: opts.slot },
    harness: { class: opts.profile === 'coding' ? 'coding' : opts.profile === 'chat' ? 'chat' : 'answer', effort: opts.effort === 'auto' ? 'high' : opts.effort, packages: opts.packages.map((p) => ({ name: p, tokens: 460 })), memory: { mode: 'on', why: null, index_notes: 10, index_tokens: 407, bodies: 0, body_tokens: 0, card_tokens: 190 } },
    memory: { kv_bytes: used * 109_056, resident_bytes: 2.4e9, sandbox_disk_bytes: 186e6, available_bytes: 38.4e9, floor_bytes: 15 * 1024 ** 3, sandbox_bytes: 412e6 },
    model: { name: MODEL, profile: opts.profile, weights: 'EXL3 4.5 bpw', drafter: 'MTP head', max_len: 262_144, effort: opts.effort, thinking: opts.thinkingOn, backend: 'engine' },
  };
}

export const COMPACTION: Compaction = {
  id: 'cp_1',
  at: iso(-26 * 60_000),
  before_tokens: 98_120,
  after_tokens: 31_400,
  summary:
    'Compared EXL3 4.5 and 6.0 on this box: 4.5 keeps 3.9 GB more room for the KV pool and decodes 7 % faster; 6.0 is closer to BF16 on code (top-1 agreement 0.986 against 0.971). The user chose 4.5 for daily use and asked for a weekly check of code quality. Open: the long-prompt (64k) number for 6.0 is missing.',
};

// ---- the sandbox's files ----------------------------------------------------------------------

export const FILES: Record<string, string> = {
  '/work/sync/README.md': '# sync\n\nA small worker that drains a queue of jobs and writes results to disk.\n\n    uv run pytest -q\n',
  '/work/sync/pyproject.toml': '[project]\nname = "sync"\nversion = "0.3.1"\nrequires-python = ">=3.12"\ndependencies = []\n\n[tool.pytest.ini_options]\naddopts = "-q"\n',
  '/work/sync/src/sync/worker.py':
    'import time\nfrom .queue import Queue\n\n\nclass Worker:\n    def __init__(self, q: Queue, tick: float = 0.05):\n        self.q = q\n        self.tick = tick\n        self.done: list[str] = []\n\n    def run_once(self) -> bool:\n        job = self.q.pop()\n        if job is None:\n            return False\n        time.sleep(self.tick)\n        self.done.append(job)\n        return True\n\n    def drain(self, deadline_s: float) -> int:\n        start = time.monotonic()\n        n = 0\n        while time.monotonic() - start < deadline_s:\n            if not self.run_once():\n                break\n            n += 1\n        return n\n',
  '/work/sync/src/sync/queue.py': 'from collections import deque\n\n\nclass Queue:\n    def __init__(self):\n        self._d: deque[str] = deque()\n\n    def push(self, job: str) -> None:\n        self._d.append(job)\n\n    def pop(self) -> str | None:\n        return self._d.popleft() if self._d else None\n\n    def __len__(self) -> int:\n        return len(self._d)\n',
  '/work/sync/tests/test_worker.py':
    'from sync.queue import Queue\nfrom sync.worker import Worker\n\n\ndef test_drain_takes_all_jobs():\n    q = Queue()\n    for i in range(20):\n        q.push(f"job-{i}")\n    w = Worker(q, tick=0.01)\n    assert w.drain(deadline_s=0.5) == 20\n\n\ndef test_drain_stops_at_deadline():\n    q = Queue()\n    for i in range(200):\n        q.push(f"job-{i}")\n    w = Worker(q, tick=0.01)\n    n = w.drain(deadline_s=0.1)\n    assert 5 <= n <= 12\n',
};

export const ARTIFACTS: Record<string, { kind: string; mime: string; title: string; content: string }> = {
  'preview:shop-page': {
    kind: 'preview',
    mime: 'text/html',
    title: 'Bike repair shop page',
    content: `<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Spoke &amp; Chain</title>
<style>
:root{--ink:#1d2a1f;--paper:#f3f0e6;--moss:#3f6b4a;--rust:#b4552d}
*{box-sizing:border-box}body{margin:0;font:17px/1.5 Georgia,serif;color:var(--ink);background:var(--paper)}
header{padding:28px 24px 8px;display:flex;justify-content:space-between;align-items:baseline}
h1{margin:0;font-size:28px;letter-spacing:-.02em}nav a{color:var(--moss);margin-left:18px;text-decoration:none}
.hero{padding:48px 24px 36px;max-width:760px}.hero h2{font-size:44px;line-height:1.05;margin:0 0 14px}
.hero p{font-size:20px;max-width:34ch}.cta{display:inline-block;margin-top:18px;background:var(--rust);color:#fff;padding:12px 20px;border-radius:4px;text-decoration:none}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:18px;padding:0 24px 48px}
.card{background:#fff;border:1px solid #d9d4c4;padding:18px;border-radius:6px}.card b{display:block;font-size:22px;color:var(--moss)}
footer{padding:20px 24px;border-top:1px solid #d9d4c4;font-size:14px;color:#555}
</style></head><body>
<header><h1>Spoke &amp; Chain</h1><nav><a href="#">Repairs</a><a href="#">Prices</a><a href="#">Hours</a></nav></header>
<section class="hero"><h2>Your bike, back on the road the same day.</h2><p>Brakes, gears, flats and full services. Walk in before noon and ride home after work.</p><a class="cta" href="#">Book a slot</a></section>
<section class="grid"><div class="card"><b>12 EUR</b>Flat fix, tube included</div><div class="card"><b>35 EUR</b>Brake and gear tune</div><div class="card"><b>89 EUR</b>Full service, parts extra</div></section>
<footer>Open Tuesday to Saturday, 9 to 18. No appointment needed for small repairs.</footer>
</body></html>`,
  },
  'diff:worker': {
    kind: 'diff',
    mime: 'text/x-diff',
    title: 'worker.py: deadline check after the job',
    content: `--- a/src/sync/worker.py
+++ b/src/sync/worker.py
@@ -17,9 +17,11 @@ class Worker:
     def drain(self, deadline_s: float) -> int:
         start = time.monotonic()
         n = 0
-        while time.monotonic() - start < deadline_s:
-            if not self.run_once():
-                break
+        while True:
+            if time.monotonic() - start >= deadline_s:
+                break
+            if not self.run_once():
+                break
             n += 1
         return n
--- a/tests/test_worker.py
+++ b/tests/test_worker.py
@@ -14,4 +14,4 @@ def test_drain_stops_at_deadline():
     w = Worker(q, tick=0.01)
     n = w.drain(deadline_s=0.1)
-    assert 5 <= n <= 12
+    assert 8 <= n <= 11
`,
  },
  'md:meal-plan': {
    kind: 'markdown',
    mime: 'text/markdown',
    title: 'Weekly meal plan',
    content: `# Weekly meal plan\n\n| Day | Lunch | Dinner |\n|-|-|-|\n| Monday | lentil soup | pasta with tomato and basil |\n| Tuesday | rice bowl, egg, greens | baked fish, potatoes |\n| Wednesday | leftover pasta | chickpea curry |\n| Thursday | tomato sandwich | vegetable stir fry |\n| Friday | soup from Monday, frozen | pizza night |\n| Saturday | big salad | roast chicken, carrots |\n| Sunday | chicken leftovers wrap | pancakes |\n\nShopping, once: lentils 500 g, pasta 1 kg, rice 1 kg, eggs 12, chickpeas 2 tins, tomatoes 2 kg, greens 3 bunches, one chicken, fish 600 g, potatoes 2 kg, carrots 1 kg, flour 1 kg, milk 2 l.\n`,
  },
  'md:plan': {
    kind: 'plan',
    mime: 'text/markdown',
    title: 'Plan: fix the flaky test',
    content: `# Plan\n\n1. Read the test and the worker (done)\n2. Run the one test 200 times to see the failure rate (done: 3 of 200)\n3. Move the deadline check after the job\n4. Tighten the assertion to the range the tick allows\n5. Run 200 times again\n\nOpen: a caller that depends on the old count.\n`,
  },
  'image:acceptance': {
    kind: 'image',
    mime: 'image/svg+xml',
    title: 'Acceptance per round, this chat',
    content: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 260" width="640" height="260"><rect width="640" height="260" fill="#161d25"/><g stroke="#2b3542"><line x1="50" y1="210" x2="610" y2="210"/><line x1="50" y1="150" x2="610" y2="150"/><line x1="50" y1="90" x2="610" y2="90"/><line x1="50" y1="30" x2="610" y2="30"/></g><polyline fill="none" stroke="#3987e5" stroke-width="2.5" points="50,120 120,98 190,110 260,84 330,92 400,70 470,76 540,62 610,66"/><circle cx="610" cy="66" r="5" fill="#f7a24f"/><g fill="#93a1b3" font-family="monospace" font-size="12"><text x="8" y="34">5.0</text><text x="8" y="94">4.0</text><text x="8" y="154">3.0</text><text x="8" y="214">2.0</text><text x="50" y="240">tokens per round, one point per turn</text></g></svg>`,
  },
};

export function iso(offsetMs: number, now = Date.now()): string {
  return new Date(now + offsetMs).toISOString();
}

// ---- web search (VIS-157): the stored settings and one made-up result list ----------------------

export const WEB_SEARCH: WebSearchSettings = { provider: 'duckduckgo', searxng_url: '', searxng_auth_header: '', searxng_auth_set: false, max_results: 8, safe_search: 'moderate', language: '', time_range: '', timeout_s: 10 };

export const SEARCH_RESULTS: WebSearchResult[] = [
  { title: 'Speculative decoding: draft, verify, accept', url: 'https://example.org/notes/spec-decoding', snippet: 'A small model drafts a block; the large one verifies it in one pass.', date: '2026-08-02' },
  { title: 'Tokens per round on your own text', url: 'https://docs.example.com/tokens-per-round', snippet: 'Acceptance and tokens per round, block size 4 to 16.' },
  { title: 'Draft models compared', url: 'https://www.example.net/blog/drafters', snippet: 'Three drafter sizes on prose, code and chat.' },
];

// ---- tasks (VIS-E15): a main agent with workers, on a made-up ocean scene --------------------------

/** the screenshot render_page took: a small ocean at sunset */
export const SHOT_SVG = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 384" width="640" height="384"><defs><linearGradient id="sky" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#2b2350"/><stop offset=".55" stop-color="#c8557a"/><stop offset="1" stop-color="#f7a24f"/></linearGradient><linearGradient id="sea" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#284c6e"/><stop offset="1" stop-color="#0d1a2a"/></linearGradient></defs><rect width="640" height="230" fill="url(#sky)"/><circle cx="320" cy="222" r="44" fill="#ffd38a"/><rect y="226" width="640" height="158" fill="url(#sea)"/><g stroke="#f7c27a" stroke-opacity=".55" stroke-width="3"><line x1="270" y1="246" x2="370" y2="246"/><line x1="285" y1="268" x2="355" y2="268"/><line x1="298" y1="292" x2="342" y2="292"/></g></svg>`;
const SHOT_URL = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(SHOT_SVG)}`;
ARTIFACTS['image:shot-3'] = { kind: 'image', mime: 'image/svg+xml', title: 'shot-3, index.html at 1280 px', content: SHOT_SVG };
ARTIFACTS['image:shot-7'] = { kind: 'image', mime: 'image/svg+xml', title: 'shot-7, index.html at 1280 px', content: SHOT_SVG };

/** a worker's finished call */
function wcall(id: string, name: string, input: Record<string, unknown>, output: string, opts: { status?: ToolCallPart['status']; shot?: string } = {}): ToolCallPart {
  return {
    type: 'tool_call',
    id,
    name,
    input: JSON.stringify(input, null, 2),
    output: { content: output, is_error: opts.status === 'failed', bytes: output.length, exit_code: opts.status === 'failed' ? 1 : name === 'bash' ? 0 : null, ...(opts.shot ? { images: [{ id: opts.shot, name: `${opts.shot}.png`, kind: 'image' as const, bytes: 41_200, mime: 'image/svg+xml', preview: SHOT_URL, url: SHOT_URL }] } : {}) },
    status: opts.status ?? 'done',
    started_at: iso(-2 * 3_600_000),
    ended_at: iso(-2 * 3_600_000 + 1_800),
    permission: null,
    approved: 'auto',
    pane: opts.shot ? { kind: 'image', ref: `image:${opts.shot}`, title: opts.shot } : null,
  };
}

function transcript(id: string, parts: Part[], status: Message['status'] = 'done'): Message[] {
  return [{ id, role: 'assistant', created_at: iso(-2 * 3_600_000), parent_id: null, run_id: null, parts, status }];
}

const OCEAN_BRIEF = (goal: string, files: string, check: string) => `Goal: ${goal}\nFiles: ${files}\nConstraints: one page, no build step; keep the WebGL2 fallback working\nCheck: ${check}`;

const REPORT_1: TaskReport = {
  status: 'done',
  summary: 'The page draws a sky gradient, a sun and a flat sea with WebGPU, and falls back to WebGL2 when the adapter is missing.',
  changed_files: [
    { path: 'index.html', change: 'the canvas, the module script and a fallback note' },
    { path: 'src/main.js', change: 'adapter request, the WebGL2 fallback, the render loop' },
    { path: 'src/ocean.wgsl', change: 'sky, sun and sea in one fragment shader' },
  ],
  checks: [
    { what: 'render_page index.html', result: '0 console errors, shot-3', call_id: 'w1_c6', shot: 'shot-3' },
    { what: 'bash "node --check src/main.js"', result: 'exit 0', call_id: 'w1_c5' },
  ],
  expected: [{ item: 'the graphics API the check browser used', answer: 'WebGL2 (the headless browser has no WebGPU adapter)' }],
  open_problems: ['none'],
};
const CHECK_1 = `- Changed in /workspace (snapshot diff): index.html (+38 -0), src/main.js (+112 -0), src/ocean.wgsl (+64 -0)
- Changed but not in the report: none
- Checks that ran: render_page index.html (call w1_c6: 0 console errors, shot-3); bash "node --check src/main.js" (call w1_c5: exit 0)
- Claimed in Checks with no matching call: none`;

const REPORT_2: TaskReport = {
  status: 'partial',
  summary: 'The banding came from 8-bit normals; they are float now. The horizon blend still steps at 1280 px: the step limit came first.',
  changed_files: [{ path: 'src/ocean.wgsl', change: 'normals in f32, smoothstep at the horizon' }],
  checks: [
    { what: 'render_page index.html', result: '0 console errors, shot-7', call_id: 'w2_c31', shot: 'shot-7' },
    { what: 'bash "npm test"', result: 'exit 0', call_id: null },
  ],
  expected: [{ item: 'the FPS line from the console after render_page', answer: 'fps 58.9' }],
  open_problems: ['the horizon still shows two bands at 1280 px', 'no test covers the shader'],
};
const CHECK_2 = `- Changed in /workspace (snapshot diff): src/ocean.wgsl (+41 -12), src/main.js (+3 -1)
- Changed but not in the report: src/main.js
- Checks that ran: render_page index.html (call w2_c31: 0 console errors, shot-7)
- Claimed in Checks with no matching call: bash "npm test"`;

const CHECK_3 = `- Changed in /workspace (snapshot diff): none
- Changed but not in the report: none
- Checks that ran: bash "npm install lil-gui" (call w3_c2: exit 1)
- Claimed in Checks with no matching call: none`;

const CHECK_4 = `- Changed in /workspace (snapshot diff): src/sky.js (+18 -0)
- Changed but not in the report: src/sky.js
- Checks that ran: none
- Claimed in Checks with no matching call: none`;

function task(n: number, p: Partial<Task> & Pick<Task, 'title' | 'status'>): Task {
  return { id: `task_ocean_${n}`, conversation_id: 'c_ocean', run_id: `run_ocean_${n}`, call_id: `call_task_${n}`, n, brief: '', expect: '', steps: 0, max_steps: 40, prompt_tokens: 0, completion_tokens: 0, cached_tokens: 0, started: iso(-2 * 3_600_000 + n * 600_000), ended: iso(-2 * 3_600_000 + n * 600_000 + 300_000), report: null, report_source: 'worker', check: null, error: null, ...p };
}

/** c_ocean's four tasks: done, partial at the step limit (the report built by chatd), failed and
 *  cancelled, each with its transcript */
export const OCEAN_TASKS: Task[] = [
  task(1, {
    title: 'Scene skeleton with WebGPU and a WebGL2 fallback',
    status: 'done',
    brief: OCEAN_BRIEF('index.html draws a sky, a sun and a sea with WebGPU, and with WebGL2 when there is no adapter', 'index.html (new), src/main.js (new), src/ocean.wgsl (new)', 'render_page index.html with 0 console errors'),
    expect: 'the graphics API the check browser used',
    steps: 9,
    prompt_tokens: 38_420,
    completion_tokens: 4_110,
    cached_tokens: 31_900,
    stats: { ms: 248_000 },
    report: REPORT_1,
    check: CHECK_1,
    messages: transcript('w1_m', [
      { type: 'thinking', text: 'Three files. Ask for the adapter first; without one, build the same scene in a WebGL2 fragment shader so the check browser can show it.', tokens: 32, done: true },
      wcall('w1_c1', 'write_file', { path: 'index.html', bytes: 1_204 }, 'wrote index.html (1,204 bytes)'),
      wcall('w1_c2', 'write_file', { path: 'src/ocean.wgsl', bytes: 2_340 }, 'wrote src/ocean.wgsl (2,340 bytes)'),
      wcall('w1_c3', 'write_file', { path: 'src/main.js', bytes: 3_980 }, 'wrote src/main.js (3,980 bytes)'),
      wcall('w1_c5', 'bash', { command: 'node --check src/main.js' }, ''),
      wcall('w1_c6', 'render_page', { path: 'index.html' }, 'loaded index.html at 1280 px: 0 console errors, WebGL2 (no WebGPU adapter), screenshot shot-3', { shot: 'shot-3' }),
      { type: 'text', text: 'Status: done\nSummary: the page draws a sky gradient, a sun and a flat sea…' },
    ]),
  }),
  task(2, {
    title: 'Ocean shader: fix the horizon banding',
    status: 'partial (step limit)',
    brief: OCEAN_BRIEF('the horizon shows no visible bands at 1280 px', 'src/ocean.wgsl (the sea and the horizon blend), src/main.js (uniforms)', 'render_page index.html; the console prints an fps line'),
    expect: 'the FPS line from the console after render_page',
    steps: 40,
    prompt_tokens: 57_990,
    completion_tokens: 3_794,
    cached_tokens: 49_100,
    stats: { ms: 372_000 },
    report: REPORT_2,
    report_source: 'chatd',
    check: CHECK_2,
    messages: transcript('w2_m', [
      { type: 'thinking', text: 'Banding at the horizon: either the normals are quantised or the blend is linear. Look at the normals first.', tokens: 24, done: true },
      wcall('w2_c1', 'read_file', { path: 'src/ocean.wgsl', offset: 1, limit: 80 }, '  1  struct Out { @builtin(position) pos: vec4f, @location(0) uv: vec2f };\n  …'),
      wcall('w2_c2', 'edit_file', { path: 'src/ocean.wgsl', find: 'vec3<u8>', replace: 'vec3f' }, 'edited src/ocean.wgsl: 1 block, +1 -1 lines'),
      wcall('w2_c31', 'render_page', { path: 'index.html' }, 'loaded index.html at 1280 px: 0 console errors; console: fps 58.9; screenshot shot-7', { shot: 'shot-7' }),
      wcall('w2_c39', 'edit_file', { path: 'src/ocean.wgsl', find: 'mix(sky, sea, t)', replace: 'mix(sky, sea, smoothstep(0.0, 1.0, t))' }, 'edited src/ocean.wgsl: 1 block, +1 -1 lines'),
    ]),
  }),
  task(3, {
    title: 'Wave height slider with lil-gui',
    status: 'failed',
    brief: OCEAN_BRIEF('a slider sets the wave height live', 'src/main.js (uniforms), index.html', 'render_page index.html shows the slider'),
    steps: 3,
    prompt_tokens: 14_220,
    completion_tokens: 610,
    cached_tokens: 11_300,
    stats: { ms: 41_000 },
    report: { status: 'failed', summary: 'lil-gui could not be installed: the sandbox has no network.', changed_files: [], checks: [{ what: 'bash "npm install lil-gui"', result: 'exit 1, ENOTFOUND registry.npmjs.org', call_id: 'w3_c2' }], expected: [], open_problems: ['turn the sandbox network on, or allow a slider without a library'] },
    check: CHECK_3,
    messages: transcript('w3_m', [wcall('w3_c2', 'bash', { command: 'npm install lil-gui' }, 'npm error code ENOTFOUND\nnpm error network request to https://registry.npmjs.org/lil-gui failed', { status: 'failed' })]),
  }),
  task(4, {
    title: 'Sun and sky colours at sunset',
    status: 'cancelled',
    brief: OCEAN_BRIEF('the sky reads as a warm sunset', 'src/sky.js (new), src/ocean.wgsl', 'render_page index.html'),
    steps: 4,
    prompt_tokens: 12_800,
    completion_tokens: 402,
    cached_tokens: 9_700,
    stats: { ms: 52_000 },
    report: null,
    report_source: 'chatd',
    check: CHECK_4,
    messages: transcript('w4_m', [wcall('w4_c3', 'write_file', { path: 'src/sky.js', bytes: 640 }, 'wrote src/sky.js (640 bytes)')], 'stopped'),
  }),
];

/** what the main agent read back for a task (chatd's render_result, short) */
export function taskResult(t: Task): string {
  const head = `Result of task ${t.n}, given to you as data: the worker's report as chatd parsed it, then chatd's check.\ntask ${t.n} "${t.title}": ${t.status}, ${t.steps} calls (worker: ${t.prompt_tokens.toLocaleString('en-US')} prompt tokens, ${t.completion_tokens.toLocaleString('en-US')} completion)`;
  const rep = t.report ? `\nStatus: ${t.report.status}\nSummary: ${t.report.summary}` : '';
  return `${head}${rep}${t.check ? `\nchatd check:\n${t.check}` : ''}\nThe transcript is in the Task ${t.n} card, not in your context.`;
}

/** a running worker's context: the brief and the tools, its steps growing; its Now and figures */
export function workerContext(t: Pick<Task, 'id' | 'n' | 'title' | 'max_steps'>, step: number, live?: { phase: RunPhase; now: string; metrics: TaskMetrics | null }): WorkerContext {
  const steps = 1_900 * step;
  const files = Math.min(6_000, 400 * step);
  const split = { system: 2_410, tools: 3_880, brief: 610, steps, files };
  const used = split.system + split.tools + split.brief + steps + files;
  return { task_id: t.id, n: t.n, title: t.title, used, window: 262_144, split, step, max_steps: t.max_steps, cache: { warm: used - 2_200, cold: 2_200, cold_eta_ms: 3_000, source: 'resident', slot: 'slot 0' }, ...(live ?? {}) };
}

/** a worker's live figures: its prefill while it reads its prompt, else its decode */
export function workerMetrics(taskId: string, phase: RunPhase, tin: number, tout: number, prefill: TaskMetrics['prefill'] = null): TaskMetrics {
  const decoding = phase === 'writing' || phase === 'thinking';
  return { task_id: taskId, phase, prefill: phase === 'prefill' ? prefill : null, decode_tps: decoding ? 37.2 : null, ttft_ms: phase === 'prefill' ? null : 640, tokens_per_round: 3.1, acceptance: 0.62, prompt_tokens: tin, completion_tokens: tout, thinking_used: Math.round(tout * 0.3) };
}
