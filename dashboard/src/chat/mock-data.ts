// The mock's content: generic conversations, the sandbox's files, the artifacts, the settings
// stores. Nothing personal: a made-up project (a sync worker), a made-up shop page, a meal plan.

import type { Capabilities, ChatSettings, Compaction, ContextReport, McpServer, PackageInfo, Preset, Prompt, SandboxStatus, ToolInfo, WebSearchResult, WebSearchSettings } from './types';

export const MODEL = 'TandemLLM';

export const DEFAULT_SETTINGS: ChatSettings = {
  model: MODEL,
  profile: 'coding',
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
  max_steps: 50,
  sandbox_network: 'off',
  permission_mode: 'auto',
};

export const CAPABILITIES: Capabilities = {
  models: [
    {
      id: MODEL,
      name: 'TandemLLM',
      weights: 'EXL3 4.5 bpw',
      max_len: 262_144,
      profiles: [
        { id: 'chat', name: 'chat', description: 'short answers, thinking off by default' },
        { id: 'coding', name: 'coding', description: 'tools on, high effort, the sandbox ready' },
        { id: 'writing', name: 'writing', description: 'longer prose, no tools' },
        { id: 'research', name: 'research', description: 'web search and notes, citations kept' },
      ],
    },
  ],
  efforts: ['auto', 'low', 'medium', 'high', 'xhigh'],
  commands: [
    { command: 'clear', description: 'start a new chat' },
    { command: 'compact', description: 'fold the older turns into a summary' },
    { command: 'model', description: 'pick the model and the profile' },
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
  { id: 'ps_coding', name: 'Coding, high effort', model: MODEL, profile: 'coding', effort: 'high', thinking: true, tools: ['bash', 'read_file', 'write_file', 'edit_file'], prompt_id: null },
  { id: 'ps_quick', name: 'Quick chat', model: MODEL, profile: 'chat', effort: 'low', thinking: false, tools: [], prompt_id: null },
  { id: 'ps_research', name: 'Research', model: MODEL, profile: 'research', effort: 'high', thinking: true, tools: ['search', 'fetch_page', 'notes_search'], prompt_id: null },
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
