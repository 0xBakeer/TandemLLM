// The Chat tab's phase 2 leftovers and phases 3 and 4 against a real chatd, a real sandbox and a
// real engine (VIS-106, VIS-108, VIS-110, SRV-358, SRV-360, SRV-362, SRV-364). Spark only:
//   PW_LIVE=1 PW_BASE=http://127.0.0.1:4181/dashboard/ npx playwright test -c pw-chat-live.config.ts e2e/chat-live3.spec.ts
// 1. a WebSocket from a page in the preview pane to a server in the sandbox;
// 2. Compact now: the divider in the thread;
// 3. two MCP servers and an imported Open WebUI tool answer in one chat;
// 4. plan mode: a plan, Approve, the run that carries it out, the todo list, Rewind;
// 5. the PWA: the service worker caches no API answer, the page is installable;
// 6. a local notification when a chat in another tab waits for permission, and the badge.
// Every test stops its chat's sandbox at the end (one container at a time on the box).
import { expect, test, type APIRequestContext, type Browser, type Page } from '@playwright/test';

const SHOTS = process.env.PW_SHOTS ?? 'e2e-live-shots';
const DESKTOP = { width: 1440, height: 900 };

test.skip(!process.env.PW_LIVE, 'the live tier runs on the Spark with PW_LIVE=1');

type Snap = {
  run: { phase: string } | null;
  settings: { plan_mode?: boolean };
  todos?: { text: string; state: string }[];
  compactions: { id: string }[];
  messages: { id: string; role: string; status: string; parts: { type: string; id?: string; status?: string; name?: string; text?: string; output?: { content: string } | null }[] }[];
};

async function open(browser: Browser, hash: string) {
  const ctx = await browser.newContext({ viewport: DESKTOP });
  const page = await ctx.newPage();
  await page.goto(`./${hash}`);
  await page.locator('qse-chat').waitFor();
  await expect(page.locator('.ch-mock-badge')).toHaveCount(0);
  return { ctx, page };
}

async function newChat(request: APIRequestContext, settings: Record<string, unknown> = {}): Promise<string> {
  const r = await request.post('../v1/chat-app/conversations', { data: { settings: { thinking: false, ...settings } } });
  return ((await r.json()) as { id: string }).id;
}

async function snap(request: APIRequestContext, id: string): Promise<Snap> {
  return (await (await request.get(`../v1/chat-app/conversations/${id}`)).json()) as Snap;
}

async function say(request: APIRequestContext, id: string, text: string) {
  const r = await request.post(`../v1/chat-app/conversations/${id}/messages`, { data: { client_msg_id: `t${Date.now()}${Math.random()}`, text } });
  expect(r.status(), await r.text()).toBe(202);
}

async function idle(request: APIRequestContext, id: string, timeout = 240_000) {
  await expect.poll(async () => (await snap(request, id)).run, { timeout, intervals: [1000] }).toBeNull();
  return snap(request, id);
}

const calls = (s: Snap) => s.messages.flatMap((m) => m.parts.filter((p) => p.type === 'tool_call'));

async function stopSandbox(request: APIRequestContext, id: string) {
  await request.post(`../v1/chat-app/conversations/${id}/sandbox/stop`);
}

/** a command typed into the chat's shell over its WebSocket, from the page */
async function shell(page: Page, id: string, command: string) {
  await page.evaluate(
    ([id, command]) =>
      new Promise<void>((resolve, reject) => {
        const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/v1/chat-app/conversations/${id}/terminals/shell/ws`);
        ws.binaryType = 'arraybuffer';
        ws.onopen = () => {
          ws.send(new TextEncoder().encode(command + '\r'));
          setTimeout(() => (ws.close(), resolve()), 2500);
        };
        ws.onerror = () => reject(new Error('the terminal did not open'));
      }),
    [id, command],
  );
}

const WS_SERVER = `import base64, hashlib, socket, sys, threading
G = b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
def serve(c):
    h = c.recv(4096).decode()
    key = [l.split(":", 1)[1].strip() for l in h.split("\\r\\n") if l.lower().startswith("sec-websocket-key")][0]
    acc = base64.b64encode(hashlib.sha1(key.encode() + G).digest()).decode()
    c.sendall(("HTTP/1.1 101 Switching Protocols\\r\\nUpgrade: websocket\\r\\nConnection: Upgrade\\r\\nSec-WebSocket-Accept: " + acc + "\\r\\n\\r\\n").encode())
    while True:
        b = c.recv(2)
        if len(b) < 2 or b[0] & 15 == 8: return
        n = b[1] & 127
        mask = c.recv(4)
        data = bytes(x ^ mask[i % 4] for i, x in enumerate(c.recv(n)))
        out = b"echo:" + data
        c.sendall(bytes([0x81, len(out)]) + out)
s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(("127.0.0.1", int(sys.argv[1]))); s.listen(8)
while True:
    c, _ = s.accept(); threading.Thread(target=serve, args=(c,), daemon=True).start()
`;

const WS_PAGE = `<!doctype html><html><body style="font-family:sans-serif;background:#f6f1e7;padding:24px">
<h1>WebSocket through the preview</h1><p id="s">connecting</p><script>
const u = new URL('../8124/', location.href); u.protocol = u.protocol === 'https:' ? 'wss:' : 'ws:';
const ws = new WebSocket(u);
ws.onopen = () => { document.getElementById('s').textContent = 'open'; ws.send('hello from the preview'); };
ws.onmessage = (e) => { document.getElementById('s').textContent = e.data; };
ws.onerror = () => { document.getElementById('s').textContent = 'error'; };
</script></body></html>`;

test('a WebSocket from the preview pane reaches a server in the sandbox', async ({ browser, request }) => {
  const id = await newChat(request);
  for (const [path, body] of [['/workspace/ws.py', WS_SERVER], ['/workspace/index.html', WS_PAGE]]) {
    const r = await request.put(`../v1/chat-app/conversations/${id}/sandbox/file?path=${path}`, { data: body, headers: { 'Content-Type': 'text/plain' } });
    expect(r.status()).toBe(204);
  }
  expect((await request.post(`../v1/chat-app/conversations/${id}/sandbox/start`)).status()).toBe(200);
  const a = await open(browser, `#/chat/${id}`);
  await shell(a.page, id, 'cd /workspace && (nohup python3 ws.py 8124 >/tmp/ws.log 2>&1 &) && (nohup python3 -m http.server 8123 --bind 127.0.0.1 >/tmp/http.log 2>&1 &)');
  await request.patch(`../v1/chat-app/conversations/${id}`, { data: { ui: { panes: [{ id: 'preview:port:8123', kind: 'preview', ref: 'port:8123', title: 'port 8123' }], active: 'preview:port:8123', split: 0.6 } } });
  await a.page.reload();
  const frame = a.page.frameLocator('.ch-preview iframe');
  await expect(frame.locator('#s')).toHaveText('echo:hello from the preview', { timeout: 30_000 });
  await a.page.screenshot({ path: `${SHOTS}/p3-1-preview-websocket.png` });
  await stopSandbox(request, id);
  await a.ctx.close();
});

test('Compact now: the divider sits above the turns the model still reads', async ({ browser, request }) => {
  const id = await newChat(request);
  for (const q of ['Name one prime number. One word.', 'Name one even number. One word.', 'Name one colour. One word.']) {
    await say(request, id, q);
    await idle(request, id);
  }
  const r = await request.post(`../v1/chat-app/conversations/${id}/compact`);
  expect(r.status()).toBe(202);
  await expect.poll(async () => (await snap(request, id)).compactions.length, { timeout: 180_000 }).toBe(1);
  const a = await open(browser, `#/chat/${id}`);
  const d = a.page.locator('.ch-divider');
  await expect(d).toContainText('Earlier messages summarized');
  await d.getByRole('button').click();
  await expect(d.locator('.ch-cp-summary')).not.toBeEmpty();
  await a.page.screenshot({ path: `${SHOTS}/p3-2-compaction-divider.png` });
  await a.ctx.close();
});

const CALC = `"""
title: Calculator
description: Arithmetic the model should not do in its head.
"""


class Tools:
    def calculate(self, expression: str) -> str:
        """
        Evaluate an arithmetic expression.
        :param expression: for example "17 * 23"
        """
        return str(eval(expression, {"__builtins__": {}}, {}))
`;

test('two MCP servers and an imported Open WebUI tool answer in one chat', async ({ browser, request }) => {
  const servers = (await (await request.get('../v1/chat-app/mcp/servers')).json()) as { id: string; name: string }[];
  const want = [
    { name: 'deepwiki', transport: 'http', url: 'https://mcp.deepwiki.com/mcp' },
    { name: 'time', transport: 'stdio', command: 'uvx', args: ['mcp-server-time'] },
  ];
  for (const w of want) if (!servers.some((s) => s.name === w.name)) expect((await request.post('../v1/chat-app/mcp/servers', { data: w })).status()).toBe(200);
  const a = await open(browser, '#/chat/settings/tools');
  await a.page.locator('.ch-file-btn', { hasText: 'Import a tools file' }).locator('input[type=file]').setInputFiles({ name: 'calculator.py', mimeType: 'text/x-python', buffer: Buffer.from(CALC) });
  await expect(a.page.locator('.ch-kit', { hasText: 'Calculator' }).first()).toBeVisible({ timeout: 10_000 });
  await a.page.screenshot({ path: `${SHOTS}/p3-3-owui-imported.png`, fullPage: true });
  const ids = ((await (await request.get('../v1/chat-app/mcp/servers')).json()) as { id: string; name: string }[]).filter((s) => want.some((w) => w.name === s.name)).map((s) => s.id);
  const id = await newChat(request, { mcp: ids });
  await request.patch(`../v1/chat-app/conversations/${id}/settings`, { data: { policies: { 'mcp__time__get_current_time': 'always', 'mcp__deepwiki__read_wiki_structure': 'always', 'owui__calculator__calculate': 'always' } } });
  await say(request, id, 'Do three tool calls, one after the other: mcp__time__get_current_time with timezone "Europe/Berlin"; owui__calculator__calculate with expression "17 * 23"; mcp__deepwiki__read_wiki_structure with repoName "facebook/react". Then answer in one line: the Berlin hour, the product, and how many top-level topics the wiki has.');
  const s = await idle(request, id, 420_000);
  const done = calls(s).filter((c) => c.status === 'done').map((c) => c.name);
  console.log('calls:', calls(s).map((c) => `${c.name}:${c.status}`).join(' '));
  expect(done).toContain('owui__calculator__calculate');
  expect(done.some((n) => n?.startsWith('mcp__time__'))).toBe(true);
  expect(done.some((n) => n?.startsWith('mcp__deepwiki__'))).toBe(true);
  expect(calls(s).find((c) => c.name === 'owui__calculator__calculate')?.output?.content).toBe('391');
  await a.page.goto(`./#/chat/${id}`);
  await a.page.locator('.ch-foot').last().waitFor();
  await a.page.screenshot({ path: `${SHOTS}/p3-4-mcp-and-owui.png` });
  await stopSandbox(request, id);
  await a.ctx.close();
});

test('plan mode: a plan, Approve, the run that carries it out, then Rewind', async ({ browser, request }) => {
  const id = await newChat(request, { tools: ['bash', 'read_file', 'write_file', 'edit_file', 'glob', 'grep'] });
  await request.patch(`../v1/chat-app/conversations/${id}/settings`, { data: { policies: { write_file: 'always', edit_file: 'always' } } });
  const a = await open(browser, `#/chat/${id}`);
  await a.page.locator('.ch-pill', { hasText: 'plan' }).click();
  await expect(a.page.locator('.ch-pill', { hasText: 'plan' })).toHaveAttribute('aria-pressed', 'true');
  await a.page.locator('#ch-input').fill('Plan how to create notes/plan.md holding three lines: alpha, beta, gamma. Answer with a numbered plan of at most three steps; do not do it yet.');
  await a.page.locator('.ch-send').click();
  await expect(a.page.locator('.ch-planbar')).toBeVisible({ timeout: 300_000 });
  const first = await snap(request, id);
  expect(calls(first).every((c) => ['read_file', 'glob', 'grep', 'todo_write'].includes(c.name ?? '') || c.output?.content?.startsWith('plan mode'))).toBe(true);
  await a.page.screenshot({ path: `${SHOTS}/p4-1-plan.png` });
  await a.page.locator('.ch-planbar').getByRole('button', { name: 'Approve plan' }).click();
  await expect(a.page.locator('.ch-pill', { hasText: 'plan' })).toHaveAttribute('aria-pressed', 'false');
  const s = await idle(request, id, 420_000);
  expect((s.todos ?? []).length).toBeGreaterThan(0);
  const f = await request.get(`../v1/chat-app/conversations/${id}/sandbox/file?path=/workspace/notes/plan.md`);
  expect(f.status()).toBe(200);
  expect(((await f.json()) as { content: string }).content).toContain('beta');
  if (!(await a.page.locator('.ch-panes').count())) await a.page.getByRole('button', { name: 'Panes', exact: true }).click();
  await a.page.locator('qse-chat-panes').getByRole('button', { name: 'Open a pane' }).click();
  await a.page.locator('.ch-pane-menu [role="menuitem"]', { hasText: 'Plan' }).click();
  await expect(a.page.locator('.ch-todos .ch-todo').first()).toBeVisible();
  await a.page.screenshot({ path: `${SHOTS}/p4-2-approved-and-done.png` });
  // rewind to the approval: the file goes, the message comes back to the composer
  a.page.once('dialog', (d) => void d.accept());
  await a.page.locator('.ch-user').last().getByRole('button', { name: 'Rewind to here' }).click();
  await expect(a.page.locator('#ch-input')).toHaveValue('Carry out the approved plan.');
  const g = await request.get(`../v1/chat-app/conversations/${id}/sandbox/file?path=/workspace/notes/plan.md`);
  expect(g.status()).not.toBe(200);
  await a.page.screenshot({ path: `${SHOTS}/p4-3-rewound.png` });
  await stopSandbox(request, id);
  await a.ctx.close();
});

test('the PWA: the service worker caches the shell only, and the page is installable', async ({ browser }) => {
  const a = await open(browser, '#/chat');
  await expect.poll(async () => a.page.evaluate(async () => !!(await navigator.serviceWorker.getRegistration())), { timeout: 15_000 }).toBe(true);
  await a.page.reload();
  await expect.poll(async () => a.page.evaluate(() => !!navigator.serviceWorker.controller), { timeout: 15_000 }).toBe(true);
  await a.page.locator('qse-chat').waitFor();
  await a.page.waitForTimeout(1500);
  const cached = await a.page.evaluate(async () => {
    const out: string[] = [];
    for (const k of await caches.keys()) for (const r of await (await caches.open(k)).keys()) out.push(r.url);
    return out;
  });
  console.log('cached:', cached.length, 'urls');
  expect(cached.length).toBeGreaterThan(3);
  expect(cached.filter((u) => u.includes('/v1/'))).toEqual([]);
  const cdp = await a.ctx.newCDPSession(a.page);
  const inst = (await cdp.send('Page.getInstallabilityErrors')) as { installabilityErrors: { errorId: string }[] };
  console.log('installability errors:', JSON.stringify(inst.installabilityErrors));
  expect(inst.installabilityErrors).toEqual([]);
  await a.ctx.close();
});

test('a chat that waits for permission notifies another tab, with no text, and counts on the badge', async ({ browser, request }) => {
  const ctx = await browser.newContext({ viewport: DESKTOP });
  await ctx.grantPermissions(['notifications']);
  await ctx.addInitScript(() => {
    const seen: { title: string; body?: string }[] = [];
    (window as unknown as { __notes: typeof seen }).__notes = seen;
    ServiceWorkerRegistration.prototype.showNotification = async function (title: string, opts?: NotificationOptions) {
      seen.push({ title, body: opts?.body });
    };
    Object.defineProperty(document, 'hidden', { get: () => true });
  });
  const page = await ctx.newPage();
  await page.goto('./#/chat');
  await page.locator('qse-chat').waitFor();
  const id = await newChat(request, { tools: ['bash'] });
  await request.patch(`../v1/chat-app/conversations/${id}`, { data: { title: 'Notify probe' } });
  await say(request, id, 'Use the bash tool to run: echo secret-words-here. Do nothing else.');
  await expect.poll(async () => page.evaluate(() => (window as unknown as { __notes: { title: string; body?: string }[] }).__notes), { timeout: 300_000 }).toContainEqual({ title: 'Notify probe', body: 'waits for permission' });
  const notes = await page.evaluate(() => JSON.stringify((window as unknown as { __notes: unknown[] }).__notes));
  expect(notes).not.toContain('secret-words-here');
  await expect.poll(async () => page.title(), { timeout: 15_000 }).toMatch(/^\(\d+\) /);
  await page.goto('./#/chat/settings/device');
  await page.screenshot({ path: `${SHOTS}/p4-4-this-device.png` });
  const s = await snap(request, id);
  const ask = calls(s).find((c) => c.status === 'asking');
  if (ask) await request.post(`../v1/chat-app/conversations/${id}/permissions/${ask.id ?? ''}`, { data: { decision: 'deny' } });
  await request.post(`../v1/chat-app/conversations/${id}/stop`);
  await stopSandbox(request, id);
  await ctx.close();
});
