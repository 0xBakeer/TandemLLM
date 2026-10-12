// The Chat tab's phase 2 against a real chatd, a real sandbox and a real engine (SRV-351..357,
// VIS-105/107). Spark only, never on a laptop:
//   PW_LIVE=1 PW_BASE=http://127.0.0.1:4181/dashboard/ npx playwright test -c pw-chat-live.config.ts e2e/chat-live2.spec.ts
// 1. a bash call asks for permission on the desktop and on a phone at once; the phone allows it,
//    the desktop's prompt closes, the command runs in the sandbox and the answer uses its output;
// 2. the terminal pane: a shell in the sandbox, typed on the desktop, the same screen on the phone;
// 3. the preview pane shows a workspace page in a sandboxed iframe that cannot read the cookie;
// 4. the files and diff panes after the agent wrote a file.
// Every test stops its chat's sandbox at the end (one container at a time on the box).
import { expect, test, type APIRequestContext, type Browser, type Page } from '@playwright/test';

const SHOTS = process.env.PW_SHOTS ?? 'e2e-live-shots';
const DESKTOP = { width: 1440, height: 900 };
const PHONE = { width: 390, height: 844 };

test.skip(!process.env.PW_LIVE, 'the live tier runs on the Spark with PW_LIVE=1');

async function open(browser: Browser, hash: string, viewport = DESKTOP) {
  const ctx = await browser.newContext({ viewport, ...(viewport === PHONE ? { isMobile: true, hasTouch: true } : {}) });
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

async function snap(request: APIRequestContext, id: string) {
  return (await (await request.get(`../v1/chat-app/conversations/${id}`)).json()) as {
    run: { phase: string } | null;
    messages: { id: string; role: string; status: string; parts: { type: string; status?: string; name?: string; text?: string; output?: { content: string } | null }[] }[];
    sandbox: { state: string };
  };
}

async function stopSandbox(request: APIRequestContext, id: string) {
  await request.post(`../v1/chat-app/conversations/${id}/sandbox/stop`);
}

/** the header's Panes button opens the terminal first; the + menu adds the others */
async function openPane(page: Page, kind: 'Terminal' | 'Preview' | 'Files' | 'Diff') {
  if (!(await page.locator('.ch-panes').count())) await page.locator('button[aria-label="Panes"]').click();
  if (kind === 'Terminal') return;
  await page.locator('button[aria-label="Open a pane"]').first().click();
  await page.locator('.ch-pane-menu [role="menuitem"]', { hasText: kind }).click();
}

async function send(page: Page, text: string) {
  await page.locator('#ch-input').fill(text);
  await page.locator('.ch-send').click();
}

test('a permission prompt on two devices: the phone allows, the desktop closes, bash runs', async ({ browser, request }) => {
  const id = await newChat(request);
  const a = await open(browser, `#/chat/${id}`);
  const b = await open(browser, `#/chat/${id}`, PHONE);
  await send(a.page, 'Use the bash tool to run this command: python3 -c "print(6*7)" . Then tell me the number it printed, in one short sentence.');
  await expect(a.page.locator('.ch-perm')).toBeVisible({ timeout: 120_000 });
  await expect(b.page.locator('.ch-perm')).toBeVisible({ timeout: 10_000 });
  await expect(a.page.locator('.ch-perm-text')).toContainText('Run: python3');
  await a.page.screenshot({ path: `${SHOTS}/p2-1-permission-desktop.png` });
  await b.page.screenshot({ path: `${SHOTS}/p2-2-permission-phone.png` });
  await b.page.locator('.ch-perm button', { hasText: 'Allow once' }).click();
  await expect(a.page.locator('.ch-perm')).toHaveCount(0, { timeout: 10_000 });
  await expect.poll(async () => (await snap(request, id)).run, { timeout: 180_000 }).toBeNull();
  const s = await snap(request, id);
  const call = s.messages[1].parts.find((p) => p.type === 'tool_call')!;
  expect(call.status).toBe('done');
  expect(call.output?.content).toContain('42');
  expect(s.messages[1].parts.filter((p) => p.type === 'text').map((p) => p.text).join('')).toContain('42');
  await a.page.waitForTimeout(600);
  await a.page.screenshot({ path: `${SHOTS}/p2-3-tool-done-desktop.png` });
  await stopSandbox(request, id);
  await a.ctx.close();
  await b.ctx.close();
});

test('the terminal pane: a shell in the sandbox, the phone sees the same screen', async ({ browser, request }) => {
  const id = await newChat(request);
  const a = await open(browser, `#/chat/${id}`);
  await openPane(a.page, 'Terminal');
  await expect(a.page.locator('.ch-term-screen .xterm-rows')).toContainText('sandbox', { timeout: 60_000 });
  await a.page.locator('.ch-term-screen').click();
  await a.page.keyboard.type('echo $((6*7)) && uname -m');
  await a.page.keyboard.press('Enter');
  await expect(a.page.locator('.ch-term-screen .xterm-rows')).toContainText('42', { timeout: 15_000 });
  await a.page.screenshot({ path: `${SHOTS}/p2-4-terminal-desktop.png` });
  const b = await open(browser, `#/chat/${id}`, PHONE);
  await b.page.locator('button[aria-label="Panes"]').click();
  await expect(b.page.locator('.ch-term-screen .xterm-rows')).toContainText('42', { timeout: 15_000 });
  await b.page.screenshot({ path: `${SHOTS}/p2-5-terminal-phone.png` });
  await stopSandbox(request, id);
  await a.ctx.close();
  await b.ctx.close();
});

test('the preview pane: a workspace page in a sandboxed iframe that cannot read the cookie', async ({ browser, request }) => {
  const id = await newChat(request);
  const page = `<!doctype html><html><body style="font-family:sans-serif;background:#f6f1e7;padding:24px">
<h1>Preview from the sandbox</h1><p id="c"></p><script>
let r; try { r = 'cookie: ' + JSON.stringify(document.cookie); } catch (e) { r = 'cookie blocked: ' + e.name; }
document.getElementById('c').textContent = r;</script></body></html>`;
  const put = await request.put(`../v1/chat-app/conversations/${id}/sandbox/file?path=/workspace/index.html`, { data: page, headers: { 'Content-Type': 'text/plain' } });
  expect(put.status()).toBe(204);
  const a = await open(browser, `#/chat/${id}`);
  await a.page.context().addCookies([{ name: 'qse_probe', value: 'secret', url: a.page.url() }]);
  await openPane(a.page, 'Preview');
  const frame = a.page.frameLocator('.ch-preview iframe');
  await expect(frame.locator('h1')).toHaveText('Preview from the sandbox', { timeout: 15_000 });
  await expect(frame.locator('#c')).toContainText('cookie blocked: SecurityError');
  const sandboxAttr = await a.page.locator('.ch-preview iframe').getAttribute('sandbox');
  expect(sandboxAttr).not.toContain('allow-same-origin');
  await a.page.screenshot({ path: `${SHOTS}/p2-6-preview.png` });
  await stopSandbox(request, id);
  await a.ctx.close();
});

test('the agent writes a file: the files pane lists it, the diff pane shows the change', async ({ browser, request }) => {
  const id = await newChat(request);
  const a = await open(browser, `#/chat/${id}`);
  await send(a.page, 'Use the write_file tool to create notes/today.md with two lines: "# Today" and "- try the sandbox". Then say done.');
  await expect.poll(async () => (await snap(request, id)).run, { timeout: 180_000 }).toBeNull();
  const s = await snap(request, id);
  expect(s.messages[1].parts.some((p) => p.type === 'tool_call' && p.name === 'write_file' && p.status === 'done')).toBe(true);
  await openPane(a.page, 'Files');
  await expect(a.page.locator('.ch-files')).toContainText('notes', { timeout: 10_000 });
  await a.page.locator('.ch-file', { hasText: 'notes' }).click();
  await a.page.locator('.ch-file', { hasText: 'today.md' }).click();
  await expect(a.page.locator('.ch-file-view')).toContainText('try the sandbox');
  await a.page.screenshot({ path: `${SHOTS}/p2-7-files.png` });
  await openPane(a.page, 'Diff');
  await expect(a.page.locator('.ch-diff')).toContainText('try the sandbox', { timeout: 10_000 });
  await a.page.screenshot({ path: `${SHOTS}/p2-8-diff.png` });
  await stopSandbox(request, id);
  await a.ctx.close();
});
