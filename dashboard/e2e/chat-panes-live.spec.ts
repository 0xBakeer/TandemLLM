// The panes on a phone and on a desktop (VIS-133, VIS-135), against a throwaway Chat tab with real
// sandboxes: the plus menu opens each pane kind, the terminal runs a command in the sandbox, and an
// HTML file a tool call wrote opens in the preview from its tool card, the plus menu and Files.
// WebKit refuses some ports (4190 among them); the stack listens on 4188.
// The stack is tests/chat_e2e_stack.py (the fake engine, a chatd on its own data and sandbox label,
// the front with this build); never a chatd that holds real chats. On a box with the sandbox image:
//   python3 tests/chat_e2e_stack.py --front 127.0.0.1:4188 --dashboard dashboard/dist &
//   PW_LIVE=1 PW_BASE=http://127.0.0.1:4188/dashboard/ npx playwright test -c pw-chat-panes.config.ts
import { expect, test, type APIRequestContext, type Page } from '@playwright/test';

test.skip(!process.env.PW_LIVE, 'needs tests/chat_e2e_stack.py and the sandbox image (PW_LIVE=1)');

const API = '../v1/chat-app';
const PAGE = '<!doctype html><html><head><title>e2e page</title></head><body><h1 id="hello">Hello from the sandbox</h1><script>document.getElementById("hello").dataset.ran = "yes"</script></body></html>';

/** a chat whose one turn wrote site/page.html with write_file */
async function chatWithPage(request: APIRequestContext): Promise<string> {
  expect((await request.patch(`${API}/defaults`, { data: { permission_mode: 'yolo' } })).ok()).toBeTruthy();
  const conv = (await (await request.post(`${API}/conversations`, { data: {} })).json()) as { id: string };
  const calls = [{ name: 'write_file', arguments: { path: 'site/page.html', content: PAGE } }];
  const sent = await request.post(`${API}/conversations/${conv.id}/messages`, { data: { client_msg_id: `e2e-${Date.now()}`, text: `FAKE_CALLS ${JSON.stringify(calls)}` } });
  expect(sent.ok()).toBeTruthy();
  await expect
    .poll(async () => {
      const s = (await (await request.get(`${API}/conversations/${conv.id}`)).json()) as { run: unknown; messages: { status: string }[] };
      return s.run === null && s.messages.length >= 2 && s.messages[s.messages.length - 1].status !== 'streaming';
    }, { timeout: 60_000 })
    .toBe(true);
  return conv.id;
}

/** the element under the centre of `sel` is `sel` itself or inside it: nothing covers or clips it */
async function reachable(page: Page, sel: string): Promise<boolean> {
  return page.locator(sel).first().evaluate((el) => {
    const r = el.getBoundingClientRect();
    if (r.width < 8 || r.height < 8) return false;
    const hit = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
    return !!hit && (hit === el || el.contains(hit));
  });
}

async function plusPick(page: Page, word: string): Promise<void> {
  await page.getByRole('button', { name: 'Open a pane' }).click();
  const item = page.locator('.ch-pane-menu [role=menuitem]', { hasText: word });
  await expect(item).toBeVisible();
  expect(await reachable(page, `.ch-pane-menu [role=menuitem]:has-text("${word}")`)).toBe(true);
  await item.click();
}

for (const phone of [true, false]) {
  test(`${phone ? 'a phone' : 'a desktop'}: plus opens every pane, the terminal runs a command, the page previews`, async ({ browser, request }, info) => {
    const id = await chatWithPage(request);
    const ctx = await browser.newContext(phone ? { viewport: { width: 390, height: 844 }, hasTouch: true, isMobile: info.project.name !== 'webkit' ? true : undefined } : { viewport: { width: 1280, height: 860 } });
    const page = await ctx.newPage();
    await page.goto(`./#/chat/${id}`);
    await page.locator('qse-chat').waitFor();
    await expect(page.locator('section.ch-call[data-tool="write_file"]')).toBeVisible();

    // the tool card offers the file it wrote
    await page.locator('section.ch-call[data-tool="write_file"]').getByRole('button', { name: /Open in Preview/ }).click();
    const frame = page.frameLocator('.ch-preview iframe');
    await expect(frame.locator('#hello')).toHaveText('Hello from the sandbox', { timeout: 20_000 });
    await expect(frame.locator('#hello')).toHaveAttribute('data-ran', 'yes');
    // the last pane closed: the Panes button brings the strip back (on a phone the empty sheet the
    // report showed, "No pane open. Press + ...")
    await page.getByRole('button', { name: 'Close page.html' }).click();
    await page.getByRole('button', { name: 'Panes', exact: true }).click();
    await expect(page.getByRole('button', { name: 'Open a pane' })).toBeVisible();
    if (phone) await expect(page.locator('.ch-pane-empty')).toContainText('Press +');

    // the plus menu: every kind it lists can be reached and opened
    await page.getByRole('button', { name: 'Open a pane' }).click();
    const words = await page.locator('.ch-pane-menu [role=menuitem] .ch-pop-name').allTextContents();
    expect(words).toEqual(expect.arrayContaining(['Terminal', 'Files', 'Preview']));
    for (let i = 0; i < words.length; i++) expect(await reachable(page, `.ch-pane-menu [role=menuitem]:nth-child(${i + 1})`)).toBe(true);
    await page.getByRole('button', { name: 'Open a pane' }).click();
    await expect(page.locator('.ch-pane-menu')).toHaveCount(0);

    // Terminal: a shell in the chat's sandbox
    await plusPick(page, 'Terminal');
    await expect(page.locator('.ch-pane-body[data-kind="terminal"] .xterm')).toBeVisible({ timeout: 30_000 });
    await expect(page.locator('.ch-term-wait')).toHaveCount(0);
    await page.locator('.ch-term-screen').click();
    await page.keyboard.type('echo pane-$((6*7)) && pwd');
    await page.keyboard.press('Enter');
    await expect(page.locator('.ch-term-screen .xterm-rows')).toContainText('pane-42', { timeout: 20_000 });
    await expect(page.locator('.ch-term-screen .xterm-rows')).toContainText('/workspace');
    await page.screenshot({ path: info.outputPath('terminal.png') });

    // Files: the tree, and the HTML file's Preview button
    await plusPick(page, 'Files');
    await page.locator('.ch-files').getByRole('button', { name: /site/ }).click();
    await page.locator('.ch-files').getByRole('button', { name: /page\.html/ }).click();
    await page.locator('.ch-file-view').getByRole('button', { name: /Preview/ }).click();
    await expect(page.frameLocator('.ch-preview iframe').locator('#hello')).toHaveText('Hello from the sandbox', { timeout: 20_000 });

    // Preview from the plus menu: the newest HTML file in the sandbox
    await page.getByRole('button', { name: 'Close page.html' }).click();
    await expect(page.locator('.ch-pane-tab', { hasText: 'page.html' })).toHaveCount(0);
    await plusPick(page, 'Preview');
    await expect(page.frameLocator('.ch-preview iframe').locator('#hello')).toHaveText('Hello from the sandbox', { timeout: 20_000 });
    await page.screenshot({ path: info.outputPath('preview.png') });
    await ctx.close();
  });
}
