// The preview pane on a real sandbox (SRV-511, SRV-512): a Vite dev server, `vite preview` and a plain
// static server, each on the port it bound, and a page that needs no server, in the dashboard's
// preview iframe (opaque origin), with their modules, styles, run-time fetches and images, and the
// dev server's hot reload. The chats are made through the API (files, the dev server's Start), so
// no model is needed: it runs against the throwaway stack or a live box behind its https name.
//   python3 tests/chat_e2e_stack.py --front 127.0.0.1:4188 --dashboard dashboard/dist &
//   PW_LIVE=1 PW_BASE=http://127.0.0.1:4188/dashboard/ npx playwright test -c pw-chat-preview.config.ts
//   PW_LIVE=1 PW_BASE=https://tandem.khaled.de/dashboard/ npx playwright test -c pw-chat-preview.config.ts
// Each test makes its own chat and deletes it with its sandbox at the end. The sandboxes need the
// internet (npm install vite); WebKit runs in the Playwright image (mcr.microsoft.com/playwright).
import { expect, test, type APIRequestContext, type FrameLocator, type Page } from '@playwright/test';

test.skip(!process.env.PW_LIVE, 'needs a chatd with the sandbox image (PW_LIVE=1)');

const API = '../v1/chat-app';
const SHOTS = process.env.PW_SHOTS;

/** a Vite project: a module that imports a package by name, a style sheet with url(/...), a fetch
 *  of /data.json and an image at /logo.svg (from public/), and msg.js for the hot reload */
function viteFiles(dir: string, dev: string): Record<string, string> {
  return {
    [`${dir}/package.json`]: JSON.stringify({ name: dir, private: true, type: 'module', scripts: { dev }, devDependencies: { vite: '5.4.21', 'lil-gui': '0.19.2' } }),
    [`${dir}/index.html`]: '<!doctype html><html><head><meta charset="utf-8"><title>e2e vite</title></head><body><h1 id="app">loading</h1><img id="logo" alt="logo"><p id="data"></p><script type="module" src="/src/main.js"></script></body></html>',
    [`${dir}/src/main.js`]: "import GUI from 'lil-gui';\nimport './style.css';\nimport { msg } from './msg.js';\ndocument.getElementById('app').textContent = msg;\nnew GUI({ title: 'e2e gui' });\nconst img = document.getElementById('logo');\nimg.onload = () => (img.dataset.loaded = 'yes');\nimg.src = '/logo.svg';\nfetch('/data.json').then((r) => r.json()).then((d) => (document.getElementById('data').textContent = d.v));\n",
    [`${dir}/src/msg.js`]: "export const msg = 'vite app ok';\n",
    [`${dir}/src/style.css`]: 'body { background: rgb(1, 2, 3) url(/logo.svg) no-repeat; color: #eee; }\n',
    [`${dir}/public/data.json`]: '{"v": "data from /data.json"}',
    [`${dir}/public/logo.svg`]: '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"><rect width="10" height="10" fill="red"/></svg>',
  };
}

async function newChat(request: APIRequestContext, files: Record<string, string>): Promise<string> {
  const conv = (await (await request.post(`${API}/conversations`, { data: { title: 'e2e preview (SRV-512)' } })).json()) as { id: string };
  for (const [path, content] of Object.entries(files)) {
    const r = await request.put(`${API}/conversations/${conv.id}/sandbox/file?path=${encodeURIComponent(`/workspace/${path}`)}`, { data: content, headers: { 'Content-Type': 'text/plain' } });
    expect(r.ok(), `${path}: ${r.status()}`).toBeTruthy();
  }
  expect((await request.post(`${API}/conversations/${conv.id}/sandbox/start`)).ok()).toBeTruthy();
  return conv.id;
}

async function serve(request: APIRequestContext, id: string, dir: string): Promise<void> {
  const r = await request.post(`${API}/conversations/${id}/sandbox/serve`, { data: { dir: `/workspace/${dir}` } });
  expect(r.status(), await r.text()).toBe(202);
}

async function ports(request: APIRequestContext, id: string): Promise<number[]> {
  const r = (await (await request.get(`${API}/conversations/${id}/sandbox/ports`)).json()) as { ports: { port: number; http: boolean }[] };
  return r.ports.filter((p) => p.http).map((p) => p.port);
}

async function openChat(page: Page, id: string): Promise<void> {
  await page.goto(`./#/chat/${id}`);
  await page.locator('qse-chat').waitFor();
}

async function plusPick(page: Page, word: string): Promise<void> {
  if (!(await page.locator('.ch-pane-tabs').isVisible().catch(() => false))) await page.getByRole('button', { name: 'Panes', exact: true }).click();
  await page.getByRole('button', { name: 'Open a pane' }).click();
  await page.locator('.ch-pane-menu [role=menuitem]', { hasText: word }).click();
}

/** the Vite app ran in the preview: its module, the package it imports, its style sheet, the
 *  run-time fetch and image */
async function viteRan(frame: FrameLocator, text = 'vite app ok'): Promise<void> {
  await expect(frame.locator('#app')).toHaveText(text, { timeout: 90_000 });
  await expect(frame.locator('.lil-gui .title').first()).toHaveText('e2e gui');
  await expect(frame.locator('#data')).toHaveText('data from /data.json');
  await expect(frame.locator('#logo')).toHaveAttribute('data-loaded', 'yes');
  expect(await frame.locator('body').evaluate((b) => getComputedStyle(b).backgroundColor)).toBe('rgb(1, 2, 3)');
}

async function shot(page: Page, name: string, info: { project: { name: string } }): Promise<void> {
  if (SHOTS) await page.screenshot({ path: `${SHOTS}/${name}-${info.project.name}.png` });
}

test('a Vite dev server opens in the preview by itself, runs, and reloads on a change', async ({ page, request }, info) => {
  const id = await newChat(request, viteFiles('devapp', 'vite --port 5173'));
  try {
    await openChat(page, id);
    // the panes are open (a terminal), then the dev server starts: its port opens in a Preview tab
    await plusPick(page, 'Terminal');
    await serve(request, id, 'devapp');
    await expect(page.locator('.ch-pane-tab.is-on')).toContainText('port 5173', { timeout: 120_000 });
    const frame = page.frameLocator('.ch-preview iframe');
    await viteRan(frame);
    await expect(page.locator('.ch-preview .err-ink')).toHaveCount(0);
    await shot(page, 'preview-vite-dev', info);
    // the hot-reload socket: a change reaches the open pane without a reload
    const r = await request.put(`${API}/conversations/${id}/sandbox/file?path=${encodeURIComponent('/workspace/devapp/src/msg.js')}`, { data: "export const msg = 'vite app changed';\n", headers: { 'Content-Type': 'text/plain' } });
    expect(r.ok()).toBeTruthy();
    await expect(frame.locator('#app')).toHaveText('vite app changed', { timeout: 30_000 });
    // the project's source index.html goes to its dev server
    await page.getByRole('button', { name: 'Close port 5173' }).click();
    await plusPick(page, 'Files');
    await page.locator('.ch-file', { hasText: 'devapp' }).click();
    await page.locator('.ch-file', { hasText: 'index.html' }).first().click();
    await page.locator('.ch-file-view').getByRole('button', { name: /Preview/ }).click();
    await expect(page.locator('.ch-preview-note')).toContainText('index.html is the source of a Vite project');
    await viteRan(page.frameLocator('.ch-preview iframe'), 'vite app changed');
  } finally {
    // the page first: a tab on a chat that goes away starts a new one
    await page.close();
    await request.delete(`${API}/conversations/${id}?forever=1`);
  }
});

test('vite preview: the page waits, Start runs the dev script, the built app opens', async ({ page, request }, info) => {
  const id = await newChat(request, viteFiles('prodapp', 'vite build && vite preview --port 4173'));
  try {
    await openChat(page, id);
    await plusPick(page, 'Files');
    await page.locator('.ch-file', { hasText: 'prodapp' }).click();
    await page.locator('.ch-file', { hasText: 'index.html' }).first().click();
    await page.locator('.ch-file-view').getByRole('button', { name: /Preview/ }).click();
    // nothing listens: the pane says why and offers the project's dev script
    await expect(page.locator('.ch-preview-wait')).toContainText('it runs only from its dev server');
    await shot(page, 'preview-waiting', info);
    await page.locator('.ch-preview-wait').getByRole('button', { name: 'Start npm run dev' }).click();
    // vite build, then vite preview on 4173: the waiting pane opens it (absolute /assets/..., crossorigin)
    await expect(page.locator('.ch-preview-note')).toContainText('port 4173', { timeout: 150_000 });
    await viteRan(page.frameLocator('.ch-preview iframe'));
    expect(await ports(request, id)).toContain(4173);
    await shot(page, 'preview-vite-preview', info);
  } finally {
    // the page first: a tab on a chat that goes away starts a new one
    await page.close();
    await request.delete(`${API}/conversations/${id}?forever=1`);
  }
});

test('a plain static server and a page that needs no server', async ({ page, request }, info) => {
  const id = await newChat(request, {
    'site/package.json': JSON.stringify({ scripts: { dev: 'python3 -m http.server 8000' } }),
    'site/index.html': '<!doctype html><html><head><title>static</title><link rel="stylesheet" href="/s.css"></head><body><h1 id="t">static server</h1><script type="module" src="/m.js"></script></body></html>',
    'site/s.css': 'h1 { color: rgb(0, 128, 0); }',
    'site/m.js': "import { t } from './t.js'; document.getElementById('t').dataset.ran = t;",
    'site/t.js': "export const t = 'module ran';",
    'page/page.html': '<!doctype html><html><head><title>page</title></head><body><h1 id="p">plain page</h1><script type="module" src="p.js"></script></body></html>',
    'page/p.js': "document.getElementById('p').dataset.ran = 'yes';",
  });
  try {
    await openChat(page, id);
    await serve(request, id, 'site');
    await expect.poll(() => ports(request, id), { timeout: 60_000 }).toContain(8000);
    // the plus menu's Preview opens the running server
    await plusPick(page, 'Preview');
    const frame = page.frameLocator('.ch-preview iframe');
    await expect(frame.locator('#t')).toHaveAttribute('data-ran', 'module ran', { timeout: 30_000 });
    expect(await frame.locator('#t').evaluate((h) => getComputedStyle(h).color)).toBe('rgb(0, 128, 0)');
    await shot(page, 'preview-static-server', info);
    // a page that needs no server: from the view route, its module next to it
    await plusPick(page, 'Files');
    await page.locator('.ch-file', { hasText: 'page' }).first().click();
    await page.locator('.ch-file', { hasText: 'page.html' }).click();
    await page.locator('.ch-file-view').getByRole('button', { name: /Preview/ }).click();
    await expect(page.frameLocator('.ch-preview iframe').locator('#p')).toHaveAttribute('data-ran', 'yes', { timeout: 30_000 });
    await expect(page.locator('.ch-preview-note')).toHaveCount(0);
    await shot(page, 'preview-static-file', info);
  } finally {
    // the page first: a tab on a chat that goes away starts a new one
    await page.close();
    await request.delete(`${API}/conversations/${id}?forever=1`);
  }
});
