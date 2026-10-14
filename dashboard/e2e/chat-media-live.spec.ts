// A video in the Chat tab (VIS-159, QA VIS-160), against a throwaway Chat tab: a 60 s fixture mp4 of
// about 45 MB uploaded to a chat's sandbox plays in the present card and in the Files pane
// (currentTime advances), and a seek past what the player buffered asks the view route for a byte
// range (206). WebKit gets an H.264 file, Chromium (no proprietary codecs) a VP9 one, both mp4.
// The stack is tests/chat_e2e_stack.py (the fake engine, a chatd on its own data and sandbox label,
// the front with this build); never a chatd that holds real chats. On a box, never a laptop:
//   python3 tests/chat_e2e_stack.py --front 127.0.0.1:4188 --dashboard dashboard/dist &
//   PW_LIVE=1 PW_MEDIA=<dir with clip-h264.mp4 and clip-vp9.mp4> \
//     PW_BASE=http://127.0.0.1:4188/dashboard/ npx playwright test -c pw-chat-panes.config.ts chat-media
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { expect, test, type Locator, type Page } from '@playwright/test';

test.skip(!process.env.PW_LIVE || !process.env.PW_MEDIA, 'needs tests/chat_e2e_stack.py and PW_MEDIA');

const API = '../v1/chat-app';

async function plays(v: Locator, from = 0): Promise<void> {
  await v.evaluate(async (el: HTMLVideoElement) => {
    el.muted = true;
    await el.play();
  });
  await expect.poll(() => v.evaluate((el: HTMLVideoElement) => el.currentTime), { timeout: 30_000 }).toBeGreaterThan(from + 0.5);
}

function rangeLog(page: Page, part: string): { range: string; status: number }[] {
  const log: { range: string; status: number }[] = [];
  page.on('response', (r) => {
    if (r.url().includes(part)) log.push({ range: r.request().headers()['range'] ?? '', status: r.status() });
  });
  return log;
}

test('a video over 2 MiB plays in the present card and the Files pane; a seek asks for a range', async ({ page, request }, info) => {
  const name = info.project.name === 'webkit' ? 'clip-h264.mp4' : 'clip-vp9.mp4';
  const bytes = readFileSync(join(process.env.PW_MEDIA!, name));
  expect((await request.patch(`${API}/defaults`, { data: { permission_mode: 'yolo' } })).ok()).toBeTruthy();
  const { id } = (await (await request.post(`${API}/conversations`, { data: {} })).json()) as { id: string };
  const up = await request.post(`${API}/conversations/${id}/sandbox/upload?dir=/workspace/clips`, { multipart: { file: { name, mimeType: 'video/mp4', buffer: bytes } } });
  expect(up.ok()).toBeTruthy();

  // the route itself, through the front's forward: whole, a range, past the end
  const view = `${API}/conversations/${id}/sandbox/view/clips/${name}`;
  const whole = await request.get(view);
  expect(whole.status()).toBe(200);
  expect(Number(whole.headers()['content-length'])).toBe(bytes.length);
  expect(whole.headers()['accept-ranges']).toBe('bytes');
  const part = await request.get(view, { headers: { Range: 'bytes=1000000-' } });
  expect(part.status()).toBe(206);
  expect(part.headers()['content-range']).toBe(`bytes 1000000-${bytes.length - 1}/${bytes.length}`);
  expect(Buffer.compare(await part.body(), bytes.subarray(1000000))).toBe(0);
  expect((await request.get(view, { headers: { Range: `bytes=${bytes.length}-` } })).status()).toBe(416);

  // the model presents it (as a preview: the file's own kind wins)
  const calls = [{ name: 'present', arguments: { kind: 'preview', path: `clips/${name}` } }];
  expect((await request.post(`${API}/conversations/${id}/messages`, { data: { client_msg_id: `e2e-${Date.now()}`, text: `FAKE_CALLS ${JSON.stringify(calls)}` } })).ok()).toBeTruthy();
  await expect
    .poll(async () => {
      const s = (await (await request.get(`${API}/conversations/${id}`)).json()) as { run: unknown; messages: { status: string }[] };
      return s.run === null && s.messages.length >= 2 && s.messages[s.messages.length - 1].status !== 'streaming';
    }, { timeout: 60_000 })
    .toBe(true);

  const log = rangeLog(page, `/sandbox/view/clips/${name}`);
  await page.setViewportSize({ width: 1280, height: 900 });
  await page.goto(`./#/chat/${id}`);
  await page.locator('qse-chat').waitFor();
  const card = page.locator('section.ch-call[data-tool="present"]');
  await expect(card).toBeVisible();
  const cardVideo = card.locator('qse-media-view video');
  await expect(cardVideo).toBeVisible({ timeout: 20_000 });
  await expect(cardVideo).toHaveAttribute('playsinline', '');
  await expect(cardVideo).toHaveAttribute('preload', 'metadata');
  await expect(card.getByRole('link', { name: /Download/ })).toHaveAttribute('download', name);
  await plays(cardVideo);
  const before = log.length;
  await cardVideo.evaluate((el: HTMLVideoElement) => {
    el.currentTime = 50;
  });
  await expect.poll(() => cardVideo.evaluate((el: HTMLVideoElement) => el.currentTime), { timeout: 30_000 }).toBeGreaterThan(50.3);
  await expect.poll(() => log.slice(before).some((r) => /^bytes=[1-9]\d*-/.test(r.range) && r.status === 206), { timeout: 15_000, message: JSON.stringify(log) }).toBe(true);
  // no error answer (WebKit reports a request its player dropped for a seek with status 0)
  expect(log.every((r) => r.status < 400), JSON.stringify(log)).toBe(true);
  console.log(`${info.project.name} view requests: ${JSON.stringify(log)}`);
  await cardVideo.evaluate((el: HTMLVideoElement) => el.pause());

  // the card's pane button: the video pane plays it
  await card.getByRole('button', { name: 'Open in Video' }).click();
  const pane = page.locator('.ch-mediapane qse-media-view video');
  await expect(pane).toBeVisible({ timeout: 20_000 });
  await plays(pane);
  await pane.evaluate((el: HTMLVideoElement) => el.pause());

  // the Files pane: the file opens in the player, not an empty viewer
  await page.getByRole('button', { name: 'Open a pane' }).click();
  await page.locator('.ch-pane-menu [role=menuitem]', { hasText: 'Files' }).click();
  await page.locator('.ch-file', { hasText: 'clips' }).first().click();
  await page.locator('.ch-file', { hasText: name }).click();
  const paneVideo = page.locator('.ch-file-view qse-media-view video');
  await expect(paneVideo).toBeVisible({ timeout: 20_000 });
  await plays(paneVideo);
  await expect(page.locator('.ch-file-view .ch-code')).toHaveCount(0);
  await page.screenshot({ path: test.info().outputPath(`media-${info.project.name}.png`) });
});
