// VIS-19..22 — the Playground: streaming chat with thinking, roles and edits, parameters that
// send only what changed, presets, function calling, the readout, JSON and exports.
// Scenarios that assert on the mock's scripted answers are tagged @mock and skip on a real engine.
import { expect, test, type Page, type Request } from '@playwright/test';
import { IS_MOCK, login, noHorizontalOverflow, setMode } from './helpers';

type Body = Record<string, unknown> & { messages: { role: string; content: string; tool_calls?: unknown[]; tool_call_id?: string }[] };

async function open(page: Page, hash = '#/playground'): Promise<void> {
  // a fresh bench per test: the draft and the presets are cleared once per context
  await page.addInitScript(() => {
    if (!sessionStorage.getItem('pw-pg-cleared')) {
      for (const k of ['qse.playground.draft', 'qse.playground.presets']) localStorage.removeItem(k);
      sessionStorage.setItem('pw-pg-cleared', '1');
    }
  });
  await login(page, hash);
  if (hash.startsWith('#/playground')) await expect(page.locator('qse-playground')).toBeVisible();
}

const isChat = (r: Request) => r.url().includes('/v1/chat/completions') && r.method() === 'POST';

/** Type a message, send it, return the posted body; waits for the turn to finish unless told not to. */
async function send(page: Page, text: string, opts: { wait?: boolean } = {}): Promise<Body> {
  await closeSheet(page);
  const reqP = page.waitForRequest(isChat);
  await page.locator('#pg-composer').fill(text);
  await page.locator('.pg-send').click();
  const req = await reqP;
  if (opts.wait !== false) await expect(page.locator('.pg-msg.role-assistant').last().locator('.pg-msg-foot')).not.toContainText('streaming', { timeout: 30_000 });
  return JSON.parse(req.postData() ?? '{}') as Body;
}

/** On a phone the setup column is a bottom sheet: open it before touching a section. */
async function openSheet(page: Page): Promise<void> {
  const toggle = page.locator('.pg-setup-toggle');
  if ((await toggle.isVisible()) && (await page.locator('.pg.is-setup-open').count()) === 0) {
    await toggle.click();
    await expect(page.locator('.pg.is-setup-open')).toBeVisible();
    await page.waitForTimeout(250); // the sheet's slide-in
  }
}
async function closeSheet(page: Page): Promise<void> {
  if ((await page.locator('.pg.is-setup-open').count()) > 0) {
    await page.locator('.pg-setup-close').click();
    await expect(page.locator('.pg.is-setup-open')).toHaveCount(0);
    await page.waitForTimeout(250);
  }
}

async function openSection(page: Page, id: string): Promise<void> {
  await openSheet(page);
  const head = page.locator(`.pg-section[data-section="${id}"] .pg-section-head`);
  if ((await head.getAttribute('aria-expanded')) !== 'true') await head.click();
}

async function setParam(page: Page, key: string, value: string): Promise<void> {
  await openSection(page, 'params');
  await page.locator(`#pg-param-${key}`).fill(value);
}

test.describe('playground', () => {
  test.beforeEach(async ({ request }) => {
    if (IS_MOCK) await setMode(request, 'ok');
  });

  // ---- VIS-19 chat ----------------------------------------------------------------------------
  test('the fifth tab: #/playground renders, the tab is current, reload keeps it', async ({ page }) => {
    await open(page, '#/usage');
    await page.getByRole('link', { name: 'Playground' }).click();
    await expect(page.locator('qse-playground')).toBeVisible();
    await expect(page.locator('.rail-tab.is-on')).toHaveText(/Playground/);
    await page.reload();
    await expect(page.locator('qse-playground')).toBeVisible();
    expect(page.url()).toContain('#/playground');
  });

  test('send and stream with thinking: a thinking block, a Markdown answer with a code block, a readout line', async ({ page }) => {
    await open(page);
    const body = await send(page, 'Explain tokens per block. Show a code example.');
    expect(body.stream).toBe(true);
    expect(body.stream_options).toEqual({ include_usage: true });
    expect(body).not.toHaveProperty('chat_template_kwargs');
    const turn = page.locator('.pg-msg.role-assistant').last();
    await expect(turn.locator('.pg-think')).toBeVisible();
    await expect(turn.locator('.pg-think-body')).toContainText(/\w+/);
    await expect(turn.locator('.pg-md')).toContainText(/\w+/);
    await expect(turn.locator('.pg-md')).not.toContainText('<think>');
    if (IS_MOCK) await expect(turn.locator('.md-code')).toBeVisible();
    await expect(turn.locator('.pg-msg-foot')).toContainText(/\d+(\.\d+)? tok\/s/);
    await expect(turn.locator('.pg-msg-foot')).toContainText(/stop|length/);
  });

  test('reasoning formats: reasoning_content and both show the reasoning once and no tag in the answer', async ({ page }) => {
    await open(page);
    await openSection(page, 'params');
    await page.locator('#pg-param-reasoning_format').selectOption('reasoning_content');
    let body = await send(page, 'Say hi');
    expect(body.reasoning_format).toBe('reasoning_content');
    let turn = page.locator('.pg-msg.role-assistant').last();
    await expect(turn.locator('.pg-think-body')).toContainText(/\w+/);
    await expect(turn.locator('.pg-md')).not.toContainText('think');
    await page.locator('#pg-param-reasoning_format').selectOption('both');
    body = await send(page, 'Say hi again');
    expect(body.reasoning_format).toBe('both');
    turn = page.locator('.pg-msg.role-assistant').last();
    const reasoning = (await turn.locator('.pg-think-body').textContent()) ?? '';
    expect(reasoning.trim().length).toBeGreaterThan(0);
    // shown once: the body is not the reasoning twice
    expect(reasoning.indexOf(reasoning.slice(0, 20), 5)).toBe(-1);
    await expect(turn.locator('.pg-md')).not.toContainText('<think>');
  });

  test('stop: aborts the stream, keeps the partial text, marks the turn aborted @mock', async ({ page, request }) => {
    test.skip(!IS_MOCK, 'needs the mock slow switch');
    await setMode(request, 'slow');
    await open(page);
    await send(page, 'Tell me a long story.', { wait: false });
    await expect(page.locator('.pg-stop')).toBeVisible();
    const turn = page.locator('.pg-msg.role-assistant').last();
    // wait for text, not for the element: the answer's cursor placeholder exists before any token
    await expect(turn.locator('.pg-msg-body')).toContainText(/\S/, { timeout: 15_000 });
    await page.locator('.pg-stop').click();
    await expect(turn.locator('.pg-msg-foot')).toContainText('aborted');
    await expect(page.locator('.pg-send')).toBeVisible();
    const text = (await turn.locator('.pg-think-body, .pg-md').allTextContents()).join('');
    expect(text.trim().length).toBeGreaterThan(0);
    await setMode(request, 'ok');
  });

  test('regenerate replaces the last assistant turn and keeps the transcript before it', async ({ page }) => {
    await open(page);
    await send(page, 'Say hi');
    const before = await page.locator('.pg-msg.role-assistant').last().getAttribute('data-uid');
    const reqP = page.waitForRequest(isChat);
    await page.getByRole('button', { name: 'Regenerate' }).click();
    const body = JSON.parse((await reqP).postData() ?? '{}') as Body;
    expect(body.messages.map((m) => m.role)).toEqual(['user']);
    await expect(page.locator('.pg-msg.role-assistant').last().locator('.pg-msg-foot')).not.toContainText('streaming', { timeout: 30_000 });
    await expect(page.locator('.pg-msg.role-assistant')).toHaveCount(1);
    expect(await page.locator('.pg-msg.role-assistant').last().getAttribute('data-uid')).not.toBe(before);
    await expect(page.locator('.pg-msg.role-user')).toHaveCount(1);
  });

  test('edit and resend: everything after the edited user message is removed and a new answer streams', async ({ page }) => {
    await open(page);
    await send(page, 'Say hi');
    await send(page, 'And again');
    await expect(page.locator('.pg-msg')).toHaveCount(4);
    const first = page.locator('.pg-msg.role-user').first();
    await first.hover();
    await first.getByRole('button', { name: 'Edit message' }).click();
    await first.locator('textarea').fill('Say hi, edited');
    const reqP = page.waitForRequest(isChat);
    await first.getByRole('button', { name: 'Save & resend' }).click();
    const body = JSON.parse((await reqP).postData() ?? '{}') as Body;
    expect(body.messages).toEqual([{ role: 'user', content: 'Say hi, edited' }]);
    await expect(page.locator('.pg-msg.role-assistant').last().locator('.pg-msg-foot')).not.toContainText('streaming', { timeout: 30_000 });
    await expect(page.locator('.pg-msg')).toHaveCount(2);
    await expect(page.locator('.pg-msg.role-user .pg-text')).toHaveText('Say hi, edited');
  });

  test('roles: an added message can be given the system role and shows so in the raw request', async ({ page }) => {
    await open(page);
    await page.getByLabel('Add a message with a role').selectOption('assistant');
    const msg = page.locator('.pg-msg.is-editing');
    await expect(msg).toBeVisible();
    await msg.locator('textarea').fill('Answer only in haiku.');
    await msg.locator('select').selectOption('system');
    await msg.getByRole('button', { name: 'Save', exact: true }).click();
    await expect(page.locator('.pg-msg.role-system')).toHaveCount(1);
    await closeSheet(page);
    await page.getByRole('button', { name: 'Raw request' }).click();
    const raw = JSON.parse(await page.getByTestId('raw-request').innerText()) as Body;
    expect(raw.messages).toEqual([{ role: 'system', content: 'Answer only in haiku.' }]);
  });

  test('raw request equals the body that is posted', async ({ page }) => {
    await open(page);
    await setParam(page, 'temperature', '0.7');
    await closeSheet(page);
    await page.locator('#pg-composer').fill('Compare the two.');
    await page.getByRole('button', { name: 'Raw request' }).click();
    const shown = JSON.parse(await page.getByTestId('raw-request').innerText());
    const reqP = page.waitForRequest(isChat);
    await page.locator('.pg-send').click();
    const posted = JSON.parse((await reqP).postData() ?? '{}');
    expect(posted).toEqual(shown);
    expect(shown.temperature).toBe(0.7);
  });

  test('busy engine: the 503 message and its Retry-After are shown in the transcript @mock', async ({ page, request }) => {
    test.skip(!IS_MOCK, 'needs the mock busy switch');
    await setMode(request, 'busy');
    await open(page);
    await send(page, 'Say hi');
    const err = page.locator('.pg-msg.role-assistant .pg-msg-error');
    await expect(err).toContainText('503');
    await expect(err).toContainText('retry after 5 s');
    await expect(page.locator('.pg-msg.role-assistant .pg-msg-foot')).toContainText('error');
    await setMode(request, 'ok');
  });

  test('phone: no horizontal overflow, the Setup button opens the sheet and the close button shuts it', async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await open(page);
    await send(page, 'Say hi');
    expect(await noHorizontalOverflow(page)).toBe(true);
    await page.locator('.pg-setup-toggle').click();
    await expect(page.locator('.pg.is-setup-open')).toBeVisible();
    await expect(page.locator('#pg-system')).toBeVisible();
    await page.locator('.pg-setup-close').click();
    await expect(page.locator('.pg.is-setup-open')).toHaveCount(0);
    // the composer stays in view above the tab bar
    const box = await page.locator('#pg-composer').boundingBox();
    expect(box).not.toBeNull();
    expect(box!.y + box!.height).toBeLessThan(844 - 60);
  });

  // ---- VIS-20 parameters and presets --------------------------------------------------------
  test('defaults from the engine are shown next to each field and nothing is marked changed', async ({ page }) => {
    await open(page);
    await openSection(page, 'params');
    await expect(page.locator('#pg-param-temperature')).toHaveAttribute('placeholder', /^0(\.0)?$/);
    await expect(page.locator('#pg-param-top_p')).toHaveAttribute('placeholder', /^1(\.0)?$/);
    await expect(page.locator('#pg-param-max_tokens')).toHaveAttribute('placeholder', /^[\d,]+$/);
    await expect(page.locator('#pg-param-reasoning_effort option').first()).toContainText(/default \(.+\)/);
    await expect(page.locator('.pg-param.is-changed')).toHaveCount(0);
    await expect(page.locator('.pg-section[data-section="params"] .pg-section-badge')).toHaveText('defaults');
  });

  test('only changed parameters are sent; no tools, no tool_choice', async ({ page }) => {
    await open(page);
    await setParam(page, 'temperature', '0.7');
    await setParam(page, 'max_tokens', '128');
    await expect(page.locator('.pg-param.is-changed')).toHaveCount(2);
    const body = await send(page, 'Say hi');
    expect(body.temperature).toBe(0.7);
    expect(body.max_tokens).toBe(128);
    for (const k of ['top_p', 'top_k', 'seed', 'presence_penalty', 'frequency_penalty', 'repetition_penalty', 'no_repeat_ngram_size', 'reasoning_effort', 'reasoning_format', 'draft_temperature', 'max_reasoning_tokens', 'tools', 'tool_choice', 'chat_template_kwargs', 'stop']) expect(body, k).not.toHaveProperty(k);
  });

  test('reset returns a field to the default and the next request has no such field', async ({ page }) => {
    await open(page);
    await setParam(page, 'temperature', '0.7');
    await page.locator('[data-param="temperature"] .pg-reset').click();
    await expect(page.locator('#pg-param-temperature')).toHaveValue('');
    await expect(page.locator('[data-param="temperature"]')).not.toHaveClass(/is-changed/);
    const body = await send(page, 'Say hi');
    expect(body).not.toHaveProperty('temperature');
  });

  test('thinking switch: off sends enable_thinking false, on again sends true', async ({ page }) => {
    await open(page);
    await openSection(page, 'params');
    await page.locator('#pg-param-thinking').uncheck();
    let body = await send(page, 'Say hi');
    expect(body.chat_template_kwargs).toEqual({ enable_thinking: false });
    await expect(page.locator('.pg-msg.role-assistant').last().locator('.pg-think')).toHaveCount(0);
    await openSection(page, 'params'); // the phone's sheet closed for the send
    await page.locator('#pg-param-thinking').check();
    body = await send(page, 'Say hi');
    expect(body.chat_template_kwargs).toEqual({ enable_thinking: true });
  });

  test('stop sequences are chips and go out as an array', async ({ page }) => {
    await open(page);
    await openSection(page, 'params');
    const input = page.locator('#pg-param-stop');
    await input.fill('###');
    await input.press('Enter');
    await input.fill('END');
    await input.press('Enter');
    await expect(page.locator('[data-param="stop"] .pg-chip')).toHaveCount(2);
    const body = await send(page, 'Say hi');
    expect(body.stop).toEqual(['###', 'END']);
  });

  test('an invalid value marks the field, disables Send and names the range', async ({ page }) => {
    await open(page);
    await setParam(page, 'temperature', '3');
    await expect(page.locator('[data-param="temperature"] .pg-param-error')).toContainText('at most 2');
    await closeSheet(page);
    await page.locator('#pg-composer').fill('x');
    await expect(page.locator('.pg-send')).toBeDisabled();
    await expect(page.locator('.pg-blocked')).toContainText('at most 2');
    await setParam(page, 'temperature', '0.5');
    await closeSheet(page);
    await expect(page.locator('.pg-send')).toBeEnabled();
  });

  test('preset round trip: save, reload, load — system prompt, temperature and tool are back', async ({ page }) => {
    await open(page);
    await openSection(page, 'system');
    await page.locator('#pg-system').fill('Be terse.');
    await setParam(page, 'temperature', '0.7');
    await openSection(page, 'tools');
    await page.getByRole('button', { name: 'get_weather(city, unit?)' }).click();
    await openSection(page, 'presets');
    await page.locator('#pg-preset-name').fill('terse');
    await page.getByRole('button', { name: 'Save preset' }).last().click();
    await expect(page.locator('.pg-notice')).toContainText('Saved preset "terse"');
    await page.reload();
    await expect(page.locator('qse-playground')).toBeVisible();
    await page.getByRole('button', { name: 'New chat' }).click();
    await openSection(page, 'system');
    await page.locator('#pg-system').fill('');
    await openSection(page, 'params');
    await page.locator('[data-param="temperature"] .pg-reset').click();
    await expect(page.locator('#pg-param-temperature')).toHaveValue('');
    if (await page.locator('.pg-setup-toggle').isVisible()) {
      await openSection(page, 'presets');
      await page.locator('.pg-preset-load', { hasText: 'terse' }).click();
    } else await page.getByLabel('Load a preset').selectOption('terse');
    await openSection(page, 'system');
    await expect(page.locator('#pg-system')).toHaveValue('Be terse.');
    await expect(page.locator('#pg-param-temperature')).toHaveValue('0.7');
    await openSection(page, 'tools');
    await expect(page.locator('.pg-tool-list')).toContainText('get_weather');
  });

  test('import refused: a file that is not a preset array changes nothing and says why', async ({ page }) => {
    await open(page);
    await openSection(page, 'presets');
    await page.locator('.pg-file input').setInputFiles({ name: 'x.json', mimeType: 'application/json', buffer: Buffer.from('{"hello": 1}') });
    await expect(page.locator('.pg-notice-warn')).toContainText('Import refused');
    await expect(page.locator('.pg-preset-list')).toHaveCount(0);
  });

  test('storage blocked: saving a preset still works for the page and says storage is blocked', async ({ page }) => {
    await page.addInitScript(() => {
      const orig = Storage.prototype.setItem;
      Storage.prototype.setItem = function (k: string, v: string) {
        if (k.startsWith('qse.playground')) throw new Error('QuotaExceededError');
        return orig.call(this, k, v);
      };
    });
    await open(page);
    await openSection(page, 'presets');
    await page.locator('#pg-preset-name').fill('x');
    await page.getByRole('button', { name: 'Save preset' }).last().click();
    await expect(page.locator('.pg-notice-warn')).toContainText('storage is blocked');
    await expect(page.locator('.pg-preset-list')).toContainText('x');
    await expect(page.locator('.pg-section[data-section="presets"]')).toContainText('cannot be saved in this browser');
  });

  // ---- VIS-21 tools -------------------------------------------------------------------------
  test('define a tool from a template: a valid array with one function, listed by name', async ({ page }) => {
    await open(page);
    await openSection(page, 'tools');
    await page.getByRole('button', { name: 'get_weather(city, unit?)' }).click();
    const text = await page.locator('#pg-tools').inputValue();
    const arr = JSON.parse(text);
    expect(arr).toHaveLength(1);
    expect(arr[0].function.name).toBe('get_weather');
    await expect(page.locator('.pg-tool-list li')).toHaveCount(1);
    await expect(page.locator('.pg-section[data-section="tools"] .pg-section-badge')).toContainText('1 defined');
    await expect(page.getByRole('button', { name: 'get_weather(city, unit?)' })).toBeDisabled();
  });

  test('validation: a bad name and broken JSON are named, Send is disabled meanwhile', async ({ page }) => {
    await open(page);
    await openSection(page, 'tools');
    await page.getByRole('button', { name: 'get_weather(city, unit?)' }).click();
    const editor = page.locator('#pg-tools');
    await editor.fill((await editor.inputValue()).replace('"get_weather"', '"get weather"'));
    await expect(page.locator('.pg-errors')).toContainText('tool 0 function.name');
    await closeSheet(page);
    await page.locator('#pg-composer').fill('x');
    await expect(page.locator('.pg-send')).toBeDisabled();
    await openSection(page, 'tools');
    await editor.fill('[{');
    await expect(page.locator('.pg-errors')).toContainText('not valid JSON');
    await editor.fill('');
    await expect(page.locator('.pg-errors')).toHaveCount(0);
    await closeSheet(page);
    await expect(page.locator('.pg-send')).toBeEnabled();
  });

  test('tool_choice shapes: auto is implicit; none, required and a named function are sent as such', async ({ page }) => {
    await open(page);
    await openSection(page, 'tools');
    await page.getByRole('button', { name: 'get_weather(city, unit?)' }).click();
    await closeSheet(page);
    await page.getByRole('button', { name: 'Raw request' }).click();
    const read = async () => {
      await closeSheet(page);
      return JSON.parse(await page.getByTestId('raw-request').innerText());
    };
    const choose = async (name: string) => {
      await openSection(page, 'tools');
      await page.getByRole('radio', { name }).click();
    };
    expect((await read()).tools).toHaveLength(1);
    expect(await read()).not.toHaveProperty('tool_choice');
    await choose('none');
    expect((await read()).tool_choice).toBe('none');
    expect((await read()).tools).toHaveLength(1);
    await choose('required');
    expect((await read()).tool_choice).toBe('required');
    await choose('named');
    expect((await read()).tool_choice).toEqual({ type: 'function', function: { name: 'get_weather' } });
  });

  test('a streamed tool call: a call card with arguments, finish tool_calls; a result continues the conversation @mock', async ({ page }) => {
    test.skip(!IS_MOCK, 'the real engine decides whether to call');
    await open(page);
    await openSection(page, 'tools');
    await page.getByRole('button', { name: 'get_weather(city, unit?)' }).click();
    const body = await send(page, 'What is the weather in Berlin?');
    expect(body.tools).toHaveLength(1);
    const turn = page.locator('.pg-msg.role-assistant').last();
    const card = turn.locator('.pg-call');
    await expect(card).toHaveCount(1);
    await expect(card.locator('.pg-call-name code')).toHaveText('get_weather');
    await expect(card.locator('.pg-call-args')).toContainText('"city": "Berlin"');
    await expect(turn.locator('.pg-msg-foot')).toContainText('tool_calls');
    // the template's example is offered and pre-filled
    await expect(card.locator('textarea')).toHaveValue(/temperature_c/);
    await card.getByRole('button', { name: 'template result' }).click();
    const reqP = page.waitForRequest(isChat);
    await page.getByRole('button', { name: /Send result/ }).click();
    const cont = JSON.parse((await reqP).postData() ?? '{}') as Body;
    const roles = cont.messages.map((m) => m.role);
    expect(roles).toEqual(['user', 'assistant', 'tool']);
    expect(cont.messages[1].tool_calls).toHaveLength(1);
    expect(cont.messages[2].tool_call_id).toBe((cont.messages[1].tool_calls![0] as { id: string }).id);
    expect(cont.messages[2].content).toContain('temperature_c');
    await expect(page.locator('.pg-msg.role-tool')).toHaveCount(1);
    const answer = page.locator('.pg-msg.role-assistant').last();
    await expect(answer.locator('.pg-msg-foot')).not.toContainText('streaming', { timeout: 30_000 });
    await expect(answer.locator('.pg-md')).toContainText('get_weather');
    await expect(answer.locator('.pg-md')).toContainText('21');
    // the result box of the answered call is gone; the raw request shows the continuation
    await expect(card.locator('textarea')).toHaveCount(0);
    await page.getByRole('button', { name: 'Raw request' }).click();
    const raw = JSON.parse(await page.getByTestId('raw-request').innerText()) as Body;
    expect(raw.messages.map((m) => m.role)).toEqual(['user', 'assistant', 'tool', 'assistant']);
    expect(raw.messages[1]).toHaveProperty('tool_calls');
    expect(raw.messages[2]).toHaveProperty('tool_call_id');
  });

  // ---- VIS-22 readout, JSON, exports ---------------------------------------------------------
  test('readout after a turn: every figure with its unit, taken from usage / timings / metrics', async ({ page }) => {
    await open(page);
    await send(page, 'Say hi');
    const readout = page.getByTestId('readout');
    for (const id of ['prompt', 'completion', 'reasoning']) await expect(readout.locator(`[data-fig="${id}"] .pg-fig-value`)).toContainText(/^[\d,]+tok$/);
    await expect(readout.locator('[data-fig="cached"] .pg-fig-value')).toContainText(/tok/);
    await expect(readout.locator('[data-fig="ttft"] .pg-fig-value')).toContainText(/\d (ms|s)$/);
    await expect(readout.locator('[data-fig="prefill"] .pg-fig-value')).toContainText(/tok\/s$/);
    await expect(readout.locator('[data-fig="decode"] .pg-fig-value')).toContainText(/^\d+\.\dtok\/s$/);
    await expect(readout.locator('[data-fig="tpb"] .pg-fig-value')).toContainText(/^\d+\.\d\dtok\/blk$/);
    await expect(readout.locator('[data-fig="acceptance"] .pg-fig-value')).toContainText(/(\d+ %|not reported)/);
    await expect(readout.locator('[data-fig="finish"] .pg-fig-value')).toHaveText(/stop|length/);
  });

  test('include_usage off: no stream_options is sent, the figures come from the finish chunk and the readout is still full', async ({ page }) => {
    await open(page);
    await openSection(page, 'params');
    await page.locator('#pg-param-include_usage').uncheck();
    const body = await send(page, 'Say hi');
    expect(body).not.toHaveProperty('stream_options');
    await expect(page.getByTestId('readout').locator('[data-fig="decode"] .pg-fig-value')).toContainText(/tok\/s/);
    await expect(page.getByTestId('readout').locator('[data-fig="prompt"] .pg-fig-value')).toContainText(/tok/);
  });

  test('JSON view: the request pane equals the posted body; the response pane holds the message, finish, usage, timings, metrics', async ({ page }) => {
    await open(page);
    const body = await send(page, 'Say hi');
    await page.getByRole('button', { name: 'JSON' }).click();
    expect(JSON.parse(await page.getByTestId('last-request').innerText())).toEqual(body);
    const res = JSON.parse(await page.getByTestId('last-response').innerText());
    expect(res.message.role).toBe('assistant');
    expect(typeof res.message.content).toBe('string');
    expect(res.finish_reason).toMatch(/stop|length/);
    expect(res.usage).toHaveProperty('completion_tokens');
    expect(res.timings).toHaveProperty('ttft_ms');
    expect(res.metrics).toHaveProperty('tokens_per_second');
  });

  test('export: JSON and Markdown files download with the messages and the readout lines', async ({ page }) => {
    await open(page);
    await send(page, 'Say hi');
    await page.getByRole('button', { name: 'Export' }).click();
    let dl = page.waitForEvent('download');
    await page.getByRole('menuitem', { name: 'as JSON' }).click();
    let d = await dl;
    expect(d.suggestedFilename()).toMatch(/^playground-\d{8}-\d{4}\.json$/);
    const json = JSON.parse(await streamText(d));
    expect(json.format).toBe('qse-playground-conversation');
    expect(json.messages.map((m: { role: string }) => m.role)).toEqual(['user', 'assistant']);
    expect(json.turns[0].timings).toHaveProperty('ttft_ms');
    await page.getByRole('button', { name: 'Export' }).click();
    dl = page.waitForEvent('download');
    await page.getByRole('menuitem', { name: 'as Markdown' }).click();
    d = await dl;
    expect(d.suggestedFilename()).toMatch(/\.md$/);
    const md = await streamText(d);
    expect(md).toContain('## user');
    expect(md).toContain('## assistant');
    expect(md).toMatch(/_.* tok\/s .*_/);
  });
});

async function streamText(d: import('@playwright/test').Download): Promise<string> {
  const s = await d.createReadStream();
  const chunks: Buffer[] = [];
  for await (const c of s) chunks.push(Buffer.from(c));
  return Buffer.concat(chunks).toString('utf8');
}
