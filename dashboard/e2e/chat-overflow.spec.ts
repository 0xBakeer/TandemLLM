// VIS-185, VIS-186: nothing in the Chat tab is wider than the screen. The thread scrolls up and
// down only: long words, URLs and inline code wrap, code blocks and tables scroll inside their own
// box, tool rows cut their argument with an ellipsis inside the card, the composer's pills stay in
// their row, the panes' tabs scroll inside their strip. Every chat of the mock, with every tool
// card and thinking block open, at 320, 390 and 430 px and on a desktop. Spark only:
//   npx playwright test e2e/chat-overflow.spec.ts --project=webkit-phone --project=chromium-desktop
// PW_CHAT_ID=<id> with PW_BASE adds a real conversation (read only: it opens the chat, nothing is sent).
import { expect, test, type Page } from '@playwright/test';
import { login } from './helpers';

const MOCK_CHATS = ['c_exl3', 'c_sql', 'c_regex', 'c_sync', 'c_rename', 'c_shop', 'c_explain', 'c_meal'];
const WIDTHS = [320, 390, 430];

/** what sticks out: elements past the chat's column or the screen, outside any box of their own
 *  that scrolls or clips; and every scroller that scrolls sideways but should not */
async function overflow(page: Page): Promise<string[]> {
  return page.evaluate(() => {
    const bad: string[] = [];
    const vw = document.documentElement.clientWidth;
    if (document.scrollingElement!.scrollWidth > vw + 1) bad.push(`page ${document.scrollingElement!.scrollWidth} > ${vw}`);
    const name = (el: Element) => `${el.tagName.toLowerCase()}.${[...el.classList].join('.')}`;
    const scroll = document.querySelector<HTMLElement>('.ch-scroll');
    if (scroll && scroll.scrollWidth > scroll.clientWidth + 1) bad.push(`.ch-scroll ${scroll.scrollWidth} > ${scroll.clientWidth}`);
    const col = document.querySelector('.ch-col')?.getBoundingClientRect();
    const main = document.querySelector('.ch-main')?.getBoundingClientRect();
    // the boxes a part may scroll or clip inside; the thread's own scroller and the layout around
    // it do not count (the thread clips sideways, so a part past the column would hide there)
    const LAYOUT = /^(ch-scroll|ch-threadwrap|ch-body|ch-main|ch|view)$/;
    for (const el of document.querySelectorAll('.ch-main *')) {
      if (el.closest('.ch-rail, .ch-ctx, .sr-only, qse-wren-figure, .ch-scrim')) continue;
      const r = el.getBoundingClientRect();
      const edge = Math.min(vw, el.closest('.ch-col') && col ? col.right : main ? main.right : vw);
      if (!r.width || (r.right <= edge + 1 && r.left >= -1)) continue;
      let boxed = false;
      for (let a = el.parentElement; a && a.tagName !== 'QSE-CHAT'; a = a.parentElement) {
        if ([...a.classList].some((c) => LAYOUT.test(c))) break;
        if (getComputedStyle(a).overflowX !== 'visible') {
          const ar = a.getBoundingClientRect();
          boxed = ar.right <= edge + 1 && ar.left >= -1;
          break;
        }
      }
      if (!boxed) bad.push(`${name(el)} ${Math.round(r.left)}..${Math.round(r.right)} past ${Math.round(edge)}`);
    }
    return bad.slice(0, 10);
  });
}

async function openAll(page: Page): Promise<void> {
  const heads = page.locator('.ch-call-head[aria-expanded="false"]');
  for (let i = await heads.count(); i > 0; i--) await heads.first().click();
  const thinks = page.locator('details.ch-think:not([open]) > summary');
  for (let i = await thinks.count(); i > 0; i--) await thinks.first().click();
}

async function openChat(page: Page, id: string, query: string): Promise<void> {
  await page.addInitScript(() => localStorage.removeItem('qse.chat.last'));
  await login(page, `#/chat/${id}`, query);
  await page.locator('.ch-col .ch-msg').first().waitFor();
  await page.waitForTimeout(300);
}

test.describe('the Chat tab never scrolls sideways', () => {
  for (const id of MOCK_CHATS) {
    test(`${id}: every width, every card open`, async ({ page }) => {
      const phone = (page.viewportSize()?.width ?? 1440) < 768;
      await openChat(page, id, '?chat=mock');
      await openAll(page);
      for (const w of phone ? WIDTHS : [1440]) {
        await page.setViewportSize({ width: w, height: phone ? 844 : 900 });
        await page.waitForTimeout(150);
        expect(await overflow(page), `${id} at ${w} px`).toEqual([]);
      }
    });
  }

  test('the figures line is one row, the composer pills stay in their row', async ({ page }) => {
    await openChat(page, 'c_sql', '?chat=mock');
    const figs = page.locator('.ch-figs').first();
    const h = await figs.evaluate((el) => el.getBoundingClientRect().height);
    expect(h).toBeLessThanOrEqual(30);
    const fit = await page.locator('.ch-pills').evaluate((el) => {
      const r = el.getBoundingClientRect();
      return [...el.querySelectorAll('.ch-pill')].every((p) => {
        const q = p.getBoundingClientRect();
        return q.left >= r.left - 1 && q.right <= r.right + 1;
      });
    });
    expect(fit).toBe(true);
  });

  test('the panes: the tabs scroll in their strip, the open tab in view, close always reachable', async ({ page }) => {
    test.skip((page.viewportSize()?.width ?? 1440) >= 768, 'the phone sheet');
    await openChat(page, 'c_exl3', '?chat=mock');
    await page.getByRole('button', { name: 'Panes', exact: true }).click();
    for (const k of ['Files', 'Terminal', 'Diff', 'Context', 'Preview']) {
      await page.locator('qse-chat-panes').getByRole('button', { name: 'Open a pane' }).click();
      await page.locator('.ch-pane-menu [role="menuitem"]', { hasText: k }).first().click();
    }
    await page.waitForTimeout(300);
    expect(await overflow(page)).toEqual([]);
    const vw = page.viewportSize()!.width;
    const on = await page.locator('.ch-pane-tab.is-on').boundingBox();
    const strip = await page.locator('.ch-pane-tabs').boundingBox();
    expect(on!.x).toBeGreaterThanOrEqual(strip!.x - 1);
    expect(on!.x + on!.width).toBeLessThanOrEqual(strip!.x + strip!.width + 1);
    const close = await page.getByRole('button', { name: 'Close panes' }).boundingBox();
    expect(close!.x + close!.width).toBeLessThanOrEqual(vw);
    // the sheet has the screen's height: the shell's bar steps down while it is open
    await expect(page.locator('.rail')).not.toBeInViewport();
  });

  test('a real conversation (PW_CHAT_ID)', async ({ page }) => {
    const id = process.env.PW_CHAT_ID;
    test.skip(!id || !process.env.PW_BASE, 'needs PW_BASE and PW_CHAT_ID');
    const phone = (page.viewportSize()?.width ?? 1440) < 768;
    await openChat(page, id!, '');
    await openAll(page);
    for (const w of phone ? WIDTHS : [1440]) {
      await page.setViewportSize({ width: w, height: phone ? 844 : 900 });
      await page.waitForTimeout(150);
      expect(await overflow(page), `${id} at ${w} px`).toEqual([]);
    }
  });
});
