// Prints the gzip size of every file in dist/ and fails above the budget: 300 KB before the Chat
// tab, 400 KB with it (VIS-99: its own chunk of about 40 KB gzip plus the mock it loads on
// demand), 405 KB since the terminal pane runs xterm.js (VIS-105, 2026-10-12: an 81 KB chunk that
// loads only when a terminal opens) and the PWA's worker and notifications (VIS-109), 410 KB since
// 0.8.0's phone bar and composer (rc-mobile, 405.0 KB) and the Chat tab's Beta pill took it to the limit
// (2026-10-13, 405.1 KB): no feature cut, as decided for the release. 415 KB since the community
// weights section of the setup guide (2026-10-13, 412.3 KB: the cards, their search and sort, the
// GitHub sync state and six inline glyphs for the networks). Room looked for first: terser is
// larger than oxc here, top-level mangling is already on, no CSS rule is dead, and one combined
// CSS file would save 2.5 KB but load every view's CSS up front; the module-preload polyfill went.
// 420 KB since 0.9.0 (2026-10-14: 415.1 KB as released, 416.4 KB with the panes' plus menu, the
// pages a tool call wrote and the Context panel's reasons).
// 423 KB since the media players (VIS-159, a 1.4 KB chunk loaded on first use) and the Context
// panel's memory notes (SRV-471): 422.2 KB. 424 KB with the Setup Guide's job log and change list
// (VIS-151, VIS-155) on top: 423.7 KB; a lazy install view measured 424.0 KB, so no split.
// 425 KB with the sandbox's reason on a call and the attachment chip's tokens (VIS-167, VIS-169):
// 424.2 KB.
// 428 KB with a live run's reasons on the Now figures and the slot row's owner (VIS-181) and the
// Chat tab's panel for a box whose chatd is down, its re-probe and banner (VIS-179): 427.3 KB.
// 430 KB with the preview's target, its waiting state and Start button, the page's report and the
// new-server watch (SRV-511): 429.7 KB. 431 KB with the Chat tab's design pass on top (VIS-185: the
// figures line as readings, the panes' tab strip that scrolls on its own, the thread's wrap rules): 430.3 KB.
// 433 KB with the Setup Guide's parallel-chats stepper (VIS-199: the field, its reasons, the cost line,
// Next's refusal and the fit search): 432.4 KB. 434 KB with its two window steppers on top (the ladder,
// typed 128k and 1m, the limits named): 433.6 KB. 435 KB with the verify phase on top (SRV-532: the
// harness's check card, the Checked / Not checked block, the composer's switch): 434.2 KB.
// 445 KB with the Setup Guide's restart card, finished screen, Retry and diagnostics and its weights list
// on top (VIS-209, VIS-211, VIS-213, VIS-215, 2026-10-17: 442.5 KB).
// 456 KB with a main agent's workers (VIS-217, VIS-219, 2026-10-17: 455.1 KB from 442.7): the Task card,
// its report, chatd's check and the transcript sheet, the worker's part of the Context panel and the
// Agents switch add 5.9 KB to the chat chunk; the mock's scripted worker and its four seeded tasks
// 5.7 KB to the mock chunk, which loads only with ?chat=mock.
// 458 KB with each running worker's live figures (SRV-579, VIS-219, 2026-10-17: 457.3 KB from
// 455.1): its own Now in the Context panel, a row per worker in the Now dock, the rate on its card,
// and the mock's two workers at once.
// #130 (2026-10-23): the size is pako's gzip at level 6 (scripts/gzip-size.mjs), the same on every Node, OS and CPU.
// 462 KB since the guide learned a simple mode (#103, 2026-10-23: 460.6 KB on the pako measure): the mode's own
// chunk (the three-page flow, the folded settings page, the card tiles and their note-line trim) and the Finish
// gate's plan-check round trip.
import { BUDGET, distSizes } from './gzip-size.mjs';

const dist = new URL('../dist/', import.meta.url).pathname;
const { files, total } = distSizes(dist);
for (const f of files) console.log(`${(f.gzip / 1024).toFixed(1).padStart(7)} KB  ${f.path}`);
console.log(`${(total / 1024).toFixed(1).padStart(7)} KB  total gzip (budget ${BUDGET / 1024} KB, ${BUDGET - total} bytes left)`);
if (total > BUDGET) {
  console.error('over budget');
  process.exit(1);
}
