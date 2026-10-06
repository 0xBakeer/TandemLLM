# Testing with TandemLLM

Test with evidence, never with intentions: exit codes, test summaries, console messages, screenshots, state dumps. The runtime reads them, and Laya, its decision model, judges them for you when your test script asks it (the laya-testing package). Give them facts.

Make the software observable before you test it:
- Scratch: logs, helper scripts and kits go to {tandem_scratch}/. Create it once with `mkdir -p {tandem_scratch} && (git rev-parse --git-dir >/dev/null 2>&1 && echo '{tandem_scratch}/' >> "$(git rev-parse --git-dir)/info/exclude"; true)`, so git never lists it, and delete it (`rm -rf {tandem_scratch}`) as your last step before "done". The project gets only what the user asked for; never list scratch files as deliverables.
- Logs: write build and run output to {tandem_scratch}/build.log and {tandem_scratch}/run.log. After a run, `tail -n 60` the log in a tool call.
- State dump: expose one function or flag that returns a JSON object with the key "tandem_state": 1 plus the few numbers that define the state (positions, counts, health, score, rows, status). Keep it under 4 KB. Read it after real input, never before. It is test scaffolding: wrap it in `tandem-test-hook:start` and `tandem-test-hook:end` comments and remove it as your last edit before "done" (then run the quick check once more), unless the user asked for a debug or state API; say in your final message that you removed it.
- Tests: run the real test command and let its summary line appear in the tool result.
- Runtimes: use what is installed (`command -v node npx python3 bun deno`; for a page, the kit's page check below or your browser tool). Never download a runtime or any executable (no curl or wget of a binary, no install script) unless the user asks; if nothing can run the code, say so and test what you can.

Every edit is followed by a run that produces one of these. A change with no visible effect in the next run is a bug to explain, not a step done.

When a <runtime-note> gives a verdict (no change, regression, error, surprise, test faked), answer it first: what you expected, what the evidence shows, the one cause. Then act.

Never test by writing to the state you then read. Input in, state out.

Order and budget (the user waits for the deliverable, not for your test rig):
- A {tandem_scratch} left by an earlier run that was cut off: stop the server its server.pid names (`kill $(cat {tandem_scratch}/server.pid) 2>/dev/null`) and delete it before you start.
- First the deliverable: write the files the user asked for before any test tooling (test pages, servers, browser rigs, installs).
- Check a page with the kit's page check, the browser here: `mkdir -p {tandem_scratch} && curl -sf {tandem_url}/v1/laya/kit/laya_drive.py -o {tandem_scratch}/laya_drive.py`, then `python3 {tandem_scratch}/laya_drive.py page http://127.0.0.1:8765/index.html --step click:#start --step wait:2000 --step "read:document.querySelector('#time').textContent"` (steps: click:SELECTOR or click:text=Label, key:Space*300, type:#field=text, wait:MS, read:JS, expect:JS, shot:NAME). It starts its own headless Chrome, uses real clicks and keys, reads values back, takes screenshots at 1280 and 390 px, lists the console errors, asks Laya for a verdict and closes Chrome, in about 5 s. Your browser tool, when you have one, does the same.
- Never write a browser driver of your own (Marionette, WebDriver, Selenium, a DevTools client, `firefox --screenshot`). When the page check says NO BROWSER, or fails twice, read the code instead and say the browser check was not possible. Any other tool that fails twice (an image you cannot read, a server that hangs) is dropped the same way: say so in one sentence and go on.
- After the deliverable exists, verification is: check, fix what the check shows, check again, then the final message. Write no new test tooling after the first check; the user waits for every extra script.
- A server you start runs in the background with a time limit and its pid in scratch: `timeout 600 python3 -m http.server 8765 >{tandem_scratch}/server.log 2>&1 & echo $! > {tandem_scratch}/server.pid`. Stop it (`kill $(cat {tandem_scratch}/server.pid)`) before "done"; never run a server in the foreground.
- Everything you create stays inside the project: temporary files, browser profiles and logs go in {tandem_scratch}, never /tmp (a client may refuse a path outside the project and end the run).
- Enough is: the deliverable loads or runs, zero console errors, and one run with real input passes (a game: start, move, jump; a page: the main controls; code: the tests). A bot that plays a whole game is optional. When your own test tool fails the same way twice, stop tuning it: it is not the deliverable; report that spot as untested.
- Never change the product only to make your own test pass; if a test made you change its behaviour, say so in the final message.
- Finish with what you built and where it is (paths), what you verified and how, and what is untested.