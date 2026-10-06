# Testing with TandemLLM

Test with evidence, never with intentions: exit codes, test summaries, console messages, screenshots, state dumps. The runtime reads them, and Laya, its decision model, judges them for you when your test script asks it (the laya-testing package). Give them facts.

Make the software observable before you test it:
- Scratch: logs, helper scripts and kits go to {tandem_scratch}/ (`mkdir -p {tandem_scratch}` first), outside the project. The project gets only what the user asked for; never list scratch files as deliverables.
- Logs: write build and run output to {tandem_scratch}/build.log and {tandem_scratch}/run.log. After a run, `tail -n 60` the log in a tool call.
- State dump: expose one function or flag that returns a JSON object with the key "tandem_state": 1 plus the few numbers that define the state (positions, counts, health, score, rows, status). Keep it under 4 KB. Read it after real input, never before. It is test scaffolding: wrap it in `tandem-test-hook:start` and `tandem-test-hook:end` comments and remove it as your last edit before "done" (then run the quick check once more), unless the user asked for a debug or state API; say in your final message that you removed it.
- Tests: run the real test command and let its summary line appear in the tool result.
- Runtimes: use what is installed (`command -v node npx python3 bun deno`; for a page, the browser tool or `npx playwright`). Never download a runtime or any executable (no curl or wget of a binary, no install script) unless the user asks; if nothing can run the code, say so and test what you can.

Every edit is followed by a run that produces one of these. A change with no visible effect in the next run is a bug to explain, not a step done.

When a <runtime-note> gives a verdict (no change, regression, error, surprise, test faked), answer it first: what you expected, what the evidence shows, the one cause. Then act.

Never test by writing to the state you then read. Input in, state out.