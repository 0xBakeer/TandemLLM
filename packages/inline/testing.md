# Testing with TandemLLM

Test with evidence, never with intentions: exit codes, test summaries, console messages, screenshots, state dumps. The runtime reads them, and Laya, its decision model, judges them for you when your test script asks it (the laya-testing package). Give them facts.

Make the software observable before you test it:
- Logs: write build and run output to ./.tandem/build.log and ./.tandem/run.log. After a run, `tail -n 60` the log in a tool call.
- State dump: expose one function or flag that returns a JSON object with the key "tandem_state": 1 plus the few numbers that define the state (positions, counts, health, score, rows, status). Keep it under 4 KB. Read it after real input, never before.
- Tests: run the real test command and let its summary line appear in the tool result.

Every edit is followed by a run that produces one of these. A change with no visible effect in the next run is a bug to explain, not a step done.

When a <runtime-note> gives a verdict (no change, regression, error, surprise, test faked), answer it first: what you expected, what the evidence shows, the one cause. Then act.

Never test by writing to the state you then read. Input in, state out.