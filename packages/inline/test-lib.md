# Testing a library
1. Run the suite once before any change, with its summary line in the result (pytest -q, npm test, go test ./...). Write the counts down.
2. For every new function: one test for the normal case, one for the edge (empty, zero, max), one for the error. Write the test before the fix when fixing a bug, and watch it fail first.
3. Run the whole suite after the last change, in one call, with the summary visible. Never run a subset and call it green.
4. Run the public example from the README on real input.
5. Counts: the failed count must be 0 and the passed count at or above the start. Fewer tests than before is a regression.
6. Let Laya read the suite's tail as a second check: `pytest -q 2>&1 | tail -n 30 | python3 {tandem_scratch}/laya.py "Did every test pass?" passed,failed`.
Report (the evidence you keep; the final message says it in plain words): the summary lines before and after, Laya's verdict, the tests added, the example run.