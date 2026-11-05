<!-- Title: `KEY: imperative summary`, at most 72 characters. KEY is the ticket key, or `#123` for work that starts from a GitHub issue. -->

Ticket: <!-- ENG-123, or #123 -->

## What and why

<!-- What changed, and the problem it fixes. Link the issue. -->

## Tests

<!-- The commands you ran and what they printed. -->

- [ ] `python tests/script_suite.py --all`, or the test files this change touches
- [ ] `npm ci && npm test` in `dashboard/` (only for a change there)
- [ ] The engine writes the same text as before this change ([docs/exactness.md](https://github.com/0xBakeer/TandemLLM/blob/main/docs/exactness.md)), or the change says why it may not
- [ ] The gate report for this head is below, or `python -m tools.gate map` says level `none` ([docs/quality-gate.md](https://github.com/0xBakeer/TandemLLM/blob/main/docs/quality-gate.md))
- [ ] I have read the Contributor License Agreement in CONTRIBUTING.md and I agree to it.

## Quality gate

<!-- Run `python -m tools.gate run` on your GPU and paste results/gate/<head>-<time>/gate-report.md here, whole. For a WORSE or CHANGED_TEXT row, add its reason first: `python -m tools.gate explain <report.json> <row> "why"`. Level none: write "level none". -->
