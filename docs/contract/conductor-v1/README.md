# Conductor API contract v1

These files are the truth for `/v1/conductor/*` (server/conductor/routes.py). The server's tests validate real responses against them (`tests/test_conductor_e2e.py`), and the dashboard's mock engine validates its fixtures and every mock response (`dashboard/tests/conductor-contract.test.ts`). They live beside `dashboard-v1` rather than inside it so the two contracts can change on their own.

| file | endpoint |
|-|-|
| `config.schema.json` | `GET /v1/conductor/config`: loaded versions, load errors, the journal |
| `meta.schema.json` | `GET /v1/conductor/meta`: enums, the task catalogue with units, the schemas |
| `profiles.schema.json` | `GET /v1/conductor/profiles`: the tiles |
| `document.schema.json` | `GET /v1/conductor/{settings,router,roles,packages}`, `GET /v1/conductor/profiles/{name}` |
| `write.schema.json` | `PUT` of a document, `POST .../restore`, `POST /v1/conductor/profiles/{name}/mode` |
| `versions.schema.json` | `GET .../versions` |
| `diff.schema.json` | `GET .../diff?from=&to=` |
| `roles.schema.json` | `GET /v1/conductor/roles`: the document and each role's health |
| `role-test.schema.json` | `POST /v1/conductor/roles/{role}/test` |
| `classify.schema.json` | `POST /v1/conductor/classify`: the test box |
| `decisions.schema.json`, `decision.schema.json` | `GET /v1/conductor/decisions`, `GET /v1/conductor/decisions/{id}`; `$defs` holds one router decision (with `events`, the monitor's decisions in the same request) and one monitor event |
| `events.schema.json`, `event.schema.json` | `GET /v1/conductor/events?task=&mode=&acted=&session=`, `GET /v1/conductor/events/{id}`: the session monitor's decisions |
| `stats.schema.json` | `GET /v1/conductor/stats?hours=` |
| `sessions.schema.json`, `session.schema.json`, `export.schema.json` | the session list, one timeline, the JSON export |
| `evals.schema.json` | `GET /v1/conductor/evals`: reports in the config directory's `evals/` |
| `scan.schema.json` | `GET /v1/conductor/packages/scan` |
| `error.schema.json` | every 4xx body; a 422 adds `errors` with one line per violation and its JSON path |
| `<kind>.config.schema.json` | the documents themselves; written from server/conductor/schema.py by `tools/conductor_contract.py --write` |
| `seeds.json` | the built-in version 0 of every document and the meta; the mock engine serves it |

Rules the schemas cannot express, which both sides test:

- A write needs a one-line note (422 without one). `base_version` is optional; when it is sent and the document moved on since, the answer is 409. `raw` and `bench` answer 409 to every write.
- A `PUT` that fails validation answers 422 and leaves the current version as it was. `POST .../check` runs the same validation without writing and returns the diff against the live version.
- A profile whose Tier 1 task is set to `act` while that task's role is offline is refused with the reason.
- `stats.precision` counts, per profile and decision type that ran in shadow, the decisions a person labelled and how many were right; the dashboard allows "Promote to on" at precision 0.9 or more over 30 or more labelled decisions.

## Changes

- v1.0, first version.
- v1.0, SRV-113/115, additive: one journal for the router and the session monitor. The `monitor` document (the monitor's policy; its schema is the policy's own, a branch per rule, and version 0 is the server's `ops/supervisor.yaml`). `events` and `event`. A router row gains the journal's common fields (`kind`, `task`, `tier`, `decision`, `action`, `trigger`, `score`, `tier1_model`, `question_id`, `probabilities`, `tier1_ms`, `hidden_text_id`, `hidden_tokens`, `position`, ...) and `events`; `session` gains `events`; `sessions` gains `acts`, `gave_ups`, `shadow` and `interventions`; `stats.monitor` is the monitor's block (null when no monitor runs); the export lists every intervention of either half (`source` router or monitor) with its exact text and says `summary`, "N interventions in M turns". A label (`POST .../decisions/{id}/label`) takes any row's id. `config.journal` names the ledger's file.
