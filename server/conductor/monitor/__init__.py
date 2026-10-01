"""The session monitor: safe points, the fixed action set, the policy file, the journal.

The public interface, kept stable for the conductor's policy engine and UI (docs/server.md,
"The conductor's session monitor"):

    policy.MODES, policy.TASK_MODES, policy.ACTIONS, policy.SAFE_POINTS
    policy.Action                          what a rule returns
    policy.load(path) -> Policy            ops/supervisor.yaml, validated
    policy.schema() -> dict                the JSON Schema (ops/supervisor.schema.json)
    tier0.register(name, sp, params, ...)  add a rule kind; its tasks then validate
    observe.RequestEvent / BlockEvent / FinishEvent / ToolCallEvent
    executor.Monitor(mode, policy, tok)    .begin(...) -> Supervisor | None, .reload(policy)
    Supervisor.at_request / at_block / at_finish / at_tool_call    SP4 / SP1 / SP2 / SP3
    Supervisor.session                     the session's state (session.SessionState)
    Journal.write(event)                   one decision; `Journal.sinks` for SRV-115
"""

from server.conductor.monitor.executor import Monitor, Supervisor  # noqa: F401
from server.conductor.monitor.journal import Journal  # noqa: F401
from server.conductor.monitor.observe import (  # noqa: F401
    BlockEvent, FinishEvent, RequestEvent, ToolCallEvent,
)
from server.conductor.monitor.policy import (  # noqa: F401
    ACTIONS, MODES, SAFE_POINTS, TASK_MODES, Action, Policy, PolicyError, load, lower,
    parse_mode, schema,
)
