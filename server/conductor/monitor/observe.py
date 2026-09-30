"""What the rules see: one event per safe point, built from committed tokens and the request.

An event is plain data. It never carries logits, the KV, a drafter or router state, and a rule
cannot change the request through it; a rule reads the event, the request's `RequestEvent` and
the session state, and returns an action or nothing.

    SP4  RequestEvent   the prompt is rendered, nothing is prefilled yet
    SP1  BlockEvent     a verify block's tokens are committed and yielded
    SP2  FinishEvent    an end token is about to end the stream
    SP3  ToolCallEvent  `</tool_call>` closed a call and the call is held
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RequestEvent:
    """SP4, and the request's context at every later safe point (`Supervisor.req`)."""
    request_id: str
    messages: list = field(default_factory=list)   # the body's messages, read-only
    tools: list = field(default_factory=list)      # the tools the model sees ([] with none)
    tool_choice: str = "auto"                      # auto | none | required | function
    n: int = 1
    stream: bool = False
    prompt_tokens: int = 0                         # as the client sent them
    in_think: bool = False                         # the prompt leaves the model thinking
    max_tokens: int = 0                            # the request's output cap, after the clamp
    client_budget: int | None = None               # max_reasoning_tokens / thinking_budget
    client: str = ""                               # ledger client_kind: opencode, openwebui, ...
    model: str = ""

    @property
    def tools_offered(self) -> bool:
        return bool(self.tools) and self.tool_choice != "none"


@dataclass
class BlockEvent:
    """SP1. `stall` is the ThinkBudget arm that fired (`loop(..)`, `repeat16(..)`,
    `novelty(..)`), `share` the history arm's last repeat share."""
    reasoning_tokens: int
    inside: bool
    stall: str | None
    share: float | None
    output_tokens: int


@dataclass
class FinishEvent:
    """SP2: the model wrote its end token. `inside`: the reasoning block was still open.
    `answer`: the text after the block, as written so far (the whole output with thinking off)."""
    kind: str
    inside: bool
    answer: str
    reasoning_tokens: int
    output_tokens: int


@dataclass
class ToolCallEvent:
    """SP3: one parsed call, held. `arguments` is the JSON string a client would receive."""
    name: str
    arguments: str
    index: int
