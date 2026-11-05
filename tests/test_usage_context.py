"""Context usage includes replayed reasoning for clients that compact from API counts.

The request and ledger retain their client counts. Only the explicitly selected API
accounting mode changes; generation, replay and the cache state are untouched.
"""
from pathlib import Path
import sys
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chatd import context
from chatd.runs import Run, client_prompt_of
from server.usage import RequestRecord


def test_replayed_reasoning_counts_toward_the_compaction_threshold():
    r = RequestRecord("test", "chat", True)
    r.prompt_tokens = 179027
    r.hidden_prompt = 82068
    r.context_prompt_tokens = 261095
    r.cached_tokens = 260000
    r.completion_tokens = 1048
    r.reasoning_tokens = 789
    r.usage_context = "engine"
    u = r.usage()
    assert u["prompt_tokens"] == 261095
    assert u["prompt_tokens"] > 200000
    assert u["total_tokens"] == 262143
    assert u["prompt_tokens_details"] == {
        "cached_tokens": 260000, "client_tokens": 179027, "client_cached_tokens": 179027,
        "hidden_tokens": 82068}
    assert u["completion_tokens_details"]["reasoning_tokens"] == 789
    assert r.prompt_tokens == 179027
    assert r.row()["prompt_tokens"] == 179027


def test_client_accounting_remains_the_default():
    r = RequestRecord("test", "chat", False)
    r.prompt_tokens, r.hidden_prompt, r.cached_tokens = 11, 227, 230
    assert r.usage()["prompt_tokens"] == 11
    assert r.usage()["prompt_tokens_details"] == {"cached_tokens": 11, "hidden_tokens": 227}


def test_plain_requests_have_the_same_usage_in_both_modes():
    r = RequestRecord("test", "chat", False)
    r.prompt_tokens, r.context_prompt_tokens, r.cached_tokens = 100, 100, 40
    r.completion_tokens = 12
    client = r.usage()
    r.usage_context = "engine"
    assert r.usage() == client


def test_continuation_uses_its_actual_context_not_the_client_prompt():
    r = RequestRecord("test", "chat", False)
    r.prompt_tokens = r.prompt_client = 7000
    r.context_prompt_tokens = 5000
    r.cached_tokens = 4500
    r.usage_context = "engine"
    assert r.usage()["prompt_tokens"] == 5000
    assert r.usage()["prompt_tokens_details"] == {
        "cached_tokens": 4500, "client_tokens": 7000, "client_cached_tokens": 4500}


def test_cached_count_cannot_exceed_the_reported_context():
    r = RequestRecord("test", "chat", False)
    r.prompt_tokens, r.context_prompt_tokens, r.cached_tokens = 20, 30, 100
    r.usage_context = "engine"
    assert r.usage()["prompt_tokens_details"]["cached_tokens"] == 30


def test_engine_usage_counts_continuation_cached_rows_in_the_occupied_context():
    r = RequestRecord("test", "chat", False)
    r.prompt_tokens = r.prompt_client = 7_000
    r.context_prompt_tokens = 10_000
    r.usage_context = "engine"
    r.absorb_prefill({"reused": 2_000, "forwarded": 2_500, "kind": "end"})
    assert r.cache_source == "continuation"
    assert r.cached_tokens == 4_500                   # ledger and tandem keep client accounting
    assert r.usage()["prompt_tokens_details"] == {
        "cached_tokens": 7_500, "client_tokens": 7_000, "client_cached_tokens": 4_500}


def test_engine_usage_counts_every_replayed_occupied_row_as_cached():
    r = RequestRecord("test", "chat", False)
    r.prompt_tokens, r.context_prompt_tokens = 7_000, 10_000
    r.usage_context = "engine"
    r.absorb_response_cache()
    assert r.cache_source == "response" and r.prompt_n == 0
    assert r.cached_tokens == 7_000                   # ledger and tandem keep client accounting
    assert r.usage()["prompt_tokens_details"] == {
        "cached_tokens": 10_000, "client_tokens": 7_000, "client_cached_tokens": 7_000}


def test_chatd_learns_and_reports_from_the_client_prompt():
    usage = {"prompt_tokens": 900,
             "prompt_tokens_details": {"cached_tokens": 100, "client_tokens": 600}}
    assert client_prompt_of(usage) == 600
    assert client_prompt_of({"prompt_tokens": 900}) == 900

    learned = []
    svc = types.SimpleNamespace(
        learn_ratio=lambda conv, prompt, chars, images: learned.append((prompt, chars, images)))
    run = Run(svc, types.SimpleNamespace(id="c1"), user_message_id="u", message_id="m",
              max_steps=10, started_by=None)
    run.step, run.request_chars, run.request_images = 1, 1_200, 0
    run.call = types.SimpleNamespace(
        events=lambda: [("data", {"usage": dict(
            usage, completion_tokens=12,
            completion_tokens_details={"reasoning_tokens": 0})})])
    assert run._consume() == "completed"
    assert learned == [(600, 1_200, 0)]
    assert run.step_usage[-1]["prompt_tokens"] == 600
    assert run.stats("stop")["prompt_tokens"] == 600


def test_chatd_pairs_the_client_prompt_with_the_client_cached_count():
    r = RequestRecord("test", "chat", True)
    r.prompt_tokens, r.hidden_prompt = 179_027, 82_068
    r.context_prompt_tokens = 261_095
    r.hidden_at = [(100_000, 182_068)]                # the retained reasoning, inside the cached span
    r.cached_tokens, r.prompt_n = 255_000, 6_095
    r.completion_tokens = 1_048
    r.usage_context = "engine"
    usage = r.usage()
    assert usage["prompt_tokens_details"]["cached_tokens"] == 255_000
    assert usage["prompt_tokens_details"]["client_cached_tokens"] == 172_932

    svc = types.SimpleNamespace(learn_ratio=lambda *a: None)
    run = Run(svc, types.SimpleNamespace(id="c1"), user_message_id="u", message_id="m",
              max_steps=10, started_by=None)
    run.step, run.request_chars, run.request_images = 1, 700_000, 0
    run.call = types.SimpleNamespace(events=lambda: [("data", {"usage": usage})])
    assert run._consume() == "completed"
    assert run.step_usage[-1]["prompt_tokens"] == 179_027
    assert run.step_usage[-1]["cached_tokens"] == 172_932
    stats = run.stats("stop")
    assert (stats["prompt_tokens"], stats["cached_tokens"]) == (179_027, 172_932)
    rep = context.report(model={"max_len": 262_144}, settings={}, branch_tokens=180_075,
                         files_tokens=0, stats=stats, run=None, row=None, facts={}, peek=None,
                         harness=None)
    assert (rep["cache"]["warm"], rep["cache"]["cold"]) == (172_932, 6_095), rep["cache"]


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
