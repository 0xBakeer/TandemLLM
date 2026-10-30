"""Context usage includes replayed reasoning for clients that compact from API counts.

The request and ledger retain their client counts. Only the explicitly selected API
accounting mode changes; generation, replay and the cache state are untouched.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
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
        "cached_tokens": 260000, "client_tokens": 179027, "hidden_tokens": 82068}
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
    assert r.usage()["prompt_tokens_details"] == {"cached_tokens": 4500, "client_tokens": 7000}


def test_cached_count_cannot_exceed_the_reported_context():
    r = RequestRecord("test", "chat", False)
    r.prompt_tokens, r.context_prompt_tokens, r.cached_tokens = 20, 30, 100
    r.usage_context = "engine"
    assert r.usage()["prompt_tokens_details"]["cached_tokens"] == 30


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
