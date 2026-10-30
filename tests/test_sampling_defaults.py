"""SPD-245: a sampled request's filters when it names none (server/sampling_defaults.py)."""

import json
import os
import tempfile

from server import sampling_defaults as sd
from server.conductor import loopescape as le

CARD = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "source": "generation_config.json"}


def test_no_filter_takes_the_card():
    assert sd.resolve({"temperature": 1.0}, {"top_p": None, "top_k": None}, CARD) == {
        "top_p": 0.95, "top_k": 20, "min_p": 0.0, "source": "card"}


def test_no_card_is_the_whole_vocabulary():
    assert sd.resolve({"temperature": 1.0}, {}, None) == {
        "top_p": 1.0, "top_k": 0, "min_p": 0.0, "source": "none"}


def test_a_named_filter_keeps_the_request_and_the_rest_plain():
    got = sd.resolve({"temperature": 1.0, "top_p": 0.8}, {}, CARD)
    assert got == {"top_p": 0.8, "top_k": 0, "min_p": 0.0, "source": "request"}
    got = sd.resolve({"min_p": 0.05}, {}, CARD)
    assert got == {"top_p": 1.0, "top_k": 0, "min_p": 0.05, "source": "request"}
    # an explicit whole-vocabulary request stays one
    got = sd.resolve({"top_k": 0, "top_p": 1.0}, {}, CARD)
    assert (got["top_k"], got["top_p"], got["source"]) == (0, 1.0, "request")


def test_server_flags_beat_the_card():
    got = sd.resolve({"temperature": 0.7}, {"top_p": 0.9, "top_k": None}, CARD)
    assert got == {"top_p": 0.9, "top_k": 0, "min_p": 0.0, "source": "server"}
    got = sd.resolve({"top_k": 40}, {"top_p": 0.9}, CARD)
    assert got == {"top_p": 0.9, "top_k": 40, "min_p": 0.0, "source": "request"}


def test_bad_field_raises():
    try:
        sd.resolve({"top_k": "many"}, {}, CARD)
    except (TypeError, ValueError):
        return
    raise AssertionError("a bad top_k was taken")


def test_the_27b_card_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "generation_config.json"), "w") as f:
            json.dump({"do_sample": True, "temperature": 1.0, "top_k": 20, "top_p": 0.95}, f)
        card = le.card_sampling(d)
    got = sd.resolve({"temperature": 1.0, "max_tokens": 8}, {"top_p": None, "top_k": None}, card)
    assert (got["top_k"], got["top_p"], got["min_p"], got["source"]) == (20, 0.95, 0.0, "card")


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
