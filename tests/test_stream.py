"""The two serving bugs that were bugs about text rather than about tokens.

Both were found by the operator driving the server from Open WebUI, and both are invisible to every
benchmark in this repository: the atlas row counts tokens and measures the gaps between them, and
neither the count nor the gap changes when the characters are wrong.

No torch, no tokeniser, no board. The fixture is a byte-level decoder written out by hand, because
what is under test is the rule for holding a piece back and not the tokeniser that produces the
situation.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.stream import (  # noqa: E402
    Detokenizer, Reasoning, StopStrings, opens_think, split_full,
)


# --- a decoder that behaves the way a byte-level one does ---------------------------------------

def byte_decoder(pieces: dict[int, bytes]):
    """`decode(ids)` over a table of id -> raw bytes, decoded with `errors="replace"`.

    This is what an HF fast tokeniser does for a prefix whose last code point is incomplete: the
    trailing bytes come back as U+FFFD, and at the next token the same prefix decodes to the real
    character. Reproducing it in four lines is what lets the rule be tested without the model.
    """
    def decode(ids):
        return b"".join(pieces[i] for i in ids).decode("utf-8", errors="replace")
    return decode


# 🤦🏼 is U+1F926 U+1F3FC: four bytes and four bytes, split across five tokens the way the real
# tokeniser splits it (SPEED-LEDGER phase 9, the probe that found this).
FACEPALM = {
    1: "Oh ".encode(),
    2: "\U0001F926".encode()[:3],      # three of the four bytes of the face
    3: "\U0001F926".encode()[3:] + "\U0001F3FC".encode()[:1],
    4: "\U0001F3FC".encode()[1:3],
    5: "\U0001F3FC".encode()[3:],
    6: " done".encode(),
}


def drive(det: Detokenizer, ids: list[int]) -> str:
    out = ""
    for n in range(1, len(ids) + 1):
        out += det.push(ids[:n])
    return out


def test_a_code_point_that_spans_tokens_is_never_emitted_in_pieces():
    """The bug the operator saw: a stray replacement character, permanently, in the middle of a word."""
    det = Detokenizer(byte_decoder(FACEPALM))
    got = drive(det, [1, 2, 3, 4, 5, 6])
    assert got == "Oh \U0001F926\U0001F3FC done", repr(got)
    assert "�" not in got


def test_the_naive_rule_is_the_one_that_was_wrong():
    """Stated as a test so the regression has a shape: decode-the-prefix-and-diff loses a
    character here, and it loses it for ever -- the prefix after the break is SHORTER in
    characters than what has already gone out, so the tail is never sent at all."""
    decode = byte_decoder(FACEPALM)
    emitted, naive = "", ""
    for n in range(1, 7):
        text = decode([1, 2, 3, 4, 5, 6][:n])
        naive += text[len(emitted):]
        emitted = text
    assert "�" in naive
    assert "\U0001F3FC" not in naive


def test_every_piece_is_final_and_the_pieces_concatenate_to_the_whole():
    det = Detokenizer(byte_decoder(FACEPALM))
    ids = [1, 2, 3, 4, 5, 6]
    pieces = [det.push(ids[:n]) for n in range(1, len(ids) + 1)]
    assert "".join(pieces) == byte_decoder(FACEPALM)(ids)
    # the two steps that carry an incomplete character send nothing at all
    assert pieces == ["Oh ", "", "\U0001F926", "", "\U0001F3FC", " done"], pieces


def test_a_broken_tail_still_goes_out_at_the_end():
    """At end of stream there is no next token, so what is held back is genuinely broken input and
    dropping it would lose bytes the model really produced."""
    table = {1: "ab".encode(), 2: "\U0001F926".encode()[:2]}
    det = Detokenizer(byte_decoder(table))
    assert drive(det, [1, 2]) == "ab"
    assert det.flush([1, 2]) == "�"


def test_a_replacement_character_in_the_middle_does_not_stall_the_stream():
    """Only the TRAILING run is held: a bad byte already followed by a good character is not
    going to become anything else, and waiting for it would stop the stream for ever."""
    table = {1: b"\xff", 2: "ok".encode()}
    det = Detokenizer(byte_decoder(table))
    assert drive(det, [1, 2]) == "�ok"


# --- the reasoning block ------------------------------------------------------------------------

def test_a_generation_prompt_with_thinking_on_leaves_the_block_open():
    assert opens_think("<|im_start|>assistant\n<think>\n")
    # thinking off: the template opens AND closes the block inside the prompt
    assert not opens_think("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert not opens_think("<|im_start|>user\nhi<|im_end|>\n")


def test_the_tagged_stream_starts_with_the_opening_tag():
    """`content` starts with `<think>` -- the property the fix is verified by. Since SRV-16 the
    tag goes out in the same delta as the first text (tests/test_app_loop.py)."""
    r = Reasoning("tags", in_think=True)
    first = [("content", "<think>\n")] + r.push("thinking")
    assert first[0] == ("content", "<think>\n")
    # and in `tags` everything reaches the content field, closing tag included -- the block's
    # part labelled `tagged` so the tool-call buffer never reads it (SRV-23)
    assert r.push("</think>\n\nanswer") == [("tagged", "</think>"), ("content", "\n\nanswer")]


def test_the_tagged_block_ends_where_the_closing_tag_does_even_split_across_pieces():
    """SRV-23: `tags` holds nothing back and still knows where the answer starts."""
    r = Reasoning("tags", in_think=True)
    got = r.push("a <tool_call> I will not make </th") + r.push("ink>\n\n<tool_call>")
    assert got == [("tagged", "a <tool_call> I will not make </th"), ("tagged", "ink>"),
                   ("content", "\n\n<tool_call>")], got
    assert r.push("more") == [("content", "more")]
    assert "".join(t for _, t in got) == "a <tool_call> I will not make </think>\n\n<tool_call>"


def test_reasoning_content_splits_at_the_closing_tag_and_drops_it():
    r = Reasoning("reasoning_content", in_think=True)
    assert r.push("weighing it up") == [("reasoning", "weighing it up")]
    assert r.push("</think>\n\nThe answer.") == [("content", "The answer.")]
    assert r.push(" More.") == [("content", " More.")]


def test_a_closing_tag_split_across_pieces_is_still_recognised():
    """The tag arrives a token at a time like everything else, and a tail that could still grow
    into one is held back rather than sent as reasoning."""
    r = Reasoning("reasoning_content", in_think=True)
    assert r.push("done") == [("reasoning", "done")]
    assert r.push("</th") == []
    assert r.push("ink>") == []
    assert r.push("\n\nyes") == [("content", "yes")]


def test_a_tail_that_cannot_become_the_tag_is_released():
    r = Reasoning("reasoning_content", in_think=True)
    assert r.push("a<") == [("reasoning", "a")]
    assert r.push("b") == [("reasoning", "<b")]


def test_both_sends_the_reasoning_twice_and_the_answer_once():
    r = Reasoning("both", in_think=True)
    assert r.push("why") == [("reasoning", "why"), ("content", "why")]
    assert r.push("</think>\n\nbecause") == [("content", "because")]


def test_what_is_still_held_back_when_the_generation_ends_is_not_lost():
    r = Reasoning("reasoning_content", in_think=True)
    r.push("mid</thin")
    assert r.finish() == [("reasoning", "</thin")]
    assert r.finish() == []


def test_a_request_with_thinking_off_is_never_split():
    for fmt in ("tags", "reasoning_content", "both"):
        r = Reasoning(fmt, in_think=False)
        assert r.push("plain </think> text") == [("content", "plain </think> text")]


def test_the_non_streamed_answer_matches_the_streamed_one():
    text = "weighing it up</think>\n\nThe answer."
    assert split_full(text, "tags") == ("<think>\nweighing it up</think>\n\nThe answer.", None)
    assert split_full(text, "reasoning_content") == ("The answer.", "weighing it up")
    assert split_full(text, "both") == ("<think>\n" + text, "weighing it up")
    assert split_full(text, "tags", in_think=False) == (text, None)


def test_an_answer_cut_off_inside_the_reasoning_block_has_no_content():
    """A generation that hits the token limit before `</think>` has produced no answer, and
    promoting the reasoning to one would be the server inventing it."""
    assert split_full("still thinking", "reasoning_content") == ("", "still thinking")


def test_an_unknown_reasoning_format_is_refused_rather_than_guessed():
    for bad in ("Tags", "openai", ""):
        try:
            Reasoning(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} was accepted")


# --- stop strings (SRV-25) ------------------------------------------------------------------------

def _stopped(pieces, stops):
    st = StopStrings(stops)
    out = [st.push(p) for p in pieces]
    return out, st.finish(), st.hit


def _whole(text, stops):
    """The non-streamed rule: cut at the earliest match."""
    hits = [text.find(s) for s in stops if s and text.find(s) >= 0]
    return text[:min(hits)] if hits else text


def test_a_stop_string_split_across_pieces_is_never_sent():
    assert _stopped(list("Stop here."), ["Stop"]) == ([""] * 10, "", True)
    assert _stopped(["Go S", "to", "p!"], ["Stop"]) == (["Go ", "", ""], "", True)


def test_a_held_prefix_is_released_when_it_cannot_match_and_at_the_end():
    assert _stopped(["Sto", "rm"], ["Stop"]) == (["", "Storm"], "", False)
    assert _stopped(["Hello Sto"], ["Stop"]) == (["Hello "], "Sto", False)


def test_the_earliest_match_wins_across_stop_strings():
    # "bc" completes first, but "abcd" started earlier and is still arriving
    assert _stopped(list("abcdef"), ["abcd", "bc"]) == (["", "", "", "", "", ""], "", True)
    out, tail, hit = _stopped(list("abcxef"), ["abcd", "bc"])
    assert "".join(out) + tail == "a" and hit
    # ended while waiting on the longer one: the shorter match still cuts
    out, tail, hit = _stopped(list("abc"), ["abcd", "bc"])
    assert "".join(out) + tail == "a" and hit


def test_every_split_of_every_case_matches_the_non_streamed_rule():
    cases = [("Stop here.", ["Stop"]), ("xaab", ["ab", "aab"]), ("aaab", ["aab"]),
             ("abcdef", ["abcd", "bc"]), ("abcxef", ["abcd", "bc"]), ("a</thi</think>b", ["</think>"]),
             ("mississippi", ["issip", "ssi"]), ("plain", []), ("plain", [""]), ("ababab", ["abb"])]
    for text, stops in cases:
        for n in range(1, len(text) + 1):             # every piece size
            pieces = [text[i:i + n] for i in range(0, len(text), n)]
            out, tail, hit = _stopped(pieces, stops)
            assert "".join(out) + tail == _whole(text, stops), (text, stops, n, out, tail)
            assert hit == (_whole(text, stops) != text), (text, stops, n)


def test_without_stop_strings_every_piece_goes_straight_through():
    st = StopStrings([])
    assert [st.push(p) for p in ("S", "to", "p")] == ["S", "to", "p"] and st.finish() == ""


if __name__ == "__main__":
    import traceback
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok  {name}")
            except Exception:
                fails += 1
                print(f"FAIL  {name}")
                traceback.print_exc()
    print(f"\n{fails} failed")
    sys.exit(1 if fails else 0)
