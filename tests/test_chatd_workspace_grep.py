"""Grep tells the truth: a file path is searchable, and size is not a mute button.

Reproduced live on a 256K conversation: the model grepped `^#{1,2} ` at the manual's .md —
the tool answered "no matches" while the file's very first line is `## TandemLLM`.

Also here: grep is bounded (a newline-less multi-MB file cannot OOM the thread — GREP_LINE_MAX
truncates the line, GREP_FILE_MAX caps the file), line numbers follow the SAME boundaries
read_file's splitlines() uses (a form feed advances both tools by one), and the line pass
runs per 1 MiB chunk instead of re-copying the buffer per line.

Run: python tests/test_chatd_workspace_grep.py   (pytest-collectable too: def test_* + assert)
"""
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatd.workspace import Workspace


def make_ws():
    d = tempfile.mkdtemp(prefix="grep-test-")
    os.makedirs(d + "/uploads", exist_ok=True)
    with open(d + "/uploads/notes.md", "w") as f:
        f.write("# Notes\nsome prose\n## Section A\n")
    # a file past the old MAX_READ cliff (2 MB): the needle sits at the end
    with open(d + "/uploads/big.txt", "w") as f:
        while os.path.getsize(d + "/uploads/big.txt") < 2_200_000:
            f.write("filler " + "x" * 60 + "\n")
        f.write("NEEDLE-PAST-CLIFF\n")
    # a newline-less 3 MB file (a request dump): the needle sits inside the first 64 KiB of
    # the one giant line, so it is found; the point is that the scan terminates bounded
    with open(d + "/uploads/dump.bin", "wb") as f:
        f.write(b"junk " + b"x" * 4096)
        f.write(b"NEEDLE-IN-DUMP")
        f.write(b"y" * (3_000_000 - f.tell()))
    # a form-feed file: grep must number lines exactly where read's splitlines() puts them
    with open(d + "/uploads/ff.md", "wb") as f:
        f.write(b"one\ntwo\x0cthree\x0cfox HITME\nseven\n")
    return Workspace(d)


WS = make_ws()


def test_a_file_path_is_searchable():
    h = WS.grep("^## ", "/workspace/uploads/notes.md")
    assert len(h) == 1 and h[0].endswith("## Section A")        # was: PathError -> 'no matches'


def test_anchor_works_per_line_on_a_single_file_grep():
    h = WS.grep("^# ", "/workspace/uploads/notes.md")
    assert len(h) == 1 and "Notes" in h[0]


def test_a_file_past_the_old_cliff_is_scanned_not_skipped():
    assert len(WS.grep("NEEDLE-PAST-CLIFF", "/workspace")) == 1


def test_the_big_file_answers_to_its_own_path():
    assert len(WS.grep("NEEDLE-PAST-CLIFF", "/workspace/uploads/big.txt")) == 1


def test_directory_walk_still_greps_siblings():
    assert len(WS.grep("Section", "/workspace")) == 1


def test_a_genuine_no_match_still_says_so():
    assert WS.grep("no-such-thing-xyz", "/workspace") == []


def test_line_numbers_follow_reads_splitlines():
    # a form feed ends a line for read(); grep must count it the same: one, two | three | hit
    h = WS.grep("HITME", "/workspace/uploads/ff.md")
    assert len(h) == 1 and ":4:" in h[0], h


def test_the_newline_less_dump_is_scanned_in_bounds():
    t0 = time.monotonic()
    h = WS.grep("NEEDLE-IN-DUMP", "/workspace/uploads/dump.bin")
    took = time.monotonic() - t0
    assert len(h) == 1 and took < 5.0


def test_long_lines_in_bulk_keep_every_number():
    # the skip of an over-window line ends at the FIRST boundary in the buffer: a
    # skip that ran to the buffer's LAST boundary swallowed whole lines sitting
    # between the long line and the chunk end — 60 lines of 200 KB answered 28
    # of 60 and numbered the rest wrong
    d = tempfile.mkdtemp(prefix="grep-long-")
    os.makedirs(d + "/uploads", exist_ok=True)
    with open(d + "/uploads/longlines.txt", "w") as f:
        for i in range(1, 61):
            f.write("needle%d " % i + "x" * (200_000 - 9) + "\n")
    ws = Workspace(d)
    h = ws.grep("^needle", "/workspace/uploads/longlines.txt")
    nums = sorted(int(line.split(":")[1]) for line in h)
    assert nums == list(range(1, 61)), "expected all 60 lines with their true numbers, got %r" % nums[:10]


def test_the_giant_line_keeps_line_number_1():
    h = WS.grep("NEEDLE-IN-DUMP", "/workspace/uploads/dump.bin")
    assert len(h) == 1 and ":1:" in h[0]


def test_a_giant_line_past_the_window_ends_the_scan_no_hang():
    t0 = time.monotonic()
    assert WS.grep("no newline here either", "/workspace/uploads/dump.bin") == []
    assert time.monotonic() - t0 < 10.0


def test_a_crlf_split_across_the_chunk_seam_counts_once():
    # GREP_CHUNK is 1 MiB: park the \r as the last byte of chunk 1 and the \n as the
    # first of chunk 2. An eager per-chunk split would count that pair twice and
    # report SEAMHIT one line late.
    d = tempfile.mkdtemp(prefix="grep-seam-")
    os.makedirs(d + "/uploads")
    with open(d + "/uploads/seam.txt", "wb") as f:
        f.write(b"x\n" * 524287)          # 1_048_574 bytes: 524287 complete lines
        f.write(b"Q")                     # byte 1_048_574
        f.write(b"\r\n")                  # 1_048_575 = '\r' (last of chunk 1), 1_048_576 = '\n'
        f.write(b"SEAMHIT\n")
    h = Workspace(d).grep("SEAMHIT", "/workspace/uploads/seam.txt")
    assert len(h) == 1, h
    assert int(h[0].split(":")[1]) == 524289, h


def test_a_multibyte_char_cut_by_the_seam_still_matches():
    # '\xc3' lands as the last byte of chunk 1, '\xa9' opens chunk 2: a per-chunk
    # decode would mint two U+FFFD where the file holds an 'é', and the grep would
    # call the line absent while read() shows it.
    d = tempfile.mkdtemp(prefix="grep-seam-")
    os.makedirs(d + "/uploads")
    with open(d + "/uploads/seam.md", "wb") as f:
        f.write(b"y\n" * 524287)           # 1_048_574 bytes: 524287 complete lines
        f.write(b"w")                     # byte 1_048_574, opens line 524288
        f.write(b"\xc3")                  # 1_048_575: last byte of chunk 1
        f.write(b"\xa9HITSPLIT\n")        # chunk 2 opens with the continuation byte
    h = Workspace(d).grep("HITSPLIT", "/workspace/uploads/seam.md")
    assert len(h) == 1 and int(h[0].split(":")[1]) == 524288, h
    # the é itself is intact for a grep — not two replacement chars:
    h2 = Workspace(d).grep("wéHITSPLIT", "/workspace/uploads/seam.md")
    assert len(h2) == 1, h2


def test_a_file_at_the_cliff_searches_its_last_chunk_and_says_nothing_more():
    # the reported repro: a file of exactly 16,777,216 bytes with the match in
    # the last 20. The loop used to drop that chunk before searching it -- no
    # match, nothing said. A cliff chunk gets searched; and a file that ENDS
    # at the cap has no rest, so it gets no "unread" line either (peek first).
    d = tempfile.mkdtemp(prefix="grep-cliff-")
    os.makedirs(d + "/uploads")
    with open(d + "/uploads/big.txt", "wb") as f:
        f.write(b"x\n" * 8388598)             # 16,777,196 bytes of filler lines
        f.write(b"NEEDLE-AT-THE-CLIFF\n")     # the last 20: chunk 16
    h = Workspace(d).grep("NEEDLE-AT-THE-CLIFF", "/workspace/uploads/big.txt")
    assert len(h) == 1, h
    assert ":8388599:" in h[0], h


def test_a_file_past_the_cliff_says_where_it_stopped():
    # one byte more than the cap: the note belongs to files that DO have a rest
    d = tempfile.mkdtemp(prefix="grep-cliff2-")
    os.makedirs(d + "/uploads")
    with open(d + "/uploads/big.txt", "wb") as f:
        f.write(b"x\n" * 8388598)             # 16,777,196 bytes of filler lines
        f.write(b"NEEDLE-AT-THE-CLIFF\n")     # the last 20: chunk 16
        f.write(b"and the rest\n")            # one byte over is enough to say it
    h = Workspace(d).grep("NEEDLE-AT-THE-CLIFF", "/workspace/uploads/big.txt")
    assert len(h) == 2, h
    assert ":8388599:" in h[0], h
    assert "16 MiB" in h[1] and "unread" in h[1], h


def test_unicode_line_boundaries_count_lines_where_splitlines_does():
    # the reviewer's case: a chunk of pure-'\n' among the ASCII controls takes the
    # str.split fast path -- but _GREP_EOL (and splitlines) also break on U+0085,
    # U+2028 and U+2029, so the fast path under-counted and the needle landed one
    # line early. The decoded head is probed for them before the fast path may run.
    d = tempfile.mkdtemp(prefix="grep-eol-")
    os.makedirs(d + "/uploads")
    with open(d + "/uploads/u.txt", "w", encoding="utf-8") as f:
        f.write("line one\u2028line two\nline three NEEDLE")
    h = Workspace(d).grep("NEEDLE", "/workspace/uploads/u.txt")
    assert len(h) == 1 and h[0].endswith("line three NEEDLE") and ":3: " in h[0], h


def test_a_fuzz_of_unicode_separators_only_agrees_with_splitlines():
    # 100 files whose lines are separated by '\n' and the three Unicode boundaries
    # only -- no ASCII control byte anywhere, so every chunk asks for the fast path.
    # The fuzz that mixed in ASCII separators hid the bug: those files took the
    # general path anyway.
    import random
    rng = random.Random(20261023)
    seps = ["\n", "\u0085", "\u2028", "\u2029"]
    d = tempfile.mkdtemp(prefix="grep-fuzz-")
    os.makedirs(d + "/uploads")
    want = {}
    for i in range(100):
        lines = [f"pad {i} {j}" for j in range(rng.randrange(20, 200))]
        hit = rng.randrange(len(lines))
        lines[hit] = f"FUZZNEEDLE-{i}-MARK"
        text = "".join(ln + rng.choice(seps) for ln in lines)
        with open(d + f"/uploads/f{i}.txt", "w", encoding="utf-8") as f:
            f.write(text)
        want[f"f{i}.txt"] = len(text.splitlines()[:hit]) + 1
    h = Workspace(d).grep("FUZZNEEDLE-", "/workspace/uploads")
    got = {}
    for line in h:
        path, num, _rest = line.split(":", 2)
        got[path.rsplit("/", 1)[1]] = int(num)
    assert sorted(got) == sorted(want), f"{len(got)} hits, expected 100"
    for name in want:
        assert got[name] == want[name], f"{name}: grep line {got[name]}, splitlines {want[name]}"


def run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok   {t.__name__}")
    print(f"PASS ({len(tests)} tests)")


if __name__ == "__main__":
    run()
