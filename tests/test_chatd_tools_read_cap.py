# read_file's result cap (issue #18): a manual of 27 pages became 2000 lines of long
# paragraphs — the default limit swallowed 178,580 tokens in one tool result, 68 % of a
# 262,144-token window, and the run died one step past it. One call now shows at most
# MAX_RESULT_BYTES bytes and names the offset to read on from. Run: python
# tests/test_chatd_tools_read_cap.py — no sandbox, no engine.

import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatd.tools import MAX_RESULT_BYTES, _read


class _Ws:
    def __init__(self, data: bytes):
        self.data = data

    def stat(self, path):
        return types.SimpleNamespace(st_size=len(self.data), st_mode=0o100644)

    def read(self, path, limit=4 * 1024 * 1024):
        return self.data[:limit]

    def resolve(self, path):
        return "/workspace/" + path.lstrip("/")


def ctx(data: bytes):
    return types.SimpleNamespace(ws=_Ws(data))


def test_one_step_cannot_swallow_a_file():
    data = ("\n".join(f"line {i} " + ("paragraph " * 40) for i in range(1, 4001))).encode()
    r = _read(ctx(data), {"path": "manual.md"})
    assert len(r.content.encode()) <= MAX_RESULT_BYTES + 4096, \
        f"a read_file result was {len(r.content.encode()):,} bytes"
    assert f"{MAX_RESULT_BYTES:,} bytes" in r.content and "read on with offset" in r.content


def test_paging_from_the_named_offset_reaches_the_end():
    data = ("\n".join(f"line {i} " + ("paragraph " * 40) for i in range(1, 4001))).encode()
    off, seen, calls = 1, 0, 0
    while off:
        r = _read(ctx(data), {"path": "manual.md", "offset": off})
        lines = [ln for ln in r.content.splitlines() if ln[:6].strip().isdigit()]
        assert lines, r.content[:200]
        seen += len(lines)
        calls += 1
        assert calls < 200
        off = int(lines[-1][:6]) + 1 if "more lines left" in r.content else 0
    assert seen == 4000, f"paged {seen} of 4000 lines in {calls} calls"


def test_short_files_are_untouched():
    data = b"\n".join(b"short %d" % i for i in range(10))
    r = _read(ctx(data), {"path": "a.txt"})
    assert "more lines left" not in r.content and "short 9" in r.content


def test_binary_still_says_not_shown():
    r = _read(ctx(b"\x89PNG\x00\x00\x00"), {"path": "x.bin"})
    assert "binary file" in r.content


def test_images_are_still_images():
    r = _read(ctx(b"\x89PNG\r\n\x1a\n" + b"0" * 100), {"path": "p.png"})
    assert r.images and "an image" in r.content


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok ", fn.__name__)
    print(f"{len(fns)}/{len(fns)} passed")
