"""CPU tests for the OCR attachment path: chatd/ocr.py and Service._ocr_ws.

A scanned page is a picture of text; when its message is sent, one OCR pass turns it into
markdown that is written next to the upload (`uploads/<name>.md`) and stored as an ordinary
text attachment, so the decisions that already exist (chatd/attach.py) run it: whole into
the prompt while it fits the budget, one line and paged by read_file past it.

Indra is not a test dependency: the engine is a fake object with the same `convert` the
tests hand to `ocr._conv`, exactly the shape `indra_ocr.load()` returns.
"""

from __future__ import annotations

import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatd import attach, ocr  # noqa: E402
from chatd.service import Service  # noqa: E402


def _engine(pages):
    """The shape indra_ocr.load() returns: convert(path) yields (index, page) with page.markdown."""
    return types.SimpleNamespace(
        convert=lambda path, **kw: [(i, types.SimpleNamespace(markdown=md))
                                    for i, md in enumerate(pages)])


def _svc(root: str) -> Service:
    return Service(root, "http://127.0.0.1:1/v1")        # no test asks the engine


def _ws(root: str, cid: str) -> str:
    ws = os.path.join(root, "ws", cid)
    os.makedirs(os.path.join(ws, "uploads"), exist_ok=True)     # the workspace exists in production
    return ws


def _placed_image(ws: str, name: str = "scan.png"):
    att = {"id": "att_1", "name": name, "kind": "image", "mime": "image/png"}
    att["path"] = attach.save_upload(ws, name, b"\x89PNG a picture of text")
    return att


def _with_engine(pages, fn):
    old = ocr._conv
    ocr._conv = _engine(pages)
    try:
        return fn()
    finally:
        ocr._conv = old


def test_wanted_reads_pdfs_and_images_only():
    assert ocr.wanted("paper.pdf", "application/pdf")
    assert ocr.wanted("photo.JPG", "image/jpeg")
    assert ocr.wanted("no-name", "image/png")
    assert not ocr.wanted("notes.md", "text/markdown")
    assert not ocr.wanted("data.bin", "application/octet-stream")


def test_convert_joins_the_pages_and_says_none_without_text():
    ws = tempfile.mkdtemp()
    path = os.path.join(ws, "x.pdf")
    with open(path, "wb") as fh:
        fh.write(b"%PDF-1.4")

    def run():
        return ocr.convert(path)
    assert _with_engine(["page one", "   ", "page two"], run) == "page one\n\npage two"
    assert _with_engine([None, "  "], run) is None         # a blank scan: nothing to say
    assert ocr.convert(path) is None                       # no engine (CI): inert


def test_the_markdown_lands_next_to_the_upload_and_becomes_a_text_attachment():
    root = tempfile.mkdtemp()
    svc = _svc(root)
    ws = _ws(root, "c1")
    a = _placed_image(ws)

    def run():
        return svc._ocr_ws(types.SimpleNamespace(id="c1"), ws, a, b"\x89PNG a picture of text")
    att = _with_engine(["READABLE TEXT 42"], run)

    assert att is not None and att["kind"] == "text" and att["name"] == "scan.png.md"
    assert int(att["tokens"]) > 0                          # a text attachment like any other
    assert att["path"] == "/workspace/uploads/scan.png.md"
    with open(os.path.join(ws, "uploads", "scan.png.md"), "rb") as fh:
        assert fh.read() == b"READABLE TEXT 42"            # next to the upload, its own file
    assert svc.attachment(att["id"])[1] == b"READABLE TEXT 42"   # and in the blob store


def test_a_second_scan_of_one_name_gets_its_own_file_and_the_same_pass_is_memoized():
    root = tempfile.mkdtemp()
    svc = _svc(root)
    ws = _ws(root, "c1")
    a = _placed_image(ws)
    calls = []
    pages = ["one"]
    eng = types.SimpleNamespace(
        convert=lambda path, **kw: (calls.append(1) or
                                    [(i, types.SimpleNamespace(markdown=md))
                                     for i, md in enumerate(pages)]))
    old = ocr._conv
    ocr._conv = eng                                        # one engine, its identity in the key
    try:
        first = svc._ocr_ws(types.SimpleNamespace(id="c1"), ws, a, b"\x89PNG same bytes")
        pages[:] = ["two"]                                 # what a second pass would say
        second = svc._ocr_ws(types.SimpleNamespace(id="c1"), ws, a, b"\x89PNG same bytes")
    finally:
        ocr._conv = old
    assert first is not None and second is not None
    assert first["path"] != second["path"]                 # save_upload's free names: -2
    assert second["path"] == "/workspace/uploads/scan.png-2.md"
    assert second["name"] == "scan.png-2.md"               # the blob name follows the saved file
    assert len(calls) == 1                                 # one OCR pass for the same bytes
    assert svc.attachment(second["id"])[1] == b"one"       # the memoized text, not "two"


def test_a_symlinked_upload_is_refused():
    root = tempfile.mkdtemp()
    ws = _ws(root, "c1")
    secret = os.path.join(root, "secret.txt")
    with open(secret, "w") as fh:
        fh.write("HOST SECRET")
    target = os.path.join(ws, "uploads", "scan.png")
    os.symlink(secret, target)                             # the sandbox owner's swap
    calls = []

    def counting(path, **kw):
        calls.append(path)
        return [(0, types.SimpleNamespace(markdown="READ"))]

    old = ocr._conv
    ocr._conv = types.SimpleNamespace(convert=counting)
    try:
        assert ocr.convert(target) is None                 # refused: nothing reaches the prompt
    finally:
        ocr._conv = old
    assert calls == []                                     # the engine never opened it


def test_a_missing_blob_says_why_unplaced():
    from chatd.service import Service as S

    class Stub(S):
        def model_info(self, model):
            return {"max_len": 0}
        def _branch(self, conv, head):
            return [], {}

    root = tempfile.mkdtemp()
    svc = Stub(root, "http://127.0.0.1:1/v1")
    ws = _ws(root, "c1")
    a = {"id": "att_missing", "name": "scan.png", "kind": "image"}
    svc._place(types.SimpleNamespace(id="c1"), {"settings": "{}", "head_id": None}, [a])
    assert a.get("unplaced") == "blob missing"             # the prompt says why, like PathError
    assert "path" not in a


def test_a_scan_that_cannot_be_read_sends_unscanned():
    root = tempfile.mkdtemp()
    svc = _svc(root)
    ws = _ws(root, "c1")
    a = _placed_image(ws)

    def broken(path, **kw):
        raise RuntimeError("an engine's surprise")
    old = ocr._conv
    ocr._conv = types.SimpleNamespace(convert=broken)
    try:
        assert svc._ocr_ws(types.SimpleNamespace(id="c1"), ws, a, b"broken bytes") is None
    finally:
        ocr._conv = old


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} passed")
