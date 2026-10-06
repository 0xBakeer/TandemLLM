"""The package texts after the UT1 self-test (2026-10-06): SRV-163 and SRV-171. CPU only.

  1  SRV-163: a literal address in front of TandemLLM's own routes is found in every form the texts
     used (a shell default, a Python default plus a path, a fetch, a session-token path), and a
     packages document that has one is refused at save with the field named; `{tandem_url}` passes
                                                        test_literal_hosts_are_refused
  2  `{tandem_url}` is the request's address, or `QWEN38_TANDEM_URL` when the operator sets one;
     `{tandem_scratch}` is `.tandem` in the project (excluded from git, deleted before done), or
     QWEN38_TANDEM_SCRATCH
                                                        test_the_tokens
  3  the shipped texts: no `.tandem/`, no literal host; the testing package has the scratch, hook
     and no-download rules; the migration leaves them unchanged (idempotent)
                                                        test_the_shipped_texts_are_clean
  4  the box's saved document of 2026-10-03 (v7: the literal host in four packages, `.tandem/` in
     eleven), migrated: clean, and the save check passes   test_a_box_document_migrates_clean
  5  the Laya helper, against an address nobody answers: exits 3 at once with the truthful line
     ("unavailable ... your code is not the cause"), never a traceback
                                                        test_the_helper_fails_fast
  6  the playtest kit's log sits beside the kit, not in the working directory; the outcome reader
     reads a log tail in the scratch directory        test_the_kit_and_the_log_tail
  7  SRV-195: the box's saved document of 0.3.43 (v11: "verify with what already works here",
     "a game or a UI flow" played by Laya) migrates to the kit's page check, no driver of one's
     own, check-fix-check, a one-screen tool without a playtest; every command the texts give
     for the page check parses with the kit's own argument parser; a second run changes nothing
                                                        test_the_page_check_edits

Run: python tests/test_package_texts.py
"""

from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from server.conductor import packages as P  # noqa: E402
from server.conductor import schema  # noqa: E402
from tools import packages_migrate as M  # noqa: E402

HOST = "https://ai-api.example.lan"


def _seed() -> dict:
    with open(os.path.join(ROOT, "packages", "seed.json")) as f:
        doc = json.load(f)
    for name, p in doc["packages"].items():
        src = p.pop("body_from", None)
        if src:
            with open(os.path.join(ROOT, "packages", src)) as f:
                p["body"] = f.read()
    assert not schema.doc_errors("packages", doc), schema.doc_errors("packages", doc)[:3]
    return doc


def test_literal_hosts_are_refused():
    forms = [f'curl -s "${{TANDEM_URL:-{HOST}}}/v1/laya/ask"',
             f'URL = os.environ.get("TANDEM_URL", "{HOST}") + "/v1/laya/ask"',
             f'await fetch("{HOST}/v1/laya/ask", {{method: "POST"}})',
             f"curl -s {HOST}/s/Ab3xYz9_kLmn/v1/library/x",
             f"curl -s {HOST}:8443/v1/memory/search?q=a"]
    for text in forms:
        assert P.literal_api_hosts(text), text
    for text in ("curl -s {tandem_url}/v1/laya/ask", "https://github.com/oven-sh/bun/v1/laya",
                 "see https://example.com/docs", "http://localhost:5173/"):
        assert P.literal_api_hosts(text) == [] or "github" in text, text
    doc = _seed()
    doc["packages"]["laya-testing"]["body"] += "\n" + forms[0]
    errs = schema.check("packages", doc)
    assert any("$.packages.laya-testing.body" in e and HOST in e and "{tandem_url}" in e
               for e in errs), errs
    doc["packages"]["laya-testing"]["body"] = doc["packages"]["laya-testing"]["body"].replace(
        HOST, "{tandem_url}")
    assert not [e for e in schema.check("packages", doc) if "calls TandemLLM" in e]
    return f"{len(forms)} literal forms found and refused at save; the token passes"


def test_the_tokens():
    url = "http://127.0.0.1:8000/s/Ab3xYz9_kLmn"
    # in the project by default: opencode's sandbox asks before any path outside it, and its
    # `run` mode rejects the request (the live check of 2026-10-06 stopped at its first log)
    assert P.with_url("{tandem_url}/v1/laya/ask > {tandem_scratch}/run.log", url) == \
        f"{url}/v1/laya/ask > .tandem/run.log"
    assert P.with_url("{tandem_url}", None) == P.URL_DEFAULT
    old_s = os.environ.pop(P.SCRATCH_ENV, None)
    try:
        os.environ[P.SCRATCH_ENV] = "/tmp/tandem-{token}"
        assert P.with_url("{tandem_scratch}/a", url) == "/tmp/tandem-Ab3xYz9_kLmn/a"
        assert P.with_url("{tandem_scratch}/a", None) == "/tmp/tandem-session/a"
    finally:
        os.environ.pop(P.SCRATCH_ENV, None)
        if old_s is not None:
            os.environ[P.SCRATCH_ENV] = old_s
    hdr = {"Host": "spark:8000"}
    old = os.environ.pop(P.URL_ENV, None)
    try:
        assert P.request_url(hdr) == "http://spark:8000"
        os.environ[P.URL_ENV] = "http://192.0.2.7:8000/"
        assert P.request_url(hdr) == "http://192.0.2.7:8000"
    finally:
        os.environ.pop(P.URL_ENV, None)
        if old is not None:
            os.environ[P.URL_ENV] = old
    return "the request's address or QWEN38_TANDEM_URL; .tandem, or QWEN38_TANDEM_SCRATCH with {token}"


def test_the_shipped_texts_are_clean():
    files = glob.glob(os.path.join(ROOT, "packages", "**", "*.md"), recursive=True)
    for path in files:
        with open(path) as f:
            text = f.read()
        assert not re.search(r"(?<![\w{])\.tandem/", text), path
        assert not P.literal_api_hosts(text), (path, P.literal_api_hosts(text))
        assert M.migrate_text(path, text)[1] == [], (path, M.migrate_text(path, text)[1])
    with open(os.path.join(ROOT, "packages", "inline", "testing.md")) as f:
        t = f.read()
    assert M.SCRATCH_RULE in t and "tandem-test-hook:start" in t
    assert M.RUNTIME_RULE.replace(M.RUNTIME_OLD_PAGE, M.RUNTIME_NEW_PAGE) in t
    with open(os.path.join(ROOT, "packages", "inline", "laya-testing.md")) as f:
        assert M.FAIL_BULLET in f.read()
    return f"{len(files)} files: no .tandem/, no literal host, already migrated"


def test_a_box_document_migrates_clean():
    # the shape of the box's v7 (2026-10-03): the shipped texts with the literal host where the
    # token was, and the old `.tandem/` paths
    doc = _seed()
    for name in ("laya-testing", "laya-scripts", "laya-playtest", "test-web", "testing"):
        p = doc["packages"][name]
        body = p["body"].replace("{tandem_url}", HOST).replace("{tandem_scratch}/", ".tandem/")
        body = body.replace("mkdir -p {tandem_scratch}", "mkdir -p .tandem")
        p["body"] = body
    assert [e for e in schema.check("packages", doc) if "calls TandemLLM" in e]
    new, report = M.migrate_doc(doc)
    assert set(report) >= {"laya-testing", "laya-scripts", "laya-playtest", "test-web"}, report
    assert not [e for e in schema.check("packages", new) if "calls TandemLLM" in e]
    for name, p in new["packages"].items():
        if p.get("source") == "inline":
            assert HOST not in p["body"] and not re.search(r"(?<![\w{])\.tandem/", p["body"]), name
    assert M.migrate_doc(new)[1] == {}, "a second run changes nothing"
    return f"{len(report)} packages migrated; the save check passes; idempotent"


def test_the_helper_fails_fast():
    with open(os.path.join(ROOT, "packages", "inline", "laya-scripts.md")) as f:
        text = f.read()
    code = re.search(r"```python\n(# \{tandem_scratch\}/laya\.py.*?)```", text, re.S).group(1)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "laya.py")
        with open(path, "w") as f:
            f.write(P.with_url(code, "http://127.0.0.1:9"))
        r = subprocess.run([sys.executable, path, "Did it pass?", "pass,fail"], input="ok",
                           capture_output=True, text=True, timeout=30,
                           env={"PATH": os.environ.get("PATH", ""), "TANDEM_URL":
                                "http://127.0.0.1:9"})
    assert r.returncode == 3, (r.returncode, r.stderr[-400:])
    assert "LAYA unavailable" in r.stderr and "your code is not the cause" in r.stderr, r.stderr
    assert "Traceback" not in r.stderr, r.stderr
    return "exit 3, one truthful line, no traceback"


def test_the_kit_and_the_log_tail():
    from server.laya_kit import laya_drive
    assert os.path.dirname(laya_drive.LOG) == os.path.dirname(os.path.abspath(laya_drive.__file__))
    rx = re.compile(schema.OUTCOMES["patterns"]["log_tail"])
    assert rx.search("tail -n 60 /tmp/tandem-Ab3xYz9_kLmn/run.log")
    assert rx.search("tail -n 60 /tmp/tandem/build.log")
    assert rx.search("tail -n 60 ./.tandem/run.log"), "a project from before keeps working"
    assert not rx.search("tail -n 60 server.log")
    return "the kit's log beside it; scratch and old log tails both read"


def test_the_page_check_edits():
    # the box's v11 texts (0.3.43-test): the shipped texts with edits 9-11 undone
    doc = _seed()
    undo = ((M.CHECK_ONE_NEW, M.CHECK_ONE_OLD), (M.LAYA_RUN_NEW, M.LAYA_RUN_OLD),
            (M.PAGE_CMD_NEW, M.PAGE_CMD_OLD), (M.WEB_CMD_NEW, M.WEB_CMD_OLD),
            (M.VERIFY_NEW, M.VERIFY_OLD), (M.RUNTIME_NEW_PAGE, M.RUNTIME_OLD_PAGE), (M.WEB1_V3, M.WEB1_V2),
            (M.WEB3_NEW, M.WEB3_OLD), (M.WEB5_NEW, M.WEB5_OLD), (M.WEB8_V3, M.WEB8_V2), (M.PLAY_V3, M.PLAY_V2))
    for name in ("testing", "test-web", "laya-playtest", "laya-testing"):
        p = doc["packages"][name]
        for new, old in undo:
            p["body"] = p["body"].replace(new, old)
        assert "laya_drive.py page" not in p["body"], name
    new, report = M.migrate_doc(doc)
    assert set(report) == {"testing", "test-web", "laya-playtest", "laya-testing"}, report
    t = new["packages"]["testing"]["body"]
    assert "laya_drive.py page http://127.0.0.1:8765/index.html --url {tandem_url}" in t
    assert "page URL --url {tandem_url} --step" in new["packages"]["test-web"]["body"]
    assert "Never write a browser driver of your own (Marionette" in t and "NO BROWSER" in t
    assert "one page check (or test run) with every main control as a step" in t and "no test copies" in t
    assert t.count("one page check (or test run)") == 1
    lt = new["packages"]["laya-testing"]["body"]
    assert lt.count("write no Laya script of your own for it") == 1, lt
    assert "npx playwright" not in t and t.count("Never download a runtime") == 1, t
    w = new["packages"]["test-web"]["body"]
    assert "--step click:text=Start" in w and "needs the page check only" in w
    assert "needs only the page check" in new["packages"]["laya-playtest"]["body"]
    assert not [e for e in schema.check("packages", new) if "calls TandemLLM" in e]
    assert M.migrate_doc(new)[1] == {}, "a second run changes nothing"
    assert any(c.startswith(M.ROUND3) for ch in report.values() for c in ch) and len(M.NOTE_345) < 200
    # every page-check command in the texts parses with the kit's own command line
    from server.laya_kit import laya_drive
    import shlex
    cmds = re.findall(r"`(python3 \{tandem_scratch\}/laya_drive\.py page [^`]+)`", t)
    assert cmds, t
    for c in cmds:
        argv = shlex.split(c.replace("{tandem_scratch}", "/x"))[2:]
        ap_err = []
        try:
            laya_drive.main(argv + ["--timeout", "1", "--no-laya", "--cdp", "http://127.0.0.1:9"])
        except SystemExit as exc:          # argparse refuses with SystemExit(2) and a usage line
            ap_err.append(exc.code)
        assert ap_err in ([], [0], [1]), (c, ap_err)
    return f"{len(report)} packages migrated to the page check; {len(cmds)} command(s) parse; idempotent"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:50s} ok   {fn() or ''}")
            passed += 1
    print(f"{passed} passed")
