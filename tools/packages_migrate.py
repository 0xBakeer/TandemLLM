"""The package texts' migration of 0.3.41-test (SRV-163, SRV-171; the UT1 self-test, 2026-10-06).

One set of text edits, applied to the texts the repository ships (`--repo`), to a box's skill
copies (`--skills DIR`) and to a box's saved `packages` document (`--doc FILE`, or `--put URL`
through the config API, which stores it as a new version with a note; the old versions stay and
`restore` brings one back). The same function edits all of them, so the shipped texts and a box's
edited ones cannot drift apart.

The edits:

  1  SRV-163: a literal address in front of TandemLLM's own routes (`/v1/laya`, `/v1/library`,
     `/v1/memory`) becomes `{tandem_url}`, the address the client used (the box's saved packages
     had `https://ai-api.khaled.de`, which failed TLS wherever the private root CA is not trusted:
     42 % of a session went to certificate debugging).
  2  SRV-163: the Laya helper fails fast with a truthful message (the call failed on this
     machine, the code is not the cause, continue without the verdict) and exits 3; the curl
     example has a time limit; laya-testing says what to do when a call fails.
  3  SRV-171: `./.tandem/` and `.tandem/` (logs, the helper, the kit, pid files, a state file)
     become `{tandem_scratch}/`, the session's directory outside the project
     (`/tmp/tandem-<session token>`); the .gitignore and .dockerignore lines for `.tandem` go.
  4  SRV-171: the testing package says where scratch goes, that a state hook is scaffolding the
     model removes before "done" (unless the user asked for one), and that it never downloads a
     runtime or an executable.

Every edit is idempotent: a text already migrated comes out unchanged.

  python tools/packages_migrate.py --repo                 # rewrite packages/ in this checkout
  python tools/packages_migrate.py --skills DIR [--dry]   # SKILL.md files under DIR
  python tools/packages_migrate.py --doc 0007.json        # print what would change
  python tools/packages_migrate.py --put http://127.0.0.1:8000 --token-file secrets.env [--dry]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from server.conductor.packages import literal_api_hosts  # noqa: E402

NOTE = ("TandemLLM's routes via {tandem_url}; Laya helper fails fast; scratch in {tandem_scratch} "
        "outside the project; state hooks removed before done; no downloads")

# 2: the helper's call and its command line (after edit 3 renamed its path)
HELPER_OLD = '''    return json.load(urllib.request.urlopen(req, timeout=90))

if __name__ == "__main__":   # python3 {tandem_scratch}/laya.py "question" pass,fail < evidence.txt
    choices = sys.argv[2].split(",")'''
HELPER_NEW = '''    try:
        return json.load(urllib.request.urlopen(req, timeout=90))
    except (OSError, ValueError) as e:   # refused, TLS, timeout, not JSON: this machine's network
        print(f"LAYA unavailable ({type(e).__name__}: {e}). The call failed on this machine, "
              "your code is not the cause: do not debug the connection or certificates, "
              "continue with your own evidence and say Laya was unavailable.", file=sys.stderr)
        raise SystemExit(3)

if __name__ == "__main__":   # python3 {tandem_scratch}/laya.py "question" pass,fail < evidence.txt
    choices = sys.argv[2].split(",")'''
CURL_OLD = '''curl -s "${TANDEM_URL:-{tandem_url}}/v1/laya/ask"'''
CURL_NEW = '''curl -sS --max-time 60 "${TANDEM_URL:-{tandem_url}}/v1/laya/ask"'''
FAIL_ANCHOR = '''- failure, no or mismatch is a bug to explain before the next edit; `"sure": false` means read the evidence yourself.'''
FAIL_BULLET = ('''- A call that fails (connection refused, a TLS or certificate error, a timeout, a 5xx) is this '''
               '''machine's network, not your code: write "Laya unavailable" once and go on with your own '''
               '''evidence. At most one retry; never debug the transport (no openssl, no CA hunting, no -k).''')

# 4: the testing package's rules
LOGS_OLD = ('''- Logs: write build and run output to {tandem_scratch}/build.log and {tandem_scratch}/run.log. '''
            '''After a run, `tail -n 60` the log in a tool call.''')
SCRATCH_RULE = ('''- Scratch: logs, helper scripts and kits go to {tandem_scratch}/ (`mkdir -p {tandem_scratch}` '''
                '''first), outside the project. The project gets only what the user asked for; never list '''
                '''scratch files as deliverables.''')
STATE_OLD = ('''Keep it under 4 KB. Read it after real input, never before.''')
STATE_NEW = ('''Keep it under 4 KB. Read it after real input, never before. It is test scaffolding: wrap it in '''
             '''`tandem-test-hook:start` and `tandem-test-hook:end` comments and remove it as your last edit '''
             '''before "done" (then run the quick check once more), unless the user asked for a debug or '''
             '''state API; say in your final message that you removed it.''')
TESTS_ANCHOR = '''- Tests: run the real test command and let its summary line appear in the tool result.'''
RUNTIME_RULE = ('''- Runtimes: use what is installed (`command -v node npx python3 bun deno`; for a page, the '''
                '''browser tool or `npx playwright`). Never download a runtime or any executable (no curl or '''
                '''wget of a binary, no install script) unless the user asks; if nothing can run the code, '''
                '''say so and test what you can.''')


def _once(text: str, anchor: str, add: str, changes: list, what: str) -> str:
    """`add` on its own line after `anchor`, unless it is there already."""
    if add in text or anchor not in text:
        return text
    changes.append(what)
    return text.replace(anchor, anchor + "\n" + add, 1)


def migrate_text(name: str, text: str) -> tuple[str, list[str]]:
    """(the text with the edits, what changed). Idempotent."""
    changes: list[str] = []
    out = text
    # 1: literal addresses of TandemLLM's routes
    for host in literal_api_hosts(out):
        out = re.sub(re.escape(host) + r"(?=\}?[\"']?\s*\)?\s*\+?\s*[\"']?/(?:s/[A-Za-z0-9_-]+/)?"
                     r"v1/(?:laya|library|memory)\b)", "{tandem_url}", out)
        changes.append(f"host {host} -> {{tandem_url}}")
    # 3: the scratch directory (the ignore lines first: they name the directory, not a file in it)
    for old, new in (("\\n.env\\n.tandem/\\n'", "\\n.env\\n'"), (", `.env`, `.tandem`.", ", `.env`.")):
        if old in out:
            out = out.replace(old, new)
            changes.append("ignore line for .tandem removed")
    n0 = out
    out = re.sub(r"mkdir -p (?:\./)?\.tandem\b/?", "mkdir -p {tandem_scratch}", out)
    out = out.replace('sys.path.insert(0, ".tandem")', 'sys.path.insert(0, "{tandem_scratch}")')
    out = re.sub(r"(?<![\w/{])\./\.tandem/", "{tandem_scratch}/", out)
    out = re.sub(r"(?<![\w/{.])\.tandem/", "{tandem_scratch}/", out)
    if out != n0:
        changes.append(".tandem/ -> {tandem_scratch}/")
    # 2: Laya fails fast and truthfully
    if HELPER_OLD in out:
        out = out.replace(HELPER_OLD, HELPER_NEW)
        changes.append("laya helper fails fast (exit 3)")
    if CURL_OLD in out:
        out = out.replace(CURL_OLD, CURL_NEW)
        changes.append("curl example has a time limit")
    out = _once(out, FAIL_ANCHOR, FAIL_BULLET, changes, "laya-testing: what to do when a call fails")
    # 4: the testing package's rules
    if LOGS_OLD in out and SCRATCH_RULE not in out:
        out = out.replace(LOGS_OLD, SCRATCH_RULE + "\n" + LOGS_OLD, 1)
        changes.append("testing: the scratch rule")
    if STATE_OLD in out and STATE_NEW not in out:
        out = out.replace(STATE_OLD, STATE_NEW, 1)
        changes.append("testing: the state hook is scaffolding")
    out = _once(out, TESTS_ANCHOR, RUNTIME_RULE, changes, "testing: no downloads")
    return out, changes


def migrate_doc(doc: dict) -> tuple[dict, dict]:
    """(the packages document with every inline body migrated, {package: changes})."""
    new = json.loads(json.dumps(doc))
    report = {}
    for name, p in (new.get("packages") or {}).items():
        if p.get("source", "inline") != "inline" or not isinstance(p.get("body"), str):
            continue
        body, ch = migrate_text(name, p["body"])
        if ch:
            p["body"], report[name] = body, ch
    return new, report


def _files(root: str) -> list[str]:
    return sorted(glob.glob(os.path.join(root, "**", "*.md"), recursive=True))


def migrate_files(root: str, dry: bool) -> dict:
    report = {}
    for path in _files(root):
        with open(path) as f:
            text = f.read()
        out, ch = migrate_text(os.path.basename(os.path.dirname(path)), text)
        if ch:
            report[os.path.relpath(path, root)] = ch
            if not dry:
                with open(path, "w") as f:
                    f.write(out)
    return report


def _token(path: str) -> str:
    with open(os.path.expanduser(path)) as f:
        for line in f:
            if line.startswith("QSE_ADMIN_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit(f"no QSE_ADMIN_TOKEN in {path}")


def _call(method: str, url: str, token: str, body=None):
    req = urllib.request.Request(url, method=method, data=json.dumps(body).encode()
                                 if body is not None else None,
                                 headers={"Authorization": f"Bearer {token}",
                                          "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as exc:            # the server's reason, not a traceback
        raise SystemExit(f"[packages_migrate] {method} {url}: {exc.code} {exc.read()[:400]!r}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", action="store_true", help="rewrite this checkout's packages/")
    ap.add_argument("--skills", help="rewrite the SKILL.md (and .md) files under this directory")
    ap.add_argument("--doc", help="a packages version file or document: print what would change")
    ap.add_argument("--put", help="a server's address: migrate its current packages document")
    ap.add_argument("--token-file", default="~/.qwen38-spark-engine/secrets.env")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args(argv)
    if a.repo:
        rep = migrate_files(os.path.join(ROOT, "packages"), a.dry)
        print(json.dumps(rep, indent=1))
    if a.skills:
        rep = migrate_files(os.path.expanduser(a.skills), a.dry)
        print(json.dumps(rep, indent=1))
    if a.doc:
        with open(a.doc) as f:
            d = json.load(f)
        _, rep = migrate_doc(d.get("doc", d))
        print(json.dumps(rep, indent=1))
    if a.put:
        tok = _token(a.token_file)
        cur = _call("GET", a.put.rstrip("/") + "/v1/conductor/packages", tok)
        new, rep = migrate_doc(cur["doc"])
        print(json.dumps(rep, indent=1))
        left = {n: literal_api_hosts(p.get("body") or "") for n, p in new["packages"].items()
                if literal_api_hosts(p.get("body") or "")}
        if left:
            raise SystemExit(f"literal hosts left: {left}")
        if not rep:
            print("[packages_migrate] nothing to change")
            return 0
        if a.dry:
            print(f"[packages_migrate] dry run: v{cur.get('version')} would get a new version")
            return 0
        got = _call("PUT", a.put.rstrip("/") + "/v1/conductor/packages", tok,
                    {"doc": new, "note": NOTE, "base_version": cur.get("version")})
        print(f"[packages_migrate] stored packages v{got.get('version')} (was v{cur.get('version')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
