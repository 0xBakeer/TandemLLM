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
     become `{tandem_scratch}/` (`.tandem` in the project by default, QWEN38_TANDEM_SCRATCH);
     the testing package has the model exclude it from git and delete it before "done"; the
     .gitignore and .dockerignore lines for `.tandem` go (the exclude does it per clone).
  4  SRV-171: the testing package says where scratch goes, that a state hook is scaffolding the
     model removes before "done" (unless the user asked for one), and that it never downloads a
     runtime or an executable.

The edits of 0.3.43-test (SRV-179, SRV-181; the UT2 self-test, 2026-10-06: an Open WebUI agent tuned
its own playtest bot for 18 minutes after the game worked and never answered; an opencode page was
written at 642 s of 733 s, after a Firefox rig and three hung servers):

  5  testing: "Order and budget": a scratch directory left by a cut run is cleaned first, the
     deliverable before any test tooling, a tool that fails twice is dropped, servers in the background with a time limit and killed before "done",
     what is enough, never change the product for one's own test, the final report.
  6  laya-playtest: a playtest when the kit runs here, a check and not a deliverable; a spot only
     the bot fails is not a bug; two runs ending at the same spot end the playtest.
  7  test-web: write the page first, the server rule, and the same stop rule for the bot.
  8  testing: everything the model creates stays inside the project, browser profiles included,
     never /tmp (opencode's run mode refuses a path outside the project and ends the run).

The edits of 0.3.45-test (SRV-195; FX2's live check, 2026-10-06: opencode wrote a pomodoro page at
176 s, found no node, npx or Chrome on its PATH, only Firefox, and spent seven minutes on a
Marionette driver of its own until the run's limit; the kit's headless Chrome was there all along):

  9  testing: a page is checked with the kit's page check (`laya_drive.py page`: its own headless
     Chrome, real clicks and keys, values read back, screenshots at 1280 and 390 px, console
     errors, Laya's verdict, Chrome closed, about 5 s); never a browser driver of one's own;
     NO BROWSER or two failed checks = read the code and say so; after the deliverable exists the
     verification is check, fix, re-check, then the report.
 10  test-web: the browser tool or the page check for every step (the same steps as `--step`).
 11  test-web and laya-playtest: a game, or a flow of many steps toward a goal, is played by Laya;
     a one-screen tool (a timer, a form, a calculator) needs the page check only.
 12  laya-testing and testing (the live n=1 of edits 9-11: 473 s, of which ~280 s checks: four Laya
     scripts of its own and a helper beside the page check, a 3-second copy of the page): the page
     check's `laya` field is Laya's verdict for a page, no script of one's own; one page check
     with every main control as a step; no test copies of the product.
 13  testing and test-web: the page-check commands carry `--url {tandem_url}`: on Open WebUI's
     terminal host the kit's default (http://127.0.0.1:8000) is not TandemLLM and Laya's verdict
     came back 404 (FX3's live chat, 2026-10-06).

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

NOTE = ("TandemLLM's routes via {tandem_url}; Laya helper fails fast; scratch {tandem_scratch} excluded "
        "from git and deleted before done; state hooks removed; no downloads")
# the store keeps a version note under 200 characters; a box that had the first edits gets this one
ROUND2 = ("testing: order", "testing: nothing outside", "laya-playtest: a check", "laya-playtest: the stop",
          "test-web: the")
NOTE_343 = ("0.3.43-test: deliverable first, a test budget, servers in the background and stopped, a cut "
            "run's scratch cleaned, the playtest stop rule (SRV-179, SRV-181)")
ROUND3 = ("testing: the kit's page check", "testing: a page runs", "test-web: the browser tool or",
          "test-web: real input", "test-web: the page check", "test-web: Laya plays", "laya-playtest: a one-screen",
          "laya-testing: the page check", "testing: one page check", "testing: the page check asks",
          "test-web: the page check asks")
NOTE_345 = ("0.3.45-test: a page is checked with the kit's page check (its own headless Chrome, Laya at "
            "{tandem_url}), never a driver of one's own; one check, fix, check, report (SRV-195)")

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
SCRATCH_RULE_V1 = ("- Scratch: logs, helper scripts and kits go to {tandem_scratch}/ (`mkdir -p {tandem_scratch}` "
                   "first), outside the project. The project gets only what the user asked for; never list "
                   "scratch files as deliverables.")
SCRATCH_RULE = ('''- Scratch: logs, helper scripts and kits go to {tandem_scratch}/. Create it once with '''
                '''`mkdir -p {tandem_scratch} && (git rev-parse --git-dir >/dev/null 2>&1 && echo '{tandem_scratch}/' >> '''
                '''"$(git rev-parse --git-dir)/info/exclude"; true)`, so git never lists it, and delete it (`rm -rf '''
                '''{tandem_scratch}`) as your last step before "done". The project gets only what the user asked for; '''
                '''never list scratch files as deliverables.''')
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


# 5: the testing package's order and budget
BUDGET_ANCHOR = "Never test by writing to the state you then read. Input in, state out."
BUDGET_HEAD = "Order and budget (the user waits for the deliverable"
BUDGET = """
Order and budget (the user waits for the deliverable, not for your test rig):
- A {tandem_scratch} left by an earlier run that was cut off: stop the server its server.pid names (`kill $(cat {tandem_scratch}/server.pid) 2>/dev/null`) and delete it before you start.
- First the deliverable: write the files the user asked for before any test tooling (test pages, servers, browser rigs, installs).
- Verify with what already works here: your browser tool, else `npx playwright` or the Laya kit when `command -v` finds them; else read the code and say which check was not possible. A tool that fails twice (a browser that does not start, an image you cannot read, a server that hangs) is dropped: say so in one sentence and go on.
- A server you start runs in the background with a time limit and its pid in scratch: `timeout 600 python3 -m http.server 8765 >{tandem_scratch}/server.log 2>&1 & echo $! > {tandem_scratch}/server.pid`. Stop it (`kill $(cat {tandem_scratch}/server.pid)`) before "done"; never run a server in the foreground.
- Enough is: the deliverable loads or runs, zero console errors, and one run with real input passes (a game: start, move, jump; a page: the main controls; code: the tests). A bot that plays a whole game is optional. When your own test tool fails the same way twice, stop tuning it: it is not the deliverable; report that spot as untested.
- Never change the product only to make your own test pass; if a test made you change its behaviour, say so in the final message.
- Finish with what you built and where it is (paths), what you verified and how, and what is untested."""
# 8: nothing outside the project (opencode's run mode refuses a path outside it and ends the run:
# a Firefox profile in /tmp ended one at 349 s)
TMP_ANCHOR = ("- A server you start runs in the background with a time limit and its pid in scratch: "
              "`timeout 600 python3 -m http.server 8765 >{tandem_scratch}/server.log 2>&1 & echo $! > "
              "{tandem_scratch}/server.pid`. Stop it (`kill $(cat {tandem_scratch}/server.pid)`) before "
              "\"done\"; never run a server in the foreground.")
TMP_RULE = ("- Everything you create stays inside the project: temporary files, browser profiles and logs "
            "go in {tandem_scratch}, never /tmp (a client may refuse a path outside the project and end "
            "the run).")
# 9: the kit's page check is the browser here (SRV-195)
VERIFY_OLD = ("- Verify with what already works here: your browser tool, else `npx playwright` or the Laya "
              "kit when `command -v` finds them; else read the code and say which check was not possible. "
              "A tool that fails twice (a browser that does not start, an image you cannot read, a server "
              "that hangs) is dropped: say so in one sentence and go on.")
VERIFY_NEW = ("- Check a page with the kit's page check, the browser here: `mkdir -p {tandem_scratch} && curl -sf "
              "{tandem_url}/v1/laya/kit/laya_drive.py -o {tandem_scratch}/laya_drive.py`, then `python3 "
              "{tandem_scratch}/laya_drive.py page http://127.0.0.1:8765/index.html --step click:#start "
              "--step wait:2000 --step \"read:document.querySelector('#time').textContent\"` (steps: "
              "click:SELECTOR or click:text=Label, key:Space*300, type:#field=text, wait:MS, read:JS, "
              "expect:JS, shot:NAME). It starts its own headless Chrome, uses real clicks and keys, "
              "reads values back, takes screenshots at 1280 and 390 px, lists the console errors, asks "
              "Laya for a verdict and closes Chrome, in about 5 s. Your browser tool, when you have one, "
              "does the same.\n"
              "- Never write a browser driver of your own (Marionette, WebDriver, Selenium, a DevTools "
              "client, `firefox --screenshot`). When the page check says NO BROWSER, or fails twice, "
              "read the code instead and say the browser check was not possible. Any other tool that "
              "fails twice (an image you cannot read, a server that hangs) is dropped the same way: say "
              "so in one sentence and go on.\n"
              "- After the deliverable exists, verification is: check, fix what the check shows, check "
              "again, then the final message. Write no new test tooling after the first check; the user "
              "waits for every extra script.")
RUNTIME_OLD_PAGE = "for a page, the browser tool or `npx playwright`"
RUNTIME_NEW_PAGE = "for a page, the kit's page check below or your browser tool"
# 10: test-web, the steps through the browser tool or the page check
WEB1_V2 = ("1. Write the page first. Serve it in the background with a time limit (the testing package's "
           "server rule), open it in the browser tool; zero console errors first.")
WEB1_V3 = ("1. Write the page first. Serve it in the background with a time limit (the testing package's "
           "server rule). Open it in your browser tool, else run the kit's page check (`python3 "
           "{tandem_scratch}/laya_drive.py page URL --step ...`, the testing package); zero console "
           "errors first.")
WEB3_OLD = ("3. Real input only: press_key with a hold time, clicks at screenshot coordinates. Wait 300 ms "
            "or more, then read window.__state() with evaluate_script. Never set keys, flags or positions "
            "in a script.")
WEB3_NEW = ("3. Real input only: press_key with a hold time, clicks at screenshot coordinates (page check: "
            "`--step key:ArrowRight*300`, `--step click:#start` or `--step click:text=Start`). Wait 300 ms "
            "or more, then read window.__state() (evaluate_script, or `--step \"read:window.__state()\"`). "
            "Never set keys, flags or positions in a script.")
WEB5_OLD = "5. Screenshots at 1280 and 390 px: start, middle, end."
WEB5_NEW = ("5. Screenshots at 1280 and 390 px: start, middle, end (the page check takes 1280 and 390 at "
            "the end; `--step shot:middle` for more).")
# 11: Laya plays games and long flows; a one-screen tool needs the page check only
WEB8_V2 = ("8. A game or a UI flow: let Laya play it, 3 episodes or more (laya-playtest: "
           "`{tandem_scratch}/laya_drive.py browser`); a death or stuck spot that real keys cannot avoid "
           "either is a bug. Two runs that end at the same spot: stop and report it as untested.")
WEB8_V3 = ("8. A game, or a flow of many steps toward a goal: let Laya play it, 3 episodes or more "
           "(laya-playtest: `{tandem_scratch}/laya_drive.py browser`); a death or stuck spot that real "
           "keys cannot avoid either is a bug. Two runs that end at the same spot: stop and report it as "
           "untested. A one-screen tool (a timer, a form, a calculator) needs the page check only.")
PLAY_V2 = ('Playtest a game, UI flow, CLI or API flow you build this way when the kit runs here (a check, '
           'not a deliverable; the testing package\'s budget applies):')
PLAY_V3 = ('Playtest a game, or a UI, CLI or API flow of many steps toward a goal, this way when the kit runs '
           'here (a check, not a deliverable; the testing package\'s budget applies). A one-screen tool '
           '(a timer, a form, a calculator) needs only the page check (`laya_drive.py page`):')

# 12: one page check is the whole check of a page (the live n=1 of edits 9-11)
LAYA_RUN_OLD = ("- After every run that produced evidence (a log, an output, the console, a state dump), your "
                "test script asks Laya and prints the verdict.")
LAYA_RUN_NEW = ("- After every run that produced evidence (a log, an output, the console, a state dump), your "
                "test script asks Laya and prints the verdict. A page checked with the kit's page check has it "
                "already (the report's `laya` field): quote it, write no Laya script of your own for it.")
CHECK_ONE_OLD = ("- After the deliverable exists, verification is: check, fix what the check shows, check again, "
                 "then the final message. Write no new test tooling after the first check; the user waits for "
                 "every extra script.")
CHECK_ONE_NEW = ("- After the deliverable exists, verification is: one page check (or test run) with every main "
                 "control as a step in the same call, fix what it shows, check again, then the final message, "
                 "in a few minutes. Write no new test tooling after the first check and no test copies of the "
                 "product (a page with shorter timers, a patched build); say what a long wait would show "
                 "instead. The user waits for every extra script.")

# 13: the page check asks Laya where the kit came from
PAGE_CMD_OLD = "{tandem_scratch}/laya_drive.py page http://127.0.0.1:8765/index.html --step"
PAGE_CMD_NEW = "{tandem_scratch}/laya_drive.py page http://127.0.0.1:8765/index.html --url {tandem_url} --step"
WEB_CMD_OLD = "(`python3 {tandem_scratch}/laya_drive.py page URL --step ...`, the testing package)"
WEB_CMD_NEW = "(`python3 {tandem_scratch}/laya_drive.py page URL --url {tandem_url} --step ...`, the testing package)"

# 6: laya-playtest's stop rule
PLAY_OLD = 'Playtest every game, UI flow, CLI or API flow you build this way before "done":'
PLAY_NEW = ('Playtest a game, UI flow, CLI or API flow you build this way when the kit runs here (a check, '
            'not a deliverable; the testing package\'s budget applies):')
REPORT_OLD = ("4. Report: deaths with place, cause and the decisions before them; stuck spots. An unavoidable "
              "death, a stuck spot or a crash is a bug: fix, rerun, quote the report.")
REPORT_NEW = ("4. Report: deaths with place, cause and the decisions before them; stuck spots. A crash, or a "
              "death or stuck spot that a player with real keys cannot avoid either, is a bug: fix the build, "
              "rerun, quote the report. A spot where only the bot fails is not a bug: report it. Two runs that "
              "end at the same spot with no change to the build: stop playtesting and report what works and "
              "what is untested.")
# 7: test-web
WEB1_OLD = "1. Serve it, open it in the browser tool; zero console errors first."
WEB1_NEW = ("1. Write the page first. Serve it in the background with a time limit (the testing package's "
            "server rule), open it in the browser tool; zero console errors first.")
WEB8_OLD = ("8. A game or a UI flow: let Laya play it, 3 episodes or more (laya-playtest: "
            "`{tandem_scratch}/laya_drive.py browser`); every death it could not avoid and every stuck spot "
            "is a bug.")
WEB8_NEW = ("8. A game or a UI flow: let Laya play it, 3 episodes or more (laya-playtest: "
            "`{tandem_scratch}/laya_drive.py browser`); a death or stuck spot that real keys cannot avoid "
            "either is a bug. Two runs that end at the same spot: stop and report it as untested.")


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
    # 13: the same host handed to the kit (`--url HOST`, Laya's address for the page check)
    for host in set(re.findall(r"(?<=--url )(https?://[^\s`'\"]+)", out)) & set(literal_api_hosts(text) or ()):
        out = out.replace("--url " + host, "--url {tandem_url}")
        changes.append(f"--url {host} -> {{tandem_url}}")
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
    if SCRATCH_RULE_V1 in out:                       # the first wording (outside the project)
        out = out.replace(SCRATCH_RULE_V1, SCRATCH_RULE)
        changes.append("testing: scratch inside the project, excluded and deleted")
    if LOGS_OLD in out and SCRATCH_RULE not in out:
        out = out.replace(LOGS_OLD, SCRATCH_RULE + "\n" + LOGS_OLD, 1)
        changes.append("testing: the scratch rule")
    if STATE_OLD in out and STATE_NEW not in out:
        out = out.replace(STATE_OLD, STATE_NEW, 1)
        changes.append("testing: the state hook is scaffolding")
    if "Never download a runtime or any executable" not in out:    # edit 9 rewords its page part
        out = _once(out, TESTS_ANCHOR, RUNTIME_RULE, changes, "testing: no downloads")
    # 5, 6, 7: the round-2 budget (SRV-179, SRV-181)
    if BUDGET_HEAD not in out:                   # the block may have grown since (edit 8)
        out = _once(out, BUDGET_ANCHOR, BUDGET, changes, "testing: order and budget")
    out = _once(out, TMP_ANCHOR, TMP_RULE, changes, "testing: nothing outside the project")
    for old, new, what in ((PLAY_OLD, PLAY_NEW, "laya-playtest: a check, not a deliverable"),
                           (REPORT_OLD, REPORT_NEW, "laya-playtest: the stop rule"),
                           (WEB1_OLD, WEB1_NEW, "test-web: the page first, the server rule"),
                           (WEB8_OLD, WEB8_NEW, "test-web: the stop rule")):
        if old in out:
            out = out.replace(old, new, 1)
            changes.append(what)
    # 6 and 7 on a box's edited wording (the design-review edit: "UI flow with steps", a sentence
    # after the first one)
    for rx, rep, what in (
            (r'Playtest every game, (.*?) you build this way before "done"([.:])',
             r"Playtest a game, \1 you build this way when the kit runs here (a check, not a "
             r"deliverable; the testing package's budget applies)\2",
             "laya-playtest: a check, not a deliverable"),
            (r"(8\. A game or a UI flow[^\n]*?); every death it could not avoid and every stuck "
             r"spot is a bug\.",
             r"\1; a death or stuck spot that real keys cannot avoid either is a bug. Two runs "
             r"that end at the same spot: stop and report it as untested.",
             "test-web: the stop rule")):
        got = re.sub(rx, rep, out, count=1)
        if got != out:
            out = got
            changes.append(what)
    # 9, 10, 11: the kit's page check (SRV-195)
    for old, new, what in ((VERIFY_OLD, VERIFY_NEW, "testing: the kit's page check, no own driver, check-fix-check"),
                           (RUNTIME_OLD_PAGE, RUNTIME_NEW_PAGE, "testing: a page runs in the page check"),
                           (WEB1_V2, WEB1_V3, "test-web: the browser tool or the page check"),
                           (WEB3_OLD, WEB3_NEW, "test-web: real input through the page check"),
                           (WEB5_OLD, WEB5_NEW, "test-web: the page check's screenshots"),
                           (WEB8_V2, WEB8_V3, "test-web: Laya plays games and long flows"),
                           (PLAY_V2, PLAY_V3, "laya-playtest: a one-screen tool needs the page check"),
                           (LAYA_RUN_OLD, LAYA_RUN_NEW, "laya-testing: the page check carries Laya's verdict"),
                           (CHECK_ONE_OLD, CHECK_ONE_NEW, "testing: one page check with every control, no test copies"),
                           (PAGE_CMD_OLD, PAGE_CMD_NEW, "testing: the page check asks Laya at {tandem_url}"),
                           (WEB_CMD_OLD, WEB_CMD_NEW, "test-web: the page check asks Laya at {tandem_url}")):
        if old in out and new not in out:          # 12's new text begins with its old one
            out = out.replace(old, new, 1)
            changes.append(what)
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
    """The admin password from the secrets file; "" when none is set (the admin routes are open
    then, VIS-56)."""
    try:
        with open(os.path.expanduser(path)) as f:
            for line in f:
                if line.startswith("QSE_ADMIN_TOKEN="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        pass
    return ""


def _call(method: str, url: str, token: str, body=None):
    req = urllib.request.Request(url, method=method, data=json.dumps(body).encode()
                                 if body is not None else None,
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {token}"} if token else {})})
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
                    {"doc": new, "note": NOTE_345 if any(
                        c.startswith(ROUND3) for ch in rep.values() for c in ch) else NOTE_343 if any(
                        c.startswith(ROUND2) for ch in rep.values() for c in ch) else NOTE,
                     "base_version": cur.get("version")})
        print(f"[packages_migrate] stored packages v{got.get('version')} (was v{cur.get('version')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
