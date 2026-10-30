"""tools/release.sh (OPS-223) on a scratch repository with a local bare remote: no network, no GitHub, no
box. bump writes the version into the pointer lines and leaves the prose; preflight passes on the bumped
tree and names what is missing on the old one; a dry run prints the tag and push commands and changes no
ref; the milestone section closes vX.Y.Z and opens the next patch through a stand-in gh. Under
/bin/bash where it is bash 3.2 (macOS), else the bash on PATH.

Run: python tests/test_release_script.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "tools", "release.sh")
BASH = "/bin/bash" if os.path.exists("/bin/bash") else (shutil.which("bash") or "bash")
ID = ["-c", "user.name=Release Test", "-c", "user.email=release@example.invalid"]

FILES = {
    "VERSION": "1.2.3\n",
    "install.sh": '#!/bin/bash\n# The engine release. v1.2.3 adds a thing.\nTANDEM_REF="${TANDEM_REF:-v1.2.3}"\n',
    "README.md": ("curl -fsSLO https://raw.githubusercontent.com/o/r/v1.2.3/install.sh\n\n"
                  "It clones the engine at the tag `v1.2.3` into `~/T/src`, via install.sh.\n\n"
                  "Where the tag `v1.2.3` is missing, the install.sh step stops.\n\n"
                  "| `--ref REF` | the engine version, default `v1.2.3`; |\n\n"
                  "The current release is `v1.2.3`. `v1.2.2` fixed install.sh.\n"),
    "docs/setup.md": "curl -fsSLO https://raw.githubusercontent.com/o/r/v1.2.3/install.sh\nthe tag `v1.2.3`.\n",
    "manual/preamble.tex": "\\newcommand{\\version}{1.2.3}\n",
    "manual/chapters/install-engine.tex": "curl -fsSLO https://raw.githubusercontent.com/o/r/v1.2.3/install.sh\n",
    "manual/chapters/troubleshooting.tex": "no tag, branch or commit 'v1.2.3'; \\code{v1.2.3} by default\n",
    "manual/generated/install-flags.tex": "\\texttt{-{}-ref REF} & the engine release (default v1.2.3) \\\\\n",
    "manual/chapters/changelog.tex": "1.2.3 & 10-20 & a thing \\\\\n\\section{1.2.3, a thing}\n",
    "CHANGELOG.md": "# Changelog\n\n## 1.2.3, 2026-10-20\n\nA thing.\n",
}


def _git(cwd, *args, check=True):
    return subprocess.run(["git", *ID, *args], cwd=cwd, check=check, capture_output=True, text=True)


def _repo():
    d = tempfile.mkdtemp(prefix="release-test-")
    remote, work = os.path.join(d, "remote.git"), os.path.join(d, "work")
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", remote], check=True)
    os.makedirs(work)
    _git(work, "init", "-q", "-b", "main")
    for rel, text in FILES.items():
        os.makedirs(os.path.dirname(os.path.join(work, rel)) or work, exist_ok=True)
        with open(os.path.join(work, rel), "w") as fh:
            fh.write(text)
    os.makedirs(os.path.join(work, "tools"))
    shutil.copy(SCRIPT, os.path.join(work, "tools", "release.sh"))
    _git(work, "add", "-A")
    _git(work, "commit", "-q", "-m", "1.2.3")
    _git(work, "tag", "-a", "v1.2.3", "-m", "TandemLLM 1.2.3")
    _git(work, "remote", "add", "origin", remote)
    _git(work, "push", "-q", "origin", "main", "v1.2.3")
    return d, work


def _release(work, *args):
    env = dict(os.environ, RELEASE_NOTES_CHECK="", RELEASE_NOTES_DENY="", GIT_CONFIG_COUNT="2",
               GIT_CONFIG_KEY_0="user.name", GIT_CONFIG_VALUE_0="Release Test",
               GIT_CONFIG_KEY_1="user.email", GIT_CONFIG_VALUE_1="release@example.invalid")
    return subprocess.run([BASH, os.path.join(work, "tools", "release.sh"), *args, "--repo", "o/r"],
                          cwd=work, capture_output=True, text=True, env=env, timeout=120)


def _read(work, rel):
    with open(os.path.join(work, rel)) as fh:
        return fh.read()


def test_bump_then_preflight_then_a_dry_run():
    d, work = _repo()
    try:
        r = _release(work, "1.2.4", "--only", "preflight")
        assert r.returncode != 0, r.stdout
        assert "FAIL  VERSION says 1.2.3" in r.stdout and "still points at v1.2.3" in r.stdout, r.stdout

        r = _release(work, "1.2.4", "--only", "bump")
        assert r.returncode == 0, r.stdout + r.stderr
        assert _read(work, "VERSION") == "1.2.4\n"
        inst = _read(work, "install.sh")
        assert 'TANDEM_REF="${TANDEM_REF:-v1.2.4}"' in inst and "# The engine release. v1.2.3" in inst, inst
        readme = _read(work, "README.md")
        assert "/v1.2.4/install.sh" in readme and "default `v1.2.4`" in readme, readme
        assert "at the tag `v1.2.4`" in readme and "Where the tag `v1.2.4`" in readme, readme
        assert "The current release is `v1.2.3`. `v1.2.2` fixed install.sh." in readme, "the paragraph is prose"
        assert "\\version}{1.2.4}" in _read(work, "manual/preamble.tex")
        assert "v1.2.3" not in _read(work, "manual/chapters/troubleshooting.tex")
        assert "(default v1.2.4)" in _read(work, "manual/generated/install-flags.tex")

        # the author's part: the release paragraph, the changelogs; then the release commit
        with open(os.path.join(work, "README.md"), "w") as fh:
            fh.write(readme.replace("The current release is `v1.2.3`.", "The current release is `v1.2.4`. In `v1.2.3` a thing."))
        with open(os.path.join(work, "CHANGELOG.md"), "w") as fh:
            fh.write("# Changelog\n\n## 1.2.4, 2026-10-22\n\nAnother thing.\n\n## 1.2.3, 2026-10-20\n\nA thing.\n")
        with open(os.path.join(work, "manual/chapters/changelog.tex"), "a") as fh:
            fh.write("1.2.4 & 10-22 & another thing \\\\\n\\section{1.2.4, another thing}\n")
        notes = os.path.join(d, "notes.md")
        with open(notes, "w") as fh:
            fh.write("Release notes.\n")
        _git(work, "add", "-A")
        _git(work, "commit", "-q", "-m", "Release 1.2.4")

        r = _release(work, "1.2.4", "--only", "preflight", "--notes", notes)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "PREFLIGHT: all checks pass" in r.stdout and "previous release v1.2.3" in r.stdout, r.stdout

        head = _git(work, "rev-parse", "HEAD").stdout.strip()
        r = _release(work, "1.2.4", "--only", "preflight,tag,push", "--notes", notes, "--dry-run")
        assert r.returncode == 0, r.stdout + r.stderr
        assert "+ git -C" in r.stdout and "tag -a v1.2.4 -m TandemLLM 1.2.4 HEAD" in r.stdout, r.stdout
        assert "push origin HEAD:refs/heads/main refs/tags/v1.2.4:refs/tags/v1.2.4" in r.stdout, r.stdout
        assert _git(work, "tag", "-l", "v1.2.4").stdout.strip() == "", "a dry run made the tag"
        assert _git(work, "ls-remote", "origin", "main").stdout.split()[0] != head, "a dry run pushed"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_a_trailer_stops_the_preflight():
    d, work = _repo()
    try:
        _release(work, "1.2.4", "--only", "bump")
        with open(os.path.join(work, "CHANGELOG.md"), "w") as fh:
            fh.write("# Changelog\n\n## 1.2.4, 2026-10-22\n\nAnother thing.\n")
        readme = _read(work, "README.md").replace("current release is `v1.2.3`", "current release is `v1.2.4`")
        with open(os.path.join(work, "README.md"), "w") as fh:
            fh.write(readme)
        with open(os.path.join(work, "manual/chapters/changelog.tex"), "a") as fh:
            fh.write("1.2.4 & 10-22 & x \\\\\n\\section{1.2.4, x}\n")
        _git(work, "add", "-A")
        _git(work, "commit", "-q", "-m", "Release 1.2.4\n\nCo-Authored-By: Someone <someone@example.invalid>")
        r = _release(work, "1.2.4", "--only", "preflight")
        assert r.returncode != 0 and "carries a co-author or session line" in r.stdout, r.stdout
    finally:
        shutil.rmtree(d, ignore_errors=True)


FAKE_GH = """#!/bin/sh
# a stand-in for gh: the milestone list from $FAKE_GH_LIST, every call logged to $FAKE_GH_LOG
echo "$*" >> "$FAKE_GH_LOG"
case "$*" in
    *"milestones?state=all"*) cat "$FAKE_GH_LIST" ;;
esac
"""


def _milestone_run(work, listing, *args):
    d = os.path.dirname(work)
    bin_dir, log, lst = os.path.join(d, "bin"), os.path.join(d, "gh.log"), os.path.join(d, "milestones")
    os.makedirs(bin_dir, exist_ok=True)
    with open(os.path.join(bin_dir, "gh"), "w") as fh:
        fh.write(FAKE_GH)
    os.chmod(os.path.join(bin_dir, "gh"), 0o755)
    with open(lst, "w") as fh:
        fh.write(listing)
    open(log, "w").close()
    env = dict(os.environ, PATH=bin_dir + os.pathsep + os.environ["PATH"], FAKE_GH_LOG=log, FAKE_GH_LIST=lst)
    r = subprocess.run([BASH, os.path.join(work, "tools", "release.sh"), "1.2.4", "--only", "milestone",
                        "--repo", "o/r", *args], cwd=work, capture_output=True, text=True, env=env, timeout=60)
    with open(log) as fh:
        return r, fh.read().splitlines()


def test_the_milestone_section_closes_the_release_and_opens_the_next_patch():
    d, work = _repo()
    try:
        listing = "3 open v1.2.4\n4 open v1.3.0\n5 open Later\n1 closed Nemotron community packs\n"
        r, calls = _milestone_run(work, listing, "--dry-run")
        assert r.returncode == 0, r.stdout + r.stderr
        assert "+ gh api -X PATCH repos/o/r/milestones/3 -f state=closed" in r.stdout, r.stdout
        assert "+ gh api -X POST repos/o/r/milestones -f title=v1.2.5" in r.stdout, r.stdout
        assert all("-X" not in c for c in calls), ("a dry run changed a milestone", calls)

        r, calls = _milestone_run(work, listing)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "api -X PATCH repos/o/r/milestones/3 -f state=closed" in calls, calls
        assert any(c.startswith("api -X POST repos/o/r/milestones -f title=v1.2.5") for c in calls), calls

        r, calls = _milestone_run(work, "3 closed v1.2.4\n6 open v1.2.5\n")
        assert r.returncode == 0, r.stdout + r.stderr
        assert "already closed" in r.stdout and "milestone v1.2.5 exists" in r.stdout, r.stdout
        assert all("-X" not in c for c in calls), calls

        r, calls = _milestone_run(work, "")
        assert r.returncode == 0 and "nothing to close" in r.stdout, r.stdout
        assert any("title=v1.2.5" in c for c in calls), calls
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_the_scripts_parse():
    for f in ("tools/release.sh", "tools/release_box_gate.sh"):
        r = subprocess.run([BASH, "-n", os.path.join(ROOT, f)], capture_output=True, text=True)
        assert r.returncode == 0, (f, r.stderr)


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
