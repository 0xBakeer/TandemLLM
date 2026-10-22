"""docs/labels.md holds the table that tools/gh_labels.sh --table prints (OPS-229)."""
import os
import re
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _table():
    out = subprocess.run(["bash", os.path.join(ROOT, "tools", "gh_labels.sh"), "--table"],
                         check=True, capture_output=True, text=True).stdout
    return out.strip().splitlines()


def test_docs_table_is_the_scripts_table():
    with open(os.path.join(ROOT, "docs", "labels.md"), encoding="utf-8") as f:
        doc = [line for line in f.read().splitlines() if line.startswith("|")]
    assert doc == _table()


def test_every_label_has_a_short_description_and_a_colour():
    rows = _table()[2:]
    assert len(rows) >= 50
    names = set()
    for row in rows:
        _, group, name, desc, _ = row.split("|")
        name = name.strip().strip("`")
        assert name not in names, name
        names.add(name)
        assert 0 < len(desc.strip()) <= 100, name
    for need in ("bug", "feature", "perf", "docs", "needs-triage", "run-gpu", "skip-changelog"):
        assert need in names, need
    with open(os.path.join(ROOT, "tools", "gh_labels.sh"), encoding="utf-8") as f:
        script = f.read()
    for colour in re.findall(r"^\w+\|[^|]+\|([^|]*)\|", script, re.M):
        assert re.fullmatch(r"[0-9a-f]{6}", colour), colour


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
