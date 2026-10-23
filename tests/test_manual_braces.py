"""#126: every table manual/tools/gen_reference.py writes closes each brace it opens, no GPU.

manual/generated/ledger.tex opened a `{\\hbadness=3000` group around its last paragraph and never
closed it, so the setting held to the end of the book and LuaLaTeX ended with
`(\\end occurred inside a group at level 1)`.

  * the counter: escaped braces (`\\{`, `\\}`), `\\\\` and comments do not count, and a `}` before
    its `{` is an error, as is a `{` left open;
  * every .tex file the generator writes, built fresh from this checkout, is balanced;
  * every committed manual/generated/*.tex is balanced, so a hand edit cannot open one either;
  * the ledger table's journal paragraph ends its group with `\\par}`.
"""

from __future__ import annotations

import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "manual", "tools")
sys.path.insert(0, TOOLS)

_COMMENT = re.compile(r"(?<!\\)%.*")
_ESCAPED = re.compile(r"\\[\\{}%]")


def unbalanced(text: str) -> str:
    """'' when every brace closes in order, else where it goes wrong."""
    depth, opened = 0, []
    for n, line in enumerate(text.splitlines(), 1):
        line = _COMMENT.sub("", _ESCAPED.sub("", line))
        for ch in line:
            if ch == "{":
                depth += 1
                opened.append(n)
            elif ch == "}":
                if not depth:
                    return f"line {n}: a }} with no {{ open"
                depth -= 1
                opened.pop()
    return f"{depth} {{ left open, the first on line {opened[0]}" if depth else ""


def test_the_counter():
    assert unbalanced(r"\texttt{a\_\allowbreak{}b} & \{x\} % {") == ""
    assert unbalanced(r"a \\{b}") == ""
    assert unbalanced(r"50\% {x}") == ""
    assert unbalanced("{\\hbadness=3000 \\noindent text.\n").startswith("1 { left open, the first on line 1")
    assert unbalanced("x}\n{").startswith("line 1:")
    return "escapes, comments, an open group, a stray close"


def test_every_generated_table_is_balanced():
    import gen_reference
    files = gen_reference.build(ROOT)
    tex = {os.path.relpath(p, ROOT): t for p, t in files.items() if p.endswith(".tex")}
    assert len(tex) >= 15, sorted(tex)
    bad = {p: why for p, t in tex.items() if (why := unbalanced(t))}
    assert not bad, bad
    return f"{len(tex)} tables built from the code"


def test_every_committed_table_is_balanced():
    paths = sorted(glob.glob(os.path.join(ROOT, "manual", "generated", "*.tex")))
    assert len(paths) >= 15, paths
    bad = {os.path.basename(p): why for p in paths if (why := unbalanced(open(p, encoding="utf-8").read()))}
    assert not bad, f"run manual/tools/gen_reference.py: {bad}"
    return f"{len(paths)} files in manual/generated"


def test_the_ledger_journal_closes_its_group():
    import gen_reference
    text = gen_reference.gen_ledger(ROOT)
    last = text.rstrip("\n").splitlines()[-1]
    assert last.startswith(r"{\hbadness=3000 ") and last.endswith(r"\par}"), last[-80:]
    return "{\\hbadness=3000 ... \\par}"


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<50} ok   {fn() or ''}")
        except AssertionError as e:
            fails += 1
            print(f"  {name:<50} FAIL {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
