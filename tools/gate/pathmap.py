"""Changed paths -> the gate level and the GPU tests a pull request needs (tools/gate/map.yaml)."""

from __future__ import annotations

import functools
import os
import re
import subprocess

from tools.gate import LEVEL_SUITES, LEVELS, level_index, max_level

HERE = os.path.dirname(os.path.abspath(__file__))
MAP = os.path.join(HERE, "map.yaml")


def braces(pattern: str) -> list[str]:
    """`a/{b,c}.py` -> ['a/b.py', 'a/c.py'] (nested and repeated groups too)."""
    m = re.search(r"\{([^{}]*)\}", pattern)
    if not m:
        return [pattern]
    out = []
    for alt in m.group(1).split(","):
        out += braces(pattern[:m.start()] + alt + pattern[m.end():])
    return out


@functools.lru_cache(maxsize=None)
def _regex(pattern: str) -> re.Pattern:
    out, i = "", 0
    while i < len(pattern):
        c = pattern[i]
        if pattern.startswith("**/", i):
            out, i = out + "(?:.*/)?", i + 3
        elif pattern.startswith("**", i):
            out, i = out + ".*", i + 2
        elif c == "*":
            out, i = out + "[^/]*", i + 1
        elif c == "?":
            out, i = out + "[^/]", i + 1
        else:
            out, i = out + re.escape(c), i + 1
    return re.compile(out + r"\Z")


def match(path: str, pattern: str) -> bool:
    return any(_regex(p).match(path) for p in braces(pattern))


def load(path: str = MAP) -> dict:
    import yaml
    with open(path) as fh:
        m = yaml.safe_load(fh)
    if m.get("default") not in LEVELS:
        raise ValueError(f"{path}: default {m.get('default')!r} is not a level")
    for r in m.get("rules") or []:
        if r.get("level") not in LEVELS or not r.get("paths"):
            raise ValueError(f"{path}: a rule needs a level and paths: {r}")
    return m


def level_of(path: str, m: dict) -> tuple[str, str | None]:
    """A file's level and the pattern that decided it (None: no rule matched, the default)."""
    best, why = None, None
    for r in m["rules"]:
        for p in r["paths"]:
            if match(path, p) and (best is None or level_index(r["level"]) > level_index(best)):
                best, why = r["level"], p
    return (best, why) if best is not None else (m["default"], None)


def kernels(files: list[str], m: dict) -> list[str]:
    out: list[str] = []
    for k in m.get("kernels") or []:
        if any(match(f, p) for f in files for p in k["paths"]):
            out += [t for t in k["tests"] if t not in out]
    return out


def required(files: list[str], m: dict | None = None) -> dict:
    """What a diff of `files` needs: its level, the suites, the GPU tests, and each file's rule."""
    m = m or load()
    by_file = {f: level_of(f, m) for f in files}
    level = max_level(lv for lv, _ in by_file.values())
    tests = kernels(files, m) if level != "none" else []
    return {"level": level, "suites": list(LEVEL_SUITES[level]), "kernels": tests,
            "files": {f: {"level": lv, "rule": why or "default"} for f, (lv, why) in by_file.items()}}


def changed_files(base: str, head: str, repo: str = ".") -> list[str]:
    """The files `head` changes against its merge base with `base` (as a pull request's diff)."""
    out = subprocess.run(["git", "-C", repo, "diff", "--name-only", f"{base}...{head}"],
                         capture_output=True, text=True, check=True).stdout
    return [ln for ln in out.splitlines() if ln.strip()]


def changed_between(a: str, b: str, repo: str = ".") -> list[str]:
    """The files that differ between two commits (a report's head and the pull request's head)."""
    out = subprocess.run(["git", "-C", repo, "diff", "--name-only", a, b],
                         capture_output=True, text=True, check=True).stdout
    return [ln for ln in out.splitlines() if ln.strip()]
