"""The text of a skill or package, for the router to inject (VIS-E6 packages, inside the harness).

A package in the `packages` document is either `inline` (its `body` is the text, edited in the
dashboard) or points at a file on the box: `skill:<name>` is `<dir>/<name>/SKILL.md` in the first
of `skill_dirs` that has it (opencode's `~/.config/opencode/skills` layout), `file:<path>` is one
file. A file is read when first needed and again when its mtime moves. The YAML frontmatter
(`---` ... `---`) is the client's metadata, not instructions, so it is stripped from the text and
its `description` is offered to the dashboard's import.

A library skill (R1, `library: true`, server/conductor/libsearch.py) is a SKILL.md whose text before
its `## Reference` heading is what goes in; the reference is read only on request
(`GET /v1/library/{name}`).
"""

from __future__ import annotations

import os
import re
import threading

CHARS_PER_TOKEN = 3.5
# what ends a text cut to the package's max_tokens (the dashboard's catalog looks for it)
CUT_NOTE = "\n[... cut to the package's max_tokens]"
_FRONT = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)
_REFERENCE = re.compile(r"(?m)^## Reference\b")


def split_frontmatter(text: str) -> tuple[dict, str]:
    """(`{name, description}` from the frontmatter, the body after it)."""
    m = _FRONT.match(text)
    if not m:
        return {}, text
    meta = {}
    for line in m.group(1).splitlines():
        k, sep, v = line.partition(":")
        if sep and k.strip() in ("name", "description"):
            meta[k.strip()] = v.strip().strip('"').strip("'")
    return meta, text[m.end():]


def split_reference(body: str) -> tuple[str, str]:
    """(the text before the `## Reference` heading, the reference from it on); the whole text and
    "" when there is none."""
    m = _REFERENCE.search(body)
    return (body[:m.start()], body[m.start():]) if m else (body, "")


class Library:
    """Resolves package names to text. One per server; reads files lazily."""

    def __init__(self):
        self._files: dict[str, tuple[float, str]] = {}
        self._lock = threading.Lock()

    def _dirs(self, doc: dict) -> list[str]:
        return [os.path.expanduser(d) for d in doc.get("skill_dirs") or []]

    def path(self, doc: dict, name: str) -> str | None:
        p = (doc.get("packages") or {}).get(name)
        if p is None:
            return None
        src = p.get("source", "inline")
        if src.startswith("file:"):
            return os.path.expanduser(src[5:])
        if src.startswith("skill:"):
            skill = src[6:]
            for d in self._dirs(doc):
                cand = os.path.join(d, skill, "SKILL.md")
                if os.path.isfile(cand):
                    return cand
            return os.path.join(self._dirs(doc)[0], skill, "SKILL.md") if self._dirs(doc) else None
        return None

    def file_text(self, path: str) -> str | None:
        """A file's text, read again when its mtime moves (the library's index reads with it)."""
        return self._read(path)

    def _read(self, path: str) -> str | None:
        try:
            mt = os.stat(path).st_mtime
        except OSError:
            return None
        with self._lock:
            got = self._files.get(path)
            if got is not None and got[0] == mt:
                return got[1]
            try:
                with open(path, encoding="utf-8") as f:
                    text = f.read()
            except (OSError, UnicodeDecodeError):
                return None
            self._files[path] = (mt, text)
            return text

    def text(self, doc: dict, name: str) -> tuple[str | None, str | None]:
        """(the package's instruction text, None) or (None, why it is unavailable)."""
        p = (doc.get("packages") or {}).get(name)
        if p is None:
            return None, "package_missing"
        if not p.get("enabled", True):
            return None, "package_disabled"
        if p.get("source", "inline") == "inline":
            body = p.get("body") or ""
        else:
            path = self.path(doc, name)
            raw = self._read(path) if path else None
            if raw is None:
                return None, "package_file_missing"
            body = split_frontmatter(raw)[1]
        if p.get("library"):
            body = split_reference(body)[0]
        body = body.strip()
        if not body:
            return None, "package_empty"
        cap = int(p.get("max_tokens") or 4000) * CHARS_PER_TOKEN
        if len(body) > cap:
            cut = body[:int(cap)]
            body = cut[:cut.rfind("\n")] if "\n" in cut else cut
            body += CUT_NOTE
        return body, None

    def status(self, doc: dict) -> dict:
        """Per package: where its text comes from, whether it is there, its size."""
        out = {}
        for name, p in (doc.get("packages") or {}).items():
            text, err = self.text(doc, name)
            out[name] = {"available": text is not None, "error": err,
                         "path": self.path(doc, name), "chars": len(text or ""),
                         "tokens_est": int(len(text or "") / CHARS_PER_TOKEN)}
        return out

    def scan(self, doc: dict) -> list[dict]:
        """Skills found in `skill_dirs`, for the dashboard's import: name, description, path."""
        seen, out = set(), []
        for d in self._dirs(doc):
            try:
                names = sorted(os.listdir(d))
            except OSError:
                continue
            for n in names:
                path = os.path.join(d, n, "SKILL.md")
                if n in seen or not os.path.isfile(path):
                    continue
                raw = self._read(path) or ""
                meta, body = split_frontmatter(raw)
                seen.add(n)
                out.append({"name": n, "description": meta.get("description", "")[:600],
                            "path": path, "chars": len(body),
                            "registered": n in (doc.get("packages") or {})})
        return out
