"""The record store: one JSON per run under DIR/runs/ (its records by row id), and DIR/index.json, keyed
by (row id, build, pack, box, knobs hash) -> the run file that holds the row. The index is rebuilt from the
run files, so two copies of a store (the box's and the laptop's) join by copying run files.

A key holds one record. A second record for a key is refused unless it is forced (`run --force`), or
it is an import of a row another source already holds: then the first stays the row's record and the
second is listed beside it (`also`), so a view shows one number and every source."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

from tools.measure import KNOB_KEYS, SCHEMA


def build_of(version: str | None) -> str:
    """A server's version string -> its build (the release): "0.15.2+flashnext-3.05" -> "0.15.2"."""
    return re.split(r"[+\s]", str(version or "unknown"))[0] or "unknown"


def limit_knobs(knobs: dict | None) -> dict:
    return {k: (knobs or {}).get(k) for k in KNOB_KEYS}


def knobs_hash(knobs: dict | None) -> str:
    return hashlib.sha256(json.dumps(limit_knobs(knobs), sort_keys=True, default=str).encode()).hexdigest()[:12]


def key(row: str, build: str, pack: str, box: str, khash: str) -> str:
    return "|".join((row, build, pack, box, khash))


class Duplicate(SystemExit):
    def __init__(self, keys: list[str]):
        super().__init__("[measure] refused: these rows already have a record (a row is measured once; "
                         "--force measures it again): " + ", ".join(k.split("|")[0] for k in keys))
        self.keys = keys


class Store:
    def __init__(self, root):
        self.root = Path(os.path.expanduser(str(root)))
        self.runs = self.root / "runs"

    # --- reading ---------------------------------------------------------------------------------
    def run_files(self) -> list[Path]:
        if not self.runs.is_dir():
            return []
        files = [p for p in self.runs.glob("*.json")]

        def added(p):
            try:
                return json.loads(p.read_text()).get("run", {}).get("added") or ""
            except ValueError:
                return ""
        return sorted(files, key=lambda p: (added(p), p.name))

    def load_runs(self) -> list[tuple[Path, dict]]:
        out = []
        for p in self.run_files():
            try:
                out.append((p, json.loads(p.read_text())))
            except ValueError as exc:
                raise SystemExit(f"[measure] {p}: not JSON ({exc})")
        return out

    def index(self) -> dict:
        """key -> {"path": the record's run file, "also": [other run files with a record of the key]}. The
        first record of a key wins, unless a later one was forced."""
        idx: dict = {}
        for p, doc in self.load_runs():
            rel = str(p.relative_to(self.root))
            for rid, rec in (doc.get("records") or {}).items():
                k = key(rid, rec["build"], rec["pack"], rec["box"], rec["knobs_hash"])
                if k not in idx:
                    idx[k] = {"path": rel, "also": []}
                elif rec.get("forced"):
                    idx[k] = {"path": rel, "also": idx[k]["also"] + [idx[k]["path"]]}
                else:
                    idx[k]["also"].append(rel)
        return idx

    def write_index(self) -> dict:
        idx = self.index()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "index.json").write_text(json.dumps(idx, indent=1, sort_keys=True) + "\n")
        return idx

    def record(self, rel: str, rid: str) -> dict:
        doc = json.loads((self.root / rel).read_text())
        rec = dict(doc["records"][rid])
        rec["_run"] = doc.get("run") or {}
        rec["_path"] = rel
        return rec

    def find(self, rid: str, build: str, pack: str, box: str | None = None, khash: str | None = None) -> list[dict]:
        """The records of a row for (build, pack[, box[, knobs]]): each key's own record first, then its
        others (`also`), keys in the order they were added."""
        out = []
        for k, ent in self.index().items():
            r, b, p, x, h = k.split("|")
            if r == rid and b == build and p == pack and (box is None or x == box) and (khash is None or h == khash):
                out.append(dict(self.record(ent["path"], rid), _primary=True))
                out += [dict(self.record(a, rid), _primary=False) for a in ent["also"]]
        return out

    def primary(self, rid: str, build: str, pack: str, box: str | None = None,
                khash: str | None = None) -> dict | None:
        got = [r for r in self.find(rid, build, pack, box, khash) if r["_primary"]]
        return got[0] if got else None

    def has(self, rid: str, build: str, pack: str, box: str, khash: str | None = None) -> bool:
        return self.primary(rid, build, pack, box, khash) is not None

    def packs(self, build: str) -> list[str]:
        seen = []
        for k in self.index():
            _, b, p, _, _ = k.split("|")
            if b == build and p not in seen:
                seen.append(p)
        return seen

    # --- writing ---------------------------------------------------------------------------------
    def add_run(self, run: dict, records: dict, force: bool = False, imported: bool = False) -> Path:
        """One run's records. A row whose key holds a record already is refused (Duplicate) unless
        `force` (the new one becomes the key's record) or `imported` (listed beside the first)."""
        idx = self.index()
        dup = [key(rid, r["build"], r["pack"], r["box"], r["knobs_hash"]) for rid, r in records.items()]
        dup = [k for k in dup if k in idx]
        if dup and not force and not imported:
            raise Duplicate(dup)
        run = dict(run, added=run.get("added") or time.strftime("%Y-%m-%dT%H:%M:%S%z"), schema=SCHEMA)
        for rid, r in records.items():
            r.setdefault("row", rid)
            if force and key(rid, r["build"], r["pack"], r["box"], r["knobs_hash"]) in idx:
                r["forced"] = True
        self.runs.mkdir(parents=True, exist_ok=True)
        name = re.sub(r"[^\w.-]+", "-", run["id"])
        p = self.runs / f"{name}.json"
        if p.exists():
            raise SystemExit(f"[measure] {p} exists: a run id is used once")
        p.write_text(json.dumps({"kind": "tandem-measure-run", "run": run, "records": records}, indent=1,
                                default=str) + "\n")
        self.write_index()
        return p
