"""Fingerprint v3 (ENG-243): what a model's stored rows depend on, as the model's adapter lists it.

The disk tier's fingerprint v2 (`engine/disk_cache.fingerprint`) hashes every file under `engine/`
but a few, so moving method code -- StairCut, the router, sampling, the caches, the server's loop --
sweeps every saved conversation, and it misses the tile tables that set the kernels' summation order
(a knob names the file; v2 hashed the name, never the table). v3 hashes the adapter's `manifest()`:

  modules   the files whose code computes stored rows (forward, verify and commit bodies, the
            drafters whose state is cached), and the `tools/` modules they import, transitively
  knobs     the defaults of the knobs those files own (engine/settings.py `KNOBS`); a value set in
            the environment is in the caller's `extra`, as under v2
  data      the tables those files read that set the arithmetic (tile tables), by content: the
            manifest's own, every .json a knob of theirs names, a module's default table
  extra     the weights, the environment, the grid, the libraries (server/diskcache.py)

Method code is not in it: it chooses which rows are computed, never how.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
from dataclasses import dataclass

from engine import disk_cache as dc

#: fp4 (rc-0.8.0): fp3's hash over the server's extra by content (server/diskcache.py
#: `fingerprint_extra`), where fp3's named files and tables by path; fp3 is still computed, by path,
#: to adopt the files the builds before wrote (`server/diskcache.py legacy_fingerprints`)
FORMULA = "fp4"


@dataclass(frozen=True)
class Manifest:
    arch: str                   # the registry name (engine/models: qwen3_5, qwen4_exp)
    modules: tuple              # repo-relative files that compute stored rows
    data: tuple = ()            # tables they read that set the arithmetic (repo-relative or absolute)


def files(root: str, m: Manifest) -> list[str]:
    """The manifest's modules and the `tools/` modules they import (absolute paths, in order)."""
    mods = [os.path.join(root, p) for p in m.modules]
    return mods + dc._tools_modules(root, mods)


#: tables a module reads when its knob names none (the knob's value, when set, is read instead)
DEFAULT_TABLES = {"tools/exl3_linear.py": ("EXL3_TILES", os.path.join("ops", "exl3-tiles-fused.json"))}


def _knob_table(settings_path: str) -> dict:
    with open(settings_path, "rb") as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        target = node.target if isinstance(node, ast.AnnAssign) else (
            node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1 else None)
        if isinstance(target, ast.Name) and target.id == "KNOBS" and node.value is not None:
            return ast.literal_eval(node.value)
    raise ValueError(f"{settings_path} has no literal KNOBS table")


def knobs_of(settings_path: str, owners: set) -> bytes:
    """(name, default) of every knob in `settings_path` whose owner is one of `owners`, in name order."""
    knobs = _knob_table(settings_path)
    return json.dumps(sorted((str(k), v[0]) for k, v in knobs.items()
                             if len(v) > 1 and str(v[1]) in owners), default=str).encode()


def tables(root: str, owners: set, env=None) -> list[str]:
    """The tables the owners read: a knob of theirs whose value (the environment's, else its default)
    names a .json file, and a module's own default table when its knob is unset (`DEFAULT_TABLES`)."""
    env = os.environ if env is None else env
    knobs = _knob_table(os.path.join(root, dc.SETTINGS_FILE))
    out = set()
    for k, v in knobs.items():
        if len(v) > 1 and str(v[1]) in owners:
            val = env.get("QWEN38_" + k, v[0])
            if isinstance(val, str) and val.endswith(".json"):
                out.add(val)
    for mod, (k, path) in DEFAULT_TABLES.items():
        if mod in owners and not env.get("QWEN38_" + k, knobs.get(k, (None,))[0]):
            out.add(path)
    return sorted(out)


def fingerprint(root: str, m: Manifest, extra: dict, formula: str = FORMULA) -> str:
    h = hashlib.sha256()
    h.update(formula.encode())
    h.update(m.arch.encode())
    paths = files(root, m)
    owners = {os.path.relpath(p, root) for p in paths}
    for p in paths:
        rel = os.path.relpath(p, root)
        h.update(rel.encode())
        if rel == dc.SETTINGS_FILE:
            data = knobs_of(p, owners)
        else:
            with open(p, "rb") as f:
                data = f.read()
        h.update(hashlib.sha256(data).digest())
    for d in sorted(set(m.data) | set(tables(root, owners))):
        full = os.path.expanduser(d) if os.path.isabs(os.path.expanduser(d)) else os.path.join(root, d)
        rel = os.path.relpath(full, root)
        h.update((full if rel.startswith("..") else rel).encode())
        try:
            with open(full, "rb") as f:
                h.update(hashlib.sha256(f.read()).digest())
        except OSError:
            h.update(b"missing")
    h.update(json.dumps(extra, sort_keys=True, default=str).encode())
    return h.hexdigest()
