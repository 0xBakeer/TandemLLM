"""The row list (tools/measure/rows.yaml), expanded: one Row per id, in the file's order."""

from __future__ import annotations

import dataclasses
import fnmatch
import itertools
from pathlib import Path

ROWS = Path(__file__).resolve().parent / "rows.yaml"

#: a length label's prompt tokens (the report's ladder, tools/release_report/plan.py)
TOKENS = {"1k": 1024, "2k": 2048, "4k": 4096, "8k": 8192, "16k": 16384, "32k": 32768, "55k": 56320,
          "57k": 57344, "64k": 65536, "96k": 98304, "128k": 131072, "131k": 131072, "192k": 196608,
          "240k": 245760}


@dataclasses.dataclass
class Row:
    id: str
    measures: str
    shape: dict
    views: list
    yields: list
    exec: str
    server: str | None
    est: float
    after: list
    order: int
    param: dict                     # the `for` values of this row ({"k": "32k"} ...)

    @property
    def tokens(self) -> int | None:
        """The prompt length of a length-family row, in tokens."""
        k = self.param.get("k") or next((x for x in self.id.split(".") if x in TOKENS), None)
        return TOKENS.get(k) if k else None

    @property
    def n(self) -> int | None:
        v = self.param.get("n")
        return int(v) if v is not None else None


def _fill(x, vals: dict):
    if isinstance(x, str):
        for k, v in vals.items():
            x = x.replace("{" + k + "}", str(v))
        return x
    if isinstance(x, list):
        return [_fill(v, vals) for v in x]
    if isinstance(x, dict):
        return {k: _fill(v, vals) for k, v in x.items()}
    return x


def load(path: Path = ROWS) -> tuple[dict, dict]:
    """(rows by id in order, groups)."""
    import yaml
    doc = yaml.safe_load(path.read_text())
    out: dict = {}
    order = 0
    for spec in doc["rows"]:
        fam = spec.get("for") or {}
        keys = list(fam)
        for combo in itertools.product(*(fam[k] for k in keys)) if keys else [()]:
            vals = dict(zip(keys, combo))
            rid = _fill(spec["id"], vals)
            if rid in out:
                raise ValueError(f"rows.yaml: the id {rid} twice")
            est = spec.get("est", 0)
            if isinstance(est, dict):
                est = est.get(combo[0], est.get(str(combo[0]), 0)) if combo else 0
            out[rid] = Row(id=rid, measures=_fill(spec["measures"], vals), shape=_fill(spec.get("shape") or {}, vals),
                           views=list(spec.get("views") or []), yields=list(spec.get("yields") or []),
                           exec=spec["exec"], server=spec.get("server"), est=float(est or 0),
                           after=list(spec.get("after") or []), order=order, param=vals)
            order += 1
    return out, doc.get("groups") or {}


def for_view(rows: dict, view: str, pack_rows: dict | None = None) -> list[Row]:
    """The rows a view uses (the gate's c8 only of the card's levels: its profile's level)."""
    out = [r for r in rows.values() if view in r.views and r.exec != "every-run"]
    if view == "gate":
        out = [r for r in out if not r.id.startswith("conc.card.c") or r.id == "conc.card.c8"]
    return out


def select(rows: dict, spec: str) -> list[Row]:
    """`a,b,conc.2k.*`: ids and glob patterns, in the row list's order; an unknown id is an error."""
    want = [s.strip() for s in spec.split(",") if s.strip()]
    got = []
    for w in want:
        hit = [r for r in rows.values() if fnmatch.fnmatchcase(r.id, w)]
        if not hit:
            raise SystemExit(f"[measure] no row {w!r}: see `python -m tools.measure rows`")
        got += [r for r in hit if r not in got]
    return sorted(got, key=lambda r: r.order)


def matches(rid: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(rid, p) for p in patterns)
