"""The weight cards' speed rows (SPD-245): tools/parallel_card.py's row and the registry's rules for the
`parallel` section of a model.yaml (server/setup/registry.py `_parallel_errors`, `_PAR_ROW`).

The test matrix:

  1  the row is the raw file's: build_row takes every number from the run (decode medians, prefill
     medians, each level's medians and its proof), the YAML the tool writes loads back to the same
     row, and the registry takes it                     test_the_row_is_the_raw_file
  2  a level whose rounds did not decode N at once is refused ("refused: 7 decoded at once, not 8"),
     and so is a level with fewer valid rounds than the row's
                                                        test_a_level_without_proof_is_refused
  3  the raw file: missing, or a row that differs from the file's, is refused; a raw path outside the
     model's measurements/ is refused                   test_the_raw_file_is_required
  4  the weight set: an unknown profile, a community pack the model does not have and a second row
     for one weight set are refused; a community pack that is there is taken
                                                        test_the_row_names_a_weight_set
  5  the schema: levels 1/2/4/8 only (3 is an unknown key), level 1 required, a commit that is not
     hex refused                                        test_the_schema
  6  every shipped row loads, proves every level and is its raw file's row
                                                        test_the_shipped_rows

Run: python tests/test_parallel_card.py
"""

from __future__ import annotations

import copy
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml  # noqa: E402

os.environ.setdefault("QSE_COMMUNITY_DIR", tempfile.mkdtemp(prefix="parcard-community-"))
os.environ.setdefault("QSE_COMMUNITY_URL", "off")

from server.conductor.schema import validate  # noqa: E402
from server.setup import registry  # noqa: E402
from tools import parallel_card as pc  # noqa: E402

MID = "qwen3.8-27b"


def _level(n: int, at_once: int | None = None, rounds: int = 3) -> dict:
    rs = [{"wall_tok_s": 30.0 * n + i, "decode_tok_s": 32.0 * n + i, "proof": {"max_decoding_at_once": n if at_once is None else at_once},
           "streams": [{"ttft_s": 0.2 + 0.1 * i}] * n} for i in range(rounds)]
    return {"rounds": rs, "valid": at_once in (None, n), "wall_tok_s": 30.0 * n + 1, "decode_tok_s": 32.0 * n + 1,
            "ttft_s": 0.3, "at_once": n if at_once is None else at_once}


def _raw(profile: str = "exl3-4.0", **over) -> dict:
    raw = {"profile": profile, "build": "0.14.2", "commit": "345ab0fa", "date": "2026-10-26",
           "raw_path": f"models/{MID}/measurements/parallel-test-{profile.replace('community/', '')}.json",
           "args": {"rounds": 3},
           "setup": {"env": "live8-cold", "lanes": 8, "window": 262144, "max_tokens": 512, "suffix_store": False,
                     "response_cache": False,
                     "sampling": {"temperature": 1.0, "top_k": 20, "top_p": 0.95, "min_p": 0.0, "seed": "one per request"},
                     "prompts": {"set": "card-v1", "sha256": "0123456789abcdef0123", "prompt_tokens": {"code": 246, "prose": 190}}},
           "decode": {"median": {"code": 71.23, "prose": 36.44}},
           "prefill": {"depths": {"2048": {"ttft_s": 1.612, "tok_s": 1268.4, "prompt_tokens": 2045},
                                  "8192": {"ttft_s": 5.871, "tok_s": 1395.2, "prompt_tokens": 8190}}},
           "levels": {str(n): _level(n) for n in (1, 2, 4, 8)}}
    raw.update(over)
    return raw


def _root(rows_raw: list[dict], community: tuple = ()) -> str:
    root = tempfile.mkdtemp(prefix="parcard-")
    os.makedirs(os.path.join(root, "models", MID, "measurements"))
    os.makedirs(os.path.join(root, "models", MID, "community"))
    for c in community:
        open(os.path.join(root, "models", MID, "community", c + ".json"), "w").write("{}")
    for raw in rows_raw:
        raw = dict(raw, row=pc.build_row(raw))
        with open(os.path.join(root, raw["raw_path"]), "w") as fh:
            json.dump(raw, fh)
    return root


def _model(rows: list[dict]) -> dict:
    m = registry.get(MID)
    m = copy.deepcopy({k: m[k] for k in ("id", "profiles")})
    m["parallel"] = {"method": "tools/parallel_card.py", "rows": rows}
    return m


def test_the_row_is_the_raw_file():
    raw = _raw()
    row = pc.build_row(raw)
    assert row["decode"] == {"code": 71.2, "prose": 36.4}
    assert row["prefill"]["2048"] == {"ttft_s": 1.61, "tok_s": 1268.0, "prompt_tokens": 2045}
    assert row["levels"]["8"] == {"wall_tok_s": 241.0, "decode_tok_s": 257.0, "ttft_s": 0.3, "at_once": 8, "rounds": 3}
    assert row["setup"]["prompts"]["sha256"] == "0123456789abcdef" and row["setup"]["thinking"] is False
    back = yaml.safe_load("rows:\n" + pc.row_yaml(row))["rows"][0]
    assert back == row, (back, row)
    assert not validate(row, registry._PAR_ROW), validate(row, registry._PAR_ROW)
    assert registry._parallel_errors(_model([back]), _root([raw])) == []
    # a workload without a decode window (one event) is left out of the row
    raw2 = _raw(decode={"median": {"code": None, "prose": 126.29}})
    assert pc.build_row(raw2)["decode"] == {"prose": 126.3}
    assert not validate(pc.build_row(raw2), registry._PAR_ROW)


def test_a_level_without_proof_is_refused():
    raw = _raw(levels={"1": _level(1), "2": _level(2), "4": _level(4), "8": _level(8, at_once=7)})
    errs = registry._parallel_errors(_model([pc.build_row(raw)]), _root([raw]))
    assert any("levels.8: refused: 7 decoded at once, not 8" in e for e in errs), errs
    raw = _raw(levels={"1": _level(1), "8": _level(8, rounds=2)})
    errs = registry._parallel_errors(_model([pc.build_row(raw)]), _root([raw]))
    assert any("levels.8: 2 valid rounds of 3" in e for e in errs), errs


def test_the_raw_file_is_required():
    raw = _raw()
    row = pc.build_row(raw)
    errs = registry._parallel_errors(_model([row]), _root([]))
    assert any("no raw file" in e for e in errs), errs
    changed = copy.deepcopy(row)
    changed["levels"]["8"]["wall_tok_s"] = 999.0
    errs = registry._parallel_errors(_model([changed]), _root([raw]))
    assert any("not the raw file's row" in e for e in errs), errs
    outside = dict(row, raw="models/flint/measurements/x.json")
    errs = registry._parallel_errors(_model([outside]), _root([raw]))
    assert any("not under models/qwen3.8-27b/measurements/" in e for e in errs), errs


def test_the_row_names_a_weight_set():
    tc = "community/thinkingcap-27b-kearuga-exl3-4.00"
    a, b, c = _raw("exl3-9.9"), _raw(tc), _raw("community/nobody")
    root = _root([a, b, c], community=("thinkingcap-27b-kearuga-exl3-4.00",))
    errs = registry._parallel_errors(_model([pc.build_row(x) for x in (a, b, c)]), root)
    assert any("no profile 'exl3-9.9'" in e for e in errs), errs
    assert any("no community pack 'community/nobody'" in e for e in errs), errs
    assert not any(tc in e for e in errs), errs
    d = _raw()
    errs = registry._parallel_errors(_model([pc.build_row(d), pc.build_row(d)]), _root([d]))
    assert any("a second row for 'exl3-4.0'" in e for e in errs), errs


def test_the_schema():
    row = pc.build_row(_raw())
    bad = copy.deepcopy(row)
    bad["levels"]["3"] = bad["levels"]["2"]
    assert any("unknown key '3'" in e for e in validate(bad, registry._PAR_ROW))
    bad = copy.deepcopy(row)
    del bad["levels"]["1"]
    assert any("missing '1'" in e for e in validate(bad, registry._PAR_ROW))
    assert any("commit" in e for e in validate(dict(row, commit="HEAD"), registry._PAR_ROW))


def test_the_shipped_rows():
    models, broken = registry.load_all()
    assert not broken, broken
    for mid, m in models.items():
        for r in (m.get("parallel") or {}).get("rows") or []:
            with open(os.path.join(registry.ROOT, r["raw"])) as fh:
                raw = json.load(fh)
            assert raw["row"] == r and raw["valid"], (mid, r["profile"])
            for n, lv in r["levels"].items():
                assert lv["at_once"] == int(n) and raw["levels"][n]["valid"], (mid, r["profile"], n)
                assert all(x["proof"]["max_decoding_at_once"] == int(n) and x["proof"]["max_running"] <= int(n)
                           for x in raw["levels"][n]["rounds"]), (mid, r["profile"], n)


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:68s} ok")
            passed += 1
    print(f"{passed} passed")
