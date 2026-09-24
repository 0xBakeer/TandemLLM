"""CPU tests for TRN-7's data: the holdout check and the pool.

The check is the one guard against the 2026-09-17 failure (identical prompts on both sides of a
split faked +69 %), so each of its three detectors is tested to fire on the case it exists for and
to stay quiet on unrelated text. The pool is tested for its quota, its exclusion list and its split.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.h100.common import iter_shards, write_jsonl  # noqa: E402
from tools.h100.opd_data import QUOTA, dedup_key, overlaps, pick  # noqa: E402

LONG = ("Explain in three short paragraphs why a memory bound decoder spends most of its step "
        "waiting on weights rather than on arithmetic, and what that means for batch size")


def _train(*texts):
    return [{"name": f"p{i:06d}", "text": t} for i, t in enumerate(texts)]


def test_dedup_key_is_prompts_py_key():
    """Whitespace and case do not make a prompt new: the key folds both, as prompts.py does."""
    assert dedup_key("Hello   World\n") == dedup_key("hello world")
    assert dedup_key("hello world") != dedup_key("hello worlds")


def test_exact_prompt_is_a_hit():
    hits = overlaps([{"name": "row-a", "text": LONG}], _train("unrelated text", LONG.upper()))
    assert any(h["why"] == "dedup key" and h["train"] == "p000001" for h in hits)


def test_wrapped_prompt_is_a_hit():
    """A template around the holdout prompt (containment), in either direction."""
    wrapped = "Please answer the following question carefully.\n\n" + LONG + "\n\nThanks."
    hits = overlaps([{"name": "row-a", "text": LONG}], _train(wrapped))
    assert any(h["why"] == "containment" for h in hits)
    hits = overlaps([{"name": "row-a", "text": wrapped}], _train(LONG))
    assert any(h["why"] == "containment" for h in hits)


def test_paraphrase_with_a_long_shared_run_is_a_hit():
    para = "My question: " + " ".join(LONG.split()[:14]) + " -- and nothing else is the same here."
    hits = overlaps([{"name": "row-a", "text": LONG}], _train(para), n=13)
    assert any("shared 13-word" in h["why"] for h in hits)


def test_unrelated_prompts_do_not_hit():
    hits = overlaps([{"name": "row-a", "text": LONG}],
                    _train("Write a Python function that reverses a linked list in place.",
                           "Erkläre kurz, wie ein Kühlschrank funktioniert.",
                           "why a memory bound decoder"))       # short: below every threshold
    assert hits == []


def _seqs():
    out = []
    i = 0
    for klass in ("prose", "code", "de"):
        for think in (False, True):
            for _ in range(40):
                out.append({"name": f"p{i:06d}", "klass": klass, "think": think, "kind": "gen",
                            "split": "train", "n": 300})
                i += 1
    out += [{"name": f"h{j}", "klass": "prose", "think": False, "kind": "gen", "split": "heldout",
             "n": 300} for j in range(3)]
    return out


def test_pick_follows_the_quota_and_is_seeded():
    seqs = _seqs()
    got = pick(seqs, 80, seed=7, exclude=set())
    assert len(got) == 80 and len({s["name"] for s in got}) == 80
    for (klass, think), q in QUOTA.items():
        n = sum(1 for s in got if s["klass"] == klass and bool(s["think"]) == think)
        assert n == round(80 * q), (klass, think, n)
    assert all(s["split"] == "train" for s in got)
    assert [s["name"] for s in got] == [s["name"] for s in pick(seqs, 80, 7, set())]
    assert [s["name"] for s in got] != [s["name"] for s in pick(seqs, 80, 8, set())]


def test_pick_fills_a_short_cell_from_the_others_and_never_takes_an_excluded_one():
    seqs = _seqs()
    exclude = {s["name"] for s in seqs if s["klass"] == "prose" and not s["think"]}
    got = pick(seqs, 100, seed=1, exclude=exclude)
    assert len(got) == 100
    assert not any(s["name"] in exclude for s in got)


def test_pool_writes_the_split_and_the_rows():
    from tools.h100 import opd_data
    seqs = _seqs()
    with tempfile.TemporaryDirectory() as d:
        man = os.path.join(d, "manifest.json")
        json.dump({"sequences": seqs}, open(man, "w"))
        gen = os.path.join(d, "gen")
        write_jsonl(os.path.join(gen, "shard-0000.jsonl"),
                    [{"name": s["name"], "prompt_ids": [1, 2], "gen_ids": [3] * 10} for s in seqs])
        out = os.path.join(d, "out")
        argv = sys.argv
        sys.argv = ["opd_data.py", "pool", "--manifest", man, "--gen", gen, "--out", out,
                    "--n", "40", "--shard-size", "16"]
        try:
            opd_data.main()
        finally:
            sys.argv = argv
        rows = list(iter_shards(out))
        assert len(rows) == 43
        assert sum(r["opd_split"] == "heldout" for r in rows) == 3
        assert {r["name"] for r in rows if r["opd_split"] == "heldout"} == {"h0", "h1", "h2"}
        pool = json.load(open(os.path.join(out, "pool.json")))
        assert pool["n_train"] == 40 and pool["n_heldout"] == 3


def test_agree_counts_generated_positions_only():
    """The label drift is read over the continuation, not the prompt, against the right rows of
    the right shard."""
    import torch
    from tools.h100.opd_data import agree
    from tools.h100.shard import FIELDS
    with tempfile.TemporaryDirectory() as d:
        old, new = os.path.join(d, "old"), os.path.join(d, "new")
        os.makedirs(old)
        os.makedirs(new)
        n = 10
        seqs, rows = [], {f: [] for f in FIELDS}
        for j, name in enumerate(("a", "b")):
            lab = torch.arange(n, dtype=torch.int32) + 100 * j
            rows["ids"].append(lab.clone())
            rows["label"].append(lab.clone())
            rows["fused"].append(torch.zeros(n, 4, dtype=torch.bfloat16))
            rows["top_ids"].append(torch.zeros(n, 2, dtype=torch.int32))
            rows["top_lp"].append(torch.full((n, 2), -0.5, dtype=torch.float16))
            seqs.append({"name": name, "shard": 0, "offset": j * n, "n": n, "gen_start": 4})
            new_lab = lab.clone()
            if name == "b":
                new_lab[2] = -1          # in the prompt: not counted
                new_lab[6] = -1          # generated: one disagreement
            torch.save({"ids": lab, "label": new_lab, "fused": torch.zeros(n, 4),
                        "top_ids": torch.zeros(n, 2, dtype=torch.int32),
                        "top_lp": torch.full((n, 2), -1.0, dtype=torch.float16)},
                       os.path.join(new, f"{name}.pt"))
        torch.save({f: torch.cat(v) for f, v in rows.items()}, os.path.join(old, "shard-0000.pt"))
        json.dump({"sequences": seqs, "sharded": True}, open(os.path.join(old, "manifest.json"), "w"))
        json.dump({"sequences": [{"name": "a", "gen_start": 4}, {"name": "b", "gen_start": 4}]},
                  open(os.path.join(new, "manifest.json"), "w"))
        msg = agree(new, old, 10)
        # 2 sequences x positions 4..8 (5 each, the last position has no next token) = 10; 9 agree
        assert "10 generated positions" in msg and "0.9000" in msg, msg


def _main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = 0
    for fn in fns:
        try:
            fn()
            print(f"  ok  {fn.__name__}")
        except Exception as exc:                                    # noqa: BLE001
            bad += 1
            import traceback
            traceback.print_exc()
            print(f"FAIL  {fn.__name__}: {exc}")
    print(f"\n{len(fns) - bad}/{len(fns)} passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(_main())
