"""SPD-245: an unseeded sampled request draws its own seed and takes the keyed walk (server/sampling_defaults.py
`admit_seed`), on the CPU.

The seed is fresh per request (two requests, two seeds, two texts), the mode decides where it is drawn, the
admitted request takes the seeded request's keyed walk (`round_walks`, `tree_walk_logits`), and the keyed walk
over many fresh seeds samples the filtered distribution the unseeded multinomial sampled: per position (a
chi-square against `_filter`'s probabilities) and from one position to the next (the second token of a
two-deep chain, given the first, follows its own row).
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402

from engine import sample as S  # noqa: E402
from engine.sample import Sampler  # noqa: E402
from server import sampling_defaults as sd  # noqa: E402


def _chi2_crit(dof: int, z: float = 3.09) -> float:
    """The chi-square quantile at p = 0.001 (Wilson-Hilferty)."""
    a = 2.0 / (9.0 * dof)
    return dof * (1.0 - a + z * math.sqrt(a)) ** 3


def _chi2(counts: list, probs: list) -> tuple[float, int]:
    n = sum(counts)
    keep = [i for i, p in enumerate(probs) if p > 0.0]
    stat = sum((counts[i] - n * probs[i]) ** 2 / (n * probs[i]) for i in keep)
    return stat, len(keep) - 1


def _row(v=10, seed=3, scale=1.6):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(v, generator=g) * scale


def test_fresh_seeds_differ_and_fit_an_int64():
    seeds = [sd.fresh_seed() for _ in range(2000)]
    assert len(set(seeds)) == len(seeds), "two draws gave one seed"
    assert all(0 <= s < 1 << 63 for s in seeds)
    return "2000 draws, all distinct, all in [0, 2^63)"


def test_the_mode_decides_where_a_seed_is_drawn():
    def unseeded():
        return Sampler(temperature=1.0, top_p=0.95, top_k=20)

    for mode, q_free, drawn in (("auto", True, True), ("auto", False, False), (None, True, True),
                                ("on", False, True), ("on", True, True), ("off", True, False),
                                (" Auto ", True, True)):
        s = unseeded()
        got = sd.admit_seed(s, mode, q_free)
        assert (got is not None) == drawn, (mode, q_free, got)
        assert s.seed == got and s.coupled == drawn, (mode, q_free, s.seed)
        assert sd.keys_unseeded(mode, q_free) == drawn
    own = Sampler(temperature=1.0, top_k=20, seed=7)
    assert sd.admit_seed(own, "on", True) is None and own.seed == 7, "a client's seed is kept"
    greedy = Sampler(temperature=0.0, top_k=20)
    assert sd.admit_seed(greedy, "on", True) is None and greedy.seed is None, "greedy stays greedy"
    assert sd.admit_seed(None, "on", True) is None
    try:
        sd.admit_seed(unseeded(), "sometimes", True)
    except ValueError:
        pass
    else:
        raise AssertionError("a bad QWEN38_FRESH_SEED was taken")
    return "auto: drawn where q-free; on: always; off: never; seeded and greedy untouched"


def test_two_unseeded_requests_get_two_seeds_and_two_texts():
    row = torch.softmax(torch.zeros(64), dim=-1)          # a flat row: any two seeds disagree somewhere
    a, b = Sampler(temperature=1.0), Sampler(temperature=1.0)
    sa, sb = sd.admit_seed(a, "auto", True), sd.admit_seed(b, "auto", True)
    assert sa != sb
    ta = [a.pick_at(row, 100 + i) for i in range(32)]
    tb = [b.pick_at(row, 100 + i) for i in range(32)]
    assert ta != tb, "two requests with one prompt gave one text"
    again = Sampler(temperature=1.0, seed=sa)                # the recorded seed reproduces the text
    assert [again.pick_at(row, 100 + i) for i in range(32)] == ta
    return "two seeds, two texts; the drawn seed sent back gives the first text again"


def _spec(sm, rows, tokens, parents, start=40):
    return {"sampler": sm, "rows": rows, "tokens": tokens, "parents": parents, "start": start,
            "q": [None] * len(tokens), "picks": None}


def test_an_admitted_request_takes_the_keyed_walk():
    """Path selection: before admission an unseeded slot draws by multinomial (`round_walks` "draws"), after
    it the slot takes the keyed walk, the same path and tokens as a client-seeded request with that seed."""
    old_batch = S.BATCH
    S.BATCH = True
    calls = []
    real = S._keyed_draws

    def spy(*a, **k):
        calls.append(a[0])
        return real(*a, **k)

    S._keyed_draws = spy
    try:
        rows = torch.stack([_row(12, seed=i) for i in range(4)])
        tokens, parents = [5, 1, 7, 2], [-1, 0, 0, 1]
        plain = Sampler(temperature=0.9, top_p=0.95, top_k=6)
        S.round_walks([_spec(plain, rows, tokens, parents)])
        assert not calls, "an unseeded slot took the keyed walk before admission"
        adm = Sampler(temperature=0.9, top_p=0.95, top_k=6)
        seed = sd.admit_seed(adm, "auto", True)
        got = S.round_walks([_spec(adm, rows, tokens, parents)])[0]
        assert calls == [adm], "the admitted slot did not take the keyed walk"
        ref = Sampler(temperature=0.9, top_p=0.95, top_k=6, seed=seed)
        want = ref._tree_walk_keyed(ref.probs_rows(rows), tokens, parents, 40)
        assert got == ("walk", want), (got, want)
        # the lone slot's walk (`tree_walk_logits`) takes the keyed walk too
        lone = Sampler(temperature=0.9, top_p=0.95, top_k=6, seed=seed)
        assert lone.tree_walk_logits(rows, tokens, parents, start=40) == want
    finally:
        S._keyed_draws = real
        S.BATCH = old_batch
    return "unseeded: multinomial draws; admitted: the keyed walk, equal to the client-seeded one"


def test_the_keyed_walk_samples_the_filtered_distribution():
    """Over fresh seeds, the keyed draw at one position follows `_filter`'s probabilities (top_k, top_p and the
    temperature applied), as the unseeded multinomial does: chi-square within p = 0.001 for both."""
    logits = _row(10, seed=3)
    n = 6000
    probs = Sampler(temperature=0.8, top_p=0.9, top_k=6)._filter(logits.float())
    p = probs.tolist()
    keyed, dense = [0] * 10, [0] * 10
    for _ in range(n):
        sm = Sampler(temperature=0.8, top_p=0.9, top_k=6)
        sd.admit_seed(sm, "on", False)
        _, new = sm._tree_walk_keyed(sm.probs_rows(logits[None]), [0], [-1], 500)
        keyed[new[0]] += 1
        un = Sampler(temperature=0.8, top_p=0.9, top_k=6)
        dense[un.pick(probs)] += 1
    for name, c in (("keyed", keyed), ("dense", dense)):
        assert all(c[i] == 0 for i in range(10) if p[i] == 0.0), (name, c, p)
        stat, dof = _chi2(c, p)
        assert stat < _chi2_crit(dof), (name, stat, dof, c, p)
    sk, dof = _chi2(keyed, p)
    sdn, _ = _chi2(dense, p)
    return f"{n} seeds, {dof + 1} kept tokens: chi2 keyed {sk:.1f}, dense {sdn:.1f} (p=0.001 bound {_chi2_crit(dof):.1f})"


def test_the_keyed_walk_draws_each_position_afresh():
    """A two-deep chain [anchor, b]: where the first draw is b the walk goes on, and the second token must
    follow row 1's own filtered distribution, whatever noise the first position had (the positions' noise
    rows are independent): chi-square within p = 0.001, and the first token follows row 0."""
    l0 = torch.tensor([3.0, 0.2, 0.1, -0.5, -1.0, 0.4])
    l1 = _row(6, seed=11, scale=1.2)
    rows = torch.stack([l0, l1])
    b = 0
    first, second = [0] * 6, [0] * 6
    proto = Sampler(temperature=1.0, top_k=5)
    p0, p1 = (proto._filter(r.float()).tolist() for r in rows)
    for _ in range(6000):
        sm = Sampler(temperature=1.0, top_k=5)
        sd.admit_seed(sm, "on", False)
        _, new = sm._tree_walk_keyed(sm.probs_rows(rows), [9, b], [-1, 0], 200)
        first[new[0]] += 1
        if new[0] == b:
            second[new[1]] += 1
    for c, p in ((first, p0), (second, p1)):
        stat, dof = _chi2(c, p)
        assert stat < _chi2_crit(dof), (stat, dof, c, p)
    return f"first token over 6000 seeds, second over the {sum(second)} that took b: both within p=0.001"


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
