"""Choosing the BLOCK LENGTH once per step, and the drafter that goes with it.

Phase 5 measured both lengths on the same five prompts, with the reasoning block closed, and the
table is the whole argument for this file (SPEED-LEDGER, 17:00-17:14):

| workload | fine-tuned, 8 | fine-tuned, 16 |
|---|---|---|
| prose | 2.75 / 18.53 | 2.36 / 14.24 |
| chat  | 3.94 / 26.56 | 3.86 / 23.15 |
| code  | 5.45 / 36.95 | 6.10 / 36.55 |
| edit  | 7.64 / 52.62 | 13.26 / 80.56 |
| quote | 8.00 / 55.77 | 14.86 / 89.51 |

A sixteen-wide block verifies for 132.38 ms against an eight-wide one's 116.22, and its draft costs
about 32 ms against 26, so it has to commit **15.6 %** more tokens to break even. On `edit` it
commits 74 % more and on `quote` 86 %; on `prose` it commits 14 % fewer. Neither length is right,
and which one is right is a property of the text being written rather than of the model -- so the
engine should be choosing it, per step, and it needs no training to do so.

**The asymmetry that makes this cheap.** A block of sixteen that accepted `n` tokens tells you
exactly what a block of eight would have done with the same draft: `min(n, 7)`. The narrower block
is a prefix of the wider one, the verify pass computes the same rows for it, and the target's argmax
at row `i` does not depend on rows after `i`. So **every wide block scores the narrow option for
free**, and coming back down never costs an experiment.

The other direction is censored. A block of eight that accepted all seven is an observation that
the truth was "at least seven"; what sixteen would have committed is not in the data at all. That is
the 11:03 trap in a new costume -- *a policy that only learns about the option it took keeps taking
it* -- and the only honest fix is to pay for the information. So the router explores upward, and the
rate at which it explores is driven by the one signal that says information is worth buying: how
often the narrow block is hitting its own ceiling.

Three cheap signals, in the order they matter:

1. **the ceiling rate** -- the fraction of recent narrow blocks that accepted every slot. On `quote`
   this is 1.00 and on `prose` it is ~0.00, which is the regime distinction the table is made of;
2. **the free counterfactual** -- `min(n, 7)` on every wide block, which prices the narrow option
   without running it;
3. **the drafter's own lattice** -- once the wide drafter has run, the path probability of slots 8
   to 15 says whether the far half of the block is worth submitting at all, and that decision is
   free because the draft has already been paid for.

Everything is priced on the measured curve -- `verify(w)` from `tools/profile_block.py`, the draft
cost timed around the call the router itself makes, and the verify cost re-learned from the decode
loop that pays it (`on_verify`), because a policy constant belongs to the loop that pays it and a
microbenchmark once read a rollback at 23.4 ms that the loop reads at 6.4.
"""

from __future__ import annotations

import time

from engine.drafters import Drafter

# tools/profile_block.py --blocks 1,8,16 --reps 6, NVFP4 weights + fp8 head (SPEED-LEDGER 17:07).
# Keyed by the TOTAL block width, anchor included, which is what `forward_block` is handed.
VERIFY_MS_B = {1: 95.56, 8: 116.22, 16: 132.38}
# Timed around `DFlash2Drafter.propose` in this file; these are only the priors the router starts
# from and it replaces them with its own measurements after a handful of blocks.
DRAFT_MS_B = {8: 26.0, 16: 32.0}
ROLLBACK_MS_B = {8: 6.40, 16: 7.07}


def verify_ms_b(width: int, table: dict[int, float] | None = None) -> float:
    """The measured verify cost of a block of `width` rows, anchor included.

    Interpolated between measured points and flat-extrapolated past the ends. The interpolation is
    known to be wrong in a specific way -- the FP4 kernel tiles sixteen rows, so the curve is a
    staircase and not a line (SPEED-LEDGER 10:37, which cost 5 % of the prose row) -- and the
    router's action set is deliberately the two widths the staircase has steps at, so it never
    asks this function about a width between them.
    """
    table = table or VERIFY_MS_B
    keys = sorted(table)
    if width <= keys[0]:
        return table[keys[0]]
    if width >= keys[-1]:
        slope = (table[keys[-1]] - table[keys[-2]]) / (keys[-1] - keys[-2])
        return table[keys[-1]] + slope * (width - keys[-1])
    for lo, hi in zip(keys, keys[1:]):
        if lo <= width <= hi:
            f = (width - lo) / (hi - lo)
            return table[lo] + f * (table[hi] - table[lo])
    return table[keys[-1]]


class _Est:
    """A sample mean that becomes an exponential one once it has seen enough.

    The prior matters here in a way it does not in a long-running estimator: a generation is thirty
    to ninety blocks, so an arm that is judged on its first two blocks is judged on noise, and an
    arm that never sheds its prior is not judged at all. Sample mean until `warm`, EMA after.
    """

    def __init__(self, init: float, alpha: float = 0.25, warm: int = 4):
        self.value = float(init)
        self.alpha = alpha
        self.warm = warm
        self.n = 0
        self._sum = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        self._sum += float(x)
        if self.n <= self.warm:
            self.value = self._sum / self.n
        else:
            self.value += self.alpha * (float(x) - self.value)


class LengthRouter(Drafter):
    """Two block drafters at two block lengths, one of them chosen per step.

    `small` drafts an eight-wide block (seven proposals) and `large` a sixteen-wide one (fifteen).
    They are separate checkpoints -- the released weights were trained at eight and their top-1
    falls off past slot 7, so a wide block is only interesting with weights fine-tuned at that
    length -- and they are held at the same time. That costs 7.7 GB of the board's 121 and about
    1.5 ms a step of keeping both draft caches current; it buys the choice.

    Only ONE of them drafts on any given step. Running both would cost 26 to 32 ms of a 142 to
    164 ms block, which is a fifth of the budget spent on a proposal that is thrown away.
    """

    name = "lenrouter"
    wants_rows = True

    def __init__(self, small, large, *, alpha: float = 0.25,
                 explore_period: int = 32, ceiling_period: int = 4,
                 ceiling_trigger: float = 0.25,
                 verify_table: dict[int, float] | None = None,
                 draft_table: dict[int, float] | None = None,
                 width_trim: bool = True, fixed: int = 0, learn_cost: bool = True):
        self.small = small
        self.large = large
        self.eng = small.eng
        self.w_small = int(small.cfg.block_size)
        self.w_large = int(large.cfg.block_size)
        if self.w_large <= self.w_small:
            raise ValueError(f"large block {self.w_large} must exceed small block {self.w_small}")
        self.alpha = alpha
        self.explore_period = int(explore_period)
        self.ceiling_period = int(ceiling_period)
        self.ceiling_trigger = float(ceiling_trigger)
        self.width_trim = bool(width_trim)
        # Whether the router replaces its measured priors with what this process actually pays.
        # On the board it should; in a test whose drafters return instantly it must not, or the
        # draft call prices at zero and the wide drafter looks free.
        self.learn_cost = bool(learn_cost)
        # 0 = route. 8 or 16 pins the router to one configuration, which is how the fixed-length
        # baselines are measured through exactly the same code as the routed one.
        self.fixed = int(fixed)

        vt = dict(verify_table or VERIFY_MS_B)
        dt = dict(draft_table or DRAFT_MS_B)
        self.vms = {w: _Est(verify_ms_b(w, vt), alpha, warm=6)
                    for w in (self.w_small, self.w_large)}
        self.dms = {"s": _Est(dt.get(self.w_small, 26.0), alpha, warm=6),
                    "l": _Est(dt.get(self.w_large, 32.0), alpha, warm=6)}
        self.rb = dict(ROLLBACK_MS_B)

        # Committed tokens a block, including the model's own bonus token, per configuration.
        # ("l", small) is the wide drafter's block submitted narrow -- the option the free
        # counterfactual prices, and the one `width_trim` takes.
        self.acc = {("s", self.w_small): _Est(3.2, alpha),
                    ("l", self.w_large): _Est(3.6, alpha),
                    ("l", self.w_small): _Est(3.2, alpha)}
        # How often the narrow block accepted every slot it had: the signal that says the truth is
        # censored and information about the wide block is worth buying.
        self.ceiling = _Est(0.0, alpha)
        # How optimistic the wide drafter's own lattice has been, as one scalar.
        self.calib = _Est(1.0, alpha, warm=6)

        self.since = {self.w_small: 0, self.w_large: 0}
        self.blocks = 0
        self.last_key = None
        self.last_width = 0
        self.last_expected = 0.0
        self.last_forced = False
        # One bound method, held. `self._on_tap is self._on_tap` is False in CPython -- a bound
        # method is built fresh on every attribute access -- so an attach/detach pair that compares
        # them with `is` never detaches anything.
        self._tap_cb = self._on_tap
        self.stats = {"blocks": 0, "small": 0, "large": 0, "trims": 0, "forced": 0,
                      "tokens_small": 0, "tokens_large": 0, "declined": 0,
                      "ceiling_hits": 0, "width_hist": {}}
        self.attach()

    # --- the tap, shared -------------------------------------------------------------------

    def attach(self) -> None:
        """One tap, two consumers.

        `DFlash2Drafter.attach` overwrites `eng.tap` with its own callback, so two of them cannot
        both be attached; the engine emits one row per layer per forward and whoever holds the tap
        gets it. The router holds it and hands each row to both, which is what lets the drafter
        that did NOT propose this step stay current -- and it has to stay current, because its own
        cache is indexed by absolute position and a gap in it makes it decline for ever after.
        """
        self.eng.tap = self._tap_cb

    def detach(self) -> None:
        if self.eng.tap is self._tap_cb:
            self.eng.tap = None

    def _on_tap(self, h) -> None:
        self.small._on_tap(h)
        self.large._on_tap(h)

    # --- drafter interface -----------------------------------------------------------------

    def reset(self) -> None:
        """A new request: the drafters' caches go, and so does the evidence about this text.

        What the router has learned splits cleanly in two. The COSTS -- what a verify of eight or
        sixteen rows takes, what a draft call takes -- are properties of the board and they are
        kept, so a long-lived server's constants get better rather than being thrown away every
        request. The ACCEPTANCE is a property of the text being written, and carrying one request's
        beliefs into the next would start a fresh-prose request with a quotation's policy. So the
        arms, the ceiling rate and the calibration are cleared.
        """
        self.small.reset()
        self.large.reset()
        for key, width in list(self.acc):
            init = 3.6 if width == self.w_large else 3.2
            self.acc[(key, width)] = _Est(init, self.alpha)
        self.ceiling = _Est(0.0, self.alpha)
        self.calib = _Est(1.0, self.alpha, warm=6)
        self.since = {self.w_small: 0, self.w_large: 0}
        self.blocks = 0
        self.last_key = None
        self.last_width = 0
        # per request, so `report()` after a generation describes that generation
        self.stats = {k: ({} if isinstance(v, dict) else 0) for k, v in self.stats.items()}

    def prime(self, tokens: list[int]) -> None:
        for d in (self.small, self.large):
            if hasattr(d, "prime"):
                d.prime(tokens)

    def sync(self, tokens, hidden, first_pos, rows=None) -> None:
        for d in (self.small, self.large):
            if getattr(d, "wants_rows", False):
                d.sync(tokens, hidden, first_pos, rows=rows)
            else:
                d.sync(tokens, hidden, first_pos)

    def on_verify(self, width: int, ms: float) -> None:
        """What the decode loop actually paid for the block it just verified.

        `engine/spec.py` calls this after the block's logits have been read back, so the number
        includes the synchronisation the loop was going to pay anyway and nothing it was not.
        """
        if self.learn_cost and width in self.vms and ms > 0:
            self.vms[width].update(ms)

    # --- pricing ---------------------------------------------------------------------------

    def _arm(self, width: int) -> int:
        """Which of the two arms a submitted width belongs to."""
        return self.w_small if width <= self.w_small else self.w_large

    def _cost_ms(self, key: str, width: int, expected: float) -> float:
        p_reject = min(1.0, max(0.0, 1.0 - expected / max(width - 1, 1)))
        return (self.vms[width].value + self.dms[key].value
                + self.rb.get(width, 6.5) * p_reject)

    def _value(self, key: str, width: int) -> float:
        """Committed tokens per millisecond if every step looked like this one."""
        est = self.acc[(key, width)]
        expected = max(est.value - 1.0, 0.0)          # accepted drafts, not committed tokens
        return est.value / self._cost_ms(key, width, expected)

    def _wide_value(self) -> float:
        """What running the WIDE drafter is worth, at whichever width it would be submitted at.

        Two configurations share one draft call, and the router does not have to choose between
        them before paying for it: the sixteen-wide submission, and the same draft truncated to
        eight. The second is not a hypothetical -- `_trim` plays it -- and its acceptance is known
        exactly on every wide block that has ever run, because a narrow block is a prefix of a wide
        one. So the value of asking the wide drafter is the better of the two, and the width itself
        is settled afterwards, when the drafter's own scores for slots 8 to 15 are in hand.
        """
        v = self._value("l", self.w_large)
        if self.width_trim and self.acc[("l", self.w_small)].n:
            v = max(v, self._value("l", self.w_small))
        return v

    # --- the choice ------------------------------------------------------------------------

    def _choose(self, k: int) -> str:
        if self.fixed == self.w_small:
            return "s"
        if self.fixed == self.w_large:
            return "l"
        if k < self.w_large - 1:
            return "s"
        period = (self.ceiling_period if self.ceiling.value >= self.ceiling_trigger
                  else self.explore_period)
        if self.since[self.w_large] >= period:
            self.last_forced = True
            self.stats["forced"] += 1
            return "l"
        self.last_forced = False
        return "l" if self._wide_value() > self._value("s", self.w_small) else "s"

    # --- the lattice, for the width the wide drafter's own draft deserves --------------------

    def _path_prob(self, child) -> list[float] | None:
        """Per-slot conditional probability along the greedy path, from the drafter's lattice.

        `scores[l, p, c]` is the selector's score for candidate `c` at slot `l` given candidate `p`
        at slot `l-1`; a softmax along `c` of the row the greedy walk actually took is the cheapest
        thing that is monotone in the right direction, which is all this is used for. It is not a
        calibrated probability, and the one scalar `calib` is what stands between it and reality.
        """
        lat = getattr(child, "_lattice", None)
        if lat is None:
            return None
        import torch

        cand, scores = lat
        if scores.shape[0] < 2:
            return None
        idx = int(scores[0, 0].argmax())
        path = [idx]
        local = scores[1:].argmax(dim=-1)                       # [L-1, k]
        for e in range(local.shape[0]):
            idx = int(local[e, idx])
            path.append(idx)
        sel = torch.as_tensor(path, dtype=torch.long, device=scores.device)
        k = scores.shape[-1]
        rest = scores[1:].gather(1, sel[:-1].view(-1, 1, 1).expand(-1, 1, k)).squeeze(1)
        rows = torch.cat([scores[0, 0].unsqueeze(0), rest], dim=0).float()
        temp = float(getattr(child, "tree_temp", 1.0)) or 1.0
        lp = torch.log_softmax(rows / temp, dim=-1)
        return lp.gather(1, sel[:, None])[:, 0].exp().tolist()

    @staticmethod
    def _expected_prefix(probs: list[float], slots: int) -> float:
        """Expected accepted length of the first `slots` of a chain.

        A chain is accepted prefix-first, so slot `i` counts only if every slot above it was
        accepted: the expectation is the sum of the running products, which is what makes slot 0
        multiply every later term and why a maximiser of the total score is the wrong draft
        (SPEED-LEDGER 10:28, Viterbi at 3.22 against greedy's 4.30).
        """
        total, run = 0.0, 1.0
        for i in range(min(slots, len(probs))):
            run *= probs[i]
            total += run
        return total

    def _trim(self, draft: list[int], probs: list[float] | None) -> tuple[list[int], int]:
        """Submit the wide draft at whichever width prices better, now that it is paid for.

        This is the second half of the decision and the free half: the draft cost is sunk, so the
        only term that changes with the width is the verify, and the drafter's own scores for slots
        8 to 15 are the evidence for whether that half of the block is worth 16 ms.
        """
        if not self.width_trim or len(draft) < self.w_large - 1:
            return draft, min(len(draft) + 1, self.w_large)
        if probs is None:
            # No lattice to read -- the drafter is running without its selector. The width is then
            # decided on the two arms' own histories, which is the same comparison one step coarser.
            if (self.acc[("l", self.w_small)].n
                    and self._value("l", self.w_small) > self._value("l", self.w_large)):
                self.stats["trims"] += 1
                return draft[:self.w_small - 1], self.w_small
            return draft, self.w_large
        c = max(self.calib.value, 0.05)
        full = min(c * self._expected_prefix(probs, self.w_large - 1), float(self.w_large - 1))
        narrow = min(c * self._expected_prefix(probs, self.w_small - 1), float(self.w_small - 1))
        v_full = (full + 1.0) / self._cost_ms("l", self.w_large, full)
        v_narrow = (narrow + 1.0) / self._cost_ms("l", self.w_small, narrow)
        if v_narrow > v_full:
            self.stats["trims"] += 1
            return draft[:self.w_small - 1], self.w_small
        return draft, self.w_large

    # --- proposing -------------------------------------------------------------------------

    def propose(self, context: list[int], k: int) -> list[int]:
        if k <= 0:
            return []
        key = self._choose(k)
        child = self.small if key == "s" else self.large
        want = (self.w_small if key == "s" else self.w_large) - 1
        t0 = time.perf_counter()
        draft = child.propose(context, min(k, want))
        if not draft and key == "l":
            # The wide drafter declines where the narrow one would not only at the very end of a
            # sequence, where its block runs past `max_len`. Fall back rather than take the
            # one-token path, which is the expensive one.
            key, child, want = "s", self.small, self.w_small - 1
            draft = child.propose(context, min(k, want))
        if self.learn_cost:
            self.dms[key].update((time.perf_counter() - t0) * 1e3)
        if not draft:
            self.stats["declined"] += 1
            self.last_key, self.last_width, self.last_expected = None, 0, 0.0
            return []
        probs = self._path_prob(child) if key == "l" else None
        if key == "l" and not self.fixed:
            draft, width = self._trim(draft, probs)
        else:
            width = len(draft) + 1
        self.last_key, self.last_width = key, width
        self.last_expected = (self._expected_prefix(probs, width - 1) if probs is not None
                              else 0.0)
        self.blocks += 1
        self.stats["blocks"] += 1
        self.stats["small" if key == "s" else "large"] += 1
        self.stats["width_hist"][width] = self.stats["width_hist"].get(width, 0) + 1
        for w in self.since:
            self.since[w] = 0 if w == width else self.since[w] + 1
        return draft

    def observe(self, tokens: list[int]) -> None:
        """Learn from the block, including about the width that was not submitted.

        `tokens` is what the loop committed: the accepted prefix of the draft plus the model's own
        token, so `len(tokens)` is the block's yield and `len(tokens) - 1` is how much of the draft
        survived. Three things are learned from it and only the first needed the block to be run at
        this configuration.
        """
        if self.last_key is None or self.last_width == 0:
            return
        committed = len(tokens)
        accepted = committed - 1
        width = self.last_width
        key = self.last_key
        # The last block of a generation is whatever is left of the token budget, so the submitted
        # width can be any number between two and the arm's own. Its evidence belongs to the arm it
        # came from, not to a width the router can never choose on purpose.
        self.acc[(key, self._arm(width))].update(committed)
        self.stats["tokens_small" if key == "s" else "tokens_large"] += committed

        if width <= self.w_small:
            hit = 1.0 if accepted >= width - 1 else 0.0
            self.ceiling.update(hit)
            self.stats["ceiling_hits"] += int(hit)
        elif key == "l":
            # The free counterfactual. The narrow block is a prefix of the wide one and the
            # target's argmax at row i does not depend on rows after i, so this is not an estimate
            # of what eight would have committed -- it is what eight would have committed.
            self.acc[("l", self.w_small)].update(min(accepted, self.w_small - 1) + 1)

        if self.last_expected > 0:
            self.calib.update(min(accepted, width - 1) / self.last_expected)
        self.last_key, self.last_width, self.last_expected = None, 0, 0.0

    # --- reporting -------------------------------------------------------------------------

    def report(self) -> str:
        s, l = self.stats["small"], self.stats["large"]
        a_s = self.stats["tokens_small"] / s if s else 0.0
        a_l = self.stats["tokens_large"] / l if l else 0.0
        return (f"lenrouter {self.w_small}x{s} ({a_s:.2f} tok/block) "
                f"{self.w_large}x{l} ({a_l:.2f} tok/block) "
                f"trims {self.stats['trims']} forced {self.stats['forced']} "
                f"ceiling {self.ceiling.value:.2f} calib {self.calib.value:.2f} "
                f"verify {self.vms[self.w_small].value:.1f}/{self.vms[self.w_large].value:.1f} ms "
                f"draft {self.dms['s'].value:.1f}/{self.dms['l'].value:.1f} ms "
                f"widths {dict(sorted(self.stats['width_hist'].items()))}")
