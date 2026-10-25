"""The runner-owned round (SPD-201, stage A: Python): one thread runs every registered slot's decode round.

With side slots each decoding slot's thread runs its own round: it drafts, submits its block to the
round's barrier, waits, walks, commits, syncs its drafter and drafts again, every step taking the GPU turn
and the GIL in turn with seven others. At 8 slots that is ~47 ms of a 204 ms round on the critical path
(SPD-161's profiles), most of it Python and thread hand-offs. Here a slot whose request qualifies
(`eligible`) hands its decode state to the runner at its first block and its thread only streams what the
runner pushes; the runner holds the turn for a round and does, for all its tasks at once:

  1. each task's proposal up to its draft (the length router's generator, run to the point where the draft
     is queued: `DRAFT_MEET` returns a list the runner fills), then ONE packed draft over every drafter of a
     checkpoint, then each proposal's rest (its tree);
  2. one joint verify pass over every tree, cut to the family's row cap and tree cap (the joint graphs);
  3. the round's walks in one batch (`engine.sample.round_walks`, each slot's penalties first);
  4. each task's tail, bound to its slot (`slots.bind`): the end cut, the commit, the drafter's sync and
     observe, the penalties' and pattern stop's bookkeeping, and the tokens onto the slot's queue.

Every statement of a task's round is the slot loop's (server/app.py `_generate_stream`, the tree branch), in
its order, so the tokens are the ones the slot's own loop would have written: the proof is
tests/test_round_runner.py (texts equal on the CPU) and the board's seeded byte identity.

Not taken (the slot keeps its own loop, `why_not` names the reason, /health counts it): a supervisor, a
probe, log-probabilities, a reasoning budget or stall arms (something that can close the block), a request
without a tree drafter. Slot 0's side lane (server/preempt.py) is served by the runner at each round's start,
the safe point the slot loop serves it at (no proposal in flight). A passive reasoning block (`ThinkBudget` with neither) is followed
as the slot loop follows it: `inside` masks the penalties, every block is observed. QSE_ROUND_RUNNER=1
turns it on.
"""

from __future__ import annotations

import os
import queue
import threading
import time

import torch

ON = os.environ.get("QSE_ROUND_RUNNER", "1") == "1"
#: SPD-201 (row invariance): a round of one task verifies through the family's joint pass too, so a request's
#: rows take the same statements alone as among other slots' (the lone pass is another kernel set). 0: the
#: lone `forward_tree`
JOINT1 = os.environ.get("QSE_RUNNER_JOINT1", "0") == "1"
#: SPD-201: the tails' row copies (the drafters' tap rows, the hidden rows, the cache compaction) recorded and
#: run as one launch at the end of the tails (engine/rowcopy.py); 0: each its own torch statement
ROWCOPY = os.environ.get("QSE_RUNNER_ROWCOPY", "1") == "1"
#: SPD-201: a round's tails in two halves around the next round's packed draft: what the draft reads first (the
#: accepted tokens, positions, the drafter syncs, which tasks end), then the draft is queued, then the rest (the
#: target's commit, penalties, stop patterns, the stream) runs on the host while the GPU drafts; 0: one tail a task
PIPE = os.environ.get("QSE_RUNNER_PIPE", "1") == "1"
#: SRV-709 (a test knob): the pool refuses a lane once after it wrote this many tokens (0: never), so a test
#: can watch a lane give way and go on; QSE_KV_TEST_REFUSE_SLOTS limits it to these slots (comma list)
#: SPD-215: a joint pass hands each drafter that asks (`joint_priced`) its share of the pass by rows
JOINT_PRICE = os.environ.get("QSE_RUNNER_JOINT_PRICE", "0") == "1"
#: SPD-225: a joint round keeps the rows where its expected tokens a ms are the most, on the pass prices the
#: runner learns from its own rounds (server/slots.py `priced_cut`); every QSE_RUNNER_PRICED_EXPLORE-th priced
#: round keeps a random row budget instead, so the prices of every row count stay measured
PRICED_CUT = os.environ.get("QSE_RUNNER_PRICED_CUT", "0") == "1"
PRICED_EXPLORE = int(os.environ.get("QSE_RUNNER_PRICED_EXPLORE", "8") or 8)
#: SPD-235: a round's MTP drafts in one packed pass (engine/methods/draft_meet.py): the lanes' heads stage their
#: drafts and the runner runs them together after the proposals (heads with `run_packed`, the MoE families' MTP
#: heads); 0: each head its own graph. Served 2026-10-25, C8 of the two MoE families: 74.8 -> 83.6, 72.8 -> 79.7
MTP_PACK = os.environ.get("QSE_RUNNER_MTP_PACK", "1") == "1"
TEST_REFUSE_AFTER = int(os.environ.get("QSE_KV_TEST_REFUSE_AFTER", "0") or 0)
TEST_REFUSE_SLOTS = {int(x) for x in (os.environ.get("QSE_KV_TEST_REFUSE_SLOTS") or "").split(",") if x.strip()}

#: requests the runner did not take, by reason (`why_not`), for /health
SKIPS: dict = {}



def why_not(*, sup, probe, lpr, think, ahead, drafter, lane, turn) -> str | None:
    """None when the runner may take this slot's decode, else the reason it may not (counted)."""
    why = ("supervisor" if sup is not None else "probe" if probe is not None
           else "logprobs" if lpr is not None
           else "think_budget" if think is not None and (think.budget or think.arms_cfg is not None)
           else "launched" if ahead is not None else "no_drafter" if drafter is None
           else "no_turn" if turn is None else None)
    if why is not None:
        SKIPS[why] = SKIPS.get(why, 0) + 1
    return why


class Task:
    """One slot's decode state, owned by the runner while the request is registered."""

    def __init__(self, slot, eng, drafter, sampler, pen, pstop, ctx, pos, tok, n_out, max_new, eos, bs, k,
                 deadline, device, think=None):
        self.think = think                  # a passive reasoning block (no budget, no arms) or None
        self.slot, self.eng, self.drafter, self.sampler, self.pen, self.pstop = slot, eng, drafter, sampler, pen, pstop
        self.ctx, self.pos, self.tok, self.n_out, self.max_new, self.eos = ctx, pos, tok, n_out, max_new, eos
        self.bs, self.k, self.deadline, self.device = bs, k, deadline, device
        self.q: queue.SimpleQueue = queue.SimpleQueue()
        self.cancel = False                 # the client left: dropped at the next round
        self.done = False
        self.steps = None                   # the proposal generator stopped at its draft
        self.pre = False                    # PIPE: the last round's tails queued this task's draft
        self.tree = None
        self.err = None

    def push(self, toks) -> None:
        self.q.put(("tokens", list(toks)))

    def end(self, why: str) -> None:
        self.done = True
        self.q.put(("end", why))


class Runner:
    def __init__(self, pool, app):
        self.pool, self.app = pool, app
        self.tasks: list[Task] = []
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        self.thread = None
        self.stats = {"rounds": 0, "tasks": 0, "joint": 0, "plain": 0, "errors": 0, "lane_jobs": 0,
                      # rounds by the slots in their verify pass (the concurrency proof: N rows a round)
                      "joint_hist": {}}
        # host ms a round by phase (summed; `report`): the proposals to their drafts, the packed draft's
        # issue, the proposals' rest (it reads the draft back), the verify's issue, the walk (it reads the
        # pass back), the tails
        self.ms = {k: 0.0 for k in ("prop", "packed", "rest", "verify", "walk", "tails", "round")}
        # the same phases on the GPU's timeline (SPD-201 host cut, step a): a CUDA event at the round's start and
        # at each phase's end, read one round later (never waited on). A segment's GPU time is its end event
        # minus the one before: the packed draft and the verify pass are the GPU's own work, the others the
        # time the GPU waited for the host (or ran its small kernels); `between` is one round's end to the next
        # round's start
        self.gpu = {k: 0.0 for k in ("prop", "packed", "rest", "verify", "walk", "tails", "between", "period")}
        self.gpu_n = 0
        self._ev: list = []                 # this round's (label, event)
        self._ev_done: list = []            # finished rounds' event lists, read when complete
        self._ev_last = None                # the last round's final event
        # steady state by the slots in a round's verify (SPD-201): each round's GPU period (one round end to the
        # next) and tokens, the last 512 a slot count; `report` gives the medians and their tokens a second
        self._steady: dict = {}
        # SPD-239: the same phases by the slots in a round's verify ({slots: {phase: ms summed}}, "n" the rounds),
        # host and GPU, so a round of 8 lanes is read apart from the mix
        self._ms_e: dict = {}
        self._gpu_e: dict = {}
        self._tails_e: dict = {}
        self._round_e = 0
        self._round_tok = 0
        self._round_rows = 0                # the rows of the round's joint pass (0: none, or more than one)
        self._round_exp = 0.0               # the kept drafts' expected tokens (scores), for the calibration
        self._round_acc = 0                 # the drafts the walks accepted
        from server import slots as slots_mod
        self.prices = slots_mod.RowPrices()
        import random
        self._rng = random.Random(225)
        self._collect: list | None = None    # the runner thread's pending draft calls, while proposals run
        self._mtp_collect: list | None = None    # SPD-235: draft heads whose drafts the runner packs
        if MTP_PACK:
            from engine.methods import draft_meet
            draft_meet.MEET = self.mtp_meet

    # ---- registration (the slot's thread) ---------------------------------------------------
    def add(self, task: Task) -> None:
        with self.cv:
            self.tasks.append(task)
            self.stats["tasks"] += 1
            if self.thread is None or not self.thread.is_alive():
                self.thread = threading.Thread(target=self._loop, name="round-runner", daemon=True)
                self.thread.start()
            self.cv.notify_all()

    # ---- the draft hook (dflash2.DRAFT_MEET, from the runner thread) -------------------------
    def meet(self, drafter, anchor, pos0):
        """While the runner drives proposals, a draft call joins the round's packed draft: a list the
        runner fills once every task has reached its draft (the generator reads it after its yield)."""
        if self._collect is None or threading.current_thread() is not self.thread:
            return None
        out: list = []
        self._collect.append((drafter, anchor, pos0, out))
        return out

    def mtp_meet(self, head) -> bool:
        """SPD-235 (`draft_meet.MEET`): from the runner thread while it collects, the head's draft joins the
        round's packed pass."""
        if self._mtp_collect is None or threading.current_thread() is not self.thread:
            return False
        self._mtp_collect.append(head)
        return True

    # ---- the loop ---------------------------------------------------------------------------
    def _loop(self) -> None:
        turn = self.pool.turn
        while True:
            with self.cv:
                live = [t for t in self.tasks if not t.done]
                for t in self.tasks:
                    if t.cancel and not t.done:
                        t.end("cancelled")
                self.tasks = [t for t in self.tasks if not t.done]
                if not self.tasks:
                    self.cv.wait(timeout=0.05)
                    if not self.tasks:
                        continue
                live = list(self.tasks)
            if turn is not None:
                turn.acquire("runner")
            try:
                self._round(live)
            except BaseException as exc:                                # noqa: BLE001
                self.stats["errors"] += 1
                print(f"[runner] the round failed ({type(exc).__name__}: {exc}); its tasks end", flush=True)
                import traceback
                traceback.print_exc()
                for t in live:
                    t.err = exc
                    if not t.done:
                        t.end("error")
            finally:
                if turn is not None:
                    turn.release("runner")

    def _bind(self, t):
        from server import slots as slots_mod
        slots_mod.bind(None if t.slot.index == 0 else t.slot)

    def report(self) -> dict:
        n = max(1, self.stats["rounds"])
        g = max(1, self.gpu_n)
        steady = {}
        for e, st in sorted(self._steady.items()):
            if len(st) >= 8:
                per = sorted(p for p, _ in st)[len(st) // 2]
                tok = sorted(t for _, t in st)[len(st) // 2]
                mean_tok = sum(t for _, t in st) / len(st)
                steady[e] = {"rounds": len(st), "period_ms": round(per, 2), "tokens": tok,
                             "tok_s": round(1e3 * mean_tok / per, 1)}
        return dict(self.stats, skips=dict(SKIPS),
                    ms_a_round={k: round(v / n, 2) for k, v in self.ms.items()},
                    gpu_ms_a_round={k: round(v / g, 2) for k, v in self.gpu.items()}, gpu_rounds=self.gpu_n,
                    steady=steady, prices=self.prices.report() if PRICED_CUT else None,
                    ms_by_slots=self._by_e(self._ms_e), gpu_ms_by_slots=self._by_e(self._gpu_e),
                    tails_by_slots=self._by_e(self._tails_e))

    @staticmethod
    def _by_e(d: dict) -> dict:
        """SPD-239: {slots: {phase: ms a round}} for slot counts with 8 rounds or more."""
        return {e: {k: (v if k == "n" else round(v / v_n, 2)) for k, v in sorted(ph.items())}
                for e, ph in sorted(d.items()) if (v_n := ph["n"]) >= 8}

    def _t(self, key, t0):
        t1 = time.perf_counter()
        self.ms[key] += (t1 - t0) * 1e3
        self._mark(key)
        return t1

    def _mark(self, label) -> None:
        if torch.cuda.is_available():
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self._ev.append((label, ev))

    def _gpu_read(self) -> None:
        """The finished rounds' GPU segments into `gpu` (a round whose last event is not done waits)."""
        keep = []
        for rnd, prev, e, tok, rows in self._ev_done:
            if not rnd[-1][1].query():
                keep.append((rnd, prev, e, tok, rows))
                continue
            if prev is not None and prev.query() and e:
                per = prev.elapsed_time(rnd[-1][1])
                st = self._steady.setdefault(e, [])
                st.append((per, tok))
                if len(st) > 512:
                    del st[0]
                if rows and e > 1:
                    # SPD-225: the pass's GPU segment (its issue to its end event) at these lanes and rows
                    seg = [a.elapsed_time(b) for (_, a), (lab, b) in zip(rnd, rnd[1:]) if lab == "verify"]
                    if len(seg) == 1:
                        self.prices.note(e, rows, seg[0], per)
            ge = self._gpu_e.setdefault(e, {"n": 0})
            ge["n"] += 1
            for (_, a), (lab, b) in zip(rnd, rnd[1:]):
                if lab in self.gpu:
                    self.gpu[lab] += a.elapsed_time(b)
                    ge[lab] = ge.get(lab, 0.0) + a.elapsed_time(b)
            if prev is not None and prev.query():
                self.gpu["between"] += prev.elapsed_time(rnd[0][1])
                self.gpu["period"] += prev.elapsed_time(rnd[0][1]) + rnd[0][1].elapsed_time(rnd[-1][1])
                ge["between"] = ge.get("between", 0.0) + prev.elapsed_time(rnd[0][1])
            self.gpu_n += 1
        self._ev_done = keep[-8:]

    def _round(self, tasks: list[Task]) -> None:
        app = self.app
        from engine.drafters import tree_steps
        from engine.model import h2d
        from server import slots as slots_mod
        self.stats["rounds"] += 1
        self._gpu_read()
        self._ev = []
        self._round_e, self._round_tok = 0, 0
        self._round_rows, self._round_exp, self._round_acc, passes = 0, 0.0, 0, 0
        self._mark("start")
        ms0 = dict(self.ms)
        t_r = t0 = time.perf_counter()
        # 1. proposals to their drafts, one packed draft, the proposals' rest
        for t in tasks:
            if t.slot.index == 0 and not t.done:
                # slot 0's side lane at the round's safe point, as the slot loop serves it between blocks
                self._bind(t)
                lane = app.STATE.get("lane")
                if lane is not None and lane.jobs:
                    if t.pre:
                        # PIPE: the draft the last tails queued is read before the lane runs on this slot
                        t.tree, t.steps = app._collect(t.steps, t.tree), None
                    self.stats["lane_jobs"] += lane.serve("decode")
        for t in tasks:
            if t.pre and not t.done and t.deadline is not None and t.deadline.expired():
                t.end("deadline")
        t0 = self._propose([t for t in tasks if not t.pre], t0)
        for t in tasks:
            t.pre = False
            if t.done:
                if t.steps is not None:
                    t.steps.close()             # a draft queued for a task that ended after it
                    t.steps = None
                continue
            self._bind(t)
            t.tree = app._collect(t.steps, t.tree)
            t.steps = None
            if t.tree is not None:
                t.tree = t.tree.truncate(t.eng.max_len - t.pos)
        # 2. the joint verify pass (a task without a draft takes its plain step alone)
        t0 = self._t("rest", t0)
        live = [t for t in tasks if not t.done]
        if len(live) > 1:
            from engine.tree import DraftTree
            for t in live:
                if ((t.tree is None or t.tree.n_draft == 0) and getattr(t.eng.caps, "joint_plain", False)):
                    # ENG-241: no draft this round, other lanes decoding: the plain step joins the round's
                    # pass as its anchor alone, so the lanes read the weights once (the slot loop's
                    # `joint_anchor`; NemotronH, Flint, LFM2.5 serve it as one joint decode)
                    t.tree = DraftTree(tokens=[int(t.tok)], parents=[-1])
        group = [t for t in live if t.tree is not None and (t.tree.n_draft > 0 or (
            len(live) > 1 and getattr(t.eng.caps, "joint_plain", False)))]
        for t in tasks:
            if not t.done and t not in group:
                self._bind(t)
                self._plain(t)
        if group:
            # SRV-707: more decoding lanes than one pass holds (two rows each) verify in more passes
            meet = self.pool.meeting
            k = meet.per_pass() if meet is not None and hasattr(meet, "per_pass") else len(group)
            for i in range(0, len(group), max(1, k)):
                self._verify(group[i:i + k])
                passes += 1
            if passes > 1:
                self._round_rows = 0
        if self._round_exp > 0:
            self.prices.note_accept(self._round_acc, self._round_exp)
        self._bind_none()
        self._t("round", t_r)
        me = self._ms_e.setdefault(self._round_e, {"n": 0})
        me["n"] += 1
        for k, v in self.ms.items():
            me[k] = me.get(k, 0.0) + v - ms0[k]
        if self._ev:
            self._ev_done.append((self._ev, self._ev_last, self._round_e, self._round_tok, self._round_rows))
            self._ev_last = self._ev[-1][1]
        if self.stats["rounds"] % 200 == 0:
            print(f"[runner] {self.report()}", flush=True)

    def _propose(self, tasks: list, t0: float, deadline: bool = True) -> float:
        """Each task's proposal to its draft, then one packed draft for them (queued, not read). `deadline`
        False (PIPE, inside a tail): the deadline is the next round's to check, after the stream."""
        from engine.drafters import tree_steps
        app = self.app
        self._collect = []
        if MTP_PACK and sum(1 for t in tasks if not t.done) > 1:
            self._mtp_collect = []
        for t in tasks:
            if t.done:
                continue
            self._bind(t)
            if deadline and t.deadline is not None and t.deadline.expired():
                t.end("deadline")
                continue
            app._row_cap(self.pool, t.drafter)
            t.steps, t.tree = app._launch(tree_steps(t.drafter, t.ctx, min(t.k, t.max_new - t.n_out)))
        calls, self._collect = self._collect, None
        heads, self._mtp_collect = self._mtp_collect, None
        self._bind_none()
        t0 = self._t("prop", t0)
        if heads:
            by: dict = {}
            for h in heads:
                by.setdefault((type(h), id(h.fn)), []).append(h)
            for grp in by.values():
                type(grp[0]).run_packed(grp)
        if calls:
            self._packed(calls)
        return self._t("packed", t0)

    def _bind_none(self):
        from server import slots as slots_mod
        slots_mod.bind(None)

    def _packed(self, calls: list) -> None:
        """`_run_draft_round` for the collected calls: grouped by checkpoint, in slot order."""
        from engine.drafters.draft_graph import packed_draft
        by: dict = {}
        for c in calls:
            by.setdefault(id(c[0].module.w), []).append(c)
        for grp in by.values():
            if len(grp) == 1:
                d, a, p, out = grp[0]
                out[:] = list(d.graph_run(a, p))
                continue
            res = packed_draft([(d, a, p) for d, a, p, _ in grp])
            for (_, _, _, out), r in zip(grp, res):
                out[:] = list(r)

    def _verify(self, group: list[Task]) -> None:
        app = self.app
        from engine.model import forward_tree_multi, h2d
        from engine.sample import round_walks
        from server import slots as slots_mod
        meet = self.pool.meeting
        group.sort(key=lambda t: t.slot.index)
        self._round_e = len(group)
        h = self.stats["joint_hist"]
        h[len(group)] = h.get(len(group), 0) + 1
        trees = [t.tree for t in group]
        if (len(group) > 1 or JOINT1) and meet is not None:
            tc = getattr(meet, "tree_cap", None)
            cap = tc(len(group), meet.max_rows) if tc is not None else None
            # SPD-233: the family's rows at this many lanes (`joint_policy` "rows_at"), never above the cap
            rows = min(meet.max_rows, max(2 * len(group),
                                          (getattr(meet, "rows_at", None) or {}).get(len(group), meet.max_rows)))
            if sum(len(x.parents) for x in trees) > rows or (
                    cap is not None and any(len(x.parents) > cap for x in trees)) or meet.joint_for(len(group)):
                keeps = slots_mod.joint_cut(trees, rows, cap)
                trees = [x.subset(kp) if len(kp) < len(x.tokens) else x for x, kp in zip(trees, keeps)]
            if PRICED_CUT and len(group) > 1:
                # SPD-225: the rows where the round's expected tokens a ms are the most (an exploring round:
                # a random row budget), on the pass prices measured at these lanes
                pr = self.prices
                budget = None
                if pr.cost(len(group), rows) is None or (pr.priced + pr.explored) % PRICED_EXPLORE == 0:
                    budget = self._rng.randint(len(group), rows)
                alone = all(getattr(t.eng.caps, "joint_plain", False) for t in group)
                got = slots_mod.priced_cut(trees, pr, rows, cap, budget, anchor_alone=alone)
                if got is not None:
                    keeps, _, offered = got
                    if budget is None:
                        pr.priced += 1
                        pr.rows_offered += offered
                        pr.rows_kept += sum(len(kp) for kp in keeps)
                    else:
                        pr.explored += 1
                    trees = [x.subset(kp) if len(kp) < len(x.tokens) else x for x, kp in zip(trees, keeps)]
            # SPD-223: no tree deeper than the family's joint kernels carry (a 32-deep lookup chain past
            # the grouped GDN's 16 levels: a ValueError eager, an illegal address from a joint graph)
            trees = [slots_mod.depth_cut(x, getattr(meet, "max_depth", None)) for x in trees]
        # SRV-709: every lane's pages before the joint pass, so a pool that cannot give them stops that
        # lane alone (its request gives way and goes on: server/app.py `Handler._recomputing`), not the round
        from engine import kvpool
        if kvpool.POOL is not None and all(callable(getattr(t.eng.kv, "ensure", None)) for t in group):
            # a family whose KV has no `ensure` (its pages are committed inside the pass) is left as it was
            kept = []
            for t, x in zip(group, trees):
                self._bind(t)
                try:
                    if (TEST_REFUSE_AFTER and t.n_out >= TEST_REFUSE_AFTER
                            and (not TEST_REFUSE_SLOTS or t.slot.index in TEST_REFUSE_SLOTS)
                            and not getattr(t.eng, "_test_refused", False)):
                        t.eng._test_refused = True
                        raise kvpool.PoolFull(f"QSE_KV_TEST_REFUSE_AFTER: slot {t.slot.index} after "
                                              f"{t.n_out} tokens")
                    t.eng.kv.ensure(t.pos + len(x.tokens))
                except kvpool.PoolFull as exc:
                    t.err = exc
                    t.end("error")
                    continue
                kept.append((t, x))
            if len(kept) < len(group):
                if not kept:
                    return
                group, trees = [t for t, _ in kept], [x for _, x in kept]
        if len(group) > 1:
            self._round_rows = sum(len(x.tokens) for x in trees)
            self._round_exp += sum(sum(x.scores[1:]) for x in trees)
        t0 = time.perf_counter()
        blocks = [h2d(x.tokens, torch.long, t.device) for x, t in zip(trees, group)]
        if len(group) == 1 and not (JOINT1 and getattr(group[0].eng, "forward_tree_multi", None) is not None):
            t = group[0]
            self._bind(t)
            outs = [t.eng.forward_tree(blocks[0], trees[0].parents, start=t.pos)]
            on_verify = getattr(t.drafter, "on_verify", None)
            if on_verify is not None:
                # a lone block's time, as the slot loop hands it (a joint round's is the round's)
                on_verify(trees[0].n_draft + 1, (time.perf_counter() - t0) * 1e3)
        else:
            multi = getattr(group[0].eng, "forward_tree_multi", None) or forward_tree_multi
            outs = multi([(t.eng, b, x.parents, t.pos) for t, b, x in zip(group, blocks, trees)])
            self.stats["joint"] += 1
            if JOINT_PRICE:
                # a drafter that prices its rounds by its verify (the MTP tree drafter's StairCut) gets its
                # share of the joint pass by rows; without it the joint rounds go unpriced
                ms = (time.perf_counter() - t0) * 1e3
                total = sum(len(x.tokens) for x in trees) or 1
                for t, x in zip(group, trees):
                    ov = getattr(t.drafter, "on_verify", None)
                    if ov is not None and getattr(t.drafter, "joint_priced", False):
                        ov(len(x.tokens), ms * len(x.tokens) / total)
        t0 = self._t("verify", t0)
        # 3. the walks in one batch, each slot's penalties first
        specs = []
        for t, x, lg in zip(group, trees, outs):
            if t.pen is not None:
                t.pen.mask = bool(t.think is not None and t.think.inside)
                t.pen.apply_tree(lg, x)
            sm = t.sampler if t.sampler is not None and t.sampler.on else None
            specs.append({"sampler": sm, "rows": lg, "tokens": x.tokens, "parents": x.parents,
                          "start": len(t.ctx), "q": x.q, "picks": None})
        walked = round_walks(specs)
        t0 = self._t("walk", t0)
        # 4. each task's tail, as the slot loop runs it
        from engine import rowcopy
        with (rowcopy.recording() if ROWCOPY else _nothing()):
            self._tails(group, trees, outs, walked)
        self._t("tails", t0)

    def _tails(self, group, trees, outs, walked) -> None:
        if PIPE:
            t0 = time.perf_counter()
            halves = [self._tail_a(t, x, path, new) for t, x, path, new in self._paths(group, trees, outs, walked)]
            go = [t for t, h in zip(group, halves) if h[2] is None]
            t1 = time.perf_counter()
            self._propose(go, t1, deadline=False)
            for t in go:
                t.pre = True
            t2 = time.perf_counter()
            for t, h in zip(group, halves):
                self._bind(t)
                self._tail_b(t, *h)
            # SPD-239: the tails' host ms by half (the walks' paths and drafter syncs, the next draft's issue, the
            # commits), summed by the slots in the round (`report`'s tails_by_slots)
            ta = self._tails_e.setdefault(len(group), {"n": 0, "a": 0.0, "draft": 0.0, "b": 0.0})
            ta["n"] += 1
            ta["a"] += 1e3 * (t1 - t0)
            ta["draft"] += 1e3 * (t2 - t1)
            ta["b"] += 1e3 * (time.perf_counter() - t2)
            return
        for t, x, path, new in self._paths(group, trees, outs, walked):
            self._tail(t, x, path, new)

    def _paths(self, group, trees, outs, walked):
        # ENG-E8: the walk every model's blocks take (a family's engine need not carry `accept_tree`)
        from engine.methods import walk as walk_mod
        for t, x, lg, w in zip(group, trees, outs, walked):
            self._bind(t)
            if w is not None and w[0] == "walk":
                path, new = w[1]
            elif w is not None:
                path, new = walk_mod.accept_tree(x, w[1])
            elif t.sampler is not None and t.sampler.on:
                path, new = t.sampler.tree_walk_logits(lg, x.tokens, x.parents, start=len(t.ctx), q=x.q)
            else:
                path, new = walk_mod.accept_tree(x, lg.argmax(-1).tolist())
            if len(group) > 1:
                self._round_acc += len(path) - 1
            yield t, x, path, new

    def _tail(self, t: Task, tree, path, new) -> None:
        from engine.model import h2d
        app = self.app
        accepted = len(path) - 1
        cut = app._end_cut(new, t.eos, t.max_new - t.n_out)
        if cut < len(new):
            path, new = path[:cut + 1], new[:cut]
        t.eng.commit_tree(path)
        d = t.drafter
        if hasattr(d, "sync"):
            from engine import rowcopy
            if rowcopy.active():
                hidden = rowcopy.rows_of(t.eng.hidden_post_norm[0], path)
                if not getattr(d, "hidden_free", False):
                    rowcopy.flush()             # this drafter's sync reads the hidden rows at once
            else:
                hidden = t.eng.hidden_post_norm[0, h2d(path, torch.long, t.device)]
            toks = [int(tree.tokens[i]) for i in path]
            if getattr(d, "wants_rows", False):
                d.sync(toks, hidden, t.pos, rows=path)
            else:
                d.sync(toks, hidden, t.pos)
        t.pos += len(path)
        t.bs.block(max(tree.depths()), accepted)
        d.observe(new)
        if t.pen is not None:
            t.pen.commit(new)
        stop_now = t.pstop is not None and t.pstop.observe(new)
        t.ctx.extend(new)
        t.n_out += len(new)
        self._round_tok += len(new)
        t.tok = t.ctx[-1]
        t.push(new)
        if stop_now:
            t.end("stop")
        elif any(x in t.eos for x in new) or t.n_out >= t.max_new or t.pos >= t.eng.max_len - 1:
            t.end("eos" if any(x in t.eos for x in new) else "length")
        elif t.think is not None:
            t.think.observe(new)            # the slot loop looks at a block it goes on after

    def _tail_a(self, t: Task, tree, path, new) -> tuple:
        """PIPE: the part of `_tail` the next draft reads: the cut, the drafter's sync and observation, the position
        and context, and whether the task ends (it ends in `_tail_b`, after its tokens are pushed). Returns
        (path, new, the end reason or None)."""
        from engine.model import h2d
        app = self.app
        accepted = len(path) - 1
        cut = app._end_cut(new, t.eos, t.max_new - t.n_out)
        if cut < len(new):
            path, new = path[:cut + 1], new[:cut]
        d = t.drafter
        if hasattr(d, "sync"):
            from engine import rowcopy
            if rowcopy.active():
                hidden = rowcopy.rows_of(t.eng.hidden_post_norm[0], path)
                if not getattr(d, "hidden_free", False):
                    rowcopy.flush()             # this drafter's sync reads the hidden rows at once
            else:
                hidden = t.eng.hidden_post_norm[0, h2d(path, torch.long, t.device)]
            toks = [int(tree.tokens[i]) for i in path]
            if getattr(d, "wants_rows", False):
                d.sync(toks, hidden, t.pos, rows=path)
            else:
                d.sync(toks, hidden, t.pos)
        t.bs.block(max(tree.depths()), accepted)
        d.observe(new)
        end = None
        if any(x in t.eos for x in new) or t.n_out + len(new) >= t.max_new or t.pos + len(path) >= t.eng.max_len - 1:
            end = "eos" if any(x in t.eos for x in new) else "length"
        t.pos += len(path)
        t.ctx.extend(new)
        t.n_out += len(new)
        self._round_tok += len(new)
        t.tok = t.ctx[-1]
        if end is None and t.think is not None:
            t.think.observe(new)
        return path, new, end

    def _tail_b(self, t: Task, path, new, end) -> None:
        """PIPE: the rest of `_tail` while the GPU drafts: the target's commit, the penalties, the stop patterns,
        the stream, the end."""
        t.eng.commit_tree(path)
        if t.pen is not None:
            t.pen.commit(new)
        stop_now = t.pstop is not None and t.pstop.observe(new)
        t.push(new)
        if stop_now:
            t.end("stop")
        elif end is not None:
            t.end(end)

    def _plain(self, t: Task) -> None:
        """A task with no draft this round: the slot loop's plain step (one row, alone)."""
        from engine.model import h2d
        self.stats["plain"] += 1
        eng, d = t.eng, t.drafter
        on_plain = getattr(d, "on_plain", None) if d is not None else None
        t_plain = time.perf_counter()
        logits = eng.forward(h2d([t.tok], torch.long, t.device), start=t.pos, last_only=True)
        if d is not None and hasattr(d, "sync"):
            d.sync([t.tok], eng.hidden_post_norm[0], t.pos)
        t.pos += 1
        t.bs.block()
        if t.pen is not None:
            t.pen.mask = bool(t.think is not None and t.think.inside)
            t.pen.apply_single(logits[0, -1])
        tok = (t.sampler(logits[0, -1], index=len(t.ctx)) if t.sampler is not None and t.sampler.on
               else int(logits[0, -1].argmax()))
        if on_plain is not None:
            on_plain((time.perf_counter() - t_plain) * 1e3)
        t.ctx.append(tok)
        t.n_out += 1
        t.tok = tok
        if t.pen is not None:
            t.pen.commit([tok])
        if d is not None:
            d.observe([tok])
        stop_now = t.pstop is not None and t.pstop.observe([tok])
        if t.think is not None and not stop_now and tok not in t.eos:
            t.think.observe([tok])          # the slot loop's plain step: past a stop or an end token, not
        t.push([tok])
        if stop_now:
            t.end("stop")
        elif tok in t.eos or t.n_out >= t.max_new or t.pos >= eng.max_len - 1:
            t.end("eos" if tok in t.eos else "length")


class _nothing:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


RUNNER: Runner | None = None


def runner_for(pool, app) -> Runner:
    global RUNNER
    if RUNNER is None or RUNNER.pool is not pool:
        RUNNER = Runner(pool, app)
    return RUNNER
