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

ON = os.environ.get("QSE_ROUND_RUNNER", "0") == "1"

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
        self._collect: list | None = None    # the runner thread's pending draft calls, while proposals run

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
        return dict(self.stats, skips=dict(SKIPS),
                    ms_a_round={k: round(v / n, 2) for k, v in self.ms.items()})

    def _t(self, key, t0):
        t1 = time.perf_counter()
        self.ms[key] += (t1 - t0) * 1e3
        return t1

    def _round(self, tasks: list[Task]) -> None:
        app = self.app
        from engine.drafters import tree_steps
        from engine.model import h2d
        from server import slots as slots_mod
        self.stats["rounds"] += 1
        t_r = t0 = time.perf_counter()
        # 1. proposals to their drafts, one packed draft, the proposals' rest
        for t in tasks:
            if t.slot.index == 0 and not t.done:
                # slot 0's side lane at the round's safe point, as the slot loop serves it between blocks
                self._bind(t)
                lane = app.STATE.get("lane")
                if lane is not None and lane.jobs:
                    self.stats["lane_jobs"] += lane.serve("decode")
        self._collect = []
        for t in tasks:
            self._bind(t)
            if t.deadline is not None and t.deadline.expired():
                t.end("deadline")
                continue
            app._row_cap(self.pool, t.drafter)
            t.steps, t.tree = app._launch(tree_steps(t.drafter, t.ctx, min(t.k, t.max_new - t.n_out)))
        calls, self._collect = self._collect, None
        self._bind_none()
        t0 = self._t("prop", t0)
        self._packed(calls)
        t0 = self._t("packed", t0)
        for t in tasks:
            if t.done:
                continue
            self._bind(t)
            t.tree = app._collect(t.steps, t.tree)
            t.steps = None
            if t.tree is not None:
                t.tree = t.tree.truncate(t.eng.max_len - t.pos)
        # 2. the joint verify pass (a task without a draft takes its plain step alone)
        t0 = self._t("rest", t0)
        group = [t for t in tasks if not t.done and t.tree is not None and t.tree.n_draft > 0]
        for t in tasks:
            if not t.done and t not in group:
                self._bind(t)
                self._plain(t)
        if group:
            self._verify(group)
        self._bind_none()
        self._t("round", t_r)
        if self.stats["rounds"] % 200 == 0:
            print(f"[runner] {self.report()}", flush=True)

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
        h = self.stats["joint_hist"]
        h[len(group)] = h.get(len(group), 0) + 1
        trees = [t.tree for t in group]
        if len(group) > 1 and meet is not None:
            tc = getattr(meet, "tree_cap", None)
            cap = tc(len(group), meet.max_rows) if tc is not None else None
            if sum(len(x.parents) for x in trees) > meet.max_rows or (
                    cap is not None and any(len(x.parents) > cap for x in trees)) or meet.joint_for(len(group)):
                keeps = slots_mod.joint_cut(trees, meet.max_rows, cap)
                trees = [x.subset(kp) if len(kp) < len(x.tokens) else x for x, kp in zip(trees, keeps)]
        t0 = time.perf_counter()
        blocks = [h2d(x.tokens, torch.long, t.device) for x, t in zip(trees, group)]
        if len(group) == 1:
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
        for t, x, lg, w in zip(group, trees, outs, walked):
            self._bind(t)
            if w is not None and w[0] == "walk":
                path, new = w[1]
            elif w is not None:
                path, new = t.eng.accept_tree(x, w[1])
            elif t.sampler is not None and t.sampler.on:
                path, new = t.sampler.tree_walk_logits(lg, x.tokens, x.parents, start=len(t.ctx), q=x.q)
            else:
                path, new = t.eng.accept_tree(x, lg.argmax(-1).tolist())
            self._tail(t, x, path, new)
        self._t("tails", t0)

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
            sel = h2d(path, torch.long, t.device)
            toks = [int(tree.tokens[i]) for i in path]
            if getattr(d, "wants_rows", False):
                d.sync(toks, t.eng.hidden_post_norm[0, sel], t.pos, rows=path)
            else:
                d.sync(toks, t.eng.hidden_post_norm[0, sel], t.pos)
        t.pos += len(path)
        t.bs.block(max(tree.depths()), accepted)
        d.observe(new)
        if t.pen is not None:
            t.pen.commit(new)
        stop_now = t.pstop is not None and t.pstop.observe(new)
        t.ctx.extend(new)
        t.n_out += len(new)
        t.tok = t.ctx[-1]
        t.push(new)
        if stop_now:
            t.end("stop")
        elif any(x in t.eos for x in new) or t.n_out >= t.max_new or t.pos >= t.eng.max_len - 1:
            t.end("eos" if any(x in t.eos for x in new) else "length")
        elif t.think is not None:
            t.think.observe(new)            # the slot loop looks at a block it goes on after

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


RUNNER: Runner | None = None


def runner_for(pool, app) -> Runner:
    global RUNNER
    if RUNNER is None or RUNNER.pool is not pool:
        RUNNER = Runner(pool, app)
    return RUNNER
