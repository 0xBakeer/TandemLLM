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

Not taken (the slot keeps its own loop, `why_not` names the reason, /health counts it): a probe,
log-probabilities, a request without a tree drafter. Slot 0's side lane (server/preempt.py) is served by the
runner at each round's start, the safe point the slot loop serves it at (no proposal in flight). A passive
reasoning block (`ThinkBudget` with neither budget nor arms) is followed as the slot loop follows it: `inside`
masks the penalties, every block is observed. QSE_ROUND_RUNNER=1 turns it on.

SPD-249 (QSE_RUNNER_SUPERVISED): a supervised request and a reasoning block that can be closed (a budget,
stall arms) are taken too. Where the slot loop asks the supervisor or closes the block, the runner hands the
request back to its slot's thread (`Task.back`), which does what the slot loop does there and gives it to the
runner again:
  - SP2, an end token inside the budget: the rows before it are committed, nothing of it is taken (the
    tokens before it, the end token and the block's room go back; the slot's thread asks `_sp2`);
  - SP1, a block inside the reasoning that the supervisor watches: the tokens go out with a marker behind
    them, the slot's thread asks `_sp1` when it reaches the marker, and the task sits out the rounds until
    the verdict is in (None: it goes on; an action: it goes back);
  - the budget spent (`think.hit`): back, for the forced close.
A task that waits for its verdict is never drafted ahead (PIPE). One difference from the slot loop: when
SP2 decides nothing, the end token's row is not committed (the slot loop commits it when the end token was a
drafted node). The tokens are the same; the turn then ends as it does on a plain step or a bonus end token,
with the end token not forwarded, and the next turn reads that one token again.
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
#: SPD-239: a joint round's commits at once where the family commits several slots in one call (its
#: `commit_tree_multi`: every lane's copies in one launch); 0: each tail commits its own
COMMIT_MULTI = os.environ.get("QSE_RUNNER_COMMIT_MULTI", "1") == "1"
#: SPD-249: a joint pass hands the families that read a block on the host (`host_blocks`) a host tensor of its
#: tokens instead of a device copy each lane (which the family read back to the host, a stream sync each lane); the
#: same tokens. 0: device blocks for every family
HOST_BLOCKS = os.environ.get("QSE_RUNNER_HOST_BLOCKS", "0") == "1"
#: SPD-233: at E lanes of families that join a plain step to the joint pass (`joint_plain`), the runner weighs its
#: drafted rounds' tokens a GPU ms against its plain rounds' (one joint decode step, a token a lane) and takes the
#: better: in plain mode every lane's drafter skips its draft. Each mode's rate is a running mean of the rounds
#: that ran it (the period from one round's end to the next, as `steady`), every QSE_RUNNER_LANE_PROBE-th round at
#: E runs the other mode, and a switch needs QSE_RUNNER_LANE_MARGIN over the current mode. Bolt EXL3 4.0, 8 lanes,
#: T1 prose: a drafted round 114.6 ms for 11 tokens, the joint plain step 64.3 ms for 8 (in process); served C8
#: 78.3 / 78.1 -> 90.5 / 90.1 tok/s. "auto" (the default): the families whose serving engine opts in (`lane_plain`
#: True, measured), "1": every joint-plain family, "0": off
LANE_MODE = os.environ.get("QSE_RUNNER_LANE_PLAIN", "auto")
LANE_PLAIN = LANE_MODE != "0"
LANE_PROBE = max(2, int(os.environ.get("QSE_RUNNER_LANE_PROBE", "16") or 16))
LANE_MARGIN = float(os.environ.get("QSE_RUNNER_LANE_MARGIN", "0.03") or 0.03)
#: SPD-249: supervised requests and closable reasoning blocks go with the runner, handed back at SP1, SP2
#: and the budget (the module's header); 0: they keep the slot loop ("supervisor", "think_budget" skips)
SUPERVISED = os.environ.get("QSE_RUNNER_SUPERVISED", "1") == "1"
#: SPD-249: how long a round waits for the SP1 verdicts its tasks' slot threads owe (ms) before the tasks
#: still waiting sit it out: the threads answer once the block's tokens are read, and a lane that misses the
#: round's joint pass costs it its share of that pass. 0: no wait
VERDICT_WAIT_MS = float(os.environ.get("QSE_RUNNER_VERDICT_WAIT_MS", "4") or 0)
TEST_REFUSE_AFTER = int(os.environ.get("QSE_KV_TEST_REFUSE_AFTER", "0") or 0)
TEST_REFUSE_SLOTS = {int(x) for x in (os.environ.get("QSE_KV_TEST_REFUSE_SLOTS") or "").split(",") if x.strip()}
#: SPD-249 (a measurement): host counters by lane count in `report()["counters"]`: each round cut into contiguous
#: host segments (wall, the runner thread's CPU, the process's CPU, garbage-collection ms), the timed host calls
#: of the drafters and of the families that hand their own (`host_timers`), the verified trees' row depths, rows and
#: accepted lengths. Timers only: what a round computes does not change. 0: off
COUNTERS = os.environ.get("QSE_RUNNER_COUNTERS", "0") == "1"
#: the timed host calls' sums, {name: [wall ms, thread CPU ms, calls, hits]} (`host_time`), across threads
HOST_CALLS: dict = {}
#: SPD-249: the host work no round waits for runs while the GPU runs the next joint pass, not in the host phases the
#: GPU waits through. A joint round's token pushes are held and handed to the slots' threads in the next pass's GPU wait
#: (engine/methods/busy.py: the family calls `busy()` between a pass's issue and its wait); a task that ends, goes back
#: or reaches an SP1 marker gets its held tokens at once, and tokens no wait took go at the next round's end. Before the
#: runner waits for anything but its own next pass (the turn to another holder, slot 0's lane job, SP1 verdicts) the
#: held tokens go and the windows are given up (`_hand_off`). The runner's report line is printed in that wait, and
#: /health's full body and the live sampler's ticks wait for it (`busy.gate`). The same tokens, in the same order,
#: later on the wire by up to a round's own work. 0: off
BUSY_WORK = os.environ.get("QSE_RUNNER_BUSY_WORK", "0") == "1"
#: SPD-249: the collector kept out of the rounds. The objects the server holds after its load are frozen
#: (`gc_after_load`: never scanned again); while the runner's rounds run there is no automatic collection: the young
#: generations are collected in a joint pass's GPU wait at Python's own thresholds (at a round's end when no wait
#: came), the full collection when the runner goes idle and, under load with no idle gap, in a GPU wait every
#: QSE_RUNNER_GC_FULL_ROUNDS rounds. What a round computes does not change. 0: off (Python's own collector)
GC_CTL = os.environ.get("QSE_RUNNER_GC", "0") == "1"
GC_FULL_ROUNDS = max(1, int(os.environ.get("QSE_RUNNER_GC_FULL_ROUNDS", "200") or 200))
#: the full collection of an idle gap waits this long after the runner's last round (and for no request in flight)
IDLE_GC_S = 0.25
#: the collections the runner made, {kind: [count, ms, max ms]}, and what the load froze
GC_STATS: dict = {"frozen": None, "load_ms": None}
#: garbage-collection pauses (any thread): [ms, collections] and {generation: [ms, collections]}
_GC = [0.0, 0]
_GC_GEN: dict = {}

#: requests the runner did not take, by reason (`why_not`), for /health
SKIPS: dict = {}


def host_time(obj, attr: str, name: str, hit=None) -> None:
    """`obj.attr` (a bound method) replaced on the instance by a timed call of the same method; `hit(result)`
    True counts a hit. Once an instance and attribute."""
    fn = getattr(obj, attr, None)
    if fn is None or getattr(fn, "_hc", False):
        return

    def timed(*a, **k):
        w, c = time.perf_counter(), time.thread_time()
        try:
            r = fn(*a, **k)
        finally:
            s = HOST_CALLS.setdefault(name, [0.0, 0.0, 0, 0])
            s[0] += 1e3 * (time.perf_counter() - w)
            s[1] += 1e3 * (time.thread_time() - c)
            s[2] += 1
        if hit is not None and hit(r):
            s[3] += 1
        return r
    timed._hc = True
    setattr(obj, attr, timed)


def _gc_cb(phase, info, _t=[0.0]) -> None:
    if phase == "start":
        _t[0] = time.perf_counter()
        return
    ms = 1e3 * (time.perf_counter() - _t[0])
    _GC[0] += ms
    _GC[1] += 1
    g = _GC_GEN.setdefault(info.get("generation"), [0.0, 0])
    g[0] += ms
    g[1] += 1



def gc_after_load(log=print) -> None:
    """SPD-249 (QSE_RUNNER_GC): after the server's load and warm-up, one full collection and the survivors frozen
    (`gc.freeze`: the weights' Python side, the tokenizer, the tables; a full collection scanned them all, ~216 ms
    a pause at 8 lanes). Nothing with the knob off."""
    if not GC_CTL:
        return
    import gc
    from engine import capture_lock
    t0 = time.perf_counter()
    with capture_lock.off_thread("runner_gc"):      # a finalizer's CUDA call never lands inside an idle capture
        n = gc.collect()
    gc.freeze()
    GC_STATS["frozen"] = gc.get_freeze_count()
    GC_STATS["load_ms"] = round(1e3 * (time.perf_counter() - t0), 1)
    log(f"[runner] gc: {GC_STATS['frozen']} objects frozen after the load (collection {GC_STATS['load_ms']} ms, "
        f"{n} unreachable); no automatic collection while the rounds run")


def _gc_note(kind: str, ms: float) -> None:
    s = GC_STATS.setdefault(kind, [0, 0.0, 0.0])
    s[0] += 1
    s[1] += ms
    s[2] = max(s[2], ms)


def _gc_old() -> None:
    """After a full collection: the objects left in the old generation (what the load froze is not there),
    the last count and the most, the memory check of a long run."""
    import gc
    n = len(gc.get_objects(2))
    GC_STATS["old_after_full"] = n
    GC_STATS["old_after_full_max"] = max(GC_STATS.get("old_after_full_max", 0), n)


def why_not(*, sup, probe, lpr, think, ahead, drafter, lane, turn) -> str | None:
    """None when the runner may take this slot's decode, else the reason it may not (counted)."""
    keep = not SUPERVISED                   # SPD-249: else handed back at SP1, SP2 and the budget
    why = ("supervisor" if sup is not None and keep else "probe" if probe is not None
           else "logprobs" if lpr is not None
           else "think_budget" if keep and think is not None and (think.budget or think.arms_cfg is not None)
           else "launched" if ahead is not None else "no_drafter" if drafter is None
           else "no_turn" if turn is None else None)
    if why is not None:
        SKIPS[why] = SKIPS.get(why, 0) + 1
    return why


class Task:
    """One slot's decode state, owned by the runner while the request is registered."""

    def __init__(self, slot, eng, drafter, sampler, pen, pstop, ctx, pos, tok, n_out, max_new, eos, bs, k,
                 deadline, device, think=None, sup=None):
        self.think = think                  # the reasoning block's budget (passive, or SPD-249: armed) or None
        self.sup = sup                      # SPD-249: the request's supervisor, or None
        self.gate = False                   # SPD-249: SP1's marker is out; the task waits for its verdict
        self.verdict = None                 # SPD-249: [the slot thread's SP1 action or None], once given
        self.next = None                    # SPD-249: "budget" or "sp1" after this block's tokens (`_tail_a`)
        self.back = None                    # SPD-249: the hand-back (`end_back`), for the slot's thread
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
        self.held: list = []                # SPD-249: messages held for the next GPU wait (the runner's thread)
        self.held_at = 0                    # the round the oldest of them is from

    def push(self, toks) -> None:
        self.flush()
        self.q.put(("tokens", list(toks)))

    def hold(self, toks, rnd: int) -> None:
        """SPD-249: tokens for the slot's thread, handed over later (`flush`), in order."""
        if not self.held:
            self.held_at = rnd
        self.held.append(("tokens", list(toks)))

    def flush(self) -> None:
        if self.held:
            held, self.held = self.held, []
            for m in held:
                self.q.put(m)

    def put(self, msg) -> None:
        """A message after every held one."""
        self.flush()
        self.q.put(msg)

    def end(self, why: str) -> None:
        self.flush()
        self.done = True
        self.q.put(("end", why))

    def end_back(self, back: dict) -> None:
        """SPD-249: the request goes back to its slot's thread at `back["why"]` (sp2, sp2_plain, sp1,
        budget), with what that point needs; the slot's thread gives it to the runner again after."""
        self.back = back
        self.end("back")


class Runner:
    def __init__(self, pool, app):
        self.pool, self.app = pool, app
        self.tasks: list[Task] = []
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        self.thread = None
        self.stats = {"rounds": 0, "tasks": 0, "joint": 0, "plain": 0, "errors": 0, "lane_jobs": 0,
                      # rounds by the slots in their verify pass (the concurrency proof: N rows a round)
                      "joint_hist": {},
                      # SPD-249: hand-backs by point, and the rounds a task sat out for its SP1 verdict
                      "back": {}, "sp1_marks": 0, "sat_out": 0, "verdict_wait_ms": 0.0}
        # host ms a round by phase (summed; `report`): the proposals to their drafts, the packed draft's
        # issue, the proposals' rest (it reads the draft back), the verify's issue, the walk (it reads the
        # pass back), the tails
        self.ms = {k: 0.0 for k in ("prop", "packed", "rest", "verify", "walk", "tails", "round")}
        # SPD-249: the round's thread CPU ms outside the verify pass's wait, beside its wall ms (`round` - `verify`): the
        # gap is time the runner's thread did not run (the GIL held by another thread, or the OS)
        self.ms["out_cpu"] = 0.0
        # the same phases on the GPU's timeline (SPD-201 host cut, step a): a CUDA event at the round's start and
        # at each phase's end, read one round later (never waited on). A segment's GPU time is its end event
        # minus the one before: the packed draft and the verify pass are the GPU's own work, the others the
        # time the GPU waited for the host (or ran its small kernels); `between` is one round's end to the next
        # round's start
        self.gpu = {k: 0.0 for k in ("prop", "packed", "rest", "verify", "walk", "tails", "between", "period")}
        self.gpu_n = 0
        self._verify_cpu = 0.0
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
        self._round_plain = False           # SPD-233: every block of the round's pass was its anchor alone
        self._lane: dict = {}               # SPD-233: by lanes, the drafted and plain rounds' rates (`_lane_plain`)
        self._lane_at = (None, False)       # SPD-233: the round the last mode decision was for, and the decision
        self._round_cap0 = None             # SPD-233: `_captures` at the round's start
        from server import slots as slots_mod
        self.prices = slots_mod.RowPrices()
        import random
        self._rng = random.Random(225)
        self._collect: list | None = None    # the runner thread's pending draft calls, while proposals run
        self._mtp_collect: list | None = None    # SPD-235: draft heads whose drafts the runner packs
        if MTP_PACK:
            from engine.methods import draft_meet
            draft_meet.MEET = self.mtp_meet
        # SPD-249: this round's segments {label: [wall, cpu, process cpu, gc ms]}, the clocks at the last cut, the
        # timed calls at the round's start, and the sums by lane count (`_hc_end`)
        self._hc_round: dict = {}
        self._hc_at = None
        self._hc_c0: dict = {}
        self._hc_e: dict = {}
        self._hc_gpu = None                 # the last round's GPU work was done when this round's rest began
        self._hc_gen0: dict = {}
        if COUNTERS:
            import gc
            if _gc_cb not in gc.callbacks:
                gc.callbacks.append(_gc_cb)
        # SPD-249: the GPU-wait hook (engine/methods/busy.py) and its state
        self._held: dict = {}               # tasks with held tokens, by id
        self._defer = False                 # this round's pushes are held (its pass had a GPU wait)
        self._busy_at = -1                  # the round whose pass last called the hook
        self._busy_on = False               # the runner holds windows (`busy.hold`)
        self._report_due = False            # the report line waits for the next GPU wait
        self._gc_off = False                # the runner turned automatic collection off for its rounds
        self._gc_rounds = 0                 # rounds since the last full collection
        self._gc_full_due = False           # the full collection waits for an idle gap
        self._gc_idle_at = 0.0              # when the runner went idle
        if BUSY_WORK or GC_CTL:
            self.stats["busy"] = {"waits": 0, "flushed": 0, "late": 0, "handoff": 0}
            from engine.methods import busy as busy_mod
            busy_mod.HOOK = self._busy

    # ---- registration (the slot's thread) ---------------------------------------------------
    def add(self, task: Task) -> None:
        js = getattr(task.eng, "joint_stats", None)
        if callable(js):
            self._joint_stats = js                # SPD-249: the family's joint-pass shape, for the report
        if COUNTERS:
            self._hc_wrap(task)
        with self.cv:
            self.tasks.append(task)
            self.stats["tasks"] += 1
            if self.thread is None or not self.thread.is_alive():
                self.thread = threading.Thread(target=self._loop, name="round-runner", daemon=True)
                self.thread.start()
            self.cv.notify_all()

    def poke(self) -> None:
        """SPD-249: a slot's thread gave its SP1 verdict: the loop looks again."""
        with self.cv:
            self.cv.notify_all()

    def _ready(self, t: Task) -> bool:
        """SPD-249 (under `cv`): whether `t` decodes this round. A task whose SP1 verdict is not in sits it
        out; a verdict of None lets it go on; an action sends it back to its slot's thread."""
        if not t.gate:
            return True
        v = t.verdict
        if v is None:
            self.stats["sat_out"] += 1
            return False
        t.gate, t.verdict = False, None
        if v[0] is None:
            return True
        self._back(t, {"why": "sp1", "act": v[0]})
        return False

    def _back(self, t: Task, back: dict) -> None:
        b = self.stats["back"]
        b[back["why"]] = b.get(back["why"], 0) + 1
        t.end_back(back)

    def _sp2_at(self, t: Task, new) -> int | None:
        """SPD-249: where SP2 asks the supervisor in this block: its first end token, when it is inside the
        output budget (`app._sp2_block`'s rule); None for a request without a supervisor."""
        if t.sup is None:
            return None
        j = next((i for i, x in enumerate(new) if x in t.eos), None)
        return None if j is None or t.n_out + j >= t.max_new else j

    @staticmethod
    def _next_stop(t: Task) -> str | None:
        """SPD-249, after a block's `think.observe`: "budget" when the budget is spent (the forced close),
        "sp1" when the supervisor watches this block inside the reasoning (`app._sp1`'s rule), else None."""
        th = t.think
        if th is None:
            return None
        if th.hit:
            return "budget"
        if t.sup is not None and th.inside and getattr(t.sup, "watches_blocks", False):
            return "sp1"
        return None

    def _after(self, t: Task) -> None:
        """SPD-249: a block's tokens are out and the task goes on: back for the budget, or SP1's marker behind
        the tokens (the slot's thread asks there, with the values the slot loop passes)."""
        nxt, t.next = t.next, None
        if nxt == "budget":
            self._back(t, {"why": "budget"})
        elif nxt == "sp1":
            self.stats["sp1_marks"] += 1
            t.gate = True
            t.put(("sp1", (t.n_out, t.eng.max_len - t.pos, len(t.ctx))))

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
        owed = None                         # SPD-249: since when the round waits for SP1 verdicts
        while True:
            idle = False
            with self.cv:
                live = [t for t in self.tasks if not t.done]
                for t in self.tasks:
                    if t.cancel and not t.done:
                        t.end("cancelled")
                self.tasks = [t for t in self.tasks if not t.done]
                if not self.tasks and (self._gc_off or self._busy_on or self._held or self._gc_full_due):
                    idle = True                 # SPD-249: the idle work, after the lock (`_idle`)
                elif not self.tasks:
                    self.cv.wait(timeout=0.05)
                    if not self.tasks:
                        continue
            if idle:
                self._idle()
                continue
            with self.cv:
                if not self.tasks:
                    continue
                if VERDICT_WAIT_MS > 0 and any(t.gate and t.verdict is None for t in self.tasks):
                    now = time.perf_counter()
                    owed = now if owed is None else owed
                    left = VERDICT_WAIT_MS / 1e3 - (now - owed)
                    if left > 0:
                        self._hand_off()            # nothing held through the wait
                        self.cv.wait(timeout=left)
                        continue
                if owed is not None:
                    self.stats["verdict_wait_ms"] += (time.perf_counter() - owed) * 1e3
                    owed = None
                live = [t for t in self.tasks if not t.done and self._ready(t)]
                if not live:
                    # SPD-249: every task waits for its SP1 verdict (or went back): no round, no turn
                    self._hand_off()
                    self.cv.wait(timeout=0.005)
                    continue
            if turn is not None:
                self._acquire(turn)
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
                    if BUSY_WORK and turn.waiting():
                        self._hand_off()            # another holder takes the turn next
                    turn.release("runner")

    def _bind(self, t):
        from server import slots as slots_mod
        slots_mod.bind(None if t.slot.index == 0 else t.slot)

    # ---- SPD-249: the GPU's wait and the idle gaps ------------------------------------
    def _busy(self) -> None:
        """`busy.HOOK`: a joint pass of this runner's round is queued and its thread is about to wait for it. The
        young generations collected and, owed, the full collection; the held tokens to the slots' threads, the
        report line, and a window for the gated work of other threads."""
        if threading.current_thread() is not self.thread:
            return
        self._busy_at = self.stats["rounds"]
        self.stats["busy"]["waits"] += 1
        if GC_CTL and self._gc_off:
            self._gc_collect(full=self._gc_rounds >= GC_FULL_ROUNDS, kind="busy")
        if BUSY_WORK:
            for t in self._held.values():
                self.stats["busy"]["flushed"] += len(t.held)
                t.flush()
            self._held.clear()
            if self._report_due:
                self._report_due = False
                print(f"[runner] {self.report()}", flush=True)
            from engine.methods import busy as busy_mod
            if not self._busy_on:
                self._busy_on = True
                busy_mod.hold(True)
            busy_mod.opened()

    def _acquire(self, turn) -> None:
        """The turn for a round. With QSE_RUNNER_BUSY_WORK, when it would wait (another holder has it or is owed it next), the held tokens
        and the windows go first (`_hand_off`), so no stream waits through a prefill chunk, a burst or a lane job."""
        if BUSY_WORK and (self._held or self._busy_on or self._report_due):
            try_ = getattr(turn, "try_acquire", None)
            if try_ is not None and try_("runner"):
                return
            self._hand_off()
        turn.acquire("runner")

    def _hand_off(self) -> None:
        """The runner is about to wait for something other than its own next pass: the tokens it holds go to the
        slots' threads now, in order, a due report line is printed, and its windows are given up (`busy.hold(False)`:
        every gated thread goes at once). The next pass's GPU wait holds windows again."""
        if not BUSY_WORK:
            return
        if self._held:
            for t in self._held.values():
                self.stats["busy"]["handoff"] += len(t.held)
                t.flush()
            self._held.clear()
        if self._report_due:
            self._report_due = False
            print(f"[runner] {self.report()}", flush=True)
        if self._busy_on:
            from engine.methods import busy as busy_mod
            self._busy_on = False
            busy_mod.hold(False)

    def _push(self, t: Task, toks) -> None:
        """A tail's tokens: held for the next GPU wait when this round's pass had one, else onto the queue."""
        if self._defer:
            t.hold(toks, self.stats["rounds"])
            self._held[id(t)] = t
        else:
            t.push(toks)

    def _round_end_busy(self) -> None:
        """At a round's end: tokens held since an earlier round (no GPU wait took them) go now; a round that
        had no GPU wait gives its windows up, prints a due report line and collects as Python would have."""
        r = self.stats["rounds"]
        if self._held:
            for k, t in list(self._held.items()):
                if not t.held:
                    del self._held[k]
                elif t.held_at < r or t.done:
                    self.stats["busy"]["late"] += len(t.held)
                    t.flush()
                    del self._held[k]
        if self._busy_at != r:
            if self._busy_on:
                from engine.methods import busy as busy_mod
                self._busy_on = False
                busy_mod.hold(False)
            if self._report_due:
                self._report_due = False
                print(f"[runner] {self.report()}", flush=True)
            if GC_CTL and self._gc_off:
                self._gc_collect(full=self._gc_rounds >= GC_FULL_ROUNDS, kind="late")

    def _gc_collect(self, full: bool, kind: str) -> None:
        """The full collection when `full`, else the young generations at Python's own thresholds (generation 0
        past its threshold, generation 1 when its count is past its own), never generation 2 on its own."""
        import gc
        if full:
            t0 = time.perf_counter()
            gc.collect(2)
            self._gc_rounds = 0
            _gc_note("full_" + kind, 1e3 * (time.perf_counter() - t0))
            _gc_old()
            return
        c0, c1, _ = gc.get_count()
        th0, th1, _ = gc.get_threshold()
        if c0 <= th0:
            return
        t0 = time.perf_counter()
        gc.collect(1 if c1 > th1 else 0)
        _gc_note("young_" + kind, 1e3 * (time.perf_counter() - t0))

    def _idle(self) -> None:
        """The runner's tasks all gone: held tokens out, the windows given up, Python's own collector back on
        at once (whatever runs while no round does, e.g. a prefill, collects as before); the full collection once the
        runner has been idle IDLE_GC_S and no request is in flight (a burst's prefill is not held up by it)."""
        for t in self._held.values():
            t.flush()
        self._held.clear()
        if self._busy_on:
            from engine.methods import busy as busy_mod
            self._busy_on = False
            busy_mod.hold(False)
        if self._report_due:
            self._report_due = False
            print(f"[runner] {self.report()}", flush=True)
        import gc
        now = time.perf_counter()
        if self._gc_off:
            gc.enable()
            self._gc_off = False
            self._gc_full_due, self._gc_idle_at = True, now
        if self._gc_full_due:
            inflight = int((getattr(self.app, "INFLIGHT", None) or {}).get("running", 0) or 0)
            if now - self._gc_idle_at >= IDLE_GC_S and inflight == 0:
                self._gc_full_due = False
                from engine import capture_lock
                t0 = time.perf_counter()
                with capture_lock.off_thread("runner_gc"):   # the idle thread's captures run in these gaps
                    gc.collect(2)
                self._gc_rounds = 0
                _gc_note("full_idle", 1e3 * (time.perf_counter() - t0))
                _gc_old()
                return
        with self.cv:
            if not self.tasks:
                self.cv.wait(timeout=0.05)

    # ---- SPD-249 host counters (QSE_RUNNER_COUNTERS=1) ------------------------------------------
    @staticmethod
    def _hc_wrap(task: Task) -> None:
        """A task's drafter calls timed (the lookup with its hits, the cut, the feed, the head's launch) and the
        family's own (`host_timers(host_time)` on its serving engine, e.g. the draft read's wait, the PLE ids and rows)."""
        d = task.drafter
        if d is not None:
            host_time(d, "_look", "look", hit=lambda r: r is not None)
            host_time(d, "_feed", "feed")
            if getattr(d, "cut", None) is not None:
                host_time(d.cut, "cut", "cut")
            if getattr(d, "head", None) is not None:
                host_time(d.head, "launch", "launch")
        fw = getattr(task.eng, "host_timers", None)
        if callable(fw):
            fw(host_time)

    def _hc_seg(self, label: str) -> None:
        """The host time since the last cut into this round's segment `label`."""
        if not COUNTERS:
            return
        now = (time.perf_counter(), time.thread_time(), time.process_time(), _GC[0])
        at, self._hc_at = getattr(self, "_hc_at", None), now
        if at is None:
            return
        s = self.__dict__.setdefault("_hc_round", {}).setdefault(label, [0.0, 0.0, 0.0, 0.0, 0])
        s[0] += 1e3 * (now[0] - at[0])
        s[1] += 1e3 * (now[1] - at[1])
        s[2] += 1e3 * (now[2] - at[2])
        s[3] += now[3] - at[3]
        s[4] += 1

    def _hc_start(self) -> None:
        """At a round's start: the time since the last round's end is its `between` unless the runner was
        idle (more than 100 ms: no round waited); the timed calls' sums noted."""
        if not COUNTERS:
            return
        if self._hc_at is not None and time.perf_counter() - self._hc_at[0] > 0.1:
            self._hc_at = None
        self._hc_round = {}
        self._hc_seg("between")
        if self._hc_at is None:
            self._hc_at = (time.perf_counter(), time.thread_time(), time.process_time(), _GC[0])
        self._hc_c0 = {k: list(v) for k, v in HOST_CALLS.items()}
        self._hc_gen0 = {g: list(v) for g, v in _GC_GEN.items()}
        self._hc_gpu = None

    def _hc_tree(self, trees: list) -> None:
        """A verify's trees after its cuts: rows by depth, deepest row a tree, rows a pass, at these lanes."""
        if not COUNTERS:
            return
        e = self.__dict__.setdefault("_hc_e", {}).setdefault(len(trees), {"n": 0, "seg": {}, "calls": {}, "depth": {}, "deepest": {},
                                               "rows": {}, "acc": {}, "gc_gen": {}, "gpu_done": 0, "gpu_asked": 0})
        rows = 0
        for x in trees:
            dep = x.depths()
            rows += len(dep)
            for k in dep:
                e["depth"][k] = e["depth"].get(k, 0) + 1
            m = max(dep)
            e["deepest"][m] = e["deepest"].get(m, 0) + 1
        e["rows"][rows] = e["rows"].get(rows, 0) + 1

    def _hc_acc(self, lanes: int, accepted: int) -> None:
        if COUNTERS and lanes in self.__dict__.get("_hc_e", {}):
            a = self._hc_e[lanes]["acc"]
            a[accepted] = a.get(accepted, 0) + 1

    def _hc_end(self, e: int) -> None:
        """At a round's end: its segments and the timed calls' growth into the sums of its lane count."""
        if not COUNTERS:
            return
        self._hc_seg("end")
        if e not in self.__dict__.get("_hc_e", {}):
            return
        d = self._hc_e[e]
        d["n"] += 1
        if self._hc_gpu is not None:
            d["gpu_asked"] += 1
            d["gpu_done"] += int(self._hc_gpu)
        for k, v in self._hc_round.items():
            s = d["seg"].setdefault(k, [0.0, 0.0, 0.0, 0.0, 0])
            for i in range(5):
                s[i] += v[i]
        for k, v in HOST_CALLS.items():
            v0 = self._hc_c0.get(k, [0.0, 0.0, 0, 0])
            s = d["calls"].setdefault(k, [0.0, 0.0, 0, 0])
            for i in range(4):
                s[i] += v[i] - v0[i]
        for g, v in _GC_GEN.items():
            v0 = self._hc_gen0.get(g, [0.0, 0])
            s = d["gc_gen"].setdefault(g, [0.0, 0])
            s[0] += v[0] - v0[0]
            s[1] += v[1] - v0[1]

    def _hc_report(self) -> dict:
        """By lane count (8 rounds or more): segment ms a round [wall, thread CPU, process CPU, gc], the timed calls
        a round [wall ms, CPU ms, calls, hits], the histograms (summed), the share of rounds whose GPU had finished the
        last round's work when the proposals' rest began."""
        out = {}
        for e, d in sorted(self._hc_e.items()):
            n = d["n"]
            if n < 8:
                continue
            out[e] = {"rounds": n,
                      "seg": {k: [round(v[0] / n, 3), round(v[1] / n, 3), round(v[2] / n, 3), round(v[3] / n, 3)]
                              for k, v in d["seg"].items()},
                      "calls": {k: [round(v[0] / n, 3), round(v[1] / n, 3), round(v[2] / n, 3), round(v[3] / n, 3)]
                                for k, v in d["calls"].items()},
                      "depth": dict(sorted(d["depth"].items())), "deepest": dict(sorted(d["deepest"].items())),
                      "rows": dict(sorted(d["rows"].items())), "acc": dict(sorted(d["acc"].items())),
                      "gc_gen": {g: [round(v[0], 2), v[1]] for g, v in d["gc_gen"].items()},
                      "gpu_done_at_rest": [d["gpu_done"], d["gpu_asked"]]}
        return {"by_lanes": out, "gc_total": [round(_GC[0], 1), _GC[1]]}

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
                    lane_plain=({e: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in st.items()}
                                 for e, st in sorted(self._lane.items())} if LANE_PLAIN else None),
                    ms_by_slots=self._by_e(self._ms_e), gpu_ms_by_slots=self._by_e(self._gpu_e),
                    tails_by_slots=self._by_e(self._tails_e), joint_stats=self._joint_stats_now(),
                    **({"counters": self._hc_report()} if COUNTERS else {}),
                    **({"host_work": self._busy_report()} if BUSY_WORK or GC_CTL else {}))

    def _busy_report(self) -> dict:
        """SPD-249: the knobs, the GPU waits the hook ran in and the tokens it handed over (`late`: at a
        round's end instead), the gated work's windows, and the collections (`young_busy` / `full_busy` in a GPU wait,
        `*_late` at a round's end, `full_idle` in an idle gap: [count, ms, max ms]; the objects in the old generation
        after the last full collection, and the most)."""
        import gc
        from engine.methods import busy as busy_mod
        gs = {k: ([v[0], round(v[1], 1), round(v[2], 2)] if isinstance(v, list) else v) for k, v in GC_STATS.items()}
        return {"busy_work": BUSY_WORK, "gc": GC_CTL, "gc_full_rounds": GC_FULL_ROUNDS, "hook": dict(self.stats["busy"]),
                "windows": busy_mod.report(), "gc_stats": gs, "gc_count": list(gc.get_count()),
                "gc_enabled": gc.isenabled()}

    def _joint_stats_now(self):
        """SPD-249: the family's joint-pass shape (its engine's `joint_stats`, e.g. the distinct experts a layer)."""
        js = getattr(self, "_joint_stats", None)
        try:
            return js() if js is not None else None
        except Exception:                                          # noqa: BLE001
            return None

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
        for rnd, prev, e, tok, rows, plain, cap in self._ev_done:
            if not rnd[-1][1].query():
                keep.append((rnd, prev, e, tok, rows, plain, cap))
                continue
            if prev is not None and prev.query() and e:
                per = prev.elapsed_time(rnd[-1][1])
                if LANE_PLAIN and e > 1 and per > 0 and not cap:
                    # a round that captured a graph is the capture's time, not its mode's
                    self._lane_note(e, plain, tok / per)
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
        self._round_plain = False
        self._round_cap0 = self._captures(tasks)
        self._defer = False
        if GC_CTL:
            import gc
            if gc.isenabled():
                gc.disable()                # no automatic collection while the rounds run
                self._gc_off = True
            self._gc_rounds += 1
        self._hc_start()
        self._mark("start")
        ms0 = dict(self.ms)
        t_r = t0 = time.perf_counter()
        c_r = time.thread_time()
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
                    self._hand_off()                # no stream waits through the lane's job
                    self.stats["lane_jobs"] += lane.serve("decode")
        for t in tasks:
            if t.pre and not t.done and t.deadline is not None and t.deadline.expired():
                t.end("deadline")
        t0 = self._propose([t for t in tasks if not t.pre], t0)
        if COUNTERS:
            self._hc_seg("prop0")
            if self._ev_last is not None:
                self._hc_gpu = bool(self._ev_last.query())
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
        self._prefetch(tasks)
        self._hc_seg("rest")
        # 2. the joint verify pass (a task without a draft takes its plain step alone)
        t0 = self._t("rest", t0)
        live = [t for t in tasks if not t.done]
        for t in live:
            if hasattr(t.drafter, "lanes"):
                # SPD-239: the drafter prices this round by its wall time at its next proposal; a shared one
                # is not a lone round's price
                t.drafter.lanes = len(live)
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
        if BUSY_WORK or GC_CTL:
            self._round_end_busy()
        self._t("round", t_r)
        self._hc_end(self._round_e)
        self.ms["out_cpu"] += 1e3 * (time.thread_time() - c_r) - self._verify_cpu
        self._verify_cpu = 0.0
        me = self._ms_e.setdefault(self._round_e, {"n": 0})
        me["n"] += 1
        for k, v in self.ms.items():
            me[k] = me.get(k, 0.0) + v - ms0[k]
        if self._ev:
            self._ev_done.append((self._ev, self._ev_last, self._round_e, self._round_tok, self._round_rows,
                                  self._round_plain, self._captures(tasks) != self._round_cap0))
            self._ev_last = self._ev[-1][1]
        if self.stats["rounds"] % 200 == 0:
            if BUSY_WORK and self._busy_at == self.stats["rounds"]:
                self._report_due = True     # printed in the next GPU wait (`_busy`)
            else:
                print(f"[runner] {self.report()}", flush=True)

    def _propose(self, tasks: list, t0: float, deadline: bool = True) -> float:
        """Each task's proposal to its draft, then one packed draft for them (queued, not read). `deadline`
        False (PIPE, inside a tail): the deadline is the next round's to check, after the stream."""
        from engine.drafters import tree_steps
        app = self.app
        self._collect = []
        if MTP_PACK and sum(1 for t in tasks if not t.done) > 1:
            self._mtp_collect = []
        # SPD-233: the mode of the round these drafts are for (a tail's are the next round's), decided once a round
        # and only in a call with live tasks (the round start's call of the tasks not yet drafted is mostly empty)
        skip = (self._lane_decide(self.stats["rounds"] + (0 if deadline else 1))
                if LANE_PLAIN and any(not t.done for t in tasks) else False)
        for t in tasks:
            if t.done:
                continue
            if LANE_PLAIN and hasattr(t.drafter, "lane_skip"):
                t.drafter.lane_skip = skip
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
        if not deadline:
            self._hc_seg("tprop")
        if heads:
            by: dict = {}
            for h in heads:
                by.setdefault((type(h), id(h.fn)), []).append(h)
            for grp in by.values():
                type(grp[0]).run_packed(grp)
        if calls:
            self._packed(calls)
        if not deadline:
            self._hc_seg("tpacked")
        return self._t("packed", t0)

    def _prefetch(self, tasks: list) -> None:
        """SPD-249: the drafted trees' tokens are known: a family that reads host rows for a tree's rows (its
        engine's `prefetch_rows`, e.g. its per-token embedding rows) may start that read now, before the round's cuts and
        staging; what the verify computes does not change."""
        by: dict = {}
        for t in tasks:
            if t.done or t.tree is None or t.tree.n_draft == 0:
                continue
            pf = getattr(type(t.eng), "prefetch_rows", None)
            if pf is not None:
                by.setdefault(pf, []).append((t.eng, t.tree.tokens, t.tree.parents))
        for pf, items in by.items():
            try:
                pf(items)
            except Exception as exc:                               # noqa: BLE001
                print(f"[runner] prefetch failed ({type(exc).__name__}: {exc}); the verify reads its own rows",
                      flush=True)

    def _lane_decide(self, target: int) -> bool:
        """`_lane_plain` once for the round `target` (the probe counter steps once a decided round); a second call
        for the same round takes the first's decision."""
        if self._lane_at[0] != target:
            self._lane_at = (target, self._lane_plain())
        return self._lane_at[1]

    @staticmethod
    def _captures(tasks: list) -> tuple:
        """SPD-233: what tells a round captured a graph, compared at its start and its end: the captures of the
        joint graph caches the lanes' engines keep (engine/models/graph_lru.py, `fn.joint_graphs`), the lanes' draft
        heads' captures, and the lanes whose engine says its last call captured (`captured`)."""
        jg = {id(g): g for t in tasks
              if (g := getattr(getattr(t.eng, "fn", None), "__dict__", {}).get("joint_graphs")) is not None}
        joint = sum(int(getattr(g, "captures", 0) or 0) for g in jg.values())
        heads = sum(int(getattr(getattr(t.drafter, "head", None), "captures", 0) or 0) for t in tasks)
        engs = tuple(bool(getattr(t.eng, "captured", False)) for t in tasks)
        return joint, heads, engs

    def _lane_plain(self) -> bool:
        """SPD-233: whether the next round's lanes take the plain step (no drafts): at E live lanes, all of them of
        families that join a plain step to the joint pass, the mode whose rounds made more tokens a GPU ms; the
        drafted rate is learned first, then the plain rate, and every LANE_PROBE-th round at E runs the other mode."""
        live = [t for t in self.tasks if not t.done]
        e = len(live)
        if e < 2 or not all(getattr(t.eng.caps, "joint_plain", False) and hasattr(t.drafter, "lane_skip")
                            and (LANE_MODE == "1" or getattr(t.eng, "lane_plain", False)) for t in live):
            return False
        st = self._lane.setdefault(e, {"drafted": None, "plain_rate": None, "nd": 0, "np": 0, "n": 0, "plain": False})
        st["n"] += 1
        if st["nd"] < 4:
            return False
        if st["np"] < 2:
            return True
        d, p = st["drafted"], st["plain_rate"]
        if st["plain"] and d > p * (1 + LANE_MARGIN):
            st["plain"] = False
        elif not st["plain"] and p > d * (1 + LANE_MARGIN):
            st["plain"] = True
        return (not st["plain"]) if st["n"] % LANE_PROBE == 0 else st["plain"]

    def _lane_note(self, e: int, plain: bool, rate: float) -> None:
        """A finished round at e lanes: its tokens a GPU ms into its mode's running mean (a fifth of the new)."""
        st = self._lane.setdefault(e, {"drafted": None, "plain_rate": None, "nd": 0, "np": 0, "n": 0, "plain": False})
        k, n = ("plain_rate", "np") if plain else ("drafted", "nd")
        st[k] = rate if st[k] is None else 0.8 * st[k] + 0.2 * rate
        st[n] += 1

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
            self._round_plain = all(len(x.tokens) == 1 for x in trees)
            self._round_rows = sum(len(x.tokens) for x in trees)
            self._round_exp += sum(sum(x.scores[1:]) for x in trees)
        self._hc_tree(trees)
        self._hc_seg("cut")
        t0 = time.perf_counter()
        if HOST_BLOCKS and len(group) > 1 and getattr(group[0].eng, "host_blocks", False):
            blocks = [torch.tensor(x.tokens, dtype=torch.long) for x in trees]
        else:
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
            if hasattr(group[0].eng, "joint_capture_ok"):
                # SPD-249: a family's joint graph may be captured in this pass: on this thread, while no request
                # waits for the GPU (an admission's prefill), never otherwise
                tn = self.pool.turn
                group[0].eng.joint_capture_ok = tn is None or tn.waiting() == 0
            c_v = time.thread_time()
            outs = multi([(t.eng, b, x.parents, t.pos) for t, b, x in zip(group, blocks, trees)])
            self._verify_cpu += 1e3 * (time.thread_time() - c_v)
            if self._busy_at == self.stats["rounds"]:
                # SPD-249: the pass's GPU wait is over: no new gated work, this round's pushes held for the next
                self._defer = BUSY_WORK
                if BUSY_WORK:
                    from engine.methods import busy as busy_mod
                    busy_mod.closed()
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
        self._hc_seg("verify")
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
        self._hc_seg("walk")
        # 4. each task's tail, as the slot loop runs it
        from engine import rowcopy
        with (rowcopy.recording() if ROWCOPY else _nothing()):
            self._tails(group, trees, outs, walked)
        self._t("tails", t0)

    def _tails(self, group, trees, outs, walked) -> None:
        if PIPE:
            t0 = time.perf_counter()
            halves = [self._tail_a(t, x, path, new) for t, x, path, new in self._paths(group, trees, outs, walked)]
            go = [t for t, h in zip(group, halves) if h[2] is None and t.next is None]
            t1 = time.perf_counter()
            self._hc_seg("tail_a")
            self._propose(go, t1, deadline=False)
            for t in go:
                t.pre = True
            t2 = time.perf_counter()
            multi = getattr(group[0].eng, "commit_tree_multi", None) if COMMIT_MULTI and len(group) > 1 else None
            if multi is not None:
                multi([(t.eng, h[0]) for t, h in zip(group, halves)])
            self._hc_seg("commit")
            for t, h in zip(group, halves):
                self._bind(t)
                self._tail_b(t, *h, committed=multi is not None)
            self._hc_seg("tail_b")
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
            self._hc_acc(len(group), len(path) - 1)
            yield t, x, path, new

    def _cut(self, t: Task, path, new) -> tuple:
        """The block's kept rows and tokens: through its first end token and inside the budget
        (`app._end_cut`); SPD-249: before the end token where SP2 asks, with the hand-back (else None)."""
        j = self._sp2_at(t, new)
        if j is not None:
            back = {"why": "sp2", "extra": list(new[:j]), "end": int(new[j]), "room": t.eng.max_len - t.pos}
            return path[:j + 1], new[:j], back
        cut = self.app._end_cut(new, t.eos, t.max_new - t.n_out)
        if cut < len(new):
            path, new = path[:cut + 1], new[:cut]
        return path, new, None

    def _tail(self, t: Task, tree, path, new) -> None:
        from engine.model import h2d
        accepted = len(path) - 1
        path, new, back = self._cut(t, path, new)
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
        if back is not None:
            # SPD-249: SP2 is the slot thread's; the tokens before the end token go with the hand-back
            self._round_tok += len(new)
            self._back(t, back)
            return
        d.observe(new)
        if t.pen is not None:
            t.pen.commit(new)
        stop_now = t.pstop is not None and t.pstop.observe(new)
        t.ctx.extend(new)
        t.n_out += len(new)
        self._round_tok += len(new)
        t.tok = t.ctx[-1]
        self._push(t, new)
        if stop_now:
            t.end("stop")
        elif any(x in t.eos for x in new) or t.n_out >= t.max_new or t.pos >= t.eng.max_len - 1:
            t.end("eos" if any(x in t.eos for x in new) else "length")
        elif t.think is not None:
            t.think.observe(new)            # the slot loop looks at a block it goes on after
            t.next = self._next_stop(t)
            self._after(t)

    def _tail_a(self, t: Task, tree, path, new) -> tuple:
        """PIPE: the part of `_tail` the next draft reads: the cut, the drafter's sync and observation, the position
        and context, and whether the task ends (it ends in `_tail_b`, after its tokens are pushed). Returns
        (path, new, the end reason or None)."""
        from engine.model import h2d
        accepted = len(path) - 1
        path, new, back = self._cut(t, path, new)
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
        if back is not None:
            # SPD-249: SP2 is the slot thread's: the rows before the end token are committed (`_tail_b`), and
            # the drafter's observation, the penalties, the stop patterns and the context wait for its answer
            t.pos += len(path)
            self._round_tok += len(new)
            t.back = back
            return path, new, "back"
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
            t.next = self._next_stop(t)     # SPD-249: no draft ahead for a block that goes back or waits
        return path, new, end

    def _tail_b(self, t: Task, path, new, end, committed: bool = False) -> None:
        """PIPE: the rest of `_tail` while the GPU drafts: the target's commit (unless the round committed every
        slot at once), the penalties, the stop patterns, the stream, the end."""
        if not committed:
            t.eng.commit_tree(path)
        if end == "back":
            back, t.back = t.back, None
            self._back(t, back)             # SPD-249: SP2's (`_tail_a`)
            return
        if t.pen is not None:
            t.pen.commit(new)
        stop_now = t.pstop is not None and t.pstop.observe(new)
        self._push(t, new)
        if stop_now:
            t.next = None
            t.end("stop")
        elif end is not None:
            t.end(end)
        else:
            self._after(t)

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
        if t.sup is not None and tok in t.eos:
            # SPD-249: SP2 on the plain step, the slot thread's: the end token was never forwarded, and nothing
            # has taken it yet
            self._back(t, {"why": "sp2_plain", "extra": [], "end": int(tok), "room": eng.max_len - t.pos})
            return
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
            t.next = self._next_stop(t)
        t.push([tok])
        if stop_now:
            t.next = None
            t.end("stop")
        elif tok in t.eos or t.n_out >= t.max_new or t.pos >= eng.max_len - 1:
            t.next = None
            t.end("eos" if tok in t.eos else "length")
        else:
            self._after(t)


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
