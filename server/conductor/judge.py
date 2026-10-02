"""Laya as the test judge, in shadow (J1 step 7, Fable's plan of 2026-10-02 sections 1.3 and 4).

Laya (the engine's own decision model, a cross-encoder of about 1B parameters) answers typed
questions on a digest of 320 to 768 tokens in 20 to 44 ms when the engine is idle. It reads text
only and was trained for routing, not for test evidence, so it is asked only where Tier 0 narrowed
the case and a text reading is still needed, and it acts on nothing here:

    failure   a run's output with no exit code and no test runner's counts: does it show a
              failure?  failure | clean | unclear; the Tier 0 twin is the error-line pattern
    build     a `.tandem/*.log` tail the model printed: a successful build or run?
              ok | failed | incomplete; the twin is the error-line pattern again

The questions go out after the request ended (`Conductor.finish`), on this module's own thread, to
the `decider` role's endpoint (in process: the engine's registry, idle window only) with the
router document's `outcomes.judge.deadline_ms` (100) as the deadline. Every answer, a timeout and
a failure is one journal row (`kind: outcome`, task `judge`): the question, the answer with its
probabilities, the twin, the time. Those rows, labelled by hand, are the set a question has to
pass (confident precision at or above 0.95 and coverage at or above 0.3,
`tools/judge_report.py`) before it may act; until then Qwen in context reads the narrowed cases.
Fail-open: nothing here can reach a request.
"""

from __future__ import annotations

import queue
import threading
import time

from server.conductor.classify import decider_for

QUEUE_MAX = 64


class Judge:
    """The judge's thread. `roles()`: the roles document; `journal`: the conductor's journal;
    `factory(endpoint, model, timeout_ms)`: the decider client (tests put a fake here)."""

    def __init__(self, roles, journal, factory=decider_for):
        self.roles, self.journal, self.factory = roles, journal, factory
        self._q: queue.Queue = queue.Queue(QUEUE_MAX)
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.stats = {"asked": 0, "answered": 0, "failed": 0, "dropped": 0}

    def submit(self, plan) -> bool:
        """Queue the plan's questions (`plan.judge`); False when there are none or the queue is
        full (counted)."""
        qs = getattr(plan, "judge", None)
        if not qs:
            return False
        with self._lock:
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="conductor-judge",
                                                daemon=True)
                self._thread.start()
        try:
            self._q.put_nowait((plan.session.key if plan.session else None,
                                (plan.row or {}).get("request_id"), list(qs)))
            return True
        except queue.Full:
            self.stats["dropped"] += 1
            return False

    def flush(self, timeout: float = 5.0) -> None:
        end = time.time() + timeout
        while self._q.unfinished_tasks and time.time() < end:
            time.sleep(0.01)

    def _run(self) -> None:
        while True:
            item = self._q.get()
            try:
                self.ask(*item)
            except Exception as exc:                               # noqa: BLE001
                print(f"[conductor] the judge failed: {type(exc).__name__}: {exc}", flush=True)
            finally:
                self._q.task_done()

    def ask(self, session_key, request_id, questions: list[dict]) -> list[dict]:
        """Each question to the decider, journaled; the rows written."""
        rows = []
        role = (self.roles() or {}).get("decider") or {}
        for q in questions:
            detail = {"question": q["qid"], "twin": q.get("twin"), "call": q.get("call"),
                      "chars": len(q["digest"]), "deadline_ms": q["deadline_ms"]}
            label = "unknown"
            self.stats["asked"] += 1
            if role.get("mode", "off") == "off" or not role.get("endpoint"):
                detail["error"] = "role_off" if role.get("mode", "off") == "off" else "no_endpoint"
            else:
                client = self.factory(role["endpoint"], role.get("model") or "auto",
                                      int(q["deadline_ms"]))
                if client is None:
                    detail["error"] = "no_client"
                else:
                    ans = client.choice(q["digest"], q["question"], q["options"])
                    detail.update(ms=round(ans.ms, 1), model=ans.model, error=ans.error,
                                  confidence=round(float(ans.confidence or 0.0), 4),
                                  probs={k: round(float(v), 4) for k, v in
                                         (ans.probs or {}).items()})
                    if ans.error is None and ans.label in q["options"]:
                        label = ans.label
                        detail["agree"] = (label == q.get("twin")) if q.get("twin") else None
            self.stats["answered" if label != "unknown" else "failed"] += 1
            rows.append(self.journal.note("outcome", "judge", decision=label, detail=detail,
                                          session_key=session_key, request_id=request_id))
        return rows


def questions(snap: dict, cfg: dict) -> list[dict]:
    """The judge's questions for a snapshot's narrowed cases (`snap["_narrow"]`), at most one of
    each kind, digests within `judge.digest_chars`."""
    jc = cfg["judge"]
    out, seen = [], set()
    for n in reversed(snap.get("_narrow") or ()):
        qid = n["q"]
        spec = (jc.get("questions") or {}).get(qid)
        if spec is None or qid in seen:
            continue
        seen.add(qid)
        text = n["text"]
        lim = int(jc["digest_chars"])
        if len(text) > lim:
            head = int(lim * 0.3)
            text = text[:head] + "\n[...]\n" + text[-(lim - head - 7):]
        out.append({"qid": qid, "question": spec["question"], "options": dict(spec["options"]),
                    "digest": text, "twin": n.get("twin"), "call": n.get("call"),
                    "deadline_ms": int(jc["deadline_ms"])})
    return out
