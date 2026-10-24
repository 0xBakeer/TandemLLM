"""An error frame that rides with choices: text consumed (as on main), error recorded,
run completes carrying a remark — never a silent done, never a second end-of-run event.

On main an {"error": ..., "choices": [...]} frame keeps streaming its choices; only
the error text is lost, so a plausible truncated answer arrives with no reason
anywhere. Policy here: the text is kept as before, the error is recorded whatever
the frame carries, and a run that completes with an error on record carries a
run.remark — which service persists as a notice part at the answer's end. The run's
end stays finish_run's alone: a run.error mid-run is the reducer's end-of-run signal
and a live run is not over.

Run: python tests/test_chatd_runs_error_frames.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatd.runs import Run


def _run_with(frames):
    r = Run.__new__(Run)
    r.call = type("C", (), {"events": staticmethod(lambda: iter(frames))})()
    r.request_id = None
    r.request_ids = set()
    r.usage = None
    r.step_usage = []
    r.step = 1
    r.acc = {"completion": 0, "reasoning": 0}
    r.call_base = (0, 0)
    r.request_chars = 0
    r.request_images = 0
    r.stop_mode = None
    r.carry = ""
    r.in_tag = False
    r.finish_reason = None
    r.error = None
    r.notice = None
    r.remark = None
    r.phase = "queued"
    r.now = ""
    r.trim = False
    r.spec = {"predicted_n": 0}
    r.message_id = None
    r.id = None
    r.conv = None
    r._content_texts = []
    r._content = lambda s: r._content_texts.append(s)     # type: ignore[method-assign]
    r._add = lambda kind, s: None                         # type: ignore[method-assign]
    r._tool_delta = lambda tc: None                       # type: ignore[method-assign]
    r.set_phase = lambda *a, **kw: None                   # type: ignore[method-assign]
    r.svc = type("S", (), {"flush": staticmethod(lambda run: None),
                           "learn_ratio": staticmethod(lambda *a: None)})()
    r.add_spec = lambda t: None                           # type: ignore[method-assign]
    return r


def test_an_error_frame_with_choices_records_the_error_and_keeps_streaming():
    run = _run_with([
        ("data", {"error": {"message": "engine hiccup"},
                  "choices": [{"delta": {"content": "still here"}}]}),
    ])
    out = run._consume()
    assert run.error == "engine hiccup", run.error
    assert run._content_texts == ["still here"], run._content_texts
    assert out != "failed", "choices on the frame mean the stream is still live"


def test_an_error_frame_without_choices_fails_the_run_as_on_main():
    run = _run_with([("data", {"error": {"message": "no room at all"}})])
    assert run._consume() == "failed"
    assert run.error == "no room at all"


def test_a_completed_run_with_an_error_on_record_carries_a_remark():
    run = _run_with([
        ("data", {"error": {"message": "half-incident"},
                  "choices": [{"delta": {"content": "tail"}}]}),
        ("data", {"choices": [{"delta": {}, "finish_reason": "stop"}]}),
    ])
    emitted = []
    run.svc = type("S", (), {"emit": lambda self, conv, ev, **k: emitted.append(ev),
                             "flush": lambda self, *a, **k: None,
                             "learn_ratio": lambda self, *a, **k: None})()
    out = run._consume()
    assert out == "completed" and run.error == "half-incident", (out, run.error)
    assert run.remark and "half-incident" in run.remark, run.remark
    ends = [e for e in emitted if isinstance(e, dict)
            and e.get("type") in ("run.error", "run.done", "run.finished")]
    assert not ends, f"consume ended a live run: {ends}"


def test_a_later_step_with_the_error_still_on_record_adds_no_second_remark():
    run = _run_with([
        ("data", {"error": {"message": "first incident"},
                  "choices": [{"delta": {"content": "a"}}]}),
        ("data", {"choices": [{"delta": {}, "finish_reason": "stop"}]}),
    ])
    run._consume()
    first = run.remark
    assert first, "the first step did not stamp the remark"
    # self.error is never reset between model calls: a later completing step must
    # not stamp a second remark (and must not overwrite the first)
    run2_frames = [("data", {"choices": [{"delta": {"content": "b"},
                                          "finish_reason": "stop"}]})]
    run.call = type("C", (), {"events": staticmethod(lambda: iter(run2_frames))})()
    run._consume()
    assert run.remark == first, "the remark was re-stamped on a later step"


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    n = ok = 0
    for name, fn in tests:
        n += 1
        try:
            fn()
            print(f"  {name:56s} ok")
            ok += 1
        except Exception as exc:                                       # noqa: BLE001
            print(f"  {name:56s} FAIL {type(exc).__name__}: {exc}")
    print(f"{ok}/{n}")
    sys.exit(0 if ok == n else 1)
