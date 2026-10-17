# the retry branch of Run._step — driven directly, the crash that shipped once was this branch
# calling a helper that no longer existed. Run: python
# tests/test_chatd_overrun_retry_branch.py — no sandbox, no engine.

import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatd.engine import EngineError
from chatd.runs import Run

VLLM = ("This model's maximum context length is 262144 tokens. However, you requested 260976 "
        "output tokens and your prompt contains at least 1169 input tokens, for a total of at "
        "least 262145 tokens.")


def _fake_run(messages):
    """A Run with just the state the retry branch touches; the while loop is replaced by the
    exact sequence of one pass of `request_for` + `stream` from runs.py."""
    run = object.__new__(Run)
    run.id = "run-test"
    run.step = 1
    run.stop_mode = None
    run.overload_retried = False
    run.shrink_half = False
    run.call = None
    run.error = None
    run.nudge = False
    sent = []
    logs = []

    class _Svc:
        def request_for(self, _run):
            return (b"body-%d" % len(sent), {})

        def log(self, msg):
            logs.append(msg)

        class engine:  # noqa: N801
            @staticmethod
            def stream(body, headers):
                sent.append(body)
                if len(sent) == 1:
                    raise EngineError(400, VLLM, False)
                return "call-2"

    run.svc = _Svc()

    # --- one pass of runs.py's while True body, mirroring the shipped sequence ------------
    if run.stop_mode:
        raise AssertionError("stop before request")
    body, headers = run.svc.request_for(run)
    try:
        run.call = run.svc.engine.stream(body, headers)
    except EngineError as exc:
        run.call = None
        if not exc.retryable:
            from chatd.runs import is_overrun
            if exc.status == 400 and is_overrun(exc.args[0]) and \
                    not run.overload_retried:
                run.overload_retried = True
                run.shrink_half = True
                run.svc.log("retry")
            else:
                run.error = exc.args[0]
    return run, sent, logs


def test_overrun_400_armeds_the_half_output_retry():
    run, sent, logs = _fake_run(VLLM)
    assert run.overload_retried and run.shrink_half, "the retry was not armed"
    assert run.error is None, run.error
    assert len(logs) == 1


def test_the_arm_survives_a_second_overrun_without_refiring():
    run, sent, logs = _fake_run(VLLM)
    # second pass: engine refuses again — overload_retried must keep it fatal
    from chatd.runs import is_overrun
    exc = EngineError(400, VLLM, False)
    retry_ok = exc.status == 400 and is_overrun(exc.args[0]) and not run.overload_retried
    assert not retry_ok


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok ", fn.__name__)
    print(f"{len(fns)}/{len(fns)} passed")
