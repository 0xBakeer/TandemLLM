"""SRV-717: a sampler of every thread's stack for a box without ptrace (QSE_STACK_SAMPLE=<path>[,<ms>]).

Every `ms` milliseconds (10 by default) a daemon thread reads `sys._current_frames()` and counts, per thread
name, the innermost frames (file:line function, three deep) of each thread that is not waiting (its innermost
function is not a wait, a sleep, a socket or a select). Every 20 s and at exit it writes the counts as JSON to
`path`: the busiest stacks per thread first. It only reads frames; a sample costs a few tens of microseconds
under the GIL, so it is a measurement tool, off unless named."""

from __future__ import annotations

import atexit
import collections
import json
import os
import sys
import threading
import time

IDLE = {"wait", "sleep", "select", "poll", "accept", "recv", "recv_into", "readinto", "read", "get",
        "_wait_for_tstate_lock", "acquire", "epoll", "readline", "handle_request", "serve_forever",
        "_recv_bytes", "_poll", "join"}


def start(spec: str) -> threading.Thread | None:
    path, _, ms = spec.partition(",")
    if not path:
        return None
    every = max(1.0, float(ms or 10)) / 1e3
    counts: dict = collections.defaultdict(collections.Counter)
    busy: collections.Counter = collections.Counter()
    seen: collections.Counter = collections.Counter()
    me = threading.get_ident()
    t0 = time.monotonic()

    def dump():
        names = {t.ident: t.name for t in threading.enumerate()}
        out = {"seconds": round(time.monotonic() - t0, 1), "interval_ms": every * 1e3, "threads": {}}
        for tid, c in counts.items():
            nm = names.get(tid, str(tid))
            out["threads"][f"{nm} ({tid})"] = {"busy": busy[tid], "samples": seen[tid],
                                              "top": [[k, v] for k, v in c.most_common(25)]}
        out["threads"] = dict(sorted(out["threads"].items(), key=lambda kv: -kv[1]["busy"]))
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(out, fh, indent=1)
        os.replace(tmp, path)

    def run():
        last = time.monotonic()
        while True:
            time.sleep(every)
            for tid, fr in sys._current_frames().items():
                if tid == me:
                    continue
                seen[tid] += 1
                if fr.f_code.co_name in IDLE:
                    continue
                busy[tid] += 1
                key, f, n = [], fr, 0
                while f is not None and n < 3:
                    key.append(f"{os.path.basename(f.f_code.co_filename)}:{f.f_lineno} {f.f_code.co_name}")
                    f, n = f.f_back, n + 1
                counts[tid][" < ".join(key)] += 1
            if time.monotonic() - last >= 20.0:
                last = time.monotonic()
                try:
                    dump()
                except OSError:
                    pass

    atexit.register(lambda: dump())
    th = threading.Thread(target=run, name="stack-sampler", daemon=True)
    th.start()
    return th
