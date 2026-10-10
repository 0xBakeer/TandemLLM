"""On the board: Wren, the setup guide's assistant (server/setup/wren.py), loaded, asked and unloaded
on the GPU, alone or borrowing the narrator's copy. No model of Qwen's in this process.

  own     Wren as setup mode loads it: its own Qwen3.5-2B at 8,192 rows through the planner. The
          load's time and MemAvailable drop, three turns (a question about a tool, "switch the
          narrator off", "use two slots") with the time to the first text and the tool calls
          the server checked, then the unload and MemAvailable back (within 0.3 GiB).
  borrow  the narrator's slot as the server builds it (Qwen3.5-2B, 4,096 rows, its own pool),
          one narrator line N0; Wren borrows the loaded slot (MemAvailable must not move by more
          than 0.3 GiB), one turn; the narrator's line again must equal N0 to the token.

Run on the box while :8000 has been idle for at least 10 minutes, under the box lock, with the
served environment and a private lock path for the planner (the real one is the test's own):
    flock -n ~/.qwen38-box.flock env BOX_LOCK=/tmp/wren-gpu.flock \\
        bash -c 'set -a; . ~/tandem-live/ops/serve.env; set +a; \\
                 $PY tests/gpu/setup_wren_gpu.py own borrow'
It prints one JSON line per part.
"""

from __future__ import annotations

import json
import os
import sys
import time
import types

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from server import app  # noqa: E402
from server.setup import wren  # noqa: E402

GIB = 1 << 30
PAGE = {"model": "qwen3.8-27b", "step": "components", "step_index": 3,
        "selection": {"profile": "exl3-4.5", "slots": 4,
                      "components": {"narrator": True, "shadow": True, "laya": True}}}


def avail() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024 / GIB
    raise RuntimeError("no MemAvailable")


def settle() -> float:
    import torch
    torch.cuda.synchronize()
    time.sleep(1.0)
    return avail()


def turn(w, text: str, page: dict) -> dict:
    t0 = time.perf_counter()
    first, said, calls, events = None, "", [], []
    for ev in w.chat({"messages": [{"role": "user", "content": text}], "page": page}):
        events.append(ev["type"])
        if ev["type"] == "text":
            first = first or time.perf_counter()
            said += ev["delta"]
        elif ev["type"] == "tool_call":
            calls.append({"name": ev["name"], "args": ev["args"]})
        elif ev["type"] in ("done", "error"):
            end = ev
    took = time.perf_counter() - t0
    return {"question": text, "first_text_s": round(first - t0, 2) if first else None,
            "s": round(took, 2), "text": said, "tool_calls": calls, "end": end}


def own() -> dict:
    app.STATE.clear()
    w = wren.Wren(app, device_facts=lambda: {"mem_available_gb": round(avail(), 1)},
                  setup_mode=True, log=lambda s: print(s, flush=True))
    app.STATE["wren"] = w
    before = settle()
    t0 = time.perf_counter()
    st = w.load()
    while w.state == "loading":
        time.sleep(0.2)
    load_s = time.perf_counter() - t0
    after = settle()
    st = w.status()
    if st["state"] != "loaded":
        return {"part": "own", "state": st}
    turns = [turn(w, "What does the shadow reader do, and how much memory does it take?", PAGE),
             turn(w, "Please switch the narrator off.", PAGE),
             turn(w, "Use two slots.", dict(PAGE, step="parallelism", step_index=4))]
    w.unload("asked")
    back = settle()
    # a second load and unload: what the first left behind is the process's once (the CUDA
    # context's modules and workspaces), what the second leaves would be a leak
    w.load()
    while w.state == "loading":
        time.sleep(0.2)
    again = settle()
    w.unload("asked")
    back2 = settle()
    return {"part": "own", "model": st["model"], "window": st["window"],
            "load_s": round(load_s, 1), "slot_load_s": st["load_s"],
            "device_gib": st["device_gib"], "mem_available_before": round(before, 2),
            "mem_available_loaded": round(after, 2), "took_gib": round(before - after, 2),
            "mem_available_unloaded": round(back, 2),
            "second_load_took_gib": round(back - again, 2),
            "mem_available_unloaded_again": round(back2, 2),
            "second_cycle_back_within_0_3": abs(back2 - back) <= 0.3,
            "turns": turns, "stats": dict(w.stats)}


def borrow() -> dict:
    from engine.decision.device import Device
    from engine.generative.qwen35 import Qwen35Model
    from engine.generative.slot import Slot
    from server.conductor import narrator as narrator_mod
    app.STATE.clear()
    name, folder, _ = wren.choose()
    before = settle()
    slot = Slot(Qwen35Model(name, folder), role="narrator", gate=app.LOCK, planner=None,
                device=Device("auto"), log=lambda s: print(s, flush=True))
    slot.load(wait=True)
    narr_loaded = settle()
    msgs = narrator_mod.messages({"kind": "route", "facts": {"cls": "code_edit",
                                                             "effort": "medium effort"}})
    n0 = slot.run(msgs, max_new=24, budget_ms=10_000, held=False)
    app.STATE["narrator"] = types.SimpleNamespace(slot=slot)
    w = wren.Wren(app, device_facts=lambda: {"mem_available_gb": round(avail(), 1)},
                  log=lambda s: print(s, flush=True))
    app.STATE["wren"] = w
    st = w.load()
    borrowed = settle()
    t = turn(w, "What does the narrator do?", PAGE)
    n1 = slot.run(msgs, max_new=24, budget_ms=10_000, held=False)
    w.unload("asked")
    still = slot.state
    slot.unload()
    return {"part": "borrow", "model": name, "borrowed": st["borrowed"], "window": st["window"],
            "claim_gib": st["claim_gib"], "narrator_took_gib": round(before - narr_loaded, 2),
            "borrow_took_gib": round(narr_loaded - borrowed, 2), "turn": t,
            "narrator_line_before": n0.get("text"), "narrator_line_after": n1.get("text"),
            "narrator_same": n0.get("text") == n1.get("text") and bool(n0.get("text")),
            "narrator_still_loaded_after_wren_unload": still == "loaded"}


if __name__ == "__main__":
    for part in sys.argv[1:] or ["own", "borrow"]:
        print(json.dumps({"own": own, "borrow": borrow}[part]()), flush=True)
