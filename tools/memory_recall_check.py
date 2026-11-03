#!/usr/bin/env python3
"""Memory recall on the chat path (SPD-249): questions whose answer is only in a memory note.

    python3 tools/memory_recall_check.py seed --conductor DIR [--filler 34] [--force]
    python3 tools/memory_recall_check.py ask --out FILE [--base-url URL] [--tag LABEL]
    python3 tools/memory_recall_check.py compare FILE FILE ...

`seed` writes synthetic notes for the user `owner` into the memory of the conductor config at DIR
(a copy, never a production config: its memory is DIR/memory, `transcripts.memory_root`). It refuses the
default config, the one `QSE_CONDUCTOR_CONFIG` names, and a memory that holds notes already (`--force`
seeds one anyway, for a copy seeded before): FACTS, one
note each with a fact no model knows, and `--filler` notes about other things, so the index holds
about 40 lines as a real one does. `ask` sends each fact's question in a fresh chat (its own session,
model TandemLLM, no raw header: the conductor, its memory reads and its notes as the server is set up)
and scores an answer right when it carries the fact's key. `compare` prints the runs side by side.

What a run compares is what the server was started with: the notes' layout (QWEN38_NOTES_HEAD), the
index (`memory.notes.index_order`, `index_tokens`). Run each arm on a fresh start; run one arm twice
for the run-to-run noise. Standard library only (`seed` imports the server's note store).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
import uuid

#: (title, body, question, key): the key must appear in a right answer
FACTS = [
    ("bike lock code", "The combination of the user's bike lock is 4719.",
     "What is the combination of my bike lock?", "4719"),
    ("dentist appointment weekday", "The user's dentist appointments are always on Thursday mornings.",
     "On which weekday are my dentist appointments?", "thursday"),
    ("favourite tea", "The user's favourite tea is lapsang souchong, brewed for three minutes.",
     "Which tea do I like best?", "lapsang"),
    ("server rack label", "The home server rack is labelled ORCHID-7 on its front panel.",
     "What label is on my home server rack?", "orchid-7"),
    ("allergy", "The user is allergic to hazelnuts.", "What am I allergic to?", "hazelnut"),
    ("backup drive name", "The weekly backup goes to the external drive named Kestrel.",
     "What is the name of the drive my weekly backup goes to?", "kestrel"),
    ("piano teacher", "The user's piano teacher is called Ms. Varga.", "What is my piano teacher's name?",
     "varga"),
    ("garden shed key", "The garden shed key hangs on the blue hook behind the kitchen door.",
     "Where does the garden shed key hang?", "blue hook"),
    ("running distance", "The user's usual Sunday run is 14 kilometres along the river.",
     "How far is my usual Sunday run?", "14"),
    ("car colour", "The user's car is a dark green estate.", "What colour is my car?", "green"),
    ("wifi channel", "The upstairs access point is fixed to Wi-Fi channel 44.",
     "Which Wi-Fi channel is the upstairs access point fixed to?", "44"),
    ("book club day", "The user's book club meets on the first Tuesday of each month.",
     "When does my book club meet?", "first tuesday"),
]

FILLER = ["prefers short answers", "likes dark mode", "writes Python first", "uses metric units",
          "reads on an e-reader", "plays chess on weekends", "keeps notes in Markdown", "drinks no coffee",
          "uses a split keyboard", "likes long walks", "prefers trains to planes", "grows tomatoes",
          "cooks on Sundays", "learns Spanish", "listens to jazz", "uses Linux at work",
          "names files with dates", "plans the week on Mondays", "likes board games", "cycles to work",
          "keeps a paper calendar", "prefers email to calls", "takes photos on film", "runs a home lab",
          "reads history books", "uses a standing desk", "likes spicy food", "volunteers monthly",
          "watches documentaries", "collects postcards", "swims in summer", "builds model trains",
          "prefers tea to coffee", "uses two monitors", "likes crosswords", "keeps houseplants"]


def refusal(root: str, memory: str, force: bool) -> str | None:
    """Why `seed` must not write into the config at `root` (its memory at `memory`), or None."""
    real = os.path.realpath(root)
    served = [os.path.expanduser("~/.qwen38-spark-engine/conductor"), os.environ.get("QSE_CONDUCTOR_CONFIG") or ""]
    if any(p and p != "off" and real == os.path.realpath(os.path.expanduser(p)) for p in served):
        return "seed writes into a copy of the conductor config, never the one a server runs with"
    if not force and os.path.isdir(memory) and any(os.scandir(memory)):
        return f"{memory} holds notes already (--force seeds it anyway)"
    return None


def seed(a) -> int:
    root = os.path.abspath(os.path.expanduser(a.conductor))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from server.conductor.notes import NoteStore
    from server.conductor.transcripts import memory_root
    why = refusal(root, memory_root(root), a.force)
    if why is not None:
        print(f"refused: {why}")
        return 2
    st = NoteStore(memory_root(root), settings=lambda: {"dedupe_threshold": 0.95})
    n = 0
    for title, body, _, _ in FACTS:
        st.write("owner", {"op": "add", "scope": "user", "kind": "reference", "title": title, "body": body},
                 writer="human", project=None, session=None, request_id="recall-seed", from_user=True)
        n += 1
    for f in FILLER[:a.filler]:
        st.write("owner", {"op": "add", "scope": "user", "kind": "reference", "title": f"the user {f}",
                           "body": f"The user {f}."}, writer="human", project=None, session=None,
                 request_id="recall-seed", from_user=True)
        n += 1
    print(f"seeded {n} notes for owner in {memory_root(root)}")
    return 0


def _post(base, body, timeout):
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "x-session-id": f"recall-{uuid.uuid4().hex[:12]}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def ask(a) -> int:
    rows = []
    for title, _, q, key in FACTS:
        t0 = time.time()
        try:
            r = _post(a.base_url, {"model": "TandemLLM", "stream": False, "max_tokens": a.max_tokens,
                                   "messages": [{"role": "user", "content": q}]}, a.timeout)
            msg = (r.get("choices") or [{}])[0].get("message") or {}
            text = msg.get("content") or ""
            ok = key.lower() in text.lower()
            err = None
        except Exception as e:  # noqa: BLE001
            text, ok, err = "", False, repr(e)
        rows.append({"fact": title, "question": q, "key": key, "right": ok, "answer": text[:400],
                     "error": err, "s": round(time.time() - t0, 1)})
        print(f"{'right' if ok else 'WRONG'} {title}: {text[:90]!r}", flush=True)
    score = sum(r["right"] for r in rows)
    with open(a.out, "w") as fh:
        json.dump({"tag": a.tag, "score": score, "of": len(rows), "rows": rows,
                   "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, fh, indent=1)
    print(f"{a.tag}: {score}/{len(rows)} right")
    return 0


def compare(a) -> int:
    runs = [json.load(open(f)) for f in a.files]
    for r in runs:
        print(f"{r['tag']:24s} {r['score']}/{r['of']}  missed: "
              f"{[x['fact'] for x in r['rows'] if not x['right']]}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("seed")
    s.add_argument("--conductor", required=True)
    s.add_argument("--filler", type=int, default=34)
    s.add_argument("--force", action="store_true", help="seed a memory that holds notes already")
    q = sub.add_parser("ask")
    q.add_argument("--out", required=True)
    q.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    q.add_argument("--tag", default="run")
    q.add_argument("--max-tokens", type=int, default=2048)
    q.add_argument("--timeout", type=float, default=900)
    c = sub.add_parser("compare")
    c.add_argument("files", nargs="+")
    a = ap.parse_args()
    return {"seed": seed, "ask": ask, "compare": compare}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
