"""Four requests that arrive together on a fresh server all find the tool-call formats: the lazy
discovery marked them found before their modules had registered, so a request that looked in
between found none and answered 500 ("no tool-call format 'xml'"). Found is marked last now, and
the discovery runs under a lock, once."""

from __future__ import annotations

import threading
import time


def test_found_is_marked_after_every_module_and_once(monkeypatch):
    from server import toolformats as tf
    monkeypatch.setattr(tf, "_found", [])
    seen, calls = [], []
    real = tf.importlib.import_module

    def watch(name, *a, **k):
        seen.append(bool(tf._found))
        calls.append(name)
        time.sleep(0.02)
        return real(name, *a, **k)
    monkeypatch.setattr(tf.importlib, "import_module", watch)
    ts = [threading.Thread(target=tf.get, args=("xml",)) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert seen and not any(seen), "found was marked before a module had registered"
    assert len(calls) == len(set(calls)), "the discovery ran more than once"
    assert tf._found and tf.get("xml").name == "xml"


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
