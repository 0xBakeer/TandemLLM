"""test_autodetect_wire — the auto-wire gate matrix, fully offline.

wire_llms is the ONLY place the observer writes config, so every gate gets a
case of its own: chat-proven LLM wires; STT (404 chat route), weak port-hint,
non-loopback bind, duplicate URL, the fleet cap, and a human-deleted backend
(must stay deleted, persisted) must all refuse.
"""

from proxy import autodetect as A


class FakeCfg:
    def __init__(self, backends=None):
        self.b = [dict(x) for x in (backends or [])]
        self.settings = {}

    def backends(self):
        return [dict(x) for x in self.b]

    def setting(self, key, default=None):
        return self.settings.get(key, default)

    def set_setting(self, key, value, who="gui"):
        self.settings[key] = value

    def create_backend(self, doc, who="gui"):
        if any(b["id"] == doc["id"] for b in self.b):
            raise RuntimeError("exists")
        assert who == "autodetect"            # the audit line must name the wire, not 'gui'
        self.b.append(dict(doc))
        return doc


def _fetch(url, timeout):
    port = int(url.split("//")[1].split(":")[1].split("/")[0])
    if "/v1/chat/completions" in url:
        return (404, "") if port == 8907 else (405, "")   # parakeet: no chat route
    return (200, "")


def _eng(port, fam="vllm", how="metrics:vllm:", bind="127.0.0.1", model="M", kind=None):
    return {"port": port, "family": fam, "how": how, "bind": bind, "model": model,
            "kind": kind or fam}


def setup_function(fn):
    pass   # no process state anymore — dismissal is fully persisted (settings)


def test_llm_gets_wired():
    cfg = FakeCfg()
    out = A.wire_llms({"engines": [_eng(8000)]}, cfg, fetch=_fetch, post=_fetch)
    assert out == ["auto-8000"]
    b = cfg.b[0]
    assert b["url"] == "http://127.0.0.1:8000/v1" and b["kind"] == "vllm"
    assert b["models_mode"] == "discover" and b["enabled"] is True


def test_stt_shown_but_never_wired():
    cfg = FakeCfg()
    e = _eng(8907, fam="openai", how="v1/models shape", model="parakeet-tdt-0.6b-v3")
    out = A.wire_llms({"engines": [e]}, cfg, fetch=_fetch, post=_fetch)
    assert out == [] and cfg.b == []


def test_weak_hint_never_mints():
    cfg = FakeCfg()
    e = _eng(11434, fam="ollama", how="port-hint:11434 (weak)")
    assert A.wire_llms({"engines": [e]}, cfg, fetch=_fetch, post=_fetch) == []


def test_loopback_only_and_dedupe():
    cfg = FakeCfg([{"id": "mine", "url": "http://127.0.0.1:8001/v1"}])
    out = A.wire_llms({"engines": [_eng(8002, bind="192.168.1.5"), _eng(8001)]},
                      cfg, fetch=_fetch, post=_fetch)
    assert out == []                              # LAN bind refused; 8001 already owned


def test_wildcard_bind_wires_loopback_url():
    cfg = FakeCfg()                                # docker's 0.0.0.0 default
    out = A.wire_llms({"engines": [_eng(8000, bind="0.0.0.0")]},
                      cfg, fetch=_fetch, post=_fetch)
    assert out == ["auto-8000"] and cfg.b[0]["url"] == "http://127.0.0.1:8000/v1"


def test_fleet_cap():
    auto = [{"id": f"auto-{9000 + i}", "url": f"http://127.0.0.1:{9000 + i}/v1"}
            for i in range(A.AUTO_WIRE_MAX)]
    cfg = FakeCfg(auto)
    out = A.wire_llms({"engines": [_eng(9100)]}, cfg, fetch=_fetch, post=_fetch)
    assert out == []                              # cap holds; the loop stays alive


def test_human_deletion_sticks_across_restarts():
    """The bug this encodes: dismissal used to live in process memory, so a
    service restart resurrected a deleted card. 'Was ours' is now a persisted
    prev-pass diff — a fresh process with the same settings keeps refusing."""
    cfg = FakeCfg()
    assert A.wire_llms({"engines": [_eng(8000)]}, cfg, fetch=_fetch, post=_fetch) == ["auto-8000"]
    assert cfg.settings.get(A.CREATED_KEY) == [8000]         # registered immediately
    cfg.b = []                                               # the human deletes it
    assert A.wire_llms({"engines": [_eng(8000)]}, cfg, fetch=_fetch, post=_fetch) == []
    assert cfg.settings.get(A.DISMISSED_KEY) == [8000]
    assert cfg.settings.get(A.CREATED_KEY) == []
    # a 'restart' is just a new pass reading the same settings: still refuses
    assert A.wire_llms({"engines": [_eng(8000)]}, cfg, fetch=_fetch, post=_fetch) == []


def test_delete_between_create_and_next_pass_still_dismisses():
    """The gap between a pass that creates and a pass that registers is closed
    by wire_llms persisting the created set itself, right after minting."""
    cfg = FakeCfg()
    assert A.wire_llms({"engines": [_eng(8003)]}, cfg, fetch=_fetch, post=_fetch) == ["auto-8003"]
    cfg.b = []                                               # deleted before ANY later pass
    assert A.reconcile(cfg) == {8003}                        # prev-pass diff catches it
    assert A.wire_llms({"engines": [_eng(8003)]}, cfg, fetch=_fetch, post=_fetch) == []


# ---------------------------------------------------------------- detection

def test_tensorfold_detected_by_metrics_prefix():
    """0.3.0-fork compatibility: the tensorfold engine exports its stats under
    the `tensorfold:` prefix (verified live 2026-10-18 — no vLLM-named mirror;
    the four keys resolve endswith-style through the prefix) — the scan must
    NAME it, not guess it from port 8122."""
    body = ("# HELP tensorfold:requests_running Running.\n"
            "tensorfold:requests_running 2\n"
            "tensorfold:requests_waiting 0\n"
            "tensorfold:prompt_tokens_total 123\n")

    def fetch(url, timeout):
        return (200, body) if url.endswith("/metrics") else (404, "")

    def fetch_json(url, timeout):
        return None

    r = A.fingerprint("http://127.0.0.1:8122", 8122, fetch=fetch,
                      fetch_json=fetch_json)
    assert r["family"] == "tensorfold" and r["how"] == "metrics:tensorfold:"

    # and its vLLM-named mirror set resolves through the generic suffix scan:
    stats = A._stats_snapshot("http://127.0.0.1:8122", "tensorfold", fetch=fetch)
    assert stats == {"requests_running": 2.0, "requests_waiting": 0.0,
                     "prompt_tokens": 123.0}
