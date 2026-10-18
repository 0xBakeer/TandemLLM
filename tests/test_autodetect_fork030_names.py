"""The 0.3.0 fork's sglang and exl3 name-tables, validated against the observer.

The live engines retired (sglang when the vLLM v5.2 recipe replaced
qwen38-sglang-fn; exl3 with the 0.3.0 proxy). The source of truth for their
metric names is the old fork's own adapters — sglang.mjs / exl3.mjs — which
parsed these EXACT names while those engines actually ran on this box. This
spec replays their name-tables through fingerprint() and _stats_snapshot()
so the observer's naming stays honest to what those families emitted.

Run:  python3 tests/test_autodetect_fork030_names.py
"""
import sys

sys.path.insert(0, ".")

from proxy import autodetect as A   # noqa: E402

PASS = FAIL = 0


def check(label, ok):
    global PASS, FAIL
    if not ok:
        FAIL += 1
        print(f"FAIL {label}")
    else:
        PASS += 1
        print(f"  ok  {label}")


# exact names from TandemDash-Fork/proxy/adapters/sglang.mjs (adapters kept in
# our old fork: these are the names sglang emits with --enable-metrics)
SGLANG_METRICS = """# TYPE sglang:num_running_reqs gauge
sglang:num_running_reqs{engine="0"} 3
sglang:num_queue_reqs{engine="0"} 1
sglang:num_retracted_reqs{engine="0"} 0
sglang:num_paused_reqs{engine="0"} 0
sglang:prompt_tokens_total{engine="0"} 1000
sglang:generation_tokens_total{engine="0"} 250
sglang:cached_tokens_total{engine="0"} 40
sglang:cache_hit_rate{engine="0"} 0.99
sglang:realtime_tokens_total{mode="decode"} 9000
"""

# exact names from exl3.mjs — the 0.3.0 serve-side ledger (server/metrics.py)
EXL3_METRICS = """exl3_requests_running 2
exl3_requests_waiting 0
exl3_requests_total{finish_reason="stop"} 77
exl3_prompt_tokens_total 500
exl3_generation_tokens_total 1400
exl3_accepted_draft_tokens_total 300
exl3_rejected_draft_tokens_total 40
exl3_decode_tokens_per_second 61.2
exl3_kv_capacity_tokens 262144
exl3_kv_used_tokens 31000
"""


def fetch_factory(text):
    def fetch(url, timeout=0.8):
        return (200, text)          # _get's own shape: (status, body-head)
    return fetch


def fetch_json_factory(routes):
    def fj(url, timeout=0.8):
        for suffix, payload in routes.items():
            if url.endswith(suffix):
                return payload
        return None
    return fj


sg = fetch_factory(SGLANG_METRICS)
fp = A.fingerprint("http://127.0.0.1:9", 9, fetch=sg,
                   fetch_json=fetch_json_factory({"/v1/models": {"data": []}}))
check("sglang: metrics name-table names the family",
      fp["family"] == "sglang" and fp["how"] == "metrics:sglang:")

snap = A._stats_snapshot("http://127.0.0.1:9", "sglang", fetch=sg)
check("sglang: running resolves (num_running_reqs)", snap.get("requests_running") == 3)
check("sglang: waiting resolves (num_queue_reqs)", snap.get("requests_waiting") == 1)
check("sglang: prompt tokens resolve", snap.get("prompt_tokens") == 1000)
check("sglang: generation tokens resolve", snap.get("generation_tokens") == 250)
# deliberately unmapped: cache_hit_rate spikes 0<->0.99 per batch — the 0.3.0
# adapter itself documents it as jittery; a 'hit rate' that jitters reads as a bug
check("sglang: the jittery per-batch cache_hit_rate stays unmapped",
      "cache_hit_rate" not in snap)

ex = fetch_factory(EXL3_METRICS)
fp2 = A.fingerprint("http://127.0.0.1:9", 9, fetch=ex,
                    fetch_json=fetch_json_factory({"/v1/models": {"data": []}}))
check("exl3_ name-table names the family",
      fp2["family"] == "exl3" and fp2["how"] == "metrics:exl3_")
snap2 = A._stats_snapshot("http://127.0.0.1:9", "exl3", fetch=ex)
check("exl3: running resolves", snap2.get("requests_running") == 2)
check("exl3: waiting resolves", snap2.get("requests_waiting") == 0)
check("exl3: prompt tokens resolve", snap2.get("prompt_tokens") == 500)
check("exl3: generation tokens resolve", snap2.get("generation_tokens") == 1400)

print(f"\n{PASS}/{PASS + FAIL} passed")
sys.exit(1 if FAIL else 0)
