#!/usr/bin/env python3
"""kinds: table-driven detection, status and usage normalization against fake
engines shaped like the real ones (T10).

Run:  python3 tests/test_kinds.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from proxy import kinds as K                                            # noqa: E402

FAILS: list[str] = []


def check(cond, what):
    if cond:
        print(f"ok  - {what}")
    else:
        FAILS.append(what)
        print(f"FAIL- {what}")


# ------------------------------------------------------- fake fetch responses

# llama.cpp /props (real server: model_path, n_ctx, n_batch, ... and no model_type)
PROPS_LC = {"model_path": "/models/qwen/qwen3-8b-q4_k_m.gguf", "n_ctx": 32768,
            "n_batch": 2048, "n_predict": -1, "n_gpu_layers": 99}

# tabby /v1/api/props (model_info + completion sections)
PROPS_TABBY = {"model_info": {"model": "Mistral-Nemo-Instruct-2407-exl3", "org": "ignite",
                              "max_seq_len": 65536},
               "completion": {"prompt_template": "mistral"}}

# ollama /api/version + /api/ps (running slots)
VER_OLL = {"version": "0.4.1"}
PS_OLL = {"models": [{"name": "llama3.2:3b", "model": "llama3.2:3b",
                      "size_vram": 3200000000, "context_window": 8192}]}

# a vLLM-shape /v1/models + /metrics excerpt (autodetect family, not kinds' job)
MODELS_VLLM = {"data": [{"id": "Qwen/Qwen3-8B", "object": "model"}]}
METRICS_VLLM = "# HELP vllm:num_requests_running\nvllm:num_requests_running{engine=\"0\"} 2.0\n"


def fake_fetch_json(pages: dict):
    def fetch_json(url, timeout=None):
        return pages.get(url)
    return fetch_json


def fetch_text(pages: dict):
    """fetch(url) -> (code, text) | None, in autodetect's seam."""
    def fetch(url, timeout=None):
        got = pages.get(url)
        if got is None:
            return None
        if isinstance(got, str):
            return 200, got
        import json as j
        return 200, j.dumps(got)
    return fetch


URL = "http://127.0.0.1:9999"

# ------------------------------------------------------------------ detection

check(K.detect(URL, fetch_json=fake_fetch_json({f"{URL}/props": PROPS_LC}))
      == "llama_cpp", "detect: /props {model_path,n_ctx} -> llama_cpp")
check(K.detect(URL, fetch_json=fake_fetch_json(
          {f"{URL}/props": {}, f"{URL}/api/version": VER_OLL}))
      == "ollama", "detect: /api/version -> ollama")
check(K.detect(URL, fetch_json=fake_fetch_json(
          {f"{URL}/props": {}, f"{URL}/api/version": {},
           f"{URL}/v1/api/props": PROPS_TABBY}))
      == "tabby", "detect: /v1/api/props -> tabby")
check(K.detect(URL, fetch_json=fake_fetch_json(
          {f"{URL}/props": {}, f"{URL}/api/version": {}, f"{URL}/v1/api/props": {}}))
      == "generic", "detect: nothing answers -> generic")
check(K.detect(URL, fetch_json=fake_fetch_json({})) == "generic",
      "detect: connection refused (None) -> generic")

# precedence: llama.cpp's probe wins even if ollama's endpoint also answers
check(K.detect(URL, fake_fetch_json({f"{URL}/props": PROPS_LC,
                                     f"{URL}/api/version": VER_OLL}))
      == "llama_cpp", "detect precedence: llama_cpp beats ollama probe order")

# family mapping (fingerprint reuse, no extra probes)
check(K.kind_for_family("tabby") == "tabby" and
      K.kind_for_family("vllm") == "vllm" and
      K.kind_for_family("llama.cpp") == "llama_cpp" and
      K.kind_for_family("openai") == "generic" and
      K.kind_for_family("koboldcpp") == "generic",
      "kind_for_family: families map; openai/koboldcpp -> generic")

# ------------------------------------------------------------------- status

st = K.status("llama_cpp", URL, fetch_json=fake_fetch_json({f"{URL}/props": PROPS_LC}))
check(st["loaded"] and st["models"] == ["qwen3-8b-q4_k_m.gguf"] and
      st["ctx_tokens"] == 32768, "status llama_cpp: loaded + basename + n_ctx")
check(K.status("llama_cpp", URL, fetch_json=fake_fetch_json({}))["loaded"] is False,
      "status llama_cpp: no props -> not loaded, no raise")

st = K.status("tabby", URL, fetch_json=fake_fetch_json({f"{URL}/v1/api/props": PROPS_TABBY}))
check(st["loaded"] and st["models"] == ["Mistral-Nemo-Instruct-2407-exl3"] and
      st["ctx_tokens"] == 65536, "status tabby: model_info + max_seq_len")

st = K.status("ollama", URL, fetch_json=fake_fetch_json(
    {f"{URL}/api/version": VER_OLL, f"{URL}/api/ps": PS_OLL}))
check(st["loaded"] and st["models"] == ["llama3.2:3b"] and st["ctx_tokens"] == 8192
      and st["detail"]["version"] == "0.4.1", "status ollama: /api/ps + version")
check(K.status("ollama", URL, fetch_json=fake_fetch_json(
    {f"{URL}/api/version": VER_OLL}))["loaded"] is False,
    "status ollama: version but no slots -> not loaded")

st = K.status("generic", URL, fetch_json=fake_fetch_json({f"{URL}/v1/models": MODELS_VLLM}))
check(st["loaded"] and st["models"] == ["Qwen/Qwen3-8B"], "status generic: /v1/models")

st = K.status("vllm", URL, fetch_json=fake_fetch_json({f"{URL}/version": {"version": "0.30.0"}}))
check(st["loaded"] and st["detail"]["version"] == "0.30.0",
      "status vllm: mirror profile answers from /version")

# ------------------------------------------------------------------- usage

check(K.normalize_usage("ollama", {"prompt_eval_count": 120, "eval_count": 45})
      == {"prompt_eval_count": 120, "eval_count": 45,
          "completion_tokens": 45, "prompt_tokens": 120},
      "normalize_usage: ollama native counters mapped to OpenAI names")
check(K.normalize_usage("llama_cpp",
                        {"tokens_evaluated": 120, "tokens_predicted": 45})
      .get("completion_tokens") == 45, "normalize_usage: llama.cpp native names")
check(K.normalize_usage("generic", {"prompt_tokens": 1, "completion_tokens": 2})
      == {"prompt_tokens": 1, "completion_tokens": 2},
      "normalize_usage: already-OpenAI passes through unchanged")
check(K.normalize_usage("generic", None) is None, "normalize_usage: None stays None")

# ------------------------------------------------------------------- errors

e = K.normalize_error("tabby", 403, "Not authenticated")
check(e["is_auth"] and "auth" in e["meaning"], "normalize_error: tabby 403 -> auth")
e = K.normalize_error("ollama", 200, '{"error":"model \'llama3\' not found"}')
check(e["is_overloaded"], "normalize_error: ollama in-band 200 error flagged")
e = K.normalize_error("llama_cpp", 400, "context overflow")
check("overflow" in e["meaning"], "normalize_error: llama.cpp 400 -> n_ctx overflow")
check(K.normalize_error("generic", 502, "")["meaning"] == "error",
      "normalize_error: generic passthrough")

print()
if FAILS:
    print(f"{len(FAILS)} FAILURES: " + "; ".join(FAILS))
    sys.exit(1)
print("all kinds tests pass")
