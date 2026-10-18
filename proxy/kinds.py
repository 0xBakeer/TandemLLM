"""kinds — first-class backend kinds beyond vllm/sglang (T10 prototype).

config.KINDS already names kinds and engines.py knows vllm/sglang deeply; this
module adds llama.cpp, tabby (TabbyAPI) and ollama as named, detectable, status-
queried kinds, plus the generic fallback. Pure and fetch-injectable, like
autodetect: `fetch(url)` -> (status, body-text) | None, `fetch_json(url)` -> obj
| None. Nothing here relays or configures — detection refines what autodetect
saw; the relay stays "any URL is a backend".

Detection order (docs/BACKEND-KINDS.md): explicit config > fingerprint > generic.
vllm/sglang profiles are mirrors-by-reference: engines.py already owns their
stats, so their detect/status delegate to autodetect's existing answer and the
poller; carrying them here keeps the registry complete for callers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

PROBE_TIMEOUT = 0.8          # same bound as autodetect


# ------------------------------------------------------------------ profiles


@dataclass
class KindProfile:
    """One engine kind's face. Flags mirror docs/BACKEND-KINDS.md's table."""
    name: str
    openai_chat: bool = True
    native_completion: bool = False        # a non-OpenAI body /completion-style API
    tokenizer_stats: bool = False          # a /tokenize-style endpoint
    has_metrics: bool = False              # Prometheus /metrics worth polling
    engine_status: str = ""                # how "what is loaded" is answered
    #: autodetect family names that ARE this kind (fingerprint reuse)
    families: tuple = ()
    probes: tuple = field(default=())      # ((path, kind-name), ...) detect order


LLAMA_CPP = KindProfile(
    "llama_cpp", native_completion=True, tokenizer_stats=True, has_metrics=True,
    engine_status="/props {model_path, n_ctx}; kv_usage via /metrics (to-verify)",
    families=("llama.cpp",),
    probes=(("/props", "llama_cpp"),),
)

TABBY = KindProfile(
    "tabby", has_metrics=True,
    engine_status="/v1/api/props {model_info}; kv usage not public (to-verify)",
    families=("tabby",),
    probes=(("/v1/api/props", "tabby"),),
)

OLLAMA = KindProfile(
    "ollama", native_completion=True,
    engine_status="/api/ps running slots + /api/version; no metrics",
    families=("ollama",),
    probes=(("/api/version", "ollama"),),
)

VLLM = KindProfile("vllm", has_metrics=True,
                   engine_status="engines.py TABLES (owner), /version",
                   families=("vllm",))
SGLANG = KindProfile("sglang", has_metrics=True,
                     engine_status="engines.py TABLES (owner), /server_info",
                     families=("sglang",))
GENERIC = KindProfile("generic", engine_status="none — plain relay")

REGISTRY = {p.name: p for p in (VLLM, SGLANG, LLAMA_CPP, TABBY, OLLAMA, GENERIC)}

FAMILY_TO_KIND = {fam: p.name for p in REGISTRY.values() for fam in p.families}
FAMILY_TO_KIND["exl3"] = "generic"     # exl3/gguf-http servers run as generic
FAMILY_TO_KIND["openai"] = "generic"   # refined below when the probes say otherwise


def profile(kind: str) -> KindProfile:
    return REGISTRY.get(kind, GENERIC)


# ---------------------------------------------------------------- detection


def detect(url: str, fetch_json=None) -> str:
    """Kind for a URL autodetect could not name precisely (family 'openai'/unknown).
    Probe order is evidence-strength: llama.cpp's /props is unambiguous, ollama's
    /api/version is its native root, tabby's props endpoint is namespaced under
    /v1/api — an OpenAI-only shim answers none of them, so generic."""
    fetch_json = fetch_json or _fetch_json
    for path, kind in (LLAMA_CPP.probes + OLLAMA.probes + TABBY.probes):
        got = fetch_json(f"{url}{path}")
        if got is None:
            continue
        if kind == "llama_cpp" and isinstance(got, dict) and \
                ("model_path" in got or "model_type" in got):
            return kind
        if kind == "ollama" and isinstance(got, dict) and "version" in got:
            return kind
        if kind == "tabby" and isinstance(got, dict) and \
                ("model_info" in got or "completion" in got or "model" in got):
            return kind
    return "generic"


def kind_for_family(family: str | None) -> str:
    """A fingerprint family already known -> its kind; 'openai' -> generic unless
    the caller refines with detect()."""
    return FAMILY_TO_KIND.get(family or "", "generic")


def _fetch_json(url: str):
    from proxy import autodetect                      # reuse the real seam, not a copy
    got = autodetect._get(url, PROBE_TIMEOUT)
    if not got or got[0] >= 400:
        return None
    try:
        return json.loads(got[1])
    except ValueError:
        return None


# ---------------------------------------------------------------- status


def status(kind: str, url: str, fetch=None, fetch_json=None) -> dict:
    """'What is loaded / context or KV usage' for one kind. Never raises: a probe
    failure reads as not-loaded, so a mid-restart engine answers honestly."""
    fetch_json = fetch_json or _fetch_json
    out = {"kind": kind, "loaded": False, "models": [], "ctx_tokens": None,
           "kv_usage": None, "detail": {}}
    if kind == "ollama":
        ver = fetch_json(f"{url}/api/version")
        ps = fetch_json(f"{url}/api/ps")
        psd = ps if isinstance(ps, dict) else {}
        ver = ver if isinstance(ver, dict) else {}
        models = psd.get("models")
        if isinstance(models, list) and models:
            out["loaded"] = True
            out["models"] = [str(m.get("name") or m.get("model") or "?")
                             for m in models if isinstance(m, dict)]
            # size_vram/context_window field names: to-verify across ollama versions
            first = next((m for m in models if isinstance(m, dict)), {})
            out["detail"] = {"version": ver.get("version"),
                             "size_vram": first.get("size_vram"),
                             "context_window": first.get("context_window")}
            cw = first.get("context_window")
            if isinstance(cw, int) and cw > 0:
                out["ctx_tokens"] = cw
        elif ver:
            out["loaded"] = False
            out["detail"] = {"version": ver.get("version")}
        return out
    if kind == "llama_cpp":
        props = fetch_json(f"{url}/props")
        if isinstance(props, dict) and (props.get("model_path") or props.get("model_type")):
            out["loaded"] = True
            mp = str(props.get("model_path") or props.get("model_type"))
            out["models"] = [mp.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]]
            n = props.get("n_ctx")
            out["ctx_tokens"] = int(n) if isinstance(n, (int, float)) else None
            out["detail"] = {"model_path": mp, "n_batch": props.get("n_batch")}
        return out
    if kind == "tabby":
        props = fetch_json(f"{url}/v1/api/props")
        if isinstance(props, dict):
            mi = props.get("model_info") if isinstance(props.get("model_info"), dict) else props
            model = mi.get("model") or mi.get("id")
            if model:
                out["loaded"] = True
                out["models"] = [str(model)]
                msl = mi.get("max_seq_len") or mi.get("max_length")
                out["ctx_tokens"] = int(msl) if isinstance(msl, (int, float)) else None
                out["detail"] = {"org": mi.get("org"), "max_seq_len": msl}
        return out
    if kind == "generic":
        models = fetch_json(f"{url}/v1/models")
        if isinstance(models, dict) and isinstance(models.get("data"), list):
            ids = [m.get("id") for m in models["data"] if isinstance(m, dict) and m.get("id")]
            if ids:
                out["loaded"] = True
                out["models"] = [str(ids[0])]
                out["detail"] = {"count": len(ids)}
        return out
    # vllm / sglang: engines.py owns this; a summary from the poller's info is
    # enough here (kept fetch-shaped so tests stay table-driven)
    path = "/version" if kind == "vllm" else "/get_server_info"
    info = fetch_json(f"{url}{path}")
    if isinstance(info, dict):
        out["loaded"] = True
        if info.get("version"):
            out["detail"] = {"version": str(info["version"])}
    return out


# ---------------------------------------------------------------- usage + errors


def normalize_usage(kind: str, usage: dict | None) -> dict | None:
    """OpenAI-shaped usage from whatever the kind actually returned. Pass-through
    when it already is one (v1 endpoints of every kind); native-API counters are
    mapped. Unknown fields are kept, never dropped — the caller decides."""
    if not isinstance(usage, dict):
        return None
    if "completion_tokens" in usage or "eval_count" not in usage and \
            "tokens_predicted" not in usage:
        return dict(usage)
    out = dict(usage)
    if "completion_tokens" not in usage:
        out["completion_tokens"] = usage.get("eval_count",
                                             usage.get("tokens_predicted"))
    if "prompt_tokens" not in usage:
        out["prompt_tokens"] = usage.get("prompt_eval_count",
                                         usage.get("tokens_evaluated"))
    return out


def normalize_error(kind: str, code: int, body: str) -> dict:
    """What a status/body from this kind MEANS. A read-only lens — the relay is
    untouched in this slice. Returns {meaning, is_auth, is_overloaded}."""
    meaning = "error"
    is_auth = is_overloaded = False
    text = (body or "")[:400].strip()
    if kind == "tabby" and code == 403:
        meaning, is_auth = "auth (TabbyAPI key required)", True
    elif kind == "llama_cpp" and code == 404 and "chat/completions" not in text \
            and "completion" in text.lower():
        meaning = "no OpenAI endpoint on this build (use /completion)"
    elif kind == "llama_cpp" and code == 400:
        meaning = "likely context overflow (n_ctx), not a malformed request"
    elif kind == "ollama" and code == 404:
        meaning = "model name missing its tag? (llama3 vs llama3:latest)"
    elif kind == "tabby" and code == 503:
        meaning, is_overloaded = "model still loading", True
    elif kind == "ollama" and code == 200 and '"error"' in text:
        meaning = "in-band error from a native ollama API (200 + error body)"
        is_overloaded = True
    return {"meaning": meaning, "is_auth": is_auth, "is_overloaded": is_overloaded}
