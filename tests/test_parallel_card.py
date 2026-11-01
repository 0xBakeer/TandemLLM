"""The weight cards' speed rows (SPD-245): tools/parallel_card.py's row and the registry's rules for the
`parallel` section of a model.yaml (server/setup/registry.py `_parallel_errors`, `_PAR_ROW`).

The test matrix:

  1  the row is the raw file's: build_row takes every number from the run (decode medians, prefill
     medians, each level's medians and its proof), the YAML the tool writes loads back to the same
     row, and the registry takes it                     test_the_row_is_the_raw_file
  2  a level whose rounds did not decode N at once is refused ("refused: 7 decoded at once, not 8"),
     and so is a level with fewer valid rounds than the row's
                                                        test_a_level_without_proof_is_refused
  3  the raw file: missing, or a row that differs from the file's, is refused; a raw path outside the
     model's measurements/ is refused                   test_the_raw_file_is_required
  4  the weight set: an unknown profile, a community pack the model does not have and a second row
     for one weight set are refused; a community pack that is there is taken
                                                        test_the_row_names_a_weight_set
  5  the schema: levels 1/2/4/8 only (3 is an unknown key), level 1 required, a commit that is not
     hex refused                                        test_the_schema
  6  every shipped row loads, proves every level and is its raw file's row
                                                        test_the_shipped_rows
  7  SPD-249, the engine only: a prompt is the chat template's ids (thinking off) as text, read back as the same
     ids, with a tokenizer's leading BOS left to the engine; ids that differ stop the run
                                                        test_a_prompt_is_the_chat_templates_ids_as_text
  8  SPD-249: a request goes to /v1/completions with the prompt as text and `X-Tandem: raw`, no chat field; its
     streamed text events are timed and its usage read          test_a_request_goes_to_the_engine
  9  the tokenizer's directory from the env file's QSE_WEIGHTS_DIR          test_the_weights_dir_of_the_env_file
 10  against the server (fake engine): `/v1/tokenize` gives the chat prompt's ids, their text reads back as them,
     and `/v1/completions` streams an answer whose prompt_tokens is their count       test_the_tool_against_the_server
 11  SPD-249: a family's draft corrector (/health weights.corrector) goes to the raw file's /health records and
     setup, and the row says whether it loaded (`setup.corrector_loaded`); a row without one is as before
                                                        test_the_corrector_is_recorded
 12  SPD-249, `--fixed-prompts`: request i of every round carries the nonce 0000-<i> and the seed 1000 + i, a
     level's N requests are requests 0 .. N-1, a prompt is rendered once, a level waits for an idle server once
     and runs its rounds back to back; without it every request has its own nonce and seed and every round waits
                                                        test_fixed_prompts_repeat_back_to_back
 13  SPD-249, `--levels-thinking template`: a prompt is the chat template's own default rendered with the model's
     tokenizer, never the server's chat rendering, checked against the engine's ids of its text (a leading BOS
     the engine's); ids that differ stop the run; `template_thinks` says whether that default thinks
                                                        test_levels_thinking_template

Run: python tests/test_parallel_card.py
"""

from __future__ import annotations

import copy
import http.server
import json
import os
import sys
import tempfile
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml  # noqa: E402

os.environ.setdefault("QSE_COMMUNITY_DIR", tempfile.mkdtemp(prefix="parcard-community-"))
os.environ.setdefault("QSE_COMMUNITY_URL", "off")

from server.conductor.schema import validate  # noqa: E402
from server.setup import registry  # noqa: E402
from tools import parallel_card as pc  # noqa: E402

MID = "qwen3.8-27b"


def _level(n: int, at_once: int | None = None, rounds: int = 3) -> dict:
    rs = [{"wall_tok_s": 30.0 * n + i, "decode_tok_s": 32.0 * n + i, "proof": {"max_decoding_at_once": n if at_once is None else at_once},
           "streams": [{"ttft_s": 0.2 + 0.1 * i}] * n} for i in range(rounds)]
    return {"rounds": rs, "valid": at_once in (None, n), "wall_tok_s": 30.0 * n + 1, "decode_tok_s": 32.0 * n + 1,
            "ttft_s": 0.3, "at_once": n if at_once is None else at_once}


def _raw(profile: str = "exl3-4.0", **over) -> dict:
    raw = {"profile": profile, "build": "0.14.2", "commit": "345ab0fa", "date": "2026-10-26",
           "raw_path": f"models/{MID}/measurements/parallel-test-{profile.replace('community/', '')}.json",
           "args": {"rounds": 3},
           "setup": {"env": "live8-cold", "lanes": 8, "window": 262144, "max_tokens": 512, "suffix_store": False,
                     "response_cache": False,
                     "sampling": {"temperature": 1.0, "top_k": 20, "top_p": 0.95, "min_p": 0.0, "seed": "one per request"},
                     "prompts": {"set": "card-v1", "sha256": "0123456789abcdef0123", "prompt_tokens": {"code": 246, "prose": 190}}},
           "decode": {"median": {"code": 71.23, "prose": 36.44}},
           "prefill": {"depths": {"2048": {"ttft_s": 1.612, "tok_s": 1268.4, "prompt_tokens": 2045},
                                  "8192": {"ttft_s": 5.871, "tok_s": 1395.2, "prompt_tokens": 8190}}},
           "levels": {str(n): _level(n) for n in (1, 2, 4, 8)}}
    raw.update(over)
    return raw


def _root(rows_raw: list[dict], community: tuple = ()) -> str:
    root = tempfile.mkdtemp(prefix="parcard-")
    os.makedirs(os.path.join(root, "models", MID, "measurements"))
    os.makedirs(os.path.join(root, "models", MID, "community"))
    for c in community:
        open(os.path.join(root, "models", MID, "community", c + ".json"), "w").write("{}")
    for raw in rows_raw:
        raw = dict(raw, row=pc.build_row(raw))
        with open(os.path.join(root, raw["raw_path"]), "w") as fh:
            json.dump(raw, fh)
    return root


def _model(rows: list[dict]) -> dict:
    m = registry.get(MID)
    m = copy.deepcopy({k: m[k] for k in ("id", "profiles")})
    m["parallel"] = {"method": "tools/parallel_card.py", "rows": rows}
    return m


def test_the_row_is_the_raw_file():
    raw = _raw()
    row = pc.build_row(raw)
    assert row["decode"] == {"code": 71.2, "prose": 36.4}
    assert row["prefill"]["2048"] == {"ttft_s": 1.61, "tok_s": 1268.0, "prompt_tokens": 2045}
    assert row["levels"]["8"] == {"wall_tok_s": 241.0, "decode_tok_s": 257.0, "ttft_s": 0.3, "at_once": 8, "rounds": 3}
    assert row["setup"]["prompts"]["sha256"] == "0123456789abcdef" and row["setup"]["thinking"] is False
    back = yaml.safe_load("rows:\n" + pc.row_yaml(row))["rows"][0]
    assert back == row, (back, row)
    assert not validate(row, registry._PAR_ROW), validate(row, registry._PAR_ROW)
    assert registry._parallel_errors(_model([back]), _root([raw])) == []
    # a workload without a decode window (one event) is left out of the row
    raw2 = _raw(decode={"median": {"code": None, "prose": 126.29}})
    assert pc.build_row(raw2)["decode"] == {"prose": 126.3}
    assert not validate(pc.build_row(raw2), registry._PAR_ROW)


def test_a_level_without_proof_is_refused():
    raw = _raw(levels={"1": _level(1), "2": _level(2), "4": _level(4), "8": _level(8, at_once=7)})
    errs = registry._parallel_errors(_model([pc.build_row(raw)]), _root([raw]))
    assert any("levels.8: refused: 7 decoded at once, not 8" in e for e in errs), errs
    raw = _raw(levels={"1": _level(1), "8": _level(8, rounds=2)})
    errs = registry._parallel_errors(_model([pc.build_row(raw)]), _root([raw]))
    assert any("levels.8: 2 valid rounds of 3" in e for e in errs), errs


def test_the_raw_file_is_required():
    raw = _raw()
    row = pc.build_row(raw)
    errs = registry._parallel_errors(_model([row]), _root([]))
    assert any("no raw file" in e for e in errs), errs
    changed = copy.deepcopy(row)
    changed["levels"]["8"]["wall_tok_s"] = 999.0
    errs = registry._parallel_errors(_model([changed]), _root([raw]))
    assert any("not the raw file's row" in e for e in errs), errs
    outside = dict(row, raw="models/flint/measurements/x.json")
    errs = registry._parallel_errors(_model([outside]), _root([raw]))
    assert any("not under models/qwen3.8-27b/measurements/" in e for e in errs), errs


def test_the_row_names_a_weight_set():
    tc = "community/thinkingcap-27b-kearuga-exl3-4.00"
    a, b, c = _raw("exl3-9.9"), _raw(tc), _raw("community/nobody")
    root = _root([a, b, c], community=("thinkingcap-27b-kearuga-exl3-4.00",))
    errs = registry._parallel_errors(_model([pc.build_row(x) for x in (a, b, c)]), root)
    assert any("no profile 'exl3-9.9'" in e for e in errs), errs
    assert any("no community pack 'community/nobody'" in e for e in errs), errs
    assert not any(tc in e for e in errs), errs
    d = _raw()
    errs = registry._parallel_errors(_model([pc.build_row(d), pc.build_row(d)]), _root([d]))
    assert any("a second row for 'exl3-4.0'" in e for e in errs), errs


def test_the_schema():
    row = pc.build_row(_raw())
    bad = copy.deepcopy(row)
    bad["levels"]["3"] = bad["levels"]["2"]
    assert any("unknown key '3'" in e for e in validate(bad, registry._PAR_ROW))
    bad = copy.deepcopy(row)
    del bad["levels"]["1"]
    assert any("missing '1'" in e for e in validate(bad, registry._PAR_ROW))
    assert any("commit" in e for e in validate(dict(row, commit="HEAD"), registry._PAR_ROW))


def test_the_corrector_is_recorded():
    st = {"loaded": True, "why": None, "file": "flashnext-3.05.corrector.safetensors", "sha256": "ab" * 32, "kc": 8,
          "hh": 256, "D": 4, "hot_rows": 65536, "stat": "q", "tau": 0.85,
          "sampling": {"temperature": 1.0, "top_k": 20, "top_p": 0.95}, "heads": {"65536": True, "32768": True}}
    h = {"version": "0.15.1", "weights": {"name": "flashnext-3.05", "corrector": st}, "parallel": {}, "cache": {}}
    assert pc.health_record(h)["corrector"] == st
    assert pc.health_record({"weights": {"name": "exl3-4.0"}})["corrector"] is None
    for loaded in (True, False):
        raw = _raw()
        raw["setup"]["corrector"] = dict(st, loaded=loaded, why=None if loaded else "no file")
        row = pc.build_row(raw)
        assert row["setup"]["corrector_loaded"] is loaded
        assert not validate(row, registry._PAR_ROW), validate(row, registry._PAR_ROW)
        assert yaml.safe_load("rows:\n" + pc.row_yaml(row))["rows"][0] == row
    raw = _raw()
    raw["setup"]["corrector"] = None
    assert "corrector_loaded" not in pc.build_row(raw)["setup"]
    assert any("corrector_loaded" in e for e in validate(dict(row, setup=dict(row["setup"], corrector_loaded="yes")),
                                                          registry._PAR_ROW))
    return "/health's corrector in the raw records; the row's setup.corrector true/false; none: no key"


def test_the_shipped_rows():
    models, broken = registry.load_all()
    assert not broken, broken
    for mid, m in models.items():
        for r in (m.get("parallel") or {}).get("rows") or []:
            with open(os.path.join(registry.ROOT, r["raw"])) as fh:
                raw = json.load(fh)
            assert raw["row"] == r and raw["valid"], (mid, r["profile"])
            for n, lv in r["levels"].items():
                assert lv["at_once"] == int(n) and raw["levels"][n]["valid"], (mid, r["profile"], n)
                assert all(x["proof"]["max_decoding_at_once"] == int(n) and x["proof"]["max_running"] <= int(n)
                           for x in raw["levels"][n]["rounds"]), (mid, r["profile"], n)


class _Tok:
    """The model's tokenizer as the tool uses it: id 1 is `<s>`, every other id a character."""

    def decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=True):
        assert not skip_special_tokens and not clean_up_tokenization_spaces
        return "".join("<s>" if i == 1 else chr(i) for i in ids)


def _tokenize_server(bos: bool):
    """`/v1/tokenize` with `ids`: the chat template's ids (thinking off, a BOS in the template when the tokenizer has
    one) and a text's ids as /v1/completions reads it (a BOS put in front of any text when `bos`)."""
    def enc(text):
        out, i = [], 0
        while i < len(text):
            if text.startswith("<s>", i):
                out.append(1)
                i += 3
            else:
                out.append(ord(text[i]))
                i += 1
        return out
    calls = []

    def post(url, body, timeout=60.0):
        calls.append(body)
        assert url.endswith("/v1/tokenize") and body.get("ids") is True, (url, body)
        if "messages" in body:
            assert body["chat_template_kwargs"] == {"enable_thinking": False}
            ids = enc(("<s>" if bos else "") + "".join(f"<{m['role']}>{m['content']}" for m in body["messages"])
                      + "<a>")
        else:
            ids = ([1] if bos else []) + enc(body["text"])
        return {"tokens": len(ids), "ids": ids}
    return post, calls


class _LossyTok(_Tok):
    """A tokenizer whose text does not read back as its ids (it drops the last character)."""

    def decode(self, ids, **kw):
        return super().decode(ids, **kw)[:-1]


def test_a_prompt_is_the_chat_templates_ids_as_text():
    msgs = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hello"}]
    real = pc._post_json
    try:
        for bos in (False, True):
            pc._post_json, calls = _tokenize_server(bos)
            cl = pc.Client("http://x", "TandemLLM", {}, {}, tok=_Tok())
            text = cl.render(msgs)
            assert text == "<system>be brief<user>hello<a>", text                  # no BOS in the text: the engine adds it
            assert cl.rendered == 1 and calls[-1] == {"text": text, "ids": True}, calls
            cl.render(msgs)
            assert sum(1 for c in calls if c.get("text") == "") == 1, "the leading ids asked once"
        pc._post_json, _ = _tokenize_server(False)
        cl = pc.Client("http://x", "TandemLLM", {}, {}, tok=_LossyTok())
        try:
            cl.render([{"role": "user", "content": "yes"}])
            raise AssertionError("ids that differ must stop the run")
        except pc.PromptMismatch as exc:
            assert "first difference" in str(exc), exc
        assert cl.rendered == 0
    finally:
        pc._post_json = real
    return "the chat ids as text, a leading BOS left to the engine; a mismatch stops"


def test_a_request_goes_to_the_engine():
    seen = {}
    events = [{"choices": [{"index": 0, "text": "Hel"}]}, {"choices": [{"index": 0, "text": "lo!"}]},
              {"choices": [{"index": 0, "text": "", "finish_reason": "length"}]},
              {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 3,
                                        "prompt_tokens_details": {"cached_tokens": 0}}}]

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            seen["path"] = self.path
            seen["raw"] = self.headers.get("X-Tandem")
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for ev in events:
                self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")

        def log_message(self, *a):
            pass
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        cl = pc.Client(f"http://127.0.0.1:{srv.server_address[1]}", "TandemLLM", {"temperature": 1.0, "top_k": 20},
                       {"X-Tandem": "raw"}, tok=_Tok())
        r = cl.stream("<user>hi<a>", 3, 5)
    finally:
        srv.shutdown()
    b = seen["body"]
    assert seen["path"] == "/v1/completions" == pc.ENDPOINT and seen["raw"] == "raw", seen
    assert b["prompt"] == "<user>hi<a>" and "messages" not in b and "chat_template_kwargs" not in b, b
    assert b["cache_prompt"] is False and b["seed"] == 5 and b["stream"] and b["top_k"] == 20, b
    assert r["completion_tokens"] == 3 and r["prompt_tokens"] == 7 and r["events"] == 2, r
    assert r["finish_reason"] == "length" and r["cached_tokens"] == 0 and r["ttft_s"] is not None, r
    return "/v1/completions, the prompt as text, X-Tandem raw; text events timed, usage read"


def test_the_tool_against_the_server():
    import subprocess
    import time
    import urllib.request
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ports import free_port
    from server.fake_engine import FakeTokenizer
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    launch = ("import sys, runpy; sys.modules['triton'] = None; sys.modules['triton.language'] = None;"
              "sys.argv = ['server/app.py'] + sys.argv[1:]; runpy.run_path('server/app.py', run_name='__main__')")
    port = free_port()
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": tempfile.mkdtemp(prefix="parcard-home-"),
           "PYTHONPATH": root, "CUDA_VISIBLE_DEVICES": ""}
    ledger = os.path.join(env["HOME"], "ledger.db")
    proc = subprocess.Popen([sys.executable, "-c", launch, "--fake-engine", "--port", str(port), "--fake-tps", "400",
                             "--usage-ledger", ledger], cwd=root, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    base = f"http://127.0.0.1:{port}"
    try:
        t0 = time.monotonic()
        while time.monotonic() - t0 < 120:
            try:
                urllib.request.urlopen(base + "/health", timeout=2).read()
                break
            except Exception:                                      # noqa: BLE001
                if proc.poll() is not None:
                    raise AssertionError(proc.stdout.read()[-2000:])
                time.sleep(0.3)

        class Tok(FakeTokenizer):
            def decode(self, seq, skip_special_tokens=True, clean_up_tokenization_spaces=False):
                return super().decode(seq, skip_special_tokens=skip_special_tokens)
        cl = pc.Client(base, "TandemLLM", {"temperature": 0.0}, {"X-Tandem": "raw"}, tok=Tok())
        msgs = cl.messages("Be brief.", "Nonce 1234. Write two lines about rivers.")
        want = cl._ids({"messages": msgs, "chat_template_kwargs": {"enable_thinking": False}})
        text = cl.render(msgs)
        assert cl.rendered == 1 and cl._ids({"text": text}) == want and len(want) > 10, (len(want), text[:200])
        r = cl.stream(text, 16, 7)
        assert r["prompt_tokens"] == len(want) and r["completion_tokens"] > 0, r
        assert r["finish_reason"] in ("length", "stop") and r["events"] > 0, r
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
    return f"{len(want)} prompt ids, read back as text and as the completion's prompt_tokens"


def test_the_weights_dir_of_the_env_file():
    env = {"QSE_FN_BPW": "3.05", "QSE_WEIGHTS_DIR": "${HOME}/flashnext-packs/${QSE_FN_BPW}"}
    assert pc.weights_dir(env) == os.path.join(os.environ["HOME"], "flashnext-packs", "3.05")
    assert pc.weights_dir({"QSE_WEIGHTS_DIR": "${NOPE_NOT_SET_X}/w"}) is None
    assert pc.weights_dir({}) is None
    with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as fh:
        fh.write("# a comment\nQSE_WEIGHTS_DIR='/w/x'\nQSE_PARALLEL=8\n")
    try:
        assert pc.envfile(fh.name) == {"QSE_WEIGHTS_DIR": "/w/x", "QSE_PARALLEL": "8"}
    finally:
        os.unlink(fh.name)


def test_fixed_prompts_repeat_back_to_back():
    import time
    import types
    pj = {"system": "S", "nonce_line": "Request nonce: {nonce}\n\n", "workloads": {"prose": "Write."}}
    sent, renders = [], []

    class Cl:
        base = "http://127.0.0.1:9"                      # nothing listens: /health answers nothing at once

        def messages(self, system, user):
            return [{"role": "system", "content": system}, {"role": "user", "content": user}]

        def render(self, messages, template=False):
            renders.append(messages[1]["content"])
            return messages[1]["content"]

        def stream(self, prompt, max_tokens, seed, t_send=None):
            sent.append((prompt, seed))
            t = time.monotonic()
            return {"completion_tokens": 4, "ttft_s": 0.01, "t_send": t, "t_first": t + 0.01, "t_last": t + 0.02,
                    "t_end": t + 0.03}

    for fixed in (True, False):
        sent.clear()
        for tag in ("r1", "r2"):
            pc.one_round(Cl(), pj, 4, tag, 8, fixed)
        by_round = [sorted(sent[:4], key=lambda x: x[1]), sorted(sent[4:], key=lambda x: x[1])]
        if fixed:
            want = [(f"Request nonce: 0000-0{i}\n\n\n\nWrite.", 1000 + i) for i in range(4)]
            assert by_round[0] == by_round[1] == want, by_round
            assert pc.fixed_user(pj, 7) == "Request nonce: 0000-07\n\n\n\nWrite.", pc.fixed_user(pj, 7)
        else:
            assert not set(by_round[0]) & set(by_round[1]), "fresh requests repeat nothing"
            assert len({seed for _, seed in sent}) == 8, sent
    real = (pc.one_round, pc.wait_idle)
    calls, waits = [], []
    try:
        pc.wait_idle = lambda base, limit_s=120.0: waits.append(base) or True
        pc.one_round = lambda cl, pj, n, tag, mt, fixed=False, template=False: calls.append((n, tag, fixed)) or {
            "valid": True, "wall_tok_s": 1.0, "decode_tok_s": 1.0, "streams": [{"ttft_s": 0.1, "completion_tokens": 4}],
            "proof": {"max_decoding_at_once": n, "max_running": n, "decoding_hist": {}}}
        for fixed, want_waits in ((True, 2), (False, 8)):
            calls.clear()
            waits.clear()
            a = types.SimpleNamespace(levels="1,2", warm_rounds=1, rounds=3, retries=0, max_tokens=8,
                                      fixed_prompts=fixed, levels_thinking="off")
            out = pc.run_levels(Cl(), pj, a, lambda *x: None)
            assert [c[0] for c in calls] == [1] * 4 + [2] * 4 and all(c[2] is fixed for c in calls), calls
            assert len(waits) == want_waits and out["2"]["valid"], (fixed, len(waits))
    finally:
        pc.one_round, pc.wait_idle = real
    return "fixed: the same nonce and seed a request every round, one wait a level; fresh: a wait a round"


class _TplTok(_Tok):
    """A tokenizer with a chat template that thinks by default (`</t>` closes the thinking when it is off)."""

    def __init__(self, thinks: bool = True, lossy: bool = False):
        self.thinks, self.lossy = thinks, lossy

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
        assert not tokenize and add_generation_prompt
        off = kw.get("enable_thinking", True) is False and self.thinks
        return "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<a>" + ("</t>" if off else "")

    def __call__(self, text, add_special_tokens=True):
        assert not add_special_tokens
        ids = [ord(c) for c in text]
        return {"input_ids": ids[:-1] if self.lossy else ids}


def test_levels_thinking_template():
    msgs = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hello"}]
    real = pc._post_json
    try:
        for bos in (False, True):
            pc._post_json, calls = _tokenize_server(bos)
            cl = pc.Client("http://x", "TandemLLM", {}, {}, tok=_TplTok())
            text = cl.render(msgs, template=True)
            assert text == "<system>be brief<user>hello<a>", text
            assert not any("messages" in c for c in calls), "the server's chat rendering is not asked"
            assert cl.rendered == 1 and calls[-1] == {"text": text, "ids": True}, calls
        pc._post_json, _ = _tokenize_server(False)
        cl = pc.Client("http://x", "TandemLLM", {}, {}, tok=_TplTok(lossy=True))
        try:
            cl.render(msgs, template=True)
            raise AssertionError("ids that differ must stop the run")
        except pc.PromptMismatch:
            pass
    finally:
        pc._post_json = real
    assert pc.template_thinks(_TplTok(), msgs) and not pc.template_thinks(_TplTok(thinks=False), msgs)
    return "the template's default rendered by the tokenizer, checked on the engine; thinks or not"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:68s} ok")
            passed += 1
    print(f"{passed} passed")
