"""Compare a reference greedy run (tools/nv_ref_greedy.py JSON) with a vLLM server's greedy output for the same prompt ids.

    python3 tools/nv_compare_vllm.py .nv_ref.json http://127.0.0.1:8000
"""
import json, sys, urllib.request

ref = json.load(open(sys.argv[1]))
url = sys.argv[2].rstrip("/") + "/v1/completions"
body = {"model": "Qwen/Qwen3.8-Flash-Next", "prompt": ref["prompt_ids"], "max_tokens": len(ref["ids"]), "temperature": 0,
        "logprobs": 1, "return_tokens_as_token_ids": True}
r = json.load(urllib.request.urlopen(urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))
toks = [int(t.split(":")[1]) for t in r["choices"][0]["logprobs"]["tokens"]]
ref_ids = ref["ids"]
n = min(len(toks), len(ref_ids))
first = next((i for i in range(n) if toks[i] != ref_ids[i]), None)
print(f"reference {len(ref_ids)} tokens, vLLM {len(toks)} tokens")
print("identical prefix:", n if first is None else first, "tokens", "(all)" if first is None and len(toks) == len(ref_ids) else "")
if first is not None:
    print(f"first difference at {first}: reference {ref_ids[first]} vs vLLM {toks[first]}; reference top-2 margin there {ref['top2_margin'][first]}")
print("vLLM text:", repr(r["choices"][0]["text"][:300]))
