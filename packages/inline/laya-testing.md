# Laya: the test judge you can call

TandemLLM serves Laya, a fast decision model (about 30 ms). It reads a short text (about 1,100 characters, head and tail) and answers a typed question with a probability. It writes nothing. Call it from your scripts through the provider:

```
curl -sS --max-time 60 "${TANDEM_URL:-{tandem_url}}/v1/laya/ask" -H 'Content-Type: application/json' \
  -d '{"question":"Does this output show a failure?","choices":["failure","clean"],"text":"<log tail>"}'
# {"label":"failure","p":0.93,"sure":true,...}
```

Also `scale` (levels, lowest first), or no choices for a yes-no question; `texts: [...]` judges up to 16. No key. `curl -s .../v1/laya` shows the contract.

You must use it:
- After every run that produced evidence (a log, an output, the console, a state dump), your test script asks Laya and prints the verdict.
- Before "done", run the check script and quote Laya's verdicts.
- failure, no or mismatch is a bug to explain before the next edit; `"sure": false` means read the evidence yourself.
- A call that fails (connection refused, a TLS or certificate error, a timeout, a 5xx) is this machine's network, not your code: write "Laya unavailable" once and go on with your own evidence. At most one retry; never debug the transport (no openssl, no CA hunting, no -k).
- An interactive build (a game, a UI flow, a CLI): Laya also plays it, step by step, and reports every death and stuck spot (laya-playtest).