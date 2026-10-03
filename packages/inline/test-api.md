# Testing a backend or API
1. Start the server in the background (`cmd > ./.tandem/run.log 2>&1 &`), wait for the port, read the log tail. Use the absolute project path.
2. Add GET /__state: {"tandem_state":1, "uptime":…, "routes":[…], "db":{"ok":true,"rows":…}, "last_error":…}. Add /healthz if missing.
3. Call every route with curl: a valid request, an invalid body, a missing auth, a not-found id. Read the status code and the body each time. A 500 is a bug; a 200 on bad input is a bug.
4. State after writes: create, then read back, then /__state. Numbers must agree.
5. Stop the server you started at the end; say that you did.
6. Let Laya judge every response: `(echo "expected: 404 with an error body"; curl -s -w '\nHTTP %{http_code}' localhost:8080/items/999) | python3 .tandem/laya.py "Is this response what the request should get?" right,wrong`.
Report: the routes called with their codes, Laya's verdicts, the read-back checks, the log tail.