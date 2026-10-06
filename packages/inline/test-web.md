# Testing a web app or game
1. Serve it, open it in the browser tool; zero console errors first.
2. State hook: window.__state() returns {"tandem_state":1, ...} with what matters (player x/y, lives, score, level, route); window.__log keeps the last 200 lines.
3. Real input only: press_key with a hold time, clicks at screenshot coordinates. Wait 300 ms or more, then read window.__state() with evaluate_script. Never set keys, flags or positions in a script.
4. A scenario = start state, inputs, expected state, read state; walk the whole flow.
5. Screenshots at 1280 and 390 px: start, middle, end.
6. After each scenario, ask Laya from the page (evaluate script): `return await (await fetch("{tandem_url}/v1/laya/ask", {method: "POST", body: JSON.stringify({question: "Did the player move right and stay alive?", text: JSON.stringify(window.__state())})})).json()`. A "no" is a bug.
7. A surprise (lives gone in 2 s, a blank canvas) is a bug: name the line before the next edit.
8. A game or a UI flow: let Laya play it, 3 episodes or more (laya-playtest: `{tandem_scratch}/laya_drive.py browser`); every death it could not avoid and every stuck spot is a bug.
Report: console count, scenarios, states, Laya's verdicts.