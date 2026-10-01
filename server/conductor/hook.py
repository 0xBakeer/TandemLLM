"""The conductor as `server/app.py` sees it: one object in `STATE["conductor"]`.

app.py calls it in five places and nowhere else: `/v1/models` (`models()`), the start of a
completion (`plan()`, before the queue, where an unknown profile is the client's 400), the prompt
under the lock (`Plan.apply_prompt`), the response headers
(`Plan.headers`) and the end of the request (`finish()`). A raw request gets a plan with
`active = False`, whose every method returns its input unchanged, so the raw engine's path is the
release's path plus two response headers.

The frame of the session monitor (SRV-109, `server/conductor/monitor/`) runs under the request's
profile: `monitor_mode()` lowers its mode to the profile's, `monitor_policy()` gives it the
`monitor` document (the policy, versioned and edited like the others) with the profile's task
modes, tool-gate packs, caps and thresholds on top.

One journal for both (SRV-115): the router's rows and the monitor's events go to the same
`decisions.Journal`, which the server opens on the usage ledger's file and hands in here.
"""

from __future__ import annotations

import os
import threading
import time

from server.conductor import schema
from server.conductor.classify import Decider, Qwen, decider_for
from server.conductor.decisions import Journal
from server.conductor.packages import Library
from server.conductor.router import COMPOSITE, Plan, Router
from server.conductor.store import DEFAULT_ROOT, Store

HEALTH_TTL_S = 10.0


class Conductor:
    def __init__(self, root: str | None = None, *, state: dict | None = None,
                 journal: "bool | Journal" = True, readonly: bool = False, factory=None,
                 monitor_seed: dict | None = None):
        """`journal`: the server's shared journal, or True for one of its own beside the config
        (a standalone conductor: tests, tools), or False for none on disk."""
        self.state = state if state is not None else {}
        self.root = os.path.realpath(os.path.expanduser(root or DEFAULT_ROOT))
        self.store = Store(self.root, readonly=readonly, roles_up=self.roles_up,
                           monitor_seed=monitor_seed)
        self.library = Library()
        _, settings = self.store.get("settings")
        if isinstance(journal, Journal):
            self.journal = journal
        else:
            self.journal = Journal(os.path.join(self.root, "decisions.sqlite3") if journal
                                   else None,
                                   retention_days=settings["journal"]["retention_days"])
        self.router = Router(self.store, self.library, self.journal,
                             raw_model=lambda: str(self.state.get("model") or
                                                   "qwen38-spark-engine"),
                             self_url=self.self_url,
                             server_effort=lambda: self.state.get("reasoning_effort"),
                             global_mode=lambda: self.state.get("conductor_mode"),
                             factory=factory)
        self._health: dict = {}
        self._health_at = 0.0
        self._policies: dict = {}
        self._hlock = threading.Lock()

    # ------------------------------------------------------------------ app.py's calls
    def self_url(self) -> str | None:
        port = self.state.get("port")
        return f"http://127.0.0.1:{port}" if port else None

    def models(self, raw_model: str, created: int) -> list[dict]:
        """`/v1/models`: the composite first, then the raw engine unchanged."""
        _, settings = self.store.get("settings")
        return [{"id": COMPOSITE, "object": "model", "created": created, "owned_by": "tandemllm",
                 "profile": settings["default_profile"],
                 "description": "Qwen3.8-27B with the conductor: per-turn task routing, effort "
                                "and hidden skills, under a profile"},
                {"id": raw_model, "object": "model", "created": created, "owned_by": "local"}]

    def plan(self, body: dict, headers, *, client_kind: str, request_id: str,
             chat: bool) -> Plan:
        return self.router.plan(body, headers, client_kind=client_kind, request_id=request_id,
                                chat=chat)

    def finish(self, plan: Plan | None, rec) -> None:
        if plan is None:
            return
        rec.model_served = COMPOSITE if plan.active else str(self.state.get("model") or "")
        rec.profile, rec.profile_version = plan.res.profile, plan.res.version
        rec.conductor_mode = plan.res.mode
        if plan.active:
            try:
                plan.finish(rec, self.journal)
            except Exception as exc:                               # noqa: BLE001
                print(f"[conductor] journal row failed: {type(exc).__name__}: {exc}", flush=True)

    # ------------------------------------------------------------------ roles
    def role_client(self, role: str, doc: dict):
        r = doc.get(role) or {}
        endpoint = r.get("endpoint") or ""
        if endpoint == "self":
            endpoint = self.self_url() or ""
        if not endpoint:
            return None
        if r.get("contract") == "jev":
            return decider_for(endpoint, r.get("model") or "auto", int(r.get("timeout_ms") or 400))
        if r.get("contract") == "openai-chat":
            return Qwen(endpoint, r.get("model") or str(self.state.get("model") or ""),
                        int(r.get("timeout_ms") or 8000))
        return None

    def roles_health(self, force: bool = False) -> dict:
        """role -> {up, ms, mode, error|detail}; checked at most every 10 s, in parallel."""
        now = time.monotonic()
        if not force and self._health and now - self._health_at < HEALTH_TTL_S:
            return self._health
        with self._hlock:
            _, doc = self.store.get("roles")
            out, threads = {}, []

            def one(name, r):
                if r.get("mode", "off") == "off":
                    out[name] = {"up": False, "mode": "off", "ms": None, "error": "off"}
                    return
                c = self.role_client(name, doc)
                if c is None:
                    out[name] = {"up": False, "mode": r.get("mode"), "ms": None,
                                 "error": "no endpoint" if r.get("contract") != "rerank"
                                 else "no client for this contract yet"}
                    return
                h = c.health()
                out[name] = dict(h, mode=r.get("mode"))

            for name, r in doc.items():
                t = threading.Thread(target=one, args=(name, r), daemon=True)
                t.start()
                threads.append(t)
            for t in threads:
                t.join(3.0)
            self._health, self._health_at = out, time.monotonic()
            return out

    def roles_up(self) -> dict:
        return {k: bool(v.get("up")) for k, v in self.roles_health().items()}

    def role_test(self, role: str) -> dict:
        """One question to one role: its latency and its answer, or why it failed."""
        _, doc = self.store.get("roles")
        if role not in doc:
            return {"role": role, "ok": False, "error": "no such role"}
        c = self.role_client(role, doc)
        if c is None:
            return {"role": role, "ok": False, "error": "no endpoint or no client for its "
                                                         "contract"}
        _, router = self.store.get("router")
        opts = {c_["id"]: c_["description"] for c_ in router["classes"]}
        text = "Rename the variable `cnt` to `count` in utils.py and run the tests."
        if isinstance(c, Decider):
            a = c.choice(text, router["classifier"]["question"], opts)
        else:
            listing = "\n".join(f"- {k}: {v}" for k, v in opts.items())
            a = c.choice(text, router["classifier"]["qwen_prompt"].replace("{classes}", listing),
                         opts)
        out = {"role": role, "ok": a.error is None, "ms": a.ms, "question": text,
               "answer": a.as_dict()}
        if a.error:
            out["error"] = a.error
        with self._hlock:
            self._health_at = 0.0                            # the next health read is fresh
        return out

    # ------------------------------------------------------------------ the monitor's hand-over
    # profile threshold -> the monitor policy's parameter path, for the tasks the frame runs
    THRESHOLDS = {
        ("D1", "repeat_share"): ("D1", "arms", "repeat", "act"),
        ("D1", "grey_low"): ("D1", "arms", "repeat", "grey"),
        ("D1", "min_tokens"): ("D1", "arms", "repeat", "min_tokens"),
        ("D1", "every_tokens"): ("D1", "arms", "repeat", "every"),
        ("D1b", "min_span"): ("D1", "arms", "loop", "min_span"),
        ("D2", "backstop"): ("D2", "cap"),
        ("D2", "answer_room"): ("D2", "reserve"),
        ("E1", "tries"): ("E1", "tries"),
        ("C2", "max_file_lines"): ("C2", "max_lines"),
        ("C3", "repeats"): ("C3", "repeats"),
        ("C4", "repairs"): ("C4", "regenerations"),
    }
    # the tool-gate pack each tool-call task belongs to: a pack that is off turns its task off
    PACKS = {"C1": "risk", "C2": "project", "C3": "same_call", "C4": "malformed"}

    def monitor_mode(self, plan: Plan | None, server_mode: str | None) -> str | None:
        """The session monitor's mode for a request: the server's (`--conductor`, lowered by
        `X-Tandem`), and with a profile resolved never above the profile's: the raw model name
        runs the monitor raw too (SRV-117)."""
        if plan is None:
            return server_mode
        from server.conductor.monitor.policy import lower
        return lower(server_mode or "raw", plan.res.mode)

    def monitor_policy(self, plan: Plan | None, base):
        """The monitor's policy under this request's profile: the `monitor` document (version 0
        is the server's policy file, `base`) with the profile's task modes and tool-gate packs,
        caps and the thresholds that map onto the rules' parameters. Tasks the profile does not
        know keep the document's mode. Cached per profile and document version."""
        if plan is None or not plan.active or base is None:
            return None
        mv, mdoc = self.store.peek("monitor")
        key = (plan.res.profile, plan.res.version, mv, id(base))
        got = self._policies.get(key)
        if got is not None:
            return got
        import copy
        from server.conductor.monitor.policy import Policy
        data = copy.deepcopy(mdoc if mv else base.data)
        doc = plan.res.doc
        for tid, task in data["tasks"].items():
            if tid in doc["tasks"]:
                task["mode"] = doc["tasks"][tid]["mode"]
            pack = self.PACKS.get(tid)
            if pack and not doc["toolgate"].get(pack, False):
                task["mode"] = "off"
        for (tid, name), path in self.THRESHOLDS.items():
            val = doc["tasks"].get(tid, {}).get("thresholds", {}).get(name)
            node = data["tasks"].get(path[0])
            if val is None or node is None:
                continue
            for k in path[1:-1]:
                node = node.get(k) if isinstance(node, dict) else None
                if node is None:
                    break
            if isinstance(node, dict) and path[-1] in node:
                old = node[path[-1]]
                node[path[-1]] = int(val) if isinstance(old, int) and float(val).is_integer() \
                    else val
        data["caps"]["acts_per_request"] = doc["caps"]["acts_per_request"]
        data["caps"]["acts_per_session_hour"] = doc["caps"]["acts_per_session_hour"]
        pol = Policy(data, base.path)
        if len(self._policies) > 32:
            self._policies.clear()
        self._policies[key] = pol
        return pol

    def close(self) -> None:
        self.journal.close()


def locked_names() -> list[str]:
    return sorted(schema.LOCKED)
