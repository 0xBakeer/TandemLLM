"""Skills, prompts and presets (SRV-361; plan note Chat 3). Mixed into `chatd.service.Service`.

SKILLS are the engine's: the library (`/v1/library`, BM25 over the packages' SKILL.md files) and
the router that injects a package at the prompt's tail by the turn's class. A chat controls them
with `packages_mode`:

    auto     (the default) the harness picks; the chat's pins go in too, its blocks never
    manual   the pins only: every other known skill is sent as blocked
    model    the model reads the skill index (name and one line each, ~30 tokens a skill) in its
             system block and pulls a body with `library_read`; the harness injects nothing but
             the pins

Pins and blocks travel in the body as `tandem.packages` (server/conductor/packages.py
`body_packages`, honoured by the router). A skill switched off in the settings is blocked in every
chat. A server without the library (another engine, a cloud backend) lists no skills and the
modes change nothing.

PROMPTS are slash commands with a template ({{CLIPBOARD}}, {{selection}}, {{input}} filled on the
device); a chat's `prompt_id` makes one its system prompt. PRESETS are a model, a prompt, an
effort, tools, skills and MCP servers; a new chat can start from one, and changing the preset later
leaves existing chats as they are.
"""

from __future__ import annotations

import json
import re
import time

from chatd.ids import ulid

LIBRARY_TTL_S = 300
INDEX_MAX = 80


class SkillsMixin:
    def _api(self, status, kind, msg):
        from chatd.service import ApiError
        return ApiError(status, kind, msg)

    # ------------------------------------------------------------------ the library
    def library(self) -> list[dict]:
        """The engine's skills [{name, description, classes}], cached."""
        c = getattr(self, "_lib_cache", None)
        if c is not None and time.monotonic() - c[0] < LIBRARY_TTL_S:
            return c[1]
        st, obj = self.engine.get_json("/v1/library", timeout=5.0)
        rows = []
        if st == 200 and isinstance(obj, dict):
            for s in obj.get("skills") or obj.get("packages") or []:
                if isinstance(s, dict) and s.get("name"):
                    rows.append({"name": s["name"], "description": str(s.get("description")
                                                                       or "")[:300],
                                 "classes": s.get("classes") or [],
                                 "tokens": s.get("tokens")})
        self._lib_cache = (time.monotonic(), rows)
        return rows

    def _off(self) -> set:
        return {r["tool"][4:] for r in self.store.all(
            "SELECT tool FROM tool_prefs WHERE tool LIKE 'pkg:%' AND enabled = 0")}

    def packages(self) -> list[dict]:
        off = self._off()
        return [{"id": s["name"], "name": s["name"], "description": s["description"],
                 "kind": "skill", "version": "", "enabled": s["name"] not in off, "tools": [],
                 "tokens": s.get("tokens")} for s in self.library()]

    def set_package(self, pid: str, body: dict) -> dict:
        if not any(s["name"] == pid for s in self.library()):
            raise self._api(404, "not_found", "no such skill")
        self.store.run("INSERT OR REPLACE INTO tool_prefs VALUES (?, ?, NULL)",
                       ("pkg:" + pid, 1 if body.get("enabled", True) else 0))
        return next(p for p in self.packages() if p["id"] == pid)

    def skill_index(self, settings: dict) -> str | None:
        if settings.get("packages_mode") != "model":
            return None
        off = self._off() | set(settings.get("packages_blocked") or [])
        lines = [f"- {s['name']}: {s['description'].splitlines()[0][:160] if s['description'] else ''}"
                 for s in self.library() if s["name"] not in off][:INDEX_MAX]
        return "\n".join(lines) or None

    def tandem_field(self, settings: dict) -> dict:
        """What our engine reads from the body: the compaction marker (SRV-351) and the chat's
        pins and blocks by its mode (SRV-361)."""
        t: dict = {"compaction": False}
        pin = list(settings.get("packages") or [])
        block = set(settings.get("packages_blocked") or []) | self._off()
        mode = settings.get("packages_mode") or "auto"
        if mode in ("manual", "model"):
            block |= {s["name"] for s in self.library() if s["name"] not in pin}
        block -= set(pin)
        if pin or block:
            t["packages"] = {"pin": pin, "block": sorted(block)}
        return t

    # ------------------------------------------------------------------ prompts
    def prompts(self) -> list[dict]:
        return [{"id": r["id"], "title": r["title"], "command": r["command"], "body": r["template"]}
                for r in self.store.all("SELECT * FROM prompts ORDER BY command")]

    def save_prompt(self, pid: str | None, body: dict) -> dict:
        cmd = re.sub(r"[^a-z0-9_-]", "", str(body.get("command") or "").lower().lstrip("/"))[:40]
        if not cmd or not str(body.get("body") or "").strip():
            raise self._api(400, "bad_request", "a prompt needs a command and a body")
        clash = self.store.one("SELECT id FROM prompts WHERE command = ?", (cmd,))
        if clash is not None and clash["id"] != pid:
            raise self._api(409, "taken", f"/{cmd} is taken")
        pid = pid or ulid()
        self.store.run("INSERT OR REPLACE INTO prompts VALUES (?, ?, ?, ?)",
                       (pid, cmd, str(body.get("title") or cmd)[:120], str(body["body"])[:20000]))
        return next(p for p in self.prompts() if p["id"] == pid)

    def delete_prompt(self, pid: str) -> None:
        self.store.run("DELETE FROM prompts WHERE id = ?", (pid,))

    def commands(self) -> list[dict]:
        return [{"command": "clear", "description": "start a new chat"},
                {"command": "compact", "description": "summarize the older turns now"}] + \
            [{"command": p["command"], "description": p["title"], "prompt_id": p["id"]}
             for p in self.prompts()]

    # ------------------------------------------------------------------ presets
    PRESET_KEYS = ("model", "profile", "effort", "thinking", "tools", "prompt_id", "packages",
                   "mcp")

    def presets(self) -> list[dict]:
        out = []
        for r in self.store.all("SELECT * FROM presets ORDER BY name"):
            s = json.loads(r["settings"] or "{}")
            out.append({"id": r["id"], "name": r["name"], **{k: s.get(k) for k in
                                                            self.PRESET_KEYS}})
        return out

    def save_preset(self, pid: str | None, body: dict) -> dict:
        name = str(body.get("name") or "").strip()[:80]
        if not name:
            raise self._api(400, "bad_request", "a preset needs a name")
        pid = pid or ulid()
        s = {k: body.get(k) for k in self.PRESET_KEYS if k in body}
        self.store.run("INSERT OR REPLACE INTO presets VALUES (?, ?, ?, ?, ?)",
                       (pid, name, s.get("model"), None, json.dumps(s)))
        return next(p for p in self.presets() if p["id"] == pid)

    def delete_preset(self, pid: str) -> None:
        self.store.run("DELETE FROM presets WHERE id = ?", (pid,))

    def preset_settings(self, pid: str | None) -> dict:
        if not pid:
            return {}
        p = next((x for x in self.presets() if x["id"] == pid), None)
        if p is None:
            raise self._api(404, "not_found", "no such preset")
        return {k: v for k, v in p.items() if k in self.PRESET_KEYS and v is not None}
