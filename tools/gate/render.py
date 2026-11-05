"""gate-report.md: the header and one table a PR comment can hold, then the report's JSON under the
marker the CI check looks for (tools/gate/check.py)."""

from __future__ import annotations

import json
import re

from tools.gate import MARKER

#: GitHub's limit for a comment body is 65,536 characters; the report stays well under it
LIMIT = 60000


def _num(x, unit: str = "") -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        x = f"{x:.2f}" if abs(x) < 1000 else f"{x:,.0f}"
    return f"{x}{(' ' + unit) if unit else ''}"


def _side(s) -> str:
    if not isinstance(s, dict) or not s.get("n"):
        return "-"
    return f"{_num(s['median'])} ({_num(s['min'])}-{_num(s['max'])}, n={s['n']})"


def cell(r: dict, side: str) -> str:
    if r.get("unit"):
        return _side(r.get(side))
    if side == "base" and r.get("base_tokens") is not None:
        return f"{r['base_tokens']} tok"
    if side == "head" and r.get("head_tokens") is not None:
        return f"{r['head_tokens']} tok"
    return "-"


def table(rows: list[dict]) -> str:
    out = ["| suite | metric | base | head | delta | noise | verdict |",
           "|---|---|---|---|---|---|---|"]
    for r in rows:
        unit = f" ({r['unit']})" if r.get("unit") else ""
        delta = f"{r['delta_pct']:+.2f} %" if r.get("delta_pct") is not None else "-"
        noise = f"{r['noise_pct']:.2f} %" if r.get("noise_pct") is not None else "-"
        verdict = r["verdict"] if r["verdict"] == "PASS" else f"**{r['verdict']}**"
        resolved = r.get("delta_pct") is not None and not r["why"].startswith("not resolved")
        if r["verdict"] != "PASS" or r.get("delta_pct") is None or resolved:
            verdict += f": {r['why']}"
        out.append(f"| {r['suite']} | {r['metric']}{unit} | {cell(r, 'base')} | "
                   f"{cell(r, 'head')} | {delta} | {noise} | {verdict.replace('|', '/')} |")
    return "\n".join(out)


def knob_line(k: dict) -> str:
    keys = ("lanes", "max_running", "max_queue", "max_len", "prefill_chunk", "prefix_cache",
            "prefix_chunk", "cache_gb", "ceiling_gb", "peak_gb", "suffix_store",
            "response_cache", "drafter")
    return ", ".join(f"{x} {k.get(x)}" for x in keys if x in k)


def markdown(rep: dict, with_json: bool = True) -> str:
    hw, pack = rep.get("hardware") or {}, rep.get("pack") or {}
    b, h = rep.get("base") or {}, rep.get("head") or {}
    kb = (rep.get("knobs") or {}).get("head") or {}
    lines = [
        f"## Quality gate: {rep.get('verdict')} ({rep.get('level')})",
        "",
        f"- Hardware: {hw.get('name')}, {_num(hw.get('memory_gib'))} GiB, CC {hw.get('cc')}, "
        f"driver {hw.get('driver')}, CUDA {hw.get('cuda')}, torch {hw.get('torch')} "
        f"(profile `{hw.get('profile')}`)",
        f"- Pack: `{pack.get('id')}` (profile `{pack.get('profile')}`, weights "
        f"{str(pack.get('weights_sha256') or '-')[:16]})",
        f"- Base: `{str(b.get('sha'))[:12]}` ({b.get('ref')}), tree {str(b.get('tree_sha256'))[:16]}",
        f"- Head: `{str(h.get('sha'))[:12]}` ({h.get('ref')}), tree {str(h.get('tree_sha256'))[:16]}",
        f"- Knobs (head): {knob_line(kb) or '-'}",
        f"- Suites: {', '.join(rep.get('suites') or []) or '-'}; GPU tests: "
        f"{len(rep.get('kernels') or [])}; wall {_num(rep.get('wall_s'))} s; tool "
        f"{(rep.get('tool') or {}).get('version')}",
    ]
    if rep.get("planted"):
        lines.append(f"- PLANTED (dry run, not mergeable): {rep['planted'].get('note')}")
    lines += ["", table(rep.get("rows") or [])]
    reasons = rep.get("reasons") or {}
    if reasons:
        lines += ["", "Reasons:", *[f"- `{k}`: {v}" for k, v in sorted(reasons.items())]]
    if with_json:
        lines += ["", MARKER, "<details><summary>gate-report.json</summary>", "", "```json",
                  json.dumps(rep, separators=(",", ":"), sort_keys=True), "```", "</details>"]
    text = "\n".join(lines) + "\n"
    if len(text) > LIMIT:
        raise ValueError(f"the report is {len(text)} characters, over {LIMIT}: a comment cannot hold it")
    return text


BLOCK = re.compile(re.escape(MARKER) + r".*?```json\s*\n(.*?)\n\s*```", re.S)


def extract(texts: list[str]) -> tuple[dict | None, str | None]:
    """The last report in `texts` (a pull request's body, then its comments, oldest first), and
    None or why the last marked block could not be read."""
    found, err = None, None
    for t in texts:
        for m in BLOCK.finditer(t or ""):
            try:
                found, err = json.loads(m.group(1)), None
            except ValueError as exc:
                err = f"the marked JSON block does not parse: {exc}"
    return found, err
