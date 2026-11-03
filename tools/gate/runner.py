"""`python -m tools.gate run`: base and head side by side on this box, one fresh server per arm.

    python -m tools.gate run [--suite auto|quick|quick+long|full] [--base origin/main] [--head HEAD]
                             [--pack ID] [--hw auto] [--python PY] [--port 8100] [--pairs N]
                             [--site FILE] [--lock FILE] [--out DIR] [--dry]

The order: the head's GPU tests (no server); then per pair the base's server and the head's, each
running the suites; the head without speculation (lossless: spec vs plain) and, at quick+long and
full, each side with the wide prefill chunk (long); last, the base's run of any GPU test the head
failed. Each server is started fresh from a `git archive` of its commit, with its own config copy
and state directory, on a free port from --port up (never below 8100), and stopped before the next
starts; a server the gate did not start is never touched. The measuring code (these suites,
tools/parallel_card.py, tools/soak.py) is this checkout's, the same instrument for both sides.

`--site FILE` (YAML) names what this box adds beside a checkout: `serve_env` (its ops/serve.env;
REPO is rewritten for each tree), `setup_env` (its run/setup.env, under the arm's own settings),
`local_env`, `links` / `copies` ({path in the tree: path on the box}, a file or a folder), and
`conductor_from` (a conductor config each arm copies, without its decisions or tokens). Without it
an install's run/local.env is used where there is one.

`--head-patch FILE` plants a change in the head's tree and `--head-env K=V` a setting in the head's
servers (a dry run of the gate itself); the report says so and the CI check refuses it.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path

from tools.gate import LEVEL_SUITES, LEVELS, SCHEMA, VERSION, level_index
from tools.gate import hw as hwmod
from tools.gate import pathmap, server
from tools.gate.client import client, prompts
from tools.gate.suites import Ctx, Plan, module
from tools.gate.suites import kernels as ksuite
from tools.gate.verdict import overall, skipped

REPO = Path(__file__).resolve().parent.parent.parent
PACKS = Path(__file__).resolve().parent / "packs.yaml"


def load_packs() -> dict:
    import yaml
    return yaml.safe_load(PACKS.read_text())


def gate_code_sha256() -> str:
    h = hashlib.sha256()
    here = Path(__file__).resolve().parent
    for p in sorted(here.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            h.update(str(p.relative_to(here)).encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


def plan_of(a, prof, pack: dict, files: list[str]) -> Plan:
    need = pathmap.required(files)
    level = need["level"] if a.suite == "auto" else a.suite
    if level == "none":
        return Plan("none", [], [], prof, pack, a.pairs)
    if level_index(level) < level_index(need["level"]):
        print(f"[gate] note: --suite {level} is below what the diff needs ({need['level']}); the CI "
              "check will refuse the report", file=sys.stderr)
    suites = list(LEVEL_SUITES[level])
    if a.only:
        suites = [s for s in suites if s in a.only.split(",")]
    return Plan(level, suites, need["kernels"] if "kernels" in suites else [], prof, pack,
                a.pairs if a.pairs else (2 if level == "full" else 1))


def arms_of(plan: Plan) -> list[tuple[str, str, list[str]]]:
    """(side, variant, suites) in the order they run."""
    main = [s for s in plan.suites if s != "kernels"]
    out = []
    for _ in range(plan.pairs):
        out += [("base", "main", main), ("head", "main", main)]
    if "lossless" in plan.suites and plan.profile.plain_variant and plan.pack.get("plain_env"):
        out.append(("head", "plain", ["lossless:plain"]))
    if "long" in plan.suites and plan.level in ("quick+long", "full") and plan.pack.get("wide_chunk_env"):
        out += [("base", "wide", ["long:wide"]), ("head", "wide", ["long:wide"])]
    return out


def arm_env(prof, pack: dict, side: str, variant: str, head_env: dict | None = None) -> dict:
    """An arm's environment: the hardware profile's, the variant's (plain: speculation off; wide:
    the wide prefill chunk), and on the head a dry run's planted knobs (`--head-env`)."""
    env = dict(prof.arm_env())
    if variant == "plain":
        env.update(pack.get("plain_env") or {})
    if variant == "wide":
        env.update(pack.get("wide_chunk_env") or {})
    if side == "head":
        env.update(head_env or {})
    return env


@contextlib.contextmanager
def box_lock(path: str | None):
    if not path:
        yield
        return
    fh = open(os.path.expanduser(path), "a")
    print(f"[gate] waiting for the lock {path}", flush=True)
    fcntl.flock(fh, fcntl.LOCK_EX)
    try:
        yield
    finally:
        fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()


def site_of(a) -> dict:
    site: dict = {}
    if a.site:
        import yaml
        site = yaml.safe_load(Path(a.site).read_text()) or {}
    local = REPO / "run" / "local.env"
    if not site.get("local_env") and local.exists():
        site["local_env"] = str(local)
    return site


def run_arm(arm, launch, suites: list[str], plan: Plan, ctx_args: dict, raw: dict, knobs: dict,
            log) -> None:
    log(f"[gate] {arm.key}: starting on :{arm.port}")
    try:
        k = launch.start(arm)
        k["server_start_s"] = arm.started_s
        knobs.setdefault(arm.key, k)
        h = server.pc.health(arm.base)
        if "weights_sha256" not in ctx_args:
            ctx_args["weights_sha256"] = server.weights_sha256((h.get("weights") or {}).get("weights"))
        log(f"[gate] {arm.key}: up in {arm.started_s} s; lanes {k['lanes']}, max_running "
            f"{k['max_running']} ({k['max_running_reason']}), max_len {k['max_len']}, ceiling/peak "
            f"{k['ceiling_gb']}/{k['peak_gb']} GB, drafter {k['drafter']}")
        tok = ctx_args["tokenizer"](h)
        cl = client(arm.base, h.get("model") or ctx_args["served"], tok)
        ctx = Ctx(arm.side, arm.variant, arm.base, cl, h, k, plan.profile, plan.level, plan.pack,
                  ctx_args["out"], ctx_args["pj"], log)
        for s in suites:
            name, _, part = s.partition(":")
            mod = module(name)
            fn = getattr(mod, f"measure_{part}" if part else "measure")
            t0 = time.monotonic()
            raw.setdefault(name, {}).setdefault(arm.key, []).append(fn(ctx))
            log(f"[gate] {arm.key}: {s} done in {time.monotonic() - t0:.0f} s")
    finally:
        launch.stop(arm)
        log(f"[gate] {arm.key}: stopped")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tools.gate run", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="auto", choices=("auto",) + LEVELS[1:])
    ap.add_argument("--base", default="origin/main")
    ap.add_argument("--head", default="HEAD")
    ap.add_argument("--pack", default=None, help="default: the hardware profile's reference pack")
    ap.add_argument("--hw", default="auto")
    ap.add_argument("--python", default=sys.executable, help="the engine's Python (its torch)")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--ports", default="", help="comma list: the arms take these ports in order "
                    "before --port's scan (a test suite hands out its own ports)")
    ap.add_argument("--pairs", type=int, default=0, help="servers a side (default 1; 2 at full)")
    ap.add_argument("--site", default=None)
    ap.add_argument("--lock", default=None, help="a lock file to hold for the GPU part (flock)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", default=None, help="these suites only (comma list; a partial report)")
    ap.add_argument("--head-patch", default=None, help="plant this patch in the head's tree (dry run)")
    ap.add_argument("--head-env", action="append", default=[], metavar="K=V",
                    help="plant this setting in the head's servers (dry run; repeatable)")
    ap.add_argument("--note", default=None)
    ap.add_argument("--keep-trees", action="store_true")
    ap.add_argument("--dry", action="store_true", help="print the plan, run nothing")
    a = ap.parse_args(argv)
    log = lambda *x: print(*x, flush=True)  # noqa: E731

    head_env = dict(kv.split("=", 1) for kv in a.head_env)
    repo = str(REPO)
    base_sha = server.git(repo, "rev-parse", f"{a.base}^{{commit}}")
    head_sha = server.git(repo, "rev-parse", f"{a.head}^{{commit}}")
    dirty = server.git(repo, "status", "--porcelain", "--untracked-files=no")
    if dirty and a.head == "HEAD":
        log("[gate] note: uncommitted changes are not measured (the head is the commit)")
    files = pathmap.changed_files(base_sha, head_sha, repo)
    prof = hwmod.profile(a.hw, python=a.python)
    packs = load_packs()
    pack_id = a.pack or prof.pack()
    if not pack_id or pack_id not in packs:
        log(f"[gate] no pack: pass --pack (one of {', '.join(packs)}); profile {prof.id} names "
            f"{prof.pack()!r}")
        return 2
    pack = dict(packs[pack_id], id=pack_id)
    plan = plan_of(a, prof, pack, files)
    arms = arms_of(plan)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = Path(a.out or REPO / "results" / "gate" / f"{head_sha[:12]}-{stamp}")
    log(f"[gate] base {base_sha[:12]} ({a.base}), head {head_sha[:12]} ({a.head}), {len(files)} files; "
        f"level {plan.level}; hardware {prof.id} ({prof.fp.name}); pack {pack_id}")
    log(f"[gate] suites {', '.join(plan.suites) or '-'}; GPU tests {len(plan.kernels)}; servers: "
        + ", ".join(f"{s}{'' if v == 'main' else ':' + v}" for s, v, _ in arms))
    if a.dry or plan.level == "none":
        if plan.level == "none":
            log("[gate] the diff needs no GPU report")
        return 0
    if pack["launcher"] == "ops" and hwmod.no_gpu():
        log("[gate] no GPU here: an ops pack needs one (the fake pack runs without)")
        return 2

    t_start = time.monotonic()
    out.mkdir(parents=True, exist_ok=True)
    work = out / "trees"
    trees = {"base": work / f"base-{base_sha[:12]}", "head": work / f"head-{head_sha[:12]}"}
    tree_sha = {"base": server.make_tree(repo, base_sha, trees["base"]),
                "head": server.make_tree(repo, head_sha, trees["head"], a.head_patch)}
    site = site_of(a)
    launch = server.launcher(pack, prof, a.python, site)
    if hasattr(launch, "prepare"):
        for t in trees.values():
            launch.prepare(t)
    pj = prompts()
    raw: dict = {}
    knobs: dict = {}
    kern: dict = {"head": {}, "base": {}}
    k_none = None if pack["launcher"] == "ops" else "the fake engine has no GPU"
    ctx_args = {"out": out, "pj": pj, "served": "TandemLLM",
                "tokenizer": (lambda h: None) if pack["launcher"] == "fake" else server.load_tokenizer}
    taken: set = set()

    def bail(signum, _frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, bail)

    with box_lock(a.lock):
        if plan.kernels and not k_none:
            kern["head"] = ksuite.run_all(trees["head"], plan.kernels, a.python, log)
        for side, variant, suites in arms:
            given = [int(p) for p in a.ports.split(",") if p.strip()]
            if any(p < server.MIN_PORT for p in given):
                raise SystemExit(f"[gate] --ports below {server.MIN_PORT}: the gate never starts "
                                 "a server beside the box's own")
            left = [p for p in given if p not in taken and not server.port_busy(p, strict=True)]
            port = left[0] if left else server.free_port(a.port, taken)
            taken.add(port)
            env = arm_env(prof, pack, side, variant, head_env)
            arm = server.Arm(side, variant, trees[side], port,
                             out / "arms" / f"{side}-{variant}-{len(knobs)}", env)
            if pack["launcher"] == "ops":
                free = prof.free_gib()
                if free is not None and free < prof.min_free_gib():
                    raise SystemExit(f"[gate] {free:.0f} GiB free, the profile needs "
                                     f"{prof.min_free_gib():.0f} before a server starts: is "
                                     "another engine running?")
            run_arm(arm, launch, suites, plan, ctx_args, raw, knobs, log)
        failed = [f for f, r in kern["head"].items() if r["rc"] != 0 and not r["skipped"]]
        if failed:
            kern["base"] = ksuite.run_all(trees["base"], failed, a.python, log)

    rows: list = []
    for s in plan.suites:
        if s == "kernels":
            rows += ksuite.rows(kern["head"], kern["base"], plan.kernels, k_none)
        else:
            rows += module(s).rows(raw.get(s) or {}, plan)
    weights = (knobs.get("head") or {}).get("weights")
    wsha = (hashlib.sha256(pack_id.encode()).hexdigest() if pack["launcher"] == "fake"
            else ctx_args.get("weights_sha256"))
    rep = {"kind": "tandem-gate-report", "schema": SCHEMA,
           "tool": {"version": VERSION, "code_sha256": gate_code_sha256()},
           "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "level": plan.level, "suites": plan.suites, "kernels": plan.kernels,
           "hardware": {"profile": prof.id, **prof.fp.as_dict()},
           "pack": {"id": pack_id, "profile": pack.get("profile"), "weights": weights,
                    "weights_sha256": wsha},
           "base": {"ref": a.base, "sha": base_sha, "tree_sha256": tree_sha["base"]},
           "head": {"ref": a.head, "sha": head_sha, "tree_sha256": tree_sha["head"]},
           "knobs": knobs, "profile": prof.describe(), "pairs": plan.pairs,
           "planted": ({"patch_sha256": hashlib.sha256(Path(a.head_patch).read_bytes()).hexdigest()
                        if a.head_patch else None, "env": head_env or None,
                        "note": a.note or os.path.basename(a.head_patch or "") or "head settings"}
                       if a.head_patch or head_env else None),
           "rows": rows, "reasons": {}, "verdict": overall(rows),
           "wall_s": round(time.monotonic() - t_start, 1)}
    (out / "raw.json").write_text(server.dumps({"raw": raw, "kernels": kern}))
    from tools.gate.__main__ import write
    jp, mp = write(rep, str(out))
    if not a.keep_trees:
        import shutil
        shutil.rmtree(work, ignore_errors=True)
    log(f"[gate] {rep['verdict']} in {rep['wall_s']:.0f} s: {jp}")
    log(open(mp).read().split("\n<!--")[0])
    return 0 if rep["verdict"] in ("PASS", "WORSE", "CHANGED_TEXT") else 1
