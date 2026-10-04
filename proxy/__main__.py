"""`python -m proxy`: the standalone proxy and dashboard (docs/proxy.md).

    python -m proxy                                  ~/.tandemllm/config.sqlite3, 127.0.0.1:8080
    python -m proxy --env-file proxy.env --host 0.0.0.0 --port 8080
    python -m proxy --config-db /data/config.sqlite3 --usage-ledger /data/ledger.sqlite3

Needs Python 3.11 or newer, the standard library and `cryptography` (for keys at rest). No torch,
no GPU. A fresh `--config-db` path is created at schema version 1.
"""

from __future__ import annotations

import argparse
import os
import sys

if sys.version_info < (3, 11):                                    # pragma: no cover
    raise SystemExit("the proxy needs Python 3.11 or newer")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proxy import config as config_mod  # noqa: E402
from server import ledger as ledger_mod  # noqa: E402


def load_env_file(path: str, env=None) -> list[str]:
    """`KEY=VALUE` lines into the environment; a variable already set keeps its value (the
    process environment wins over the file). Returns the names it set."""
    env = os.environ if env is None else env
    got = []
    with open(os.path.expanduser(path)) as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].lstrip()
            k, sep, v = line.partition("=")
            k = k.strip()
            if not sep or not k.replace("_", "").isalnum():
                raise SystemExit(f"[proxy] {path}:{n}: not a KEY=VALUE line")
            v = v.strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
                v = v[1:-1]
            if k not in env:
                env[k] = v
                got.append(k)
    return got


def parser() -> argparse.ArgumentParser:
    env = os.environ
    ap = argparse.ArgumentParser(prog="python -m proxy", description=__doc__.split("\n")[0])
    ap.add_argument("--config-db", default=env.get("TANDEM_CONFIG_DB", config_mod.DEFAULT_PATH),
                    help="the config database (created when missing); TANDEM_CONFIG_DB")
    ap.add_argument("--env-file", default=None,
                    help="KEY=VALUE lines read before the start; the process environment wins")
    ap.add_argument("--host", default=env.get("TANDEM_HOST", "127.0.0.1"),
                    help="the address to bind (default 127.0.0.1; the page is for a local "
                         "network: never put it on the internet without a login and TLS)")
    ap.add_argument("--port", type=int, default=int(env.get("TANDEM_PORT", "8080")))
    ap.add_argument("--usage-ledger", default=env.get("TANDEM_USAGE_LEDGER",
                                                      "~/.tandemllm/ledger.sqlite3"),
                    help="the usage ledger (one row per request, no text); 'off' for none")
    ap.add_argument("--usage-retention-days", type=int, default=ledger_mod.MIN_RETENTION_DAYS)
    ap.add_argument("--dashboard-login",
                    default=env.get("TANDEM_DASHBOARD_LOGIN", env.get("QSE_DASHBOARD_LOGIN",
                                                                      "off")),
                    choices=("on", "off"),
                    help="on: the dashboard API needs TANDEM_ADMIN_TOKEN or a session")
    ap.add_argument("--dashboard-dir", default=None, help="default: dashboard/dist")
    ap.add_argument("--trust-loopback", default="on", choices=("on", "off"),
                    help="a request from this machine without proxy headers counts as the admin")
    ap.add_argument("--no-keychain", action="store_true",
                    help="do not ask the OS keychain for the master key")
    ap.add_argument("--verbose", action="store_true")
    return ap


def main(argv=None) -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--env-file")
    known, _ = pre.parse_known_args(argv)
    if known.env_file:
        load_env_file(known.env_file)
    args = parser().parse_args(argv)
    from proxy import app
    try:
        app.serve(args)
    except config_mod.ConfigError as exc:
        raise SystemExit(f"[proxy] configuration error: {exc.message}")


if __name__ == "__main__":
    main()
