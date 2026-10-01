"""The secret-pattern refusal (SRV-137; the note rules of SRV-141 use the same list).

Nothing the runtime writes to disk from a session may hold a secret: a transcript block passes
through `redact()` before it is written, and every span that matches one of the shapes below is
replaced by `[redacted:<kind>]`. The caller journals the counts per kind, never the value.

The shapes are the ones that turn up in a coding session: private-key blocks, bearer and basic
authorisation headers, the token formats of the services a project talks to, JWTs, credentials in
a URL or a connection string, and a password, secret or token assigned to a quoted string or an
upper-case environment name. A shape errs on the side of redacting: an identifier that only looks
like a token loses its value in the transcript, which costs a reader a little; a real key in a file
on the box would cost much more. Plain code (`token = tok(text)`, `password: str`) does not match:
the assignment shapes want a quoted string or an environment name.
"""

from __future__ import annotations

import re

# (kind, pattern, the group that is the secret: 0 for the whole match)
PATTERNS: list[tuple[str, re.Pattern, int]] = [
    ("private_key", re.compile(r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----"
                               r"[\s\S]*?-----END (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----"), 0),
    # a key block whose end was cut off still holds key material
    ("private_key", re.compile(r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----[\s\S]*\Z"), 0),
    ("bearer", re.compile(r"(?i)\bbearer\s+([A-Za-z0-9\-._~+/]{16,}=*)"), 1),
    ("basic_auth", re.compile(r"(?i)\bauthorization:\s*basic\s+([A-Za-z0-9+/]{8,}=*)"), 1),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA|ABIA|ACCA)[0-9A-Z]{16}\b"), 0),
    ("aws_secret", re.compile(r"(?i)aws_secret_access_key\s*[=:]\s*[\"']?([A-Za-z0-9/+=]{40})"), 1),
    ("github_token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{22,})"), 0),
    ("gitlab_token", re.compile(r"\bglpat-[A-Za-z0-9_\-]{20,}"), 0),
    ("api_key", re.compile(r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_\-]{20,}"), 0),
    ("stripe_key", re.compile(r"\b(?:sk|rk|pk)_(?:live|test)_[0-9A-Za-z]{16,}"), 0),
    ("slack_token", re.compile(r"\bxox[abposr]-[A-Za-z0-9\-]{10,}"), 0),
    ("google_key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"), 0),
    ("hf_token", re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"), 0),
    ("npm_token", re.compile(r"\bnpm_[A-Za-z0-9]{36}\b"), 0),
    ("ntfy_token", re.compile(r"\btk_[a-z0-9]{29}\b"), 0),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"), 0),
    ("url_credentials", re.compile(r"\b[a-z][a-z0-9+.\-]*://[^\s:/@\"'<>]+:([^\s@/\"'<>]+)@"), 1),
    ("connection_string", re.compile(r"(?i)\b(?:AccountKey|SharedAccessKey|Password|Pwd)=([^;\s\"']{6,})"), 1),
    ("password", re.compile(r"(?i)(?:password|passwd|passphrase|secret|api[_-]?key|access[_-]?token|"
                            r"auth[_-]?token|client[_-]?secret|private[_-]?key)[\"']?\s*[:=]\s*"
                            r"[\"']([^\"'\s]{6,})[\"']"), 1),
    ("password", re.compile(r"(?i)--(?:password|passwd|token|api-key|secret)(?:=|\s+)"
                            r"(?![-$])([^\s\"']{6,})"), 1),
    ("env_secret", re.compile(r"\b[A-Z][A-Z0-9_]*(?:PASSWORD|PASSWD|SECRET|TOKEN|API_KEY|ACCESS_KEY|"
                              r"PRIVATE_KEY)[A-Z0-9_]*\s*[=:]\s*[\"']?([^\s\"'$][^\s\"']{5,})"), 1),
]
KINDS = tuple(dict.fromkeys(k for k, _, _ in PATTERNS))


def redact(text: str) -> tuple[str, dict[str, int]]:
    """`text` with every secret span replaced by `[redacted:<kind>]`, and the count per kind.
    The shapes run in order on what the earlier ones left, so a key inside a URL inside an
    assignment is counted once."""
    counts: dict[str, int] = {}
    if not text:
        return text, counts
    for kind, rx, group in PATTERNS:
        def sub(m: re.Match, kind=kind, group=group) -> str:
            if m.group(group).startswith("[redacted:"):
                return m.group(0)                  # an earlier shape took it already
            counts[kind] = counts.get(kind, 0) + 1
            if group == 0:
                return f"[redacted:{kind}]"
            s, e = m.span(group)
            return m.group(0)[:s - m.start()] + f"[redacted:{kind}]" + m.group(0)[e - m.start():]
        text = rx.sub(sub, text)
    return text, counts


def merge(into: dict[str, int], more: dict[str, int]) -> dict[str, int]:
    for k, v in more.items():
        into[k] = into.get(k, 0) + v
    return into
