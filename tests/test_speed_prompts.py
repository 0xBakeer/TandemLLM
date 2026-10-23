"""SPD-183: the decode speed prompts are versioned. The recorded sha256 of each text must match it, so an
edit that keeps the version fails here; the recorded token counts stay in the 1.25k-1.36k band."""

import hashlib
import json
from pathlib import Path

P = Path(__file__).resolve().parent.parent / "bench" / "speed-prompts-v1.json"


def test_texts_match_their_recorded_hashes():
    p = json.loads(P.read_text())
    assert p["version"] == "spd183-v1"
    assert hashlib.sha256(p["system"].encode()).hexdigest() == p["sha256"]["system"]
    for k in ("code", "prose"):
        assert hashlib.sha256(p["workloads"][k].encode()).hexdigest() == p["sha256"][k]


def test_token_counts_in_band():
    p = json.loads(P.read_text())
    for k in ("code", "prose"):
        lo, hi = p["token_counts"][k]
        assert 1250 <= lo <= hi <= 1360
    assert "{nonce}" in p["nonce_line"]
