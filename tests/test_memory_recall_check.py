"""SPD-249: tools/memory_recall_check.py `seed` writes synthetic notes only into a copy of a conductor config.

  * the default config (~/.qwen38-spark-engine/conductor) and the one QSE_CONDUCTOR_CONFIG names are refused, by
    their real paths (a symlink to either too);
  * a copy whose memory holds notes already is refused unless `--force`; an empty copy is seeded.

Run: python tests/test_memory_recall_check.py
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import memory_recall_check as mrc  # noqa: E402


def test_seed_refuses_a_served_config():
    with tempfile.TemporaryDirectory() as d:
        served = os.path.join(d, "served")
        os.makedirs(os.path.join(served, "memory"))
        link = os.path.join(d, "link")
        os.symlink(served, link)
        was = os.environ.get("QSE_CONDUCTOR_CONFIG")
        os.environ["QSE_CONDUCTOR_CONFIG"] = served
        try:
            assert "never the one a server runs with" in mrc.refusal(served, os.path.join(served, "memory"), True)
            assert mrc.refusal(link, os.path.join(link, "memory"), True) is not None, "the real path counts"
            default = os.path.expanduser("~/.qwen38-spark-engine/conductor")
            assert mrc.refusal(default, os.path.join(d, "x"), True) is not None
            os.environ["QSE_CONDUCTOR_CONFIG"] = "off"
            assert mrc.refusal(served, os.path.join(d, "empty"), False) is None, "off names no config"
        finally:
            if was is None:
                os.environ.pop("QSE_CONDUCTOR_CONFIG", None)
            else:
                os.environ["QSE_CONDUCTOR_CONFIG"] = was
    return "the default, QSE_CONDUCTOR_CONFIG and a link to it: refused"


def test_seed_refuses_a_memory_with_notes_unless_forced():
    was = os.environ.pop("QSE_CONDUCTOR_CONFIG", None)
    try:
        with tempfile.TemporaryDirectory() as d:
            mem = os.path.join(d, "memory")
            assert mrc.refusal(d, mem, False) is None, "no memory yet: seeded"
            os.makedirs(mem)
            assert mrc.refusal(d, mem, False) is None, "an empty memory: seeded"
            open(os.path.join(mem, "note.md"), "w").write("a note")
            assert "holds notes already" in mrc.refusal(d, mem, False)
            assert mrc.refusal(d, mem, True) is None, "--force"
    finally:
        if was is not None:
            os.environ["QSE_CONDUCTOR_CONFIG"] = was
    return "notes there: refused without --force"


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
