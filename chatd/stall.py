"""The run monitor's stall rule: a run that edits the same few files and checks the same thing
again and again with no change in what the check shows (SRV-677).

THE CASE (live 2026-10-21, chat 01M5F3ARQQ4XM4QF4YZTHXKZYB, a voxel reef in three.js). The run
rendered its page at step 96: the voxels near black. Steps 97-133 (37 steps) edited exposure,
lights, fog and colour grading in Atmosphere.js, App.js and PostFX.js and rendered again seven
times, each render with no console error and the voxels still near black; the cause was elsewhere
(the material's colours), and the user stopped it ("Stop changing lights and exposure; that is not
the problem"). The same-call guard never fired: no call repeated with the same arguments.

THE RULE. Every call is observed after it ran (`Watch.observe`):

  * a check is a render_page call, or a bash command the run already ran once (a test, a build).
    Its target: the page's path (the port and the size left out), or the command. Its outcome: the
    result's text without its first line (the render's time and hash) with every number as `#`
    (a counter or a frame rate is no change), and for a render the screenshot as a 32 x 18 grey
    thumbnail. A check makes PROGRESS when its target is new to the run, when its outcome text
    differs from the last check of that target (an error fixed, a new one), or when its picture
    differs from the last one's (correlation of the thumbnails under PICTURE_SAME: the page looks
    different, not only brighter or darker);
  * the steps between two progress checks are the run's own pace (`gaps`, the median of the last
    ones); a STREAK is the steps since the last progress check, with its checks and its edits
    (write_file and edit_file calls by path, a refused one too).

The streak fires when it holds at least MIN_CHECKS checks, has lasted max(MIN_STEPS, PACE x the
run's median gap; DEFAULT_GAP until the run has three gaps) steps, and its edits stay on a few
files (at least MIN_EDITS edits, the FEW most edited files at least FEW_SHARE of them). Then, once:

  1. STEER: the check's result gets a note that the last steps changed the same files and the
     checks showed the same outcome, and asks to step back, list the hypotheses, and test the
     cheapest one that would tell them apart (a minimal repro) before the next edit;
  2. STOP: when the streak goes on as long again (the same step count and MIN_CHECKS - 1 more
     checks) the result gets a stronger note and the run ends `interrupted` (`stall`) with a
     notice; Continue goes on from there and the model reads the note first.

A progress check resets the streak and the level. The note rides in the tool result as it is
stored, so every later request renders the same bytes (no hidden tail, the prefix stays). Standard
library only: the thumbnail is decoded here (8-bit PNG, every filter; 0.1-0.4 s for a 1280 x 720
screenshot on the Spark).
"""

from __future__ import annotations

import collections
import math
import re
import struct
import threading
import zlib

MIN_CHECKS = 4
MIN_STEPS = 12
PACE = 3
DEFAULT_GAP = 5
MIN_EDITS = 3
FEW = 3
FEW_SHARE = 0.6
PICTURE_SAME = 0.3
THUMB = (32, 18)
GAPS_KEPT = 20

CHECK_TOOLS = ("render_page",)
EDIT_TOOLS = ("write_file", "edit_file")
_NUM = re.compile(r"\d+(?:[.,]\d+)*")
_WS = re.compile(r"\s+")


def thumb(data: bytes, gw: int = THUMB[0], gh: int = THUMB[1]) -> list[int] | None:
    """A gw x gh grey thumbnail of an 8-bit, non-interlaced PNG (grey, RGB, grey+alpha, RGBA),
    from a sample of rows and columns (about 8 rows and 4 columns a cell); None for anything
    else."""
    if not data or data[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    i, idat, w, h, bpp = 8, [], 0, 0, 0
    try:
        while i + 8 <= len(data):
            n, t = struct.unpack(">I4s", data[i:i + 8])
            d = data[i + 8:i + 8 + n]
            i += 12 + n
            if t == b"IHDR":
                w, h, depth, ctype, _, _, inter = struct.unpack(">IIBBBBB", d)
                if depth != 8 or inter or ctype not in (0, 2, 4, 6):
                    return None
                bpp = {0: 1, 2: 3, 4: 2, 6: 4}[ctype]
            elif t == b"IDAT":
                idat.append(d)
            elif t == b"IEND":
                break
        if not (w and h and bpp):
            return None
        raw = zlib.decompress(b"".join(idat))
    except (struct.error, zlib.error, KeyError):
        return None
    stride = w * bpp
    if len(raw) < h * (stride + 1):
        return None
    prev = bytearray(stride)
    sums, cnt = [0] * (gw * gh), [0] * (gw * gh)
    ystep = max(1, h // (gh * 8))
    xstep = max(1, w // (gw * 4))         # every cell gets columns, a narrow picture too
    pos = 0
    for y in range(h):
        f, line = raw[pos], bytearray(raw[pos + 1:pos + 1 + stride])
        pos += 1 + stride
        if f == 1:
            for x in range(bpp, stride):
                line[x] = (line[x] + line[x - bpp]) & 255
        elif f == 2:
            line = bytearray((a + b) & 255 for a, b in zip(line, prev))
        elif f == 3:
            for x in range(stride):
                left = line[x - bpp] if x >= bpp else 0
                line[x] = (line[x] + ((left + prev[x]) >> 1)) & 255
        elif f == 4:
            for x in range(stride):
                a = line[x - bpp] if x >= bpp else 0
                b, c = prev[x], (prev[x - bpp] if x >= bpp else 0)
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                line[x] = (line[x] + (a if pa <= pb and pa <= pc else b if pb <= pc else c)) & 255
        prev = line
        if y % ystep:
            continue
        gy = y * gh // h
        for x in range(0, w, xstep):
            o = x * bpp
            v = line[o] if bpp < 3 else (line[o] * 3 + line[o + 1] * 6 + line[o + 2]) // 10
            k = gy * gw + x * gw // w
            sums[k] += v
            cnt[k] += 1
    return [s // max(1, c) for s, c in zip(sums, cnt)]


def correlation(a: list[int], b: list[int]) -> float:
    """Pearson correlation of two thumbnails: 1 the same picture up to brightness and contrast.
    Two flat pictures (one colour each) count as the same."""
    n = len(a)
    if not n or n != len(b):
        return 0.0
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((x - mb) ** 2 for x in b)
    if va < n and vb < n:                   # both flat (a blank page, a black canvas)
        return 1.0
    if va < n or vb < n:
        return 0.0
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(va * vb)


def outcome_text(text: str) -> str:
    """What a check's result says, without what changes on every run of it: its first line (the
    render's time, hash and path) and every number."""
    lines = (text or "").splitlines()[1:]
    return _WS.sub(" ", _NUM.sub("#", "\n".join(lines))).strip()


def _target(name: str, args: dict) -> str:
    if name == "render_page":
        where = str(args.get("path") or args.get("url") or "/")
        where = re.sub(r"^[a-z]+://[^/]+", "", where)
        return "page:" + (where or "/")
    return "bash:" + _WS.sub(" ", str(args.get("command") or "")).strip()


class Watch:
    """One run's stall rule (`observe` after every call). Thread-safe: read calls may run in
    parallel."""

    def __init__(self):
        self.lock = threading.Lock()
        self.last: dict[str, tuple[str, list | None]] = {}   # target -> (outcome, thumbnail)
        self.commands: set[str] = set()      # bash commands run once (a second run is a check)
        self.gaps: collections.deque = collections.deque(maxlen=GAPS_KEPT)
        self.progress_step: int | None = None
        self.start = 0                       # the streak's first step
        self.checks = 0                      # checks in the streak
        self.edits: collections.Counter = collections.Counter()
        self.level = 0                       # 0 quiet, 1 steered, 2 stopped
        self.fired_at = 0                    # the step of the last note
        self.fired_checks = 0                # the streak's checks at the last note
        self.stop = False                    # the run ends after this step
        self.events: list[dict] = []         # what fired, for the log and the replay

    # ------------------------------------------------------------------ the run's pace
    def threshold(self) -> int:
        gaps = sorted(self.gaps)
        gap = gaps[len(gaps) // 2] if len(gaps) >= 3 else DEFAULT_GAP
        return max(MIN_STEPS, PACE * gap)

    def _progress(self, step: int) -> None:
        if self.progress_step is not None and step > self.progress_step:
            self.gaps.append(step - self.progress_step)
        self.progress_step = step
        self.start, self.checks, self.level = step, 1, 0
        self.edits = collections.Counter()

    def _few(self) -> tuple[bool, list[str]]:
        n = sum(self.edits.values())
        top = self.edits.most_common(FEW)
        return (n >= MIN_EDITS and sum(c for _, c in top) >= FEW_SHARE * n,
                [p for p, _ in top])

    # ------------------------------------------------------------------ one call
    def observe(self, step: int, name: str, args: dict, text: str, is_error: bool = False,
                images=()) -> str | None:
        """After a call ran: the note to append to its result, or None. Sets `stop` when the run
        should end after this step."""
        with self.lock:
            if name in EDIT_TOOLS:
                # an attempt counts, done or refused: the run is working on that file
                if args.get("path"):
                    self.edits[str(args["path"]).removeprefix("/workspace/")] += 1
                return None
            if name == "bash":
                cmd = _target(name, args)
                if cmd not in self.commands:
                    self.commands.add(cmd)
                    self.last[cmd] = (outcome_text("\n" + (text or "")), None)
                    return None
            elif name not in CHECK_TOOLS:
                return None
            target = _target(name, args)
            out = outcome_text(text if name != "bash" else "\n" + (text or ""))
            pic = None
            for mime, data in images or ():
                pic = thumb(data)
                break
            before = self.last.get(target)
            self.last[target] = (out, pic)
            if before is None or before[0] != out or (
                    pic is not None and before[1] is not None and
                    correlation(pic, before[1]) < PICTURE_SAME):
                self._progress(step)
                return None
            self.checks += 1
            return self._fire(step)

    def _fire(self, step: int) -> str | None:
        few, files = self._few()
        span = self.threshold()
        steps = step - self.start
        if not few:
            return None
        if self.level == 0 and self.checks >= MIN_CHECKS and steps >= span:
            self.level, self.fired_at, self.fired_checks = 1, step, self.checks
            self.events.append({"step": step, "level": "steer", "steps": steps,
                                "checks": self.checks, "files": files, "span": span})
            return steer_note(steps, self.checks, files)
        if self.level == 1 and step - self.fired_at >= span and \
                self.checks - self.fired_checks >= MIN_CHECKS - 1:
            self.level, self.stop = 2, True
            self.events.append({"step": step, "level": "stop", "steps": steps,
                                "checks": self.checks, "files": files, "span": span})
            return stop_note(steps, self.checks, files)
        return None


def _names(files: list[str]) -> str:
    short = [f.rsplit("/", 1)[-1] for f in files]
    return ", ".join(short[:-1]) + (" and " if len(short) > 1 else "") + short[-1]


def steer_note(steps: int, checks: int, files: list[str]) -> str:
    return (f"\n\n[run monitor: the last {steps} steps changed the same few files ({_names(files)})"
            f" and the last {checks} checks of this showed the same outcome: no error changed and "
            f"the picture did not change in kind. The cause is likely not where the edits have "
            f"been. Before the next edit, step back: list the hypotheses that could explain the "
            f"symptom, including ones outside these files; pick the cheapest test that would tell "
            f"them apart (for example a minimal page or script that isolates the suspected part, "
            f"with known inputs and an expected output) and run it; then fix what it shows.]")


def stop_note(steps: int, checks: int, files: list[str]) -> str:
    return (f"\n\n[run monitor: still the same outcome after {steps} steps and {checks} checks "
            f"of edits to {_names(files)}. The run stops here so the user can look. When it "
            f"goes on: do not edit these files again first. Write down what the checks have "
            f"ruled out, then build a minimal repro of the symptom in a new file and find the "
            f"cause there.]")


def stop_notice(ev: dict) -> str:
    return (f"The run stopped: {ev['steps']} steps changed the same files "
            f"({_names(ev['files'])}) and {ev['checks']} checks showed no change (the run "
            f"monitor). Continue goes on from here.")

