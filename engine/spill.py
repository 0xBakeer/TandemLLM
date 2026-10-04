"""Live sessions on the NVMe: a conversation's rows, anchors and end anchor written whole and read
back byte for byte (0.3.22-test).

THE TWO RE-READS (the owner, 2026-10-03). (1) The 15:35 deploy restarted the engine and the next turn
of his 130k opencode conversation re-read 120,102 tokens: TTFT 158 s. The disk tier
(engine/disk_cache.py) keeps only what a cold grid prefill wrote, and in an agent conversation
everything past its first answer was written by verify blocks or resumed from an end anchor (the
resident prefix's `taint`), so only the session's head (25,600 tokens) came back. (2) At 15:37 a
10,216-token request from another client was too long to be a guest (`--resident-guest-max`
8,192); it replaced the resident prompt, the RAM park holds at most about 35,000 rows above where
the two part, and the conversation's next turn re-read everything above 32,768: 165 s.

WHAT THIS KEEPS. One ENTRY per conversation (its session key, else the longest shared prefix):
the rows [0, valid) of every position-indexed buffer (the 16 attention layers' KV and the drafters'
own), the recurrent state and convolution tails at the end of the last generation (the end anchor,
with its hidden spans and the drafter's mark) and at the newest grid anchors below it, the token
ids, the `taint` line and the session key. These are the bytes that were in the engine, not a
cold prefill's: restored, the next turn resumes exactly where it would have resumed had the rows
never left the buffer, and the resume is held to the gate it was held to there (bit equality with
a cold prefill below `taint`, the greedy gate above it; docs/exactness.md).

WHEN IT IS WRITTEN (engine/cache.py `ResidentPrefix.begin`, server/app.py):
  * a planned stop (SIGTERM, the drain): the resident conversation and the parked one, after the
    work in flight has finished, before the process exits; the next start reads the resident one
    back before it serves (`restore_drained`), and the parked one on demand;
  * an eviction: a request of another conversation that is not a guest is about to write over
    a resident prompt of at least `min_tokens` whose rows above the shared head the RAM park
    cannot hold; and a session snapshot from the state store, which clears the resident prefix.
    The rows go to the disk before the new prefill writes a row (the request waits for it).

WHEN IT IS READ: a request whose prompt shares more with an entry than with every cache in RAM
(the rows in place, the end anchor, the pin, the park) by more than the read costs: before the
handler's replay of the hidden spans (so the end anchor's replay works on the restored rows), and
at the start of `cache.prefill` for the rest (images' keys, the in-context stage). A resident
conversation that the restore replaces is written first, under the same rule as an eviction.

INCREMENTAL, BY CONTENT. A conversation that is evicted again a turn later must not write 14 GB
again. The rows are written in blocks of `chunk` rows, each with two 64-bit sums of its bytes per
buffer (`block_sums`, computed where the rows are, on the device); a block whose tokens and sums
equal the entry's file is not written again, and on a restore a block whose rows in place already
hold the file's bytes is not read. The same sums are checked after every block is copied in:
a file that does not hold what was written fails the restore before the engine resumes from it.

ITS OWN I/O PATH (0.3.23-test). 0.3.22's first eviction wrote 5.20 GB in 7.23 s while
a drain of the same size had taken 0.88 s: the save took the disk tier's I/O lock and staging
buffers, and the tier's writer was writing that conversation's head (1.8 GB, fsync per file) at
the same moment. Now an entry has its own lock and its own two staging buffers, the tier's writer
waits between two of its files while a save runs (`DiskTier.spill_active`), a block's copy to the
host overlaps the write of the one before it (one writer thread, two buffers), and a block's sums
over every buffer are one device sync (`block_sums_views`), not one per buffer.

THE FINGERPRINT. An entry is the bytes of one engine build's arithmetic: it is read back only under
the same weights, arithmetic flags, grid and engine code, the caching files (engine/cache.py,
engine/disk_cache.py, this file, engine/capture_lock.py) left out, so a deploy that changes only
how state is kept still finds its sessions (`fingerprint`). The layout of every buffer (dtype,
shape, axis) is checked before a byte is read.

THE DISK. `sessions/` under the disk tier's directory, 0700; each entry a directory whose
`manifest.json` is written last and removed first, so a crash leaves an entry without one, which
the next open deletes. Its own budget (`budget_gb`, 40 GB: a 256k conversation is about 27 GB),
the tier's free-space floor (never under 50 GB), a TTL (48 h unused); least recently used first.
Token ids and engine state only, never text.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time

from engine import disk_cache as dc

FORMAT = 1
MAGIC = b"QSESPIL1"
GB = dc.GB
ALIGN = dc.ALIGN
SETTINGS = {"budget_gb": 40.0, "ttl_hours": 48.0, "min_tokens": 16384,
            "drain_min_tokens": 4096, "anchors": 4}
#: engine files that change how state is kept, not the arithmetic that writes it
NOT_ARITHMETIC = ("cache.py", "disk_cache.py", "spill.py", "capture_lock.py")
#: the drained conversation is read back at the next start only when the stop was this recent
DRAIN_FRESH_S = 6 * 3600.0


def fingerprint(root: str, extra: dict) -> str:
    """The disk tier's fingerprint without the caching files (`NOT_ARITHMETIC`)."""
    return dc.fingerprint(root, extra, skip=NOT_ARITHMETIC)


def block_sums(t, ax: int, s: int, r: int) -> list[int]:
    """Two 64-bit sums of the bytes of rows [s, s + r) of `t` along `ax`, on its device: the plain
    sum of its 32-bit words and a position-weighted sum of each innermost run's sum. Identical
    bytes give identical sums; any changed word changes the first or the second (a permutation
    of whole runs the second). Nothing is copied off the device but two integers."""
    import torch
    x = t.narrow(ax, s, r)
    try:
        v = x.view(torch.int32)
    except RuntimeError:
        v = x.contiguous().view(torch.uint8).to(torch.int32)
    rs = v.sum(dim=-1, dtype=torch.int64).reshape(-1).remainder(2147483647)
    idx = torch.arange(rs.numel(), device=rs.device, dtype=torch.int64)
    w = (idx * 2654435761 + 40503) % 2147483647 + 1
    return [int(rs.sum()), int((rs * w).sum())]


def block_sums_views(views, s: int, r: int) -> list[list[int]]:
    """`block_sums` of rows [s, s + r) of every `(tensor, axis)` in `views`, with one copy off the
    device for all of them: the same integers, computed the same way."""
    import torch
    outs = []
    for t, ax in views:
        x = t.narrow(ax, s, r)
        try:
            v = x.view(torch.int32)
        except RuntimeError:
            v = x.contiguous().view(torch.uint8).to(torch.int32)
        rs = v.sum(dim=-1, dtype=torch.int64).reshape(-1).remainder(2147483647)
        idx = torch.arange(rs.numel(), device=rs.device, dtype=torch.int64)
        w = (idx * 2654435761 + 40503) % 2147483647 + 1
        outs.append(torch.stack([rs.sum(), (rs * w).sum()]).to(views[0][0].device))
    if not outs:
        return []
    return [[int(a), int(b)] for a, b in torch.stack(outs).tolist()]


def _layout(views) -> list:
    out = []
    for t, ax in views:
        shape = list(t.shape)
        shape[ax] = 1
        out.append([str(t.dtype), shape, int(ax)])
    return out


def _common(a, b) -> int:
    n = min(len(a), len(b))
    i, step = 0, 4096
    while i < n:
        j = min(n, i + step)
        if a[i:j] != b[i:j]:
            while a[i] == b[i]:
                i += 1
            return i
        i = j
    return n


class _Shim:
    """What `ResidentPrefix.replay` reads, for an entry: its end anchor, its rows' tokens, the
    server's whitespace tokens."""

    def __init__(self, end, tokens, glue):
        self.end, self.tokens, self.glue = end, tokens, glue
        self.stats = {"end_replayed": 0}


class Entry:
    """One conversation on the disk: its manifest and its tokens."""

    def __init__(self, path: str, man: dict, tokens: list[int]):
        self.path, self.man, self.tokens = path, man, tokens

    @property
    def eid(self) -> str:
        return self.man["eid"]

    @property
    def valid(self) -> int:
        return int(self.man["valid"])

    @property
    def bytes(self) -> int:
        return int(self.man.get("bytes") or 0)

    def offer(self, keys, max_len: int, glue=frozenset(), replay: bool = False,
              mm_ok: bool = False) -> int:
        """The line a prompt would resume at after a restore of this entry: the end anchor when
        the prompt starts with all of it (the hidden spans put back when `replay`), else the
        largest saved grid anchor it shares; 0 for none."""
        c = _common(keys, self.tokens)
        end = self.man.get("end")
        if end is not None and end.get("mm") and not mm_ok:
            end = None                     # an end after images: a compaction's alone
        if end is not None and int(end["at"]) <= max_len:
            e = int(end["at"])
            if c >= e:
                return e
            if replay and end.get("hidden"):
                from engine.cache import ResidentPrefix
                shim = _Shim({"at": e, "hidden": [(int(p), list(h)) for p, h in end["hidden"]],
                              "mark": end.get("mark")}, self.tokens, glue)
                _, spans = ResidentPrefix.replay(shim, list(keys))
                if spans:
                    return e
        return max((b for b in self.man["anchors"] if b <= min(c, max_len)), default=0)


class SessionSpill:
    """The entries under `root` (see the module's docstring). `tier`: the disk tier whose staging
    buffers, I/O lock and fence this shares (engine/disk_cache.py `DiskTier`)."""

    def __init__(self, root: str, *, chunk: int, fingerprint: str, tier=None,
                 settings: dict | None = None, on_event=None, clock=time.time, free_bytes=None,
                 floor_bytes: int | None = None):
        self.root = os.path.realpath(os.path.expanduser(root))
        self.chunk = int(chunk)
        self.fp = str(fingerprint)
        self.tier = tier
        self.cfg = dict(SETTINGS)
        self.cfg.update({k: v for k, v in (settings or {}).items() if k in SETTINGS
                         and v is not None})
        self.on_event = on_event
        self.clock = clock
        self._free = free_bytes or dc._statvfs_free
        self.floor = int(floor_bytes if floor_bytes is not None else
                         (tier.floor if tier is not None else dc.FLOOR_BYTES))
        self.entries: dict[str, Entry] = {}
        self.lock = threading.RLock()
        self._own_io = threading.Lock()
        self._bufs: list = []
        self._hbufs: list = []
        self._abuf = None
        self._pool = None
        self._wpool = None
        self.read_bps = dc.READ_BPS
        self.prefill_tok_s = dc.PREFILL_TOK_S
        # what the caller's server adds to an entry (the end alignment's regions) and reads back
        self.extra_of = None
        self.on_adopt = None
        self.stats = {"saves": 0, "save_ms": 0.0, "blocks_written": 0, "blocks_kept": 0,
                      "bytes_written": 0, "restores": 0, "restore_ms": 0.0, "blocks_read": 0,
                      "blocks_in_place": 0, "bytes_read": 0, "tokens_restored": 0,
                      "failed": {}, "refused": {}, "evicted": {}, "last_save": None,
                      "last_restore": None, "swept": 0}
        self.events: list[dict] = []

    # --- settings and the directory ----------------------------------------------------------
    @property
    def budget(self) -> int:
        return int(float(self.cfg["budget_gb"]) * GB)

    @property
    def min_tokens(self) -> int:
        return int(self.cfg["min_tokens"])

    def apply(self, settings: dict) -> None:
        self.cfg.update({k: v for k, v in (settings or {}).items() if k in SETTINGS
                         and v is not None})

    def open(self) -> "SessionSpill":
        os.makedirs(self.root, mode=0o700, exist_ok=True)
        os.chmod(self.root, 0o700)
        for name in sorted(os.listdir(self.root)):
            p = os.path.join(self.root, name)
            if not os.path.isdir(p):
                continue
            try:
                with open(os.path.join(p, "manifest.json")) as f:
                    man = json.load(f)
                if man.get("format") != FORMAT or man.get("chunk") != self.chunk:
                    raise ValueError("format")
                if man.get("fp") != self.fp:
                    raise ValueError("fingerprint")
                tokens = self._read_tokens(p, int(man["valid"]))
                if _tsha(tokens) != man["tsha"]:
                    raise ValueError("tokens")
                self.entries[man["eid"]] = Entry(p, man, tokens)
            except Exception:                                      # noqa: BLE001
                shutil.rmtree(p, ignore_errors=True)
                self.stats["swept"] += 1
        self._sweep("open")
        return self

    def _read_tokens(self, path: str, n: int) -> list[int]:
        import numpy as np
        a = np.fromfile(os.path.join(path, "tokens.bin"), dtype="<i8")
        if a.size != n:
            raise ValueError("tokens")
        return a.tolist()

    def _event(self, kind: str, **detail) -> None:
        ev = {"kind": kind, "ts": round(self.clock(), 3), **detail}
        with self.lock:
            self.events.append(ev)
            del self.events[:-30]
        if self.on_event is not None:
            try:
                self.on_event(kind, detail)
            except Exception:                                      # noqa: BLE001
                pass

    def _count(self, what: str, why: str) -> None:
        d = self.stats[what]
        d[why] = d.get(why, 0) + 1

    def _bytes(self) -> int:
        return sum(e.bytes for e in self.entries.values())

    def _drop(self, eid: str, why: str) -> int:
        with self.lock:
            e = self.entries.pop(eid, None)
        if e is None:
            return 0
        try:
            os.unlink(os.path.join(e.path, "manifest.json"))
        except OSError:
            pass
        shutil.rmtree(e.path, ignore_errors=True)
        self._count("evicted", why)
        self._event("evict", why=why, tokens=e.valid, bytes=e.bytes)
        return e.bytes

    def _sweep(self, why: str, room: int = 0, protect: str | None = None) -> bool:
        """TTL first, then least recently used until `room` more bytes fit the budget and leave
        the floor free. False when they cannot."""
        now = self.clock()
        ttl = float(self.cfg["ttl_hours"]) * 3600.0
        for eid, e in list(self.entries.items()):
            if eid != protect and now - float(e.man.get("used", 0)) > ttl:
                self._drop(eid, "ttl")
        while True:
            over = self._bytes() + room > self.budget
            low = self._free(self.root) - room < self.floor
            if not (over or low):
                return True
            olds = sorted((float(e.man.get("used", 0)), eid) for eid, e in self.entries.items()
                          if eid != protect)
            if not olds:
                return False
            self._drop(olds[0][1], "budget" if over else "floor")

    def clear(self) -> int:
        n = 0
        for eid in list(self.entries):
            n += self._drop(eid, "clear")
        return n

    # --- the staging buffers ------------------------------------------------------------------
    def _io(self):
        """This object's own lock (0.3.23-test): a save or a restore never waits for the tier."""
        return self._own_io

    def _tier_hold(self, d: int) -> None:
        """While a save runs the tier's writer waits between two of its files."""
        if self.tier is not None and hasattr(self.tier, "spill_active"):
            with self.lock:
                self.tier.spill_active = max(0, int(self.tier.spill_active) + d)

    def _writer(self):
        if self._wpool is None:
            from concurrent.futures import ThreadPoolExecutor
            self._wpool = ThreadPoolExecutor(1, thread_name_prefix="spill-write")
        return self._wpool

    def _staging(self, views):
        """Two block buffers of `chunk` rows of every view, this object's own (0.3.23-test: never
        the tier's chunk slots, which its writer fills at the same time)."""
        payload = dc._sections([dc._row_bytes(t, ax) * self.chunk for t, ax in views])[1]
        if not self._bufs or self._bufs[0].nbytes < payload:
            for b in self._bufs:
                b.close()
            cuda = any(t.is_cuda for t, _ in views)
            self._bufs = [dc._Buf(payload, cuda) for _ in range(2)]
        return self._bufs

    def _hdr(self, i: int = 0):
        if not self._hbufs:
            self._hbufs = [dc._Buf(ALIGN, False) for _ in range(2)]
        return self._hbufs[i]

    def _anchor_buf(self, eng):
        need = dc._sections([dc._nb(eng.state.S), dc._nb(eng.state.conv)])[1]
        if self._abuf is None or self._abuf.nbytes < need:
            if self._abuf is not None:
                self._abuf.close()
            self._abuf = dc._Buf(need, bool(eng.state.S.is_cuda))
        return self._abuf

    def _reader(self):
        if self._pool is None:
            from concurrent.futures import ThreadPoolExecutor
            self._pool = ThreadPoolExecutor(1, thread_name_prefix="spill-read")
        return self._pool

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        if self._wpool is not None:
            self._wpool.shutdown(wait=True)
            self._wpool = None
        for b in self._bufs + self._hbufs:
            b.close()
        self._bufs, self._hbufs = [], []
        if self._abuf is not None:
            self._abuf.close()
            self._abuf = None

    @staticmethod
    def _put_header(buf, meta: dict):
        raw = json.dumps(meta, separators=(",", ":")).encode()
        if len(raw) > ALIGN - 12:
            raise ValueError(f"a spill header of {len(raw)} bytes")
        hdr = buf.view(0, ALIGN)
        hdr[:ALIGN] = b"\0" * ALIGN
        hdr[:8] = MAGIC
        hdr[8:12] = len(raw).to_bytes(4, "little")
        hdr[12:12 + len(raw)] = raw
        return hdr

    @staticmethod
    def _get_header(buf) -> dict:
        if bytes(buf.view(0, 8)) != MAGIC:
            raise ValueError("magic")
        k = int.from_bytes(bytes(buf.view(8, 4)), "little")
        return json.loads(bytes(buf.view(12, k)).decode())

    # --- writing ------------------------------------------------------------------------------
    def _eid_for(self, owner: str | None, tokens: list[int]) -> str:
        with self.lock:
            if owner is not None:
                for e in self.entries.values():
                    if e.man.get("owner") == owner:
                        return e.eid
            best, eid = 4 * self.chunk, None
            for e in self.entries.values():
                if e.man.get("owner") != owner:
                    continue
                c = _common(tokens, e.tokens)
                if c >= best:
                    best, eid = c, e.eid
        if eid is not None:
            return eid
        h = hashlib.sha256(f"{owner}|{self.clock()}|{tokens[:64]}".encode()).hexdigest()
        return h[:16]

    def save_resident(self, res, eng, drafter, reason: str) -> dict | None:
        """The resident conversation: rows [0, valid), the end anchor and the newest anchors."""
        if res.valid <= 0 or res._stash is not None:
            return None
        n = int(res.valid)
        anchors = {b: pair for b, pair in res.anchors.items() if b <= n}
        end = None
        if res.end is not None and res._end_bufs is not None and int(res.end["at"]) == n:
            end = {"at": n, "hidden": [[int(p), [int(x) for x in h]] for p, h in res.end["hidden"]],
                   "mark": res.end.get("mark"), "pair": res._end_bufs}
            if res.end.get("mm") or res.end.get("delta"):
                end.update(delta=int(res.end.get("delta") or 0), mm=bool(res.end.get("mm")))
        if end is None and not anchors:
            return None
        return self._save(eng, drafter, tokens=res.tokens[:n], n=n, rows_from=None,
                          anchors=anchors, end=end, taint=res.taint, owner=res.owner,
                          reason=reason, extra=self._extra(res) if end is not None else None,
                          resident=True)

    def save_park(self, res, eng, drafter, reason: str) -> dict | None:
        """The parked conversation: its head is the rows in place (the resident prompt shares it),
        the rest the park's buffers."""
        if not res.park_rows or not res.park_anchors or res._park_bufs is None:
            return None
        base, top = int(res.park_base), int(res.park_base + res.park_rows)
        if res.common(res.park_tokens[:base]) < base or len(res.park_tokens) < top:
            return None
        from engine.cache import _kv_views
        views = _kv_views(eng, drafter)
        if len(views) != len(res._park_bufs):
            return None

        def rows_from(i, s, r):
            t, ax = views[i]
            out = []
            if s < base:
                out.append((0, t.narrow(ax, s, min(r, base - s))))
            if s + r > base:
                a = max(s, base)
                out.append((a - s, res._park_bufs[i].narrow(ax, a - base, s + r - a)))
            return out
        return self._save(eng, drafter, tokens=res.park_tokens[:top], n=top, rows_from=rows_from,
                          anchors=dict(res.park_anchors), end=None, taint=res.park_taint,
                          owner=res.park_owner, reason=reason, extra=None, resident=False)

    def _extra(self, res) -> dict | None:
        if self.extra_of is None:
            return None
        try:
            return self.extra_of(res)
        except Exception:                                          # noqa: BLE001
            return None

    def _save(self, eng, drafter, *, tokens, n, rows_from, anchors, end, taint, owner, reason,
              extra, resident) -> dict | None:
        import torch
        from engine.cache import _kv_views
        t0 = time.perf_counter()
        C = self.chunk
        tokens = [int(x) for x in tokens[:n]]
        views = _kv_views(eng, drafter)
        layout = _layout(views)
        keep = sorted(anchors)[-int(self.cfg["anchors"]):]
        per_row = sum(dc._row_bytes(t, ax) for t, ax in views)
        S, conv = eng.state.S, eng.state.conv
        (oS, nS), (oc, nc) = dc._sections([dc._nb(S), dc._nb(conv)])[0]
        anc_size = ALIGN + dc.align(oc + nc)
        eid = self._eid_for(owner, tokens)
        old = self.entries.get(eid)
        want = (n * per_row + -(-n // C) * 2 * ALIGN
                + (len(keep) + (end is not None)) * anc_size)
        if not self._sweep("save", room=max(0, want - (old.bytes if old else 0)), protect=eid):
            self._count("refused", "room")
            self._event("refused", why="room", tokens=n, reason=reason)
            return None
        path = os.path.join(self.root, eid)
        os.makedirs(path, mode=0o700, exist_ok=True)
        # the entry is incomplete from here until its new manifest is in place
        dc._unlink(os.path.join(path, "manifest.json"))
        with self.lock:
            self.entries.pop(eid, None)
        same = old is not None and old.man.get("layout") == layout
        old_blocks = {int(b["b"]): b for b in old.man["blocks"]} if same else {}
        cuda = any(t.is_cuda for t, _ in views)
        blocks, written, kept, nbytes, wbytes = [], 0, 0, 0, 0
        carried = None
        # 0.3.24-test: where a save's time goes, in ms (`info["parts"]`; the 0.3.23 live case of
        # 7.0 GB in 4.5 s had no breakdown)
        tm = {"prep": (time.perf_counter() - t0) * 1e3, "lock": 0.0, "alloc": 0.0, "sums": 0.0,
              "copy": 0.0, "wait": 0.0, "tail": 0.0, "anchors": 0.0, "fsync": 0.0,
              "manifest": 0.0}
        tw = [0.0]                             # the writer thread's own time in write_file

        def _write(p, views_, fs):
            a = time.perf_counter()
            try:
                return dc.write_file(p, views_, fs)
            finally:
                tw[0] += time.perf_counter() - a
        try:
            q = time.perf_counter()
            with self._io():
                tm["lock"] = (time.perf_counter() - q) * 1e3
                self._tier_hold(+1)
                pending: dict = {}             # staging slot -> the write still using it
                try:
                    bufs = None
                    for b in range(0, n, C):
                        r = min(C, n - b)
                        fname = f"r{b:07d}.blk"
                        pieces = None if rows_from is None else [rows_from(i, b, r)
                                                                 for i in range(len(views))]
                        split = pieces is not None and any(len(p) > 1 for p in pieces)
                        sums = None
                        q = time.perf_counter()
                        if not split:
                            sums = (block_sums_views(views, b, r) if pieces is None else
                                    block_sums_views([(pieces[i][0][1], ax) for i, (t, ax)
                                                      in enumerate(views)], 0, r))
                        tm["sums"] += (time.perf_counter() - q) * 1e3
                        ob = old_blocks.get(b)
                        if (sums is not None and ob is not None and ob["rows"] == r
                                and ob["sums"] == sums and old.tokens[b:b + r] == tokens[b:b + r]
                                and os.path.exists(os.path.join(path, fname))):
                            blocks.append(ob)
                            kept += 1
                            nbytes += int(ob["bytes"])
                            continue
                        if bufs is None:
                            q = time.perf_counter()
                            bufs = self._staging(views)
                            tm["alloc"] += (time.perf_counter() - q) * 1e3
                        slot = written % 2
                        f = pending.pop(slot, None)
                        if f is not None:
                            q = time.perf_counter()
                            f.result()             # the write that used this buffer is done
                            tm["wait"] += (time.perf_counter() - q) * 1e3
                        q = time.perf_counter()
                        secs, payload = dc._sections([dc._row_bytes(t, ax) * r for t, ax in views])
                        buf = bufs[slot]
                        dsts = []
                        for i, ((t, ax), (off, nb)) in enumerate(zip(views, secs)):
                            dst = buf.typed(off, t.dtype, dc._row_shape(t, ax, r))
                            if pieces is None:
                                dst.copy_(t.narrow(ax, b, r), non_blocking=cuda and buf.pinned)
                            else:
                                for at, x in pieces[i]:
                                    dst.narrow(ax, at, x.shape[ax]).copy_(
                                        x, non_blocking=cuda and buf.pinned)
                            dsts.append((dst, ax))
                        if cuda:
                            torch.cuda.current_stream().synchronize()
                        tm["copy"] += (time.perf_counter() - q) * 1e3
                        if sums is None:
                            # a block in two pieces: its sums are the whole block's, as a restore
                            # reads them in place (on the host here, once)
                            sums = [block_sums(dst, ax, 0, r) for dst, ax in dsts]
                        hdr = self._put_header(self._hdr(slot), {"b": b, "rows": r, "sums": sums,
                                                                 "sections": secs})
                        size = ALIGN + payload
                        pending[slot] = self._writer().submit(
                            _write, os.path.join(path, fname), [hdr, buf.view(0, payload)], False)
                        blocks.append({"b": b, "rows": r, "sums": sums, "bytes": size})
                        written += 1
                        nbytes += size
                        wbytes += size
                    q = time.perf_counter()
                    for f in list(pending.values()):
                        f.result()
                    pending = {}
                    tm["tail"] += (time.perf_counter() - q) * 1e3
                finally:
                    for f in list(pending.values()):
                        try:
                            f.result()
                        except Exception:                          # noqa: BLE001
                            pass
                    self._tier_hold(-1)
                # the anchors: the newest grid lines' and the end's, one file each
                q = time.perf_counter()
                abuf = self._anchor_buf(eng)
                files = []
                pairs = [(f"a{b:07d}.anc", b, anchors[b]) for b in keep]
                if end is not None:
                    pairs.append(("end.anc", int(end["at"]), end["pair"]))
                old_files = ({f["file"]: f for f in old.man.get("anchor_files") or []}
                             if same else {})
                # 0.3.24-test: an end anchor already on the disk stays when the rows under it did
                # not move (every block kept, the same tokens) and the engine no longer holds it
                # (a cache level gave its buffers back, or the request that evicts this
                # conversation dropped it first): its state is still the state at those rows
                oe = old.man.get("end") if same else None
                if (end is None and oe is not None and int(oe["at"]) == n and written == 0
                        and old.valid == n and old.tokens[:n] == tokens
                        and "end.anc" in old_files
                        and os.path.exists(os.path.join(path, "end.anc"))):
                    carried = dict(oe)
                if carried is not None:
                    files.append(old_files["end.anc"])
                    nbytes += anc_size
                for fname, b, (aS, ac) in pairs:
                    sums = [state_sums(aS), state_sums(ac)]
                    of = old_files.get(fname)
                    if (of is not None and of["b"] == b and of["sums"] == sums
                            and os.path.exists(os.path.join(path, fname))):
                        files.append(of)           # 0.3.23-test: an anchor already on the disk
                        nbytes += anc_size
                        continue
                    abuf.typed(oS, S.dtype, S.shape).copy_(aS, non_blocking=cuda and abuf.pinned)
                    abuf.typed(oc, conv.dtype, conv.shape).copy_(ac,
                                                                 non_blocking=cuda and abuf.pinned)
                    if cuda:
                        torch.cuda.current_stream().synchronize()
                    hdr = self._put_header(self._hdr(), {"b": b, "sums": sums})
                    dc.write_file(os.path.join(path, fname),
                                  [hdr, abuf.view(0, dc.align(oc + nc))], fsync=False)
                    files.append({"file": fname, "b": b, "sums": sums})
                    nbytes += anc_size
                    wbytes += anc_size
            tm["anchors"] = (time.perf_counter() - q) * 1e3
            q = time.perf_counter()
            names = {f"r{bl['b']:07d}.blk" for bl in blocks} | {f["file"] for f in files}
            for name in os.listdir(path):
                if name not in names:
                    dc._unlink(os.path.join(path, name))
            import numpy as np
            np.asarray(tokens, dtype="<i8").tofile(os.path.join(path, "tokens.bin"))
            _fsync_dir_files(path)
            tm["fsync"] = (time.perf_counter() - q) * 1e3
            q = time.perf_counter()
            now = self.clock()
            man = {"format": FORMAT, "eid": eid, "fp": self.fp, "chunk": C, "valid": n,
                   "tsha": _tsha(tokens), "owner": owner, "reason": reason, "resident": resident,
                   "created": round(now, 3), "used": round(now, 3), "taint": taint,
                   "layout": layout,
                   "state": [[str(S.dtype), list(S.shape)], [str(conv.dtype), list(conv.shape)]],
                   "anchors": [int(b) for b in keep], "anchor_files": files,
                   "end": ({"at": int(end["at"]), "hidden": end["hidden"], "mark": end["mark"],
                            **({"delta": end["delta"], "mm": end["mm"]} if "mm" in end else {})}
                           if end is not None else carried),
                   "blocks": blocks, "bytes": nbytes,
                   "extra": extra if carried is None else old.man.get("extra")}
            self._write_manifest(path, man)
            tm["manifest"] = (time.perf_counter() - q) * 1e3
        except Exception as exc:                                   # noqa: BLE001
            shutil.rmtree(path, ignore_errors=True)
            self._count("failed", "save")
            self._event("failed", what="save", error=f"{type(exc).__name__}: {exc}"[:200])
            return None
        with self.lock:
            self.entries[eid] = Entry(path, man, tokens)
        self._sweep("save", protect=eid)           # the budget against the bytes written
        s = time.perf_counter() - t0
        info = {"eid": eid, "reason": reason, "tokens": n, "blocks_written": written,
                "blocks_kept": kept, "bytes_written": wbytes, "bytes": nbytes,
                "ms": round(s * 1e3, 1), "end": end is not None or carried is not None,
                "anchors": len(keep), **({"end_carried": True} if carried is not None else {}),
                "parts": {k: round(v, 1) for k, v in tm.items()}, "writer_ms": round(tw[0] * 1e3, 1)}
        self.stats["saves"] += 1
        self.stats["save_ms"] += s * 1e3
        self.stats["blocks_written"] += written
        self.stats["blocks_kept"] += kept
        self.stats["bytes_written"] += wbytes
        self.stats["last_save"] = info
        self._event("save", **info)
        return info

    def _write_manifest(self, path: str, man: dict) -> None:
        tmp = os.path.join(path, "manifest.json.tmp")
        with open(tmp, "w") as f:
            json.dump(man, f, separators=(",", ":"))
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, os.path.join(path, "manifest.json"))
        _fsync_dir(path)

    # --- reading ------------------------------------------------------------------------------
    def candidate(self, res, keys, max_len: int, replay: bool = False, mm_ok: bool = False):
        """(entry, line, the line the caches in RAM offer) for the entry a prompt should be served
        from, or None: it must beat them by more than a chunk and by more than its read costs."""
        if not self.entries:
            return None
        c = res.common(keys)
        here = max(res.best(c, max_len), res.end_offer(c, max_len, mm_ok=mm_ok),
                   res.pin_offer(keys, c, max_len), res.park_offer(keys, c, max_len))
        if replay and res.end is not None and res.end.get("hidden") and here < res.end["at"] \
                and res.end["at"] <= max_len:
            from engine.cache import ResidentPrefix
            if ResidentPrefix.replay(_Shim(res.end, res.tokens, res.glue), list(keys))[1]:
                here = int(res.end["at"])
        best = None
        with self.lock:
            entries = list(self.entries.values())
        for e in entries:
            if e.valid <= here + self.chunk:
                continue
            line = e.offer(keys, max_len, res.glue, replay, mm_ok=mm_ok)
            if line <= here + self.chunk:
                continue
            need = sum(int(b["bytes"]) for b in e.man["blocks"] if b["b"] + b["rows"] > c)
            if need / self.read_bps >= (line - here) / self.prefill_tok_s:
                self._count("refused", "slow")
                continue
            if best is None or line > best[1]:
                best = (e, line, here)
        return best

    def maybe_restore(self, res, eng, drafter, keys, max_len: int, replay: bool = False,
                      disk=None, mm_ok: bool = False) -> dict | None:
        """Serve `keys` from an entry when one beats the caches in RAM: the resident conversation
        written first when it is worth keeping, then the entry's rows read in place."""
        if res is None or not self.entries:
            return None
        if res._stash is not None:
            if disk is not None:
                cache_fence(disk, 0)
            res.settle_guest(eng)
        got = self.candidate(res, keys, max_len, replay, mm_ok=mm_ok)
        if got is None:
            return None
        e, line, here = got
        if (res.valid >= self.min_tokens and (res.owner is None or res.owner != e.man.get("owner"))
                and _common(res.tokens[:res.valid], e.tokens) < res.valid):
            self.save_resident(res, eng, drafter, "switch")
        return self.restore(res, eng, drafter, e, disk=disk, line=line, here=here)

    def restore(self, res, eng, drafter, e: Entry, disk=None, line: int = 0,
                here: int = 0) -> dict | None:
        """Read entry `e` into the engine's buffers and make it the resident prompt (its anchors,
        its end anchor, its line and its session). None, with the resident prefix cut below the
        first row written, when a file fails its check or a buffer is refused."""
        import torch
        from engine.cache import _kv_views
        t0 = time.perf_counter()
        man = e.man
        n, C = e.valid, self.chunk
        S, conv = eng.state.S, eng.state.conv
        if man["state"] != [[str(S.dtype), list(S.shape)], [str(conv.dtype), list(conv.shape)]]:
            return self._failed(e, "layout", res, None)
        if drafter is not None and hasattr(drafter, "state_resume"):
            drafter.state_resume(n)            # builds a draft cache that does not exist yet
        views = _kv_views(eng, drafter)
        if _layout(views) != man["layout"] or n > min(t.shape[ax] for t, ax in views):
            return self._failed(e, "layout", res, None)
        shared = _common(res.tokens[:res.valid], e.tokens) if res._stash is None else 0
        if disk is not None:
            cache_fence(disk, 0)
        low = None                              # the first row this restore writes
        nread = in_place = 0
        cuda = any(t.is_cuda for t, _ in views)
        try:
            with self._io():
                todo = []
                for bl in man["blocks"]:
                    b, r = int(bl["b"]), int(bl["rows"])
                    # rows in place that hold the file's bytes already (the prompt they share)
                    if b + r <= shared and block_sums_views(views, b, r) == bl["sums"]:
                        in_place += 1
                        continue
                    todo.append(bl)
                if todo:
                    bufs = self._staging(views)
                    pool = self._reader()
                    futs = {}

                    def submit(j):
                        if j < len(todo):
                            bl = todo[j]
                            p = os.path.join(e.path, f"r{int(bl['b']):07d}.blk")
                            size = int(bl["bytes"]) - ALIGN
                            futs[j] = (pool.submit(dc.read_into, p, 0, self._hdr(j % 2).view(0, ALIGN)),
                                       pool.submit(dc.read_into, p, ALIGN,
                                                   bufs[j % 2].view(0, size)))
                    submit(0)
                    submit(1)
                    for j, bl in enumerate(todo):
                        b, r = int(bl["b"]), int(bl["rows"])
                        buf = bufs[j % 2]
                        try:
                            fh, fb = futs.pop(j)
                            nread += fh.result() + fb.result()
                            meta = self._get_header(self._hdr(j % 2))
                            if meta["b"] != b or meta["rows"] != r:
                                raise ValueError("block")
                        except Exception:                          # noqa: BLE001
                            _cancel(futs)
                            return self._failed(e, "read", res, low)
                        low = b if low is None else min(low, b)
                        for (t, ax), (off, nb) in zip(views, meta["sections"]):
                            t.narrow(ax, b, r).copy_(
                                buf.typed(off, t.dtype, dc._row_shape(t, ax, r)),
                                non_blocking=cuda and buf.pinned)
                        if cuda:
                            torch.cuda.current_stream().synchronize()
                        submit(j + 2)
                        if block_sums_views(views, b, r) != bl["sums"]:
                            _cancel(futs)
                            return self._failed(e, "sums", res, low)
                end = None
                if man.get("end") is not None:
                    end = {"at": int(man["end"]["at"]),
                           "hidden": [(int(p), [int(x) for x in h])
                                      for p, h in man["end"]["hidden"]],
                           "mark": man["end"].get("mark"),
                           **({"delta": int(man["end"].get("delta") or 0), "mm": True}
                              if man["end"].get("mm") else {})}
                res.adopt(e.tokens, n, man.get("taint"), man.get("owner"))
                low = 0 if low is None else low
                abuf = self._anchor_buf(eng)
                (oS, nS), (oc, nc) = dc._sections([dc._nb(S), dc._nb(conv)])[0]
                for f in man["anchor_files"]:
                    p = os.path.join(e.path, f["file"])
                    try:
                        nread += dc.read_into(p, ALIGN, abuf.view(0, dc.align(oc + nc)))
                    except Exception:                              # noqa: BLE001
                        return self._failed(e, "read", res, low)
                    if f["file"] == "end.anc":
                        pair = res.adopt_end(eng, end) if end is not None else None
                    else:
                        pair = res.adopt_anchor(eng, int(f["b"]))
                    if pair is None:
                        continue                   # the memory policy refused it: not kept
                    nb = cuda and abuf.pinned
                    pair[0].copy_(abuf.typed(oS, S.dtype, S.shape), non_blocking=nb)
                    pair[1].copy_(abuf.typed(oc, conv.dtype, conv.shape), non_blocking=nb)
                    if cuda:
                        torch.cuda.current_stream().synchronize()
                    if [state_sums(pair[0]), state_sums(pair[1])] != f["sums"]:
                        return self._failed(e, "sums", res, low)
        except Exception as exc:                                   # noqa: BLE001
            self._event("failed", what="restore", error=f"{type(exc).__name__}: {exc}"[:200])
            return self._failed(e, "error", res, low)
        if res.end is not None and self.on_adopt is not None and man.get("extra"):
            try:
                self.on_adopt(res, man["extra"])
            except Exception:                                      # noqa: BLE001
                pass
        man["used"] = round(self.clock(), 3)
        s = time.perf_counter() - t0
        info = {"eid": e.eid, "tokens": n, "line": line, "before": here,
                "blocks_read": len(man["blocks"]) - in_place, "blocks_in_place": in_place,
                "bytes": nread, "ms": round(s * 1e3, 1), "end": res.end is not None,
                "anchors": sorted(res.anchors), "reason": man.get("reason")}
        self.stats["restores"] += 1
        self.stats["restore_ms"] += s * 1e3
        self.stats["blocks_read"] += info["blocks_read"]
        self.stats["blocks_in_place"] += in_place
        self.stats["bytes_read"] += nread
        self.stats["tokens_restored"] += max(0, line - here)
        self.stats["last_restore"] = info
        if s > 0 and nread >= (1 << 26):
            self.read_bps = 0.5 * self.read_bps + 0.5 * nread / s
        try:
            self._write_manifest(e.path, man)
        except OSError:
            pass
        self._event("restore", **{k: v for k, v in info.items() if k != "anchors"})
        return info

    def _failed(self, e: Entry, why: str, res, low) -> None:
        """A restore that did not happen. The entry goes when its files are bad; the rows this
        restore wrote over are not the resident prompt's any more."""
        self._count("failed", why)
        if low is not None and res is not None:
            if low <= 0:
                res.clear()
            else:
                res._cut(low)
        if why in ("read", "sums"):
            self._drop(e.eid, "corrupt")
        self._event("restore_failed", why=why, tokens=e.valid)
        return None

    def drained(self) -> Entry | None:
        """The resident conversation a planned stop wrote, when the stop was recent."""
        now = self.clock()
        best = None
        with self.lock:
            for e in self.entries.values():
                if (e.man.get("reason") == "drain" and e.man.get("resident")
                        and now - float(e.man["created"]) < DRAIN_FRESH_S):
                    if best is None or e.man["created"] > best.man["created"]:
                        best = e
        return best

    def report(self) -> dict:
        with self.lock:
            rows = [{"tokens": e.valid, "bytes": e.bytes, "reason": e.man.get("reason"),
                     "created": e.man.get("created"), "used": e.man.get("used"),
                     "end": e.man.get("end") is not None, "owner": _short(e.man.get("owner"))}
                    for e in self.entries.values()]
        rows.sort(key=lambda r: -float(r["used"] or 0))
        return {"root": self.root, "settings": dict(self.cfg), "entries": len(rows),
                "bytes": sum(r["bytes"] for r in rows), "budget": self.budget,
                "fingerprint": self.fp[:16], "read_bps": round(self.read_bps),
                "list": rows[:10], "events": self.events[-10:],
                **{k: (dict(v) if isinstance(v, dict) else v) for k, v in self.stats.items()}}


def state_sums(x) -> list[int]:
    """`block_sums` of a recurrent state or convolution tail, as rows of 64 values."""
    flat = x.reshape(-1)
    if flat.numel() % 64 == 0:
        flat = flat.view(-1, 64)
    else:
        flat = flat.view(1, -1)
    return block_sums(flat, 0, 0, flat.shape[0])


def _tsha(tokens: list[int]) -> str:
    return hashlib.sha256(json.dumps(tokens).encode()).hexdigest()[:16]


def _short(s):
    return None if s is None else (s if len(s) <= 12 else s[:12] + "…")


def _cancel(futs: dict) -> None:
    for pair in futs.values():
        for f in (pair if isinstance(pair, tuple) else (pair,)):
            f.cancel()
            try:
                f.result()
            except Exception:                                      # noqa: BLE001
                pass


def cache_fence(disk, safe: int) -> None:
    try:
        disk.fence(safe)
    except Exception:                                              # noqa: BLE001
        pass


def _fsync_dir_files(path: str) -> None:
    for name in os.listdir(path):
        p = os.path.join(path, name)
        try:
            fd = os.open(p, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            pass


def _fsync_dir(path: str) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass
