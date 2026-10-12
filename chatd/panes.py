"""The workspace API the panes read (SRV-357, Chat 5): the sandbox's state, the files, a run's
diff and its undo, artifacts and the preview. Mixed into `chatd.service.Service`.

    GET    /conversations/{id}/sandbox                    SandboxStatus
    POST   /conversations/{id}/sandbox/{start|stop|reset} SandboxStatus (reset: a new container,
                                                          the files stay)
    GET    /conversations/{id}/sandbox/files?path=        SandboxFile[] (one directory)
    GET    /conversations/{id}/sandbox/file?path=         {content, mime, bytes, etag}; ?raw=1 bytes
    PUT    /conversations/{id}/sandbox/file?path=         If-Match: etag, 412 when it moved
    DELETE /conversations/{id}/sandbox/file?path=
    POST   /conversations/{id}/sandbox/upload?dir=        multipart files, 50 MB each
    GET    /conversations/{id}/sandbox/diff?run_id=       {files: [{path, status, additions,
                                                          deletions, patch}]}
    POST   /conversations/{id}/sandbox/undo {run_id}      the run's files back as they were
    GET    /conversations/{id}/artifacts/{ref}            what a pane shows: `file:<path>`,
                                                          `diff:<run>`, `port:<n>`
    GET    /conversations/{id}/sandbox/view/<path>        a workspace file for the preview iframe,
                                                          with the preview CSP (CONTRACT 5)

Paths are resolved by chatd/workspace.py (no `..`, nothing outside /workspace, no symlink on the
way). Files are read on the host; nothing here needs the container running.
"""

from __future__ import annotations

import mimetypes
import os

from chatd.ids import ulid
from chatd.workspace import ChangedError, PathError

PREVIEW_CSP = ("default-src 'self' data: blob: https:; script-src 'self' 'unsafe-inline' "
               "'unsafe-eval' https: blob:; style-src 'self' 'unsafe-inline' https:; "
               "connect-src 'self' https: wss:; frame-ancestors 'self'")
TEXT_LIMIT = 2 * 1024 * 1024


def mime_of(path: str) -> str:
    m, _ = mimetypes.guess_type(path)
    if m:
        return m
    if path.endswith((".ts", ".tsx", ".py", ".rs", ".go", ".sh", ".toml", ".yaml", ".yml",
                      ".md", ".txt", ".log", ".cfg", ".ini", ".lock")):
        return "text/plain"
    return "application/octet-stream"


def is_text(mime: str) -> bool:
    return mime.startswith("text/") or mime in ("application/json", "application/javascript",
                                                "application/xml", "image/svg+xml")


class PanesMixin:
    def _api_error(self, status, kind, msg):
        from chatd.service import ApiError
        return ApiError(status, kind, msg)

    def _ws_for(self, cid: str):
        self.conv(cid)                       # 404 for an unknown chat
        return self.workspace(cid)

    # ------------------------------------------------------------------ the sandbox
    def sandbox_status(self, cid: str) -> dict:
        self.conv(cid)
        return self.sandboxes.status(cid)

    def sandbox_action(self, cid: str, action: str) -> dict:
        from chatd.sandbox import Unavailable
        self.conv(cid)
        if not self.sandboxes.available():
            raise self._api_error(503, "sandbox_unavailable",
                                  "the sandbox image is not installed on this box")
        try:
            if action == "start":
                why = self.sandboxes.admit() if self.sandboxes.box(cid).state != "running" else None
                if why:
                    raise self._api_error(503, "waiting_memory", why)
                self.sandboxes.ensure(cid)
            elif action == "stop":
                self.sandboxes.stop(cid, "user")
            elif action == "reset":
                self.sandboxes.remove(cid, files=False)
        except Unavailable as exc:
            raise self._api_error(503, "sandbox_unavailable", str(exc)) from None
        return self.sandboxes.status(cid)

    # ------------------------------------------------------------------ files
    def files(self, cid: str, path: str) -> list[dict]:
        ws = self._ws_for(cid)
        try:
            return ws.list(path or "/workspace")
        except FileNotFoundError:
            raise self._api_error(404, "not_found", "no such directory") from None
        except (PathError, NotADirectoryError) as exc:
            raise self._api_error(400, "bad_path", str(exc)) from None

    def file(self, cid: str, path: str) -> tuple[dict, bytes]:
        """({content?, mime, bytes, etag}, raw bytes)"""
        from chatd.workspace import etag
        ws = self._ws_for(cid)
        try:
            st = ws.stat(path)
            if os.path.isdir(ws.resolve(path)):
                raise self._api_error(400, "bad_path", "a directory")
            data = ws.read(path, limit=TEXT_LIMIT + 1)
        except PathError as exc:
            raise self._api_error(400 if "no such" not in str(exc) else 404, "bad_path",
                                  str(exc)) from None
        except FileNotFoundError:
            raise self._api_error(404, "not_found", "no such file") from None
        mime = mime_of(path)
        meta = {"mime": mime, "bytes": int(st.st_size), "etag": etag(data)
                if st.st_size <= TEXT_LIMIT else None}
        if is_text(mime) and st.st_size <= TEXT_LIMIT and b"\0" not in data[:4096]:
            meta["content"] = data.decode("utf-8", "replace")
        return meta, data

    def put_file(self, cid: str, path: str, data: bytes, if_match: str | None) -> None:
        ws = self._ws_for(cid)
        try:
            ws.write(path, data, if_match=if_match)
        except ChangedError:
            raise self._api_error(412, "changed", "the file changed on the box; reload it") \
                from None
        except PathError as exc:
            raise self._api_error(400, "bad_path", str(exc)) from None

    def delete_file(self, cid: str, path: str) -> None:
        ws = self._ws_for(cid)
        try:
            ws.delete(path)
        except (PathError, FileNotFoundError) as exc:
            raise self._api_error(400, "bad_path", str(exc)) from None

    def upload_files(self, cid: str, folder: str, content_type: str, body: bytes) -> list[dict]:
        import email.parser
        import email.policy
        ws = self._ws_for(cid)
        msg = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(
            b"Content-Type: " + content_type.encode() + b"\r\n\r\n" + body)
        out = []
        for part in msg.iter_parts() if msg.is_multipart() else []:
            name = os.path.basename(part.get_filename() or "")
            if not name:
                continue
            data = part.get_payload(decode=True) or b""
            if len(data) > 50 * 1024 * 1024:
                raise self._api_error(413, "too_large", f"{name} is over 50 MB")
            target = (folder.rstrip("/") or "/workspace") + "/" + name
            try:
                ws.write(target, data)
            except PathError as exc:
                raise self._api_error(400, "bad_path", str(exc)) from None
            full = ws.resolve(target)
            out.append({"path": ws.show(full), "kind": "file", "bytes": len(data)})
        return out

    # ------------------------------------------------------------------ diff and undo
    def diff(self, cid: str, run_id: str | None) -> dict:
        self.conv(cid)
        if not run_id:
            r = self.store.one("SELECT id FROM runs WHERE conversation_id = ? ORDER BY started "
                               "DESC LIMIT 1", (cid,))
            run_id = r["id"] if r else ""
        return {"run_id": run_id, **self.snaps.diff(cid, run_id)}

    def undo(self, cid: str, run_id: str) -> dict:
        conv = self.conv(cid)
        if conv.run is not None and conv.run.active:
            raise self._api_error(409, "run_active", "undo when the run has ended")
        try:
            n = self.snaps.undo(cid, self.workspace(cid), run_id, "undo-" + ulid())
        except ValueError as exc:
            raise self._api_error(409, "nothing_to_undo", str(exc)) from None
        self.log(f"[undo] {cid} run {run_id}: {n} file(s)")
        return {"files": n}

    # ------------------------------------------------------------------ artifacts
    def artifact(self, cid: str, ref: str) -> dict:
        """What a pane shows for `ref`; the preview loads `url` in a sandboxed iframe."""
        base = f"/v1/chat-app/conversations/{cid}"
        if ref.startswith("diff:"):
            run = ref[5:]
            d = self.diff(cid, run)
            text = "".join(f["patch"] for f in d["files"])
            return {"kind": "diff", "content": text or "No file changed in this run.\n",
                    "mime": "text/x-diff", "title": "changes", "files": d["files"]}
        if ref.startswith("port:"):
            port = int(ref[5:])
            return {"kind": "preview", "content": "", "mime": "text/html",
                    "title": f"port {port}", "url": f"{base}/sandbox/ports/{port}/"}
        path = ref[5:] if ref.startswith("file:") else ref
        meta, data = self.file(cid, path)
        mime = meta["mime"]
        rel = self.workspace(cid).rel(path)
        title = os.path.basename(rel) or "workspace"
        url = f"{base}/sandbox/view/{rel}"
        if mime.startswith("image/"):
            return {"kind": "image", "content": url, "mime": mime, "title": title, "url": url}
        if mime == "text/html":
            return {"kind": "preview", "content": meta.get("content") or "", "mime": mime,
                    "title": title, "url": url}
        if mime in ("text/markdown",) or rel.endswith(".md"):
            return {"kind": "markdown", "content": meta.get("content") or "", "mime": mime,
                    "title": title}
        return {"kind": "file", "content": meta.get("content") or "", "mime": mime,
                "title": title, "url": url}
