"""Serve the bundled Cytolume viewer against a spine directory.

The viewer is a static site that fetches ``spine/manifest.json`` relative to its
root. This server maps:

    /spine/*        -> the spine directory you built
    /vendor/hra/*   -> an optional local HRA anatomy snapshot (``anatomy=``)
    everything else -> the viewer bundled in this package
"""

from __future__ import annotations

import posixpath
import threading
import urllib.parse
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

VIEWER_DIR = Path(__file__).resolve().parent / "viewer"

# Set explicitly: on Windows, mimetypes reads the registry and can map .js to
# text/plain, which browsers refuse for module scripts.
_MIME = {
    ".js": "text/javascript",
    ".mjs": "text/javascript",
    ".css": "text/css",
    ".html": "text/html",
    ".json": "application/json",
    ".jsonld": "application/ld+json",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".gif": "image/gif",
    ".woff2": "font/woff2",
    ".glb": "model/gltf-binary",
    ".bin": "application/octet-stream",
    ".f32": "application/octet-stream",
    ".txt": "text/plain",
}


def viewer_available() -> bool:
    return (VIEWER_DIR / "index.html").exists()


class _Handler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, **_MIME}

    def __init__(self, *args, viewer: Path, spine: Path, anatomy: Path | None, quiet: bool, **kwargs):
        self._viewer = viewer
        self._spine = spine
        self._anatomy = anatomy
        self._quiet = quiet
        super().__init__(*args, directory=str(viewer), **kwargs)

    def translate_path(self, path: str) -> str:
        path = urllib.parse.unquote(path.split("?", 1)[0].split("#", 1)[0])
        path = posixpath.normpath(path)
        parts = [p for p in path.split("/") if p and p not in (".", "..")]
        if parts[:1] == ["spine"]:
            root, rest = self._spine, parts[1:]
        elif parts[:2] == ["vendor", "hra"] and self._anatomy is not None:
            root, rest = self._anatomy, parts[2:]
        else:
            root, rest = self._viewer, parts
        return str(root.joinpath(*rest)) if rest else str(root)

    def end_headers(self) -> None:
        # Spine and anatomy files are rebuilt in place; never let a browser keep stale tiles.
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, format: str, *args) -> None:  # noqa: A002 - stdlib signature
        if not self._quiet:
            super().log_message(format, *args)


class ViewerServer:
    """A running viewer. Use ``.url``, and ``.stop()`` when done (or a ``with`` block)."""

    def __init__(self, httpd: ThreadingHTTPServer, thread: threading.Thread | None):
        self._httpd = httpd
        self._thread = thread
        host, port = httpd.server_address[:2]
        self.url = f"http://{'localhost' if host in ('127.0.0.1', '0.0.0.0', '') else host}:{port}/"

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()

    def __enter__(self) -> "ViewerServer":
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def _repr_html_(self) -> str:
        return f'<iframe src="{self.url}" width="100%" height="720" style="border:0"></iframe>'

    def __repr__(self) -> str:
        return f"ViewerServer({self.url!r})"


def _make_server(spine_dir, port: int, host: str, anatomy, quiet: bool) -> ThreadingHTTPServer:
    if not viewer_available():
        raise FileNotFoundError(
            f"the bundled viewer is missing from {VIEWER_DIR}. "
            "In a source checkout, run `python scripts/bundle_viewer.py` first."
        )
    spine = Path(spine_dir).resolve()
    if not (spine / "manifest.json").exists():
        raise FileNotFoundError(f"{spine} has no manifest.json — build one with cytolume.build_spine()")
    anat = Path(anatomy).resolve() if anatomy else None
    if anat is not None and not (anat / "snapshot.json").exists():
        raise FileNotFoundError(f"{anat} is not an HRA snapshot (no snapshot.json)")
    handler = partial(_Handler, viewer=VIEWER_DIR, spine=spine, anatomy=anat, quiet=quiet)
    return ThreadingHTTPServer((host, port), handler)


def serve(
    spine_dir,
    *,
    port: int = 8765,
    host: str = "127.0.0.1",
    anatomy=None,
    open_browser: bool = True,
    quiet: bool = True,
) -> None:
    """Serve the viewer for ``spine_dir`` until interrupted (Ctrl+C)."""
    httpd = _make_server(spine_dir, port, host, anatomy, quiet)
    url = ViewerServer(httpd, None).url
    print(f"Cytolume viewer: {url}  (Ctrl+C to stop)")
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def show(spine_dir, *, port: int = 0, host: str = "127.0.0.1", anatomy=None, open_browser: bool = False, quiet: bool = True) -> ViewerServer:
    """Start the viewer in a background thread and return it.

    In Jupyter the returned object renders as an embedded viewer; elsewhere open
    ``.url``. ``port=0`` picks a free port.
    """
    httpd = _make_server(spine_dir, port, host, anatomy, quiet)
    thread = threading.Thread(target=httpd.serve_forever, name="cytolume-viewer", daemon=True)
    thread.start()
    srv = ViewerServer(httpd, thread)
    if open_browser:
        webbrowser.open(srv.url)
    return srv
