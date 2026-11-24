"""Local HTTP server (standard library only) serving the GUI and its JSON API."""
import json
import mimetypes
import os
import secrets
import shutil
import tempfile
import threading
import traceback
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from .session import JobFailed, Session, UserError

STATIC_DIR = Path(__file__).parent / "static"
API = {}


def api(name):
    """Register an API endpoint: fn(session, params) -> JSON-able data."""
    def deco(fn):
        API[name] = fn
        return fn
    return deco


def find_plotly_js():
    """Locate plotly.min.js shipped with the plotly Python package (a wawi dependency)."""
    env = os.environ.get("WAWI_GUI_PLOTLY_JS")
    if env and Path(env).is_file():
        return Path(env)
    try:
        import importlib.util
        spec = importlib.util.find_spec("plotly")
        if spec and spec.submodule_search_locations:
            for loc in spec.submodule_search_locations:
                p = Path(loc) / "package_data" / "plotly.min.js"
                if p.is_file():
                    return p
    except Exception:
        pass
    local = STATIC_DIR / "plotly.min.js"
    return local if local.is_file() else None


class _Handler(BaseHTTPRequestHandler):
    server_version = "wawi-gui"
    protocol_version = "HTTP/1.1"

    # ---------------------------------------------------------------- helpers
    def log_message(self, fmt, *args):  # quieter console
        if self.server.verbose:
            super().log_message(fmt, *args)

    def _host_ok(self):
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]").lower()
        return host in self.server.allowed_hosts

    def _token_ok(self, query=None):
        tok = self.headers.get("X-Token")
        if tok is None and query is not None:
            tok = (query.get("token") or [None])[0]
        return tok is not None and secrets.compare_digest(tok, self.server.token)

    def _send(self, code, body, ctype="application/json", extra_headers=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, allow_nan=False).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (extra_headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path, ctype=None, download_name=None):
        path = Path(path)
        ctype = ctype or mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        size = path.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(size))
        if download_name:
            self.send_header("Content-Disposition", f'attachment; filename="{download_name}"')
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        with open(path, "rb") as f:
            shutil.copyfileobj(f, self.wfile)

    # ---------------------------------------------------------------- GET
    def do_GET(self):
        if not self._host_ok():
            return self._send(403, {"ok": False, "error": "Host not allowed"})
        url = urlparse(self.path)
        q = parse_qs(url.query)
        path = unquote(url.path)
        if path in ("/", "/index.html"):
            return self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
        if path == "/plotly.js":
            p = find_plotly_js()
            if p is None:
                return self._send(404, "plotly.js not found locally", "text/plain")
            return self._send_file(p, "application/javascript")
        if path.startswith("/static/"):
            rel = path[len("/static/"):]
            target = (STATIC_DIR / rel).resolve()
            if STATIC_DIR.resolve() not in target.parents or not target.is_file():
                return self._send(404, "not found", "text/plain")
            return self._send_file(target)
        if path.startswith("/download/"):
            if not self._token_ok(q):
                return self._send(403, {"ok": False, "error": "Invalid token"})
            did = path[len("/download/"):]
            item = self.server.session.downloads.get(did)
            if not item:
                return self._send(404, "not found", "text/plain")
            return self._send_file(item["path"], download_name=item["name"])
        return self._send(404, "not found", "text/plain")

    # ---------------------------------------------------------------- POST
    def do_POST(self):
        if not self._host_ok():
            return self._send(403, {"ok": False, "error": "Host not allowed"})
        url = urlparse(self.path)
        q = parse_qs(url.query)
        if not self._token_ok():
            return self._send(403, {"ok": False, "error": "Invalid or missing session token. "
                                    "Open the URL printed in the terminal (it contains ?token=...)."})
        length = int(self.headers.get("Content-Length") or 0)
        if url.path == "/upload":
            return self._upload(q, length)
        if not url.path.startswith("/api/"):
            return self._send(404, {"ok": False, "error": "unknown endpoint"})
        name = url.path[len("/api/"):]
        fn = API.get(name)
        if fn is None:
            return self._send(404, {"ok": False, "error": f"Unknown API endpoint {name!r}"})
        try:
            raw = self.rfile.read(length) if length else b"{}"
            params = json.loads(raw or b"{}")
        except Exception as exc:
            return self._send(400, {"ok": False, "error": f"Bad JSON: {exc}"})
        try:
            data = fn(self.server.session, params)
            payload = {"ok": True, "data": data}
            try:
                body = json.dumps(payload, allow_nan=False)
            except ValueError:
                from .serialize import jsonable
                body = json.dumps({"ok": True, "data": jsonable(data)}, allow_nan=False)
            return self._send(200, body)
        except UserError as exc:
            return self._send(200, {"ok": False, "error": str(exc)})
        except JobFailed as exc:
            return self._send(200, {"ok": False, "error": exc.job.error, "traceback": exc.job.traceback,
                                    "job": exc.job.id, "log": exc.job.log.text()})
        except Exception as exc:
            return self._send(200, {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                                    "traceback": traceback.format_exc()})

    def _upload(self, q, length):
        """Raw-body upload. ?dir=<id or empty>&path=<relative path>."""
        sess = self.server.session
        rel = (q.get("path") or ["upload.bin"])[0].replace("\\", "/")
        parts = [p for p in rel.split("/") if p not in ("", ".", "..")]
        if not parts:
            return self._send(400, {"ok": False, "error": "Invalid file name"})
        dir_id = (q.get("dir") or [""])[0]
        if dir_id and dir_id in sess.uploads:
            base = Path(sess.uploads[dir_id])
        else:
            base = Path(tempfile.mkdtemp(prefix="wawi_gui_upload_"))
            dir_id = secrets.token_hex(6)
            sess.uploads[dir_id] = str(base)
        target = base.joinpath(*parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        remaining = length
        with open(target, "wb") as f:
            while remaining > 0:
                chunk = self.rfile.read(min(1 << 20, remaining))
                if not chunk:
                    break
                f.write(chunk)
                remaining -= len(chunk)
        return self._send(200, {"ok": True, "data": {"dir": dir_id, "base": str(base), "path": str(target)}})


class GuiServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, session, token, allowed_hosts, verbose=False):
        super().__init__(addr, _Handler)
        self.session = session
        self.token = token
        self.allowed_hosts = allowed_hosts
        self.verbose = verbose


def serve(host="127.0.0.1", port=8050, open_browser=True, token=None, verbose=False, allow_hosts=()):
    from . import api_all  # noqa: F401  (registers endpoints)
    from .bootstrap import build_namespace

    session = Session()
    session.init_namespace(build_namespace())
    token = token or secrets.token_urlsafe(18)
    allowed = {"localhost", "127.0.0.1", "::1"} | {h.lower() for h in allow_hosts}
    if host not in ("127.0.0.1", "localhost", "::1", ""):
        allowed.add(host.lower())

    httpd = None
    for p in range(port, port + 20):
        try:
            httpd = GuiServer((host, p), session, token, allowed, verbose)
            port = p
            break
        except OSError:
            continue
    if httpd is None:
        raise SystemExit(f"Could not bind to any port in {port}-{port + 19}")

    shown_host = "localhost" if host in ("127.0.0.1", "", "0.0.0.0") else host
    url = f"http://{shown_host}:{port}/?token={token}"
    print("\n  WAWI GUI is running")
    print(f"  Open: {url}\n  (Ctrl+C to stop)\n", flush=True)
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping WAWI GUI.")
    finally:
        httpd.server_close()
    return url
