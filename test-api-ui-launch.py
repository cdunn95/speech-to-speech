#!/usr/bin/env python3
"""Standalone web UI for testing the Qwen-Clone branch endpoints.

This is a *zero-dependency* (Python standard library only) tool that starts a
small HTTP(S) server on ``0.0.0.0:8020`` so that **any device on the LAN** can
open a browser and exercise the endpoints that were introduced on the
``Qwen-Clone`` branch of the speech-to-speech realtime server.

Endpoints covered
-----------------
Audio API (mounted at ``/v1/audio`` on the realtime server):

* ``POST /v1/audio/transcriptions``  – audio -> text (supports mic recording)
* ``POST /v1/audio/translations``    – non-English audio -> English text
* ``POST /v1/audio/speech``          – text -> audio (TTS)

Voice-clone management (mounted at ``/clone`` on the realtime server):

* ``GET  /clone/list``               – list clones
* ``POST /clone/new``                – create a clone (mp3 + name + transcription)
* ``POST /clone/default``            – set the default clone voice
* ``POST /clone/delete/``            – delete a clone

How it works
------------
The browser talks to *this* UI (port 8020). The UI forwards every API call to
the real speech-to-speech server (default ``http://127.0.0.1:8765``) and
returns the response. Acting as a same-origin proxy avoids CORS issues and
keeps the real server's configuration private. Only the seven endpoints above
are reachable through the proxy (an explicit allowlist).

Microphone note
---------------
Browsers only expose ``getUserMedia`` (the microphone) in a *secure context*.
Over plain ``http://`` that is ``localhost`` only, so the mic works when you
open the UI on the machine running it. To use the mic from *another* LAN
device, start with ``--ssl`` (a self-signed certificate is generated
automatically, with SANs for this host's LAN addresses) and accept the
browser's certificate warning.

Security posture
----------------
This is a personal, LAN-scoped test tool. It binds ``0.0.0.0`` and does not
authenticate requests. It deliberately sends **no** CORS headers, so a page on
a *different* origin cannot read responses through it. If you run it on an
untrusted network, keep it to a trusted LAN or stop it when done.

Usage
-----
    python test-api-ui-launch.py
    python test-api-ui-launch.py --api-url http://192.168.1.50:8765
    python test-api-ui-launch.py --ssl                 # HTTPS (mic from any device)
    python test-api-ui-launch.py --port 8020 --host 0.0.0.0
    python test-api-ui-launch.py --api-url https://dev:8765 --insecure-upstream
"""

from __future__ import annotations

import argparse
import atexit
import http.client
import json
import logging
import os
import shutil
import socket
import ssl
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit

# ─────────────────────────────────────────────────────────────────────
# Configuration / globals
# ─────────────────────────────────────────────────────────────────────

DEFAULT_API_URL = "http://127.0.0.1:8765"
DEFAULT_PORT = 8020

# The UI reads/writes this so the target API can be changed at runtime from the
# browser without restarting the process.
STATE = {
    "api_base": DEFAULT_API_URL,
}
STATE_LOCK = threading.Lock()

# Cap the amount of request/response data the proxy will buffer in memory. The
# real API caps uploads at 25 MB; we allow headroom for the multipart envelope.
MAX_PROXY_BYTES = 32 * 1024 * 1024

# Only these endpoints are reachable through the proxy. This doubles as a
# path-traversal guard: anything else is rejected before it reaches upstream.
ALLOWED_ENDPOINTS = frozenset(
    {
        "v1/audio/transcriptions",
        "v1/audio/translations",
        "v1/audio/speech",
        "clone/list",
        "clone/new",
        "clone/default",
        "clone/delete/",
    }
)

# When True, HTTPS upstreams are connected with certificate verification
# disabled (for self-signed dev servers). Set from --insecure-upstream.
INSECURE_UPSTREAM = False

# The self-signed cert directory (created under --ssl) is removed on exit.
_CERT_TMPDIR: str | None = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("test-api-ui")


# ─────────────────────────────────────────────────────────────────────
# LAN address discovery
# ─────────────────────────────────────────────────────────────────────


def _lan_addresses() -> list[str]:
    """Best-effort list of non-loopback IPv4 addresses on this host."""
    addrs: list[str] = []

    # 1) Connect a UDP socket to a public address (no packets are sent) to find
    #    the interface the OS would use for outbound traffic.
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.5)
        s.connect(("8.8.8.8", 80))
        addrs.append(s.getsockname()[0])
        s.close()
    except OSError:
        pass

    # 2) Fall back to the local hostname's resolved addresses.
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if ip not in addrs and not ip.startswith("127."):
                addrs.append(ip)
    except OSError:
        pass

    return addrs


# ─────────────────────────────────────────────────────────────────────
# API base URL validation
# ─────────────────────────────────────────────────────────────────────


def validate_api_base(raw: str) -> str | None:
    """Normalise and validate an API base URL.

    Returns the normalised URL (scheme://host[:port][/path]) or ``None`` when it
    is not a usable http(s) URL with a valid port.
    """
    url = (raw or "").strip()
    if not url:
        return None
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return None
    try:
        port = parts.port  # raises ValueError for out-of-range ports
    except ValueError:
        return None
    if port is not None and not (0 < port < 65536):
        return None
    return url.rstrip("/")


def resolve_endpoint(raw_path: str) -> str | None:
    """Map an incoming ``/proxy/<path>`` tail to an allowed upstream endpoint.

    Returns the canonical endpoint string, or ``None`` if it is not on the
    allowlist (which also rejects path traversal and query-string tricks).
    """
    path = unquote(raw_path).lstrip("/")
    path = path.split("?", 1)[0]  # drop any client-attached query string
    if not path or path != path.strip() or ".." in path:
        return None
    return path if path in ALLOWED_ENDPOINTS else None


# ─────────────────────────────────────────────────────────────────────
# Self-signed certificate generation (for --ssl)
# ─────────────────────────────────────────────────────────────────────


def _generate_self_signed_cert(host: str, lan_addrs: list[str]) -> tuple[str, str]:
    """Return ``(cert_path, key_path)`` for a freshly generated self-signed cert.

    The SAN includes ``localhost``, the bind host (if it is a real address), and
    every discovered LAN address, so the certificate validates for any device
    that opens the UI. Tries ``openssl`` on PATH first, then ``cryptography``.
    """
    global _CERT_TMPDIR
    _CERT_TMPDIR = tempfile.mkdtemp(prefix="s2s-test-ui-")
    atexit.register(shutil.rmtree, _CERT_TMPDIR, ignore_errors=True)
    cert_path = os.path.join(_CERT_TMPDIR, "cert.pem")
    key_path = os.path.join(_CERT_TMPDIR, "key.pem")
    common_name = host if host not in ("0.0.0.0", "::") else "localhost"

    # Build the SAN list: localhost + real IPs only (skip 0.0.0.0 / IPv6-any).
    san_dns = [common_name] if common_name != "localhost" else ["localhost"]
    san_ips: list[str] = []
    for candidate in ["127.0.0.1", host, *lan_addrs]:
        if candidate in ("0.0.0.0", "::", "", "localhost"):
            continue
        try:
            import ipaddress

            ipaddress.ip_address(candidate)
        except ValueError:
            continue
        if candidate not in san_ips:
            san_ips.append(candidate)

    # --- openssl ---
    if shutil.which("openssl"):
        san_parts = []
        for d in san_dns:
            san_parts.append(f"DNS:{d}")
        for ip in san_ips:
            san_parts.append(f"IP:{ip}")
        san = ",".join(san_parts)
        try:
            subprocess.run(
                [
                    "openssl",
                    "req",
                    "-x509",
                    "-newkey",
                    "rsa:2048",
                    "-nodes",
                    "-keyout",
                    key_path,
                    "-out",
                    cert_path,
                    "-days",
                    "365",
                    "-subj",
                    f"/CN={common_name}",
                    "-addext",
                    f"subjectAltName={san}",
                ],
                check=True,
                capture_output=True,
                timeout=60,
            )
            return cert_path, key_path
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
            log.warning("openssl cert generation failed: %s", e)

    # --- cryptography ---
    try:
        from datetime import datetime, timedelta, timezone

        import ipaddress

        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID

        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])
        now = datetime.now(timezone.utc)
        san_entries = [x509.DNSName(d) for d in san_dns]
        for ip in san_ips:
            san_entries.append(x509.IPAddress(ipaddress.ip_address(ip)))

        cert = (
            x509.CertificateBuilder()
            .subject_name(name)
            .issuer_name(name)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(minutes=5))
            .not_valid_after(now + timedelta(days=365))
            .add_extension(x509.SubjectAlternativeName(san_entries), critical=False)
            .sign(key, hashes.SHA256())
        )

        with open(key_path, "wb") as f:
            f.write(
                key.private_bytes(
                    serialization.Encoding.PEM,
                    serialization.PrivateFormat.TraditionalOpenSSL,
                    serialization.NoEncryption(),
                )
            )
        with open(cert_path, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))
        return cert_path, key_path
    except ImportError as e:
        raise RuntimeError(
            "Could not generate a self-signed certificate: neither 'openssl' on PATH "
            "nor the 'cryptography' package is available. Install one of them, or pass "
            "--cert/--key with your own certificate."
        ) from e


# ─────────────────────────────────────────────────────────────────────
# HTTP handler: serves the UI and proxies API calls
# ─────────────────────────────────────────────────────────────────────


class UIHandler(BaseHTTPRequestHandler):
    server_version = "S2STestUI/1.0"
    protocol_version = "HTTP/1.1"
    # Bound handler-thread reads so idle/malicious connections can't pin threads.
    timeout = 60

    # Set on the class by build_server() when running under TLS.
    ssl_context: ssl.SSLContext | None = None

    # ── connection setup (per-connection TLS, off the accept loop) ─────

    def setup(self) -> None:
        # Wrap the socket in the handler thread (not the accept loop) so one
        # stalled TLS handshake can't freeze every other client.
        if UIHandler.ssl_context is not None:
            self.request.settimeout(30)  # bound the handshake itself
            self.request = UIHandler.ssl_context.wrap_socket(self.request, server_side=True)
        super().setup()

    # Silence default per-request stderr logging; we log selectively.
    def log_message(self, format: str, *args) -> None:
        log.debug("%s - %s", self.address_string(), format % args)

    # ── response helpers ───────────────────────────────────────────────

    def _send(
        self,
        status: int,
        body: bytes,
        content_type: str,
        extra_headers: dict | None = None,
        close: bool = False,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if close:
            # Error paths may leave an unread request body in the stream; closing
            # the connection keeps keep-alive framing from desyncing.
            self.close_connection = True
            self.send_header("Connection", "close")
        for k, v in (extra_headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if body:
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True

    def _send_json(self, status: int, obj, close: bool = False) -> None:
        self._send(status, json.dumps(obj).encode("utf-8"), "application/json", close=close)

    def _send_error(self, status: int, message: str, close: bool = True) -> None:
        self._send_json(status, {"status": "error", "error": message}, close=close)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length <= 0:
            return b""
        if length > MAX_PROXY_BYTES:
            raise ValueError(f"Request body too large ({length} bytes; max {MAX_PROXY_BYTES})")
        return self.rfile.read(length)

    # ── CORS preflight (no CORS headers: same-origin UI only) ──────────

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.end_headers()

    # ── routes ─────────────────────────────────────────────────────────

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/index.html"):
            self._send(
                200,
                UI_HTML_BYTES,
                "text/html; charset=utf-8",
                extra_headers={"Content-Security-Policy": _CSP},
            )
            return
        if self.path == "/health":
            with STATE_LOCK:
                self._send_json(200, {"status": "ok", "api_base": STATE["api_base"]})
            return
        if self.path == "/config":
            with STATE_LOCK:
                self._send_json(200, {"api_base": STATE["api_base"]})
            return
        if self.path.startswith("/proxy/"):
            self._proxy("GET", self.path[len("/proxy/"):])
            return
        self._send_error(404, "Not found")

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/config":
            self._handle_config()
            return
        if self.path.startswith("/proxy/"):
            self._proxy("POST", self.path[len("/proxy/"):])
            return
        self._send_error(404, "Not found")

    def _handle_config(self) -> None:
        try:
            body = self._read_body()
        except ValueError as e:
            self._send_error(413, str(e))
            return
        try:
            data = json.loads(body.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self._send_error(400, "Invalid JSON body")
            return
        api_base = validate_api_base(data.get("api_base", ""))
        if api_base is None:
            self._send_error(400, "api_base must be a valid http(s) URL with a valid port")
            return
        with STATE_LOCK:
            STATE["api_base"] = api_base
        log.info("API base URL updated to %s", api_base)
        self._send_json(200, {"status": "ok", "api_base": api_base})

    # ── proxy core ─────────────────────────────────────────────────────

    def _proxy(self, method: str, raw_path: str) -> None:
        """Forward *method* ``/{raw_path}`` to the configured API server."""
        endpoint = resolve_endpoint(raw_path)
        if endpoint is None:
            self._send_error(404, f"Endpoint not allowed: {raw_path!r}")
            return

        with STATE_LOCK:
            api_base = STATE["api_base"]
        if validate_api_base(api_base) is None:
            self._send_error(502, f"Invalid configured API base URL: {api_base!r}")
            return

        parts = urlsplit(api_base)
        scheme = parts.scheme
        host = parts.hostname
        try:
            port = parts.port or (443 if scheme == "https" else 80)
        except ValueError:
            self._send_error(502, "Invalid port in API base URL")
            return
        upstream_path = (parts.path.rstrip("/") + "/" + endpoint)
        query = f"?{parts.query}" if parts.query else ""

        try:
            if scheme == "https":
                context = ssl._create_unverified_context() if INSECURE_UPSTREAM else None
                conn = http.client.HTTPSConnection(host, port, timeout=300, context=context)
            else:
                conn = http.client.HTTPConnection(host, port, timeout=300)
        except OSError as e:
            self._send_error(502, f"Cannot connect to {host}:{port}: {e}")
            return

        headers: dict[str, str] = {}
        content_type = self.headers.get("Content-Type")
        if content_type:
            headers["Content-Type"] = content_type

        body: bytes | None = None
        if method == "POST":
            try:
                body = self._read_body()
            except ValueError as e:
                self._send_error(413, str(e))
                return

        log.info("PROXY %s %s -> %s%s", method, endpoint, api_base, upstream_path)
        try:
            conn.request(method, upstream_path + query, body=body, headers=headers)
            resp = conn.getresponse()

            # Read the response in bounded chunks to cap memory usage.
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > MAX_PROXY_BYTES:
                    break
            if total > MAX_PROXY_BYTES:
                self._send_error(502, f"Upstream response too large (> {MAX_PROXY_BYTES} bytes)")
                return
            resp_body = b"".join(chunks)
        except (OSError, http.client.HTTPException) as e:
            log.error("PROXY error for %s: %s", endpoint, e)
            self._send_error(
                502,
                f"Upstream request to {api_base}{upstream_path} failed: {e}",
            )
            return
        finally:
            conn.close()

        content_type = resp.getheader("Content-Type", "application/octet-stream")
        extra: dict[str, str] = {}
        req_id = resp.getheader("X-Request-Id")
        if req_id:
            clean = req_id.replace("\r", "").replace("\n", "")[:128]
            if clean:
                extra["X-Request-Id"] = clean
        log.info("PROXY %s %s <- HTTP %d (%d bytes)", method, endpoint, resp.status, len(resp_body))
        self._send(resp.status, resp_body, content_type, extra_headers=extra)


# A minimal CSP for the inline UI: allow inline script/style (they are embedded)
# and blob: media (the TTS player uses URL.createObjectURL).
_CSP = (
    "default-src 'self'; "
    "script-src 'unsafe-inline'; "
    "style-src 'unsafe-inline'; "
    "media-src blob:; "
    "img-src 'self' data:; "
    "connect-src 'self'"
)


class QuietHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address) -> None:  # noqa: A003
        # Suppress noisy client-disconnect tracebacks; log anything else briefly.
        exc = sys.exception()
        if isinstance(exc, (BrokenPipeError, ConnectionResetError, TimeoutError, ssl.SSLError)):
            return
        log.warning("Connection error from %s: %r", client_address[0] if client_address else "?", exc)


def build_server(host: str, port: int, ssl_context: ssl.SSLContext | None) -> QuietHTTPServer:
    UIHandler.ssl_context = ssl_context
    httpd = QuietHTTPServer((host, port), UIHandler)
    return httpd


def main() -> int:
    parser = argparse.ArgumentParser(
        description="LAN web UI for testing the Qwen-Clone branch endpoints.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python test-api-ui-launch.py\n"
            "  python test-api-ui-launch.py --api-url http://192.168.1.50:8765\n"
            "  python test-api-ui-launch.py --ssl\n"
            "  python test-api-ui-launch.py --api-url https://dev:8765 --insecure-upstream\n"
        ),
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help=f"Base URL of the speech-to-speech server (default: {DEFAULT_API_URL})",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Interface to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port to listen on (default: {DEFAULT_PORT})")
    parser.add_argument(
        "--ssl",
        action="store_true",
        help="Serve over HTTPS so the microphone works from any LAN device "
        "(generates a self-signed certificate if --cert/--key are not given)",
    )
    parser.add_argument("--cert", help="Path to a PEM certificate (used with --ssl; requires --key)")
    parser.add_argument("--key", help="Path to a PEM private key (used with --ssl; requires --cert)")
    parser.add_argument(
        "--insecure-upstream",
        action="store_true",
        help="Disable TLS certificate verification when the API server uses https "
        "(for self-signed dev servers)",
    )
    args = parser.parse_args()

    global INSECURE_UPSTREAM
    INSECURE_UPSTREAM = args.insecure_upstream

    api_base = validate_api_base(args.api_url)
    if api_base is None:
        log.error("Invalid --api-url: %r", args.api_url)
        return 1
    with STATE_LOCK:
        STATE["api_base"] = api_base

    if bool(args.cert) != bool(args.key):
        log.error("--cert and --key must be provided together")
        return 1

    ssl_context: ssl.SSLContext | None = None
    if args.ssl:
        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        if args.cert and args.key:
            ssl_context.load_cert_chain(args.cert, args.key)
            log.info("Loaded certificate from %s", args.cert)
        else:
            lan_addrs = _lan_addresses()
            cert_path, key_path = _generate_self_signed_cert(args.host, lan_addrs)
            ssl_context.load_cert_chain(cert_path, key_path)
            log.info("Generated self-signed certificate (CN=localhost, %d SAN IP(s))", len(lan_addrs) + 1)

    scheme = "https" if ssl_context else "http"
    try:
        httpd = build_server(args.host, args.port, ssl_context)
    except OSError as e:
        log.error("Failed to bind %s:%s — %s", args.host, args.port, e)
        return 1

    addrs = _lan_addresses()
    print()
    print("=" * 68)
    print("  Qwen-Clone endpoint test UI")
    print("=" * 68)
    print("  UI URL(s):")
    print(f"      {scheme}://localhost:{args.port}")
    for ip in addrs:
        print(f"      {scheme}://{ip}:{args.port}   (any device on your LAN)")
    print(f"  Proxying API calls to:  {STATE['api_base']}")
    if ssl_context:
        print()
        print("  NOTE: self-signed certificate — accept the browser warning.")
        print("        The microphone works from any device over HTTPS.")
    else:
        print()
        print("  NOTE: the microphone works from any LAN device only over HTTPS.")
        print("        Re-run with --ssl if you need the mic off this machine.")
    print("  Press Ctrl+C to stop.")
    print("=" * 68)
    print()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        httpd.server_close()
    return 0


# ─────────────────────────────────────────────────────────────────────
# UI (single-page HTML/CSS/JS, embedded for zero-dependency portability)
# ─────────────────────────────────────────────────────────────────────

UI_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Qwen-Clone Endpoint Test UI</title>
<style>
  :root {
    --bg: #0f1115;
    --panel: #171a21;
    --panel-2: #1e222b;
    --border: #2a2f3a;
    --text: #e6e9ef;
    --muted: #8b93a1;
    --accent: #4f8cff;
    --accent-2: #2f6fe0;
    --ok: #34c77b;
    --err: #ff5c5c;
    --warn: #ffb020;
    --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    line-height: 1.5;
  }
  header {
    padding: 18px 22px; border-bottom: 1px solid var(--border);
    background: linear-gradient(180deg, #12151c, #0f1115);
    display: flex; flex-wrap: wrap; gap: 14px; align-items: center; justify-content: space-between;
    position: sticky; top: 0; z-index: 5;
  }
  header h1 { font-size: 17px; margin: 0; font-weight: 650; letter-spacing: .2px; }
  header h1 span { color: var(--accent); }
  .conn { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  .conn label { font-size: 12px; color: var(--muted); }
  input, select, textarea {
    background: var(--panel-2); color: var(--text); border: 1px solid var(--border);
    border-radius: 8px; padding: 8px 10px; font-size: 13px; font-family: inherit;
  }
  input:focus, select:focus, textarea:focus { outline: none; border-color: var(--accent); }
  textarea { resize: vertical; min-height: 64px; width: 100%; }
  button {
    background: var(--accent); color: #fff; border: 0; border-radius: 8px;
    padding: 9px 14px; font-size: 13px; font-weight: 600; cursor: pointer;
    transition: background .15s;
  }
  button:hover { background: var(--accent-2); }
  button:disabled { opacity: .5; cursor: not-allowed; }
  button.secondary { background: var(--panel-2); color: var(--text); border: 1px solid var(--border); }
  button.secondary:hover { background: #262b36; }
  button.danger { background: #b23b3b; }
  button.danger:hover { background: #9a3030; }
  button.rec { background: var(--err); }
  main { padding: 20px 22px 60px; max-width: 1080px; margin: 0 auto; }
  .tabs { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 18px; }
  .tab {
    padding: 9px 16px; border-radius: 9px; cursor: pointer; font-size: 13px;
    font-weight: 600; color: var(--muted); background: var(--panel); border: 1px solid var(--border);
  }
  .tab.active { color: #fff; background: var(--accent); border-color: var(--accent); }
  .card {
    background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
    padding: 18px; margin-bottom: 18px;
  }
  .card h2 { margin: 0 0 4px; font-size: 15px; }
  .card .sub { color: var(--muted); font-size: 12.5px; margin: 0 0 14px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }
  .field { display: flex; flex-direction: column; gap: 5px; }
  .field label { font-size: 12px; color: var(--muted); }
  .row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-top: 14px; }
  .status { font-size: 12.5px; padding: 6px 10px; border-radius: 8px; display: none; }
  .status.show { display: inline-block; }
  .status.ok { background: rgba(52,199,123,.15); color: var(--ok); }
  .status.err { background: rgba(255,92,92,.15); color: var(--err); }
  .status.warn { background: rgba(255,176,32,.15); color: var(--warn); }
  .result {
    margin-top: 14px; background: var(--panel-2); border: 1px solid var(--border);
    border-radius: 10px; padding: 12px; font-family: var(--mono); font-size: 12.5px;
    white-space: pre-wrap; word-break: break-word; max-height: 320px; overflow: auto;
    display: none;
  }
  .result.show { display: block; }
  .result .label { color: var(--muted); font-family: inherit; font-size: 11px; text-transform: uppercase; letter-spacing: .5px; display:block; margin-bottom: 6px; }
  .audio-out { margin-top: 12px; display: none; }
  .audio-out.show { display: block; }
  .audio-out audio { width: 100%; }
  .mic {
    display: flex; gap: 10px; align-items: center; flex-wrap: wrap;
    background: var(--panel-2); border: 1px dashed var(--border); border-radius: 10px;
    padding: 12px; margin-bottom: 12px;
  }
  .mic .rec-indicator { width: 10px; height: 10px; border-radius: 50%; background: var(--muted); }
  .mic.recording .rec-indicator { background: var(--err); animation: pulse 1s infinite; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: .3; } }
  .mic .timer { font-family: var(--mono); font-size: 13px; color: var(--muted); }
  .file-drop { font-size: 12.5px; color: var(--muted); margin-top: 8px; }
  .file-drop input[type=file] { font-size: 12px; }
  .table { width: 100%; border-collapse: collapse; font-size: 13px; }
  .table th, .table td { text-align: left; padding: 9px 10px; border-bottom: 1px solid var(--border); }
  .table th { color: var(--muted); font-weight: 600; font-size: 11.5px; text-transform: uppercase; letter-spacing: .4px; }
  .table tr:last-child td { border-bottom: 0; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 20px; font-size: 11px; font-weight: 600; }
  .badge.default { background: rgba(79,140,255,.2); color: var(--accent); }
  .badge.id { background: rgba(139,147,161,.18); color: var(--muted); font-family: var(--mono); }
  .hint { font-size: 12px; color: var(--muted); margin-top: 8px; }
  .code { font-family: var(--mono); background: var(--panel-2); padding: 1px 6px; border-radius: 5px; font-size: 12px; }
  .muted { color: var(--muted); }
  .spacer { flex: 1; }
  .endpoint { font-family: var(--mono); font-size: 12px; color: var(--accent); }
  .hidden { display: none !important; }
</style>
</head>
<body>
<header>
  <h1>Qwen-Clone <span>Endpoint Test UI</span></h1>
  <div class="conn">
    <label for="apiBase">API server</label>
    <input id="apiBase" type="text" style="min-width:260px" placeholder="http://127.0.0.1:8765">
    <button id="saveApi" class="secondary">Save</button>
    <button id="testConn" class="secondary">Test</button>
    <span id="connStatus" class="status" data-status="conn" aria-live="polite"></span>
  </div>
</header>

<main>
  <div class="tabs" role="tablist">
    <div class="tab active" data-tab="transcription" role="tab">Transcription</div>
    <div class="tab" data-tab="translation" role="tab">Translation</div>
    <div class="tab" data-tab="speech" role="tab">Speech (TTS)</div>
    <div class="tab" data-tab="clones" role="tab">Voice Clones</div>
  </div>

  <!-- ── TRANSCRIPTION ─────────────────────────────────────────── -->
  <section id="tab-transcription" class="tab-panel">
    <div class="card">
      <h2>Transcription <span class="endpoint">POST /v1/audio/transcriptions</span></h2>
      <p class="sub">Convert audio to text. Record with your microphone or upload a file.</p>

      <label style="display:flex; gap:8px; align-items:center; margin:0 0 14px; font-size:13px; font-weight:600; cursor:pointer">
        <input type="checkbox" id="rt-toggle" style="width:auto"> Realtime
        <span class="muted" style="font-weight:400; font-size:12px">— transcribe continuously as you talk</span>
      </label>

      <div class="mic" id="mic-transcription">
        <span class="rec-indicator"></span>
        <button class="rec" data-mic="transcription">● Record</button>
        <button class="secondary hidden" data-stop="transcription">■ Stop</button>
        <button class="secondary hidden" data-clear="transcription">Clear</button>
        <span class="timer" data-timer="transcription"></span>
        <span class="muted" data-micmsg="transcription">No recording yet</span>
      </div>
      <div class="file-drop">
        …or upload an audio file (mp3, wav, flac, ogg, m4a, webm):
        <input type="file" id="file-transcription" accept="audio/*">
      </div>

      <div class="grid" style="margin-top:14px">
        <div class="field"><label>model</label><input id="t-model" value="whisper"></div>
        <div class="field"><label>language (optional)</label><input id="t-language" placeholder="auto (e.g. en, fr)"></div>
        <div class="field"><label>response_format</label>
          <select id="t-format">
            <option>json</option><option>text</option><option>verbose_json</option>
            <option>srt</option><option>vtt</option>
          </select>
        </div>
        <div class="field"><label>temperature</label><input id="t-temp" type="number" step="0.1" value="0.0"></div>
      </div>
      <div class="field" style="margin-top:12px">
        <label>prompt (optional)</label>
        <textarea id="t-prompt" placeholder="Optional context to guide the model"></textarea>
      </div>

      <div class="row">
        <button data-run="transcription">Transcribe</button>
        <span class="status" data-status="transcription" aria-live="polite"></span>
        <span class="spacer"></span>
        <label class="muted" style="font-size:12px"><input type="checkbox" data-raw="transcription" style="width:auto"> show raw</label>
      </div>
      <div class="result" data-result="transcription"></div>
      <div class="result" data-realtime="transcription" style="border-color: var(--accent)"></div>
    </div>
  </section>

  <!-- ── TRANSLATION ───────────────────────────────────────────── -->
  <section id="tab-translation" class="tab-panel hidden">
    <div class="card">
      <h2>Translation <span class="endpoint">POST /v1/audio/translations</span></h2>
      <p class="sub">Translate non-English audio to English text.</p>

      <div class="mic" id="mic-translation">
        <span class="rec-indicator"></span>
        <button class="rec" data-mic="translation">● Record</button>
        <button class="secondary hidden" data-stop="translation">■ Stop</button>
        <button class="secondary hidden" data-clear="translation">Clear</button>
        <span class="timer" data-timer="translation"></span>
        <span class="muted" data-micmsg="translation">No recording yet</span>
      </div>
      <div class="file-drop">
        …or upload an audio file:
        <input type="file" id="file-translation" accept="audio/*">
      </div>

      <div class="grid" style="margin-top:14px">
        <div class="field"><label>model</label><input id="tr-model" value="whisper"></div>
        <div class="field"><label>response_format</label>
          <select id="tr-format">
            <option>json</option><option>text</option><option>verbose_json</option>
            <option>srt</option><option>vtt</option>
          </select>
        </div>
      </div>
      <div class="field" style="margin-top:12px">
        <label>prompt (optional)</label>
        <textarea id="tr-prompt" placeholder="Optional context to guide the model"></textarea>
      </div>

      <div class="row">
        <button data-run="translation">Translate</button>
        <span class="status" data-status="translation" aria-live="polite"></span>
        <span class="spacer"></span>
        <label class="muted" style="font-size:12px"><input type="checkbox" data-raw="translation" style="width:auto"> show raw</label>
      </div>
      <div class="result" data-result="translation"></div>
    </div>
  </section>

  <!-- ── SPEECH ────────────────────────────────────────────────── -->
  <section id="tab-speech" class="tab-panel hidden">
    <div class="card">
      <h2>Speech / TTS <span class="endpoint">POST /v1/audio/speech</span></h2>
      <p class="sub">Synthesise text to audio. If a default voice clone is set, synthesis routes to it.</p>

      <div class="field">
        <label>input (text to speak)</label>
        <textarea id="s-input" placeholder="Enter text to synthesise…">Hello! This is a test of the speech synthesis endpoint.</textarea>
      </div>
      <div class="grid" style="margin-top:12px">
        <div class="field"><label>model</label><input id="s-model" value="qwen3-tts"></div>
        <div class="field"><label>voice</label><input id="s-voice" value="alloy" placeholder="alloy / clone id"></div>
        <div class="field"><label>response_format</label>
          <select id="s-format">
            <option>mp3</option><option>wav</option><option>flac</option>
          </select>
        </div>
        <div class="field"><label>speed</label><input id="s-speed" type="number" step="0.1" value="1.0"></div>
      </div>
      <div class="field" style="margin-top:12px">
        <label>instructions (optional)</label>
        <input id="s-instructions" placeholder="Optional style instructions">
      </div>
      <p class="hint">Note: <span class="code">pcm</span> output is raw bytes (no container header) and can't be played inline, so it is not offered here.</p>

      <div class="row">
        <button data-run="speech">Synthesise</button>
        <span class="status" data-status="speech" aria-live="polite"></span>
      </div>
      <div class="audio-out" data-audio="speech"><audio controls></audio></div>
      <div class="result" data-result="speech"></div>
    </div>
  </section>

  <!-- ── CLONES ────────────────────────────────────────────────── -->
  <section id="tab-clones" class="tab-panel hidden">
    <div class="card">
      <h2>Voice Clones</h2>
      <p class="sub">Manage Qwen3-TTS reference-voice clones.</p>

      <div class="row" style="margin-top:0">
        <button id="refreshClones">⟳ Refresh list</button>
        <span class="status" data-status="clones" aria-live="polite"></span>
      </div>
      <div id="cloneTableWrap" style="margin-top:14px"></div>
    </div>

    <div class="card">
      <h2>Create clone <span class="endpoint">POST /clone/new</span></h2>
      <p class="sub">Upload an MP3 (2–20 s) of the voice, its exact transcription, and a name.</p>
      <div class="grid">
        <div class="field"><label>name</label><input id="c-name" placeholder="e.g. my-voice"></div>
        <div class="field"><label>description (optional)</label><input id="c-desc" placeholder="optional"></div>
      </div>
      <div class="field" style="margin-top:12px">
        <label>transcription (exact words in the MP3)</label>
        <textarea id="c-transcription" placeholder="Type the exact words spoken in the reference audio"></textarea>
      </div>
      <div class="field" style="margin-top:12px">
        <label>reference audio (MP3, 2–20 s)</label>
        <input type="file" id="c-file" accept="audio/mpeg,audio/mp3,.mp3">
      </div>
      <div class="row">
        <button data-run="clone-new">Create clone</button>
        <span class="status" data-status="clone-new" aria-live="polite"></span>
      </div>
      <div class="result" data-result="clone-new"></div>
    </div>
  </section>
</main>

<script>
(function () {
  "use strict";

  // ── tiny DOM helpers ──────────────────────────────────────────────
  const $ = (sel, root) => (root || document).querySelector(sel);
  const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));

  // ── state ─────────────────────────────────────────────────────────
  const recordings = {};      // key -> Blob
  const recorders = {};       // key -> {rec, stream, iv}
  let apiBase = "";

  // ── tabs ──────────────────────────────────────────────────────────
  $$(".tab").forEach(tab => {
    tab.addEventListener("click", () => {
      $$(".tab").forEach(t => t.classList.remove("active"));
      $$(".tab-panel").forEach(p => p.classList.add("hidden"));
      tab.classList.add("active");
      $("#tab-" + tab.dataset.tab).classList.remove("hidden");
      if (tab.dataset.tab === "clones") refreshClones();
    });
  });

  // ── status helpers ────────────────────────────────────────────────
  function setStatus(key, kind, msg) {
    const el = $(`[data-status="${key}"]`);
    if (!el) return;
    el.className = "status show " + kind;
    el.textContent = msg;
  }
  function clearStatus(key) {
    const el = $(`[data-status="${key}"]`);
    if (el) { el.className = "status"; el.textContent = ""; }
  }
  function showResult(key, label, text) {
    const el = $(`[data-result="${key}"]`);
    if (!el) return;
    el.innerHTML = "";
    const lab = document.createElement("span");
    lab.className = "label";
    lab.textContent = label;
    el.appendChild(lab);
    el.appendChild(document.createTextNode(text));
    el.classList.add("show");
  }

  // ── connection config ─────────────────────────────────────────────
  async function loadConfig() {
    try {
      const r = await fetch("/config");
      const j = await r.json();
      apiBase = j.api_base;
      $("#apiBase").value = apiBase;
    } catch (e) { /* ignore */ }
  }
  $("#saveApi").addEventListener("click", async () => {
    const val = $("#apiBase").value.trim();
    try {
      const r = await fetch("/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ api_base: val }),
      });
      const j = await r.json();
      if (j.status === "ok") {
        apiBase = j.api_base;
        $("#apiBase").value = apiBase;
        setStatus("conn", "ok", "Saved");
      } else {
        setStatus("conn", "err", j.error || "Failed");
      }
    } catch (e) {
      setStatus("conn", "err", "Failed to save: " + e.message);
    }
    setTimeout(() => clearStatus("conn"), 2500);
  });
  $("#testConn").addEventListener("click", async () => {
    setStatus("conn", "warn", "Testing…");
    try {
      const r = await fetch("/proxy/clone/list", { method: "GET" });
      const text = await r.text();
      if (r.ok) {
        let count = 0;
        try { count = JSON.parse(text).length; } catch (e) {}
        setStatus("conn", "ok", `Connected (${count} clone${count === 1 ? "" : "s"})`);
      } else {
        setStatus("conn", "err", "HTTP " + r.status);
      }
    } catch (e) {
      setStatus("conn", "err", "Connection failed");
    }
    setTimeout(() => clearStatus("conn"), 4000);
  });

  // ── microphone recording ──────────────────────────────────────────
  function pickMimeType() {
    const candidates = [
      "audio/webm;codecs=opus",
      "audio/webm",
      "audio/ogg;codecs=opus",
      "audio/mp4",
      "audio/mpeg",
    ];
    if (window.MediaRecorder) {
      for (const c of candidates) {
        try { if (MediaRecorder.isTypeSupported(c)) return c; } catch (e) {}
      }
    }
    return "";
  }
  function extForMime(mime) {
    mime = (mime || "").toLowerCase();
    if (mime.includes("webm")) return "webm";
    if (mime.includes("ogg")) return "ogg";
    if (mime.includes("mp4")) return "m4a";
    if (mime.includes("mpeg")) return "mp3";
    return "webm";
  }

  async function startRecording(key) {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      micMsg(key, "Mic unavailable — needs a secure context (use --ssl, or open via localhost).");
      return;
    }
    let stream = null;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mime = pickMimeType();
      const rec = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
      const chunks = [];
      rec.ondataavailable = e => { if (e.data && e.data.size) chunks.push(e.data); };
      rec.onstop = () => {
        const type = rec.mimeType || mime || "audio/webm";
        recordings[key] = new Blob(chunks, { type });
        micMsg(key, `Recorded ${(recordings[key].size / 1024).toFixed(0)} KB (${type})`);
        stream.getTracks().forEach(t => t.stop());
        setMicUI(key, false);
        if (recorders[key] && recorders[key].iv) clearInterval(recorders[key].iv);
        delete recorders[key];
      };
      rec.start();
      const t0 = Date.now();
      const iv = setInterval(() => {
        const s = Math.floor((Date.now() - t0) / 1000);
        $(`[data-timer="${key}"]`).textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
      }, 250);
      recorders[key] = { rec, stream, iv };
      setMicUI(key, true);
      micMsg(key, "Recording…");
    } catch (e) {
      if (stream) stream.getTracks().forEach(t => t.stop());
      const name = e && e.name ? ` (${e.name})` : "";
      micMsg(key, "Mic error" + name + ": " + (e && e.message ? e.message : String(e)));
    }
  }
  function stopRecording(key) {
    const r = recorders[key];
    if (r && r.rec && r.rec.state !== "inactive") r.rec.stop();
  }
  function clearRecording(key) {
    const r = recorders[key];
    if (r) {
      // Cancel an in-progress recording without saving a blob.
      r.rec.onstop = null;
      try { r.rec.stop(); } catch (e) {}
      r.stream.getTracks().forEach(t => t.stop());
      if (r.iv) clearInterval(r.iv);
      delete recorders[key];
    }
    delete recordings[key];
    setMicUI(key, false);
    $(`[data-timer="${key}"]`).textContent = "";
    micMsg(key, "No recording yet");
  }
  function setMicUI(key, recording) {
    $(`#mic-${key}`).classList.toggle("recording", recording);
    $(`[data-mic="${key}"]`).classList.toggle("hidden", recording);
    $(`[data-stop="${key}"]`).classList.toggle("hidden", !recording);
    // Clear is available while recording (to cancel) or when a take exists.
    $(`[data-clear="${key}"]`).classList.toggle("hidden", !recording && !recordings[key]);
  }
  function micMsg(key, msg) { $(`[data-micmsg="${key}"]`).textContent = msg; }

  // ── realtime (chunked) transcription ──────────────────────────────
  // The /transcriptions REST endpoint is request/response (no true
  // streaming), so "realtime" here means: capture raw PCM continuously
  // from the mic, and every RT_CHUNK_MS encode the accumulated audio to
  // WAV and transcribe it, appending the result to a live transcript.
  // This gives the "transcribes as I keep talking" experience.
  const RT_CHUNK_MS = 1500;
  const realtime = {};  // key -> state

  function float32ToWav(pcm, sampleRate) {
    const numChannels = 1;
    const length = pcm.length * 2 + 44;
    const buffer = new ArrayBuffer(length);
    const view = new DataView(buffer);
    let offset = 0;
    const writeString = s => { for (let i = 0; i < s.length; i++) view.setUint8(offset++, s.charCodeAt(i)); };
    const writeUint32 = v => { view.setUint32(offset, v, true); offset += 4; };
    const writeUint16 = v => { view.setUint16(offset, v, true); offset += 2; };

    writeString("RIFF");
    writeUint32(36 + length);
    writeString("WAVE");
    writeString("fmt ");
    writeUint32(16);
    writeUint16(1);               // PCM
    writeUint16(numChannels);
    writeUint32(sampleRate);
    writeUint32(sampleRate * numChannels * 2);
    writeUint16(numChannels * 2);
    writeUint16(16);              // bits per sample
    writeString("data");
    writeUint32(pcm.length * 2);

    let idx = offset;
    for (let i = 0; i < pcm.length; i++) {
      const s = Math.max(-1, Math.min(1, pcm[i]));
      view.setInt16(idx, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      idx += 2;
    }
    return buffer;
  }

  function showRealtimeTranscript(key, text) {
    const el = $(`[data-realtime="${key}"]`);
    if (!el) return;
    el.innerHTML = "";
    const lab = document.createElement("span");
    lab.className = "label";
    lab.textContent = "Live transcript";
    el.appendChild(lab);
    el.appendChild(document.createTextNode(text || ""));
    el.classList.add("show");
  }

  async function startRealtime(key) {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      micMsg(key, "Mic unavailable — needs a secure context (use --ssl, or open via localhost).");
      return;
    }
    if (!window.AudioContext && !window.webkitAudioContext) {
      micMsg(key, "Web Audio API not available in this browser.");
      return;
    }
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (e) {
      const name = e && e.name ? ` (${e.name})` : "";
      micMsg(key, "Mic error" + name + ": " + (e && e.message ? e.message : String(e)));
      return;
    }
    const Ctx = window.AudioContext || window.webkitAudioContext;
    const ctx = new Ctx();
    try { await ctx.resume(); } catch (e) {}
    const source = ctx.createMediaStreamSource(stream);
    const processor = ctx.createScriptProcessor(4096, 1, 1);
    const gain = ctx.createGain();
    gain.gain.value = 0;  // route through the processor without playing back
    source.connect(processor);
    processor.connect(gain);
    gain.connect(ctx.destination);

    const state = {
      ctx, source, processor, gain, stream,
      chunks: [], inFlight: false, running: true, transcript: "", t0: Date.now(),
    };
    realtime[key] = state;

    processor.onaudioprocess = e => {
      if (!state.running) return;
      state.chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
    };
    state.timer = setInterval(() => flushRealtimeChunk(key), RT_CHUNK_MS);
    state.clock = setInterval(() => {
      const s = Math.floor((Date.now() - state.t0) / 1000);
      $(`[data-timer="${key}"]`).textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
    }, 250);

    setMicUI(key, true);
    micMsg(key, "Live transcription running…");
    showRealtimeTranscript(key, "");
  }

  async function flushRealtimeChunk(key) {
    const state = realtime[key];
    if (!state || !state.running || state.inFlight) return;
    let total = 0;
    for (const c of state.chunks) total += c.length;
    // Require at least ~0.5s of audio before spending a request on it.
    if (total < Math.floor(0.5 * state.ctx.sampleRate)) return;

    const pcm = new Float32Array(total);
    let off = 0;
    for (const c of state.chunks) { pcm.set(c, off); off += c.length; }
    state.chunks = [];
    state.inFlight = true;
    micMsg(key, "Transcribing…");
    try {
      const blob = new Blob([float32ToWav(pcm, state.ctx.sampleRate)], { type: "audio/wav" });
      const form = new FormData();
      form.append("file", blob, `live-${Date.now()}.wav`);
      form.append("model", $("#t-model").value.trim() || "whisper");
      const lang = $("#t-language").value.trim();
      if (lang) form.append("language", lang);
      form.append("response_format", "json");
      const res = await proxyFetch("v1/audio/transcriptions", { form });
      const text = decodeText(res.buffer);
      let piece = "";
      if (res.status.toString().startsWith("2")) {
        try { piece = (JSON.parse(text).text || "").trim(); } catch (e) { piece = text.trim(); }
      } else {
        piece = `[HTTP ${res.status}]`;
      }
      if (piece && state.running) {
        state.transcript = (state.transcript ? state.transcript + " " : "") + piece;
        showRealtimeTranscript(key, state.transcript);
      }
    } catch (e) {
      /* transient network error — keep capturing */
    } finally {
      state.inFlight = false;
      if (realtime[key] && realtime[key].running) micMsg(key, "Live transcription running…");
    }
  }

  function stopRealtime(key) {
    const state = realtime[key];
    if (!state) return;
    state.running = false;
    clearInterval(state.timer);
    clearInterval(state.clock);
    try { state.processor.disconnect(); } catch (e) {}
    try { state.source.disconnect(); } catch (e) {}
    try { state.gain.disconnect(); } catch (e) {}
    state.stream.getTracks().forEach(t => t.stop());
    state.ctx.close().catch(() => {});
    delete realtime[key];
    setMicUI(key, false);
    $(`[data-timer="${key}"]`).textContent = "";
    micMsg(key, "Live transcription stopped.");
  }

  $$("[data-mic]").forEach(b => b.addEventListener("click", () => {
    const key = b.dataset.mic;
    if (key === "transcription" && $("#rt-toggle").checked) startRealtime(key);
    else startRecording(key);
  }));
  $$("[data-stop]").forEach(b => b.addEventListener("click", () => {
    const key = b.dataset.stop;
    if (realtime[key]) stopRealtime(key);
    else stopRecording(key);
  }));
  $$("[data-clear]").forEach(b => b.addEventListener("click", () => {
    const key = b.dataset.clear;
    if (realtime[key]) { stopRealtime(key); showRealtimeTranscript(key, ""); }
    else clearRecording(key);
  }));

  // Realtime toggle: in realtime mode the Transcribe button and file upload
  // are not used (transcription happens continuously while recording).
  $("#rt-toggle").addEventListener("change", () => {
    const on = $("#rt-toggle").checked;
    const runBtn = $("[data-run='transcription']");
    if (runBtn) runBtn.disabled = on;
    const fileInput = $("#file-transcription");
    if (fileInput) fileInput.disabled = on;
    if (on) {
      if (realtime["transcription"]) stopRealtime("transcription");
      micMsg("transcription", "Realtime on — press Record to start live transcription.");
    }
  });

  // Stop all microphones (recordings + realtime) if the page is hidden/closed.
  window.addEventListener("pagehide", () => {
    Object.keys(recorders).forEach(key => {
      const r = recorders[key];
      if (r) { try { r.rec.stop(); } catch (e) {} r.stream.getTracks().forEach(t => t.stop()); }
    });
    Object.keys(realtime).forEach(key => { try { stopRealtime(key); } catch (e) {} });
  });

  // ── file inputs: a chosen file is preferred over a recording ──────
  ["transcription", "translation"].forEach(key => {
    $(`#file-${key}`).addEventListener("change", e => {
      if (e.target.files && e.target.files[0]) {
        micMsg(key, "Using uploaded file: " + e.target.files[0].name);
      }
    });
  });

  // ── in-browser audio → WAV normalisation ──────────────────────────
  // The browser mic records webm/opus, which the server's soundfile decoder
  // can't read (it relies on a server-side ffmpeg/librosa fallback for that).
  // To make transcription robust regardless of the server's codec support, we
  // decode the audio here and re-encode it as 16-bit PCM WAV, which soundfile
  // reads natively.
  async function blobToWavBlob(blob) {
    const arrayBuffer = await blob.arrayBuffer();
    const Ctx = window.AudioContext || window.webkitAudioContext;
    const ctx = new Ctx();
    try {
      const audioBuffer = await ctx.decodeAudioData(arrayBuffer);
      return new Blob([audioBufferToWav(audioBuffer)], { type: "audio/wav" });
    } finally {
      ctx.close().catch(() => {});
    }
  }

  function audioBufferToWav(ab) {
    const numChannels = ab.numberOfChannels;
    const length = ab.length * numChannels * 2 + 44;
    const buffer = new ArrayBuffer(length);
    const view = new DataView(buffer);
    let offset = 0;
    const writeString = s => { for (let i = 0; i < s.length; i++) view.setUint8(offset++, s.charCodeAt(i)); };
    const writeUint32 = v => { view.setUint32(offset, v, true); offset += 4; };
    const writeUint16 = v => { view.setUint16(offset, v, true); offset += 2; };

    writeString("RIFF");
    writeUint32(36 + length);
    writeString("WAVE");
    writeString("fmt ");
    writeUint32(16);
    writeUint16(1);               // PCM
    writeUint16(numChannels);
    writeUint32(ab.sampleRate);
    writeUint32(ab.sampleRate * numChannels * 2);
    writeUint16(numChannels * 2);
    writeUint16(16);              // bits per sample
    writeString("data");
    writeUint32(ab.length * numChannels * 2);

    const channels = [];
    for (let i = 0; i < numChannels; i++) channels.push(ab.getChannelData(i));
    let idx = offset;
    for (let i = 0; i < ab.length; i++) {
      for (let c = 0; c < numChannels; c++) {
        const s = Math.max(-1, Math.min(1, channels[c][i]));
        view.setInt16(idx, s < 0 ? s * 0x8000 : s * 0x7fff, true);
        idx += 2;
      }
    }
    return buffer;
  }

  async function getSubmitAudio(key) {
    // Returns {blob, filename} — an uploaded file wins over a recording. The
    // audio is normalised to WAV in the browser (see blobToWavBlob); if that
    // fails we fall back to sending the original bytes.
    const fileInput = $(`#file-${key}`);
    let blob, baseName;
    if (fileInput && fileInput.files && fileInput.files[0]) {
      blob = fileInput.files[0];
      baseName = fileInput.files[0].name.replace(/\.[^.]+$/, "");
    } else if (recordings[key]) {
      blob = recordings[key];
      baseName = `recording-${Date.now()}`;
    } else {
      return null;
    }
    try {
      return { blob: await blobToWavBlob(blob), filename: `${baseName}.wav` };
    } catch (e) {
      const ext = extForMime(blob.type);
      return { blob, filename: `${baseName}.${ext}` };
    }
  }

  // ── proxy fetch helper ────────────────────────────────────────────
  async function proxyFetch(path, { method = "POST", form = null, json = null } = {}) {
    const opts = { method };
    let body;
    if (form) { body = form; }
    else if (json !== null) {
      body = JSON.stringify(json);
      opts.headers = { "Content-Type": "application/json" };
    }
    opts.body = body;
    const res = await fetch("/proxy/" + path, opts);
    const ct = res.headers.get("Content-Type") || "";
    const buf = await res.arrayBuffer();
    return { status: res.status, contentType: ct, buffer: buf };
  }

  function decodeText(buf) {
    return new TextDecoder("utf-8").decode(buf);
  }
  function prettyJson(text) {
    try { return JSON.stringify(JSON.parse(text), null, 2); } catch (e) { return text; }
  }
  function safeJson(text) { try { return JSON.parse(text); } catch (e) { return {}; } }
  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
  }

  // ── run handlers ──────────────────────────────────────────────────
  $$("[data-run]").forEach(btn => btn.addEventListener("click", () => {
    btn.disabled = true;
    run(btn.dataset.run).finally(() => { btn.disabled = false; });
  }));

  async function run(kind) {
    try {
      if (kind === "transcription") return await runTranscription();
      if (kind === "translation") return await runTranslation();
      if (kind === "speech") return await runSpeech();
      if (kind === "clone-new") return await runCloneNew();
    } catch (e) {
      setStatus(kind, "err", "Request failed: " + e.message);
    }
  }

  async function runTranscription() {
    const key = "transcription";
    clearStatus(key);
    setStatus(key, "warn", "Preparing audio…");
    let src;
    try {
      src = await getSubmitAudio(key);
    } catch (e) {
      setStatus(key, "err", "Could not prepare audio: " + e.message);
      return;
    }
    if (!src) { setStatus(key, "err", "Record audio or choose a file first."); return; }
    const form = new FormData();
    form.append("file", src.blob, src.filename);
    form.append("model", $("#t-model").value.trim() || "whisper");
    const lang = $("#t-language").value.trim();
    if (lang) form.append("language", lang);
    form.append("response_format", $("#t-format").value);
    form.append("temperature", $("#t-temp").value || "0.0");
    const prompt = $("#t-prompt").value.trim();
    if (prompt) form.append("prompt", prompt);

    clearStatus(key);
    setStatus(key, "warn", "Transcribing…");
    const t0 = performance.now();
    const res = await proxyFetch("v1/audio/transcriptions", { form });
    const ms = Math.round(performance.now() - t0);
    const text = decodeText(res.buffer);
    if (!res.status.toString().startsWith("2")) {
      setStatus(key, "err", `HTTP ${res.status} in ${ms}ms`);
      showResult(key, `error (${ms}ms)`, prettyJson(text));
      return;
    }
    setStatus(key, "ok", `OK · HTTP ${res.status} · ${ms}ms`);
    const showRaw = $(`[data-raw="${key}"]`).checked;
    if (showRaw) showResult(key, `raw response (${ms}ms)`, prettyJson(text));
    else {
      let display = text;
      try { display = JSON.parse(text).text; } catch (e) {}
      showResult(key, `transcript (${ms}ms)`, display);
    }
  }

  async function runTranslation() {
    const key = "translation";
    clearStatus(key);
    setStatus(key, "warn", "Preparing audio…");
    let src;
    try {
      src = await getSubmitAudio(key);
    } catch (e) {
      setStatus(key, "err", "Could not prepare audio: " + e.message);
      return;
    }
    if (!src) { setStatus(key, "err", "Record audio or choose a file first."); return; }
    const form = new FormData();
    form.append("file", src.blob, src.filename);
    form.append("model", $("#tr-model").value.trim() || "whisper");
    form.append("response_format", $("#tr-format").value);
    const prompt = $("#tr-prompt").value.trim();
    if (prompt) form.append("prompt", prompt);

    clearStatus(key);
    setStatus(key, "warn", "Translating…");
    const t0 = performance.now();
    const res = await proxyFetch("v1/audio/translations", { form });
    const ms = Math.round(performance.now() - t0);
    const text = decodeText(res.buffer);
    if (!res.status.toString().startsWith("2")) {
      setStatus(key, "err", `HTTP ${res.status} in ${ms}ms`);
      showResult(key, `error (${ms}ms)`, prettyJson(text));
      return;
    }
    setStatus(key, "ok", `OK · HTTP ${res.status} · ${ms}ms`);
    const showRaw = $(`[data-raw="${key}"]`).checked;
    if (showRaw) showResult(key, `raw response (${ms}ms)`, prettyJson(text));
    else {
      let display = text;
      try { display = JSON.parse(text).text; } catch (e) {}
      showResult(key, `translation (${ms}ms)`, display);
    }
  }

  async function runSpeech() {
    const key = "speech";
    const text = $("#s-input").value.trim();
    if (!text) { setStatus(key, "err", "Enter some text to synthesise."); return; }
    const payload = {
      model: $("#s-model").value.trim() || "qwen3-tts",
      input: text,
      voice: $("#s-voice").value.trim() || "alloy",
      response_format: $("#s-format").value,
      speed: parseFloat($("#s-speed").value || "1.0"),
    };
    const instructions = $("#s-instructions").value.trim();
    if (instructions) payload.instructions = instructions;

    clearStatus(key);
    setStatus(key, "warn", "Synthesising…");
    const t0 = performance.now();
    const res = await proxyFetch("v1/audio/speech", { json: payload });
    const ms = Math.round(performance.now() - t0);
    const wrap = $("#tab-speech [data-audio]");
    const audio = wrap.querySelector("audio");

    if (!res.status.toString().startsWith("2")) {
      setStatus(key, "err", `HTTP ${res.status} in ${ms}ms`);
      showResult(key, `error (${ms}ms)`, prettyJson(decodeText(res.buffer)));
      wrap.classList.remove("show");
      return;
    }
    setStatus(key, "ok", `OK · HTTP ${res.status} · ${ms}ms · ${(res.buffer.byteLength / 1024).toFixed(0)} KB`);
    const fmt = $("#s-format").value;
    const mime = { mp3: "audio/mpeg", wav: "audio/wav", flac: "audio/flac" }[fmt] || "audio/mpeg";
    if (audio.src && audio.src.startsWith("blob:")) URL.revokeObjectURL(audio.src);
    const url = URL.createObjectURL(new Blob([res.buffer], { type: mime }));
    audio.src = url;
    wrap.classList.add("show");
    audio.play().catch(() => {});
    showResult(key, `audio (${ms}ms)`, `Generated ${(res.buffer.byteLength / 1024).toFixed(1)} KB of ${fmt}.`);
  }

  // ── clones ────────────────────────────────────────────────────────
  async function refreshClones() {
    clearStatus("clones");
    const wrap = $("#cloneTableWrap");
    wrap.innerHTML = '<p class="muted">Loading…</p>';
    try {
      const res = await proxyFetch("clone/list", { method: "GET" });
      if (!res.status.toString().startsWith("2")) {
        wrap.innerHTML = `<p class="muted">HTTP ${res.status}: ${escapeHtml(decodeText(res.buffer))}</p>`;
        return;
      }
      let clones = [];
      try { clones = JSON.parse(decodeText(res.buffer)); } catch (e) {}
      if (!Array.isArray(clones) || clones.length === 0) {
        wrap.innerHTML = '<p class="muted">No clones yet. Create one below.</p>';
        return;
      }
      let rows = clones.map(c => {
        const desc = c.description ? `<span class="muted">${escapeHtml(c.description)}</span>` : "";
        return `<tr>
          <td><span class="badge id">#${c.id}</span></td>
          <td><strong>${escapeHtml(c.name)}</strong><br>${desc}</td>
          <td>
            <button class="secondary" data-setdefault="${c.id}">Set default</button>
            <button class="danger" data-del="${c.id}">Delete</button>
          </td>
        </tr>`;
      }).join("");
      wrap.innerHTML = `<table class="table">
        <thead><tr><th>ID</th><th>Name / description</th><th>Actions</th></tr></thead>
        <tbody>${rows}</tbody></table>`;
      $$("#cloneTableWrap [data-setdefault]").forEach(b =>
        b.addEventListener("click", () => {
          b.disabled = true;
          setDefault(b.dataset.setdefault).finally(() => { b.disabled = false; });
        }));
      $$("#cloneTableWrap [data-del]").forEach(b =>
        b.addEventListener("click", () => {
          b.disabled = true;
          delClone(b.dataset.del).finally(() => { b.disabled = false; });
        }));
    } catch (e) {
      wrap.innerHTML = `<p class="muted">Failed: ${escapeHtml(e.message)}</p>`;
    }
  }

  async function setDefault(id) {
    setStatus("clones", "warn", `Setting #${id} as default…`);
    try {
      const res = await proxyFetch("clone/default", { json: { clone: Number(id) } });
      const text = decodeText(res.buffer);
      if (res.status.toString().startsWith("2")) {
        const j = safeJson(text);
        setStatus("clones", "ok", `Default set to ${j.name || "#" + id}${j.warning ? " (warning)" : ""}`);
      } else {
        setStatus("clones", "err", `HTTP ${res.status}: ${safeJson(text).error || text}`);
      }
    } catch (e) {
      setStatus("clones", "err", "Request failed: " + e.message);
    }
  }

  async function delClone(id) {
    if (!confirm(`Delete clone #${id}? This cannot be undone.`)) return;
    setStatus("clones", "warn", `Deleting #${id}…`);
    try {
      const res = await proxyFetch("clone/delete/", { json: { clone: Number(id) } });
      const text = decodeText(res.buffer);
      if (res.status.toString().startsWith("2")) {
        setStatus("clones", "ok", `Deleted #${id}`);
        refreshClones();
      } else {
        setStatus("clones", "err", `HTTP ${res.status}: ${safeJson(text).error || text}`);
      }
    } catch (e) {
      setStatus("clones", "err", "Request failed: " + e.message);
    }
  }

  async function runCloneNew() {
    const key = "clone-new";
    const name = $("#c-name").value.trim();
    const transcription = $("#c-transcription").value.trim();
    const fileInput = $("#c-file");
    if (!name) { setStatus(key, "err", "A name is required."); return; }
    if (!transcription) { setStatus(key, "err", "A transcription is required."); return; }
    if (!fileInput.files || !fileInput.files[0]) { setStatus(key, "err", "Choose an MP3 file."); return; }

    const form = new FormData();
    form.append("file", fileInput.files[0], fileInput.files[0].name);
    form.append("name", name);
    form.append("transcription", transcription);
    const desc = $("#c-desc").value.trim();
    if (desc) form.append("description", desc);

    clearStatus(key);
    setStatus(key, "warn", "Creating clone…");
    const t0 = performance.now();
    const res = await proxyFetch("clone/new", { form });
    const ms = Math.round(performance.now() - t0);
    const text = decodeText(res.buffer);
    if (res.status.toString().startsWith("2")) {
      const j = safeJson(text);
      setStatus(key, "ok", `Created #${j.id} "${j.name}" in ${ms}ms`);
      showResult(key, `response (${ms}ms)`, prettyJson(text));
      $("#c-name").value = ""; $("#c-desc").value = ""; $("#c-transcription").value = "";
      fileInput.value = "";
      refreshClones();
    } else {
      setStatus(key, "err", `HTTP ${res.status} in ${ms}ms`);
      showResult(key, `error (${ms}ms)`, prettyJson(text));
    }
  }

  // ── init ──────────────────────────────────────────────────────────
  loadConfig();
})();
</script>
</body>
</html>
"""

UI_HTML_BYTES = UI_HTML.encode("utf-8")


if __name__ == "__main__":
    sys.exit(main())
