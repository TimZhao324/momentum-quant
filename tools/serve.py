"""Serve local web dashboard on http://127.0.0.1:8000.

Usage:
    .venv/bin/python tools/serve.py        (Ctrl+C to stop)
"""

import http.server
import os
import socketserver

PORT = 8000
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(ROOT, "web")

if not os.path.exists(os.path.join(WEB, "data", "overview.json")):
    raise SystemExit("web/data/overview.json is missing. Run: .venv/bin/python tools/pipeline.py")

os.chdir(WEB)


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        """Append Cache-Control no-store header to bypass browser caching."""
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        """Silence per-request access logs."""
        pass


socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(("127.0.0.1", PORT), Handler) as httpd:
    print(f"Dashboard available on http://localhost:{PORT} (Ctrl+C to terminate)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
