import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any


class MockServer:
    def __init__(self, log_path: str | Path, host: str = "127.0.0.1", port: int = 0):
        self._log = []
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                self._log.append(json.loads(line))
        self.host = host
        self.port = port
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(inner_self):
                length = int(inner_self.headers.get("Content-Length", "0"))
                data = inner_self.rfile.read(length).decode()
                payload = json.loads(data) if data else {}
                for row in self._log:
                    if row.get("request") == payload:
                        resp = row.get("response", {})
                        body = json.dumps(resp).encode()
                        inner_self.send_response(200)
                        inner_self.send_header("Content-Type", "application/json")
                        inner_self.end_headers()
                        inner_self.wfile.write(body)
                        return
                inner_self.send_response(404)
                inner_self.end_headers()

            def log_message(*args: Any, **kwargs: Any) -> None:  # noqa: D401
                pass

        self._server = HTTPServer((self.host, self.port), Handler)
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return f"http://{self.host}:{self.port}"

    def __exit__(self, exc_type, exc, tb):
        if self._server:
            self._server.shutdown()
        if self._thread:
            self._thread.join()
