#!/usr/bin/env python3
"""Run on the personal computer. Upstream secret stays in memory, not on GPU host.

The reverse-forward endpoint is accessible by other users of the same Unix UID.
Restrict its lifetime and request budget; this is not account-level isolation.
"""
import argparse
import getpass
import json
import ssl
import sys
import threading
import re
import shutil
import os
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=18761)
    p.add_argument("--max-requests", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--daemon", action="store_true", help="detach after hidden key input; local computer only")
    p.add_argument("--log", default="logs/local-proxy.log")
    p.add_argument("--pid-file", default="logs/local-proxy.pid")
    p.add_argument("--model-only", action="store_true", help="public model relay only; no key read or API calls")
    args = p.parse_args()
    if args.model_only:
        key = ""
        args.max_requests = 0
    else:
        print("Waiting for API key on stdin (not logged)", flush=True)
        key = (getpass.getpass("API key (hidden): ") if sys.stdin.isatty() else sys.stdin.readline()).strip()
    if not key and not args.model_only:
        raise SystemExit("No key supplied")
    if args.daemon:
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
        if os.fork():
            return
        os.setsid()
        if os.fork():
            os._exit(0)
        log = os.open(args.log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        null = os.open(os.devnull, os.O_RDONLY)
        os.dup2(null, 0)
        os.dup2(log, 1)
        os.dup2(log, 2)
        Path(args.pid_file).write_text(str(os.getpid()) + "\n")
    lock = threading.Lock()
    state = {"requests": 0}
    # Homebrew Python's default CA store can omit the system proxy CA.
    # Keep certificate validation enabled; use the OS CA bundle when present.
    tls = ssl.create_default_context(cafile="/etc/ssl/cert.pem" if Path("/etc/ssl/cert.pem").exists() else None)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, code, body):
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/hf/"):
                # Fixed read-only official model origin. Never attach API key.
                path = self.path[3:]
                allowed = re.fullmatch(r"/api/models/Qwen/Qwen2\.5-(3|7)B-Instruct/revision/main\?blobs=true", path)
                allowed = allowed or re.fullmatch(r"/Qwen/Qwen2\.5-(3|7)B-Instruct/resolve/[0-9a-f]{40}/[A-Za-z0-9_.-]+", path)
                if not allowed:
                    return self.reply(403, b'{"error":"model path not allowed"}')
                try:
                    headers = {}
                    if self.headers.get("Range"):
                        if not re.fullmatch(r"bytes=[0-9]+-", self.headers["Range"]):
                            return self.reply(400, b'{"error":"unsupported range"}')
                        headers["Range"] = self.headers["Range"]
                    request = urllib.request.Request("https://huggingface.co" + path, headers=headers)
                    with urllib.request.urlopen(request, context=tls, timeout=120) as upstream:
                        self.send_response(upstream.status)
                        self.send_header("Content-Type", upstream.headers.get("Content-Type", "application/octet-stream"))
                        if upstream.headers.get("Content-Length"):
                            self.send_header("Content-Length", upstream.headers["Content-Length"])
                        if upstream.headers.get("Content-Range"):
                            self.send_header("Content-Range", upstream.headers["Content-Range"])
                        self.end_headers()
                        shutil.copyfileobj(upstream, self.wfile, length=1024 * 1024)
                except Exception as error:
                    print(json.dumps({"model_relay_error_type": type(error).__name__, "path": path}), flush=True)
                    # Close incomplete file; downloader verifies size and hash.
                    self.close_connection = True
                return
            if self.path not in ("/models", "/v1/models"):
                return self.reply(404, b'{"error":"unsupported path"}')
            self.reply(200, b'{"object":"list","data":[{"id":"deepseek-flash","object":"model"}]}')

        def do_POST(self):
            if args.model_only:
                return self.reply(403, b'{"error":"API disabled in model-only mode"}')
            if self.path not in ("/chat/completions", "/v1/chat/completions"):
                return self.reply(404, b'{"error":"unsupported path"}')
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1024 * 1024:
                    raise ValueError("payload size")
                data = json.loads(self.rfile.read(length))
                if data.get("model") != "deepseek-flash" or data.get("stream", False):
                    raise ValueError("model/stream")
                if not 1 <= data.get("max_tokens", 0) <= args.max_tokens:
                    raise ValueError("max_tokens")
                if set(data) - {"model", "messages", "temperature", "max_tokens", "thinking", "seed"}:
                    raise ValueError("unsupported parameters")
                # DeepSeek seed is not documented. Do not imply reproducibility.
                data.pop("seed", None)
                data["thinking"] = {"type": "disabled"}
            except (ValueError, TypeError):
                return self.reply(400, b'{"error":"proxy request validation failed"}')
            with lock:
                if state["requests"] >= args.max_requests:
                    return self.reply(429, b'{"error":"local proxy request budget exhausted"}')
                state["requests"] += 1
                number = state["requests"]
            req = urllib.request.Request("https://api.deepseek.com/chat/completions",
                data=json.dumps(data).encode(), headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
            try:
                with urllib.request.urlopen(req, context=tls, timeout=120) as response:
                    body, status = response.read(), response.status
            except urllib.error.HTTPError as error:
                status = error.code
                # Never forward remote error bodies that could contain secrets.
                body = json.dumps({"error": {"message": "DeepSeek upstream HTTP error", "status": status}}).encode()
            except Exception as error:
                status, body = 502, b'{"error":"local proxy upstream connection failed"}'
                print(json.dumps({"upstream_error_type": type(error).__name__}), flush=True)
            print(json.dumps({"request": number, "status": status}), flush=True)
            self.reply(status, body)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(json.dumps({"listening": args.port, "max_requests": args.max_requests, "max_tokens": args.max_tokens}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
