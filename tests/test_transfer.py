import hashlib
import tempfile
import unittest
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from scripts.prefetch_weights import transfer


class TransferTests(unittest.TestCase):
    def test_interrupted_body_resumes_remaining_bytes(self):
        data = b"0123456789"
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                requests.append(self.headers.get("Range"))
                if len(requests) == 1:
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data[:3])
                else:
                    self.send_response(206)
                    self.send_header("Content-Range", "bytes 3-9/10")
                    self.send_header("Content-Length", "7")
                    self.end_headers()
                    self.wfile.write(data[3:])

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                transfer(f"http://127.0.0.1:{server.server_port}/weights", root / "final", root / "partial",
                         len(data), hashlib.sha256(data).hexdigest())
                self.assertEqual((root / "final").read_bytes(), data)
                self.assertEqual(requests, [None, "bytes=3-"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_git_blob_verification(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = b"small config fixture"
            (root / "source").write_bytes(data)
            blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
            transfer((root / "source").as_uri(), root / "final", root / "partial", len(data), git_blob_id=blob)
            self.assertEqual((root / "final").read_bytes(), data)

    def test_verified_atomic_promotion(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = b"offline fixture, not model weights"
            source = root / "source"
            source.write_bytes(data)
            target, temporary = root / "model/weights", root / "tmp/partial"
            result = transfer(source.as_uri(), target, temporary, len(data), hashlib.sha256(data).hexdigest())
            self.assertEqual(target.read_bytes(), data)
            self.assertFalse(temporary.exists())
            self.assertEqual(result["bytes"], len(data))

    def test_bad_hash_cannot_replace_existing_valid_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, target = root / "source", root / "target"
            source.write_bytes(b"bad")
            target.write_bytes(b"existing")
            with self.assertRaises(ValueError):
                transfer(source.as_uri(), target, root / "partial", 3, "0" * 64)
            self.assertEqual(target.read_bytes(), b"existing")


if __name__ == "__main__":
    unittest.main()
