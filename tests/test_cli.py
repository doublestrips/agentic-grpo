"""End-to-end resume checks using a local HTTP fixture, not a real model."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class CLITests(unittest.TestCase):
    def test_concurrent_resume_and_config_rejection(self):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append(payload)
                if "tools" in payload:
                    text = "I have completed checking your request."
                else:
                    text = "Hello" if len(payload["messages"]) == 2 else "###STOP###"
                result = {"id": "synthetic-test", "object": "chat.completion", "created": 0, "model": "synthetic-test",
                    "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}}
                body = json.dumps(result).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                project = root / "projects/agentic-grpo"
                source = Path(__file__).resolve().parents[1]
                shutil.copytree(source, project, ignore=shutil.ignore_patterns(".git", "__pycache__", "logs", "results", "artifacts"))
                (root / "sources").mkdir()
                (root / "sources/tau-bench").symlink_to(Path(os.environ["AGENTIC_ROOT"]) / "sources/tau-bench")

                def git(*args):
                    subprocess.run(["git", *args], cwd=project, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                git("init")
                git("config", "user.name", "offline-test")
                git("config", "user.email", "offline@example.invalid")
                git("add", ".")
                git("commit", "-m", "offline test fixture")
                env = dict(os.environ, AGENTIC_ROOT=str(root), PYTHONPATH=f"{project}:{root / 'sources/tau-bench'}")
                for prefix in ("POLICY", "USER_SIM"):
                    env.update({prefix + "_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
                                prefix + "_MODEL": "synthetic-test", prefix + "_API_KEY": "synthetic-no-secret"})
                command = [sys.executable, "-m", "eval.run_eval", "--domain", "retail", "--task-ids", "0", "1", "--trials", "2",
                           "--concurrency", "2", "--user-concurrency", "1", "--output", "results/offline"]

                def run(extra=()):
                    return subprocess.run(command + list(extra), cwd=project, env=env, capture_output=True, text=True)

                first = run()
                self.assertEqual(first.returncode, 0, first.stderr)
                result_dir = project / "results/offline"
                metrics = json.loads((result_dir / "metrics.json").read_text())
                self.assertEqual(metrics["recorded_trials"], 4)
                calls_after_first = len(requests)
                self.assertEqual(run(["--resume"]).returncode, 0)
                self.assertEqual(len(requests), calls_after_first)
                self.assertNotEqual(run(["--resume", "--seed", "99"]).returncode, 0)
                self.assertEqual(len(requests), calls_after_first)
                # Simulate interrupted storage: retain completed trials; fill only missing key.
                file = result_dir / "trials/task-0000-trial-00.json"
                file.unlink()
                restored = run(["--resume"])
                self.assertEqual(restored.returncode, 0, restored.stderr)
                self.assertEqual(len(requests) - calls_after_first, 3)
                self.assertEqual(len(list((result_dir / "trials").glob("*.json"))), 4)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
