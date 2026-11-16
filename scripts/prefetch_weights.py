"""Prefetch later immutable shards while the primary downloader progresses.

Use separate temporary files and atomic promotion only after SHA256 validation.
The primary downloader will independently validate and reuse completed shards.
"""
import argparse
import hashlib
import json
import os
import time
import http.client
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def transfer(url, destination, temporary, size, sha256=None, git_blob_id=None, attempts=5):
    destination, temporary = Path(destination), Path(temporary)
    temporary.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(attempts):
        offset = temporary.stat().st_size if temporary.exists() else 0
        if offset > size:
            raise ValueError(f"partial exceeds expected size: {destination.name}")
        if offset == size:
            break
        try:
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=180) as response:
                append = response.status == 206 and offset > 0
                if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise ValueError(f"invalid resume range: {destination.name}")
                print(json.dumps({"file": destination.name, "attempt": attempt, "resume_from_bytes": offset,
                                  "http_status": response.status, "appending": append}), flush=True)
                with temporary.open("ab" if append else "wb") as out:
                    while chunk := response.read(8 * 1024 * 1024):
                        out.write(chunk)
                    out.flush()
                    os.fsync(out.fileno())
            if temporary.stat().st_size != size:
                raise OSError("incomplete response body")
            break
        except (OSError, urllib.error.URLError, http.client.IncompleteRead) as error:
            if isinstance(error, urllib.error.HTTPError) and error.code not in (408, 429) and error.code < 500:
                raise
            print(json.dumps({"retry": attempt, "file": destination.name, "error_type": type(error).__name__,
                              "partial_bytes": temporary.stat().st_size if temporary.exists() else 0}), flush=True)
            if attempt + 1 == attempts:
                raise
            time.sleep(min(2 ** attempt, 8))
    digest = hashlib.sha256()
    blob = hashlib.sha1(f"blob {size}\0".encode()) if git_blob_id else None
    with temporary.open("rb") as src:
        for chunk in iter(lambda: src.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
            if blob is not None:
                blob.update(chunk)
    if temporary.stat().st_size != size or (sha256 and digest.hexdigest() != sha256) or (blob and blob.hexdigest() != git_blob_id):
        raise ValueError(f"prefetch size/SHA256 mismatch: {destination.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary.replace(destination)
    return {"file": destination.name, "bytes": size, "sha256": digest.hexdigest()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--indices", nargs="+", type=int, default=[3, 4])
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()
    if not 1 <= args.workers <= 2:
        p.error("prefetch worker limit is two")
    root = Path(os.environ["AGENTIC_ROOT"])
    plan = json.loads((root / "projects/agentic-grpo/artifacts/model-download-plan.json").read_text())[args.model]
    revision = plan["sha"]
    files = sorted([f for f in plan["siblings"] if f["rfilename"].endswith(".safetensors")], key=lambda f: f["rfilename"])
    if len(set(args.indices)) != len(args.indices) or any(i < 1 or i > len(files) for i in args.indices):
        p.error("invalid shard selection")
    relay = os.environ.get("MODEL_RELAY_URL", "http://127.0.0.1:18761/hf")

    def worker(index):
        item = files[index - 1]
        name = item["rfilename"]
        destination = root / "models" / args.model.split("/")[-1] / revision / name
        temporary = root / "tmp/model-prefetch" / revision / (name + ".partial")
        print(json.dumps({"prefetch_start": name}), flush=True)
        result = transfer(f"{relay}/{args.model}/resolve/{revision}/{name}", destination, temporary, item["size"], item["lfs"]["sha256"])
        print(json.dumps(result), flush=True)
        return result

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(worker, args.indices))


if __name__ == "__main__":
    main()
