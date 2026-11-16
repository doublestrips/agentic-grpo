"""Prefetch later immutable shards while the primary downloader progresses.

Use separate temporary files and atomic promotion only after SHA256 validation.
The primary downloader will independently validate and reuse completed shards.
"""
import argparse
import hashlib
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def transfer(url, destination, temporary, size, sha256):
    destination, temporary = Path(destination), Path(temporary)
    temporary.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url)
    digest = hashlib.sha256()
    total = 0
    with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as out:
        while chunk := response.read(8 * 1024 * 1024):
            out.write(chunk)
            digest.update(chunk)
            total += len(chunk)
        out.flush()
        os.fsync(out.fileno())
    if total != size or digest.hexdigest() != sha256:
        raise ValueError(f"prefetch size/SHA256 mismatch: {destination.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary.replace(destination)
    return {"file": destination.name, "bytes": total, "sha256": sha256}


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
