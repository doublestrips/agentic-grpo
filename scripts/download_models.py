"""Resolve immutable model revision, then download sequentially without GPUs."""
import hashlib
import json
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from eval.storage import atomic_json
from scripts.prefetch_weights import transfer


def main():
    root = Path(os.environ["AGENTIC_ROOT"])
    project = root / "projects/agentic-grpo"
    relay = os.environ.get("MODEL_RELAY_URL")
    api = HfApi() if not relay else None
    plan_path = project / "artifacts/model-download-plan.json"
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else {}
    if relay:
        for model in ("Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct"):
            if model not in plan:
                with urllib.request.urlopen(f"{relay}/api/models/{model}/revision/main?blobs=true", timeout=120) as response:
                    plan[model] = json.load(response)
                atomic_json(plan_path, plan)
    records = []
    for model in ("Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct"):
        if relay:
            info = plan[model]
            revision = info["sha"]
        else:
            info = api.model_info(model)
            revision = info.sha
        dest = root / "models" / model.split("/")[-1] / revision
        print(json.dumps({"model": model, "revision": revision, "destination": str(dest)}), flush=True)
        if relay:
            dest.mkdir(parents=True, exist_ok=True)
            for item in info["siblings"]:
                name = item["rfilename"]
                if "/" in name or not (name.endswith((".json", ".safetensors", ".txt", ".jinja", ".model")) or name in ("LICENSE", "README.md")):
                    continue
                path = dest / name
                expected_size = item.get("size")
                lfs_sha = (item.get("lfs") or {}).get("sha256")
                valid = path.exists() and path.stat().st_size == expected_size
                if valid and lfs_sha:
                    with path.open("rb") as src:
                        valid = hashlib.file_digest(src, "sha256").hexdigest() == lfs_sha
                elif valid:
                    valid = hashlib.sha1(f"blob {expected_size}\0".encode() + path.read_bytes()).hexdigest() == item["blobId"]
                if not valid:
                    temporary = path.with_name(path.name + ".partial")
                    print(json.dumps({"downloading": name, "expected_bytes": expected_size}), flush=True)
                    transfer(f"{relay}/{model}/resolve/{revision}/{name}", path, temporary,
                             expected_size, lfs_sha, None if lfs_sha else item["blobId"])
        else:
            snapshot_download(model, revision=revision, local_dir=dest,
                              allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model", "LICENSE", "README.md"], max_workers=2)
        files = []
        for path in sorted(dest.rglob("*")):
            if path.is_file() and ".cache" not in path.parts:
                digest = hashlib.sha256()
                with path.open("rb") as src:
                    for chunk in iter(lambda: src.read(8 * 1024 * 1024), b""):
                        digest.update(chunk)
                files.append({"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()})
        records.append({"model": model, "revision": revision, "path": str(dest), "files": files,
                        "origin": "https://huggingface.co/" + model, "transport": "local read-only relay" if relay else "direct HF Hub",
                        "recorded_at": datetime.now(timezone.utc).isoformat()})
        atomic_json(project / "artifacts/model-manifest.json", records)
        print(json.dumps({"model": model, "files": len(files), "bytes": sum(f["bytes"] for f in files)}), flush=True)


if __name__ == "__main__":
    main()
