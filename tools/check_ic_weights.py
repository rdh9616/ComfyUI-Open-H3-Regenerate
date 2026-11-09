"""Read-only check of pinned IC assets. Never downloads, renames or overwrites model files."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from urllib.parse import quote

MANIFEST = Path(__file__).resolve().parents[1] / "weights" / "ic-manifest.json"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pinned_url(asset):
    return f"https://huggingface.co/{asset['repo']}/resolve/{asset['revision']}/{quote(asset['path'])}"


def candidates(root, asset):
    relative = Path(asset["destination"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("manifest destination must be relative to the models directory")
    expected = root / relative
    found = [expected] if expected.is_file() else []
    # ComfyUI discovers subfolders. Accept an existing copy instead of suggesting another download.
    category = root / relative.parts[0]
    if category.is_dir():
        found.extend(p for p in category.rglob(relative.name) if p.is_file() and p != expected)
    return found


def inspect_asset(root, asset, verify=False):
    matches = candidates(root, asset)
    result = {"id": asset["id"], "required": asset["required"], "url": pinned_url(asset),
              "expected_bytes": asset["size"], "expected_sha256": asset["sha256"], "candidates": []}
    for path in matches:
        size = path.stat().st_size
        entry = {"path": str(path), "bytes": size, "status": "wrong_size"}
        if size == asset["size"]:
            entry["status"] = "size_matches_unverified"
            if verify:
                entry["sha256"] = sha256(path)
                entry["status"] = "verified" if entry["sha256"] == asset["sha256"] else "wrong_hash"
        result["candidates"].append(entry)
    statuses = [entry["status"] for entry in result["candidates"]]
    result["status"] = ("verified" if "verified" in statuses else "size_matches_unverified"
                        if "size_matches_unverified" in statuses else "mismatch" if statuses else "missing")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--only", help="Comma-separated asset IDs; omit to check every asset")
    parser.add_argument("--verify", action="store_true", help="Stream full files and compare SHA256 (can take minutes)")
    args = parser.parse_args()
    root = args.models_dir.resolve()
    if not root.is_dir():
        parser.error("--models-dir must be an existing ComfyUI models directory")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assets = manifest["assets"]
    selected = set(args.only.split(",")) if args.only else {a["id"] for a in assets}
    unknown = selected - {a["id"] for a in assets}
    if unknown:
        parser.error("unknown asset IDs: " + ", ".join(sorted(unknown)))
    results = [inspect_asset(root, asset, args.verify) for asset in assets if asset["id"] in selected]
    print(json.dumps({"manifest": str(MANIFEST), "verification": "sha256" if args.verify else "size_only",
                      "assets": results}, indent=2))
    acceptable = {"verified"} if args.verify else {"size_matches_unverified"}
    return int(any(r["status"] == "mismatch" or
                   ((r["required"] or args.only) and r["status"] not in acceptable) for r in results))


if __name__ == "__main__":
    sys.exit(main())
