"""Read-only checker: find existing subfolders and distinguish size checks from hashes."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile

spec = importlib.util.spec_from_file_location("check_ic_weights", Path(__file__).resolve().parents[1] / "tools/check_ic_weights.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def test_checker():
    content = b"pinned model fixture"
    asset = dict(id="tiny", required=True, repo="author/model", revision="a" * 40, path="lora/tiny.safetensors",
                 destination="loras/H3/tiny.safetensors", size=len(content), sha256=hashlib.sha256(content).hexdigest())
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        assert checker.inspect_asset(root, asset)["status"] == "missing"
        path = root / "loras/other/tiny.safetensors"
        path.parent.mkdir(parents=True)
        path.write_bytes(content)
        assert checker.inspect_asset(root, asset)["status"] == "size_matches_unverified"
        assert checker.inspect_asset(root, asset, True)["status"] == "verified"
        path.write_bytes(b"X" * len(content))
        bad = checker.inspect_asset(root, asset, True)
        assert bad["status"] == "mismatch" and bad["candidates"][0]["status"] == "wrong_hash"
        path.write_bytes(b"short")
        assert checker.inspect_asset(root, asset)["status"] == "mismatch"
        assert path.read_bytes() == b"short"  # checking never repairs or overwrites
    print("IC weight checker: missing/subfolder/size/hash/mismatch/read-only PASS")


if __name__ == "__main__":
    test_checker()
