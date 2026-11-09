"""Build additive IC examples from the published graph conventions; never edits legacy examples."""
from pathlib import Path
import json
import uuid

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / "example_workflows"

RECOVERY_NOTE = """# Open-H3 IC - Decode Saved Latent
Recover a completed IC refinement without loading the transformer or repeating its sampling.

1. Choose the exact saved `open_h3_ic_*.safetensors` file in **Load refined latent**. `(newest)` can select a
   different run; check the filename before queueing.
2. **Video VAE** uses the standard H3 decoder. Start the validation instance with `--fp32-vae`.
3. Run. **Finish Frames** uses the stored real frame count and original soundtrack. The final video is 24 fps.

The sampler result must have been saved before decoding, with the original draft audio connected to
**Save Refined Latent**. If sampling itself failed, there is no completed result to recover.

Optional fast decoding: replace **VAE Decode** with **Open-H3 Fast VAE Decode**, choose the pinned v2 decoder,
connect the loaded samples to it, and keep **Video VAE** connected as its memory fallback. This pack's plain
PyTorch decoder does not require FA4. Decoder choice changes the pixels; record it in comparisons.

This workflow does not download models, overwrite a base model, or submit another transformer run.
See `docs/ic-setup.md` for pinned assets and the separate, still-pending matched upstream comparison.
"""


def build_recovery():
    graph = json.loads((WORKFLOWS / "Open-H3 Regenerate 2K - Decode Saved Latent.json").read_text(encoding="utf-8"))
    name = "Open-H3 IC - Decode Saved Latent"
    graph["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, "open-h3/" + name))
    for node in graph["nodes"]:
        kind = node["type"]
        if kind == "MarkdownNote":
            node["widgets_values"] = [RECOVERY_NOTE]
        elif kind == "OpenH3RegenFastVAEDecode":
            node["type"] = "VAEDecode"
            node["title"] = "Standard H3 VAE decode (no sampling)"
            node["properties"] = {"Node name for S&R": "VAEDecode"}
            node["widgets_values"] = []
        elif kind == "OpenH3LoadRefinedLatent":
            node["title"] = "Load refined latent - choose the exact IC file"
        elif kind == "SaveVideo":
            node["widgets_values"][0] = "video/open_h3_ic_recovered"
    return name, graph


def main():
    for name, graph in [build_recovery()]:
        path = WORKFLOWS / (name + ".json")
        path.write_text(json.dumps(graph, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(path.name)


if __name__ == "__main__":
    main()
