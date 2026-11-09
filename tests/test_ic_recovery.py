"""CPU recovery regression: AV latent -> saved soundtrack -> video-only VAE fallback."""
from pathlib import Path
import importlib.util
import sys

COMFY, SCRATCH = sys.argv[1:3]
sys.path.insert(0, COMFY)
sys.argv = [sys.argv[0], "--cpu"]
import comfy.options
comfy.options.enable_args_parsing()
import comfy.nested_tensor
import folder_paths
import nodes
import torch

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("open_h3_ic_recovery", root / "__init__.py",
                                             submodule_search_locations=[str(root)])
pack = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pack
spec.loader.exec_module(pack)


def test_recovery():
    folder_paths.set_output_directory(str(Path(SCRATCH).resolve()))
    video = torch.arange(24 * 2 * 2 * 2, dtype=torch.float32).reshape(1, 24, 2, 2, 2)
    sound = torch.ones(1, 32, 2, 10)
    waveform = torch.arange(400, dtype=torch.float32).reshape(1, 2, 200) / 400
    audio = {"waveform": waveform, "sample_rate": 48000}
    original = {"samples": comfy.nested_tensor.NestedTensor((video, sound))}
    saved = pack.OpenH3SaveRefinedLatent().save(original, 5, "ic_recovery_test", audio)
    assert saved["result"][0] is original
    restored, count, soundtrack = pack.OpenH3LoadRefinedLatent().load(pack.NEWEST)
    assert count == 5 and soundtrack["sample_rate"] == 48000
    assert torch.equal(soundtrack["waveform"], waveform)
    assert torch.equal(restored["samples"].tensors[0], video)
    assert torch.equal(restored["samples"].tensors[1], sound)

    class VideoVAE:
        def decode(self, latent):
            # Native VAEDecode unwraps AV before calling VAE.decode. Its raw VAE expects a tensor.
            assert isinstance(latent, torch.Tensor), "video VAE received an AV NestedTensor"
            assert torch.equal(latent, video)
            return torch.zeros(1, 5, 32, 32, 3)

    expected, = nodes.VAEDecode().decode(VideoVAE(), restored)
    result, = pack.OpenH3RegenFastVAEDecode._native(VideoVAE(), restored, "CPU forced fallback")
    assert result.shape == expected.shape == (5, 32, 32, 3)
    assert torch.equal(result, expected)
    plain, = pack.OpenH3RegenFastVAEDecode._native(VideoVAE(), {"samples": video}, "plain video")
    assert torch.equal(plain, expected)
    print("IC recovery: saved AV/frame count/original soundtrack and native video-only fallback PASS")


if __name__ == "__main__":
    test_recovery()
