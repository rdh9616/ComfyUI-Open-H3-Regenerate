"""CPU IC contracts. Run with ComfyUI's Python:

python -B tests/test_ic.py <ComfyUI directory> <scratch directory> [pinned before_denoise.py]

The optional oracle is Diffusers 8c54b38fea3d624d23d0e43c0c96037f080a79f9:
src/diffusers/modular_pipelines/minimax_h3/before_denoise.py (blob 3cf5a7048a5d8f10b81bb54eb941b4a3eda57bc6).
Only its pure layout functions are evaluated, after verifying the Git blob hash. No downloads or weights.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

ORACLE = Path(sys.argv[3]) if len(sys.argv) > 3 else None
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_comfy_nodes as base  # sets up native ComfyUI on CPU, without running its tests
import numpy as np
import torch
from open_h3_regen import ic_core as ic, ic_nodes
from comfy.ldm.minimax.model import PackedLayout, patchify_video
from comfy.model_sampling import CONST

torch.set_num_threads(4)


def rejects(call, text):
    try:
        call()
    except ValueError as e:
        assert text in str(e), str(e)
    else:
        raise AssertionError(f"expected error containing {text!r}")


def test_draft():
    for n in (113, 121, 124):
        frames = torch.arange(n).float()[:, None, None, None].expand(n, 4, 8, 3) / 255
        out, count = ic.prepare_draft(frames, 24, (4, 8))
        assert count == n and out.shape == (124, 4, 8, 3)
        assert torch.equal(out[:n], frames) and torch.equal(out[-1], frames[-1])
    assert ic.draft_indices(62, 12) == [i // 2 for i in range(124)]
    assert ic.draft_indices(155, 30) == [int(i * 30 / 24) for i in range(124)]
    # Floor sampling differs from the legacy nearest-frame rule at i=1, fps=30.
    assert ic.draft_indices(155, 30)[1] == 1
    for n in (0, 112, 125, 362):
        rejects(lambda: ic.draft_indices(n, 24), "IC")
    for fps in (0, -1, float("nan"), float("inf")):
        rejects(lambda: ic.draft_indices(124, fps), "frame rate")
    # Independent one-frame cover resize/crop, as in Sana IC/media.py.
    x = torch.rand(113, 7, 13, 3)
    out, _ = ic.prepare_draft(x, 24, (4, 8), chunk=9)
    expected = torch.nn.functional.interpolate(x.movedim(-1, 1), (4, 8), mode="bilinear",
                                               align_corners=False, antialias=True).movedim(1, -1)
    assert torch.equal(out[:113], expected)
    assert ic.instruction("A fox.") == ic.TASK + "\n\nScene description: A fox."
    assert ic.instruction("").endswith("Scene description: ")
    print("IC draft: floor resampling, cover crop, tail pad, range rejection, instruction PASS")


def test_posterior():
    g = torch.Generator().manual_seed(8)
    moments = torch.randn((1, 48, 7, 4, 6), generator=g)
    moments[:, 24, 0, 0, 0], moments[:, 25, 0, 0, 0] = -100, 100
    offset, scale = torch.arange(24).float() / 24, torch.arange(24).float() / 24 + 1
    mean, logvar = moments.chunk(2, 1)
    noise = torch.randn(mean.shape, generator=torch.Generator().manual_seed(42))
    expected = ((mean + torch.exp(0.5 * logvar.clamp(-30, 20)) * noise - offset[None, :, None, None, None]) /
                scale[None, :, None, None, None]).bfloat16()
    state = torch.random.get_rng_state()
    actual = ic.sample_posterior(moments, offset, scale, 42)
    assert torch.equal(actual, expected) and torch.equal(state, torch.random.get_rng_state())
    assert torch.equal(actual, ic.sample_posterior(moments, offset, scale, 42))
    assert not torch.equal(actual, ic.sample_posterior(moments, offset, scale, 43))
    assert not torch.equal(actual, ((mean - offset[None, :, None, None, None]) /
                                   scale[None, :, None, None, None]).bfloat16())
    rejects(lambda: ic.sample_posterior(moments[:, :24], offset, scale, 42), "posterior")
    print("IC posterior: seeded sample, clamped logvar, normalization, RNG isolation PASS")


def test_encode_and_target():
    ref = torch.zeros(1, 24, 37, 48, 84)
    target = torch.full((1, 24, 37, 90, 160), 0.25)
    # Expanded pixels keep this a tiny-memory mocked encoder boundary check, not a GPU encode.
    pixels = torch.full((124, 1, 1, 3), 0.501).expand(124, 768, 1344, 3)

    class Clip:
        def tokenize(self, prompt, **kwargs):
            self.prompt, self.refs = prompt, kwargs["minimax_ref_items"]
            return "tokens"

        def encode_from_tokens_scheduled(self, tokens):
            assert tokens == "tokens"
            return [[torch.zeros(1, 4, 5120), {"minimax_token_tags": torch.tensor([1, 0, 0, 1])}]]

    clip = Clip()
    with patch.object(ic_nodes, "encode_draft", return_value=ref) as encode:
        cond, lat, prompt = ic_nodes.OpenH3ICEncode().encode(clip, object(), pixels, "A fox.", 42)
    assert encode.call_args.args[1] is pixels and encode.call_args.args[2] == 42
    assert prompt == clip.prompt == ic.instruction("A fox.")
    assert len(clip.refs) == 1 and clip.refs[0]["data"].shape[0] == 11
    assert clip.refs[0]["timestamps"] == [i / 2 for i in range(11)]
    assert torch.all(clip.refs[0]["data"] == round(0.501 * 255) / 255)
    block = cond[0][1]["minimax_refs"][0]
    assert block["latent"] is lat["samples"] is ref and block["audio_latent"] is None
    assert block["latent_t"] == 37 and cond[0][1]["minimax_visual_cond_noise_aug"] == 0.999
    rejects(lambda: ic.validate_conditioning([[torch.zeros(1, 2, 5120), {}]]), "vision/text tags")

    class Audio:
        from comfy.sd import VAE

        audio_sample_rate = 32000
        crop_input, output_channels = True, 2
        spacial_compression_encode = staticmethod(lambda: 800)
        crop = VAE.vae_encode_crop_pixels

        def encode(self, x):
            assert x.count_nonzero() == 0
            # Exercise native pre-encode cropping, which caused 206 rather than 207 audio latents.
            cropped = self.crop(x)
            return torch.full((1, 32, 2, cropped.shape[1] // 800), 0.125)

    for schedule, sigma in (("up3", 0.834942), ("up4", 0.923077), ("hf8", 1.0)):
        result, = ic_nodes.OpenH3ICTargetLatent().build(lat, {"samples": target}, Audio(), schedule)
        video, sound = result["samples"].tensors
        vm, am = result["noise_mask"].tensors
        assert torch.all(vm == 1) and torch.all(am == 0) and torch.all(sound == 0.125)
        noise = torch.full_like(video, 0.6)
        x = CONST().noise_scaling(torch.tensor(sigma), noise, video)
        expected = noise if schedule == "hf8" else (1 - sigma) * target + sigma * noise
        assert torch.allclose(x, expected)
        assert torch.all(target == 0.25) and torch.all(ref == 0)  # no input mutation
    rejects(lambda: ic.validate_pair(ref[:, :, :32], target), "T=37")
    rejects(lambda: ic.validate_pair(ref, target[:, :, :, :-2]), "2560x1440")
    rejects(lambda: ic.validate_pair(ref[:, :, :, :, :-2], target), "1344x768")
    rejects(lambda: ic.validate_pair(ref, torch.zeros(2, 24, 37, 90, 160)), "[1, 24")
    print("IC nodes: draft/Qwen/reference wiring, shared shape, silence mask, target re-noise PASS")


def test_vae_facade():
    from comfy.sd import VAE
    from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE, ResnetShortcut3D
    import comfy.model_management as mm

    model = MiniMaxH3VideoVAE(encoder=False, decoder=False)
    model.encoder = torch.nn.Identity()
    moments = torch.randn(1, 48, 37, 4, 6)
    calls = []

    def temporal(pixels, device):
        assert pixels.shape == (1, 3, 124, 32, 32) and device == torch.device("cpu")
        calls.append(1)
        return moments

    model.encode_temporal = temporal

    class Wrapper:
        encode = VAE.encode  # exercise the real ComfyUI wrapper and its fallback
        _encode_tiled_owned = VAE._encode_tiled_owned
        latent_dim, not_video, disable_offload, handles_tiling = 3, False, False, True
        vae_dtype = torch.float32
        device = output_device = torch.device("cpu")
        format_encoded = None
        patcher = SimpleNamespace(get_free_memory=lambda device: 100000)
        memory_used_encode = staticmethod(lambda shape, dtype: 100)
        process_input = staticmethod(lambda x: x * 2 - 1)
        vae_encode_crop_pixels = staticmethod(lambda x: x)
        vae_output_dtype = staticmethod(lambda: torch.float32)
        throw_exception_if_invalid = staticmethod(lambda: None)

    vae = Wrapper()
    vae.first_stage_model = model
    images = torch.zeros(124, 32, 32, 3)
    expected = ic.sample_posterior(moments, model.latents_mean, model.latents_std, 17).float()
    with patch.object(mm, "load_models_gpu"):
        actual = ic_nodes.encode_draft(vae, images, 17)
        assert torch.equal(actual, expected) and len(calls) == 1
        assert vae.first_stage_model is model and model.encode_temporal is temporal
        # The first attempt OOMs before encoding; fallback must still sample, never use mode.
        with patch.object(mm, "load_models_gpu", side_effect=torch.OutOfMemoryError("synthetic test")):
            fallback = ic_nodes.encode_draft(vae, images, 17)
        assert torch.equal(fallback, expected) and len(calls) == 2
    vae.vae_dtype = torch.float16
    rejects(lambda: ic_nodes.encode_draft(vae, images, 17), "FP32")
    vae.vae_dtype = torch.float32
    model.encoder = ResnetShortcut3D(32, 32)
    rejects(lambda: ic_nodes.encode_draft(vae, images, 17), "LightVAE")
    print("IC VAE facade: real native wrapper, sampled OOM fallback, shared model unchanged PASS")


def oracle_functions(path):
    source = path.read_text(encoding="utf-8").replace("\r\n", "\n").rstrip("\n") + "\n"
    data = source.encode()
    git_blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    assert git_blob == "3cf5a7048a5d8f10b81bb54eb941b4a3eda57bc6", git_blob
    tree = ast.parse(source)
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id.startswith("_ROPE_") for t in node.targets):
            selected.append(node)
        elif isinstance(node, ast.FunctionDef):
            selected.append(node)
        elif isinstance(node, ast.ClassDef):
            for method in node.body:
                if isinstance(method, ast.FunctionDef) and method.name in ("build_ref2va_packed_sequence", "build_row_timesteps"):
                    method.decorator_list = []
                    selected.append(method)
    namespace = {"torch": torch, "np": np}
    # Future annotations avoid needing the pipeline's heavyweight runtime classes for type hints.
    prefix = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[prefix] + selected, type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def test_native_layout(oracle):
    # Same T but different reference/target spatial grids, including real T=37 and a larger text span.
    for t, rh, rw, h, w, a, nt in ((7, 4, 6, 8, 10, 37, 4), (37, 6, 8, 10, 12, 207, 13)):
        ref = torch.arange(24 * t * rh * rw).reshape(1, 24, t, rh, rw).float() / 100
        tags = torch.ones(nt, dtype=torch.long)
        tags[1:3] = 0
        blk = {"kind": "video", "latent_t": t, "latent_h": rh, "latent_w": rw, "ref_audio_t": 0, "latent": ref}
        layout = PackedLayout(nt, t, h, w, a, refs=[blk])
        assert [k for _, _, k in layout.segments] == ["text", "ref_img", "audio", "video"]
        nref = t * (rh // 2) * (rw // 2)
        assert not layout.img_update[:nref].any() and layout.img_update[nref:].all()
        if oracle:
            pos, packed_tags, vi, ai, ti, nv, na = oracle["build_ref2va_packed_sequence"](
                tags, [SimpleNamespace(kind="video", has_audio=False)], [ref], [], t, h, w, a, (1, 2, 2), 2, 2, 0)
            assert torch.allclose(layout.position_ids, pos, atol=1e-12, rtol=0)
            assert torch.equal(layout.img_pos, vi) and torch.equal(layout.audio_pos, ai)
            assert nv == nref and na == 0 and torch.equal(ti, torch.arange(nt))
            assert torch.equal(patchify_video(ref), oracle["patchify_video_latents"](ref, (1, 2, 2)))
            assert torch.equal(packed_tags[:nt], tags)

        dm = base.tiny().model.diffusion_model
        payload = {"refs": [blk], "cond_video_latents": [ref], "seed": 42, "text_token_tags": tags}
        rows = dm._cond_video_rows(payload, torch.device("cpu"))
        raw = patchify_video(ref)
        expected = 0.999 * raw + 0.001 * torch.randn(raw.shape, generator=torch.Generator().manual_seed(42))
        assert torch.allclose(rows, expected) and torch.equal(rows, dm._cond_video_rows(payload, torch.device("cpu")))

        # Capture the real native forward's mod-row assignments before its first transformer block.
        class Captured(Exception):
            pass

        capture = {}

        def observe(args, extra):
            capture["segments"] = args["mod_segments"]
            raise Captured

        hook = dm.time_embedder.register_forward_pre_hook(lambda module, args: capture.update(t=args[0].clone()))
        for sigma in (0.834942, 0.468753, 0.0001):
            try:
                dm._forward([torch.zeros(1, 24, t, h, w), torch.zeros(1, 32, 2, a)], torch.tensor([sigma * 1000]),
                            torch.zeros(1, nt, 32), minimax_payload=payload,
                            denoise_mask=torch.ones(1, 1, t, h, w), audio_denoise_mask=torch.zeros(1, 1, 2, a),
                            transformer_options={"patches_replace": {"dit": {("double_block", 0): observe}}})
            except Captured:
                pass
            row_t, row_tags = torch.empty(layout.seq_len), torch.empty(layout.seq_len, dtype=torch.long)
            for lo, hi, index in capture["segments"]:
                row_t[lo:hi] = capture["t"][index // 3]
                row_tags[lo:hi] = index % 3
            tv = float(1 - (torch.tensor([sigma * 1000])[0] / 1000))
            assert torch.all(row_t[:nt] == tv) and torch.equal(row_tags[:nt], tags)
            assert torch.all(row_t[layout.audio_pos] == 1)
            assert torch.allclose(row_t[layout.img_pos[:nref]], torch.full((nref,), max(tv, 0.999)))
            assert torch.all(row_t[layout.img_pos[nref:]] == tv)
            if oracle:
                levels, indices = oracle["build_row_timesteps"](vi, ai, nv, na, nt, tv, 1., max(tv, .999), 1.)
                assert torch.equal(row_t, levels[indices]) and torch.equal(row_tags, packed_tags)
        hook.remove()
    print("IC native layout: geometry, tags, fixed seeded reference noise, row times, target update mask PASS" +
          (" (pinned Diffusers oracle)" if oracle else " (oracle comparison not requested)"))


if __name__ == "__main__":
    test_draft()
    test_posterior()
    test_vae_facade()
    test_encode_and_target()
    test_native_layout(oracle_functions(ORACLE) if ORACLE else None)
    print("IC CPU contracts passed; no GPU inference or image-quality claim")
