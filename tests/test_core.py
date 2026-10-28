"""Checks for regen_core without ComfyUI: sigma grids, frame math, LoRA key mapping, Sharp curve math.

Run:  python -I tests/test_core.py [lora_header.json] [pruned_checkpoint.safetensors]
lora_header.json: the safetensors header of open_h3_regenerate_2k_lora.safetensors (all 312 modules mapped);
the checkpoint: a curve-form H3 model whose adaln_t_table is used for the inversion test.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("regen_core", os.path.join(HERE, "..", "regen_core.py"))
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


def close(a, b, tol=1e-6):
    assert len(a) == len(b) and all(abs(x - y) < tol for x, y in zip(a, b)), (a, b)


def test_sigmas():
    close(core.refine_sigmas(3), [0.909375, 6 / 7, 2 / 3, 0.0])
    close(core.refine_sigmas(2), [0.7, 0.494118, 0.0], 1e-6)
    # the refiner's original 4-step grid (open_h3/sigmas.py docstring)
    close(core.shifted_grid(0.909375, 4), [0.909375, 0.841446, 0.732075, 0.526697, 0.0], 1e-6)
    # the 3-step release grid's later points are shift-6 of 0.5 and 0.25 (6/7, 2/3), not the rule from 0.909375
    close([6 * x / (1 + 5 * x) for x in (0.5, 0.25)], core.refine_sigmas(3)[1:3])
    close(core.refine_sigmas(3, 0.8)[:1], [0.8])
    try:
        core.refine_sigmas(5)
        raise AssertionError("5 steps without start_sigma should fail")
    except ValueError:
        pass


def test_frames():
    assert core.align_frame_count(121) == 124 and core.video_latent_t(124) == 37
    assert core.align_frame_count(360) == 362 and core.video_latent_t(362) == 107
    assert core.align_frame_count(124) == 124 and core.align_frame_count(1) == 5
    for t in range(2, 200, 5):
        assert core.video_latent_t(core.frames_for_latent_t(t)) == t
    assert round(124 / 24 * 40) == 207                      # reference num_audio_latents
    assert core.resample_indices(48, 24.0) == list(range(48))
    assert core.resample_indices(30, 30.0) == [0, 1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 14, 15, 16, 18, 19, 20, 21, 22, 24,
                                               25, 26, 28, 29]
    assert len(core.resample_indices(300, 29.97)) == 240 and core.resample_indices(10, 12.0) == [0, 0, 1, 2, 2, 2, 3, 4, 4, 4, 5, 6, 6, 6, 7, 8, 8, 8, 9, 9]
    assert core.cover_crop_box(1920, 1080, 1344, 768) == (1365, 768, 10, 0)
    assert core.cover_crop_box(1344, 768, 1344, 768) == (1344, 768, 0, 0)
    assert core.cover_crop_box(720, 1280, 768, 1344) == (768, 1365, 0, 10)
    assert core.orient((1344, 768), 720, 1280) == (768, 1344)
    assert core.output_size("2k", 1344, 768) == (2560, 1440, 2560, 1440)
    assert core.output_size("1080p", 1344, 768) == (1920, 1088, 1920, 1080)
    assert core.output_size("4k", 768, 1344) == (2176, 3840, 2160, 3840)
    # latent grids: 2K 90 x 160, 1080P 68 x 120, 4K 136 x 240 (even, H3 patch 2)
    for name, (w, h, _) in core.RESOLUTIONS.items():
        assert w % 32 == 0 and h % 32 == 0, name


def test_keys(header_path=None):
    assert core.comfy_target("transformer.transformer_blocks.3.attn.to_q") == ("diffusion_model.blocks.3.attn.qkv_proj.weight", 0)
    assert core.comfy_target("transformer.transformer_blocks.3.attn.to_v") == ("diffusion_model.blocks.3.attn.qkv_proj.weight", 2)
    assert core.comfy_target("transformer.transformer_blocks.49.attn.to_out.0") == ("diffusion_model.blocks.49.attn.out_proj.weight", None)
    assert core.comfy_target("transformer.transformer_blocks.0.ff.net.0.proj") == ("diffusion_model.blocks.0.mlp.fc1.weight", "swiglu")
    assert core.comfy_target("transformer.token_refiner.refiner_blocks.1.ff.net.2") == ("diffusion_model.token_refiner.blocks.1.mlp.fc2.weight", None)
    for bad in ("transformer.transformer_blocks.0.adaln_proj.linear", "transformer.time_embedder.proj_in", "x"):
        try:
            core.comfy_target(bad)
            raise AssertionError(bad)
        except KeyError:
            pass
    b = torch.arange(8.0).reshape(4, 2)
    assert torch.equal(core.swap_swiglu_halves(b), torch.tensor([[4., 5.], [6., 7.], [0., 1.], [2., 3.]]))
    if header_path:
        header = json.load(open(header_path, encoding="utf-8"))
        header.pop("__metadata__", None)
        modules = core.lora_modules(set(header))
        targets = {}
        for module in modules:
            key, slot = core.comfy_target(module)
            targets.setdefault(key, set()).add(slot)
            a, b = header[module + ".lora_A.weight"]["shape"], header[module + ".lora_B.weight"]["shape"]
            assert a[0] == b[1] == 576, (module, a, b)
        assert len(modules) == 312 and len(targets) == 52 * 4, (len(modules), len(targets))
        assert all(v == {0, 1, 2} for k, v in targets.items() if k.endswith("qkv_proj.weight"))
        print(f"  header: {len(modules)} modules -> {len(targets)} ComfyUI weights")


def test_curve(checkpoint=None):
    torch.manual_seed(0)
    if checkpoint:
        from safetensors import safe_open
        with safe_open(checkpoint, framework="pt", device="cpu") as f:
            table = f.get_tensor(next(k for k in f.keys() if k.endswith("adaln_t_table"))).float()
    else:   # a smooth random 8-D curve
        x = torch.linspace(0, 1, 1025)[:, None]
        table = torch.cat([torch.sin(x * (i + 1) * 2.1 + i) for i in range(8)], dim=1)
    t = torch.cat([torch.rand(64), torch.tensor([0.0, 1.0, 1 - 0.909375, 1 - 0.857143, 1 / 3, 0.999])])
    coords = core.table_coords(table, t)
    back, err = core.invert_table(table, coords)
    worst = float((back - t).abs().max())
    assert err < 1e-4 and worst < 1e-4, (err, worst)
    a, b = torch.randn(64, 8), torch.randn(96, 64)
    ref = torch.nn.functional.linear(torch.nn.functional.linear(coords, a), b)
    assert torch.allclose(core.curve_delta(b @ a, table, t), ref, atol=1e-4, rtol=1e-4)
    print(f"  curve inversion: max |dt| {worst:.2e} ({'checkpoint table' if checkpoint else 'synthetic table'})")


if __name__ == "__main__":
    test_sigmas()
    test_frames()
    test_keys(sys.argv[1] if len(sys.argv) > 1 else None)
    test_curve(sys.argv[2] if len(sys.argv) > 2 else None)
    print("regen_core: all checks passed")
