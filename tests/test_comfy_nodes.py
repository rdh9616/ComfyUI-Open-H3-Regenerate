"""Node checks inside ComfyUI's Python, on tiny native MiniMax H3 models (no weights downloaded, CPU only).

Run:  <ComfyUI python> tests/test_comfy_nodes.py <ComfyUI dir> <scratch dir>

- LoRA: a synthetic Diffusers PEFT LoRA in the merged file's layout is converted and patched; every patched ComfyUI
  weight must equal W + B @ A with the q/k/v slices and the SwiGLU half swap (verified against the real Diffusers and
  Comfy-Org checkpoints separately).
- Sharp curve: AdaLN output must equal base + B @ A @ lerp(reference table, t) for a full model (t recorded from the
  time embedder) and for curve-form models whose own table is / is not the reference table; unpatching restores all.
- Sigmas, Prepare Draft, Finish Frames and Refine Latent on dummy tensors.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import torch

COMFY, SCRATCH = sys.argv[1], sys.argv[2]
sys.path.insert(0, COMFY)
sys.argv = sys.argv[:1] + ["--cpu"]
import comfy.options  # noqa: E402
comfy.options.enable_args_parsing()
import folder_paths  # noqa: E402
import comfy.model_patcher  # noqa: E402
import comfy.nested_tensor  # noqa: E402
import comfy.ops  # noqa: E402
from comfy.ldm.minimax.model import MiniMaxH3Model  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("open_h3_regen", os.path.join(HERE, "..", "__init__.py"),
                                              submodule_search_locations=[os.path.join(HERE, "..")])
pack = importlib.util.module_from_spec(spec)
sys.modules["open_h3_regen"] = pack
spec.loader.exec_module(pack)
core = pack.core

torch.manual_seed(0)
H, HEADS, HD, FFN, TDIM, LAYERS = 64, 2, 32, 96, 16, 3
INNER = HEADS * HD
RANK = 4
os.makedirs(os.path.join(SCRATCH, "loras"), exist_ok=True)
os.makedirs(os.path.join(SCRATCH, "diffusion_models"), exist_ok=True)
folder_paths.add_model_folder_path("loras", os.path.join(SCRATCH, "loras"), is_default=True)
folder_paths.add_model_folder_path("diffusion_models", os.path.join(SCRATCH, "diffusion_models"), is_default=True)


class Wrapper(torch.nn.Module):
    def __init__(self, dm):
        super().__init__()
        self.diffusion_model = dm


def tiny(curve_grid=None):
    dm = MiniMaxH3Model(hidden_size=H, num_layers=LAYERS, token_refiner_num_layers=2, num_attention_heads=HEADS,
                        attention_head_dim=HD, ffn_hidden_size=FFN, text_dim=32, timestep_input_dim=16,
                        time_embed_hidden_size=32, time_embed_dim=8 if curve_grid else TDIM,
                        adaln_curve_grid=curve_grid, dtype=torch.float32, device="cpu",
                        operations=comfy.ops.disable_weight_init)
    with torch.no_grad():
        for p in dm.parameters():
            p.copy_(torch.randn_like(p) * 0.05)
        if curve_grid:
            x = torch.linspace(0, 1, curve_grid)[:, None]
            dm.adaln_t_table.copy_(torch.cat([torch.cos(x * (i + 2) * 1.7 + i) for i in range(8)], dim=1))
    model = Wrapper(dm)
    return comfy.model_patcher.ModelPatcher(model, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))


def write_lora():
    sd = {}
    for stack, count in (("transformer_blocks", LAYERS), ("token_refiner.refiner_blocks", 2)):
        for i in range(count):
            for leaf, (out, inp) in {"attn.to_q": (INNER, H), "attn.to_k": (INNER, H), "attn.to_v": (INNER, H),
                                     "attn.to_out.0": (H, INNER), "ff.net.0.proj": (2 * FFN, H),
                                     "ff.net.2": (H, FFN)}.items():
                name = f"transformer.{stack}.{i}.{leaf}"
                sd[name + ".lora_A.weight"] = torch.randn(RANK, inp) * 0.1
                sd[name + ".lora_B.weight"] = torch.randn(out, RANK) * 0.1
    save_file(sd, os.path.join(SCRATCH, "loras", "tiny_regen_lora.safetensors"))
    return sd


def write_curves(out_features):
    sd = {}
    for i in range(LAYERS):
        sd[f"transformer.transformer_blocks.{i}.adaln_proj.linear.curve_A.weight"] = torch.randn(RANK, 8) * 0.2
        sd[f"transformer.transformer_blocks.{i}.adaln_proj.linear.curve_B.weight"] = torch.randn(out_features, RANK) * 0.2
    save_file(sd, os.path.join(SCRATCH, "loras", "tiny_curves.safetensors"))
    return sd


def write_table(name):
    x = torch.linspace(0, 1, 1025)[:, None]
    table = torch.cat([torch.sin(x * (i + 1) * 2.3 + 0.5 * i) for i in range(8)], dim=1)
    save_file({"adaln_t_table": table}, os.path.join(SCRATCH, "diffusion_models", name))
    return table


def check_lora(patcher, lora, strength):
    sd0 = {k: v.clone() for k, v in patcher.model.state_dict().items()}
    patcher.patch_model()
    sd1 = patcher.model.state_dict()
    checked = 0
    for module in sorted({k.rsplit(".lora_", 1)[0] for k in lora}):
        key, slot = core.comfy_target(module)
        sdkey = key
        a, b = lora[module + ".lora_A.weight"], lora[module + ".lora_B.weight"]
        delta = strength * (b @ a)
        got, expect = sd1[sdkey], sd0[sdkey].clone()
        if isinstance(slot, int):
            rows = slice(slot * INNER, (slot + 1) * INNER)
            got, expect = got[rows], expect[rows] + delta
        elif slot == "swiglu":
            expect += strength * (core.swap_swiglu_halves(b) @ a)
        else:
            expect += delta
        assert torch.allclose(got, expect, atol=1e-5), module
        checked += 1
    untouched = [k for k in sd0 if k.endswith(".weight") and ("adaln" in k or "norm" in k)]
    assert all(torch.equal(sd0[k], sd1[k]) for k in untouched)
    return sd0, checked


def check_curves(patcher, curves, table, strength, own_table=None):
    dm = patcher.model.diffusion_model
    t = torch.tensor([1 - 0.909375, 1 - 0.857143, 0.999, 1.0, 0.25])
    with torch.no_grad():
        if own_table is None:
            t_emb = dm.time_embedder(t)
        else:
            pos = t * 1024
            lo = pos.floor().long().clamp(max=1023)
            t_emb = torch.lerp(own_table[lo], own_table[lo + 1], (pos - lo)[:, None])
        for i, block in enumerate(dm.blocks):
            got = torch.cat(block.adaln_proj(t_emb), dim=-1)
            linear = block.adaln_proj.linear
            x = torch.nn.functional.linear(torch.nn.functional.silu(t_emb) if block.adaln_proj.apply_silu else t_emb,
                                           linear.weight, linear.bias)
            a = curves[f"transformer.transformer_blocks.{i}.adaln_proj.linear.curve_A.weight"]
            b = curves[f"transformer.transformer_blocks.{i}.adaln_proj.linear.curve_B.weight"]
            x = x + strength * torch.nn.functional.linear(torch.nn.functional.linear(core.table_coords(table, t), a), b)
            mod = block.adaln_proj
            expect = torch.cat(x.view(x.shape[0] * mod.modalities, mod.expand * mod.hidden).chunk(mod.expand, dim=-1), dim=-1)
            assert torch.allclose(got, expect, atol=1e-5), (i, (got - expect).abs().max())


def run_case(label, curve_grid, table_choice, expect_mode):
    patcher = tiny(curve_grid)
    dm = patcher.model.diffusion_model
    out_features = dm.blocks[0].adaln_proj.linear.weight.shape[0]
    lora, curves = write_lora(), write_curves(out_features)
    ref = write_table("ref_table.safetensors")
    if table_choice == "own":
        with torch.no_grad():
            dm.adaln_t_table.copy_(ref)
        choice = pack.OWN_TABLE
    else:
        choice = "ref_table.safetensors"
    own = dm.adaln_t_table.detach().clone() if curve_grid else None
    node = pack.OpenH3RegenLora()
    m, report = node.apply(patcher, "tiny_regen_lora.safetensors", "tiny_curves.safetensors", choice, 0.8, 1.3, "fp32")
    assert f"{expect_mode}" in report or expect_mode == "", report
    assert torch.equal(patcher.model.state_dict()[next(iter(patcher.model.state_dict()))],
                       m.model.state_dict()[next(iter(m.model.state_dict()))])
    sd0, checked = check_lora(m, lora, 0.8)
    check_curves(m, curves, ref if table_choice != "own" else own, 1.3, own_table=own)
    m.unpatch_model()
    sd2 = m.model.state_dict()
    assert all(torch.equal(sd0[k], sd2[k]) for k in sd0), "unpatch did not restore weights"
    assert type(m.model.diffusion_model.blocks[0].adaln_proj.linear).__name__ != "OpenH3SharpCurveLinear"
    print(f"  {label}: {checked} LoRA modules, {LAYERS} curve branches OK — {report.splitlines()[1]}")


def check_frames():
    s = pack.OpenH3RegenSigmas().sigmas(3, 0.0)[0]
    assert torch.allclose(s, torch.tensor([0.909375, 6 / 7, 2 / 3, 0.0]))
    frames = torch.rand(130, 90, 160, 3)
    out, count, w, h = pack.OpenH3RegenPrepareDraft().prepare(frames, "576p (1024x576)", 121)
    assert (count, w, h) == (121, 1024, 576) and out.shape == (124, 576, 1024, 3)
    assert torch.equal(out[121], out[120]) and torch.equal(out[123], out[120])
    out, count, w, h = pack.OpenH3RegenPrepareDraft().prepare(torch.rand(30, 1280, 720, 3), "768p (1344x768)", 0)
    assert (count, w, h) == (30, 768, 1344) and out.shape == (39, 1344, 768, 3)
    out, *_ = pack.OpenH3RegenPrepareDraft().prepare(torch.rand(5, 100, 70, 3), "keep (snap to 32)", 0)
    assert out.shape == (5, 96, 64, 3)
    fin = pack.OpenH3RegenFinish().finish(torch.rand(124, 1088, 1920, 3)[:, :, :8], 121, "auto (1088 -> 1080, 2176 -> 2160)")[0]
    assert fin.shape == (121, 1088, 8, 3)             # 8 px wide: portrait rule must not crop width 8
    fin, sound = pack.OpenH3RegenFinish().finish(torch.rand(6, 1088, 1920, 3), 5, "auto (1088 -> 1080, 2176 -> 2160)",
                                                 {"waveform": torch.randn(1, 2, 48000), "sample_rate": 48000})
    assert fin.shape == (5, 1080, 1920, 3) and sound["waveform"].shape == (1, 2, 10000)
    out, count, *_ = pack.OpenH3RegenPrepareDraft().prepare(torch.rand(150, 64, 64, 3), "keep (snap to 32)", 0, 30.0)
    assert count == 120 and out.shape[0] == 124

    class FakeAudioVAE:
        audio_sample_rate = 32000

        def encode(self, waveform):          # [B, L, C] -> [B, 32, 2, ~L/800]
            assert waveform.shape[-1] == 2
            return torch.full((1, 32, 2, round(waveform.shape[1] / 800)), 0.5)

    video = torch.randn(1, 24, 37, 90, 160)
    latent, frames = pack.OpenH3RegenRefineLatent().build({"samples": video}, pack.AUDIO_MODES[0], FakeAudioVAE())
    v, a = latent["samples"].tensors
    mv, ma = latent["noise_mask"].tensors
    assert frames == 124 and a.shape == (1, 32, 2, 207) and torch.equal(v, video)
    assert bool((mv == 1).all()) and bool((ma == 0).all())
    audio = {"waveform": torch.randn(1, 1, 44100 * 3), "sample_rate": 44100}
    latent, _ = pack.OpenH3RegenRefineLatent().build({"samples": video}, pack.AUDIO_MODES[1], FakeAudioVAE(), audio)
    assert latent["samples"].tensors[1].shape == (1, 32, 2, 207)
    av = {"samples": comfy.nested_tensor.NestedTensor((video, torch.randn(1, 32, 2, 300)))}
    latent, _ = pack.OpenH3RegenRefineLatent().build(av, pack.AUDIO_MODES[2])
    assert latent["samples"].tensors[1].shape == (1, 32, 2, 207)
    print("  sigmas / prepare / finish / refine latent OK")


def check_fast_vae_lean():
    from open_h3_regen.fast_vae_decoder import KEEP, NonCausalViTDecoder
    import open_h3_regen.fast_vae_lean as lean
    rows = lean.ROWS
    lean.ROWS = 50                            # force the chunked paths on tiny inputs
    for T, H, W in ((7, 5, 6), (12, 18, 20), (5, 16, 16), (1, 3, 4), (13, 33, 17)):
        torch.manual_seed(T)
        ref_model = NonCausalViTDecoder([0.1] * 24, [1.3] * 24, KEEP[6], time_norm=5, heads=2, dim_head=8).eval()
        with torch.no_grad():
            for p in ref_model.parameters():
                p.copy_(torch.randn_like(p) * 0.05)
        lean_model = lean.LeanNonCausalViTDecoder([0.1] * 24, [1.3] * 24, KEEP[6], time_norm=5, heads=2, dim_head=8).eval()
        lean_model.load_state_dict(ref_model.state_dict())
        z = torch.randn(1, T, 24, H, W)
        with torch.no_grad():
            a, b = ref_model(z), lean_model(z)
        assert a.shape == b.shape and torch.allclose(a, b, atol=1e-5), (T, H, W, a.shape, b.shape, (a - b).abs().max())
        streamed = torch.full_like(a, float("nan"))
        with torch.no_grad():
            assert lean_model(z, sink=lambda f, p: streamed[:, :, f:f + p.shape[2]].copy_(p)) is None
        assert torch.equal(streamed, b), (T, H, W)
        assert a.shape[2] == 4 * T - 3 * ((T + 4) // 5)           # the node's frame count formula
    lean.ROWS = rows
    print("  fast VAE: lean forward == reference forward (5 shapes, shifted layers, partial blocks)")


def check_video_prompt_helpers():
    import io
    import av
    from open_h3_regen import llm_prompt as lp
    fmt = lp.FORMATS[0]
    good = "integrated_multimodal_description:\n[Shot 1] A fox.\n\noverall_soundscape:\nWind.\n\nnon_diegetic_music:\nN/A"
    for raw in (good, "<think>plan the shots</think>\n" + good, "Sure! Here is the prompt:\n```text\n" + good + "\n```",
                "reasoning without an opening tag</think>" + good, "**integrated_multimodal_description:**\n[Shot 1] A fox."
                "\n\n**overall_soundscape:**\nWind.\n\n**non_diegetic_music:**\nN/A"):
        assert lp.clean_response(raw, fmt) == good, lp.clean_response(raw, fmt)
    assert lp.missing_sections("integrated_multimodal_description:\n[Shot 1] x") == ["overall_soundscape", "non_diegetic_music"]
    assert "Additional instructions" in lp.instructions(fmt, 5.0, "call her Mia") and "5.0 seconds" in lp.instructions(fmt, 5.0)
    os.environ.pop("OPEN_H3_TEST_KEY", None)
    try:
        lp.api_key("OPEN_H3_TEST_KEY")
        raise AssertionError("missing key must raise")
    except RuntimeError:
        pass
    assert lp.api_key("") == ""
    data, sent, size = lp.encode_video(torch.rand(121, 270, 480, 3), 24.0)
    assert sent == 20 and size == (448, 252), (sent, size)
    with av.open(io.BytesIO(data)) as c:
        assert sum(1 for _ in c.decode(video=0)) == 20
    stills = lp.encode_frames(torch.rand(121, 270, 480, 3), 24.0)
    assert len(stills) == 8 and stills[-1][1] == 5.0
    node = pack.OpenH3VideoToPrompt()
    assert node.describe(torch.rand(5, 8, 8, 3), 24.0, "http://unused", "", "OPEN_H3_TEST_KEY", pack.SEND_AS[0], fmt, "",
                         100, 0.3, 0, False, "  my prompt  ")[0] == "my prompt"
    print("  video -> H3 prompt helpers OK (cleanup, sections, key handling, H.264 + JPEG encoding, manual prompt)")


def check_latent_roundtrip():
    out_dir = os.path.join(SCRATCH, "output")
    folder_paths.set_output_directory(out_dir)
    video, sound = torch.randn(1, 24, 12, 6, 10), torch.randn(1, 32, 2, 65)
    audio = {"waveform": torch.randn(1, 2, 44100), "sample_rate": 44100}
    av = {"samples": comfy.nested_tensor.NestedTensor((video, sound))}
    saver, loader = pack.OpenH3SaveRefinedLatent(), pack.OpenH3LoadRefinedLatent()
    for name in os.listdir(os.path.join(out_dir, pack.LATENT_DIR)) if os.path.isdir(os.path.join(out_dir, pack.LATENT_DIR)) else []:
        os.remove(os.path.join(out_dir, pack.LATENT_DIR, name))
    res = saver.save(av, 39, "rt", audio)
    assert res["result"][0] is av and "rt_00001.safetensors" in res["ui"]["text"][0]
    saver.save({"samples": video}, 0, "rt")                       # video-only, no audio -> rt_00002
    files = loader.INPUT_TYPES()["required"]["latent_file"][0]
    assert files[0] == pack.NEWEST and set(files[1:]) == {"rt_00001.safetensors", "rt_00002.safetensors"}, files
    os.utime(os.path.join(out_dir, pack.LATENT_DIR, "rt_00002.safetensors"), (2e9, 2e9))
    assert pack._resolve_latent(pack.NEWEST) == "rt_00002.safetensors"
    assert torch.equal(loader.load(pack.NEWEST)[0]["samples"], video)
    lat, frames, a = loader.load("rt_00001.safetensors")
    v2, s2 = lat["samples"].tensors
    assert torch.equal(v2, video) and torch.equal(s2, sound) and frames == 39
    assert torch.equal(a["waveform"], audio["waveform"]) and a["sample_rate"] == 44100
    lat, frames, a = loader.load("rt_00002.safetensors")
    assert torch.equal(lat["samples"], video) and frames == 0 and a["waveform"].abs().sum() == 0
    print("  refined latent save / load round trip OK")


if __name__ == "__main__":
    check_latent_roundtrip()
    check_video_prompt_helpers()
    check_fast_vae_lean()
    run_case("full model (recorded t)", None, "ref", "recorded")
    run_case("curve-form, own table = reference", 1025, "own", "the model's AdaLN curve is the reference curve")
    run_case("curve-form, different table (inverted t)", 1025, "ref", "read back")
    check_frames()
    print("comfy node checks passed")
