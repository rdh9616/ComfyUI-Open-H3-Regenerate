"""Additive IC conditioning nodes; native ComfyUI still owns the Ref2VA packed transformer."""
from __future__ import annotations

import copy

import torch
import comfy.nested_tensor
import node_helpers

from . import ic_core as ic

CATEGORY = "MiniMax H3/Open-H3 Regenerate/IC"


class _SampledPosterior:
    """Per-call VAE facade. No monkey-patching of the shared model or its encoder methods."""

    def __init__(self, model, seed):
        self.model, self.seed = model, seed

    def __getattr__(self, name):
        return getattr(self.model, name)

    def encode(self, pixels, device=None):
        moments = self.model.encode_temporal(pixels, device or pixels.device)
        return ic.sample_posterior(moments, self.model.latents_mean, self.model.latents_std, self.seed)

    def encode_tiled(self, pixels, **kwargs):
        # Comfy's OOM fallback must not silently return to the native posterior mode.
        # MiniMax already performs its semantic spatial/temporal tiling internally.
        return self.encode(pixels)


def encode_draft(vae, images, seed):
    from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE, ResnetShortcut3D

    model = vae.first_stage_model
    if not isinstance(model, MiniMaxH3VideoVAE) or not hasattr(model, "encoder"):
        raise ValueError("IC requires the stock MiniMax H3 video VAE encoder, not an audio or preview VAE")
    if any(isinstance(module, ResnetShortcut3D) for module in model.encoder.modules()):
        raise ValueError("IC requires the stock H3 encoder; LightVAE pruned encoders are not validated")
    if vae.vae_dtype != torch.float32:
        raise ValueError("IC matches the upstream FP32 VAE; load the video VAE in FP32 (--fp32-vae)")
    if not model.tiling or (model.tile_size, model.tile_overlap_min, model.clip_length, model.token_drop) != (256, 64, 17, 3):
        raise ValueError("IC requires stock H3 VAE 256/64 spatial tiles and 17-frame temporal chunks")
    # Keep ComfyUI's memory management, CPU-to-device chunking, and output-device handling.
    sampled = copy.copy(vae)
    sampled.first_stage_model = _SampledPosterior(model, seed)
    result = sampled.encode(images)
    ic.validate_video_latent(result, "encoded draft")
    return result


class OpenH3ICPrepareDraft:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "source_fps": ("FLOAT", {"default": 24.0, "min": 0.001, "max": 240.0}),
        }}

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "frame_count")
    FUNCTION = "prepare"
    CATEGORY = CATEGORY
    DESCRIPTION = ("IC only: select 113-124 real frames at 24 fps, cover-crop to 1344x768, "
                   "then repeat the last frame to 124. Longer/shorter clips are rejected. "
                   "Source FPS must describe the supplied images, after any loader resampling.")

    def prepare(self, images, source_fps):
        return ic.prepare_draft(images, source_fps)


class OpenH3ICEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "clip": ("CLIP",),
            "video_vae": ("VAE",),
            "images": ("IMAGE",),
            "caption": ("STRING", {"multiline": True, "default": ""}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
        }}

    RETURN_TYPES = ("CONDITIONING", "LATENT", "STRING")
    RETURN_NAMES = ("conditioning", "draft_latent", "instruction")
    FUNCTION = "encode"
    CATEGORY = CATEGORY
    DESCRIPTION = ("IC draft conditioning: exact enhancement instruction plus the original caption; "
                   "Qwen sees the draft at 2 fps, and Ref2VA receives its seeded VAE posterior sample. "
                   "Connect draft_latent to the latent upscaler. Audio is kept outside this conditioning.")

    def encode(self, clip, video_vae, images, caption, seed):
        if tuple(images.shape) != (ic.FRAMES, *ic.LQ_SIZE, 3):
            raise ValueError("Connect IC Prepare Draft: expected 124 RGB frames at 1344x768")
        # Encode once: the same sampled draft conditions every step and initializes the upscaled target.
        draft = encode_draft(video_vae, images, seed)
        prompt = ic.instruction(caption)
        qwen_frames = (images[::12].float().clamp(0, 1) * 255).round() / 255
        refs = [{"type": "video", "data": qwen_frames,
                 "timestamps": [i / 2.0 for i in range(len(qwen_frames))]}]
        conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(prompt, minimax_ref_items=refs))
        ic.validate_conditioning(conditioning)
        block = {"kind": "video", "latent_t": ic.LATENT_T, "latent_h": 48, "latent_w": 84,
                 "ref_audio_t": 0, "latent": draft, "audio_latent": None}
        conditioning = node_helpers.conditioning_set_values(conditioning, {
            "minimax_refs": [block], "minimax_visual_cond_noise_aug": ic.REFERENCE_T,
        })
        return conditioning, {"samples": draft}, prompt


class OpenH3ICTargetLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "reference_latent": ("LATENT",),
            "target_latent": ("LATENT",),
            "audio_vae": ("VAE",),
            "schedule": (["up3", "up4", "hf8"], {"default": "up3"}),
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "build"
    CATEGORY = CATEGORY
    DESCRIPTION = ("IC 124-frame 2K target: up3/up4 preserve the upscaled latent for sampler re-noising; "
                   "hf8 starts from pure noise. Audio is encoded silence with a zero denoise mask. "
                   "Use the matching IC schedule/sampler; the old Regenerate sigma grid is incompatible.")

    def build(self, reference_latent, target_latent, audio_vae, schedule):
        reference, target = reference_latent["samples"], target_latent["samples"]
        ic.validate_pair(reference, target)
        if schedule not in ("up3", "up4", "hf8"):
            raise ValueError("IC schedule must be up3, up4 or hf8")
        if int(getattr(audio_vae, "audio_sample_rate", 0)) != 32000:
            raise ValueError("IC needs the MiniMax H3 32000 Hz audio VAE to encode silence")
        audio_t = round(ic.FRAMES / 24 * 40)
        # VAE.encode crops input to its stride *before* the audio model can right-pad.
        # 124/24*32000 rounds to 165333, then crops to 206 hops; IC needs all 207.
        length = audio_t * 800
        sound = audio_vae.encode(torch.zeros(1, length, 2))
        if (sound.ndim != 4 or sound.shape[:3] != (1, 32, 2) or sound.shape[-1] < audio_t or
                not torch.isfinite(sound).all()):
            raise ValueError(f"IC audio VAE must yield finite [1, 32, 2, >=207] silence; got {tuple(sound.shape)}")
        sound = sound[..., :audio_t].to(device=target.device, dtype=torch.float32).contiguous()
        video = torch.zeros_like(target) if schedule == "hf8" else target
        latent = {"samples": comfy.nested_tensor.NestedTensor((video, sound)),
                  "noise_mask": comfy.nested_tensor.NestedTensor((torch.ones_like(video), torch.zeros_like(sound)))}
        return (latent,)


NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in (OpenH3ICPrepareDraft, OpenH3ICEncode, OpenH3ICTargetLatent)}
NODE_DISPLAY_NAME_MAPPINGS = {"OpenH3ICPrepareDraft": "Open-H3 IC · Prepare Draft (124 frames)",
                              "OpenH3ICEncode": "Open-H3 IC · Encode Draft Context",
                              "OpenH3ICTargetLatent": "Open-H3 IC · Target Latent (2K)"}
