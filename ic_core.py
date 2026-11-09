"""IC input contracts from NVlabs/Sana 1e3c9c2, separate from the original refiner."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

FRAMES = 124
MIN_REAL = 113
LATENT_T = 37
LQ_SIZE = (768, 1344)  # height, width
TARGET_SIZE = (1440, 2560)
REFERENCE_T = 0.999
TASK = ("Enhance <Video 1> into a high-quality version of the same video. "
        "Preserve the scene, objects, camera motion, timing, and composition of every frame. "
        "Restore fine details and remove degradation. Do not continue or change the scene.")


def instruction(caption):
    return TASK + "\n\nScene description: " + (caption or "")


def draft_indices(count, fps):
    """Floor sampling as in IC/media.py; reject clips outside this initial 124-frame scope."""
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("IC source_fps must be a finite positive frame rate")
    if count <= 0:
        raise ValueError("IC draft has no frames")
    # A 125th output frame would be real: require the user to select the intended clip.
    if int(FRAMES * fps / 24.0) < count:
        raise ValueError("IC supports at most 124 frames at 24 fps; trim the input clip explicitly")
    indices = [int(i * fps / 24.0) for i in range(FRAMES) if int(i * fps / 24.0) < count]
    if len(indices) < MIN_REAL:
        raise ValueError(f"IC needs 113-124 real frames at 24 fps; received {len(indices)}")
    return indices


def prepare_draft(images, fps, size=LQ_SIZE, chunk=8):
    """BHWC [0,1] -> 124 cover-resized frames; return the unpadded frame count."""
    if images.ndim != 4 or images.shape[-1] != 3 or min(images.shape[1:3]) <= 0:
        raise ValueError("IC draft must be RGB IMAGE [frames, height, width, 3]")
    indices = draft_indices(images.shape[0], float(fps))
    height, width = size
    h, w = images.shape[1:3]
    scale = max(height / h, width / w)
    sh, sw = max(height, round(h * scale)), max(width, round(w * scale))
    y, x = (sh - height) // 2, (sw - width) // 2
    # Preallocate: holding both a list of every full-size frame and its concatenation doubles RAM.
    result = torch.empty((FRAMES, height, width, 3), dtype=torch.float32, device="cpu")
    for start in range(0, len(indices), chunk):
        part = images[indices[start:start + chunk]].movedim(-1, 1).float()
        if not torch.isfinite(part).all():
            raise ValueError("IC draft contains non-finite pixels")
        if (h, w) != (height, width):
            part = F.interpolate(part, size=(sh, sw), mode="bilinear", align_corners=False, antialias=scale < 1)
        part = part[:, :, y:y + height, x:x + width].clamp(0, 1).movedim(1, -1)
        result[start:start + len(part)].copy_(part)
    result[len(indices):] = result[len(indices) - 1]
    return result, len(indices)


def sample_posterior(moments, latents_mean, latents_std, seed):
    """Diffusers DiagonalGaussianDistribution sample: FP32, logvar clamped [-30,20], CPU RNG.

    The upstream generator is a CPU generator even when the VAE runs on CUDA. Sampling after the
    spatial blends/temporal concatenation is essential; sampling each tile gives different noise.
    """
    if moments.ndim != 5 or moments.shape[:2] != (1, 48):
        raise ValueError(f"IC posterior must be [1, 48, T, H, W], got {tuple(moments.shape)}")
    mean, logvar = moments.float().chunk(2, dim=1)
    noise = torch.randn(mean.shape, generator=torch.Generator(device="cpu").manual_seed(seed), dtype=torch.float32)
    sample = mean + (0.5 * logvar.clamp(-30.0, 20.0)).exp() * noise.to(mean.device)
    offset = latents_mean.to(sample).view(1, 24, 1, 1, 1)
    scale = latents_std.to(sample).view(1, 24, 1, 1, 1)
    if not torch.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError("IC VAE latents_std must be finite and positive")
    result = ((sample - offset) / scale).to(device="cpu", dtype=torch.bfloat16)
    if not torch.isfinite(result).all():
        raise ValueError("IC VAE produced non-finite sampled latents")
    return result


def validate_video_latent(video, name):
    if not isinstance(video, torch.Tensor) or video.ndim != 5 or video.shape[:2] != (1, 24):
        raise ValueError(f"IC {name} must be a video LATENT [1, 24, 37, H, W]")
    if video.shape[2] != LATENT_T or min(video.shape[3:]) <= 0 or any(n % 2 for n in video.shape[3:]):
        raise ValueError(f"IC {name} needs T=37 and positive even spatial dimensions; got {tuple(video.shape)}")
    if not torch.isfinite(video).all():
        raise ValueError(f"IC {name} contains non-finite values")


def validate_pair(reference, target):
    validate_video_latent(reference, "reference")
    validate_video_latent(target, "target")
    if tuple(reference.shape[-2:]) != (48, 84):
        raise ValueError("IC reference must be the 1344x768 draft latent (48x84)")
    if tuple(target.shape[-2:]) != (90, 160):
        raise ValueError("This IC path currently supports only a 2560x1440 target latent (90x160)")


def validate_conditioning(conditioning):
    if not conditioning:
        raise ValueError("IC requires native MiniMax H3 Qwen conditioning")
    for embeds, metadata in conditioning:
        if not isinstance(embeds, torch.Tensor) or embeds.ndim != 3 or embeds.shape[0] != 1 or embeds.shape[2] != 5120:
            raise ValueError("IC requires H3 Qwen embeddings [1, tokens, 5120]")
        tags = metadata.get("minimax_token_tags")
        if (tags is None or tags.numel() != embeds.shape[1] or
                not (tags == 0).any() or not (tags == 1).any() or not ((tags == 0) | (tags == 1)).all()):
            raise ValueError("IC needs MiniMax H3 Qwen vision/text tags; use the H3 text encoder with the draft video")
        if not torch.isfinite(embeds).all():
            raise ValueError("IC text encoder produced non-finite embeddings")
