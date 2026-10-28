"""ComfyUI-free pieces of Open-H3-Regenerate-2K (NVlabs/Sana, branch sol-engine, models/Open-H3-Regenerate-2K).

- refine sigma grids (open_h3/sigmas.py): Euler, CFG 1, model time 1 - sigma
- diffusers -> ComfyUI key mapping for the merged refiner + Turbo + LMS Sharp LoRA
- the LMS Sharp curve branches: delta(t) = B @ A @ lerp(adaln_t_table, t), added to every block's AdaLN projection
- draft-frame and output-size helpers (H3 frames come in 17k + 5 steps; latent frames = 5k + 2)
"""
from __future__ import annotations

import re

import torch

REFINE_SHIFT = 6.0
SCHEDULES = {3: [0.909375, 0.8571428571428571, 0.6666666666666666, 0.0]}

# draft sizes (W, H) the reference encodes a video at
INPUT_SIZES = {"768p": (1344, 768), "576p": (1024, 576)}
# output size -> (computed width, computed height, kept height): 1080P / 4K run on whole latent patches (1088 / 2176
# rows) and the centre 1080 / 2160 rows are kept
RESOLUTIONS = {"768p": (1344, 768, 768), "1080p": (1920, 1088, 1080), "2k": (2560, 1440, 1440),
               "4k": (3840, 2176, 2160)}
UPSCALER_MAX = RESOLUTIONS["2k"]       # one upscaler pass stays within its trained range; 4K goes through 2K first

# sha256 of the 1025 x 8 adaln_t_table the reference reads (Comfy-Org minimax_h3_ref2va_pruned_int8_convrot)
REFERENCE_TABLE_SHA256 = "c02a6c11888297688c1e6278185ea1f947023acfc69f9003bbcdcec9a229a8e7"


def shifted_grid(start, steps, shift=REFINE_SHIFT):
    """Euler grid from `start` to 0: even spacing in unshifted sigma, then s*x / (1 + (s - 1)*x)."""
    raw = start / (shift - (shift - 1) * start)
    return [start] + [shift * x / (1 + (shift - 1) * x) for x in (raw * (1 - i / steps) for i in range(1, steps))] + [0.0]


SCHEDULES[2] = shifted_grid(0.7, 2)


def refine_sigmas(steps, start_sigma=0.0):
    """The release grid for `steps` (3 or 2), or the shift-6 grid from `start_sigma` when it is > 0."""
    if start_sigma and start_sigma > 0:
        return shifted_grid(float(start_sigma), int(steps))
    if steps not in SCHEDULES:
        raise ValueError(f"steps must be one of {sorted(SCHEDULES)} without a start_sigma")
    return list(SCHEDULES[steps])


# ------------------------------------------------------------------ frames

def align_frame_count(n):
    """H3 clip lengths are 17k + 5 frames at 24 fps (124 = 5 s, 362 = 15 s)."""
    n = max(5, int(n))
    return n + (5 - n) % 17


def video_latent_t(frame_count):
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2


def frames_for_latent_t(latent_t):
    return 5 if latent_t <= 2 else (latent_t - 2) // 5 * 17 + 5


def resample_indices(count, fps, target=24.0):
    """Source frame per output frame for a `count`-frame clip at `fps` resampled to `target` (ffmpeg fps filter:
    output frame i shows the source frame nearest to time i / target)."""
    out = max(1, int(round(count / fps * target)))
    return [min(count - 1, int(round(i * fps / target))) for i in range(out)]


def cover_crop_box(src_w, src_h, dst_w, dst_h):
    """(scaled_w, scaled_h, x0, y0): scale to cover dst (ffmpeg force_original_aspect_ratio=increase), centre crop."""
    scale = max(dst_w / src_w, dst_h / src_h)
    sw, sh = max(dst_w, round(src_w * scale)), max(dst_h, round(src_h * scale))
    return sw, sh, (sw - dst_w) // 2, (sh - dst_h) // 2


def orient(size, width, height):
    """A landscape (W, H) preset turned to the draft's orientation."""
    w, h = size
    return (h, w) if height > width else (w, h)


def output_size(resolution, width, height):
    """(computed W, computed H, kept W, kept H) for a draft of width x height."""
    w, h, keep = RESOLUTIONS[resolution]
    if height > width:
        return h, w, keep, w
    return w, h, w, keep


# ------------------------------------------------------------------ LoRA keys

_BLOCK = re.compile(r"^transformer\.(transformer_blocks|token_refiner\.refiner_blocks)\.(\d+)\.(.+)$")
_TARGET = {
    "attn.to_q": ("attn.qkv_proj", 0),
    "attn.to_k": ("attn.qkv_proj", 1),
    "attn.to_v": ("attn.qkv_proj", 2),
    "attn.to_out.0": ("attn.out_proj", None),
    "ff.net.0.proj": ("mlp.fc1", "swiglu"),
    "ff.net.2": ("mlp.fc2", None),
}


def comfy_target(module):
    """Diffusers module name ('transformer.transformer_blocks.3.attn.to_q') ->
    (ComfyUI weight key, qkv slot or 'swiglu' or None)."""
    match = _BLOCK.match(module)
    if match is None or match.group(3) not in _TARGET:
        raise KeyError(f"unexpected LoRA module {module}")
    stack, index, leaf = match.groups()
    root = "blocks" if stack == "transformer_blocks" else "token_refiner.blocks"
    name, slot = _TARGET[leaf]
    return f"diffusion_model.{root}.{index}.{name}.weight", slot


def swap_swiglu_halves(b):
    """LoRA B of ff.net.0.proj: Diffusers rows [value; gate] -> ComfyUI fc1 rows [gate; value]."""
    half = b.shape[0] // 2
    return torch.cat([b[half:], b[:half]], dim=0)


def lora_modules(keys):
    """Module names of a PEFT LoRA ('<module>.lora_A.weight' / '.lora_B.weight')."""
    names = sorted(k[:-len(".lora_A.weight")] for k in keys if k.endswith(".lora_A.weight"))
    missing = [n for n in names if f"{n}.lora_B.weight" not in keys]
    if missing or len(names) * 2 != len(keys):
        raise ValueError("expected only lora_A / lora_B pairs" + (f", missing B for {missing[:3]}" if missing else ""))
    return names


# ------------------------------------------------------------------ Sharp curve branches

_CURVE = re.compile(r"^transformer\.transformer_blocks\.(\d+)\.adaln_proj\.linear\.curve_A\.weight$")


def curve_blocks(keys):
    """{block index: (curve_A key, curve_B key)} of a sharp_curve_branches file."""
    out = {}
    for key in keys:
        match = _CURVE.match(key)
        if match:
            out[int(match.group(1))] = (key, key.replace(".curve_A.", ".curve_B."))
    return out


def table_coords(table, t):
    """Interpolate the 8-D curve at model time t in [0, 1] (rows of `table` span t = 0..1)."""
    t = t.to(torch.float32).clamp(0.0, 1.0)
    pos = t * (table.shape[0] - 1)
    lo = pos.floor().long().clamp(max=table.shape[0] - 2)
    return torch.lerp(table[lo], table[lo + 1], (pos - lo).unsqueeze(-1))


def invert_table(table, coords):
    """Model time t of each row of `coords`, points on the piecewise-linear curve `table` (as ComfyUI's curve-form
    H3 computes its AdaLN input).  Returns (t, max distance from the curve)."""
    table = table.to(coords.device, torch.float32)
    coords = coords.to(torch.float32)
    p0, d = table[:-1], table[1:] - table[:-1]                       # segments [S, C]
    rel = coords[:, None, :] - p0[None]                               # [M, S, C]
    s = ((rel * d[None]).sum(-1) / (d * d).sum(-1).clamp(min=1e-20)[None]).clamp(0.0, 1.0)
    dist = (rel - s[..., None] * d[None]).norm(dim=-1)               # [M, S]
    best = dist.argmin(dim=1)
    rows = torch.arange(coords.shape[0], device=coords.device)
    t = (best.to(torch.float32) + s[rows, best]) / (table.shape[0] - 1)
    return t, float(dist[rows, best].max()) if coords.shape[0] else 0.0


def curve_delta(weight, table, t):
    """B @ A @ curve(t) for model times t [M] -> [M, out]; weight = B @ A [out, 8]."""
    return table_coords(table.to(t.device), t) @ weight.to(t.device, torch.float32).t()
