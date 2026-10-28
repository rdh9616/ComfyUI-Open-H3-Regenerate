"""Open-H3-Regenerate-2K for ComfyUI: regenerate a low-resolution MiniMax H3 (or other model's) draft at 768P-4K.

NVlabs' open replacement for the closed H3-Regenerate-2K API (https://nvlabs.github.io/Sana/Open-H3-Regenerate-2K/):
draft -> H3 VAE encode -> H3-Latent-Upscaler (LBH 3D) onto the target grid -> partial re-noise to sigma 0.909 ->
3 Euler steps (CFG 1) with one merged LoRA (refiner + LightX2V FL2VA Turbo + LMS Sharp) -> H3 VAE decode.

Open-H3 Regenerate · LoRA            -> loads the merged Diffusers LoRA (12.4 GB FP32) and the 50 Sharp curve
                                        branches onto ComfyUI's native H3 model
Open-H3 Regenerate · Sigmas          -> 0.909375, 0.857, 0.667, 0 (3 steps) / 0.7, 0.494, 0 (2 steps) / shift-6 grid
Open-H3 Regenerate · Prepare Draft   -> cover-scale + centre-crop the draft to 768P / 576P, pad to H3's 17k+5 frames
Open-H3 Regenerate · Latent Upscale  -> LBH H3 latent upscaler to 768P / 1080P / 2K / 4K (4K = two passes via 2K)
Open-H3 Regenerate · Refine Latent   -> AV latent for the refine: video re-noised, audio held clean (silence by default)
Open-H3 Regenerate · Finish Frames   -> trim the padded frames and crop 1088 -> 1080 / 2176 -> 2160 rows

Needs Comfyui_Minimax_h3_latent_Upscaler (LBH-123-AI) for the latent upscale node.
"""
from __future__ import annotations

import copy
import hashlib
import logging
import os
import sys
import time

import torch
import torch.nn.functional as F

import comfy.lora
import comfy.model_management
import comfy.nested_tensor
import comfy.utils
import folder_paths
import nodes

from . import regen_core as core

CATEGORY = "MiniMax H3/Open-H3 Regenerate"
LOG = logging.getLogger("OpenH3Regenerate")
OWN_TABLE = "(model's own table)"
NO_CURVE = "none"
SIZES = ["768p (1344x768)", "576p (1024x576)", "keep (snap to 32)"]
RESOLUTION_NAMES = {"768p (1344x768)": "768p", "1080p (1920x1080)": "1080p", "2k (2560x1440)": "2k",
                    "4k (3840x2160)": "4k"}
AUDIO_MODES = ["silence (reference)", "draft audio (AUDIO input)", "keep latent audio"]
LBH_NODE = "MinimaxH3LatentUpscaler3D"
LBH_FOLDER = "latent_upscale_models"

_LORA_CACHE = {}


def _pick(options, *needles):
    for needle in needles:
        for name in options:
            if needle in name.lower():
                return name
    return options[0] if options else ""


def _nested(samples):
    return getattr(samples, "is_nested", False)


def _node_result(output):
    if isinstance(output, (tuple, list)):
        return output
    return output.result if hasattr(output, "result") else output.args


# ------------------------------------------------------------------ LoRA + Sharp curve branches

def _load_lora(path, dtype):
    """{diffusers module: (A, B in ComfyUI row order, ComfyUI weight key, slot)} of the merged LoRA, read one tensor
    at a time so peak RAM is the converted copy; the last file stays cached."""
    from safetensors import safe_open
    key = (path, os.path.getmtime(path), dtype)
    if key not in _LORA_CACHE:
        _LORA_CACHE.clear()
        pairs = {}
        with safe_open(path, framework="pt", device="cpu") as f:
            for module in core.lora_modules(set(f.keys())):
                target, slot = core.comfy_target(module)
                b = f.get_tensor(module + ".lora_B.weight").to(dtype)
                if slot == "swiglu":
                    b = core.swap_swiglu_halves(b)
                pairs[module] = (f.get_tensor(module + ".lora_A.weight").to(dtype), b, target, slot)
        _LORA_CACHE[key] = pairs
    return _LORA_CACHE[key]


def _read_table(path):
    from safetensors import safe_open
    with safe_open(path, framework="pt", device="cpu") as f:
        names = [k for k in f.keys() if k.endswith("adaln_t_table")]
        if not names:
            raise ValueError(f"{os.path.basename(path)} has no adaln_t_table (pick a pruned / curve-form H3 checkpoint, "
                             "e.g. minimax_h3_ref2va_pruned_int8_convrot)")
        return f.get_tensor(names[0]).to(torch.float32).contiguous()


def _sha256(table):
    return hashlib.sha256(table.to(torch.float32).contiguous().numpy().tobytes()).hexdigest()


class _CurveState:
    """Shared by every block's wrapper: where the model time of a forward comes from."""

    def __init__(self, table, mode, model_table=None):
        self.table = table              # reference curve (1025 x 8) the Sharp branches were trained on
        self.mode = mode                # "same": the AdaLN input already is that curve; "invert": find t on the
        self.model_table = model_table  # model's own curve; "recorded": t from the time embedder
        self.t = None
        self.warned = False
        self._device_tables = {}

    def table_on(self, device):
        if device not in self._device_tables:
            self._device_tables[device] = self.table.to(device)
        return self._device_tables[device]

    def coords(self, adaln_input):
        device = adaln_input.device
        if self.mode == "same":
            return adaln_input.to(torch.float32)
        if self.mode == "invert":
            t, err = core.invert_table(self.model_table, adaln_input.to(torch.float32))
            if err > 1e-3 and not self.warned:
                self.warned = True
                LOG.warning("Open-H3 Sharp curve: AdaLN input is %.2g off the model's curve", err)
            return core.table_coords(self.table_on(device), t)
        t = self.t
        if t is None or t.shape[0] != adaln_input.shape[0]:
            if not self.warned:
                self.warned = True
                LOG.warning("Open-H3 Sharp curve: no matching timestep recorded, curve branches skipped")
            return None
        return core.table_coords(self.table_on(device), t.to(device))


def _subclass(module, name, body):
    """Shallow copy of `module` with a subclass of its class: same parameters / child modules, so every weight key and
    ComfyUI's casting / LoRA patching are unchanged; only forward differs."""
    out = copy.copy(module)
    out.__class__ = type(name, (module.__class__,), body)
    return out


def _curve_linear(linear, weight, state, strength):
    def forward(self, *args, **kwargs):
        out = super(self.__class__, self).forward(*args, **kwargs)
        coords = state.coords(args[0] if args else kwargs["input"])
        if coords is None:
            return out
        w = self._open_h3_weights.get(coords.device)
        if w is None:
            w = self._open_h3_weights[coords.device] = weight.to(coords.device, torch.float32)
        return out + (coords @ w.t() * strength).to(out.dtype)

    wrapped = _subclass(linear, "OpenH3SharpCurveLinear", {"forward": forward})
    object.__setattr__(wrapped, "_open_h3_weights", {})
    return wrapped


def _recording_embedder(embedder, state):
    def forward(self, t):
        state.t = t.detach().to(torch.float32)
        return super(self.__class__, self).forward(t)

    return _subclass(embedder, "OpenH3RecordingTimeEmbedder", {"forward": forward})


class OpenH3RegenLora:
    @classmethod
    def INPUT_TYPES(cls):
        loras = folder_paths.get_filename_list("loras")
        checkpoints = folder_paths.get_filename_list("diffusion_models")
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (loras, {"default": _pick(loras, "open_h3_regenerate"),
                                      "tooltip": "lora/open_h3_regenerate_2k_lora.safetensors (merged refiner + Turbo + "
                                                 "LMS Sharp, Diffusers keys, FP32). Converted to ComfyUI's H3 keys here."}),
                "curve_name": ([NO_CURVE] + loras, {"default": _pick(loras, "sharp_curve_branches") or NO_CURVE,
                                                   "tooltip": "lora/sharp_curve_branches.safetensors: LMS Sharp's 50 "
                                                              "AdaLN curve branches (part of the reference recipe)."}),
                "curve_table": ([OWN_TABLE] + checkpoints, {
                    "default": _pick(checkpoints, "ref2va_pruned_int8_convrot", "ref2va_pruned") or OWN_TABLE,
                    "tooltip": "Curve the Sharp branches read: adaln_t_table of the reference checkpoint "
                               "(Comfy-Org minimax_h3_ref2va_pruned_int8_convrot). Only a few KB are read."}),
                "strength": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01,
                                       "tooltip": "The merge is exact at 1.0 (strengths are folded in)."}),
                "curve_strength": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "lora_dtype": (["bf16", "fp32"], {"default": "bf16",
                                                  "tooltip": "RAM copy of the rank-576 factors: bf16 6.2 GB, fp32 12.4 GB."}),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "report")
    FUNCTION = "apply"
    CATEGORY = CATEGORY
    DESCRIPTION = "Applies the Open-H3-Regenerate-2K LoRA (refiner + LightX2V FL2VA Turbo + LMS Sharp) to a MiniMax H3 model."

    def apply(self, model, lora_name, curve_name, curve_table, strength, curve_strength, lora_dtype):
        dtype = torch.bfloat16 if lora_dtype == "bf16" else torch.float32
        pairs = _load_lora(folder_paths.get_full_path_or_raise("loras", lora_name), dtype)
        state_dict = model.model.state_dict()
        lora, to_load = {}, {}
        for module, (a, b, key, slot) in pairs.items():
            if key not in state_dict:
                raise ValueError(f"{key} not in the model: connect a MiniMax H3 model ({module})")
            rows = state_dict[key].shape[0]
            if isinstance(slot, int):
                if rows != 3 * b.shape[0]:
                    raise ValueError(f"{key}: {rows} rows, LoRA {module} gives {b.shape[0]} per q/k/v")
                target = (key, (0, slot * b.shape[0], b.shape[0]))
            else:
                if rows != b.shape[0]:
                    raise ValueError(f"{key}: {rows} rows, LoRA {module} gives {b.shape[0]}")
                target = key
            lora[module + ".lora_A.weight"], lora[module + ".lora_B.weight"] = a, b
            to_load[module] = target
        patches = comfy.lora.load_lora(lora, to_load)
        m = model.clone()
        applied = m.add_patches(patches, strength)
        report = [f"LoRA {os.path.basename(lora_name)}: {len(pairs)} modules ({len(applied)} weight patches), "
                  f"strength {strength:g}, {lora_dtype}"]

        if curve_name != NO_CURVE and curve_strength != 0:
            report.append(self._apply_curves(m, curve_name, curve_table, curve_strength))
        else:
            report.append("Sharp curve branches: off (not the reference recipe)")
        text = "\n".join(report)
        LOG.info(text)
        return (m, text)

    @staticmethod
    def _apply_curves(m, curve_name, curve_table, strength):
        from safetensors import safe_open
        dm = m.get_model_object("diffusion_model")
        curve_form = bool(getattr(dm, "use_adaln_curves", False))
        own = dm.adaln_t_table.detach().to("cpu", torch.float32) if curve_form else None
        if curve_table == OWN_TABLE:
            if own is None:
                raise ValueError("this H3 model has no adaln_t_table: pick a pruned checkpoint in curve_table")
            table = own
        else:
            table = _read_table(folder_paths.get_full_path_or_raise("diffusion_models", curve_table))
        sha = _sha256(table)
        tag = "reference table" if sha == core.REFERENCE_TABLE_SHA256 else f"NOT the reference table (sha {sha[:12]})"

        if curve_form:
            mode = "same" if own.shape == table.shape and torch.equal(own, table) else "invert"
        else:
            mode = "recorded"
        state = _CurveState(table, mode, own)
        path = folder_paths.get_full_path_or_raise("loras", curve_name)
        with safe_open(path, framework="pt", device="cpu") as f:
            blocks = core.curve_blocks(set(f.keys()))
            if len(blocks) != len(dm.blocks):
                raise ValueError(f"{curve_name}: {len(blocks)} curve branches for {len(dm.blocks)} H3 blocks")
            for index, (key_a, key_b) in sorted(blocks.items()):
                a, b = f.get_tensor(key_a).to(torch.float32), f.get_tensor(key_b).to(torch.float32)
                name = f"diffusion_model.blocks.{index}.adaln_proj.linear"
                linear = m.get_model_object(name)
                if b.shape[0] != linear.weight.shape[0] or a.shape[1] != table.shape[1]:
                    raise ValueError(f"{curve_name}: branch {index} {tuple(b.shape)} @ {tuple(a.shape)} does not fit "
                                     f"{name} {tuple(linear.weight.shape)}")
                m.add_object_patch(name, _curve_linear(linear, b @ a, state, strength))
        if mode == "recorded":
            m.add_object_patch("diffusion_model.time_embedder",
                               _recording_embedder(m.get_model_object("diffusion_model.time_embedder"), state))
        how = {"same": "the model's AdaLN curve is the reference curve",
               "invert": "model time read back from the model's own curve, branches use the reference curve",
               "recorded": "full (non-pruned) model, model time recorded from the time embedder"}[mode]
        return f"Sharp curve branches: {len(blocks)} at {strength:g}, {tag}; {how}"


# ------------------------------------------------------------------ sigmas

class OpenH3RegenSigmas:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "steps": ("INT", {"default": 3, "min": 1, "max": 8,
                              "tooltip": "3 = release setting (0.909375, 0.857, 0.667). 2 = 0.7, 0.494: ~30% faster, "
                                         "softer fine texture."}),
            "start_sigma": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001,
                                      "tooltip": "0 = the release grid for 2 / 3 steps. Above 0: the refiner's shift-6 "
                                                 "grid from this start noise (e.g. 0.8 for the resolution ladder)."}),
        }}

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "sigmas"
    CATEGORY = CATEGORY

    def sigmas(self, steps, start_sigma):
        return (torch.tensor(core.refine_sigmas(steps, start_sigma), dtype=torch.float32),)


# ------------------------------------------------------------------ frames

class OpenH3RegenPrepareDraft:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE", {"tooltip": "Draft frames at 24 fps (load with force_rate 24)."}),
            "size": (SIZES, {"default": SIZES[0],
                             "tooltip": "768p for H3 drafts, 576p for drafts from other video models (as in the "
                                        "reference). Portrait drafts get the portrait size."}),
            "max_frames": ("INT", {"default": 121, "min": 0, "max": 3600,
                                   "tooltip": "121 = 5 s (reference). The refine supports 5-15 s (360 frames) in one "
                                              "pass. 0 = all frames."}),
            "source_fps": ("FLOAT", {"default": 24.0, "min": 0.0, "max": 240.0, "step": 0.001,
                                     "tooltip": "Frame rate of the frames (Get Video Components fps). Anything other "
                                                "than 24 is resampled to 24 fps (nearest frame, like ffmpeg fps=24). "
                                                "0 = treat as 24."}),
        }}

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT")
    RETURN_NAMES = ("images", "frame_count", "width", "height")
    FUNCTION = "prepare"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Cover-scales (bilinear) and centre-crops the draft, cuts it to max_frames and holds the last frame up "
                   "to H3's 17k+5 frame grid (121 -> 124 = 37 latent frames).")

    def prepare(self, images, size, max_frames, source_fps=24.0, chunk=16):
        if source_fps and abs(source_fps - 24.0) > 1e-3:
            images = images[core.resample_indices(images.shape[0], source_fps)]
        count = images.shape[0] if max_frames == 0 else min(images.shape[0], max_frames)
        src_h, src_w = images.shape[1], images.shape[2]
        if size.startswith("keep"):
            width, height = max(32, round(src_w / 32) * 32), max(32, round(src_h / 32) * 32)
        else:
            width, height = core.orient(core.INPUT_SIZES[size.split()[0]], src_w, src_h)
        sw, sh, x0, y0 = core.cover_crop_box(src_w, src_h, width, height)
        out = []
        for start in range(0, count, chunk):
            part = images[start:min(count, start + chunk), :, :, :3].movedim(-1, 1).to(torch.float32)
            if (sw, sh) != (src_w, src_h):
                part = F.interpolate(part, size=(sh, sw), mode="bilinear", align_corners=False,
                                     antialias=sw < src_w or sh < src_h)
            out.append(part[:, :, y0:y0 + height, x0:x0 + width].clamp(0, 1).movedim(1, -1))
        frames = torch.cat(out)
        padded = core.align_frame_count(count)
        if padded > count:
            frames = torch.cat([frames, frames[-1:].expand(padded - count, -1, -1, -1)])
        return (frames.contiguous(), count, width, height)


class OpenH3RegenFinish:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "frame_count": ("INT", {"default": 121, "min": 0, "max": 100000,
                                    "tooltip": "Frames to keep (Prepare Draft's frame_count). 0 = all."}),
            "crop": (["auto (1088 -> 1080, 2176 -> 2160)", "none"], {"default": "auto (1088 -> 1080, 2176 -> 2160)"}),
        }, "optional": {
            "audio": ("AUDIO", {"tooltip": "Soundtrack to mux; cut to the kept frames (frame_count / 24 s)."}),
        }}

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "finish"
    CATEGORY = CATEGORY

    def finish(self, images, frame_count, crop, audio=None):
        if frame_count:
            images = images[:frame_count]
        if audio is not None:
            keep = round(images.shape[0] / 24 * int(audio["sample_rate"]))
            waveform = audio["waveform"][..., :keep]
            audio = {**audio, "waveform": F.pad(waveform, (0, keep - waveform.shape[-1]))}
        if crop.startswith("auto"):
            trim = {1088: 1080, 2176: 2160}
            h, w = images.shape[1], images.shape[2]
            if h in trim and w > h:
                top = (h - trim[h]) // 2
                images = images[:, top:top + trim[h]]
            elif w in trim and h > w:
                left = (w - trim[w]) // 2
                images = images[:, :, left:left + trim[w]]
        return (images.contiguous(), audio)


# ------------------------------------------------------------------ latents

class OpenH3RegenUpscale:
    @classmethod
    def INPUT_TYPES(cls):
        try:
            models = [n for n in folder_paths.get_filename_list(LBH_FOLDER)
                      if os.path.splitext(n)[1].lower() in (".pth", ".safetensors")]
        except Exception:
            models = []
        return {"required": {
            "latent": ("LATENT", {"tooltip": "Draft latent (video or H3 AV latent; audio passes through)."}),
            "model_name": (models or ["(put the H3 latent upscaler in models/latent_upscale_models)"],
                           {"default": _pick(models, "minimax_h3_latent_upscaler_3d")}),
            "resolution": (list(RESOLUTION_NAMES), {"default": "2k (2560x1440)"}),
            "precision": (["bf16", "fp16", "fp32"], {"default": "bf16", "tooltip": "Reference: bf16."}),
            "temporal_chunking": ("BOOLEAN", {"default": False, "tooltip": "Reference: off. On saves VRAM."}),
        }}

    RETURN_TYPES = ("LATENT", "INT", "INT")
    RETURN_NAMES = ("latent", "width", "height")
    FUNCTION = "upscale"
    CATEGORY = CATEGORY
    DESCRIPTION = ("H3-Latent-Upscaler (LBH 3D conv) straight onto the target grid, no decode / re-encode. 4K runs two "
                   "passes (to 2K, then 4K) so each stays within the trained range.")

    def upscale(self, latent, model_name, resolution, precision, temporal_chunking):
        cls = nodes.NODE_CLASS_MAPPINGS.get(LBH_NODE)
        if cls is None:
            raise RuntimeError("Open-H3 Regenerate · Latent Upscale needs Comfyui_Minimax_h3_latent_Upscaler "
                               "(github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler)")
        mode_enum = sys.modules[cls.__module__].UpscaleMode.TARGET_DIMENSIONS
        samples = latent["samples"]
        video = samples.tensors[0] if _nested(samples) else samples
        lat_h, lat_w = video.shape[-2], video.shape[-1]
        tw, th, kw, kh = core.output_size(RESOLUTION_NAMES[resolution], lat_w * 16, lat_h * 16)
        device = "cuda" if comfy.model_management.get_torch_device().type == "cuda" else "cpu"

        steps = []
        cap_w, cap_h = core.orient(core.UPSCALER_MAX[:2], tw, th)
        if th * tw > cap_w * cap_h and lat_h * lat_w * 256 < cap_w * cap_h:
            steps.append((cap_w, cap_h))
        steps.append((tw, th))
        for w, h in steps:
            if (video.shape[-1] * 16, video.shape[-2] * 16) == (w, h):
                continue
            out = _node_result(cls.execute({"samples": video}, model_name, {"mode": mode_enum, "width": w, "height": h},
                                           align=32, enable_temporal_chunking=temporal_chunking,
                                           force_unload=True, device=device, precision=precision))
            video = out[0]["samples"]
        if (video.shape[-1] * 16, video.shape[-2] * 16) != (tw, th):
            raise ValueError(f"upscaler returned {video.shape[-1] * 16}x{video.shape[-2] * 16}, expected {tw}x{th}")
        result = {k: v for k, v in latent.items() if k not in ("samples", "noise_mask")}
        result["samples"] = comfy.nested_tensor.NestedTensor((video, samples.tensors[1])) if _nested(samples) else video
        return (result, kw, kh)


class OpenH3RegenRefineLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "Upscaled latent (video or AV)."}),
                "audio_mode": (AUDIO_MODES, {"default": AUDIO_MODES[0], "tooltip":
                    "Audio stream held clean (never denoised) during the refine. Reference: encoded silence. "
                    "The output soundtrack is not affected - mux the draft's audio at the end."}),
            },
            "optional": {
                "audio_vae": ("VAE", {"tooltip": "H3 audio VAE (for silence / draft audio)."}),
                "audio": ("AUDIO", {"tooltip": "Draft soundtrack, for 'draft audio (AUDIO input)'."}),
            },
        }

    RETURN_TYPES = ("LATENT", "INT")
    RETURN_NAMES = ("latent", "frame_count")
    FUNCTION = "build"
    CATEGORY = CATEGORY
    DESCRIPTION = ("H3 AV latent for the refine: the upscaled video is re-noised by the sampler (start sigma 0.909), "
                   "the audio stream is masked out and stays clean at model time 1.")

    def build(self, latent, audio_mode, audio_vae=None, audio=None):
        samples = latent["samples"]
        video = samples.tensors[0] if _nested(samples) else samples
        if video.ndim != 5 or video.shape[1] != 24:
            raise ValueError(f"expected an H3 video latent [B, 24, T, H, W], got {tuple(video.shape)}")
        frames = core.frames_for_latent_t(video.shape[2])
        audio_t = round(frames / 24 * 40)
        if audio_mode == AUDIO_MODES[2]:
            if not _nested(samples):
                raise ValueError("'keep latent audio' needs an H3 AV latent")
            sound = samples.tensors[1]
        else:
            if audio_vae is None:
                raise ValueError(f"'{audio_mode}' needs the H3 audio VAE connected")
            rate = int(getattr(audio_vae, "audio_sample_rate", 32000))
            length = round(frames / 24 * rate)      # reference: 124 frames -> 5.17 s of silence -> 207 latents
            if audio_mode == AUDIO_MODES[0]:
                waveform = torch.zeros(1, 2, length)
            else:
                if audio is None:
                    raise ValueError("'draft audio (AUDIO input)' needs the draft's AUDIO connected")
                waveform = audio["waveform"][:1].to(torch.float32)
                if int(audio["sample_rate"]) != rate:
                    import torchaudio
                    waveform = torchaudio.functional.resample(waveform, int(audio["sample_rate"]), rate)
                waveform = waveform.expand(-1, 2, -1) if waveform.shape[1] == 1 else waveform[:, :2]
                waveform = F.pad(waveform, (0, max(0, length - waveform.shape[-1])))[..., :length]
            sound = audio_vae.encode(waveform.movedim(1, -1))
        sound = sound.to(video.device, torch.float32)
        if sound.shape[-1] < audio_t:
            sound = torch.cat([sound, sound[..., -1:].expand(*sound.shape[:-1], audio_t - sound.shape[-1])], dim=-1)
        sound = sound[..., :audio_t].contiguous()
        result = {k: v for k, v in latent.items() if k not in ("samples", "noise_mask")}
        result["samples"] = comfy.nested_tensor.NestedTensor((video, sound))
        result["noise_mask"] = comfy.nested_tensor.NestedTensor((torch.ones_like(video), torch.zeros_like(sound)))
        return (result, frames)


# ------------------------------------------------------------------ fast H3 VAE decoder

_FAST_VAE = {}
def _fast_vae(path):
    key = (path, os.path.getmtime(path))
    if key not in _FAST_VAE:
        _FAST_VAE.clear()
        from comfy.ldm.minimax.vae import LATENTS_MEAN, LATENTS_STD
        from .fast_vae_decoder import KEEP
        from .fast_vae_lean import LeanNonCausalViTDecoder
        # released H3 VAE decoder config (MiniMax-H3 vae/config.json): 32 heads x 64, RoPE theta 100 on 3/4 of each head
        model = LeanNonCausalViTDecoder(LATENTS_MEAN, LATENTS_STD, KEEP[24], time_norm=5, heads=32, dim_head=64,
                                    rope_theta=100.0, rope_dim_ratio=0.75, eps=1e-5, latent_channels=24)
        model.load_state_dict(comfy.utils.load_torch_file(path, safe_load=True), strict=True)
        model.eval().requires_grad_(False)
        _FAST_VAE[key] = model
    return _FAST_VAE[key]


class OpenH3RegenFastVAEDecode:
    @classmethod
    def INPUT_TYPES(cls):
        vaes = folder_paths.get_filename_list("vae")
        return {"required": {
            "samples": ("LATENT", {"tooltip": "H3 video or AV latent (audio is ignored)."}),
            "vae_name": (vaes, {"default": _pick(vaes, "fast_h3_vae_noncausal"),
                                "tooltip": "vae_fast/fast_h3_vae_noncausal_24l.safetensors (StreamRender-H3, 6.5 GB)."}),
            "keep_loaded": ("BOOLEAN", {"default": False,
                                        "tooltip": "Keep the decoder on the GPU after decoding (6.5 GB)."}),
        }, "optional": {
            "vae": ("VAE", {"tooltip": "H3 video VAE. Used instead when the clip is too big for the fast decoder on this "
                                       "GPU (it then decodes tiled, like VAE Decode)."}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = CATEGORY
    DESCRIPTION = ("NVIDIA's fast H3 VAE decoder (StreamRender-H3): the released 36-layer decoder pruned to 24 layers and "
                   "distilled, decoding whole 5-latent blocks with window attention and no tiling. Plain PyTorch "
                   "(SDPA) engine. Slightly lossy (about 30 dB against the released decoder). VRAM grows with clip "
                   "size (~26 KB per latent token + 6.5 GB weights: 5 s 2K ≈ 20 GB, 15 s 2K ≈ 47 GB).")

    @staticmethod
    def _native(vae, samples, why):
        LOG.warning("Open-H3 fast VAE decode: %s - decoding with the connected H3 VAE instead", why)
        images = vae.decode(samples["samples"])
        return (images.reshape(-1, *images.shape[-3:]) if images.ndim == 5 else images,)

    def decode(self, samples, vae_name, keep_loaded, vae=None):
        model = _fast_vae(folder_paths.get_full_path_or_raise("vae", vae_name))
        z = samples["samples"]
        z = z.tensors[0] if _nested(z) else z
        if z.ndim != 5 or z.shape[1] != 24:
            raise ValueError(f"expected an H3 video latent [B, 24, T, H, W], got {tuple(z.shape)}")
        device = comfy.model_management.get_torch_device()
        B, _, T, lat_h, lat_w = z.shape
        frames = 4 * T - 3 * ((T + 4) // 5)                    # the first latent of every 5-latent block gives 1 frame
        weights = sum(p.numel() * p.element_size() for p in model.parameters())
        # x (fp32) + q / k / v / attention output (bf16) for the whole clip + one window's / chunk's temporaries
        need = weights + T * lat_h * lat_w * 26 * 1024 + 3 * 2 ** 30
        comfy.model_management.free_memory(need, device)
        if vae is not None and device.type == "cuda" and need > comfy.model_management.get_free_memory(device):
            return self._native(vae, samples, f"needs ~{need / 2 ** 30:.0f} GB, "
                                              f"{comfy.model_management.get_free_memory(device) / 2 ** 30:.0f} GB free")
        out = torch.empty(B * frames, lat_h * 16, lat_w * 16, 3, device=comfy.model_management.intermediate_device())
        mean = torch.tensor((.485, .456, .406), device=device).view(1, 3, 1, 1, 1)
        std = torch.tensor((.229, .224, .225), device=device).view(1, 3, 1, 1, 1)
        started = time.perf_counter()
        try:
            model.to(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16):
                for b in range(B):
                    def sink(f, pixels, base=b * frames):         # pixels [1, 3, n, H, W], ImageNet-normalized
                        part = (pixels * std + mean).clamp_(0, 1)[0].permute(1, 2, 3, 0)
                        out[base + f:base + f + part.shape[0]].copy_(part)
                    model(z[b:b + 1].to(device, torch.float32).transpose(1, 2), sink=sink)
            peak = torch.cuda.max_memory_allocated(device) / 2 ** 30 if device.type == "cuda" else 0.0
            LOG.info("Open-H3 fast VAE decode: %s latent -> %d frames in %.1f s, peak %.1f GB", tuple(z.shape),
                     out.shape[0], time.perf_counter() - started, peak)
        except torch.OutOfMemoryError:
            if vae is None:
                raise RuntimeError(f"fast H3 VAE decoder ran out of VRAM ({tuple(z.shape)} latent, needs ~"
                                   f"{need / 2 ** 30:.0f} GB): connect the H3 video VAE to this node's `vae` input to "
                                   "fall back to the tiled decoder automatically") from None
            out = None
        finally:
            if not keep_loaded:
                model.to(comfy.model_management.unet_offload_device())
            comfy.model_management.soft_empty_cache()
        if out is None:
            return self._native(vae, samples, "out of VRAM")
        return (out,)


# ------------------------------------------------------------------ video -> H3 prompt (vision LLM server)

SEND_AS = ["video (motion)", "frames (8 stills)"]


class OpenH3VideoToPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        from . import llm_prompt
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The clip's frames (Get Video Components images)."}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001,
                                  "tooltip": "Frame rate of the frames (Get Video Components fps)."}),
                "server_url": ("STRING", {"default": "http://127.0.0.1:8000/v1",
                                          "tooltip": "OpenAI-compatible base URL (vLLM, SGLang, llama.cpp, LM Studio). "
                                                     "HTTPS certificates are always verified."}),
                "model": ("STRING", {"default": "", "tooltip": "Model id; empty = the server's first listed model."}),
                "api_key_env": ("STRING", {"default": "H3_LLM_API_KEY",
                                           "tooltip": "Name of the environment variable holding the API key (the key "
                                                      "itself never goes into the workflow). Empty = no key."}),
                "send_as": (SEND_AS, {"default": SEND_AS[0],
                                      "tooltip": "video: a small H.264 clip at 4 fps (the model sees the motion). "
                                                 "frames: 8 JPEG stills, for servers without video input."}),
                "format": (llm_prompt.FORMATS, {"default": llm_prompt.FORMATS[0]}),
                "extra_instructions": ("STRING", {"multiline": True, "default": "",
                                                  "tooltip": "Optional notes for the model: names, style words, what "
                                                             "to emphasise."}),
                "max_tokens": ("INT", {"default": 1500, "min": 64, "max": 16384}),
                "temperature": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 2.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2 ** 31 - 1,
                                 "tooltip": "0 = server default. Change it to get a new variant."}),
                "thinking": ("BOOLEAN", {"default": False, "tooltip": "Qwen thinking mode (slower)."}),
                "manual_prompt": ("STRING", {"multiline": True, "default": "",
                                             "tooltip": "Leave empty to generate. Any text here is passed through "
                                                        "as the prompt and the server is not called."}),
            },
            "optional": {
                "ca_file": ("STRING", {"default": "", "tooltip": "PEM file of the CA that signed the server's "
                                                                 "certificate, if Windows does not trust it already."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "report")
    FUNCTION = "describe"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Writes a MiniMax H3 prompt for an existing video with a vision model on your inference server "
                   "(OpenAI-compatible API).")

    def describe(self, images, fps, server_url, model, api_key_env, send_as, format, extra_instructions, max_tokens,
                 temperature, seed, thinking, manual_prompt, ca_file=""):
        from . import llm_prompt
        if manual_prompt.strip():
            return (manual_prompt.strip(), "manual_prompt used as is; the server was not called")
        key = llm_prompt.api_key(api_key_env)
        model = model.strip() or llm_prompt.first_model(server_url, key, ca_file)
        duration = images.shape[0] / fps
        started = time.perf_counter()
        if send_as == SEND_AS[0]:
            data, sent, size = llm_prompt.encode_video(images, fps)
            media, what = ("video", data), f"{sent} frames as {size[0]}x{size[1]} H.264 ({len(data) // 1024} KB)"
        else:
            stills = llm_prompt.encode_frames(images, fps)
            media, what = ("frames", stills), f"{len(stills)} JPEG frames"
        text = llm_prompt.instructions(format, duration, extra_instructions)
        raw, response = llm_prompt.generate(server_url, key, model, text, media, max_tokens, temperature, seed,
                                            thinking, ca_file)
        prompt = llm_prompt.clean_response(raw, format)
        usage = response.get("usage") or {}
        report = (f"{model} @ {server_url}: {duration:.1f} s clip sent as {what}; {time.perf_counter() - started:.1f} s, "
                  f"{usage.get('prompt_tokens', '?')} prompt + {usage.get('completion_tokens', '?')} completion tokens")
        if response["choices"][0].get("finish_reason") == "length":
            report += "\nWARNING: cut off at max_tokens — raise max_tokens"
        if format == llm_prompt.FORMATS[0] and llm_prompt.missing_sections(prompt):
            report += "\nWARNING: missing H3 sections: " + ", ".join(llm_prompt.missing_sections(prompt))
        if not prompt:
            raise RuntimeError("the server returned an empty prompt (" + report + ")")
        LOG.info(report)
        return (prompt, report)


NODE_CLASS_MAPPINGS = {
    "OpenH3VideoToPrompt": OpenH3VideoToPrompt,
    "OpenH3RegenFastVAEDecode": OpenH3RegenFastVAEDecode,
    "OpenH3RegenLora": OpenH3RegenLora,
    "OpenH3RegenSigmas": OpenH3RegenSigmas,
    "OpenH3RegenPrepareDraft": OpenH3RegenPrepareDraft,
    "OpenH3RegenUpscale": OpenH3RegenUpscale,
    "OpenH3RegenRefineLatent": OpenH3RegenRefineLatent,
    "OpenH3RegenFinish": OpenH3RegenFinish,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenH3VideoToPrompt": "Open-H3 · Video to H3 Prompt (inference server)",
    "OpenH3RegenFastVAEDecode": "Open-H3 Regenerate · Fast VAE Decode (StreamRender-H3)",
    "OpenH3RegenLora": "Open-H3 Regenerate · LoRA (refiner + Turbo + Sharp)",
    "OpenH3RegenSigmas": "Open-H3 Regenerate · Sigmas",
    "OpenH3RegenPrepareDraft": "Open-H3 Regenerate · Prepare Draft",
    "OpenH3RegenUpscale": "Open-H3 Regenerate · Latent Upscale (LBH 3D)",
    "OpenH3RegenRefineLatent": "Open-H3 Regenerate · Refine Latent",
    "OpenH3RegenFinish": "Open-H3 Regenerate · Finish Frames",
}
