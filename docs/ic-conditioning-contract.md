# IC draft conditioning contract

This is an additive IC path. Existing Regenerate nodes, JSON workflows and 0.8 LoRA defaults keep their behavior.
The initial IC graph accepts **113-124 real frames at 24 fps**, pads to 124, and targets **2560x1440**.
Other lengths and output grids are rejected here until separately validated. A 15-second legacy render does not
establish 15-second IC support.

The source contract is [Sana IC at 1e3c9c2](https://github.com/NVlabs/Sana/tree/1e3c9c2/models/Open-H3-Regenerate-2K/IC).
The native implementation reviewed is ComfyUI `b5a432157de6b8884f81104cd02926adf4d597e4`.
Layout comparisons use Diffusers `8c54b38fea3d624d23d0e43c0c96037f080a79f9`, the dependency used by IC's layout module.

## Nodes and graph wiring

1. **Open-H3 IC · Prepare Draft (124 frames)** takes RGB IMAGE and its actual frame rate. It uses floor indices
   `int(i * source_fps / 24)`, bilinear cover resize (antialiased when shrinking), and a centered 1344x768 crop.
   It repeats the final real frame to 124. `frame_count` is the number of real resampled frames, for Finish Frames.
   Trim longer input explicitly before this node. Portrait input receives the same landscape cover crop; this is
   not a portrait-output mode. If the loader already resampled, supply its resulting FPS.
2. **Open-H3 IC · Encode Draft Context** takes the prepared images, H3 CLIP, stock video VAE, original caption and seed.
   It returns native CONDITIONING and the normalized video LATENT `[1,24,37,48,84]`. Upscale **that same latent** with
   the existing LBH upscaler. The caption can be empty. It is appended verbatim after the fixed IC enhancement
   instruction and `Scene description: `, rather than replacing the instruction.
3. **Open-H3 IC · Target Latent (2K)** takes reference and upscaled target LATENTs plus the H3 audio VAE. Both video
   latents must have T=37 and batch=1; the target is `[1,24,37,90,160]`. It produces a native AV LATENT containing
   encoded silence `[1,32,2,207]`, with video mask=1 and audio mask=0. Supply the corresponding IC schedule and
   HyperFlow adapter from the sampler integration. This node does not make the legacy sigma schedule IC-compatible.

Keep the original soundtrack outside conditioning and denoising. Connect it to Finish Frames / Create Video, and
to Save Refined Latent when retaining a recoverable output. The reference has no audio-reference block.

## Encoding and normalization

IC's VAE uses FP32 arithmetic, ImageNet pixel statistics, 256-pixel tiles with at least 64-pixel overlap, and
17-frame temporal chunks. The stock native H3 VAE supplies these operations. Native `encode` normally returns the
posterior **mean**; IC instead samples its diagonal Gaussian with a seeded CPU generator, clamps log variance to
[-30,20], then normalizes `(sample - latents_mean) / latents_std` and rounds to bfloat16.

The IC facade changes this last operation per call; it does not patch a shared VAE or modify ComfyUI. Both ordinary
and tiled fallback entry points sample after spatial blending and temporal concatenation. ComfyUI still manages
loading, offloading and chunked input transfer. An FP16 compute VAE is rejected; start the isolated validation
instance with `--fp32-vae`. FP16-stored weights promoted to FP32 are still quantized weights and must be reported as
such in comparisons with upstream's original weights.

Qwen receives 11 selected frames (indices 0,12,...,120), quantized to 8-bit RGB as upstream, with timestamps
0,0.5,...,5.0 seconds. The native tokenizer pads its final two-frame vision block. Native H3 CLIP supplies tag 0
for vision and tag 1 for text. The node rejects missing vision/text tags. Native Ref2VA packing consumes the
`minimax_refs` video block; no separate transformer or new packed-layout implementation is introduced.

## Runtime and sampler boundary

The sequence is `[Qwen vision/text | draft reference video | fixed silence | target video]`. Native patching uses
1x2x2 video patches and channel-major stereo audio. Reference and target share temporal length, with distinct
spatial grids and a common timeline after the reference block.

| Stream | Input at target noise sigma | Model time | Updated? |
| --- | --- | --- | --- |
| Qwen context | encoded draft plus instruction/caption | `1-sigma` | no |
| Draft reference | `0.999*draft + 0.001*fixed_noise` | `max(1-sigma,0.999)` | no |
| Silence | clean encoded silence | `1` | no |
| Target, up3/up4 | initially `(1-sigma0)*upscaled + sigma0*noise` | `1-sigma` | yes |
| Target, hf8 | initially pure noise (`sigma0=1`) | `1-sigma` | yes |

Native CONST noise scaling supplies the initial mixture; the node must not pre-noise it a second time. hf8 clears
the target template, while up3/up4 retain the upscaled target. The sampler must keep noise scale 1 and CFG 1.
IC predicts data-ward velocity: `x_next = x + (sigma-sigma_next)*v_target`. Native ComfyUI converts the sign for its
Euler sampler. Update only target video; the audio mask is also passed into the transformer so audio runs at t=1.

The HyperFlow adapter must give target rows endpoint `r=1-sigma_next`; reference and audio endpoint remains `r=t`.
It needs the actual next sigma from the selected trained IC grid. Sharp curves use current t, never r. Do not
substitute the old shift-6 Regenerate grid or use a pruned checkpoint lacking `time_embedder`.

Native reference noise is a deterministic CPU RNG over patchified rows. Upstream IC uses its GPU generator.
The distributions and fixed-per-run behavior match; equal integer seeds alone do **not** imply identical random
tensors or bitwise-identical full-pipeline output. End-to-end comparisons must disclose this distinction.

## CPU checks

Run with the ComfyUI Python environment (no model downloads or GPU inference):

```text
python -B tests/test_ic.py <ComfyUI-dir> <scratch-dir> [before_denoise.py]
python -I tests/test_core.py
python -B tests/test_comfy_nodes.py <ComfyUI-dir> <scratch-dir>
python -I tests/test_workflows.py
```

For an independent oracle, save
[pinned before_denoise.py](https://github.com/huggingface/diffusers/blob/8c54b38fea3d624d23d0e43c0c96037f080a79f9/src/diffusers/modular_pipelines/minimax_h3/before_denoise.py)
and pass it as the final argument. The test checks its Git blob hash before evaluating only its pure layout
functions; it never installs Diffusers or fetches code. It compares positions, indices, modality tags and actual
native forward row times, including sigma below 0.001. Other checks cover seeded posterior sampling, unchanged
global RNG, frame bounds, fixed reference noise, target initialization, silence masking and malformed shapes.

These are CPU contract checks. They do not establish fidelity, speed or full IC compatibility. The workflow task
records the separate GPU smoke and end-to-end comparisons once the sampler/weight integration is available.
