# IC setup and validation

IC is a separate Ref2VA refinement path. Keep existing Regenerate workflows and their 0.8 defaults for reproducible
legacy results. Installing this pack or opening its workflows does not download or replace any model.

Integration status: the conditioning nodes are in Clawcus task #5 / PR #6; the endpoint-aware HyperFlow adapter and
trained sigma grid are task #6; decoder evaluation is task #7. The final example and end-to-end evidence belong to
task #4. Do not treat a conditioning-only or decoder-only check as an IC render.
The [validation record](ic-validation.md) separates completed checks and the legacy baseline from pending gates.

## Pinned assets

[weights/ic-manifest.json](../weights/ic-manifest.json) records each upstream repository, immutable revision,
filename, byte count and file SHA256. It pins Sana IC source to `1e3c9c27ce56ca03b203cff9f63bed225a1599cd` and the
reviewed native ComfyUI implementation to `b5a432157de6b8884f81104cd02926adf4d597e4`.
That checkout reports ComfyUI 0.43.0. Validation uses Python 3.13.11, PyTorch 2.10.0+cu130, comfy-kitchen 0.2.40,
frontend 1.57.6 and LBH upscaler commit `40316cf008b2fd8663263270669eb4da23f89d2c`. The revision pins identify the
tested environment; compatibility with earlier native H3 implementations is not assumed.

The supported validation base is the full **Ref2VA int8 convrot** model. It contains `time_embedder.proj_in` and
`proj_out`. The pruned Ref2VA, hybrid FL2VA/Ref2VA, and fused Turbo models are not substitutes: a file with only
`adaln_t_table` cannot supply HyperFlow's two time embedders. No BF16 base download is required by this setup.

The validation text encoder is the official Qwen3-VL MiniMax H3 NVFP4 AWQ file. Separately fine-tuned text encoders
are different baselines. The stock FP16-stored video VAE runs in **FP32 arithmetic**; launch the validation ComfyUI
instance with `--fp32-vae`. This does not recover precision already lost in stored weights. See the
[conditioning contract](ic-conditioning-contract.md) for the posterior-sampling and RNG distinctions from upstream.

Inspect the model directory before downloading anything:

```text
python tools/check_ic_weights.py --models-dir <ComfyUI-models-directory>
python tools/check_ic_weights.py --models-dir <ComfyUI-models-directory> --only ic,hyperflow,sharp --verify
```

The first command checks filenames and sizes and reports **unverified** files explicitly. The second streams
selected files and checks their SHA256. Existing copies in subfolders are accepted, so a different folder layout
does not trigger another download. A mismatch exits with an error and leaves the file intact. Each result includes
the revision-pinned download URL; download missing assets deliberately, then re-run `--verify`. The checker never
downloads, overwrites, renames, or deletes model files. A hub ETag or Xet identifier is not necessarily a file SHA256.

| Manifest ID | Role |
| --- | --- |
| `base` | full Ref2VA int8 transformer |
| `text_encoder` | official Qwen3-VL H3 NVFP4 AWQ |
| `video_vae` | stock H3 video VAE, executed in FP32 |
| `audio_vae` | H3 audio VAE for fixed encoded silence |
| `ic` | IC rank-384 LoRA, separate from HyperFlow |
| `hyperflow` | HyperFlow LoRA, endpoint embedder weights, gate and trained grid metadata |
| `sharp` | LMS Sharp LoRA and curve-branch source |
| `upscaler_fp32` | existing LBH 3D latent upscaler, FP32 PTH weights, executed in BF16 for validation |
| `upscaler` | upstream BF16 safetensors alternative; optional and separately hashed |
| `silence` | 124-frame upstream silent-audio fixture, for comparison |
| `decoder_v1` / `decoder_v2` | optional whole-clip fast decoders; standard video VAE remains available |

The IC + Sharp merge is a derived artifact. Its producing script/version, input hashes, strength and output hash
must accompany the integration receipt. That merge does **not** include HyperFlow. Do not omit the endpoint
embedder when converting keys, or feed a Comfy-key LoRA to a loader expecting Diffusers `transformer.*.lora_A/B`
keys. The HyperFlow integration must validate metadata, complete key coverage and the per-step endpoint.

## Decoder choices

The existing fast decoder node loads both v1 and v2 weight layouts. The original video VAE decoder remains the
correctness fallback when whole-clip memory does not fit. Choose the decoder explicitly; do not infer kernel
availability from the name “Blackwell.” Pinned upstream IC selects its FA4 engine only for CUDA capability major
10, while the tested RTX PRO 5000 is capability 12.0. FA4 on this Windows machine is unvalidated.

DSH's task #7 reports one decode-only 192-frame/2K comparison: v1 44.0 s, v2 84.3 s, standard VAE 96.5 s; SSIM to
standard was 0.9482 and 0.9549 respectively. Those are another agent's measurements on one saved latent, not an
end-to-end IC result or a general performance guarantee. The actual local v2 SHA256 was independently checked:
`b7527bf0dbac4b4ffc37e106478e5caabc96894f29cd1ebfe38cea0b44982eb0`, matching the pinned manifest.

## Resolution and duration boundary

The current conditioning nodes support a 1344x768 draft, 113-124 real frames at 24 fps, padded to 124, with a
2560x1440 target. Unsupported lengths and target shapes fail before sampling. The returned real frame count trims
the padded tail at final output; original audio is muxed after refinement.

Upstream's `4k`/`3840p` path computes 3840x2176 and crops to **3840x2160**. Its distinct `4k-dci` path computes
**4096x2304**, as shown in the project page examples. These are different grids. This initial example does not
validate either one, 1080P, 384P-to-768P, portrait output, or longer clips. Legacy 15-second results do not validate
IC at that length.

## Release evidence still required

Record the exact workflow, code and weight revisions, original caption, seed, input hash, actual frame count,
output grid, sampler, decoder, hardware and quantization. Compare a 124-frame 2K IC render with the legacy 0.8 path
and pinned upstream IC on identical input/caption/seed. Report stage times and peak memory with their measurement
method; assess identity, lettering, lip-sync when present, and artifacts. An unreported upstream demo caption/seed
must remain unconfirmed, not inferred from defaults.

The coordinator approved a staged handoff on 2026-11-09: publish the native workflow and native-versus-legacy
measurements while leaving the matched upstream comparison pending. The original BF16 base and a separate
upstream runtime are not installed or downloaded by this work. NVIDIA documents Linux, Python 3.11 and its
distributed launch path; Windows support for that reference runtime has not been established. A side-by-side
view of a published demo input/output is only an unmatched visual sanity check when its caption and seed are
unknown. It does not close the matched upstream comparison gate.

Local full-file SHA256 verification passed for the manifest's full Ref2VA int8 base, official Qwen NVFP4 encoder,
video VAE, audio VAE, FP32 upscaler, HyperFlow source and v2 decoder. Both existing copies of the base matched;
the checker retained both files. This verifies asset identity, not end-to-end compatibility.

Saved refined latents must retain the original soundtrack and real frame count so decoding can be retried without
another transformer run. Before publication, inspect exported workflow/media metadata for local paths, prompts
from private media, credentials and accidental user identifiers.

## Recovering a saved result

Open **Open-H3 IC - Decode Saved Latent** and choose the exact `open_h3_ic_*.safetensors` file. The `(newest)` option
can select a different run. This graph contains no transformer, sampler or LoRA loader. It uses the standard video
VAE and restores the saved real frame count and original soundtrack; the final output remains 24 fps.

To try the optional v2 decoder, replace VAE Decode with Open-H3 Fast VAE Decode, select the pinned v2 weights,
and keep the standard video VAE connected to its fallback input. The pack's fallback now extracts video from an
AV latent before calling the video VAE. That fixes a legacy bug too: previously either the memory-fit check or an
OOM could pass the combined AV value into a decoder that expects video alone. A CPU regression reproduced the
failure before the fix and passed afterward, including saved soundtrack/frame-count recovery and plain video.

Only a completed sampling result can be recovered. A failed transformer run has no refined latent to decode.
