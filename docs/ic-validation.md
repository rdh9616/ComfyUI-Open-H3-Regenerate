# IC release validation record

Recorded 2026-11-09. This is a staged validation record, not a completed IC release claim.

| Gate | Evidence / state |
| --- | --- |
| IC conditioning, geometry, row times and update rules | Task #5 / PR #6; CPU oracle/native checks and actual conditioning/upscale/audio smoke passed. See [contract](ic-conditioning-contract.md). |
| Pinned setup and existing files | [Manifest](../weights/ic-manifest.json), read-only checker; local full-file hashes passed for the base, Qwen, VAEs, upscaler, HyperFlow source and v2 decoder. |
| Recovery / standard-decoder fallback | Saved AV latent, frame count and original waveform round-trip passed. Regression failed before extracting video for the native fallback and passed afterward. Recovery example imported as seven nodes with no missing-node dialog in isolated CPU ComfyUI; no job queued there. |
| Legacy 0.8 reference run | Completed below, including standard-VAE decode, saved latent and audio mux. |
| Endpoint-aware HyperFlow and IC+Sharp merge | Task #6 dependency pending. Explicit sigmas alone do not satisfy it. |
| Native IC up3 workflow and matched native-versus-legacy comparison | Pending the adapter/merge; no native IC denoising result is claimed. |
| Matched pinned-upstream IC comparison | Pending a human decision on the original BF16 base and separate upstream runtime. Coordinator approved this staged handoff; no environment or base download was performed. |

## Public comparison input and caption

The input is [U06 from the pinned Sana-assets revision](https://huggingface.co/datasets/Efficient-Large-Model/Sana-assets/resolve/1a95af6fb9c7c5e20e12ace64928322f430304c1/Open-H3-Regenerate-2K/assets/sol-engine/ic/U06-input.mp4):
1344x768, 121 frames at 24 fps, padded to 124 inside H3. SHA256:
`9fc727dc84cd872991350de93c8e57e6f732df607b0d88ef0a0236409b34a079`.

Both native comparison runs use seed **42** and this identical authored caption. It is not recovered upstream metadata:

> A woman with long braids with gold accents, gold hoop earrings and layered gold necklaces wears a dark teal velvet
> jacket against a plain dark teal studio background. In a steady chest-up portrait she speaks toward the camera,
> with subtle head movements and a changing smile. Soft even studio lighting.

An empty caption would remove the legacy refiner's text guidance while IC still has its draft context, so it is
not used for this comparison. The [published IC output](https://huggingface.co/datasets/Efficient-Large-Model/Sana-assets/resolve/1a95af6fb9c7c5e20e12ace64928322f430304c1/Open-H3-Regenerate-2K/assets/sol-engine/ic/v2/U06-ic.mp4)
has 124 frames at 2560x1440 and no audio; SHA256 is
`e57b5e82a99f9ec83235fe366e416f53a7a9ce1ff74179911810d329e83bc30a`. Its caption and seed are unknown. It is an
**unmatched visual reference**, not a controlled comparison or a local upstream reproduction.

## Completed legacy 0.8 run

The [machine-readable receipt](validation/legacy-u06.json) includes exact unrounded times, local legacy weight
hashes, output hashes and recorded tensor/container metadata. Shared IC/legacy weights are pinned in the setup
manifest. The receipt identifies the node-pack and native ComfyUI revisions used by this run.

One isolated standalone process used the published legacy node path, full input soundtrack, official Qwen NVFP4,
FL2VA pruned int8 convrot, merged legacy refiner/Turbo/Sharp at strength 1, Sharp curves at strength 1, and stock
FP16-stored video VAE with FP32 computation. The caption was encoded as text only. VAE encoding retained the
legacy posterior mean; the legacy 206-row silence plus repeated final row was deliberately unchanged.

Sampling used Euler, CFG 1 and float32 sigmas `[0.8, 0.6857143, 0.48, 0]`. Sol-Attn settings were tau 1.2,
start/end percent 0/0.9, dense blocks `0,1,49`, minimum tokens 12288, extra tokens 256, conditioning mode
`exact_kv_and_rows`. The final sigma evaluation lies below the configured sparse cutoff. Decoding used the
**standard H3 VAE**; the planned native IC run must use the same decoder for this comparison.

| Measured stage | Seconds |
| --- | ---: |
| Text encoder load / conditioning | 8.97 / 0.52 |
| Video VAE load / draft encode | 2.85 / 38.39 |
| Latent upscale | 4.44 |
| Audio VAE load / silence preparation | 0.24 / 0.19 |
| Transformer load / LoRA and curves load | 2.83 / 14.59 |
| Sampling, including model preparation inside the sampler | 274.72 |
| Saved refined latent | 0.03 |
| Standard FP32 VAE decode | 247.23 |
| H.264 CRF-18 video save | 5.81 |
| Total timed run | **606.36** |

Hardware was an RTX PRO 5000. Peak PyTorch CUDA allocations were **36,179,198,976 bytes** (33.70 GiB), measured
with one reset before the timed run and cumulative reads between stages. This is neither whole-card usage nor
the CUDA allocator's reserved-memory peak. GPU work was coordinated serially; an independent CPU-only UI server
started during this run. Times are one observation, include data loading between stages, and exclude process
imports/startup. They are not a speed guarantee or a controlled performance benchmark.

The output has 121 frames at 2560x1440/24 fps, SHA256
`4d8d82bcd343daff4532f98d1d4a080dae9a70231d2567cad97aa099593e7610`.
The saved latent SHA256 is `85e88067bf636ab0800d2e72d96f3e48459bbb11ca23e34b7b029cdd42709bbb`; its video is finite
`[1,24,37,90,160]`, audio latent is `[1,32,2,207]`, and metadata retains real frame count 121 and sample rate 32000.
The source soundtrack is only **55,296 samples / 1.728 seconds** long. It is retained in the latent, then padded
with silence to the video's duration by Finish Frames. The published IC reference has no audio at all.

The process unloaded its models and exited 0. ComfyUI emitted ignored `ModelPatcher.__del__` exceptions during
interpreter shutdown. No request was queued on the live ComfyUI server.

## Visual inspection and limits

Aligned stills at frames 0, 24, 60, 96 and 120 show that the legacy output keeps the scene, teal clothing, braids
and overall subject placement while changing fine facial detail. At frame 24 the mouth opening differs from the
input's pursed shape; at frame 120 the legacy smile exposes more teeth. Eyelashes, highlights and jewelry are
redrawn. These observations support a local facial/expression drift finding, not a numerical identity score.

No lettering is present in this clip. The stills show mouth-shape differences at equal frame times, but full
audio/phoneme lip-sync and temporal flicker have not been established from stills. The short source audio further
limits a full-clip speech comparison. The unmatched published IC stills appear closer to the input's mouth shapes
at these sampled times; unknown caption/seed, different decoding and the absence of a local IC render prevent
attributing that observation to our implementation.

Container metadata checks found only format brands and encoder strings in the input/output MP4s, with no prompts
or local paths. Model, caption, seed and input differences must remain visible in any later comparison.
