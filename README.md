# flux2-vae-mlx-swift

The **FLUX.2 VAE** (decoder path) as a neutral, standalone Swift/MLX package.

Extracted from [lens-mlx-swift](https://github.com/xocialize/lens-mlx-swift) so multiple
text-to-image backers can **share the VAE without depending on each other's model packages**.
Both [Lens](https://github.com/xocialize/lens-mlx-swift) and
[ERNIE-Image-Turbo](https://github.com/xocialize/ernie-image-swift) depend on this package; neither
depends on the other. A foundational component — a future candidate to fold into MLXEngine's
utility layer (cf. [format-bridge](https://github.com/xocialize/format-bridge)).

Decoder-only (these t2i pipelines never encode): isomorphic to mflux's `flux2_vae` (the
implementation the Python `lens-mlx` port loads; VAE parity gate 57.65 dB / ~120 dB in-pipeline vs
PyTorch). Tensors flow NCHW between blocks with NHWC transposes around convs/norms, kept identical
to the reference. The `bn` running stats implement the T1 latent de-norm in patchified space.

## Use

```swift
import Flux2VAE
import MLX

// Load decoder weights from a diffusers `vae/` snapshot (strict two-way load).
let vae = try Flux2VAEWeights.loadVAE(directory: vaeDir, dtype: .bfloat16)

// Decode packed latents (bn de-norm in packed space → unpatchify → decode):
let image = vae.decodePackedLatents(latents)   // (B, 3, H, W)
// …or a plain decode of standard latents:
let image2 = vae.decode(latents)
```

Depends only on [mlx-swift](https://github.com/ml-explore/mlx-swift) — no engine, no tokenizers,
no model dependency. MIT.

## GPU numerics: the decoder's 3×3 convs (2026-09-24)

mlx's Metal `conv2d` takes a Winograd F(6×6,3×3) path when the conv is 3×3, stride 1, dilation 1,
groups 1, C % 32 == 0, O % 32 == 0, C + O ≥ 256 and N·H·W ≥ 4096. On M5 that path loses precision:
about 6.4e-3 relL2 per conv in fp32, because its inner GEMM runs TF32 (`MLX_ENABLE_TF32` defaults
on), and about 5.8e-2 in bf16.

This decoder has 32 such convs at 1024²: conv_in 32→512, the 512-, 256- and 128-channel resnets,
and the upsamplers. The consumers' own VAE gates (for example Lens P3) pin the CPU device, so the
GPU decode had never been gated. Every stride-1 3×3 conv is now a `WinogradFreeConv2d` with a route
(`vae.convRoute`, type `Flux2VAEConvRoute`). Shapes outside the window always take plain conv2d.

- `.winograd` is mlx's raw path. **It is the default**: the fp32 loss is invisible and the other
  routes cost decode time.
- `.conv3d` is exact (implicit GEMM). Use it for parity lanes.
- `.fp32Winograd` upcasts bf16 inputs to fp32 for the Winograd kernel. It is the bf16 middle
  ground.

Measurements, on the M5 Max with mlx-swift 0.31.6:

| Decode, compared against | Raw conv2d (Winograd) | conv3d route |
|---|---|---|
| Lens 512² golden (real T2I latent, PyTorch fp32 CPU), GPU fp32 | 1.9e-3 · 65.3 dB · max 3.4e-2 | 4.0e-4 · 78.8 dB · max 3.4e-3 |
| Same golden, GPU **bf16** (ERNIE's production decode) | 1.26e-2 · **48.7 dB** · max 0.17 | 4.7e-3 · 57.3 dB · max 0.036 |
| Same golden, GPU bf16, `.fp32Winograd` | — | **5.5e-3 · 56.0 dB** · max 0.037 |
| 1024² decode time, fp32 (isolated, median of 3 rounds) | 576–583 ms | `.conv3d` +541…750 ms |
| 1024² decode time, bf16 | 431–454 ms | `.conv3d` +677…707 ms · `.fp32Winograd` **+186 ms** |

- The CPU lane reproduces the golden to 3.5e-6.
- The ~4e-4 that remains with the route is TF32 in the mid-block attention (fp32 `Linear` and SDPA).
  With `MLX_ENABLE_TF32=0` the route is 3.4e-6 from the CPU lane, and raw Winograd is 3.6e-6.
- The fp32 loss (Klein, Lens) is below 8-bit visibility: at most 4 levels on [0, 255].
- The bf16 loss (ERNIE) sits under the 8-bit floor of about 59 dB. `.fp32Winograd` recovers 7.3 of
  the 8.6 dB that `.conv3d` does (max error 22 → 5 levels), at about a quarter of the cost. It is
  the candidate for ERNIE; adoption is a per-product decision.
- `FLUX2VAE_CONV_ROUTE=winograd|conv3d|fp32Winograd` overrides the default.

Tests:

- `swift test --filter WinogradProbeTests` is weight-free. It is the removal signal: when raw conv2d
  reports exact on a new pin, remove the route.
- `FLUX2VAE_PARITY=1 swift test -c release -Xswiftc -enable-testing --filter VAEGPULaneTests` needs
  the golden, a lossless safetensors conversion of `lens-mlx/goldens/lens_goldens.npz`.
