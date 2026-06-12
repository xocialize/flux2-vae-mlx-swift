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
