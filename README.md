# flux2-vae-mlx-swift

The **FLUX.2 VAE** (decoder, and since 0.5.0 the encoder) as a neutral, standalone Swift/MLX package.

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
// Memory-bounded, same image (0.6.0, opt-in): 1024² fp32 peak 10.7 → 2.4 GB, 2048² 42 → 6.9 GB.
let image3 = vae.decodePackedLatents(latents, tile: .standard)

// Encoder (0.5.0), and its memory-bounded form (0.7.0, opt-in): 2048² fp32 peak 22.7 → 4.6 GB.
let encoder = try Flux2VAEWeights.loadEncoder(directory: vaeDir, dtype: .float32)
let packed = encoder.encodePacked(pixels, tile: .standard)   // pixels: (B, 3, H, W) in [-1, 1]
```

**Encoder (0.5.0).** `Flux2VAEWeights.loadEncoder(directory:)` loads the encoder tower, `quant_conv` and the `bn`
stats from the same `vae/` snapshot. It came here from flux2-klein-swift's `KleinVAEEncoder`, so every FLUX.2-VAE
consumer shares one encoder.
- `encodeMean(_:)` returns the deterministic latent mean.
- `encodePacked(_:)` returns the 2×2-patchified, bn-normalised latents the FLUX.2 and FLUX 3 DiTs consume (the
  inverse of `decodePackedLatents`).
- The in-window convs run exact by default.
- Gated against BFL's FLUX 3 klein reference `AutoEncoder.encode` (its VAE is this VAE, identical weights): reference
  tokens cos 0.999999 (GPU fp32).

**Small decoder.** `loadVAE` reads the decoder widths from the snapshot's `config.json`
(`decoder_block_out_channels`, else `block_out_channels`), so BFL's
[FLUX.2-small-decoder](https://huggingface.co/black-forest-labs/FLUX.2-small-decoder) (Apache-2.0,
widths [96, 192, 384, 384], ~28M vs ~50M decoder params) loads as-is — as shipped inside the MOSS-Image
snapshots. `Flux2VAE(decoderBlockOutChannels:)` builds either; the default is the full decoder.
Gated against diffusers `AutoencoderKLFlux2` fp32 (moss-image oracle): relL2 1.8e-4 on the exact conv
route, 3.5e-3 on the default route.

Depends only on [mlx-swift](https://github.com/ml-explore/mlx-swift) — no engine, no tokenizers,
no model dependency. MIT.

## Memory-bounded decode (0.6.0)

`decodePackedLatents(_:tile:)` and `decode(_:tile:)` are an opt-in decode in row bands that returns the same image
as the untiled decode. Nothing changes for callers that don't pass `tile:`.

```swift
let image = vae.decodePackedLatents(latents, tile: .standard)   // 64 MB bands
let small = vae.decodePackedLatents(latents, tile: Flux2VAETile(bandBytes: 16 << 20))
```

**Where the untiled peak comes from.** It is temporaries, not activations. At 1024² the widest tensor is 1 GB (256
channels at full resolution, fp32). Around it:
- Each GroupNorm materialises about five copies: a transpose, the layerNorm, the transpose back and the affine.
- The exact conv route copies its input with an extra channel.
- The mid-block attention misses MLX's fused SDPA (head dim 512), so the N×N score matrix is materialised: 1 GB at
  1024², 17 GB at 2048².

**How it works.**
- Every activation lives as NHWC row bands, never as one tensor plus its temporaries.
- GroupNorm stats come from all bands (a pairwise mean/M2 combine), then each band is normalised.
- Each 3×3 conv runs per band, with one real neighbour row on each side as its halo. Zero rows appear only at the
  image edge, so every band's output is exactly the untiled conv's rows.
- The attention keeps whole-image K and V, and chunks the queries by rows.
- An input band is released once no later band needs it.

qwen-image21-swift's recipe (a global prefix, then multi-layer tiles cropped at the receptive field) does not carry
over. Every resnet here opens with a GroupNorm whose stats span the whole image, so a tile normalised on its own
stats drifts in colour. The floor is one resnet's full-resolution input plus its middle tensor, 6.7–6.9 GB fp32 at
2048² measured.

**Measured** on the M5 Max with mlx-swift 0.32.2:
- Setup: the FLUX 3 klein 512² oracle latent (`latents.15`) tiled up to the size, decoder alone, one cold run per
  process.
- `Memory.peakMemory` is read with `Memory.cacheLimit` = 2 GB (the engine's .automatic, AB-L-0192). 0.19 GB of it
  is resident weights.
- Bands are 64 MB.

| Decode | 1024² untiled → tiled | 2048² untiled → tiled |
|---|---|---|
| fp32, `.winograd` (default route) | 10.68 → **2.37 GB** (940 vs 1349 ms) | 42.10 → **6.89 GB** (4.1 vs 6.4 s) |
| fp32, `.conv3d` (exact) | 7.83 → 2.27 GB | 30.77 → 6.79 GB |
| bf16 weights + latents, `.fp32Winograd` | 9.54 → 1.55 GB | 37.85 → 3.82 GB |
| band size, fp32 `.winograd`: 16 / 64 / 256 MB | 1.87 / 2.37 / 4.35 GB | 6.39 / 6.89 / 8.87 GB |

**Parity**, tiled vs untiled (relL2 · PSNR over [−1, 1]):
- fp32 `.conv3d`: 4.0e-6 · 118.6 dB at 1024², 4.4e-6 · 117.7 dB at 2048².
- The 512² oracle decode is 85.9 dB from BFL's reference `decoded` both ways, with bands of 64, 4 and 1 MB.
- On `.winograd` the tiled and untiled decodes differ by about 1.5–1.9e-3. That gap is Winograd's own noise, not
  tiling: both sit at 65.08 dB (1024²) and 65.04 dB (2048²) from the exact untiled decode.
- On bf16 the same holds: both are 56.7–56.8 dB from fp32.

**No seams.** The per-row error profile shows no band edges. At the final stage's edges (every 128 rows at 1024²,
every 64 at 2048²) the worst row is below the worst row elsewhere, on every route.

`swift test --filter TiledDecodeTests` is weight-free. It decodes a random narrow decoder untiled and in bands down to
one row, on the CPU and the GPU. It sits at 3–5e-6 on the CPU, and a broken halo reads ~1.1.
`Flux2VAEBench` (`decode` / `compare` / `golden`) reproduces the tables.

## Memory-bounded encode (0.7.0)

`encodePacked(_:tile:)` and `encodeMean(_:tile:)` on `Flux2VAEEncoder` are the same row-band recipe applied to the
encoder, with the same `Flux2VAETile`. They return the same latents as the untiled encode. Nothing changes for callers
that don't pass `tile:`.

```swift
let packed = encoder.encodePacked(pixels, tile: .standard)   // 64 MB bands
```

**Where the untiled peak comes from.** The mid-block attention runs at H/8 with 512 channels. At 2048² that is 65 536
tokens, and the reference's explicit softmax(QKᵀ)V materialises a 17 GB score matrix. The full-resolution stage comes
next: 128 channels, 2 GB per tensor at 2048², plus its GroupNorm temporaries.

**What is new over the decode.** The encoder adds the stride-2 downsample: it pads the bottom and right by one row and
column, then runs a valid 3×3 stride-2 conv. Output row i reads input rows 2i, 2i+1 and 2i+2. So:
- an output band o0..<o1 reads input rows 2·o0..<2·o1+1;
- the zero pad row appears only when 2·o1 reaches the bottom edge;
- input rows below 2·o1 are released once that band is out.

Everything else is the decode's recipe: whole-tensor GroupNorm stats, a one-row halo per stride-1 3×3 conv, and
query-chunked attention against whole-image K/V. The attention keeps the reference's explicit softmax. Softmax is per
query row, so chunking the queries changes nothing. The floor is one full-resolution resnet's input plus its middle
tensor, about 4 GB fp32 at 2048².

**Measured** on the M5 Max with mlx-swift 0.32.2:
- Setup: a Wikimedia Commons photo, centre-cropped and Lanczos-resized to each size. Encoder alone, fp32, on the
  encoder's default `.conv3d` (exact) route. One cold run per process, behind the GPU contention gate.
- `Memory.peakMemory` is read with `Memory.cacheLimit` = 2 GB (AB-L-0192). 0.13–0.18 GB of it is resident weights
  and the input.
- Bands are 64 MB.

| Encode, fp32 `.conv3d` | untiled | tiled | time, untiled → tiled |
|---|---|---|---|
| 512² | 2.21 GB | **0.83 GB** | 162 → 238 ms |
| 1024² | 5.78 GB | **1.59 GB** | 620 → 805 ms |
| 2048² | 22.72 GB | **4.63 GB** | 3.21 → 3.50 s |

**Parity**, tiled vs untiled packed latents (relL2 · max · PSNR over [−1, 1]):
- 512²: 1.4e-5 · 4.0e-4 · 103.7 dB.
- 1024²: 1.1e-5 · 1.6e-4 · 106.1 dB.
- 2048²: 9.4e-6 · 1.4e-4 · 107.1 dB.
- The ~1e-5 is TF32 on differently shaped GEMMs, not tiling. With `MLX_ENABLE_TF32=0`, 1024² tiled vs untiled is
  2.3e-6 · 119.4 dB. TF32 alone moves the untiled encode by 2.2e-3, about 200× more.

**No seams.** The worst latent row is at most 1.6× the median row, and the worst column is at most 1.4× the median.

`swift test --filter TiledEncodeTests` is weight-free. It encodes with a random narrow encoder, untiled and in bands
down to one row, on the CPU and the GPU. Its 100-row case gives the last downsample an odd-height input.
`Flux2VAEBench encode` + `compare` reproduces the table.

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
