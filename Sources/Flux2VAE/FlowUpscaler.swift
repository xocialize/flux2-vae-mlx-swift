// FlowUpscaler — a ×2 latent upscaler in FLUX.2's latent space, Swift/MLX port.
//
// TensorForger/FlowUpscaler (Unlicense, weights and code): a 59M-parameter attention-free U-Net of SDXL-style
// ResNet blocks (GroupNorm, SiLU, 3×3 convs, a time embedding as scale/shift), whose low-resolution latents enter
// through a small encoder and FiLM. One rectified-flow step from pure noise at t = 1000:
//
//     x = noise − v(noise, t = 1000, latents_small)
//
// on bn-normalised, UNpatchified FLUX.2 latents (32 channels at 1/8). Distilled from FLUX.2 klein-4B generations at
// 512 → 1024; passes chain (×4 = two passes). Reference: its `upscaler_unet.py` and the ComfyUI node's inference
// (`TensorForger/comfyui-flow-upscaler`, `nodes.py`). Evaluated for Gaea in AB-R-0795; ported for AB-A-0257.
//
// Layout: the reference is NCHW; this port runs NHWC inside the U-Net (MLX's conv layout) and keeps the reference's
// module tree and key names, so the checkpoint loads with three renames (`Sequential` indices) and the PT→MLX conv
// transpose. Every 3×3 stride-1 conv here is 384 or 768 channels, inside mlx's lossy Metal Winograd window
// (AB-R-0340), so they take the exact route by default (`WinogradFreeConv2d`, MLXExactConv). GroupNorm runs in fp32
// whatever the compute dtype.

import Foundation
import MLX
import MLXNN
import MLXRandom

// MARK: - Blocks

/// PyTorch `make_group_norm`: the largest group count ≤ 32 that divides the channels; eps 1e-6.
private func flowGroupNorm(_ channels: Int) -> GroupNorm {
    var groups = min(32, channels)
    while channels % groups != 0 && groups > 1 { groups -= 1 }
    return GroupNorm(groupCount: groups, dimensions: channels, eps: 1e-6, pytorchCompatible: true)
}

/// GroupNorm in fp32, result back in the activation dtype.
private func fp32Norm(_ norm: GroupNorm, _ x: MLXArray) -> MLXArray {
    norm(x.asType(.float32)).asType(x.dtype)
}

/// `ConditioningEncoder`: sinusoidal embedding of the timestep (dim 512, period 10000, [sin, cos]) → Linear, SiLU,
/// Linear (1024).
final class FlowConditioning: Module {
    let timeDim: Int
    @ModuleInfo(key: "time_proj_in") var projIn: Linear
    @ModuleInfo(key: "time_proj_out") var projOut: Linear

    init(timeDim: Int = 512, condDim: Int = 1024) {
        self.timeDim = timeDim
        self._projIn.wrappedValue = Linear(timeDim, condDim)
        self._projOut.wrappedValue = Linear(condDim, condDim)
        super.init()
    }

    /// timesteps: [B] fp32 → [B, condDim] fp32.
    func callAsFunction(_ timesteps: MLXArray) -> MLXArray {
        let half = timeDim / 2
        let freqs = exp(-log(Float(10000)) * MLXArray(0 ..< half).asType(.float32) / Float(half))
        let args = timesteps.asType(.float32)[0..., .newAxis] * freqs[.newAxis, 0...]
        let emb = concatenated([sin(args), cos(args)], axis: -1)
        return projOut(silu(projIn(emb)))
    }
}

/// `ConditionedResidualBlock` (SDXL-style): GN → SiLU → conv; GN; ×(1 + scale) + shift from the time vector; SiLU →
/// conv; + skip (a bias-free 1×1 conv when the widths differ). NHWC.
final class FlowResBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: GroupNorm
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "cond_proj") var condProj: Linear
    @ModuleInfo(key: "norm2") var norm2: GroupNorm
    @ModuleInfo(key: "conv2") var conv2: Conv2d
    @ModuleInfo(key: "skip") var skip: Conv2d?

    init(inChannels: Int, outChannels: Int, condDim: Int = 1024) {
        self._norm1.wrappedValue = flowGroupNorm(inChannels)
        self._conv1.wrappedValue = WinogradFreeConv2d(
            inputChannels: inChannels, outputChannels: outChannels, kernelSize: 3, padding: 1)
        self._condProj.wrappedValue = Linear(condDim, 2 * outChannels)
        self._norm2.wrappedValue = flowGroupNorm(outChannels)
        self._conv2.wrappedValue = WinogradFreeConv2d(
            inputChannels: outChannels, outputChannels: outChannels, kernelSize: 3, padding: 1)
        self._skip.wrappedValue = inChannels != outChannels
            ? Conv2d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: 1, bias: false) : nil
        super.init()
    }

    /// x: [B, H, W, C]; cond: [B or 1, condDim].
    func callAsFunction(_ x: MLXArray, cond: MLXArray) -> MLXArray {
        let residual = skip.map { $0(x) } ?? x
        var h = silu(fp32Norm(norm1, x))
        h = conv1(h)
        let scaleShift = condProj(silu(cond)).asType(h.dtype)
        let parts = split(scaleShift, parts: 2, axis: -1)
        let (scale, shift) = (parts[0][0..., .newAxis, .newAxis, 0...], parts[1][0..., .newAxis, .newAxis, 0...])
        h = fp32Norm(norm2, h)
        h = h * (1 + scale) + shift
        h = conv2(silu(h))
        return h + residual
    }
}

/// `DownStage`: residual blocks and a 3×3 stride-2 conv, before the blocks (`downsampleFirst`, the low-res encoder)
/// or after them (the U-Net, which keeps the pre-downsample tensor as its skip).
final class FlowDownStage: Module {
    let downsampleFirst: Bool
    @ModuleInfo(key: "blocks") var blocks: [FlowResBlock]
    @ModuleInfo(key: "downsample") var downsample: Conv2d

    init(inChannels: Int, outChannels: Int, numBlocks: Int, downsampleFirst: Bool) {
        self.downsampleFirst = downsampleFirst
        self._blocks.wrappedValue = (0 ..< numBlocks).map { i in
            FlowResBlock(inChannels: i == 0 ? inChannels : outChannels, outChannels: outChannels)
        }
        // Stride 2 is outside the Winograd window: plain conv2d is mlx's exact implicit GEMM.
        self._downsample.wrappedValue = Conv2d(
            inputChannels: outChannels, outputChannels: outChannels, kernelSize: 3, stride: 2, padding: 1)
        super.init()
    }

    func callAsFunction(_ input: MLXArray, cond: MLXArray) -> (x: MLXArray, skip: MLXArray) {
        var x = downsampleFirst ? downsample(input) : input
        for block in blocks { x = block(x, cond: cond) }
        let skip = x
        if !downsampleFirst { x = downsample(x) }
        return (x, skip)
    }
}

/// `UpStage`: bilinear ×2 (to the skip's size), concat the skip, residual blocks.
final class FlowUpStage: Module {
    @ModuleInfo(key: "blocks") var blocks: [FlowResBlock]

    init(inChannels: Int, skipChannels: Int, outChannels: Int, numBlocks: Int) {
        self._blocks.wrappedValue = (0 ..< numBlocks).map { i in
            FlowResBlock(inChannels: i == 0 ? inChannels + skipChannels : outChannels, outChannels: outChannels)
        }
        super.init()
    }

    func callAsFunction(_ input: MLXArray, skip: MLXArray, cond: MLXArray) -> MLXArray {
        // nn.Upsample(scale 2) then, if the sizes differ, F.interpolate to the skip's: both bilinear with
        // align_corners False. Resizing straight to 2× and then to the skip's size is what the reference does.
        var x = FlowUpscalerUNet.bilinear(input, height: input.dim(1) * 2, width: input.dim(2) * 2)
        if x.dim(1) != skip.dim(1) || x.dim(2) != skip.dim(2) {
            x = FlowUpscalerUNet.bilinear(x, height: skip.dim(1), width: skip.dim(2))
        }
        x = concatenated([x, skip], axis: -1)
        for block in blocks { x = block(x, cond: cond) }
        return x
    }
}

/// `LowResEncoder`: 1×1 in-conv, one block, then two downsample-first stages: features at 1, 1/2 and 1/4 of the
/// low-resolution latents.
final class FlowLowResEncoder: Module {
    @ModuleInfo(key: "in_conv") var inConv: Conv2d
    @ModuleInfo(key: "block_1") var block1: FlowResBlock
    @ModuleInfo(key: "block_2") var block2: FlowDownStage
    @ModuleInfo(key: "block_3") var block3: FlowDownStage

    init(sampleChannels: Int = 32, baseChannels: Int = 384) {
        self._inConv.wrappedValue = Conv2d(inputChannels: sampleChannels, outputChannels: baseChannels, kernelSize: 1)
        self._block1.wrappedValue = FlowResBlock(inChannels: baseChannels, outChannels: baseChannels)
        self._block2.wrappedValue = FlowDownStage(
            inChannels: baseChannels, outChannels: baseChannels, numBlocks: 1, downsampleFirst: true)
        self._block3.wrappedValue = FlowDownStage(
            inChannels: baseChannels, outChannels: baseChannels, numBlocks: 1, downsampleFirst: true)
        super.init()
    }

    func callAsFunction(_ small: MLXArray, cond: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        let b1 = block1(inConv(small), cond: cond)
        let b2 = block2(b1, cond: cond).x
        let b3 = block3(b2, cond: cond).x
        return (b1, b2, b3)
    }
}

/// `FilmCond2D`: x · (1 + scale) + shift, scale/shift from SiLU → 1×1 conv of the low-res features.
final class FlowFilm: Module {
    @ModuleInfo(key: "cond_proj") var condProj: Conv2d

    init(baseChannels: Int = 384, condChannels: Int = 384) {
        self._condProj.wrappedValue = Conv2d(
            inputChannels: condChannels, outputChannels: baseChannels * 2, kernelSize: 1)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cond: MLXArray) -> MLXArray {
        let parts = split(condProj(silu(cond)), parts: 2, axis: -1)
        return x * (1 + parts[0]) + parts[1]
    }
}

// MARK: - The U-Net

/// `UpscalerUNet` (sample 32 ch, base 384, time 512, cond 1024): the velocity v(sample, t, latents_small).
public final class FlowUpscalerUNet: Module {
    public static let sampleChannels = 32

    @ModuleInfo(key: "conditioning") var conditioning: FlowConditioning
    @ModuleInfo(key: "in_conv") var inConv: Conv2d
    @ModuleInfo(key: "low_res_encoder") var lowResEncoder: FlowLowResEncoder
    @ModuleInfo(key: "film_cond_1") var film1: FlowFilm
    @ModuleInfo(key: "film_cond_2") var film2: FlowFilm
    @ModuleInfo(key: "film_cond_3") var film3: FlowFilm
    @ModuleInfo(key: "down_stages") var downStages: [FlowDownStage]
    @ModuleInfo(key: "mid_stages") var midStages: [FlowResBlock]
    @ModuleInfo(key: "up_stages") var upStages: [FlowUpStage]
    @ModuleInfo(key: "out_conv") var outConv: Conv2d

    public init(sampleChannels: Int = 32, baseChannels: Int = 384) {
        let c = baseChannels
        self._conditioning.wrappedValue = FlowConditioning()
        self._inConv.wrappedValue = Conv2d(inputChannels: sampleChannels, outputChannels: c, kernelSize: 1)
        self._lowResEncoder.wrappedValue = FlowLowResEncoder(sampleChannels: sampleChannels, baseChannels: c)
        self._film1.wrappedValue = FlowFilm(baseChannels: c, condChannels: c)
        self._film2.wrappedValue = FlowFilm(baseChannels: c, condChannels: c)
        self._film3.wrappedValue = FlowFilm(baseChannels: c, condChannels: c)
        self._downStages.wrappedValue = [
            FlowDownStage(inChannels: c, outChannels: c, numBlocks: 3, downsampleFirst: false),
            FlowDownStage(inChannels: c, outChannels: c, numBlocks: 2, downsampleFirst: false),
        ]
        self._midStages.wrappedValue = [FlowResBlock(inChannels: c, outChannels: c)]
        self._upStages.wrappedValue = [
            FlowUpStage(inChannels: c, skipChannels: c, outChannels: c, numBlocks: 2),
            FlowUpStage(inChannels: c, skipChannels: c, outChannels: c, numBlocks: 3),
        ]
        self._outConv.wrappedValue = Conv2d(inputChannels: c, outputChannels: sampleChannels, kernelSize: 1)
        super.init()
    }

    /// Route for the in-window 3×3 convs. Default exact (`.conv3d`); `.winograd` is mlx's fast lossy kernel and
    /// `.fp32Winograd` its bf16 middle ground (WinogradFreeConv2d.swift).
    public var convRoute: Flux2VAEConvRoute {
        get { modules().lazy.compactMap { ($0 as? WinogradFreeConv2d)?.route }.first ?? .conv3d }
        set { for case let conv as WinogradFreeConv2d in modules() { conv.route = newValue } }
    }

    /// v(sample, t, latents_small). NHWC: sample [B, H, W, 32], latentsSmall [B, h, w, 32] (h ≈ H/2), timestep [1] or
    /// [B]. Runs in the inputs' dtype (GroupNorm in fp32); returns [B, H, W, 32].
    public func callAsFunction(_ sample: MLXArray, timestep: MLXArray, latentsSmall: MLXArray) -> MLXArray {
        let dtype = sample.dtype
        let cond = conditioning(timestep).asType(dtype)
        let (H, W) = (sample.dim(1), sample.dim(2))

        var (lr1, lr2, lr3) = lowResEncoder(latentsSmall, cond: cond)
        // F.interpolate(mode="bilinear"), align_corners None = False. H // 2 and H // 4 as the reference floors them.
        lr1 = Self.bilinear(lr1, height: H, width: W)
        lr2 = Self.bilinear(lr2, height: H / 2, width: W / 2)
        lr3 = Self.bilinear(lr3, height: H / 4, width: W / 4)

        var x = film1(inConv(sample), cond: lr1)
        var skips: [MLXArray] = []
        var stage = downStages[0](x, cond: cond)
        skips.append(stage.skip)
        x = film2(stage.x, cond: lr2)
        stage = downStages[1](x, cond: cond)
        skips.append(stage.skip)
        x = film3(stage.x, cond: lr3)
        for mid in midStages { x = mid(x, cond: cond) }
        for up in upStages { x = up(x, skip: skips.removeLast(), cond: cond) }
        return outConv(x)
    }

    /// PyTorch bilinear resize, align_corners False (half-pixel centres, source clamped at 0), antialias off, on an
    /// NHWC tensor to an explicit size. Separable: gather two rows and blend, then two columns.
    static func bilinear(_ x: MLXArray, height: Int, width: Int) -> MLXArray {
        func axisTaps(inSize: Int, outSize: Int) -> (MLXArray, MLXArray, MLXArray) {
            let scale = Double(inSize) / Double(outSize)
            var i0 = [Int32](), i1 = [Int32](), lam = [Float]()
            for d in 0 ..< outSize {
                let src = max(scale * (Double(d) + 0.5) - 0.5, 0)
                let lo = min(Int(src), inSize - 1)
                i0.append(Int32(lo))
                i1.append(Int32(min(lo + 1, inSize - 1)))
                lam.append(Float(src - Double(lo)))
            }
            return (MLXArray(i0), MLXArray(i1), MLXArray(lam))
        }
        var y = x
        if y.dim(1) != height {
            let (a, b, l) = axisTaps(inSize: y.dim(1), outSize: height)
            let w = l.reshaped(1, height, 1, 1).asType(y.dtype)
            y = take(y, a, axis: 1) * (1 - w) + take(y, b, axis: 1) * w
        }
        if y.dim(2) != width {
            let (a, b, l) = axisTaps(inSize: y.dim(2), outSize: width)
            let w = l.reshaped(1, 1, width, 1).asType(y.dtype)
            y = take(y, a, axis: 2) * (1 - w) + take(y, b, axis: 2) * w
        }
        return y
    }
}

// MARK: - The upscaler

/// A ×2 upscale of FLUX.2 latents in one rectified-flow step, chained for ×4.
///
/// ```swift
/// let up = try FlowUpscaler.load(url: weightsURL)                       // bf16 weights, fp32 compute
/// let packed2x = up.upscalePacked(packed, seed: 7)                      // [B,128,h,w] → [B,128,2h,2w]
/// let image = vae.decodePackedLatents(packed2x, tile: .standard)       // the row-band decode
/// ```
///
/// Packed latents are the form the FLUX.2 / FLUX 3 DiTs produce and `Flux2VAE.decodePackedLatents` takes: 2×2
/// patchified (128 channels at 1/16) and bn-normalised. The model works on the same normalised latents unpatchified,
/// so no bn statistics are needed here.
public final class FlowUpscaler {
    public let unet: FlowUpscalerUNet
    /// Compute dtype of the U-Net (its weights are held in it). GroupNorm always runs in fp32.
    public let dtype: DType
    /// The timestep of the one step: the flow starts from pure noise.
    public static let timestep: Float = 1000

    public init(unet: FlowUpscalerUNet, dtype: DType) {
        self.unet = unet
        self.dtype = dtype
    }

    /// Load `flow_upscaler.safetensors` (the original fp32 file or the bf16 conversion) at `dtype`: a two-way strict
    /// load of all 166 tensors.
    ///
    /// Default fp32 compute. Against the PyTorch fp32 reference on a 2048² decode (AB-A-0257): the bf16 weights in
    /// fp32 compute are 67.7 dB, bf16 compute 60.3 dB, for about 10 % more time (334 vs 294 ms to 2048²,
    /// 1.37 vs 1.27 s to 4096²) and a pass peak (2.5 / 5.7 GB) still far under the decode's.
    public static func load(url: URL, dtype: DType = .float32) throws -> FlowUpscaler {
        let unet = FlowUpscalerUNet()
        var state: [String: MLXArray] = [:]
        for (rawKey, rawValue) in try MLX.loadArrays(url: url) {
            let key = rawKey
                .replacingOccurrences(of: "conditioning.time_proj.0.", with: "conditioning.time_proj_in.")
                .replacingOccurrences(of: "conditioning.time_proj.2.", with: "conditioning.time_proj_out.")
                .replacingOccurrences(of: ".cond_proj.1.", with: ".cond_proj.")
            var v = rawValue
            if v.ndim == 4 { v = v.transposed(0, 2, 3, 1) }  // PT (O,I,kH,kW) → MLX (O,kH,kW,I)
            state[key] = v.asType(dtype)
        }
        try Flux2VAEWeights.verifyAndLoad(model: unet, weights: state, label: "FlowUpscaler")
        return FlowUpscaler(unet: unet, dtype: dtype)
    }

    /// One ×2 pass on unpatchified, bn-normalised latents (NCHW [B, 32, h, w]) with the given noise (NCHW
    /// [B, 32, 2h, 2w]): `noise − v(noise, 1000, latents)`. Returns fp32 NCHW [B, 32, 2h, 2w].
    public func upscale(latents: MLXArray, noise: MLXArray) -> MLXArray {
        precondition(latents.dim(1) == FlowUpscalerUNet.sampleChannels && noise.dim(1) == latents.dim(1))
        let n = noise.transposed(0, 2, 3, 1).asType(dtype)
        let small = latents.transposed(0, 2, 3, 1).asType(dtype)
        let t = MLXArray([Self.timestep])
        let v = unet(n, timestep: t, latentsSmall: small)
        return (n.asType(.float32) - v.asType(.float32)).transposed(0, 3, 1, 2)
    }

    /// One ×2 pass on packed latents [B, 128, h, w] → [B, 128, 2h, 2w] (fp32), the noise drawn from `seed`.
    public func upscalePacked(_ packed: MLXArray, seed: UInt64) -> MLXArray {
        let latents = Self.unpatchify(packed)
        let noise = MLXRandom.normal(
            [latents.dim(0), latents.dim(1), latents.dim(2) * 2, latents.dim(3) * 2], dtype: .float32,
            key: MLXRandom.key(seed))
        let out = Self.patchify(upscale(latents: latents, noise: noise))
        eval(out)
        return out
    }

    /// `passes` chained ×2 passes (pass i draws its noise from `seed + i`), each evaluated before the next.
    public func upscalePacked(_ packed: MLXArray, passes: Int, seed: UInt64) -> MLXArray {
        var x = packed
        for i in 0 ..< passes {
            x = upscalePacked(x, seed: seed &+ UInt64(i))
            MLX.Memory.clearCache()
        }
        return x
    }

    /// [B, C, H, W] → [B, 4C, H/2, W/2], diffusers' patch order (the inverse of `unpatchify`).
    public static func patchify(_ x: MLXArray) -> MLXArray {
        let (b, c, h, w) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        return x.reshaped(b, c, h / 2, 2, w / 2, 2).transposed(0, 1, 3, 5, 2, 4).reshaped(b, c * 4, h / 2, w / 2)
    }

    /// [B, 4c, h, w] → [B, c, 2h, 2w], diffusers' patch order.
    public static func unpatchify(_ x: MLXArray) -> MLXArray {
        Flux2VAE.unpatchifyLatents(x)
    }
}
