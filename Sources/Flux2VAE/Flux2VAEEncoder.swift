// FLUX.2 VAE encoder — the standard diffusers AutoencoderKL Encoder with 32 latent channels + quant_conv, and the
// packed (2×2-patchified, bn-normalised) latents the FLUX.2 and FLUX 3 DiTs consume. Block structure from
// flux2-klein-swift's VAEEncoder (itself from z-image-swift's FLUX.1 AE encoder). Returns the deterministic MEAN.
//
// Every stride-1 3×3 conv is a WinogradFreeConv2d routed `.conv3d` (exact) by default — see WinogradFreeConv2d.swift.

import Foundation
import MLX
import MLXNN

private func encSilu(_ x: MLXArray) -> MLXArray { x * sigmoid(x) }
private func encGroupNorm(_ groups: Int, _ channels: Int, _ eps: Float) -> GroupNorm {
    GroupNorm(groupCount: groups, dimensions: channels, eps: eps, pytorchCompatible: true)
}

private final class EncResnetBlock2D: Module {
    @ModuleInfo var norm1: GroupNorm
    @ModuleInfo var conv1: Conv2d
    @ModuleInfo var norm2: GroupNorm
    @ModuleInfo var conv2: Conv2d
    @ModuleInfo(key: "conv_shortcut") var convShortcut: Conv2d?

    init(_ inC: Int, _ outC: Int, groups: Int = 32, eps: Float = 1e-6) {
        self._norm1.wrappedValue = encGroupNorm(groups, inC, eps)
        self._conv1.wrappedValue = WinogradFreeConv2d(inputChannels: inC, outputChannels: outC, kernelSize: 3, stride: 1, padding: 1)
        self._norm2.wrappedValue = encGroupNorm(groups, outC, eps)
        self._conv2.wrappedValue = WinogradFreeConv2d(inputChannels: outC, outputChannels: outC, kernelSize: 3, stride: 1, padding: 1)
        if inC != outC {
            self._convShortcut.wrappedValue = Conv2d(inputChannels: inC, outputChannels: outC, kernelSize: 1, stride: 1, padding: 0)
        }
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var residual = x
        var h = conv1(encSilu(norm1(x)))
        h = conv2(encSilu(norm2(h)))
        if let convShortcut { residual = convShortcut(residual) }
        return residual + h
    }
}

private final class EncDownsample2D: Module {
    @ModuleInfo var conv: Conv2d
    init(_ channels: Int) {
        self._conv.wrappedValue = Conv2d(inputChannels: channels, outputChannels: channels, kernelSize: 3, stride: 2, padding: 0)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let padded = MLX.padded(x, widths: [.init((0, 0)), .init((0, 1)), .init((0, 1)), .init((0, 0))])
        return conv(padded)
    }
}

private final class EncVAEAttention: Module {
    @ModuleInfo(key: "group_norm") var groupNorm_: GroupNorm
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: [Linear]
    let scale: Float
    init(_ channels: Int, groups: Int = 32, eps: Float = 1e-6) {
        self._groupNorm_.wrappedValue = encGroupNorm(groups, channels, eps)
        self._toQ.wrappedValue = Linear(channels, channels)
        self._toK.wrappedValue = Linear(channels, channels)
        self._toV.wrappedValue = Linear(channels, channels)
        self._toOut.wrappedValue = [Linear(channels, channels)]
        self.scale = 1.0 / Float(channels).squareRoot()
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let residual = x
        var y = groupNorm_(x).reshaped(b, h * w, c)
        let q = toQ(y), k = toK(y), v = toV(y)
        let attn = softmax(matmul(q, k.transposed(0, 2, 1)) * scale, axis: -1)
        y = toOut[0](matmul(attn, v)).reshaped(b, h, w, c)
        return residual + y
    }
}

private final class EncMidBlock2D: Module {
    @ModuleInfo var resnets: [EncResnetBlock2D]
    @ModuleInfo var attentions: [EncVAEAttention]
    init(_ channels: Int, groups: Int = 32, eps: Float = 1e-6) {
        self._resnets.wrappedValue = [EncResnetBlock2D(channels, channels, groups: groups, eps: eps),
                                      EncResnetBlock2D(channels, channels, groups: groups, eps: eps)]
        self._attentions.wrappedValue = [EncVAEAttention(channels, groups: groups, eps: eps)]
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { resnets[1](attentions[0](resnets[0](x))) }
}

private final class EncDownBlock2D: Module {
    @ModuleInfo var resnets: [EncResnetBlock2D]
    @ModuleInfo var downsamplers: [EncDownsample2D]?
    init(_ inC: Int, _ outC: Int, numLayers: Int, addDownsample: Bool, groups: Int = 32, eps: Float = 1e-6) {
        self._resnets.wrappedValue = (0..<numLayers).map { i in EncResnetBlock2D(i == 0 ? inC : outC, outC, groups: groups, eps: eps) }
        self._downsamplers.wrappedValue = addDownsample ? [EncDownsample2D(outC)] : nil
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for r in resnets { h = r(h) }
        if let downsamplers { h = downsamplers[0](h) }
        return h
    }
}

private final class Flux2EncoderTower: Module {
    @ModuleInfo(key: "conv_in") var convIn: Conv2d
    @ModuleInfo(key: "down_blocks") var downBlocks: [EncDownBlock2D]
    @ModuleInfo(key: "mid_block") var midBlock: EncMidBlock2D
    @ModuleInfo(key: "conv_norm_out") var convNormOut: GroupNorm
    @ModuleInfo(key: "conv_out") var convOut: Conv2d

    init(inC: Int, latentC: Int, blockOut: [Int], layersPerBlock: Int, groups: Int = 32, eps: Float = 1e-6) {
        self._convIn.wrappedValue = WinogradFreeConv2d(inputChannels: inC, outputChannels: blockOut[0], kernelSize: 3, stride: 1, padding: 1)
        var blocks: [EncDownBlock2D] = []
        var outC = blockOut[0]
        for (i, boc) in blockOut.enumerated() {
            let inC2 = outC; outC = boc
            blocks.append(EncDownBlock2D(inC2, outC, numLayers: layersPerBlock, addDownsample: i != blockOut.count - 1, groups: groups, eps: eps))
        }
        self._downBlocks.wrappedValue = blocks
        self._midBlock.wrappedValue = EncMidBlock2D(blockOut.last!, groups: groups, eps: eps)
        self._convNormOut.wrappedValue = encGroupNorm(groups, blockOut.last!, eps)
        self._convOut.wrappedValue = WinogradFreeConv2d(inputChannels: blockOut.last!, outputChannels: 2 * latentC, kernelSize: 3, stride: 1, padding: 1)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = convIn(x)
        for b in downBlocks { h = b(h) }
        h = midBlock(h)
        return convOut(encSilu(convNormOut(h)))
    }
}

/// FLUX.2 VAE encoder: the diffusers `encoder` tower + `quant_conv`, plus the `bn` latent stats for the packed
/// (normalised, 2×2-patchified) form the FLUX.2 / FLUX 3 DiTs consume. Moved here from flux2-klein-swift
/// (`KleinVAEEncoder`) so every FLUX.2-VAE consumer shares one encoder. Loads from the same `vae/` snapshot as the
/// decoder (`Flux2VAEWeights.loadEncoder`).
public final class Flux2VAEEncoder: Module {
    @ModuleInfo fileprivate var encoder: Flux2EncoderTower
    @ModuleInfo(key: "quant_conv") var quantConv: Conv2d
    @ModuleInfo(key: "bn") var bn: Flux2BatchNormStats
    public let latentChannels: Int

    public init(inChannels: Int = 3, latentChannels: Int = Flux2VAE.latentChannels,
                blockOutChannels: [Int] = Flux2VAE.fullDecoderBlockOutChannels, layersPerBlock: Int = 2,
                normGroups: Int = 32) {
        self.latentChannels = latentChannels
        self._encoder.wrappedValue = Flux2EncoderTower(
            inC: inChannels, latentC: latentChannels, blockOut: blockOutChannels, layersPerBlock: layersPerBlock,
            groups: normGroups)
        self._quantConv.wrappedValue = Conv2d(
            inputChannels: 2 * latentChannels, outputChannels: 2 * latentChannels, kernelSize: 1, stride: 1, padding: 0)
        self._bn.wrappedValue = Flux2BatchNormStats(numFeatures: 4 * latentChannels)
        super.init()
        convRoute = Flux2VAEConvRoute.environmentOverride ?? .conv3d
    }

    /// Route for the in-window 3×3 convs. Default `.conv3d` (exact): the Winograd loss is material for edit
    /// references (3.4e-3 vs 1.2e-3 relL2 in the latent mean, measured in flux2-klein-swift).
    public var convRoute: Flux2VAEConvRoute {
        get { modules().lazy.compactMap { ($0 as? WinogradFreeConv2d)?.route }.first ?? .conv3d }
        set { for case let conv as WinogradFreeConv2d in modules() { conv.route = newValue } }
    }

    /// image: [B, 3, H, W] in [-1, 1] → the deterministic latent MEAN [B, latentC, H/8, W/8] (NCHW).
    public func encodeMean(_ imageNCHW: MLXArray) -> MLXArray {
        let x = imageNCHW.transposed(0, 2, 3, 1)
        var moments = quantConv(encoder(x))
        moments = moments.transposed(0, 3, 1, 2)
        return moments[0..., ..<latentChannels, 0..., 0...]
    }

    /// The packed form: the mean 2×2-patchified to [B, 4·latentC, H/16, W/16] (channel = c·4 + dy·2 + dx, the
    /// inverse of `Flux2VAE.unpatchifyLatents`) and normalised by the `bn` stats, (x − mean) / sqrt(var + eps).
    /// This is BFL's `AutoEncoder.encode` before its token flatten. H and W must be multiples of 16.
    public func encodePacked(_ imageNCHW: MLXArray) -> MLXArray {
        let mean = encodeMean(imageNCHW)
        let (b, c, h, w) = (mean.dim(0), mean.dim(1), mean.dim(2), mean.dim(3))
        let packed = mean.reshaped(b, c, h / 2, 2, w / 2, 2).transposed(0, 1, 3, 5, 2, 4)
            .reshaped(b, 4 * c, h / 2, w / 2)
        let bnMean = bn.runningMean.reshaped(1, -1, 1, 1)
        let bnStd = sqrt(bn.runningVar.reshaped(1, -1, 1, 1) + bn.eps)
        return ((packed.asType(.float32) - bnMean) / bnStd).asType(packed.dtype)
    }
}
