// Memory-bounded decode: row bands with a one-row halo per 3×3 conv, GroupNorm on whole-tensor stats.
//
// The untiled decode's peak is temporaries, not activations. At 1024² the widest tensor is 1 GB (256 ch at full
// resolution, fp32), but each GroupNorm materialises a transpose, its layerNorm, the transpose back and the affine
// (≈ 5× the tensor), the exact conv route copies its input with an extra channel, and the mid-block attention falls
// off MLX's fused SDPA (head dim 512) onto a materialised N×N score matrix: 1 GB at 1024², 17 GB at 2048².
//
// Why not the qwen-image21 recipe (AB-R-0310: global prefix, then multi-layer tiles cropped at the receptive field):
// every resnet here opens with a GroupNorm, whose mean and variance span the whole image. A tile that runs several
// layers on its own stats drifts in colour from its neighbours. So this path streams layer by layer instead:
//
// - Every activation lives as NHWC row bands (`RowBands`), never as one tensor plus its temporaries.
// - GroupNorm stats come from all bands (Chan's parallel mean/M2 combine), then normalise band by band.
// - Each 3×3 conv runs per band on the band plus one real neighbour row on each side, zero rows only at the image
//   edge, so each band's output equals the untiled conv's rows.
// - The mid-block attention keeps whole-image K/V and chunks the queries by image rows.
// - Bands of an input are released as soon as no later band needs them.
//
// The result matches the untiled decode up to float reassociation in the stats and GEMM tiling, with no seams by
// construction. The floor is one resnet's input plus its middle tensor at full resolution: 6.7–6.9 GB fp32 at 2048².

import Foundation
import MLX
import MLXExactConv
import MLXFast
import MLXNN

/// Band size for `Flux2VAE.decode(_:tile:)` / `decodePackedLatents(_:tile:)`.
public struct Flux2VAETile: Sendable, Equatable {
    /// Bytes of one row band of a stage's widest tensor, in fp32: about the size of each temporary a band
    /// allocates (a GroupNorm'd band is ~5 of them). Also caps one query chunk's attention scores.
    public var bandBytes: Int

    public init(bandBytes: Int = 64 << 20) {
        precondition(bandBytes > 0, "bandBytes must be positive")
        self.bandBytes = bandBytes
    }

    public static let standard = Flux2VAETile()

    /// Rows per band for a [batch, ·, width, channels] tensor.
    func rows(batch: Int, width: Int, channels: Int) -> Int {
        max(1, bandBytes / max(1, batch * width * channels * 4))
    }
}

/// An NHWC activation [B, H, W, C] held as consecutive row bands.
final class RowBands {
    private var pieces: [MLXArray?] = []
    private var starts: [Int] = []
    private var ends: [Int] = []
    private(set) var height = 0
    let batch: Int, width: Int, channels: Int
    let dtype: DType

    init(batch: Int, width: Int, channels: Int, dtype: DType) {
        (self.batch, self.width, self.channels, self.dtype) = (batch, width, channels, dtype)
    }

    convenience init(_ x: MLXArray) {
        self.init(batch: x.dim(0), width: x.dim(2), channels: x.dim(3), dtype: x.dtype)
        append(x)
    }

    func append(_ band: MLXArray) {
        starts.append(height)
        pieces.append(band)
        height += band.dim(1)
        ends.append(height)
    }

    /// Rows lo..<hi (clipped to the tensor) as one array.
    func rows(_ lo: Int, _ hi: Int) -> MLXArray {
        let (lo, hi) = (max(lo, 0), min(hi, height))
        var parts: [MLXArray] = []
        for (i, piece) in pieces.enumerated() {
            let (s, e) = (starts[i], ends[i])
            let (a, b) = (max(lo, s), min(hi, e))
            guard a < b else { continue }
            guard let piece else { preconditionFailure("RowBands: rows \(lo)..<\(hi) were released") }
            parts.append(a == s && b == e ? piece : piece[0..., (a - s) ..< (b - s)])
        }
        return parts.count == 1 ? parts[0] : concatenated(parts, axis: 1)
    }

    /// Drops every band that ends at or above `row`.
    func release(below row: Int) {
        for i in pieces.indices where ends[i] <= row { pieces[i] = nil }
    }

    /// Per-(batch, group) mean and 1/std over the whole tensor, shaped [B, 1, 1, G, 1] for an NHWGc view.
    func groupStats(groups: Int, eps: Float) -> (mean: MLXArray, invStd: MLXArray) {
        let groupSize = channels / groups
        var count = 0.0
        var mean: MLXArray?, m2: MLXArray?
        for piece in pieces {
            guard let piece else { preconditionFailure("RowBands: stats over released rows") }
            let x = piece.asType(.float32).reshaped(batch, -1, groups, groupSize)
            let n = Double(x.dim(1) * groupSize)
            let m = x.mean(axes: [1, 3])
            let d = x - m.reshaped(batch, 1, groups, 1)
            let q = (d * d).sum(axes: [1, 3])
            if let mean0 = mean, let m20 = m2 {  // Chan et al. pairwise combine
                let delta = m - mean0
                let total = count + n
                mean = mean0 + delta * Float(n / total)
                m2 = m20 + q + delta * delta * Float(count * n / total)
                count = total
            } else {
                (mean, m2, count) = (m, q, n)
            }
            eval(mean!, m2!)
        }
        let invStd = rsqrt(m2! / Float(count) + eps)
        return (mean!.reshaped(batch, 1, 1, groups, 1), invStd.reshaped(batch, 1, 1, groups, 1))
    }
}

/// GroupNorm on a band with whole-tensor stats; fp32 out, like `GroupNorm` on an fp32 input.
private func groupNormed(_ x: MLXArray, _ norm: GroupNorm, _ stats: (mean: MLXArray, invStd: MLXArray)) -> MLXArray {
    let (b, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
    var y = x.asType(.float32).reshaped(b, h, w, norm.groupCount, c / norm.groupCount)
    y = ((y - stats.mean) * stats.invStd).reshaped(b, h, w, c)
    if let weight = norm.weight, let bias = norm.bias { y = weight * y + bias }
    return y
}

/// A stride-1 conv (1×1, or 3×3 with padding 1) over `height` rows, one band at a time. `input(lo, hi)` yields rows
/// lo..<hi of the conv's input; `epilogue(r0, r1, y)` maps each output band; `done(r1)` runs once rows < r1 are out.
private func bandedConv(
    _ conv: Conv2d, height: Int, batch: Int, width: Int, inChannels: Int, tile: Flux2VAETile,
    input: (Int, Int) -> MLXArray,
    epilogue: (Int, Int, MLXArray) -> MLXArray = { _, _, y in y },
    done: (Int) -> Void = { _ in }
) -> RowBands {
    let kernel = conv.weight.dim(1)
    precondition(conv.stride == (1, 1) && conv.dilation == (1, 1) && conv.groups == 1
        && ((kernel == 1 && conv.padding == (0, 0)) || (kernel == 3 && conv.padding == (1, 1))),
        "bandedConv: stride-1 1×1 or padded 3×3 only")
    let halo = kernel / 2
    let rows = tile.rows(batch: batch, width: width, channels: max(inChannels, conv.weight.dim(0)))
    var out: RowBands?
    for r0 in stride(from: 0, to: height, by: rows) {
        let r1 = min(r0 + rows, height)
        var x = input(r0 - halo, r1 + halo)
        var y: MLXArray
        if halo == 0 {
            y = conv(x)
        } else {
            // Real neighbour rows inside the image, zero rows at its edge; the conv pads the width itself.
            let (top, bottom) = (r0 == 0 ? 1 : 0, r1 == height ? 1 : 0)
            if top + bottom > 0 { x = padded(x, widths: [0, .init((top, bottom)), 0, 0]) }
            let route = (conv as? WinogradFreeConv2d)?.route.exactConvRoute ?? .exact
            y = ExactConv.conv2d(x, weight: conv.weight, bias: conv.bias, padding: (0, 1), route: route)
        }
        y = epilogue(r0, r1, y)
        eval(y)
        if out == nil { out = RowBands(batch: batch, width: width, channels: y.dim(3), dtype: y.dtype) }
        out!.append(y)
        done(r1)
    }
    return out!
}

extension Flux2ResnetBlock2D {
    func callAsFunction(_ x: RowBands, tile: Flux2VAETile) -> RowBands {
        let dtype = x.dtype
        let s1 = x.groupStats(groups: norm1.groupCount, eps: norm1.eps)
        let h1 = bandedConv(
            conv1, height: x.height, batch: x.batch, width: x.width, inChannels: x.channels, tile: tile,
            input: { lo, hi in silu(groupNormed(x.rows(lo, hi), self.norm1, s1).asType(dtype)) })
        let s2 = h1.groupStats(groups: norm2.groupCount, eps: norm2.eps)
        let h1Dtype = h1.dtype
        return bandedConv(
            conv2, height: h1.height, batch: x.batch, width: x.width, inChannels: h1.channels, tile: tile,
            input: { lo, hi in silu(groupNormed(h1.rows(lo, hi), self.norm2, s2).asType(h1Dtype)) },
            epilogue: { r0, r1, y in
                let residual = x.rows(r0, r1)
                return y + (self.convShortcut.map { $0(residual) } ?? residual)
            },
            done: { r1 in
                h1.release(below: r1 - 1)
                x.release(below: r1)
            })
    }
}

extension Flux2AttentionBlock {
    func callAsFunction(_ x: RowBands, tile: Flux2VAETile) -> RowBands {
        let dtype = x.dtype
        let (batch, height, width, channels) = (x.batch, x.height, x.width, x.channels)
        let stats = x.groupStats(groups: groupNorm.groupCount, eps: groupNorm.eps)
        let tokens = height * width
        func heads(_ t: MLXArray) -> MLXArray {
            t.reshaped(batch, -1, 1, channels).transposed(0, 2, 1, 3)
        }
        func normed(_ lo: Int, _ hi: Int) -> MLXArray {
            groupNormed(x.rows(lo, hi), groupNorm, stats).asType(dtype)
        }
        // Whole-image K and V; queries chunked by image rows so one chunk's scores stay ≤ bandBytes.
        let kvRows = tile.rows(batch: batch, width: width, channels: channels)
        var ks: [MLXArray] = [], vs: [MLXArray] = []
        for r0 in stride(from: 0, to: height, by: kvRows) {
            let n = normed(r0, min(r0 + kvRows, height))
            let (k, v) = (toK(n), toV(n))
            eval(k, v)
            ks.append(k)
            vs.append(v)
        }
        let k = heads(concatenated(ks, axis: 1))
        let v = heads(concatenated(vs, axis: 1))
        ks = []
        vs = []
        let scale = 1.0 / sqrt(Float(channels))
        let qRows = max(1, tile.bandBytes / max(1, batch * tokens * 4 * width))
        let out = RowBands(batch: batch, width: width, channels: channels, dtype: dtype)
        for r0 in stride(from: 0, to: height, by: qRows) {
            let r1 = min(r0 + qRows, height)
            let q = heads(toQ(normed(r0, r1)))
            let attended = MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v, scale: scale, mask: .none
            ).transposed(0, 2, 1, 3).reshaped(batch, r1 - r0, width, channels)
            let y = x.rows(r0, r1) + toOut(attended)
            eval(y)
            out.append(y)
            x.release(below: r1)
        }
        return out
    }
}

extension Flux2Upsample2D {
    func callAsFunction(_ x: RowBands, tile: Flux2VAETile) -> RowBands {
        // Rows lo..<hi of the nearest-2× upsample, built from the source rows they repeat.
        func upsampled(_ lo: Int, _ hi: Int) -> MLXArray {
            let (lo, hi) = (max(lo, 0), min(hi, 2 * x.height))
            let a = lo / 2
            var t = x.rows(a, (hi - 1) / 2 + 1)
            t = MLX.repeated(MLX.repeated(t, count: 2, axis: 1), count: 2, axis: 2)
            return t[0..., (lo - 2 * a) ..< (hi - 2 * a)]
        }
        return bandedConv(
            conv, height: 2 * x.height, batch: x.batch, width: 2 * x.width, inChannels: x.channels, tile: tile,
            input: upsampled,
            done: { r1 in x.release(below: r1 / 2 - 1) })
    }
}

extension Flux2Decoder {
    func callAsFunction(_ hiddenStates: MLXArray, tile: Flux2VAETile) -> MLXArray {
        let dtype = hiddenStates.dtype
        let latent = RowBands(hiddenStates.transposed(0, 2, 3, 1))
        var h = bandedConv(
            convIn, height: latent.height, batch: latent.batch, width: latent.width,
            inChannels: latent.channels, tile: tile, input: latent.rows)
        h = midBlock.resnets[0](h, tile: tile)
        h = midBlock.attentions[0](h, tile: tile)
        h = midBlock.resnets[1](h, tile: tile)
        for upBlock in upBlocks {
            for resnet in upBlock.resnets { h = resnet(h, tile: tile) }
            for upsampler in upBlock.upsamplers { h = upsampler(h, tile: tile) }
        }
        let x = h
        let stats = x.groupStats(groups: convNormOut.groupCount, eps: convNormOut.eps)
        let image = bandedConv(
            convOut, height: x.height, batch: x.batch, width: x.width, inChannels: x.channels, tile: tile,
            input: { lo, hi in silu(groupNormed(x.rows(lo, hi), self.convNormOut, stats).asType(dtype)) },
            done: { r1 in x.release(below: r1 - 1) })
        return image.rows(0, image.height).transposed(0, 3, 1, 2)
    }
}

extension Flux2VAE {
    /// `decode(_:)` in row bands: the same image (no seams; float-reassociation-level differences), with the peak
    /// bounded by one resnet's full-resolution input and middle tensor instead of their temporaries. Opt-in.
    public func decode(_ latents: MLXArray, tile: Flux2VAETile) -> MLXArray {
        var l = (latents / Self.scalingFactor) + Self.shiftFactor
        l = l.transposed(0, 2, 3, 1)
        l = postQuantConv(l)
        l = l.transposed(0, 3, 1, 2)
        return decoder(l, tile: tile)
    }

    /// `decodePackedLatents(_:)` through `decode(_:tile:)`.
    public func decodePackedLatents(_ packedLatents: MLXArray, tile: Flux2VAETile) -> MLXArray {
        decode(denormalizedUnpatchified(packedLatents), tile: tile)
    }
}
