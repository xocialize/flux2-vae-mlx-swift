// Memory-bounded encode: the row-band recipe of Flux2VAETiledDecode.swift applied to the encoder tower.
//
// The untiled encode's peak at 2048² is the mid-block attention: 512 channels at H/8 is 65 536 tokens, and the
// reference's explicit softmax(QKᵀ)V materialises a 17 GB score matrix (≈ 4.1 GB per reference MP in total). The
// full-resolution stage (128 ch, 2 GB per tensor at 2048²) and its GroupNorm temporaries come next.
//
// Everything the decode path does carries over unchanged: NHWC row bands, whole-tensor GroupNorm stats, one real
// halo row per stride-1 3×3 conv, query-chunked attention against whole-image K/V, and bands released once no later
// band needs them. The encoder adds the stride-2 downsample (pad bottom/right by one, then a valid 3×3 stride-2
// conv): output row i reads input rows 2i, 2i+1, 2i+2, so an output band o0..<o1 reads input rows 2·o0..<2·o1+1,
// with a zero row only where 2·o1 reaches the bottom edge, and input rows below 2·o1 are free once it is out.
//
// The result matches the untiled encode up to float reassociation, with no seams by construction. The floor is one
// full-resolution resnet's input plus its middle tensor: ~4 GB fp32 at 2048².

import Foundation
import MLX
import MLXNN

extension EncResnetBlock2D {
    func callAsFunction(_ x: RowBands, tile: Flux2VAETile) -> RowBands {
        let dtype = x.dtype
        let s1 = x.groupStats(groups: norm1.groupCount, eps: norm1.eps)
        let h1 = bandedConv(
            conv1, height: x.height, batch: x.batch, width: x.width, inChannels: x.channels, tile: tile,
            input: { lo, hi in encSilu(groupNormed(x.rows(lo, hi), self.norm1, s1).asType(dtype)) })
        let s2 = h1.groupStats(groups: norm2.groupCount, eps: norm2.eps)
        let h1Dtype = h1.dtype
        return bandedConv(
            conv2, height: h1.height, batch: x.batch, width: x.width, inChannels: h1.channels, tile: tile,
            input: { lo, hi in encSilu(groupNormed(h1.rows(lo, hi), self.norm2, s2).asType(h1Dtype)) },
            epilogue: { r0, r1, y in
                let residual = x.rows(r0, r1)
                return (self.convShortcut.map { $0(residual) } ?? residual) + y
            },
            done: { r1 in
                h1.release(below: r1 - 1)
                x.release(below: r1)
            })
    }
}

extension EncDownsample2D {
    func callAsFunction(_ x: RowBands, tile: Flux2VAETile) -> RowBands {
        let outHeight = (x.height - 2) / 2 + 1  // (H + 1 − 3) / 2 + 1 after the one-row bottom pad
        // Output rows per band, so the 2·rows + 1 input rows it reads stay about one band of the input.
        let rows = max(1, tile.rows(batch: x.batch, width: x.width, channels: x.channels) / 2)
        var out: RowBands?
        for o0 in stride(from: 0, to: outHeight, by: rows) {
            let o1 = min(o0 + rows, outHeight)
            let bottom = 2 * o1 + 1 > x.height ? 1 : 0  // the reference's zero pad row, reached by the last band
            let band = MLX.padded(
                x.rows(2 * o0, 2 * o1 + 1), widths: [.init((0, 0)), .init((0, bottom)), .init((0, 1)), .init((0, 0))])
            let y = conv(band)
            eval(y)
            if out == nil { out = RowBands(batch: x.batch, width: y.dim(2), channels: y.dim(3), dtype: y.dtype) }
            out!.append(y)
            x.release(below: 2 * o1)
        }
        return out!
    }
}

extension EncVAEAttention {
    func callAsFunction(_ x: RowBands, tile: Flux2VAETile) -> RowBands {
        let dtype = x.dtype
        let (batch, height, width, channels) = (x.batch, x.height, x.width, x.channels)
        let stats = x.groupStats(groups: groupNorm_.groupCount, eps: groupNorm_.eps)
        func normed(_ lo: Int, _ hi: Int) -> MLXArray {
            groupNormed(x.rows(lo, hi), groupNorm_, stats).asType(dtype).reshaped(batch, -1, channels)
        }
        // Whole-image K and V; queries chunked by image rows so one chunk's scores stay ≤ bandBytes. Same math as
        // the untiled block (explicit softmax(QKᵀ·scale)V): softmax is per query row, so chunking changes nothing.
        let kvRows = tile.rows(batch: batch, width: width, channels: channels)
        var ks: [MLXArray] = [], vs: [MLXArray] = []
        for r0 in stride(from: 0, to: height, by: kvRows) {
            let n = normed(r0, min(r0 + kvRows, height))
            let (k, v) = (toK(n), toV(n))
            eval(k, v)
            ks.append(k)
            vs.append(v)
        }
        let kT = concatenated(ks, axis: 1).transposed(0, 2, 1)
        let v = concatenated(vs, axis: 1)
        ks = []
        vs = []
        let qRows = max(1, tile.bandBytes / max(1, batch * height * width * 4 * width))
        let out = RowBands(batch: batch, width: width, channels: channels, dtype: dtype)
        for r0 in stride(from: 0, to: height, by: qRows) {
            let r1 = min(r0 + qRows, height)
            let attn = softmax(matmul(toQ(normed(r0, r1)), kT) * scale, axis: -1)
            let y = x.rows(r0, r1) + toOut[0](matmul(attn, v)).reshaped(batch, r1 - r0, width, channels)
            eval(y)
            out.append(y)
            x.release(below: r1)
        }
        return out
    }
}

extension Flux2EncoderTower {
    /// x: NHWC image → NHWC moments (before quant_conv).
    func callAsFunction(_ x: MLXArray, tile: Flux2VAETile) -> MLXArray {
        let dtype = x.dtype
        let image = RowBands(x)
        var h = bandedConv(
            convIn, height: image.height, batch: image.batch, width: image.width, inChannels: image.channels,
            tile: tile, input: image.rows)
        for block in downBlocks {
            for resnet in block.resnets { h = resnet(h, tile: tile) }
            if let downsamplers = block.downsamplers { h = downsamplers[0](h, tile: tile) }
        }
        h = midBlock.resnets[0](h, tile: tile)
        h = midBlock.attentions[0](h, tile: tile)
        h = midBlock.resnets[1](h, tile: tile)
        let x = h
        let stats = x.groupStats(groups: convNormOut.groupCount, eps: convNormOut.eps)
        let moments = bandedConv(
            convOut, height: x.height, batch: x.batch, width: x.width, inChannels: x.channels, tile: tile,
            input: { lo, hi in encSilu(groupNormed(x.rows(lo, hi), self.convNormOut, stats).asType(dtype)) },
            done: { r1 in x.release(below: r1 - 1) })
        return moments.rows(0, moments.height)
    }
}

extension Flux2VAEEncoder {
    /// `encodeMean(_:)` in row bands: the same latent mean (no seams; float-reassociation-level differences), with
    /// the peak bounded by one full-resolution resnet's input and middle tensor instead of the mid-block attention's
    /// score matrix and the GroupNorm temporaries. Opt-in.
    public func encodeMean(_ imageNCHW: MLXArray, tile: Flux2VAETile) -> MLXArray {
        let x = imageNCHW.transposed(0, 2, 3, 1)
        var moments = quantConv(encoder(x, tile: tile))
        moments = moments.transposed(0, 3, 1, 2)
        return moments[0..., ..<latentChannels, 0..., 0...]
    }

    /// `encodePacked(_:)` through `encodeMean(_:tile:)`.
    public func encodePacked(_ imageNCHW: MLXArray, tile: Flux2VAETile) -> MLXArray {
        packed(mean: encodeMean(imageNCHW, tile: tile))
    }
}
