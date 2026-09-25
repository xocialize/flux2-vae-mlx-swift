// Weight-free probe: mlx's Winograd conv2d path at FLUX.2 decoder shapes, raw vs the
// WinogradFreeConv2d route (Sources/Flux2VAE/WinogradFreeConv2d.swift), both against the CPU conv.
//
// Measured 2026-09-24 on the M5 Max, mlx-swift 0.31.6 (GPU vs CPU, fp32 relL2): raw conv2d
// 6.4e-3 at every shape below (Winograd) — route ~1e-6.
// Removal signal: when a new mlx-swift pin reports raw conv2d exact-class here, drop the route.
//
// Run: swift test --filter WinogradProbeTests

import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import Flux2VAE

final class WinogradProbeTests: XCTestCase {

    static func relL2(_ a: MLXArray, _ b: MLXArray) -> (rel: Float, maxAbs: Float) {
        let d = a.asType(.float32) - b.asType(.float32)
        let rel = sqrt(sum(d * d)) / sqrt(sum(square(b.asType(.float32))))
        let mx = abs(d).max()
        eval(rel, mx)
        return (rel.item(Float.self), mx.item(Float.self))
    }

    func testDecoderConvShapes() throws {
        // (C, O, spatial): conv_in, mid/up resnets, the 256/128-ch stages.
        for (c, o, s) in [(32, 512, 64), (512, 512, 64), (512, 256, 128), (256, 128, 256), (128, 128, 256)] {
            MLXRandom.seed(UInt64(c * 7 + o))
            let conv = WinogradFreeConv2d(
                inputChannels: c, outputChannels: o, kernelSize: 3, stride: 1, padding: 1)
            let x = MLXRandom.normal([1, s, s, c])
            XCTAssertTrue(
                WinogradFreeConv2d.takesWinograd(
                    input: x, weight: conv.weight, stride: conv.stride,
                    dilation: conv.dilation, groups: conv.groups),
                "\(c)→\(o) @\(s)² should be inside the Winograd predicate")
            let ref = Device.withDefaultDevice(.cpu) { () -> MLXArray in
                let r = conv2d(x, conv.weight, stride: 1, padding: 1) + conv.bias!
                eval(r)
                return r
            }
            conv.enabled = false
            let raw = conv(x)
            conv.enabled = true
            let routed = conv(x)
            eval(raw, routed)
            let r0 = Self.relL2(raw, ref)
            let r1 = Self.relL2(routed, ref)
            print(
                String(
                    format: "  %d→%d @%d²: raw conv2d relL2 %.2e (max %.1e) %@ | route relL2 %.2e",
                    c, o, s, r0.rel, r0.maxAbs,
                    r0.rel < 1e-5 ? "EXACT — route removable" : "lossy", r1.rel))
            XCTAssertLessThan(r1.rel, 1e-5, "conv3d route must be exact-class")
        }
        // conv_out (128→3) is outside the window: the class must fall through to plain conv2d.
        let out = WinogradFreeConv2d(inputChannels: 128, outputChannels: 3, kernelSize: 3, padding: 1)
        let z = MLXRandom.normal([1, 128, 128, 128])
        let a = out(z)
        let b = conv2d(z, out.weight, stride: 1, padding: 1) + out.bias!
        XCTAssertEqual(Self.relL2(a, b).maxAbs, 0, "out-of-window shapes must be untouched")
    }
}
