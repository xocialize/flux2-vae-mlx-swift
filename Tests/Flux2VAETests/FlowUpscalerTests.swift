// FlowUpscaler gates.
// - Weight-free: patchify ∘ unpatchify is the identity, and the bilinear resize matches PyTorch's
//   F.interpolate(mode="bilinear", align_corners=False) on hand-computed values.
// - Parity (runs when the fixtures exist): the PyTorch reference's pairs from `parity/flowup_pairs.py`
//   (CPU fp32, TensorForger's upscaler_unet.py) against this port, GPU fp32 on the exact route. Set
//   FLOWUP_PAIRS (default /Volumes/DEV_VOL1/flowup-port) and FLOWUP_WEIGHTS (the original fp32 file).
// Run: swift test --filter FlowUpscalerTests

import Foundation
import MLX
import MLXRandom
import XCTest

@testable import Flux2VAE

final class FlowUpscalerTests: XCTestCase {
    func testPatchifyRoundTrip() {
        MLXRandom.seed(3)
        let packed = MLXRandom.normal([2, 128, 5, 7])
        let back = FlowUpscaler.patchify(FlowUpscaler.unpatchify(packed))
        XCTAssertEqual(back.shape, packed.shape)
        XCTAssertEqual(abs(back - packed).max().item(Float.self), 0)
    }

    func testBilinearMatchesPyTorch() {
        // torch.nn.functional.interpolate(torch.arange(4.).view(1,1,1,4), size=(1,8), mode="bilinear")
        //   = [0, .25, .75, 1.25, 1.75, 2.25, 2.75, 3]; and to size 3: [0.1667, 1.5, 2.8333].
        let x = MLXArray([0, 1, 2, 3] as [Float]).reshaped(1, 1, 4, 1)
        let up = FlowUpscalerUNet.bilinear(x, height: 1, width: 8).reshaped(-1).asArray(Float.self)
        XCTAssertEqual(up, [0, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3])
        let down = FlowUpscalerUNet.bilinear(x, height: 1, width: 3).reshaped(-1).asArray(Float.self)
        for (a, b) in zip(down, [1.0 / 6, 1.5, 17.0 / 6] as [Float]) { XCTAssertEqual(a, b, accuracy: 1e-6) }
    }

    func testParityWithPyTorch() throws {
        let env = ProcessInfo.processInfo.environment
        let dir = URL(fileURLWithPath: env["FLOWUP_PAIRS"] ?? "/Volumes/DEV_VOL1/flowup-port")
        let weights = URL(fileURLWithPath: env["FLOWUP_WEIGHTS"]
            ?? "/Volumes/DEV_VOL1/Models/Models/models--TensorForger--FlowUpscaler/flow_upscaler.safetensors")
        guard FileManager.default.fileExists(atPath: weights.path) else { throw XCTSkip("no weights at \(weights.path)") }
        let up = try FlowUpscaler.load(url: weights, dtype: .float32)
        var ran = 0
        for name in ["square", "wide", "odd"] {
            let url = dir.appendingPathComponent("pair-\(name).safetensors")
            guard FileManager.default.fileExists(atPath: url.path) else { continue }
            let pair = try MLX.loadArrays(url: url)
            let out = up.upscale(latents: pair["latents"]!, noise: pair["noise"]!)
            let ref = pair["out"]!
            let (a, b) = (out.flattened(), ref.flattened())
            let cos = (sum(a * b) / (sqrt(sum(a * a)) * sqrt(sum(b * b)))).item(Float.self)
            let rel = (sqrt(sum(square(a - b))) / sqrt(sum(b * b))).item(Float.self)
            XCTAssertGreaterThan(cos, 0.999_999, name)
            XCTAssertLessThan(rel, 1e-5, "\(name): relL2 \(rel)")
            ran += 1
        }
        if ran == 0 { throw XCTSkip("no pairs under \(dir.path)") }
    }
}
