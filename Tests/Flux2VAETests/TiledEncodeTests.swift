// Weight-free gate for the row-band encode (`encodeMean(_:tile:)`): a randomly initialised narrow encoder, encoded
// untiled and in bands down to one row, must agree to float-reassociation level on both devices. Any halo, stride-2
// band, release or GroupNorm-stats bug shows up as a seam or a colour shift orders of magnitude above that.
// Run: swift test --filter TiledEncodeTests

import Foundation
import MLX
import MLXRandom
import XCTest

@testable import Flux2VAE

final class TiledEncodeTests: XCTestCase {
    func relL2(_ a: MLXArray, _ b: MLXArray) -> Float {
        let d = a - b
        return (sqrt(sum(d * d)) / sqrt(sum(b * b))).item(Float.self)
    }

    func check(device: Device, route: Flux2VAEConvRoute, tolerance: Float) {
        Device.withDefaultDevice(device) {
            MLXRandom.seed(7)
            let encoder = Flux2VAEEncoder(blockOutChannels: [32, 64, 64, 64])
            encoder.convRoute = route
            // Non-square; 120 rows downsample 120 → 60 → 30 → 15, so the last stage has odd rows.
            let image = MLXRandom.uniform(low: -1, high: 1, [2, 3, 120, 152])
            let untiled = encoder.encodeMean(image)
            eval(untiled)
            for bandBytes in [1 << 30, 64 << 10, 4 << 10, 1] {
                let tiled = encoder.encodeMean(image, tile: Flux2VAETile(bandBytes: bandBytes))
                XCTAssertEqual(tiled.shape, untiled.shape)
                let e = relL2(tiled, untiled)
                XCTAssertLessThan(e, tolerance, "\(device) \(route) band \(bandBytes) B: relL2 \(e)")
            }
            // 100 rows downsample 100 → 50 → 25 → 12: an odd-height input to the last downsample.
            let odd = MLXRandom.uniform(low: -1, high: 1, [1, 3, 100, 72])
            let u = encoder.encodeMean(odd)
            let t = encoder.encodeMean(odd, tile: Flux2VAETile(bandBytes: 1))
            XCTAssertEqual(t.shape, u.shape)
            XCTAssertLessThan(relL2(t, u), tolerance, "\(device) \(route) 100×72 one-row bands")
        }
    }

    func testTiledMatchesUntiledCPU() { check(device: .cpu, route: .conv3d, tolerance: 1e-5) }
    func testTiledMatchesUntiledGPUExact() { check(device: .gpu, route: .conv3d, tolerance: 1e-4) }
}
