// Weight-free gate for the row-band decode (`decode(_:tile:)`): a randomly initialised narrow decoder, decoded
// untiled and in bands down to one row, must agree to float-reassociation level on both devices. Any halo,
// release or GroupNorm-stats bug shows up as a seam or a colour shift orders of magnitude above that.
// Run: swift test --filter TiledDecodeTests

import Foundation
import MLX
import MLXRandom
import XCTest

@testable import Flux2VAE

final class TiledDecodeTests: XCTestCase {
    func relL2(_ a: MLXArray, _ b: MLXArray) -> Float {
        let d = a - b
        return (sqrt(sum(d * d)) / sqrt(sum(b * b))).item(Float.self)
    }

    func check(device: Device, route: Flux2VAEConvRoute, tolerance: Float) {
        Device.withDefaultDevice(device) {
            MLXRandom.seed(7)
            let vae = Flux2VAE(decoderBlockOutChannels: [32, 64, 64, 64])
            vae.convRoute = route
            // Non-square, odd latent rows: 2·7 × 2·9 latent → 112 × 144 image.
            let packed = MLXRandom.normal([2, 4 * Flux2VAE.latentChannels, 7, 9])
            let untiled = vae.decodePackedLatents(packed)
            eval(untiled)
            for bandBytes in [1 << 30, 64 << 10, 4 << 10, 1] {
                let tiled = vae.decodePackedLatents(packed, tile: Flux2VAETile(bandBytes: bandBytes))
                XCTAssertEqual(tiled.shape, untiled.shape)
                let e = relL2(tiled, untiled)
                XCTAssertLessThan(e, tolerance, "\(device) \(route) band \(bandBytes) B: relL2 \(e)")
            }
        }
    }

    func testTiledMatchesUntiledCPU() { check(device: .cpu, route: .conv3d, tolerance: 1e-5) }
    func testTiledMatchesUntiledGPUExact() { check(device: .gpu, route: .conv3d, tolerance: 1e-4) }
}
