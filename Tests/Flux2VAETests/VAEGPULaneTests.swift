// Two-lane decode gate for the Winograd-free conv route: GPU lane (route on / raw Winograd
// conv2d; fp32 and bf16 — ERNIE decodes bf16, Lens and Klein fp32) and the CPU lane, against the
// PyTorch fp32 golden. The consumers' own VAE gates (Lens P3) pin the CPU device, so the GPU
// decode was never gated before this.
//
// Golden: lens-mlx/goldens/lens_goldens.safetensors (lossless conversion of lens_goldens.npz):
// `final_latent` [1,1024,128] = the real 512² Lens T2I latent, `decoded_image` = its PyTorch fp32
// CPU decode. Production size: the same real latent tiled 2×2 (1024²), against the CPU lane.
// `testDecodeTiming1024` times the GPU lane alone (no CPU-lane buffers pooled).
//
// With the route, the GPU fp32 lane still sits ~4e-4 from the CPU lane: that residual is the
// mid-block attention (fp32 Linear + SDPA run TF32 — MLX_ENABLE_TF32 defaults on), not a conv —
// under MLX_ENABLE_TF32=0 the route decode is 3.4e-6 from the CPU lane. The gate threshold
// (1e-3) separates it from the raw Winograd decode (1.9e-3).
//
// Run: FLUX2VAE_PARITY=1 swift test -c release -Xswiftc -enable-testing --filter VAEGPULaneTests
// Overrides: FLUX2VAE_DIR (a diffusers vae/ snapshot), FLUX2VAE_GOLDEN.

import Foundation
import MLX
import XCTest

@testable import Flux2VAE

final class VAEGPULaneTests: XCTestCase {
    static let env = ProcessInfo.processInfo.environment
    static let vaeDir = URL(
        fileURLWithPath: env["FLUX2VAE_DIR"] ?? "/Volumes/DEV_ARCHIVE/lens-mlx/weights/Lens/vae")
    static let golden = URL(
        fileURLWithPath: env["FLUX2VAE_GOLDEN"]
            ?? "/Volumes/DEV_ARCHIVE/lens-mlx/goldens/lens_goldens.safetensors")

    struct Stats: CustomStringConvertible {
        let relL2: Float, maxAbs: Float, psnr: Float
        var description: String {
            String(format: "relL2 %.2e  maxAbs %.2e  PSNR %6.2f dB", relL2, maxAbs, psnr)
        }
    }

    /// relL2 / maxAbs / PSNR over [-1, 1] images (peak-to-peak 2).
    static func stats(_ a: MLXArray, _ ref: MLXArray) -> Stats {
        let d = a.asType(.float32) - ref.asType(.float32)
        let rel = sqrt(sum(d * d)) / sqrt(sum(square(ref.asType(.float32))))
        let mx = abs(d).max()
        let mse = mean(d * d)
        eval(rel, mx, mse)
        let psnr = 10 * log10(4 / max(mse.item(Float.self), 1e-30))
        return Stats(relL2: rel.item(Float.self), maxAbs: mx.item(Float.self), psnr: psnr)
    }

    static func onCPU(_ f: () -> MLXArray) -> MLXArray {
        Device.withDefaultDevice(.cpu) { () -> MLXArray in
            let r = f()
            eval(r)
            return r
        }
    }

    /// GPU decode with the route on/off; warm-up run, then the mean of `reps` timed runs.
    static func gpuDecode(_ vae: Flux2VAE, _ packed: MLXArray, route: Bool, reps: Int = 3)
        -> (MLXArray, Double)
    {
        vae.winogradFreeConvs = route
        defer { vae.winogradFreeConvs = true }
        var out = vae.decodePackedLatents(packed)
        eval(out)
        let t0 = Date()
        for _ in 0..<reps {
            out = vae.decodePackedLatents(packed)
            eval(out)
        }
        return (out, Date().timeIntervalSince(t0) / Double(reps) * 1000)
    }

    /// Lens token latents [B, h·w, 128] → packed NCHW [B, 128, h, w] (LensPipeline.packLatentsForDecode).
    static func packForDecode(_ latents: MLXArray, h: Int, w: Int) -> MLXArray {
        let b = latents.dim(0)
        let c = latents.dim(2) / 4
        var x = latents.reshaped(b, h, w, c, 2, 2).transposed(0, 3, 1, 4, 2, 5)
        x = x.reshaped(b, c, h * 2, w * 2)
        x = x.reshaped(b, c, h, 2, w, 2).transposed(0, 1, 3, 5, 2, 4)
        return x.reshaped(b, c * 4, h, w)
    }

    func testGoldenAndProductionDecode() throws {
        try XCTSkipUnless(Self.env["FLUX2VAE_PARITY"] == "1", "set FLUX2VAE_PARITY=1 to run")
        let g = try MLX.loadArrays(url: Self.golden)
        let packed = Self.packForDecode(g["final_latent"]!.asType(.float32), h: 32, w: 32)
        let ref = g["decoded_image"]!.asType(.float32)
        let vae32 = try Flux2VAEWeights.loadVAE(directory: Self.vaeDir, dtype: .float32)
        let vae16 = try Flux2VAEWeights.loadVAE(directory: Self.vaeDir, dtype: .bfloat16)

        let cpu = Self.onCPU { vae32.decodePackedLatents(packed) }
        let (r32, _) = Self.gpuDecode(vae32, packed, route: true)
        let (w32, _) = Self.gpuDecode(vae32, packed, route: false)
        let (r16, _) = Self.gpuDecode(vae16, packed.asType(.bfloat16), route: true)
        let (w16, _) = Self.gpuDecode(vae16, packed.asType(.bfloat16), route: false)
        print("[golden 512² vs PyTorch fp32 CPU]")
        print("  CPU lane fp32         \(Self.stats(cpu, ref))")
        print("  GPU fp32, route       \(Self.stats(r32, ref))")
        print("  GPU fp32, raw conv2d  \(Self.stats(w32, ref))")
        print("  GPU bf16, route       \(Self.stats(r16, ref))")
        print("  GPU bf16, raw conv2d  \(Self.stats(w16, ref))")
        print("  GPU fp32 route vs CPU lane  \(Self.stats(r32, cpu))")
        print("  GPU fp32 raw   vs CPU lane  \(Self.stats(w32, cpu))")
        XCTAssertLessThan(Self.stats(r32, cpu).relL2, 1e-3, "GPU route vs CPU lane")

        // Production size: the real latent tiled 2×2 → 1024², vs the CPU-lane fp32 decode.
        let big = tiled(packed, repetitions: [1, 1, 2, 2])
        let t0 = Date()
        let cpuBig = Self.onCPU { vae32.decodePackedLatents(big) }
        print(String(format: "[1024² (real latent tiled 2×2) vs CPU-lane fp32 — CPU decode %.1f s]",
                     Date().timeIntervalSince(t0)))
        Memory.clearCache()  // the CPU lane leaves tens of GB pooled; don't time against that
        let rows = [
            ("GPU fp32, route      ", Self.gpuDecode(vae32, big, route: true)),
            ("GPU fp32, raw conv2d ", Self.gpuDecode(vae32, big, route: false)),
            ("GPU bf16, route      ", Self.gpuDecode(vae16, big.asType(.bfloat16), route: true)),
            ("GPU bf16, raw conv2d ", Self.gpuDecode(vae16, big.asType(.bfloat16), route: false)),
        ]
        for (label, (img, ms)) in rows {
            print(String(format: "  %@ %@  %7.1f ms", label, Self.stats(img, cpuBig).description, ms))
        }
    }

    /// GPU-lane decode time at 1024², route vs raw, interleaved over rounds (no CPU lane).
    func testDecodeTiming1024() throws {
        try XCTSkipUnless(Self.env["FLUX2VAE_PARITY"] == "1", "set FLUX2VAE_PARITY=1 to run")
        let g = try MLX.loadArrays(url: Self.golden)
        let packed = Self.packForDecode(g["final_latent"]!.asType(.float32), h: 32, w: 32)
        let big = tiled(packed, repetitions: [1, 1, 2, 2])
        func median(_ v: [Double]) -> Double { v.sorted()[v.count / 2] }
        func fmt(_ v: [Double]) -> String { v.map { String(format: "%.0f", $0) }.joined(separator: "/") }
        for dtype in [DType.float32, .bfloat16] {
            let vae = try Flux2VAEWeights.loadVAE(directory: Self.vaeDir, dtype: dtype)
            let z = big.asType(dtype)
            var route = [Double](), raw = [Double]()
            for _ in 0..<3 {
                route.append(Self.gpuDecode(vae, z, route: true).1)
                raw.append(Self.gpuDecode(vae, z, route: false).1)
            }
            print(String(format: "[timing 1024² %@] route %@ ms | raw %@ ms | median delta %+.0f ms",
                         dtype == .float32 ? "fp32" : "bf16", fmt(route), fmt(raw),
                         median(route) - median(raw)))
            Memory.clearCache()
        }
    }
}
