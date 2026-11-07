// Flux2VAEBench — decode peak + parity harness for the untiled and the row-band (`tile:`) decode.
//
//   Flux2VAEBench decode <vaeDir> <latents.safetensors> <size> <untiled|tiled> <route> [--bf16] [--band MB] [--save out.safetensors]
//     Decodes the real 512² latent `latents.15` (FLUX 3 klein oracle run_512) tiled up to size² and prints the MLX
//     peak (Memory.peakMemory) with the pool capped at the engine's 2 GB (.automatic, AB-L-0192).
//   Flux2VAEBench compare <a.safetensors> <b.safetensors> [--band-rows N]
//     relL2 / maxAbs / PSNR over [-1, 1], plus the per-row and per-column error profile (the seam check).
//   Flux2VAEBench golden <vaeDir> <latents.safetensors> <route>
//     512² untiled and tiled (bands as small as 1 MB) against the reference `decoded`.
//   Flux2VAEBench encode <vaeDir> <image.safetensors> <untiled|tiled> <route> [--bf16] [--band MB] [--save out.safetensors]
//     Encodes `image` [1, 3, H, W] in [-1, 1] to packed latents and prints the MLX peak under the same 2 GB pool cap;
//     --save writes them as `image` so `compare` reads them.
//   Flux2VAEBench flowup <flow_upscaler.safetensors> <pair.safetensors> [--bf16] [--route r] [--save out.safetensors]
//     FlowUpscaler parity: `latents` [B,32,h,w] (unpatchified, bn-normalised) and `noise` [B,32,2h,2w] in, the ×2
//     latents out; against the file's `out` (the PyTorch reference's noise − v) when present: cos, relL2, maxAbs.
//   Flux2VAEBench flowpipe <flow_upscaler.safetensors> <vaeDir> <packed.safetensors> <passes> [--seed N] [--bf16-up]
//       [--route r] [--save out.safetensors]
//   Flux2VAEBench flowimg <vaeDir> <in.safetensors> <key> <out.safetensors>
//     Decodes `key` (unpatchified [1,32,H,W] or packed [1,128,h,w], bn-normalised) with the row-band decode on the exact
//     route and saves it as `image`, so `compare` can score two upscaler outputs in pixels.
//   (flowpipe:) `packed` [1,128,h,w] → `passes` ×2 passes → the row-band decode; times and MLX peaks per stage (2 GB pool cap).

import Flux2VAE
import Foundation
import MLX

func fail(_ m: String) -> Never {
    FileHandle.standardError.write((m + "\n").data(using: .utf8)!)
    exit(2)
}

let gb = Double(1 << 30)
var args = Array(CommandLine.arguments.dropFirst())
@MainActor func flag(_ name: String) -> String? {
    guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
    let v = args[i + 1]
    args.removeSubrange(i...(i + 1))
    return v
}
@MainActor func has(_ name: String) -> Bool {
    guard let i = args.firstIndex(of: name) else { return false }
    args.remove(at: i)
    return true
}

/// FLUX 3 klein tokens [1024, 128] → packed NCHW [1, 128, 32, 32] (Flux3Latents.packed).
func packedGolden(_ url: URL) throws -> MLXArray {
    let run = try MLX.loadArrays(url: url)
    guard let t = run["latents.15"] else { fail("no latents.15 in \(url.path)") }
    return t.asType(.float32).reshaped(32, 32, 128).transposed(2, 0, 1)[.newAxis]
}

struct Stats: CustomStringConvertible {
    let relL2: Float, maxAbs: Float, psnr: Double
    var description: String { String(format: "relL2 %.2e  maxAbs %.2e  PSNR %6.2f dB", relL2, maxAbs, psnr) }
}

func stats(_ a: MLXArray, _ ref: MLXArray) -> Stats {
    let d = a.asType(.float32) - ref.asType(.float32)
    let rel = sqrt(sum(d * d)) / sqrt(sum(square(ref.asType(.float32))))
    let mx = abs(d).max()
    let mse = mean(d * d)
    eval(rel, mx, mse)
    return Stats(relL2: rel.item(Float.self), maxAbs: mx.item(Float.self),
                 psnr: 10 * log10(4 / max(Double(mse.item(Float.self)), 1e-30)))
}

func route(_ s: String) -> Flux2VAEConvRoute {
    guard let r = Flux2VAEConvRoute(rawValue: s) else { fail("route: winograd | conv3d | fp32Winograd") }
    return r
}

guard let mode = args.first else { fail("usage: decode | compare | golden | encode (see the header)") }
args.removeFirst()

switch mode {
case "decode":
    let bf16 = has("--bf16")
    let bandMB = flag("--band").flatMap(Int.init)
    let save = flag("--save")
    guard args.count == 5, let size = Int(args[2]), size % 512 == 0 else {
        fail("decode <vaeDir> <latents> <size multiple of 512> <untiled|tiled> <route>")
    }
    Memory.cacheLimit = 2 << 30  // MLXServeCore .automatic
    let dtype: DType = bf16 ? .bfloat16 : .float32
    let vae = try Flux2VAEWeights.loadVAE(directory: URL(fileURLWithPath: args[0]), dtype: dtype)
    vae.convRoute = route(args[4])
    let reps = size / 512
    let packed = tiled(try packedGolden(URL(fileURLWithPath: args[1])), repetitions: [1, 1, reps, reps])
        .asType(dtype)
    eval(packed)
    let tile = args[3] == "tiled" ? Flux2VAETile(bandBytes: (bandMB ?? 64) << 20) : nil
    Memory.clearCache()
    let resident = Memory.activeMemory
    Memory.peakMemory = 0
    let t0 = Date()
    let image = tile.map { vae.decodePackedLatents(packed, tile: $0) } ?? vae.decodePackedLatents(packed)
    eval(image)
    let ms = Date().timeIntervalSince(t0) * 1000
    let peak = Memory.peakMemory
    print(String(
        format: "[decode] %d² %@ %@ %@%@: peak %.2f GB (resident %.2f GB, peak over resident %.2f GB), %.0f ms",
        size, bf16 ? "bf16" : "fp32", args[4], args[3], tile.map { " band \($0.bandBytes >> 20) MB" } ?? "",
        Double(peak) / gb, Double(resident) / gb, Double(peak - resident) / gb, ms))
    if let save { try MLX.save(arrays: ["image": image.asType(.float32)], url: URL(fileURLWithPath: save)) }

case "compare":
    let bandRows = flag("--band-rows").flatMap(Int.init)
    guard args.count == 2 else { fail("compare <a> <b>") }
    let a = try MLX.loadArrays(url: URL(fileURLWithPath: args[0]))["image"]!
    let b = try MLX.loadArrays(url: URL(fileURLWithPath: args[1]))["image"]!
    print("[compare] \(stats(a, b))")
    // Seam check: RMS error per image row / column. A seam shows as rows that stand out from the median.
    let d = (a - b).asType(.float32)
    let rowRMS = sqrt(mean(d * d, axes: [0, 1, 3]))
    let colRMS = sqrt(mean(d * d, axes: [0, 1, 2]))
    eval(rowRMS, colRMS)
    for (name, profile) in [("row", rowRMS.asArray(Float.self)), ("col", colRMS.asArray(Float.self))] {
        let sorted = profile.sorted()
        let median = sorted[sorted.count / 2]
        let worst = profile.indices.max { profile[$0] < profile[$1] }!
        var line = String(format: "  %@ RMS: median %.2e, worst %.2e at %d (%.1f× median)",
                          name, median, profile[worst], worst, profile[worst] / max(median, 1e-30))
        if let bandRows, name == "row" {
            let edges = Set(stride(from: bandRows, to: profile.count, by: bandRows).flatMap { [$0 - 1, $0] })
            let atEdges = edges.map { profile[$0] }.max() ?? 0
            let inside = profile.indices.filter { !edges.contains($0) }.map { profile[$0] }.max() ?? 0
            line += String(format: "; band edges (every %d rows) max %.2e vs elsewhere max %.2e", bandRows, atEdges, inside)
        }
        print(line)
    }

case "encode":
    let bf16 = has("--bf16")
    let bandMB = flag("--band").flatMap(Int.init)
    let save = flag("--save")
    guard args.count == 4 else { fail("encode <vaeDir> <image.safetensors> <untiled|tiled> <route>") }
    Memory.cacheLimit = 2 << 30  // MLXServeCore .automatic
    let dtype: DType = bf16 ? .bfloat16 : .float32
    let encoder = try Flux2VAEWeights.loadEncoder(directory: URL(fileURLWithPath: args[0]), dtype: dtype)
    encoder.convRoute = route(args[3])
    guard let image = try MLX.loadArrays(url: URL(fileURLWithPath: args[1]))["image"]?.asType(dtype) else {
        fail("no `image` in \(args[1])")
    }
    eval(image)
    let tile = args[2] == "tiled" ? Flux2VAETile(bandBytes: (bandMB ?? 64) << 20) : nil
    Memory.clearCache()
    let resident = Memory.activeMemory
    Memory.peakMemory = 0
    let t0 = Date()
    let packed = tile.map { encoder.encodePacked(image, tile: $0) } ?? encoder.encodePacked(image)
    eval(packed)
    let ms = Date().timeIntervalSince(t0) * 1000
    let peak = Memory.peakMemory
    print(String(
        format: "[encode] %d×%d %@ %@ %@%@: peak %.2f GB (resident %.2f GB, peak over resident %.2f GB), %.0f ms",
        image.dim(2), image.dim(3), bf16 ? "bf16" : "fp32", args[3], args[2],
        tile.map { " band \($0.bandBytes >> 20) MB" } ?? "",
        Double(peak) / gb, Double(resident) / gb, Double(peak - resident) / gb, ms))
    if let save { try MLX.save(arrays: ["image": packed.asType(.float32)], url: URL(fileURLWithPath: save)) }

case "golden":
    guard args.count == 3 else { fail("golden <vaeDir> <run_512.safetensors> <route>") }
    let vae = try Flux2VAEWeights.loadVAE(directory: URL(fileURLWithPath: args[0]), dtype: .float32)
    vae.convRoute = route(args[2])
    let url = URL(fileURLWithPath: args[1])
    let packed = try packedGolden(url)
    let ref = try MLX.loadArrays(url: url)["decoded"]!
    let untiled = vae.decodePackedLatents(packed)[0]
    eval(untiled)
    print("[golden 512² \(args[2])] untiled vs reference  \(stats(untiled, ref))")
    for mb in [64, 4, 1] {
        let t = vae.decodePackedLatents(packed, tile: Flux2VAETile(bandBytes: mb << 20))[0]
        eval(t)
        print("  tiled band \(mb) MB vs reference  \(stats(t, ref))   vs untiled  \(stats(t, untiled))")
    }

case "flowup":
    let bf16 = has("--bf16")
    let r = flag("--route").map(route) ?? .conv3d
    let save = flag("--save")
    guard args.count == 2 else { fail("flowup <weights> <pair.safetensors>") }
    let up = try FlowUpscaler.load(url: URL(fileURLWithPath: args[0]), dtype: bf16 ? .bfloat16 : .float32)
    up.unet.convRoute = r
    let pair = try MLX.loadArrays(url: URL(fileURLWithPath: args[1]))
    guard let latents = pair["latents"], let noise = pair["noise"] else { fail("pair needs latents + noise") }
    var out = up.upscale(latents: latents, noise: noise)
    eval(out)
    let t0 = Date()
    out = up.upscale(latents: latents, noise: noise)
    eval(out)
    let ms = Date().timeIntervalSince(t0) * 1000
    print(String(format: "[flowup %@ %@] %@ → %@, warm %.0f ms", bf16 ? "bf16" : "fp32", r.rawValue,
                 "\(latents.shape)", "\(out.shape)", ms))
    if let ref = pair["out"] {
        let a = out.asType(.float32).flattened(), b = ref.asType(.float32).flattened()
        let cos = sum(a * b) / (sqrt(sum(a * a)) * sqrt(sum(b * b)))
        eval(cos)
        print(String(format: "  vs reference: cos %.8f  ", cos.item(Float.self)) + "\(stats(out, ref))")
        if let refV = pair["v"] {
            let v = noise.asType(.float32) - out
            print("  velocity vs reference v: \(stats(v, refV))")
        }
    }
    if let save { try MLX.save(arrays: ["out": out], url: URL(fileURLWithPath: save)) }

case "flowimg":
    guard args.count == 4 else { fail("flowimg <vaeDir> <in> <key> <out>") }
    let vae = try Flux2VAEWeights.loadVAE(directory: URL(fileURLWithPath: args[0]), dtype: .float32)
    vae.convRoute = .conv3d
    guard var z = try MLX.loadArrays(url: URL(fileURLWithPath: args[1]))[args[2]]?.asType(.float32) else {
        fail("no `\(args[2])` in \(args[1])")
    }
    if z.dim(1) == FlowUpscalerUNet.sampleChannels { z = FlowUpscaler.patchify(z) }
    let image = vae.decodePackedLatents(z, tile: .standard)
    eval(image)
    try MLX.save(arrays: ["image": image], url: URL(fileURLWithPath: args[3]))
    print("[flowimg] \(args[2]) → \(image.shape)")

case "flowpipe":
    let bf16Up = has("--bf16-up")
    let seed = flag("--seed").flatMap(UInt64.init) ?? 0
    let r = flag("--route").map(route) ?? .conv3d
    let save = flag("--save")
    guard args.count == 4, let passes = Int(args[3]) else { fail("flowpipe <weights> <vaeDir> <packed> <passes>") }
    Memory.cacheLimit = 2 << 30  // MLXServeCore .automatic
    let up = try FlowUpscaler.load(url: URL(fileURLWithPath: args[0]), dtype: bf16Up ? .bfloat16 : .float32)
    up.unet.convRoute = r
    let vae = try Flux2VAEWeights.loadVAE(directory: URL(fileURLWithPath: args[1]), dtype: .float32)
    guard var packed = try MLX.loadArrays(url: URL(fileURLWithPath: args[2]))["packed"]?.asType(.float32) else {
        fail("no `packed` in \(args[2])")
    }
    eval(packed)
    // Warm the kernels on the input size (a real worker has run one before).
    eval(up.upscalePacked(packed, seed: seed))
    Memory.clearCache()
    let resident = Memory.activeMemory
    for i in 0 ..< passes {
        Memory.peakMemory = 0
        let t0 = Date()
        packed = up.upscalePacked(packed, seed: seed &+ UInt64(i))
        let ms = Date().timeIntervalSince(t0) * 1000
        print(String(format: "[pass %d] → %d² latent px (%d² image): %.0f ms, peak over resident %.2f GB", i + 1,
                     packed.dim(2) * 2, packed.dim(2) * 16, ms, Double(Memory.peakMemory - resident) / gb))
        Memory.clearCache()
    }
    Memory.peakMemory = 0
    let t0 = Date()
    let image = vae.decodePackedLatents(packed, tile: .standard)
    eval(image)
    let ms = Date().timeIntervalSince(t0) * 1000
    print(String(format: "[decode] %d×%d row-band fp32: %.0f ms, peak over resident %.2f GB", image.dim(3), image.dim(2),
                 ms, Double(Memory.peakMemory - resident) / gb))
    if let save {
        try MLX.save(arrays: ["image": image.asType(.float32), "packed": packed], url: URL(fileURLWithPath: save))
    }

default:
    fail("unknown mode \(mode)")
}
