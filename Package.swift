// swift-tools-version: 6.2
// flux2-vae-mlx-swift — the FLUX.2 VAE (decoder path) as a neutral, standalone Swift/MLX package.
//
// Extracted from lens-mlx-swift so multiple text-to-image backers (Lens, ERNIE-Image-Turbo, …)
// can share the VAE without depending on each other's MODEL packages. A foundational component:
// a future candidate to fold into MLXEngine's utility layer (cf. format-bridge).

import PackageDescription

let package = Package(
    name: "Flux2VAE",
    platforms: [
        .macOS(.v26)
    ],
    products: [
        .library(name: "Flux2VAE", targets: ["Flux2VAE"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.30.0"),
    ],
    targets: [
        .target(
            name: "Flux2VAE",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
            ],
            path: "Sources/Flux2VAE"
        ),
    ]
)
